#!/usr/bin/env python
"""Train the spectra-to-motif multi-label predictor."""

import os
import sys
import json
import argparse
import importlib.util
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from rdkit import Chem, RDLogger
from sklearn.metrics import precision_score, recall_score, f1_score

# Suppress RDKit warnings.
RDLogger.DisableLog('rdApp.*')


def setup_ddp():
    """Initialize DDP if launched with distributed environment variables."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        
        return True, rank, world_size, local_rank
    else:
        return False, 0, 1, 0


def cleanup_ddp():
    """Tear down DDP."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank):
    """Return whether the current rank is the main process."""
    return rank == 0

# Add the repository root for local imports.
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets import get_dataset
from spectra.Sp import SpecFormer


def load_config(config_path: str):
    """Load a config module from disk."""
    spec = importlib.util.spec_from_file_location("config", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return config_module.get_config()


def filter_substructures(freq_file: str, threshold: float = 0.01) -> List[str]:
    """
    Select frequent motifs from a JSON frequency file.
    """
    with open(freq_file, 'r') as f:
        freq_data = json.load(f)
    
    selected = [smi for smi, data in freq_data.items() if data['freq'] >= threshold]
    selected.sort(key=lambda x: -freq_data[x]['freq'])
    print(f"Selected motifs: {len(freq_data)} -> {len(selected)} (threshold={threshold*100:.1f}%)")
    return selected


def compute_substructure_labels(mol, substructure_smiles: List[str]) -> torch.Tensor:
    """
    Build the multi-label motif target for one molecule.
    """
    labels = torch.zeros(len(substructure_smiles))
    
    if mol is None:
        return labels
    
    for i, smi in enumerate(substructure_smiles):
        pattern = Chem.MolFromSmiles(smi)
        if pattern is not None:
            if mol.HasSubstructMatch(pattern):
                labels[i] = 1.0
    
    return labels


class SubstructureDataset(Dataset):
    """
    Spectra-to-motif dataset with cached multi-label targets.
    """
    
    def __init__(self, base_dataset, substructure_smiles: List[str],
                 cache_dir: str = None, split_name: str = 'train'):
        """
        Args:
            base_dataset: source dataset split
            substructure_smiles: motif vocabulary as SMILES strings
            cache_dir: optional cache directory for precomputed labels
            split_name: split name used for cache filenames
        """
        self.base_dataset = base_dataset
        self.substructure_smiles = substructure_smiles
        self.num_classes = len(substructure_smiles)
        self.cache_dir = cache_dir
        self.split_name = split_name
        
        self.labels, self.valid_indices = self._load_or_compute_labels()
        
        print(f"Valid samples: {len(self.valid_indices)} / {len(base_dataset)}")
        label_sums = self.labels.sum(dim=0)
        print(f"Average motifs per sample: {self.labels.sum(dim=1).mean():.2f}")
        print(f"Least frequent motif count: {label_sums.min():.0f}")
        print(f"Most frequent motif count: {label_sums.max():.0f}")
    
    def _get_cache_path(self):
        """Return the cache file path for this split."""
        if self.cache_dir is None:
            return None
        return os.path.join(self.cache_dir, f'substructure_labels_{self.split_name}.npz')
    
    def _load_or_compute_labels(self):
        """Load cached labels or build them from scratch."""
        cache_path = self._get_cache_path()
        
        if cache_path and os.path.exists(cache_path):
            print(f"Loading cached labels from {cache_path}")
            try:
                data = np.load(cache_path, allow_pickle=True)
                cached_labels = torch.from_numpy(data['labels']).float()
                cached_indices = list(data['valid_indices'])
                cached_smiles = list(data['substructure_smiles'])
                
                if cached_smiles == self.substructure_smiles:
                    return cached_labels, cached_indices
                else:
                    print("Cached motif vocabulary does not match. Recomputing labels.")
            except Exception as e:
                print(f"Failed to load cache ({e}). Recomputing labels.")
        
        print("Precomputing motif labels...")
        labels = []
        valid_indices = []
        
        for idx in tqdm(range(len(self.base_dataset))):
            data = self.base_dataset[idx]
            if not hasattr(data, 'rdmol') or data.rdmol is None:
                continue
            if not hasattr(data, 'spectra') or data.spectra is None:
                continue
            
            label = compute_substructure_labels(data.rdmol, self.substructure_smiles)
            labels.append(label)
            valid_indices.append(idx)
        
        labels = torch.stack(labels)
        
        if cache_path:
            self._save_labels(labels, valid_indices, cache_path)
        
        return labels, valid_indices
    
    def _save_labels(self, labels, valid_indices, cache_path):
        """Save precomputed labels to disk."""
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        
        np.savez(cache_path, 
                 labels=labels.numpy(),
                 valid_indices=np.array(valid_indices),
                 substructure_smiles=np.array(self.substructure_smiles, dtype=object))
        print(f"Saved motif labels to {cache_path}")
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        data = self.base_dataset[real_idx]
        
        spectra = data.spectra  # [uv, ir, raman]
        
        return {
            'spectra': spectra,
            'labels': self.labels[idx],
            'idx': real_idx,
        }
    
    def compute_pos_weight(self) -> torch.Tensor:
        """
        Compute per-class positive weights for imbalanced BCE training.
        """
        label_sums = self.labels.sum(dim=0)  # [num_classes]
        num_samples = len(self.labels)
        
        label_sums = torch.clamp(label_sums, min=1)
        neg_counts = num_samples - label_sums
        
        pos_weight = neg_counts / label_sums
        
        pos_weight = torch.clamp(pos_weight, max=50.0)
        
        return pos_weight


class SubstructurePredictor(nn.Module):
    """
    Spectra-to-motif predictor.
    """
    
    def __init__(self, 
                 num_classes: int,
                 spectra_modalities: Tuple = ('uv', 'ir', 'raman'),
                 d_model: int = 256,
                 depth: int = 4,
                 n_heads: int = 8,
                 output_dim: int = 128,
                 hidden_dim: int = 512,
                 num_hidden_layers: int = 3,
                 dropout: float = 0.2,
                 spectra_patch_lens: Tuple = (32, 64, 64),
                 spectra_strides: Tuple = (16, 32, 32)):
        super().__init__()
        
        self.num_classes = num_classes
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        
        # SpecFormer encoder.
        self.specformer = SpecFormer(
            patch_len=list(spectra_patch_lens),
            stride=list(spectra_strides),
            output_dim=output_dim,
            input_norm_type="minmax",
            use_dynamic_norm=False,
            n_layers=depth,
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            act="gelu",
        )
        
        # Input projection.
        self.input_proj = nn.Linear(output_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        
        # Residual classification head.
        self.hidden_layers = nn.ModuleList()
        for _ in range(num_hidden_layers):
            self.hidden_layers.append(nn.ModuleDict({
                'fc': nn.Linear(hidden_dim, hidden_dim),
                'norm': nn.LayerNorm(hidden_dim),
                'dropout': nn.Dropout(dropout),
            }))
        
        # Output projection.
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, spectra: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            spectra: list of modality tensors shaped [batch, length]
        
        Returns:
            logits: [batch, num_classes]
        """
        _, cls_features = self.specformer(spectra)
        x = self.input_proj(cls_features)
        x = self.input_norm(x)
        x = F.gelu(x)
        
        for layer in self.hidden_layers:
            residual = x
            x = layer['fc'](x)
            x = layer['norm'](x)
            x = F.gelu(x)
            x = layer['dropout'](x)
            x = x + residual
        
        return self.output_proj(x)
    
    def predict(self, spectra: List[torch.Tensor], threshold: float = 0.5) -> torch.Tensor:
        """
        Predict binary motif labels from spectra.
        """
        logits = self.forward(spectra)
        probs = torch.sigmoid(logits)
        return (probs > threshold).float()


def collate_fn(batch):
    """Collate spectra and labels for motif prediction."""
    batch_spectra = [item['spectra'] for item in batch]
    
    num_modalities = len(batch_spectra[0])
    spectra_batched = []
    for m in range(num_modalities):
        modality_batch = torch.stack([s[m] for s in batch_spectra])
        spectra_batched.append(modality_batch)
    
    labels = torch.stack([item['labels'] for item in batch])
    indices = torch.tensor([item['idx'] for item in batch])
    
    return {
        'spectra': spectra_batched,
        'labels': labels,
        'idx': indices,
    }


def train_epoch(model, dataloader, optimizer, criterion, device, pos_weight=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch in tqdm(dataloader, desc="Training"):
        spectra = [s.to(device) for s in batch['spectra']]
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        logits = model(spectra)
        
        if pos_weight is not None:
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
        loss = criterion(logits, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        preds = (torch.sigmoid(logits) > 0.5).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels.cpu().numpy())
    
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    
    precision = precision_score(all_labels, all_preds, average='samples', zero_division=1)
    recall = recall_score(all_labels, all_preds, average='samples', zero_division=1)
    f1 = f1_score(all_labels, all_preds, average='samples', zero_division=1)
    
    return total_loss / len(dataloader), precision, recall, f1


def evaluate(model, dataloader, criterion, device):
    """Evaluate the motif predictor."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            spectra = [s.to(device) for s in batch['spectra']]
            labels = batch['labels'].to(device)
            
            logits = model(spectra)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).cpu().numpy()
            
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
    
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    all_probs = np.vstack(all_probs)
    
    # Sample-level metrics.
    precision = precision_score(all_labels, all_preds, average='samples', zero_division=1)
    recall = recall_score(all_labels, all_preds, average='samples', zero_division=1)
    f1 = f1_score(all_labels, all_preds, average='samples', zero_division=1)
    
    # Macro-averaged class metrics.
    precision_macro = precision_score(all_labels, all_preds, average='macro', zero_division=1)
    recall_macro = recall_score(all_labels, all_preds, average='macro', zero_division=1)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=1)
    
    # Exact multi-label match rate.
    exact_match = (all_preds == all_labels).all(axis=1).mean()
    
    return {
        'loss': total_loss / len(dataloader),
        'precision_samples': precision,
        'recall_samples': recall,
        'f1_samples': f1,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'exact_match': exact_match,
        'preds': all_preds,
        'labels': all_labels,
        'probs': all_probs,
    }


def main():
    parser = argparse.ArgumentParser(description="Train the spectra-to-motif predictor")
    parser.add_argument('--config', type=str, default='configs/qm9sp_mast.py',
                       help='Path to a config file.')
    parser.add_argument('--freq_file', type=str,
                       default='reference/substructure_freq_all.json',
                       help='Motif frequency file.')
    parser.add_argument('--freq_threshold', type=float, default=0.01,
                       help='Minimum motif frequency threshold.')
    parser.add_argument('--output_dir', type=str, default='runs/motif_predictor',
                       help='Output directory.')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size per process.')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate.')
    parser.add_argument('--hidden_dim', type=int, default=512,
                       help='Hidden width of the classifier head.')
    parser.add_argument('--num_hidden_layers', type=int, default=3,
                       help='Number of hidden layers in the classifier head.')
    parser.add_argument('--use_pos_weight', action='store_true',
                       help='Use positive-class weighting.')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed.')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of dataloader workers.')
    parser.add_argument('--cache_dir', type=str,
                       default='data/substructure_cache',
                       help='Cache directory for precomputed motif labels.')
    
    args = parser.parse_args()
    
    use_ddp, rank, world_size, local_rank = setup_ddp()
    
    # Offset the seed by rank for distributed workers.
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    
    if use_ddp:
        device = torch.device(f'cuda:{local_rank}')
        if is_main_process(rank):
            print(f"Using distributed training on {world_size} GPUs")
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
    
    if is_main_process(rank):
        os.makedirs(args.output_dir, exist_ok=True)
    
    if use_ddp:
        dist.barrier()
    
    if is_main_process(rank):
        print(f"Loading config: {args.config}")
    config = load_config(args.config)
    
    if is_main_process(rank):
        print(f"\nLoading motif frequency file: {args.freq_file}")
    substructure_smiles = filter_substructures(args.freq_file, args.freq_threshold) if is_main_process(rank) else None
    
    if use_ddp:
        if is_main_process(rank):
            substructure_data = json.dumps(substructure_smiles)
        else:
            substructure_data = None
        
        object_list = [substructure_data]
        dist.broadcast_object_list(object_list, src=0)
        substructure_smiles = json.loads(object_list[0])
    
    num_classes = len(substructure_smiles)
    if is_main_process(rank):
        print(f"Number of motif classes: {num_classes}")
    
    if is_main_process(rank):
        label_mapping = {i: smi for i, smi in enumerate(substructure_smiles)}
        with open(os.path.join(args.output_dir, 'substructure_labels.json'), 'w') as f:
            json.dump(label_mapping, f, indent=2)
        print("Saved motif label mapping")
    
    if is_main_process(rank):
        print("\nLoading dataset...")
    train_ds, val_ds, test_ds, _ = get_dataset(config, transform=True)
    
    if is_main_process(rank):
        print(f"\nCache directory: {args.cache_dir}")
        print("\nBuilding training split...")
    train_dataset = SubstructureDataset(train_ds, substructure_smiles, 
                                        cache_dir=args.cache_dir, split_name='train')
    if is_main_process(rank):
        print("\nBuilding validation split...")
    val_dataset = SubstructureDataset(val_ds, substructure_smiles,
                                      cache_dir=args.cache_dir, split_name='val')
    if is_main_process(rank):
        print("\nBuilding test split...")
    test_dataset = SubstructureDataset(test_ds, substructure_smiles,
                                       cache_dir=args.cache_dir, split_name='test')
    
    pos_weight = None
    if args.use_pos_weight:
        pos_weight = train_dataset.compute_pos_weight()
        if is_main_process(rank):
            print(f"Positive weight range: [{pos_weight.min():.2f}, {pos_weight.max():.2f}]")
    
    if use_ddp:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                                  sampler=train_sampler, collate_fn=collate_fn, 
                                  num_workers=args.num_workers, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                                sampler=val_sampler, collate_fn=collate_fn, 
                                num_workers=args.num_workers, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                                 sampler=test_sampler, collate_fn=collate_fn, 
                                 num_workers=args.num_workers, pin_memory=True)
    else:
        train_sampler = None
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                                  shuffle=True, collate_fn=collate_fn, 
                                  num_workers=args.num_workers, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                                shuffle=False, collate_fn=collate_fn, 
                                num_workers=args.num_workers, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                                 shuffle=False, collate_fn=collate_fn, 
                                 num_workers=args.num_workers, pin_memory=True)
    
    if is_main_process(rank):
        print("\nBuilding model...")
    model = SubstructurePredictor(
        num_classes=num_classes,
        spectra_modalities=config.model.spectra_modalities,
        d_model=config.model.spectra_d_model,
        depth=config.model.spectra_depth,
        n_heads=config.model.spectra_n_heads,
        output_dim=config.model.spectra_output_dim,
        hidden_dim=args.hidden_dim,
        num_hidden_layers=args.num_hidden_layers,
        dropout=0.2,
        spectra_patch_lens=config.model.spectra_patch_lens,
        spectra_strides=config.model.spectra_strides,
    ).to(device)
    
    if use_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, 
                    find_unused_parameters=True)
    
    if is_main_process(rank):
        raw_model = model.module if use_ddp else model
        print(f"Model parameters: {sum(p.numel() for p in raw_model.parameters()):,}")
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device) if pos_weight is not None else None)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    if is_main_process(rank):
        print("\nStarting training...")
    best_f1 = 0
    best_epoch = 0
    
    for epoch in range(args.epochs):
        if use_ddp:
            train_sampler.set_epoch(epoch)
        
        if is_main_process(rank):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{args.epochs}")
            print(f"{'='*60}")
        
        train_loss, train_p, train_r, train_f1 = train_epoch(
            model, train_loader, optimizer, criterion, device, pos_weight
        )
        
        val_results = evaluate(model, val_loader, criterion, device)
        
        scheduler.step()
        
        if is_main_process(rank):
            print(f"Train - Loss: {train_loss:.4f}, P: {train_p:.4f}, R: {train_r:.4f}, F1: {train_f1:.4f}")
            print(f"Val   - Loss: {val_results['loss']:.4f}, P: {val_results['precision_samples']:.4f}, "
                  f"R: {val_results['recall_samples']:.4f}, F1: {val_results['f1_samples']:.4f}")
            print(f"Val   - Macro F1: {val_results['f1_macro']:.4f}, Exact Match: {val_results['exact_match']:.4f}")
        
        if val_results['f1_samples'] > best_f1:
            best_f1 = val_results['f1_samples']
            best_epoch = epoch + 1
            
            if is_main_process(rank):
                raw_model = model.module if use_ddp else model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': raw_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_f1': best_f1,
                    'substructure_smiles': substructure_smiles,
                    'config': {
                        'num_classes': num_classes,
                        'hidden_dim': args.hidden_dim,
                        'freq_threshold': args.freq_threshold,
                    }
                }, os.path.join(args.output_dir, 'best_model.pth'))
                print(f"  [New Best] F1={best_f1:.4f}")
    
    if is_main_process(rank):
        print(f"\nTraining complete. Best F1: {best_f1:.4f} (epoch {best_epoch})")
    
    if is_main_process(rank):
        print("\nEvaluating the best checkpoint on the test split...")
        
        test_model = SubstructurePredictor(
            num_classes=num_classes,
            spectra_modalities=config.model.spectra_modalities,
            d_model=config.model.spectra_d_model,
            depth=config.model.spectra_depth,
            n_heads=config.model.spectra_n_heads,
            output_dim=config.model.spectra_output_dim,
            hidden_dim=args.hidden_dim,
            dropout=0.1,
            spectra_patch_lens=config.model.spectra_patch_lens,
            spectra_strides=config.model.spectra_strides,
        ).to(device)
        
        checkpoint = torch.load(os.path.join(args.output_dir, 'best_model.pth'), map_location=device)
        test_model.load_state_dict(checkpoint['model_state_dict'])
        
        test_loader_single = DataLoader(test_dataset, batch_size=args.batch_size,
                                        shuffle=False, collate_fn=collate_fn, 
                                        num_workers=args.num_workers, pin_memory=True)
        
        test_results = evaluate(test_model, test_loader_single, criterion, device)
        
        print(f"\n{'='*60}")
        print("Test Results")
        print(f"{'='*60}")
        print(f"Loss: {test_results['loss']:.4f}")
        print(f"Precision (samples): {test_results['precision_samples']:.4f}")
        print(f"Recall (samples): {test_results['recall_samples']:.4f}")
        print(f"F1 (samples): {test_results['f1_samples']:.4f}")
        print(f"F1 (macro): {test_results['f1_macro']:.4f}")
        print(f"Exact Match: {test_results['exact_match']:.4f}")
        
        print("\nPer-motif performance (sorted by F1):")
        per_class_f1 = []
        for i in range(num_classes):
            y_true = test_results['labels'][:, i]
            y_pred = test_results['preds'][:, i]
            if y_true.sum() > 0:
                f1 = f1_score(y_true, y_pred, zero_division=1)
                per_class_f1.append((substructure_smiles[i], f1, y_true.sum()))
        
        per_class_f1.sort(key=lambda x: -x[1])
        for smi, f1, count in per_class_f1[:20]:
            print(f"  {smi}: F1={f1:.4f} (count={count:.0f})")
        
        np.savez(
            os.path.join(args.output_dir, 'test_results.npz'),
            preds=test_results['preds'],
            labels=test_results['labels'],
            probs=test_results['probs'],
        )
        
        print(f"\nSaved outputs to: {args.output_dir}/")
    
    cleanup_ddp()


if __name__ == "__main__":
    main()
