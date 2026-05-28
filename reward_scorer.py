import json
import math
import os

import torch
import torch.nn.functional as F

from sampling_guidance import MolSpectraGuidance
from train_motif_predictor import SubstructurePredictor


class RewardScorer:
    """Spectrum-structure reward used by MCTS."""

    def __init__(self, guidance_model, device="cuda"):
        self.guidance_model = guidance_model
        self.device = device
        self._spec_feature_cache = None

    def cache_spectra_features(self, spectra):
        with torch.no_grad():
            self._spec_feature_cache = self.guidance_model.encode_spectra(spectra)
        return self._spec_feature_cache

    def score(self, pred_t, node_mask, spectra=None, batch_idx=None):
        with torch.no_grad():
            if spectra is not None:
                spec_feature = self.guidance_model.encode_spectra(spectra)
            else:
                spec_feature = self._spec_feature_cache

            if spec_feature is None:
                raise ValueError("No cached spectra features are available.")

            z, pos, batch = self._convert_pred_to_structure(pred_t, node_mask)
            if len(z) == 0:
                return torch.zeros(pred_t.shape[0], device=self.device)

            mol_feature = self.guidance_model.encode_structure(z, pos, batch)
            if batch_idx is not None:
                spec_feature = spec_feature[batch_idx]

            spec_norm = F.normalize(spec_feature, p=2, dim=-1)
            mol_norm = F.normalize(mol_feature, p=2, dim=-1)
            return (spec_norm * mol_norm).sum(dim=-1)

    def _convert_pred_to_structure(self, pred_t, node_mask):
        type_map_tensor = torch.tensor([1, 6, 7, 8, 9], dtype=torch.long, device=pred_t.device)
        batch_size = pred_t.shape[0]
        pos_pred = pred_t[:, :, :3]
        atom_pred = pred_t[:, :, 3:]
        if atom_pred.shape[-1] > 5:
            atom_pred = atom_pred[:, :, :-1]
        atom_pred = atom_pred[:, :, :5]

        node_mask_squeezed = node_mask.squeeze(-1)
        n_valid_per_batch = node_mask_squeezed.sum(dim=1).long()

        pos_all, z_all, batch_all = [], [], []
        for batch_idx in range(batch_size):
            n_valid = n_valid_per_batch[batch_idx].item()
            if n_valid == 0:
                continue
            pos_all.append(pos_pred[batch_idx, :n_valid, :])
            atom_type_idx = atom_pred[batch_idx, :n_valid, :].argmax(dim=-1).clamp(0, 4)
            z_all.append(type_map_tensor[atom_type_idx])
            batch_all.append(torch.full((n_valid,), batch_idx, dtype=torch.long, device=pred_t.device))

        if not pos_all:
            empty_long = torch.zeros(0, dtype=torch.long, device=pred_t.device)
            empty_pos = torch.zeros(0, 3, device=pred_t.device)
            return empty_long, empty_pos, empty_long

        return torch.cat(z_all, dim=0), torch.cat(pos_all, dim=0), torch.cat(batch_all, dim=0)


class MotifRewardScorer:
    """Motif-consistency reward in the paper."""

    def __init__(self, predictor, motif_smiles, device="cuda"):
        self.predictor = predictor
        self.motif_smiles = motif_smiles
        self.device = device
        self._pred_probs_cache = None
        self.patterns = []

        from rdkit import Chem

        for smiles in motif_smiles:
            self.patterns.append(Chem.MolFromSmiles(smiles))

    def cache_predicted_probs(self, spectra):
        with torch.no_grad():
            logits = self.predictor(spectra)
            self._pred_probs_cache = torch.sigmoid(logits)
        return self._pred_probs_cache

    def compute_motif_labels(self, mol):
        labels = torch.zeros(len(self.patterns), device=self.device)
        if mol is None:
            return labels
        for i, pattern in enumerate(self.patterns):
            if pattern is not None:
                try:
                    if mol.HasSubstructMatch(pattern):
                        labels[i] = 1.0
                except Exception:
                    pass
        return labels

    def score(self, mol, batch_idx=0):
        if self._pred_probs_cache is None:
            return 0.0

        target = self.compute_motif_labels(mol)
        probs = self._pred_probs_cache[batch_idx].clamp(1e-6, 1.0 - 1e-6)
        bce = F.binary_cross_entropy(probs, target, reduction="mean")
        return math.exp(-float(bce.item()))


def create_reward_scorer(ckpt_path, device="cuda"):
    guidance_model = MolSpectraGuidance(ckpt_path, device)
    return RewardScorer(guidance_model, device)


def create_substructure_scorer(checkpoint_path, device="cuda"):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    motif_smiles = checkpoint.get("substructure_smiles", [])
    config = checkpoint.get("config", {})
    num_classes = config.get("num_classes", len(motif_smiles))
    hidden_dim = config.get("hidden_dim", 512)

    if not motif_smiles:
        json_path = os.path.join(os.path.dirname(checkpoint_path), "substructure_labels.json")
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                label_mapping = json.load(f)
            motif_smiles = [label_mapping[str(i)] for i in range(len(label_mapping))]
            num_classes = len(motif_smiles)

    if not motif_smiles:
        raise ValueError(f"Could not load motif labels from {checkpoint_path}")

    predictor = SubstructurePredictor(
        num_classes=num_classes,
        spectra_modalities=("uv", "ir", "raman"),
        d_model=256,
        depth=4,
        n_heads=8,
        output_dim=128,
        hidden_dim=hidden_dim,
        num_hidden_layers=3,
        dropout=0.2,
        spectra_patch_lens=(32, 64, 64),
        spectra_strides=(16, 32, 32),
    )
    predictor.load_state_dict(checkpoint["model_state_dict"])
    predictor.to(device)
    predictor.eval()
    return MotifRewardScorer(predictor, motif_smiles, device)
