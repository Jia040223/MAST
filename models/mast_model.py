"""Motif-augmented diffusion denoiser used in MAST."""

import importlib.util
import json
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
from torch_geometric.utils import dense_to_sparse

from . import utils
from .spectra_mol_gnn import Spectra_DGT_concat


ROOT = Path(__file__).resolve().parent.parent


def _is_main_process() -> bool:
    return (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0


def _load_motif_predictor_class():
    module_path = ROOT / "train_motif_predictor.py"
    spec = importlib.util.spec_from_file_location("train_motif_predictor", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.SubstructurePredictor


class SubstructureEmbedding(nn.Module):
    """Project motif logits into the denoiser conditioning space."""

    def __init__(
        self,
        num_substructures: int,
        embedding_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        use_sigmoid: bool = True,
    ):
        super().__init__()
        self.use_sigmoid = use_sigmoid
        self.mlp = nn.Sequential(
            nn.Linear(num_substructures, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, substructure_logits: torch.Tensor) -> torch.Tensor:
        features = torch.sigmoid(substructure_logits) if self.use_sigmoid else substructure_logits
        return self.mlp(features)


class SubstructurePredictorWrapper(nn.Module):
    """Load a pretrained motif predictor and expose logits."""

    def __init__(self, checkpoint_path: str, device: torch.device, freeze: bool = True):
        super().__init__()
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        predictor_config = checkpoint.get("config", {})
        num_classes = predictor_config.get("num_classes")

        if num_classes is None:
            labels_path = Path(checkpoint_path).with_name("substructure_labels.json")
            if not labels_path.exists():
                raise ValueError(f"Could not infer motif vocabulary size from {checkpoint_path}")
            with open(labels_path, "r") as f:
                label_mapping = json.load(f)
            num_classes = len(label_mapping)

        predictor_cls = _load_motif_predictor_class()
        self.predictor = predictor_cls(
            num_classes=num_classes,
            spectra_modalities=tuple(predictor_config.get("spectra_modalities", ("uv", "ir", "raman"))),
            d_model=predictor_config.get("d_model", 256),
            depth=predictor_config.get("depth", 4),
            n_heads=predictor_config.get("n_heads", 8),
            output_dim=predictor_config.get("output_dim", 128),
            hidden_dim=predictor_config.get("hidden_dim", 512),
            num_hidden_layers=predictor_config.get("num_hidden_layers", 3),
            dropout=predictor_config.get("dropout", 0.2),
            spectra_patch_lens=tuple(predictor_config.get("spectra_patch_lens", (32, 64, 64))),
            spectra_strides=tuple(predictor_config.get("spectra_strides", (16, 32, 32))),
        ).to(device)

        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.predictor.load_state_dict(state_dict)
        self.freeze = freeze

        if self.freeze:
            for parameter in self.predictor.parameters():
                parameter.requires_grad = False
            self.predictor.eval()

        if _is_main_process():
            print(
                f"Loaded motif predictor from {checkpoint_path} "
                f"(num_classes={num_classes}, frozen={self.freeze})"
            )

    def forward(self, spectra: List[torch.Tensor]) -> torch.Tensor:
        with torch.set_grad_enabled(not self.freeze):
            return self.predictor(spectra)


@utils.register_model(name="MASTDenoiser")
class MASTDenoiser(Spectra_DGT_concat):
    """Spectra-conditioned denoiser augmented with a frozen motif predictor."""

    def __init__(self, config):
        super().__init__(config)

        self.use_substructure = getattr(config.model, "use_substructure", True)
        self.substructure_checkpoint = getattr(config.model, "substructure_checkpoint_path", "")
        self.freeze_substructure_predictor = getattr(
            config.model,
            "freeze_substructure_predictor",
            True,
        )
        self.substructure_fusion = getattr(config.model, "substructure_fusion", "add")
        self.substructure_embedding_dim = getattr(config.model, "substructure_embedding_dim", 128)

        self.substructure_predictor = None
        self.substructure_embedding = None
        self.substructure_fusion_mlp = None
        self.substructure_gate_mlp = None

        if not self.use_substructure:
            return

        if not self.substructure_checkpoint:
            raise ValueError(
                "config.model.substructure_checkpoint_path must be set when motif conditioning is enabled."
            )

        labels_path = Path(self.substructure_checkpoint).with_name("substructure_labels.json")
        if not labels_path.exists():
            raise FileNotFoundError(f"Missing motif label file: {labels_path}")
        with open(labels_path, "r") as f:
            label_mapping = json.load(f)
        num_substructures = len(label_mapping)

        predictor_device = config.device if isinstance(config.device, torch.device) else torch.device(config.device)
        self.substructure_predictor = SubstructurePredictorWrapper(
            checkpoint_path=self.substructure_checkpoint,
            device=predictor_device,
            freeze=self.freeze_substructure_predictor,
        )
        self.substructure_embedding = SubstructureEmbedding(
            num_substructures=num_substructures,
            embedding_dim=self.substructure_embedding_dim,
            hidden_dim=getattr(config.model, "substructure_hidden_dim", 256),
            dropout=getattr(config.model, "substructure_dropout", 0.1),
        )

        if not self.cond_time:
            raise ValueError("Motif conditioning requires config.model.cond_time=True.")

        time_dim = config.model.nf * 4
        if self.substructure_fusion == "add":
            self.substructure_fusion_mlp = nn.Sequential(
                nn.Linear(self.substructure_embedding_dim, time_dim),
                nn.LayerNorm(time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim),
            )
        elif self.substructure_fusion == "concat":
            self.substructure_fusion_mlp = nn.Sequential(
                nn.Linear(self.substructure_embedding_dim + time_dim, time_dim),
                nn.LayerNorm(time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim),
            )
        elif self.substructure_fusion == "gate":
            self.substructure_fusion_mlp = nn.Sequential(
                nn.Linear(self.substructure_embedding_dim, time_dim),
                nn.LayerNorm(time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim),
            )
            self.substructure_gate_mlp = nn.Sequential(
                nn.Linear(self.substructure_embedding_dim, time_dim),
                nn.Sigmoid(),
            )
        else:
            raise ValueError(f"Unknown substructure_fusion mode: {self.substructure_fusion}")

        if _is_main_process():
            print(
                f"Initialized motif-conditioned denoiser "
                f"(num_motifs={num_substructures}, fusion={self.substructure_fusion})"
            )

    def forward(self, t, xh, node_mask, edge_mask, context=None, *args, **kwargs):
        spectra = kwargs.get("spectra")
        substructure_spectra = kwargs.get("substructure_spectra", spectra)
        edge_x = kwargs["edge_x"]
        cond_x = kwargs.get("cond_x")
        cond_edge_x = kwargs.get("cond_edge_x")

        substructure_embedding = None
        if self.substructure_predictor is not None and substructure_spectra is not None:
            motif_logits = self.substructure_predictor(substructure_spectra)
            substructure_embedding = self.substructure_embedding(motif_logits)

        bs, n_nodes, _ = xh.shape
        pos_init = pos = xh[:, :, 0:3].clone().reshape(bs * n_nodes, -1)
        h = xh[:, :, 3:].clone().reshape(bs * n_nodes, -1)

        adj_mask = edge_mask.reshape(bs, n_nodes, n_nodes)
        dense_index = adj_mask.nonzero(as_tuple=True)
        edge_index, _ = dense_to_sparse(adj_mask)

        if cond_x is None:
            cond_x = torch.zeros_like(xh)
            cond_edge_x = torch.zeros_like(edge_x)
            cond_adj_2d = torch.ones((edge_index.size(1), 1), device=edge_x.device)
        else:
            with torch.no_grad():
                cond_adj_2d = cond_edge_x[dense_index][:, 0:1].clone()
                cond_adj_2d[cond_adj_2d >= self.edge_th] = 1.0
                cond_adj_2d[cond_adj_2d < self.edge_th] = 0.0

        cond_pos = cond_x[:, :, 0:3].clone().reshape(bs * n_nodes, -1)
        cond_h = cond_x[:, :, 3:].clone().reshape(bs * n_nodes, -1)
        h = torch.cat([h, cond_h], dim=-1)

        spectra_embedding = None
        spectra_cls = None
        if self.use_spectra and spectra is not None:
            _, spectra_cls = self.spectra_encoder(spectra)
            spectra_embedding = self.spectra_mlp(spectra_cls)

        node_time_emb = None
        edge_time_emb = None
        if self.cond_time:
            noise_level = kwargs["noise_level"]
            time_emb = self.time_mlp(noise_level)

            if spectra_embedding is not None:
                if self.spectra_fusion == "add":
                    time_emb = time_emb + spectra_embedding
                elif self.spectra_fusion == "gate":
                    gate = self.gate_mlp(spectra_cls)
                    time_emb = time_emb * gate + spectra_embedding * (1 - gate)
                elif self.spectra_fusion == "concat":
                    if time_emb.dim() == 3:
                        time_emb = time_emb.squeeze(1)
                    if spectra_embedding.dim() == 3:
                        spectra_embedding = spectra_embedding.squeeze(1)
                    time_emb = self.fusion_proj(torch.cat([time_emb, spectra_embedding], dim=-1))

            if substructure_embedding is not None:
                if self.substructure_fusion == "add":
                    time_emb = time_emb + self.substructure_fusion_mlp(substructure_embedding)
                elif self.substructure_fusion == "concat":
                    time_emb = self.substructure_fusion_mlp(
                        torch.cat([time_emb, substructure_embedding], dim=-1)
                    )
                elif self.substructure_fusion == "gate":
                    motif_proj = self.substructure_fusion_mlp(substructure_embedding)
                    motif_gate = self.substructure_gate_mlp(substructure_embedding)
                    time_emb = time_emb * motif_gate + motif_proj * (1 - motif_gate)

            node_time_emb = time_emb.unsqueeze(1).expand(-1, n_nodes, -1).reshape(bs * n_nodes, -1)
            edge_batch_id = torch.div(edge_index[0], n_nodes, rounding_mode="floor")
            edge_time_emb = time_emb[edge_batch_id]

        from . import utils as model_utils

        distances, cond_adj_spatial = model_utils.coord2diff_adj(cond_pos, edge_index, self.spatial_cut_off)
        if distances.sum() == 0:
            distances = distances.repeat(1, self.dist_dim)
        elif self.dist_gbf:
            distances = self.dist_layer(distances, edge_time_emb)

        cur_edge_attr = edge_x[dense_index]
        cond_edge_attr = cond_edge_x[dense_index]

        extra_adj = torch.cat([cond_adj_2d, cond_adj_spatial], dim=-1)
        edge_attr = torch.cat([cur_edge_attr, cond_edge_attr, distances], dim=-1)

        h = self.node_emb(h)
        edge_attr = self.edge_emb(edge_attr)

        atom_hids = [h]
        edge_hids = [edge_attr]
        for layer_idx in range(self.n_layers):
            h, edge_attr, pos = self._modules[f"e_block_{layer_idx}"](
                pos,
                h,
                edge_attr,
                edge_index,
                node_mask.reshape(-1, 1),
                extra_adj,
                node_time_emb,
                edge_time_emb,
            )
            if self.CoM:
                pos = model_utils.remove_mean_with_mask(
                    pos.reshape(bs, n_nodes, -1),
                    node_mask,
                ).reshape(bs * n_nodes, -1)
            atom_hids.append(self._modules[f"node_{layer_idx}"](h))
            edge_hids.append(self._modules[f"edge_{layer_idx}"](edge_attr))

        atom_hids = torch.cat(atom_hids, dim=-1)
        edge_hids = torch.cat(edge_hids, dim=-1)
        atom_pred = self.node_pred_mlp(atom_hids).reshape(bs, n_nodes, -1) * node_mask
        edge_pred = torch.cat([self.edge_exist_mlp(edge_hids), self.edge_type_mlp(edge_hids)], dim=-1)

        edge_final = torch.zeros_like(edge_x).reshape(bs * n_nodes * n_nodes, -1)
        edge_final = model_utils.to_dense_edge_attr(edge_index, edge_pred, edge_final, bs, n_nodes)
        edge_final = 0.5 * (edge_final + edge_final.permute(0, 2, 1, 3))

        if self.pred_data:
            pos = pos * node_mask.reshape(-1, 1)
        else:
            pos = (pos - pos_init) * node_mask.reshape(-1, 1)

        if torch.any(torch.isnan(pos)):
            print("Warning: detected NaN in predicted positions; resetting to zero.")
            pos = torch.zeros_like(pos)

        pos = pos.reshape(bs, n_nodes, -1)
        pos = model_utils.remove_mean_with_mask(pos, node_mask)
        return torch.cat([pos, atom_pred], dim=2), edge_final
