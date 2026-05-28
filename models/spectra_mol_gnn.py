"""Spectra-conditioned diffusion graph transformer."""

from torch import nn
import torch
from . import utils
from .layers import *
from .mol_gnn import EquivariantMixBlock, modulate
from torch_geometric.utils import dense_to_sparse
import functools
from torch_scatter import scatter

from spectra.Sp import SpecFormer


@utils.register_model(name='Spectra_DGT_concat')
class Spectra_DGT_concat(nn.Module):
    """
    Diffusion graph transformer with spectral conditioning.
    """
    
    def __init__(self, config):
        super().__init__()
        
        # Base architecture settings.
        in_node_dim = config.data.atom_types + int(config.model.include_fc_charge)
        hidden_dim = config.model.nf
        edge_hidden_dim = config.model.nf // 4
        n_heads = config.model.n_heads
        dropout = config.model.dropout
        self.dist_gbf = dist_gbf = config.model.dist_gbf
        gbf_name = config.model.gbf_name
        self.edge_th = config.model.edge_quan_th
        n_extra_heads = config.model.n_extra_heads
        self.CoM = config.model.CoM
        mlp_ratio = config.model.mlp_ratio
        self.spatial_cut_off = config.model.spatial_cut_off
        softmax_inf = config.model.softmax_inf
        
        if dist_gbf:
            dist_dim = edge_hidden_dim
        else:
            dist_dim = 1
        in_edge_dim = config.model.edge_ch * 2 + dist_dim
        self.cond_time = cond_time = config.model.cond_time
        self.n_layers = n_layers = config.model.n_layers
        self.pred_data = config.model.pred_data
        time_dim = hidden_dim * 4
        self.dist_dim = dist_dim
        self.config = config
        
        # Node and edge embeddings.
        self.node_emb = nn.Linear(in_node_dim * 2, hidden_dim)
        self.edge_emb = nn.Linear(in_edge_dim, edge_hidden_dim)
        
        if self.dist_gbf:
            self.dist_layer = eval(gbf_name)(dist_dim, time_dim)
        
        cat_node_dim = (hidden_dim * 2) // n_layers
        cat_edge_dim = (edge_hidden_dim * 2) // n_layers
        
        # Equivariant message-passing blocks.
        for i in range(n_layers):
            self.add_module("e_block_%d" % i, EquivariantMixBlock(
                hidden_dim, edge_hidden_dim, time_dim, n_extra_heads,
                n_heads, cond_time, dist_gbf, softmax_inf, 
                mlp_ratio=mlp_ratio, dropout=dropout, gbf_name=gbf_name
            ))
            self.add_module("node_%d" % i, nn.Linear(hidden_dim, cat_node_dim))
            self.add_module("edge_%d" % i, nn.Linear(edge_hidden_dim, cat_edge_dim))
        
        # Prediction heads.
        self.node_pred_mlp = nn.Sequential(
            nn.Linear(cat_node_dim * n_layers + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, in_node_dim)
        )
        self.edge_type_mlp = nn.Sequential(
            nn.Linear(cat_edge_dim * n_layers + edge_hidden_dim, edge_hidden_dim),
            nn.SiLU(),
            nn.Linear(edge_hidden_dim, edge_hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(edge_hidden_dim // 2, config.model.edge_ch - 1)
        )
        self.edge_exist_mlp = nn.Sequential(
            nn.Linear(cat_edge_dim * n_layers + edge_hidden_dim, edge_hidden_dim),
            nn.SiLU(),
            nn.Linear(edge_hidden_dim, edge_hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(edge_hidden_dim // 2, 1)
        )
        
        # Time embedding MLP
        if cond_time:
            learned_dim = 16
            sinu_pos_emb = LearnedSinusodialposEmb(learned_dim)
            self.time_mlp = nn.Sequential(
                sinu_pos_emb,
                nn.Linear(learned_dim + 1, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim)
            )
        
        # Spectral conditioning.
        spectra_modalities = getattr(config.model, 'spectra_modalities', ('uv', 'ir', 'raman'))
        spectra_output_dim = getattr(config.model, 'spectra_output_dim', 128)
        spectra_d_model = getattr(config.model, 'spectra_d_model', 128)
        spectra_depth = getattr(config.model, 'spectra_depth', 4)
        spectra_n_heads = getattr(config.model, 'spectra_n_heads', 8)
        self.use_spectra = getattr(config.model, 'use_spectra', True)
        self.spectra_fusion = getattr(config.model, 'spectra_fusion', 'add')  # 'add', 'gate', 'concat'
        
        # Patch settings for the three spectral modalities.
        patch_lens = getattr(config.model, 'spectra_patch_lens', (32, 64, 64))
        strides = getattr(config.model, 'spectra_strides', (16, 32, 32))
        
        if self.use_spectra:
            # Spectral encoder.
            self.spectra_encoder = SpecFormer(
                patch_len=list(patch_lens),
                stride=list(strides),
                output_dim=spectra_output_dim,
                d_model=spectra_d_model,
                n_layers=spectra_depth,
                n_heads=spectra_n_heads,
                dropout=dropout,
                act='gelu'
            )
            
            self.time_dim = time_dim
            
            # Project spectral features into the time-conditioning space.
            if self.spectra_fusion == 'add':
                self.spectra_mlp = nn.Sequential(
                    nn.Linear(spectra_output_dim, time_dim),
                    nn.GELU(),
                    nn.Linear(time_dim, time_dim)
                )
            elif self.spectra_fusion == 'gate':
                self.spectra_mlp = nn.Sequential(
                    nn.Linear(spectra_output_dim, time_dim),
                    nn.GELU(),
                    nn.Linear(time_dim, time_dim)
                )
                self.gate_mlp = nn.Sequential(
                    nn.Linear(spectra_output_dim, time_dim),
                    nn.Sigmoid()
                )
            elif self.spectra_fusion == 'concat':
                self.spectra_mlp = nn.Sequential(
                    nn.Linear(spectra_output_dim, time_dim),
                    nn.GELU(),
                    nn.Linear(time_dim, time_dim)
                )
                self.fusion_proj = nn.Linear(time_dim * 2, time_dim)
            else:
                raise ValueError(f"Unknown spectra_fusion: {self.spectra_fusion}")
    
    def forward(self, t, xh, node_mask, edge_mask, context=None, *args, **kwargs):
        """
        Forward pass with spectral conditioning
        
        Parameters
        ----------
        t: [B] time steps in [0, 1]
        xh: [B, N, ch1] atom feature (positions, types, formal charges)
        node_mask: [B, N, 1]
        edge_mask: [B*N*N, 1]
        context: optional conditional context
        kwargs: 
            - 'edge_x': [B, N, N, ch2] edge features
            - 'cond_x': [B, N, ch1] self-conditioning node features
            - 'cond_edge_x': [B, N, N, ch2] self-conditioning edge features
            - 'noise_level': [B, 1] noise level for time embedding
            - 'spectra': optional list of [B, L_i] spectral tensors
        
        Returns
        -------
        node_pred: [B, N, ch1] predicted node features
        edge_pred: [B, N, N, ch2] predicted edge features
        """
        edge_x = kwargs['edge_x']
        cond_x = kwargs.get('cond_x', None)
        cond_edge_x = kwargs.get('cond_edge_x', None)
        spectra = kwargs.get('spectra', None)
        
        bs, n_nodes, dims = xh.shape
        pos_init = pos = xh[:, :, 0:3].clone().reshape(bs * n_nodes, -1)
        h = xh[:, :, 3:].clone().reshape(bs * n_nodes, -1)
        
        adj_mask = edge_mask.reshape(bs, n_nodes, n_nodes)
        dense_index = adj_mask.nonzero(as_tuple=True)
        edge_index, _ = dense_to_sparse(adj_mask)
        
        # Self-conditioning features.
        if cond_x is None:
            cond_x = torch.zeros_like(xh)
            cond_edge_x = torch.zeros_like(edge_x)
            cond_adj_2d = torch.ones((edge_index.size(1), 1), device=edge_x.device)
        else:
            with torch.no_grad():
                cond_adj_2d = cond_edge_x[dense_index][:, 0:1].clone()
                cond_adj_2d[cond_adj_2d >= self.edge_th] = 1.
                cond_adj_2d[cond_adj_2d < self.edge_th] = 0.
        
        # Concatenate self-conditioning node features.
        cond_pos = cond_x[:, :, 0:3].clone().reshape(bs * n_nodes, -1)
        cond_h = cond_x[:, :, 3:].clone().reshape(bs * n_nodes, -1)
        h = torch.cat([h, cond_h], dim=-1)
        
        # Spectral conditioning.
        spectra_emb = None
        if self.use_spectra and spectra is not None:
            spectra_tokens, spectra_cls = self.spectra_encoder(spectra)
            spectra_emb = self.spectra_mlp(spectra_cls)  # [B, time_dim]
        
        # Time conditioning.
        if self.cond_time:
            noise_level = kwargs['noise_level']
            time_emb = self.time_mlp(noise_level)  # [B, time_dim]
            
            if spectra_emb is not None:
                if self.spectra_fusion == 'add':
                    time_emb = time_emb + spectra_emb
                elif self.spectra_fusion == 'gate':
                    gate = self.gate_mlp(spectra_cls)
                    time_emb = time_emb * gate + spectra_emb * (1 - gate)
                elif self.spectra_fusion == 'concat':
                    time_emb = self.fusion_proj(torch.cat([time_emb, spectra_emb], dim=-1))
            
            node_time_emb = time_emb.unsqueeze(1).expand(-1, n_nodes, -1).reshape(bs * n_nodes, -1)
            edge_batch_id = torch.div(edge_index[0], n_nodes, rounding_mode='floor')
            edge_time_emb = time_emb[edge_batch_id]
        else:
            node_time_emb = None
            edge_time_emb = None
        
        # Edge feature construction.
        distances, cond_adj_spatial = utils.coord2diff_adj(cond_pos, edge_index, self.spatial_cut_off)
        if distances.sum() == 0:
            distances = distances.repeat(1, self.dist_dim)
        else:
            if self.dist_gbf:
                distances = self.dist_layer(distances, edge_time_emb)
        
        cur_edge_attr = edge_x[dense_index]
        cond_edge_attr = cond_edge_x[dense_index]
        
        extra_adj = torch.cat([cond_adj_2d, cond_adj_spatial], dim=-1)
        edge_attr = torch.cat([cur_edge_attr, cond_edge_attr, distances], dim=-1)
        
        h = self.node_emb(h)
        edge_attr = self.edge_emb(edge_attr)
        
        atom_hids = [h]
        edge_hids = [edge_attr]
        for i in range(0, self.n_layers):
            h, edge_attr, pos = self._modules['e_block_%d' % i](
                pos, h, edge_attr, edge_index, node_mask.reshape(-1, 1),
                extra_adj, node_time_emb, edge_time_emb
            )
            if self.CoM:
                pos = utils.remove_mean_with_mask(pos.reshape(bs, n_nodes, -1), node_mask).reshape(bs * n_nodes, -1)
            atom_hids.append(self._modules['node_%d' % i](h))
            edge_hids.append(self._modules['edge_%d' % i](edge_attr))
        
        atom_hids = torch.cat(atom_hids, dim=-1)
        edge_hids = torch.cat(edge_hids, dim=-1)
        atom_pred = self.node_pred_mlp(atom_hids).reshape(bs, n_nodes, -1) * node_mask
        edge_pred = torch.cat([self.edge_exist_mlp(edge_hids), self.edge_type_mlp(edge_hids)], dim=-1)
        
        edge_final = torch.zeros_like(edge_x).reshape(bs * n_nodes * n_nodes, -1)
        edge_final = utils.to_dense_edge_attr(edge_index, edge_pred, edge_final, bs, n_nodes)
        edge_final = 0.5 * (edge_final + edge_final.permute(0, 2, 1, 3))
        
        if self.pred_data:
            pos = pos * node_mask.reshape(-1, 1)
        else:
            pos = (pos - pos_init) * node_mask.reshape(-1, 1)
        
        if torch.any(torch.isnan(pos)):
            print('Warning: detected nan in position, resetting to zero.')
            pos = torch.zeros_like(pos)
        
        pos = pos.reshape(bs, n_nodes, -1)
        pos = utils.remove_mean_with_mask(pos, node_mask)
        
        return torch.cat([pos, atom_pred], dim=2), edge_final

@utils.register_model(name='Spectra_DGT')
class Spectra_DGT(Spectra_DGT_concat):
    """
    Simplified variant without self-conditioning.
    """
    
    def forward(self, t, xh, node_mask, edge_mask, context=None, *args, **kwargs):
        if 'cond_x' in kwargs:
            kwargs['cond_x'] = None
        if 'cond_edge_x' in kwargs:
            kwargs['cond_edge_x'] = None
        
        return super().forward(t, xh, node_mask, edge_mask, context, *args, **kwargs)
