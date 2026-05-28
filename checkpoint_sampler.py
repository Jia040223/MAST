#!/usr/bin/env python
"""
Checkpoint Sampler for MCTS-based Diffusion Sampling

This module runs diffusion over checkpoint segments and supports
optional reward-model guidance during MCTS inference.
"""
import random
import numpy as np
import torch
import torch.nn.functional as F
from models.utils import (
    sample_combined_position_feature_noise,
    sample_symmetric_edge_feature_noise,
    assert_mean_zero_with_mask
)
from sampling import expand_dims


def set_seed(seed):
    """Set all relevant random seeds."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


class SegmentSampler:
    """Run diffusion over a contiguous segment of the time grid."""
    
    def __init__(self, noise_scheduler, time_steps, model_pred_data=True, 
                 pred_edge=True, self_cond=False, cond_process_fn=None,
                 guidance_model=None, apply_to_pos_only=True, time_scale_guidance=True):
        """
        Args:
            noise_scheduler: diffusion noise scheduler
            time_steps: full sampling time grid
            model_pred_data: whether the model predicts x0 directly
            pred_edge: whether edges are generated jointly
            self_cond: whether self-conditioning is enabled
            cond_process_fn: post-processing function for self-conditioning
            guidance_model: optional MolSpectraGuidance model
            apply_to_pos_only: apply guidance to coordinates only
            time_scale_guidance: scale guidance by time
        """
        self.noise_scheduler = noise_scheduler
        self.time_steps = time_steps
        self.s_array = torch.cat([time_steps[1:], torch.zeros(1, device=time_steps.device)])
        self.model_pred_data = model_pred_data
        self.pred_edge = pred_edge
        self.self_cond = self_cond
        self.cond_process_fn = cond_process_fn
        
        # Guidance settings.
        self.guidance_model = guidance_model
        self.apply_to_pos_only = apply_to_pos_only
        self.time_scale_guidance = time_scale_guidance
        
    def run_segment(self, model, x, node_mask, edge_mask, start_idx, end_idx,
                    edge_x=None, context=None, spectra=None, seed=None,
                    cond_x=None, cond_edge_x=None, guidance_scale=0.0,
                    spec_feature=None):
        """Advance the sampler from `start_idx` to `end_idx`."""
        bs, n_nodes_x, _ = x.shape
        expected_edge_mask_size = bs * n_nodes_x * n_nodes_x
        if edge_mask.shape[0] != expected_edge_mask_size:
            raise ValueError(f"edge_mask size mismatch! Expected {expected_edge_mask_size} (bs={bs}, n_nodes={n_nodes_x}), got {edge_mask.shape[0]}")
        
        if seed is not None:
            set_seed(seed)
        
        if guidance_scale > 0 and self.guidance_model is not None:
            if spec_feature is None and spectra is not None:
                with torch.no_grad():
                    spec_feature = self.guidance_model.encode_spectra(spectra)
            
        bs = x.shape[0]
        last_pred_t = None
        edge_pred_t = None
        
        with torch.no_grad():
            for i in range(start_idx, end_idx):
                t = self.time_steps[i]
                s = self.s_array[i]
                
                alpha_t, sigma_t = self.noise_scheduler.marginal_prob(t)
                alpha_s, sigma_s = self.noise_scheduler.marginal_prob(s)
                
                alpha_t_given_s = alpha_t / alpha_s
                sigma2_t_given_s = sigma_t ** 2 - alpha_t_given_s ** 2 * sigma_s ** 2
                sigma_t_given_s = torch.sqrt(sigma2_t_given_s)
                sigma = sigma_t_given_s * sigma_s / sigma_t
                
                vec_t = torch.ones(bs, device=x.device) * t
                noise_level = torch.ones(bs, device=x.device) * torch.log(alpha_t ** 2 / sigma_t ** 2)
                
                if self.pred_edge:
                    if self.self_cond:
                        assert self.model_pred_data
                        pred_t, edge_pred_t = model(vec_t, x, node_mask, edge_mask, edge_x=edge_x,
                                                   noise_level=noise_level, cond_x=cond_x,
                                                   cond_edge_x=cond_edge_x, context=context, spectra=spectra)
                        cond_x, cond_edge_x = self.cond_process_fn(pred_t, edge_pred_t)
                    else:
                        pred_t, edge_pred_t = model(vec_t, x, node_mask, edge_mask, edge_x=edge_x,
                                                   noise_level=noise_level, context=context, spectra=spectra)
                else:
                    if self.self_cond:
                        assert self.model_pred_data
                        pred_t = model(vec_t, x, node_mask, edge_mask, noise_level=noise_level,
                                      cond_x=cond_x, context=context, spectra=spectra)
                    else:
                        pred_t = model(vec_t, x, node_mask, edge_mask, noise_level=noise_level,
                                      context=context, spectra=spectra)
                
                # NaN guard: detect both explicit NaN and the model's
                # internal NaN-to-zero reset (pos all zeros after NaN).
                # The model prints "Warning: detected nan in position,
                # resetting to zero." and returns zeros for pos, which
                # hides the NaN from a simple isnan check.
                pred_bad = torch.any(torch.isnan(pred_t))
                if not pred_bad:
                    pos_pred = pred_t[:, :, :3]
                    masked_pos = pos_pred * node_mask
                    pred_bad = (masked_pos.abs().sum() < 1e-8)

                if pred_bad:
                    if last_pred_t is not None and not torch.any(torch.isnan(last_pred_t)):
                        pred_t = last_pred_t
                    # else: keep the zero prediction as last resort
                    if self.pred_edge and edge_pred_t is not None and torch.any(torch.isnan(edge_pred_t)):
                        edge_pred_t = torch.zeros_like(edge_pred_t)

                last_pred_t = pred_t
                
                # Node update.
                if self.model_pred_data:
                    x_mean = expand_dims((alpha_t_given_s * sigma_s ** 2 / sigma_t ** 2).repeat(bs), x.dim()) * x \
                             + expand_dims((alpha_s * sigma2_t_given_s / sigma_t ** 2).repeat(bs), pred_t.dim()) * pred_t
                else:
                    x_mean = x / expand_dims(alpha_t_given_s.repeat(bs), x.dim()) \
                             - expand_dims((sigma2_t_given_s / alpha_t_given_s / sigma_t).repeat(bs), pred_t.dim()) * pred_t
                
                # Apply guidance in the denoised space.
                if guidance_scale > 0 and self.guidance_model is not None and spec_feature is not None:
                    guidance_grad = self._compute_guidance_gradient(
                        pred_t.detach(), node_mask, spec_feature, t.item()
                    )
                    if guidance_grad is not None:
                        if self.time_scale_guidance:
                            t_scale = t.item() ** 2
                        else:
                            t_scale = 1.0
                        
                        scaled_guidance = guidance_scale * t_scale * guidance_grad
                        
                        if self.apply_to_pos_only:
                            x_mean[:, :, :3] = x_mean[:, :, :3] - scaled_guidance[:, :, :3]
                        else:
                            x_mean = x_mean - scaled_guidance
                        
                        pos_mean = x_mean[:, :, :3]
                        pos_mean = pos_mean - (pos_mean * node_mask).sum(dim=1, keepdim=True) / node_mask.sum(dim=1, keepdim=True).clamp(min=1)
                        x_mean[:, :, :3] = pos_mean
                
                x = x_mean + expand_dims(sigma.repeat(bs), x_mean.dim()) * \
                    sample_combined_position_feature_noise(bs, x_mean.shape[1], x_mean.shape[2] - 3, node_mask)
                
                x[:, :, :3] = x[:, :, :3].clamp(-100, 100)

                # Edge update.
                if self.pred_edge:
                    if self.model_pred_data:
                        edge_x_mean = expand_dims((alpha_t_given_s * sigma_s**2 / sigma_t ** 2).repeat(bs), edge_x.dim()) \
                                      * edge_x + expand_dims((alpha_s * sigma2_t_given_s / sigma_t ** 2).repeat(bs),
                                                            edge_pred_t.dim()) * edge_pred_t
                    else:
                        edge_x_mean = edge_x / expand_dims(alpha_t_given_s.repeat(bs), edge_x.dim()) - expand_dims(
                            (sigma2_t_given_s / alpha_t_given_s / sigma_t).repeat(bs), edge_pred_t.dim()) * edge_pred_t
                    edge_x = edge_x_mean + expand_dims(sigma.repeat(bs), edge_x_mean.dim()) * \
                             sample_symmetric_edge_feature_noise(bs, edge_x_mean.shape[1], edge_x_mean.shape[-1], edge_mask)
        
        if self.pred_edge:
            return x, edge_x, last_pred_t, cond_x, cond_edge_x
        else:
            return x, None, last_pred_t, cond_x, cond_edge_x
    
    def _compute_guidance_gradient(self, pred_t, node_mask, spec_feature, t_value):
        """Compute the reward-model guidance gradient used during MCTS."""
        try:
            with torch.enable_grad():
                bs, n_nodes, node_nf = pred_t.shape
                device = pred_t.device
                
                pos_pred = pred_t[:, :, :3].detach().clone()
                pos_pred.requires_grad_(True)
                
                atom_pred = pred_t[:, :, 3:].detach()
                if atom_pred.shape[-1] > 5:
                    atom_pred = atom_pred[:, :, :-1]
                atom_pred = atom_pred[:, :, :5]
                
                node_mask_squeezed = node_mask.squeeze(-1)
                n_valid_per_batch = node_mask_squeezed.sum(dim=1).long()
                
                type_map_tensor = torch.tensor([1, 6, 7, 8, 9], dtype=torch.long, device=device)
                
                pos_all, z_all, batch_all = [], [], []
                for b in range(bs):
                    n_valid = n_valid_per_batch[b].item()
                    if n_valid == 0:
                        continue
                    pos_all.append(pos_pred[b, :n_valid, :])
                    atom_type_idx = atom_pred[b, :n_valid, :].argmax(dim=-1).clamp(0, 4)
                    z_all.append(type_map_tensor[atom_type_idx])
                    batch_all.append(torch.full((n_valid,), b, dtype=torch.long, device=device))
                
                if len(pos_all) == 0:
                    return None
                
                pos = torch.cat(pos_all, dim=0)
                z = torch.cat(z_all, dim=0)
                batch_idx = torch.cat(batch_all, dim=0)
                
                mol_feature = self.guidance_model.encode_structure(z, pos, batch_idx)
                loss = self.guidance_model.compute_guidance_loss(spec_feature, mol_feature)
                loss.backward()
                
                pos_grad = pos_pred.grad
                if pos_grad is None:
                    return None
                
                gradient = torch.zeros_like(pred_t)
                gradient[:, :, :3] = pos_grad * node_mask
                return gradient.detach()
                
        except Exception:
            return None
    
    def run_full(self, model, z_T, node_mask, edge_mask, edge_z_T=None, 
                 context=None, spectra=None, seed=None):
        """Run the full diffusion trajectory without checkpoint splitting."""
        if seed is not None:
            set_seed(seed)
            
        x = z_T
        edge_x = edge_z_T
        cond_x, cond_edge_x = None, None
        
        x, edge_x, _, cond_x, cond_edge_x = self.run_segment(
            model, x, node_mask, edge_mask,
            start_idx=0, end_idx=len(self.time_steps),
            edge_x=edge_x, context=context, spectra=spectra,
            seed=None,
            cond_x=cond_x, cond_edge_x=cond_edge_x
        )
        
        if self.pred_edge:
            return x, edge_x
        else:
            return x, None


class MCTSState:
    """Intermediate diffusion state stored at each tree node."""
    
    def __init__(self, x, edge_x, node_mask, edge_mask, segment_idx,
                 cond_x=None, cond_edge_x=None, seed_history=None):
        """
        Args:
            x: current node state
            edge_x: current edge state
            node_mask: node mask
            edge_mask: edge mask
            segment_idx: current segment index
            cond_x, cond_edge_x: self-conditioning tensors
            seed_history: action history for the path
        """
        self.x = x
        self.edge_x = edge_x
        self.node_mask = node_mask
        self.edge_mask = edge_mask
        self.segment_idx = segment_idx
        self.cond_x = cond_x
        self.cond_edge_x = cond_edge_x
        self.seed_history = seed_history if seed_history is not None else []
        
    def clone(self):
        """Clone the current state."""
        return MCTSState(
            x=self.x.clone(),
            edge_x=self.edge_x.clone() if self.edge_x is not None else None,
            node_mask=self.node_mask,
            edge_mask=self.edge_mask,
            segment_idx=self.segment_idx,
            cond_x=self.cond_x.clone() if self.cond_x is not None else None,
            cond_edge_x=self.cond_edge_x.clone() if self.cond_edge_x is not None else None,
            seed_history=self.seed_history.copy()
        )
    
    def is_terminal(self, total_segments):
        """Return whether the trajectory already reached the terminal segment."""
        return self.segment_idx >= total_segments
    
    def __repr__(self):
        return f"MCTSState(segment_idx={self.segment_idx}, seed_history={self.seed_history})"


class CheckpointSamplerForMCTS:
    """Checkpoint-level interface used by the MCTS search loop."""
    
    def __init__(self, noise_scheduler, time_steps, checkpoints, 
                 model_pred_data=True, pred_edge=True, self_cond=False, 
                 cond_process_fn=None, guidance_model=None,
                 apply_to_pos_only=True, time_scale_guidance=True,
                 reinit_noise_at_first_segment=False, node_nf=None, edge_nf=None):
        """
        Args:
            noise_scheduler: diffusion noise scheduler
            time_steps: sampling time grid
            checkpoints: checkpoint indices
            guidance_model: optional MolSpectraGuidance model
            reinit_noise_at_first_segment: resample the initial noise on the first segment
            node_nf: node feature dimension
            edge_nf: edge feature dimension
            remaining arguments match `SegmentSampler`
        """
        self.segment_sampler = SegmentSampler(
            noise_scheduler, time_steps, model_pred_data, 
            pred_edge, self_cond, cond_process_fn,
            guidance_model=guidance_model,
            apply_to_pos_only=apply_to_pos_only,
            time_scale_guidance=time_scale_guidance
        )
        self.checkpoints = checkpoints
        self.n_segments = len(checkpoints) - 1
        self.time_steps = time_steps
        self.pred_edge = pred_edge
        self.guidance_model = guidance_model
        
        self.reinit_noise_at_first_segment = reinit_noise_at_first_segment
        self.node_nf = node_nf
        self.edge_nf = edge_nf
        
        self._spec_feature_cache = None
        
    def get_segment_range(self, segment_idx):
        """Return the time-index range for a segment."""
        if segment_idx < 0 or segment_idx >= self.n_segments:
            raise IndexError(f"Segment index {segment_idx} out of range")
        return self.checkpoints[segment_idx], self.checkpoints[segment_idx + 1]
    
    def cache_spectra_features(self, spectra):
        """Cache encoded spectra features for repeated guidance calls."""
        if self.guidance_model is not None and spectra is not None:
            with torch.no_grad():
                self._spec_feature_cache = self.guidance_model.encode_spectra(spectra)
        return self._spec_feature_cache
    
    def step(self, model, state, action, context=None, spectra=None):
        """Advance one checkpoint segment under the given action."""
        if state.is_terminal(self.n_segments):
            return state, None
        
        if isinstance(action, tuple):
            action_seed, guidance_scale = action
        else:
            action_seed = action
            guidance_scale = 0.0
        
        current_x = state.x
        current_edge_x = state.edge_x
        
        if state.segment_idx == 0 and self.reinit_noise_at_first_segment:
            if self.node_nf is not None:
                set_seed(action_seed)
                bs = state.node_mask.size(0)
                max_n_nodes = state.node_mask.size(1)
                current_x = sample_combined_position_feature_noise(
                    bs, max_n_nodes, self.node_nf, state.node_mask
                )
                assert_mean_zero_with_mask(current_x[:, :, :3], state.node_mask)
                
                if self.pred_edge and self.edge_nf is not None:
                    current_edge_x = sample_symmetric_edge_feature_noise(
                        bs, max_n_nodes, self.edge_nf, state.edge_mask
                    )
        
        start_idx, end_idx = self.get_segment_range(state.segment_idx)
        
        x, edge_x, pred_t, cond_x, cond_edge_x = self.segment_sampler.run_segment(
            model=model,
            x=current_x,
            node_mask=state.node_mask,
            edge_mask=state.edge_mask,
            start_idx=start_idx,
            end_idx=end_idx,
            edge_x=current_edge_x,
            context=context,
            spectra=spectra,
            seed=action_seed,
            cond_x=state.cond_x,
            cond_edge_x=state.cond_edge_x,
            guidance_scale=guidance_scale,
            spec_feature=self._spec_feature_cache
        )
        
        new_seed_history = state.seed_history + [action]
        
        new_state = MCTSState(
            x=x,
            edge_x=edge_x,
            node_mask=state.node_mask,
            edge_mask=state.edge_mask,
            segment_idx=state.segment_idx + 1,
            cond_x=cond_x,
            cond_edge_x=cond_edge_x,
            seed_history=new_seed_history
        )
        
        return new_state, pred_t
    
    def rollout(self, model, state, context=None, spectra=None, 
                default_seed=42, default_guidance_scale=0.0, fast_steps=20):
        """Roll out from the current state to the terminal segment."""
        if default_seed is not None:
            set_seed(default_seed)
        
        current_segment = state.segment_idx
        if current_segment >= len(self.checkpoints) - 1:
            current_step_idx = self.checkpoints[-1] - 1
        else:
            current_step_idx = self.checkpoints[current_segment]
        
        t_current = self.time_steps[current_step_idx]
        t_end = self.time_steps[-1]
        
        fast_time_steps = torch.linspace(
            t_current.item(), t_end.item(), fast_steps + 1, 
            device=state.x.device
        )
        
        x = state.x.clone()
        edge_x = state.edge_x.clone() if state.edge_x is not None else None
        node_mask = state.node_mask
        edge_mask = state.edge_mask
        bs = x.shape[0]
        
        cond_x = state.cond_x
        cond_edge_x = state.cond_edge_x
        
        pred_t = None
        last_pred_t = None
        
        with torch.no_grad():
            for i in range(fast_steps):
                t = fast_time_steps[i]
                s = fast_time_steps[i + 1]
                
                alpha_t, sigma_t = self.segment_sampler.noise_scheduler.marginal_prob(t)
                alpha_s, sigma_s = self.segment_sampler.noise_scheduler.marginal_prob(s)
                
                alpha_t_given_s = alpha_t / alpha_s
                sigma2_t_given_s = sigma_t ** 2 - alpha_t_given_s ** 2 * sigma_s ** 2
                sigma_t_given_s = torch.sqrt(sigma2_t_given_s)
                sigma = sigma_t_given_s * sigma_s / sigma_t
                
                vec_t = torch.ones(bs, device=x.device) * t
                noise_level = torch.ones(bs, device=x.device) * torch.log(alpha_t ** 2 / sigma_t ** 2)
                
                edge_pred_t = None
                if self.segment_sampler.pred_edge:
                    pred_t, edge_pred_t = model(vec_t, x, node_mask, edge_mask, edge_x=edge_x,
                                               noise_level=noise_level, cond_x=cond_x,
                                               cond_edge_x=cond_edge_x, context=context, spectra=spectra)
                    if self.segment_sampler.self_cond and self.segment_sampler.cond_process_fn is not None:
                        cond_x, cond_edge_x = self.segment_sampler.cond_process_fn(pred_t, edge_pred_t)
                else:
                    pred_t = model(vec_t, x, node_mask, edge_mask, noise_level=noise_level,
                                  cond_x=cond_x, context=context, spectra=spectra)
                    if self.segment_sampler.self_cond and self.segment_sampler.cond_process_fn is not None:
                        cond_x, _ = self.segment_sampler.cond_process_fn(pred_t, None)

                pred_bad = torch.any(torch.isnan(pred_t))
                if not pred_bad:
                    pos_pred = pred_t[:, :, :3]
                    masked_pos = pos_pred * node_mask
                    pred_bad = (masked_pos.abs().sum() < 1e-8)

                if pred_bad:
                    if last_pred_t is not None and not torch.any(torch.isnan(last_pred_t)):
                        pred_t = last_pred_t
                    if edge_pred_t is not None and torch.any(torch.isnan(edge_pred_t)):
                        edge_pred_t = torch.zeros_like(edge_pred_t)

                last_pred_t = pred_t

                # The final step uses the mean update without noise injection.
                is_last_step = (i == fast_steps - 1)
                x_mean = expand_dims((alpha_t_given_s * sigma_s ** 2 / sigma_t ** 2).repeat(bs), x.dim()) * x \
                         + expand_dims((alpha_s * sigma2_t_given_s / sigma_t ** 2).repeat(bs), pred_t.dim()) * pred_t
                
                if not is_last_step:
                    x = x_mean + expand_dims(sigma.repeat(bs), x_mean.dim()) * \
                        sample_combined_position_feature_noise(bs, x_mean.shape[1], x_mean.shape[2] - 3, node_mask)
                else:
                    x = x_mean
                
                x[:, :, :3] = x[:, :, :3].clamp(-100, 100)

                if self.segment_sampler.pred_edge and edge_x is not None:
                    edge_x_mean = expand_dims((alpha_t_given_s * sigma_s**2 / sigma_t ** 2).repeat(bs), edge_x.dim()) \
                                  * edge_x + expand_dims((alpha_s * sigma2_t_given_s / sigma_t ** 2).repeat(bs),
                                                        edge_pred_t.dim()) * edge_pred_t
                    if not is_last_step:
                        edge_x = edge_x_mean + expand_dims(sigma.repeat(bs), edge_x_mean.dim()) * \
                                 sample_symmetric_edge_feature_noise(bs, edge_x_mean.shape[1], edge_x_mean.shape[-1], edge_mask)
                    else:
                        edge_x = edge_x_mean
        
        final_state = MCTSState(
            x=x,
            edge_x=edge_x,
            node_mask=node_mask,
            edge_mask=edge_mask,
            segment_idx=self.n_segments,
            cond_x=cond_x,
            cond_edge_x=cond_edge_x,
            seed_history=state.seed_history.copy() if state.seed_history else []
        )
        
        return final_state, pred_t
    
    def create_initial_state(self, z_T, edge_z_T, node_mask, edge_mask):
        """Create the root state for MCTS."""
        return MCTSState(
            x=z_T,
            edge_x=edge_z_T,
            node_mask=node_mask,
            edge_mask=edge_mask,
            segment_idx=0
        )
