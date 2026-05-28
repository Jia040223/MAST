import random

import numpy as np
import torch
from torch.nn import functional as F

from models.utils import (
    assert_mean_zero_with_mask,
    sample_combined_position_feature_noise,
    sample_gaussian_with_mask,
    sample_symmetric_edge_feature_noise,
)
from utils import get_self_cond_fn


def mol_process(one_hot, positions, formal_charges, n_nodes, edge_types=None):
    mol_list = []
    batch_size = one_hot.shape[0]
    for i in range(batch_size):
        atom_type = one_hot[i].argmax(1).cpu().detach()
        pos = positions[i].cpu().detach()
        atom_type = atom_type[: n_nodes[i]]
        pos = pos[: n_nodes[i]]
        if edge_types is not None:
            edge_type = edge_types[i][: n_nodes[i], : n_nodes[i]].cpu().detach()
            if formal_charges.shape[-1] != 0:
                fc = formal_charges[i][: n_nodes[i], 0].long().cpu().detach()
            else:
                fc = formal_charges[i][: n_nodes[i]].cpu().detach()
            mol_list.append((pos, atom_type, edge_type, fc))
        else:
            mol_list.append((pos, atom_type))
    return mol_list


def post_process(
    xh,
    atom_types,
    include_charge,
    node_mask,
    inverse_scaler,
    edge_x=None,
    edge_mask=None,
    compress_edge=False,
):
    positions = xh[:, :, :3]
    if include_charge:
        formal_charges = xh[:, :, -1:]
        atom_logits = xh[:, :, 3:-1]
    else:
        formal_charges = torch.zeros(0, device=xh.device)
        atom_logits = xh[:, :, 3:]

    if edge_x is not None:
        positions, atom_logits, formal_charges, edge_logits = inverse_scaler(
            positions, atom_logits, formal_charges, node_mask, edge_x, edge_mask
        )
    else:
        positions, atom_logits, formal_charges = inverse_scaler(
            positions, atom_logits, formal_charges, node_mask
        )

    atom_one_hot = F.one_hot(torch.argmax(atom_logits, dim=2), atom_types) * node_mask
    formal_charges = torch.round(formal_charges).long() * node_mask

    if edge_x is None:
        return positions, atom_one_hot, formal_charges

    if compress_edge:
        edge_exist = edge_logits[:, :, :, 0]
        edge_exist = (edge_exist >= 0.5).float()
        edge_type = edge_logits[:, :, :, 1] * 3.0
        edge_type[edge_type >= 2.5] = 3.0
        edge_type[torch.bitwise_and(edge_type >= 1.5, edge_type < 2.5)] = 2.0
        edge_type[torch.bitwise_and(edge_type >= 0.5, edge_type < 1.5)] = 1.0
        edge_type[edge_type < 0.5] = 0.0
        edge_type = edge_exist * edge_type
        if edge_logits.size(-1) == 3:
            aromatic = (edge_logits[:, :, :, 2] >= 0.5).float() * edge_exist
            edge_type[torch.bitwise_and(aromatic > 0.0, edge_type == 0.0)] = 4.0
        edge_logits = edge_type
    else:
        edge_exists = torch.sum(edge_logits > 0.5, dim=-1) != 0
        edge_logits = (torch.argmax(edge_logits, dim=-1) + 1.0) * edge_exists

    return positions, atom_one_hot, formal_charges, edge_logits


def expand_dims(v, dims):
    return v[(...,) + (None,) * (dims - 1)]


def get_sampling_fn(config, noise_scheduler, nodes_dist, batch_size, n_samples, inverse_scaler, eps=1e-3):
    device = config.device
    sampling_steps = int(config.sampling.steps)
    atom_types = config.data.atom_types
    include_fc = config.model.include_fc_charge
    node_nf = atom_types + int(include_fc)
    pred_edge = config.pred_edge
    edge_nf = config.model.edge_ch
    compress_edge = config.data.compress_edge
    self_cond = config.model.self_cond

    if config.sampling.method != "ancestral":
        raise ValueError("Only ancestral sampling is supported in the open-source release.")

    time_steps = torch.linspace(noise_scheduler.T, eps, sampling_steps, device=device)
    sampler = AncestralSampler(
        noise_scheduler,
        time_steps,
        config.model.pred_data,
        pred_edge,
        self_cond,
        get_self_cond_fn(config),
    )
    num_rounds = int(np.ceil(n_samples / batch_size))

    def sampling_fn(model):
        model.eval()
        processed = []
        with torch.no_grad():
            n_nodes_all = nodes_dist.sample(num_rounds * batch_size)
            for round_idx in range(num_rounds):
                n_nodes = n_nodes_all[round_idx * batch_size : (round_idx + 1) * batch_size]
                max_n_nodes = max(n_nodes)

                node_mask = torch.zeros(batch_size, max_n_nodes)
                for i in range(batch_size):
                    node_mask[i, : n_nodes[i]] = 1
                edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
                diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
                edge_mask *= diag_mask
                edge_mask = edge_mask.view(batch_size * max_n_nodes * max_n_nodes, 1).to(device)
                node_mask = node_mask.unsqueeze(2).to(device)

                z = sample_combined_position_feature_noise(batch_size, max_n_nodes, node_nf, node_mask)
                assert_mean_zero_with_mask(z[:, :, :3], node_mask)
                edge_z = None
                if pred_edge:
                    edge_z = sample_symmetric_edge_feature_noise(batch_size, max_n_nodes, edge_nf, edge_mask)

                x_node, x_edge = sampler.sampling(model, z, node_mask, edge_mask, edge_z, context=None)
                positions, one_hot, formal_charges, edge_types = post_process(
                    x_node,
                    atom_types,
                    include_fc,
                    node_mask,
                    inverse_scaler,
                    x_edge,
                    edge_mask,
                    compress_edge,
                )
                assert_mean_zero_with_mask(positions, node_mask)
                processed.extend(mol_process(one_hot, positions, formal_charges, n_nodes, edge_types))

        random.shuffle(processed)
        return processed[:n_samples]

    return sampling_fn


class AncestralSampler:
    def __init__(self, noise_scheduler, time_steps, model_pred_data, pred_edge=False, self_cond=False, cond_process_fn=None):
        self.noise_scheduler = noise_scheduler
        self.t_array = time_steps
        self.s_array = torch.cat([time_steps[1:], torch.zeros(1, device=time_steps.device)])
        self.model_pred_data = model_pred_data
        self.pred_edge = pred_edge
        self.self_cond = self_cond
        self.cond_process_fn = cond_process_fn

    def sampling(self, model, z_t, node_mask, edge_mask, edge_z_t=None, context=None, spectra=None, substructure_spectra=None):
        x = z_t
        edge_x = edge_z_t
        batch_size = z_t.shape[0]
        cond_x, cond_edge_x = None, None

        for i in range(len(self.t_array)):
            t = self.t_array[i]
            s = self.s_array[i]
            alpha_t, sigma_t = self.noise_scheduler.marginal_prob(t)
            alpha_s, sigma_s = self.noise_scheduler.marginal_prob(s)

            alpha_t_given_s = alpha_t / alpha_s
            sigma2_t_given_s = sigma_t ** 2 - alpha_t_given_s ** 2 * sigma_s ** 2
            sigma_t_given_s = torch.sqrt(sigma2_t_given_s)
            sigma = sigma_t_given_s * sigma_s / sigma_t

            vec_t = torch.ones(batch_size, device=x.device) * t
            noise_level = torch.ones(batch_size, device=x.device) * torch.log(alpha_t ** 2 / sigma_t ** 2)

            if self.pred_edge:
                if self.self_cond:
                    pred_t, edge_pred_t = model(
                        vec_t,
                        x,
                        node_mask,
                        edge_mask,
                        edge_x=edge_x,
                        noise_level=noise_level,
                        cond_x=cond_x,
                        cond_edge_x=cond_edge_x,
                        context=context,
                        spectra=spectra,
                        substructure_spectra=substructure_spectra,
                    )
                    cond_x, cond_edge_x = self.cond_process_fn(pred_t, edge_pred_t)
                else:
                    pred_t, edge_pred_t = model(
                        vec_t,
                        x,
                        node_mask,
                        edge_mask,
                        edge_x=edge_x,
                        noise_level=noise_level,
                        context=context,
                        spectra=spectra,
                        substructure_spectra=substructure_spectra,
                    )
            else:
                pred_t = model(
                    vec_t,
                    x,
                    node_mask,
                    edge_mask,
                    noise_level=noise_level,
                    cond_x=cond_x,
                    context=context,
                    spectra=spectra,
                    substructure_spectra=substructure_spectra,
                )

            if self.model_pred_data:
                x_mean = expand_dims((alpha_t_given_s * sigma_s ** 2 / sigma_t ** 2).repeat(batch_size), x.dim()) * x
                x_mean = x_mean + expand_dims((alpha_s * sigma2_t_given_s / sigma_t ** 2).repeat(batch_size), pred_t.dim()) * pred_t
            else:
                x_mean = x / expand_dims(alpha_t_given_s.repeat(batch_size), x.dim())
                x_mean = x_mean - expand_dims((sigma2_t_given_s / alpha_t_given_s / sigma_t).repeat(batch_size), pred_t.dim()) * pred_t

            x = x_mean + expand_dims(sigma.repeat(batch_size), x_mean.dim()) * sample_combined_position_feature_noise(
                batch_size,
                x_mean.shape[1],
                x_mean.shape[2] - 3,
                node_mask,
            )

            if self.pred_edge:
                if self.model_pred_data:
                    edge_x_mean = expand_dims((alpha_t_given_s * sigma_s ** 2 / sigma_t ** 2).repeat(batch_size), edge_x.dim()) * edge_x
                    edge_x_mean = edge_x_mean + expand_dims((alpha_s * sigma2_t_given_s / sigma_t ** 2).repeat(batch_size), edge_pred_t.dim()) * edge_pred_t
                else:
                    edge_x_mean = edge_x / expand_dims(alpha_t_given_s.repeat(batch_size), edge_x.dim())
                    edge_x_mean = edge_x_mean - expand_dims(
                        (sigma2_t_given_s / alpha_t_given_s / sigma_t).repeat(batch_size),
                        edge_pred_t.dim(),
                    ) * edge_pred_t
                edge_x = edge_x_mean + expand_dims(sigma.repeat(batch_size), edge_x_mean.dim()) * sample_symmetric_edge_feature_noise(
                    batch_size,
                    edge_x_mean.shape[1],
                    edge_x_mean.shape[-1],
                    edge_mask,
                )

        assert_mean_zero_with_mask(x_mean[:, :, :3], node_mask)
        return x_mean, edge_x_mean
