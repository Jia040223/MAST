#!/usr/bin/env python
"""Run MCTS-guided inference for the MAST model."""

import argparse
import importlib.util
import json
import pickle
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from tqdm import tqdm

import losses
from checkpoint_sampler import CheckpointSamplerForMCTS
from checkpoint_schedule import make_checkpoints
from datasets import get_dataset
from diffusion import NoiseScheduleVP
from evaluation import get_2D_edm_metric, get_edm_metric
from mcts_sampler import MCTSConfig, SparseMCTS
from models import create_model
from models.ema import ExponentialMovingAverage
from models.utils import (
    assert_mean_zero_with_mask,
    sample_combined_position_feature_noise,
    sample_symmetric_edge_feature_noise,
)
from reward_scorer import create_reward_scorer, create_substructure_scorer
from sampling import mol_process, post_process
from utils import get_data_inverse_scaler, get_self_cond_fn, restore_checkpoint


def load_config(config_path: str):
    spec = importlib.util.spec_from_file_location("mast_config", config_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.get_config()


def parse_args():
    parser = argparse.ArgumentParser(description="MAST MCTS inference")
    parser.add_argument("--config", required=True, help="Path to the config file.")
    parser.add_argument("--checkpoint", required=True, help="Diffusion model checkpoint.")
    parser.add_argument("--output_dir", default="runs/mcts", help="Directory for saved results.")
    parser.add_argument("--split", choices=["train", "valid", "test"], default="test")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of split items to process.")
    parser.add_argument("--start_index", type=int, default=0, help="Start index inside the chosen split.")
    parser.add_argument(
        "--indices",
        type=str,
        default="",
        help="Comma-separated explicit indices inside the chosen split.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None, help="Override device, e.g. cuda:0 or cpu.")
    parser.add_argument("--n_simulations", type=int, default=None)
    parser.add_argument("--n_checkpoints", type=int, default=None)
    parser.add_argument("--expand_width", type=int, default=None)
    parser.add_argument("--c_param", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None, help="Override the number of returned candidates.")
    return parser.parse_args()


def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_sanitize(mol):
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None


def canonical_smiles(mol) -> Optional[str]:
    mol = safe_sanitize(mol)
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def morgan_fp(mol):
    if mol is None:
        return None
    try:
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
    except Exception:
        return None


def tanimoto(fp_a, fp_b) -> Optional[float]:
    if fp_a is None or fp_b is None:
        return None
    return float(DataStructs.TanimotoSimilarity(fp_a, fp_b))


def select_split(config, split_name: str):
    train_ds, valid_ds, test_ds, dataset_info = get_dataset(config)
    split_map = {"train": train_ds, "valid": valid_ds, "test": test_ds}
    return split_map[split_name], dataset_info


def resolve_indices(split_size: int, indices_arg: str, start_index: int, num_samples: int) -> List[int]:
    if indices_arg:
        indices = [int(item.strip()) for item in indices_arg.split(",") if item.strip()]
    else:
        end_index = min(split_size, start_index + num_samples)
        indices = list(range(start_index, end_index))
    return [idx for idx in indices if 0 <= idx < split_size]


def build_masks(max_n_nodes: int, n_nodes: int, device: torch.device):
    node_mask = torch.zeros(1, max_n_nodes, 1, device=device)
    node_mask[0, :n_nodes, 0] = 1
    edge_mask = node_mask.squeeze(-1).unsqueeze(1) * node_mask.squeeze(-1).unsqueeze(2)
    diag_mask = ~torch.eye(max_n_nodes, dtype=torch.bool, device=device).unsqueeze(0)
    edge_mask = edge_mask * diag_mask
    edge_mask = edge_mask.view(max_n_nodes * max_n_nodes, 1)
    return node_mask, edge_mask


def decode_prediction(config, inverse_scaler, pred_t, edge_x, node_mask):
    atom_types = config.data.atom_types
    include_fc = config.model.include_fc_charge
    compress_edge = config.data.compress_edge
    edge_mask = build_masks(node_mask.size(1), int(node_mask.sum().item()), pred_t.device)[1]

    with torch.no_grad():
        if config.pred_edge:
            positions, one_hot, formal_charges, edge_types = post_process(
                pred_t,
                atom_types,
                include_fc,
                node_mask,
                inverse_scaler,
                edge_x,
                edge_mask,
                compress_edge,
            )
            n_nodes = node_mask.squeeze(-1).sum(dim=1).long().tolist()
            items_3d = mol_process(one_hot, positions, formal_charges, n_nodes, edge_types)
            items_2d = [(None, item[1], item[2], item[3]) for item in items_3d]
            return items_3d, items_2d

        positions, one_hot, formal_charges = post_process(
            pred_t,
            atom_types,
            include_fc,
            node_mask,
            inverse_scaler,
        )
        n_nodes = node_mask.squeeze(-1).sum(dim=1).long().tolist()
        items_3d = mol_process(one_hot, positions, formal_charges, n_nodes)
        items_2d = [(None, item[1]) for item in items_3d]
        return items_3d, items_2d


def build_smiles_rebuilder(config, inverse_scaler, edm_metric, use_3d: bool):
    def rebuild(pred_t, edge_x, node_mask):
        items_3d, items_2d = decode_prediction(config, inverse_scaler, pred_t, edge_x, node_mask)
        items = items_3d if use_3d else items_2d
        _, _, mols = edm_metric(items)
        mol = mols[0] if mols else None
        return canonical_smiles(mol)

    return rebuild


def build_mcts(
    config,
    noise_scheduler,
    reward_scorer,
    substructure_scorer,
    inverse_scaler,
    edm_metric_3d,
    edm_metric_2d,
):
    time_steps = torch.linspace(noise_scheduler.T, 1e-3, config.sampling.steps, device=config.device)
    checkpoints = make_checkpoints(
        config.sampling.steps,
        config.mcts.n_checkpoints,
        config.mcts.checkpoint_kind,
    )

    checkpoint_sampler = CheckpointSamplerForMCTS(
        noise_scheduler=noise_scheduler,
        time_steps=time_steps,
        checkpoints=checkpoints,
        model_pred_data=config.model.pred_data,
        pred_edge=config.pred_edge,
        self_cond=config.model.self_cond,
        cond_process_fn=get_self_cond_fn(config) if config.model.self_cond else None,
        guidance_model=reward_scorer.guidance_model if config.mcts.use_guidance else None,
        apply_to_pos_only=getattr(config.mcts, "apply_to_pos_only", True),
        time_scale_guidance=getattr(config.mcts, "time_scale_guidance", True),
        reinit_noise_at_first_segment=getattr(config.mcts, "reinit_noise_at_first_segment", False),
        node_nf=config.data.atom_types + int(config.model.include_fc_charge),
        edge_nf=config.model.edge_ch,
    )

    mcts_config = MCTSConfig(
        n_checkpoints=config.mcts.n_checkpoints,
        checkpoint_kind=config.mcts.checkpoint_kind,
        n_simulations=config.mcts.n_simulations,
        c_param=config.mcts.c_param,
        expand_width=config.mcts.expand_width,
        horizon=config.mcts.horizon,
        action_seeds=list(range(config.mcts.n_actions)),
        use_guidance=config.mcts.use_guidance,
        guidance_scales=list(config.mcts.guidance_scales),
        guidance_as_action=config.mcts.guidance_as_action,
        rollout_seed=config.mcts.rollout_seed,
        rollout_guidance_scale=float(config.mcts.guidance_scales[0]) if config.mcts.use_guidance else 0.0,
        fast_rollout_steps=config.mcts.fast_rollout_steps,
        value_threshold=config.mcts.value_threshold,
        max_diffusion_steps=getattr(config.mcts, "max_diffusion_steps", None),
        max_tree_depth=getattr(config.mcts, "max_tree_depth", None),
        top_k=config.mcts.top_k,
        verbose=getattr(config.mcts, "verbose", True),
    )

    return SparseMCTS(
        checkpoint_sampler=checkpoint_sampler,
        reward_scorer=reward_scorer,
        config=mcts_config,
        substructure_scorer=substructure_scorer,
        substructure_weight=getattr(config.reward, "substructure_weight", 0.1),
        substructure_combine_mode=getattr(config.reward, "substructure_combine_mode", "sum"),
        mol_rebuild_fn=build_smiles_rebuilder(config, inverse_scaler, edm_metric_2d, use_3d=False),
        mol_rebuild_fn_3d=build_smiles_rebuilder(config, inverse_scaler, edm_metric_3d, use_3d=True),
        use_3d_validity=getattr(config.reward, "use_3d_validity", False),
        invalid_3d_penalty=getattr(config.reward, "invalid_3d_penalty", 0.5),
        backup_strategy=getattr(config.mcts, "backup_strategy", "mean"),
        softmax_temperature=getattr(config.mcts, "softmax_temperature", 1.0),
    )


def run_single_sample(
    model,
    config,
    data,
    reward_scorer,
    substructure_scorer,
    noise_scheduler,
    inverse_scaler,
    edm_metric_3d,
    edm_metric_2d,
):
    spectra_batch = [spectrum.unsqueeze(0).to(config.device) for spectrum in data.spectra]
    n_nodes = int(getattr(data, "num_nodes", len(data.z)))
    max_n_nodes = int(config.data.max_node)
    node_nf = config.data.atom_types + int(config.model.include_fc_charge)
    edge_nf = config.model.edge_ch

    node_mask, edge_mask = build_masks(max_n_nodes, n_nodes, config.device)
    z_t = sample_combined_position_feature_noise(1, max_n_nodes, node_nf, node_mask)
    assert_mean_zero_with_mask(z_t[:, :, :3], node_mask)
    edge_z_t = None
    if config.pred_edge:
        edge_z_t = sample_symmetric_edge_feature_noise(1, max_n_nodes, edge_nf, edge_mask)

    mcts = build_mcts(
        config,
        noise_scheduler,
        reward_scorer,
        substructure_scorer,
        inverse_scaler,
        edm_metric_3d,
        edm_metric_2d,
    )
    initial_state = mcts.checkpoint_sampler.create_initial_state(z_t, edge_z_t, node_mask, edge_mask)

    if substructure_scorer is not None:
        substructure_scorer.cache_predicted_probs(spectra_batch)

    best_node, stats, topk_nodes = mcts.search(
        model=model,
        initial_state=initial_state,
        spectra=spectra_batch,
        reward_spectra=spectra_batch,
        topk=config.mcts.top_k,
    )
    if not topk_nodes:
        topk_nodes = [best_node]

    candidates = []
    for node in topk_nodes:
        pred_t = node.rollout_pred_t
        final_state = node.rollout_final_state
        if pred_t is None or final_state is None:
            continue

        items_3d, items_2d = decode_prediction(config, inverse_scaler, pred_t, final_state.edge_x, final_state.node_mask)
        _, _, mols_3d = edm_metric_3d(items_3d)
        _, _, mols_2d = edm_metric_2d(items_2d)
        mol_3d = mols_3d[0] if mols_3d else None
        mol_2d = mols_2d[0] if mols_2d else None

        candidates.append(
            {
                "reward": float(node.rollout_reward),
                "depth": int(node.depth),
                "action_sequence": list(node.state.seed_history),
                "smiles_2d": canonical_smiles(mol_2d),
                "smiles_3d": canonical_smiles(mol_3d),
                "mol_2d": mol_2d,
                "mol_3d": mol_3d,
                "item_2d": items_2d[0] if items_2d else None,
                "item_3d": items_3d[0] if items_3d else None,
            }
        )

    candidates.sort(key=lambda item: item["reward"], reverse=True)
    return candidates, stats


def summarize_results(results: Sequence[Dict]):
    exact_top1 = 0
    exact_topk = 0
    tanimoto_top1 = 0
    tanimoto_topk = 0
    eligible = 0
    best_rewards = []
    search_times = []

    for result in results:
        best_rewards.append(result["stats"]["best_reward"])
        search_times.append(result["stats"]["total_time"])
        gt_smiles = result.get("gt_smiles")
        gt_mol = result.get("gt_mol")
        candidates = result.get("candidates", [])
        if gt_smiles is None or gt_mol is None or not candidates:
            continue

        eligible += 1
        gt_fp = morgan_fp(gt_mol)

        top1 = candidates[0]
        if top1["smiles_2d"] == gt_smiles:
            exact_top1 += 1

        hit_topk = False
        hit_tanimoto_topk = False
        for candidate in candidates:
            if candidate["smiles_2d"] == gt_smiles:
                hit_topk = True
            sim = tanimoto(gt_fp, morgan_fp(candidate["mol_2d"]))
            if sim is not None and sim >= 0.9999:
                hit_tanimoto_topk = True
        exact_topk += int(hit_topk)
        tanimoto_topk += int(hit_tanimoto_topk)

        top1_sim = tanimoto(gt_fp, morgan_fp(top1["mol_2d"]))
        if top1_sim is not None and top1_sim >= 0.9999:
            tanimoto_top1 += 1

    denom = max(eligible, 1)
    return {
        "num_processed": len(results),
        "num_with_gt": eligible,
        "mean_best_reward": float(np.mean(best_rewards)) if best_rewards else 0.0,
        "mean_search_time_sec": float(np.mean(search_times)) if search_times else 0.0,
        "top1_exact_match": exact_top1 / denom,
        "topk_exact_match": exact_topk / denom,
        "top1_tanimoto_1.0": tanimoto_top1 / denom,
        "topk_tanimoto_1.0": tanimoto_topk / denom,
    }


def main():
    args = parse_args()
    config = load_config(args.config)
    config.device = torch.device(
        args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    )

    if args.n_simulations is not None:
        config.mcts.n_simulations = args.n_simulations
    if args.n_checkpoints is not None:
        config.mcts.n_checkpoints = args.n_checkpoints
    if args.expand_width is not None:
        config.mcts.expand_width = args.expand_width
    if args.c_param is not None:
        config.mcts.c_param = args.c_param
    if args.top_k is not None:
        config.mcts.top_k = args.top_k

    set_random_seed(args.seed)
    config.seed = args.seed

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_dataset, dataset_info = select_split(config, args.split)
    requested = args.num_samples if args.num_samples is not None else int(config.eval.num_samples)
    sample_indices = resolve_indices(len(split_dataset), args.indices, args.start_index, requested)

    if not sample_indices:
        raise ValueError("No valid sample indices were selected.")
    if not getattr(config.reward, "ckpt_path", ""):
        raise ValueError("config.reward.ckpt_path must be set for MCTS inference.")

    model = create_model(config)
    ema = ExponentialMovingAverage(model.parameters(), decay=config.model.ema_decay)
    optimizer = losses.get_optimizer(config, model.parameters())
    state = {"optimizer": optimizer, "model": model, "ema": ema, "step": 0}
    state = restore_checkpoint(args.checkpoint, state, config.device)
    ema.copy_to(model.parameters())
    model.eval()

    reward_scorer = create_reward_scorer(config.reward.ckpt_path, device=config.device)
    substructure_scorer = None
    if getattr(config.reward, "use_substructure_reward", False):
        motif_ckpt = getattr(config.reward, "substructure_predictor_path", "")
        if not motif_ckpt:
            raise ValueError("config.reward.substructure_predictor_path must be set when motif reward is enabled.")
        substructure_scorer = create_substructure_scorer(motif_ckpt, device=config.device)

    noise_scheduler = NoiseScheduleVP(
        config.sde.schedule,
        continuous_beta_0=config.sde.continuous_beta_0,
        continuous_beta_1=config.sde.continuous_beta_1,
    )
    inverse_scaler = get_data_inverse_scaler(config)
    edm_metric_3d = get_edm_metric(dataset_info)
    edm_metric_2d = get_2D_edm_metric(dataset_info)

    results = []
    for split_index in tqdm(sample_indices, desc=f"MCTS {args.split}"):
        set_random_seed(args.seed + split_index)
        data = split_dataset[split_index]
        gt_mol = getattr(data, "rdmol", None)
        gt_smiles = canonical_smiles(gt_mol)

        candidates, stats = run_single_sample(
            model=model,
            config=config,
            data=data,
            reward_scorer=reward_scorer,
            substructure_scorer=substructure_scorer,
            noise_scheduler=noise_scheduler,
            inverse_scaler=inverse_scaler,
            edm_metric_3d=edm_metric_3d,
            edm_metric_2d=edm_metric_2d,
        )

        results.append(
            {
                "split": args.split,
                "split_index": split_index,
                "gt_smiles": gt_smiles,
                "gt_mol": gt_mol,
                "num_nodes": int(getattr(data, "num_nodes", len(data.z))),
                "candidates": candidates,
                "stats": stats,
            }
        )

    summary = summarize_results(results)
    summary["split"] = args.split
    summary["checkpoint"] = args.checkpoint
    summary["num_requested"] = len(sample_indices)

    with open(output_dir / "results.pkl", "wb") as f:
        pickle.dump(results, f)
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Saved results to {output_dir}")


if __name__ == "__main__":
    main()
