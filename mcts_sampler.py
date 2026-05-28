#!/usr/bin/env python
"""Core MCTS search used in MAST inference."""

import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from checkpoint_sampler import CheckpointSamplerForMCTS, MCTSState


@dataclass
class MCTSConfig:
    """Search hyperparameters for checkpoint-level MCTS."""

    n_checkpoints: int = 20
    checkpoint_kind: str = "geom"
    n_simulations: int = 100
    c_param: float = 1.0
    expand_width: int = 4
    horizon: int = 3
    action_seeds: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5])
    use_guidance: bool = False
    guidance_scales: List[float] = field(default_factory=lambda: [0.0, 0.5, 1.0, 2.0])
    guidance_as_action: bool = True
    rollout_seed: int = 42
    rollout_guidance_scale: float = 0.0
    fast_rollout_steps: int = 20
    value_threshold: float = 0.9
    max_diffusion_steps: Optional[int] = None
    max_tree_depth: Optional[int] = None
    top_k: int = 1
    verbose: bool = True

    def get_action_space(self) -> List[Any]:
        if self.use_guidance and self.guidance_as_action:
            return [
                (seed, guidance_scale)
                for seed in self.action_seeds
                for guidance_scale in self.guidance_scales
            ]
        return list(self.action_seeds)


class MCTSNode:
    """A node in the checkpoint-level search tree."""

    _node_counter = 0

    def __init__(
        self,
        state: MCTSState,
        parent: Optional["MCTSNode"] = None,
        action: Any = None,
        initial_value: float = 0.0,
    ):
        self.state = state
        self.parent = parent
        self.action = action
        self.children: List["MCTSNode"] = []
        self._visit_count = 0
        self._value_sum = 0.0
        self._initial_value = initial_value
        self._max_reward = -float("inf")
        self._reward_list: List[float] = []
        self.rollout_reward: Optional[float] = None
        self.rollout_pred_t = None
        self.rollout_final_state = None
        self.total_steps_to_node = 0
        MCTSNode._node_counter += 1
        self.id = MCTSNode._node_counter

    @property
    def depth(self) -> int:
        depth = 0
        current = self.parent
        while current is not None:
            depth += 1
            current = current.parent
        return depth

    @property
    def visit_count(self) -> int:
        return self._visit_count

    @property
    def value(self) -> float:
        if self._visit_count == 0:
            return self._initial_value
        return self._value_sum / self._visit_count

    @property
    def max_reward(self) -> float:
        if self._max_reward == -float("inf"):
            return self._initial_value
        return self._max_reward

    def get_softmax_value(self, temperature: float = 1.0) -> float:
        if not self._reward_list:
            return self._initial_value

        rewards = np.asarray(self._reward_list, dtype=np.float64)
        rewards = rewards[np.isfinite(rewards)]
        if rewards.size == 0:
            return self._initial_value

        temperature = max(float(temperature), 1e-8)
        rewards_shifted = rewards - np.max(rewards)
        scaled = np.clip(rewards_shifted / temperature, -60.0, 60.0)
        weights = np.exp(scaled)
        weight_sum = np.sum(weights)
        if not np.isfinite(weight_sum) or weight_sum <= 0:
            return float(np.max(rewards))
        return float(np.sum((weights / weight_sum) * rewards))

    def update(self, reward: float):
        self._visit_count += 1
        self._value_sum += reward
        self._max_reward = max(self._max_reward, reward)
        self._reward_list.append(reward)

    def add_child(self, child: "MCTSNode"):
        self.children.append(child)

    def is_leaf(self) -> bool:
        return not self.children

    def is_terminal(self, total_segments: int) -> bool:
        return self.state.is_terminal(total_segments)

    def __repr__(self) -> str:
        return (
            f"MCTSNode(id={self.id}, depth={self.depth}, visits={self.visit_count}, "
            f"value={self.value:.4f}, children={len(self.children)})"
        )


class SparseMCTS:
    """Checkpoint-level MCTS with fast rollout evaluation."""

    def __init__(
        self,
        checkpoint_sampler: CheckpointSamplerForMCTS,
        reward_scorer,
        config: Optional[MCTSConfig] = None,
        substructure_scorer=None,
        substructure_weight: float = 0.1,
        substructure_combine_mode: str = "sum",
        mol_rebuild_fn=None,
        mol_rebuild_fn_3d=None,
        use_3d_validity: bool = False,
        invalid_3d_penalty: float = 0.5,
        backup_strategy: str = "mean",
        softmax_temperature: float = 1.0,
    ):
        self.checkpoint_sampler = checkpoint_sampler
        self.reward_scorer = reward_scorer
        self.config = config or MCTSConfig()
        self.substructure_scorer = substructure_scorer
        self.substructure_weight = substructure_weight
        self.substructure_combine_mode = substructure_combine_mode
        self.mol_rebuild_fn = mol_rebuild_fn
        self.mol_rebuild_fn_3d = mol_rebuild_fn_3d
        self.use_3d_validity = use_3d_validity
        self.invalid_3d_penalty = invalid_3d_penalty
        self.backup_strategy = backup_strategy
        self.softmax_temperature = softmax_temperature
        self.n_segments = checkpoint_sampler.n_segments
        self.current_total_steps = 0

        if self.substructure_combine_mode not in {"sum", "weighted_average"}:
            raise ValueError(
                "substructure_combine_mode must be one of {'sum', 'weighted_average'}."
            )
        if self.backup_strategy not in {"mean", "max", "softmax"}:
            raise ValueError("backup_strategy must be one of {'mean', 'max', 'softmax'}.")

        self.stats = self._fresh_stats()

    def _fresh_stats(self) -> Dict[str, Any]:
        return {
            "n_simulations": 0,
            "n_expansions": 0,
            "n_rollouts": 0,
            "total_time": 0.0,
            "best_reward": -float("inf"),
            "n_invalid_mols": 0,
            "n_invalid_mols_3d": 0,
            "backup_strategy": self.backup_strategy,
            "substructure_combine_mode": self.substructure_combine_mode,
        }

    def get_node_value(self, node: MCTSNode) -> float:
        if self.backup_strategy == "max":
            return node.max_reward
        if self.backup_strategy == "softmax":
            return node.get_softmax_value(self.softmax_temperature)
        return node.value

    def ucb_score(self, node: MCTSNode, parent_visits: int) -> float:
        if node.visit_count == 0:
            return float("inf")
        exploitation = self.get_node_value(node)
        exploration = self.config.c_param * math.sqrt(
            math.log(parent_visits + 1) / (node.visit_count + 1)
        )
        return exploitation + exploration

    def select(self, root: MCTSNode) -> MCTSNode:
        current = root
        while not current.is_leaf():
            if current.is_terminal(self.n_segments):
                return current
            if self.config.max_tree_depth is not None and current.depth >= self.config.max_tree_depth:
                return current

            best_child = None
            best_score = -float("inf")
            for child in current.children:
                if child.is_terminal(self.n_segments) and child.visit_count > 0:
                    continue
                if self.config.max_tree_depth is not None and child.depth >= self.config.max_tree_depth:
                    continue
                score = self.ucb_score(child, current.visit_count)
                if score > best_score:
                    best_score = score
                    best_child = child

            if best_child is None:
                return current
            current = best_child
        return current

    def expand(self, node: MCTSNode, model, context=None, spectra=None) -> List[MCTSNode]:
        if node.is_terminal(self.n_segments):
            return []
        if self.config.max_tree_depth is not None and node.depth >= self.config.max_tree_depth:
            return []
        if (
            self.config.max_diffusion_steps is not None
            and self.current_total_steps >= self.config.max_diffusion_steps
        ):
            return []

        action_space = self.config.get_action_space()
        actions = random.sample(action_space, k=min(self.config.expand_width, len(action_space)))
        checkpoints = self.checkpoint_sampler.checkpoints
        parent_segment_idx = node.state.segment_idx
        children: List[MCTSNode] = []

        for action in actions:
            if (
                self.config.max_diffusion_steps is not None
                and self.current_total_steps >= self.config.max_diffusion_steps
            ):
                break

            if parent_segment_idx < len(checkpoints) - 1:
                estimated_segment_steps = checkpoints[parent_segment_idx + 1] - checkpoints[parent_segment_idx]
                estimated_total_steps = self.current_total_steps + estimated_segment_steps
                if (
                    self.config.max_diffusion_steps is not None
                    and estimated_total_steps > self.config.max_diffusion_steps
                ):
                    continue

            new_state, _ = self.checkpoint_sampler.step(
                model=model,
                state=node.state.clone(),
                action=action,
                context=context,
                spectra=spectra,
            )
            child = MCTSNode(state=new_state, parent=node, action=action)

            child_segment_idx = new_state.segment_idx
            if parent_segment_idx < len(checkpoints) - 1 and child_segment_idx < len(checkpoints):
                segment_steps = checkpoints[child_segment_idx] - checkpoints[parent_segment_idx]
                self.current_total_steps += segment_steps
                if (
                    self.config.max_diffusion_steps is not None
                    and self.current_total_steps > self.config.max_diffusion_steps
                ):
                    self.current_total_steps -= segment_steps
                    continue

            child.total_steps_to_node = self.current_total_steps
            node.add_child(child)
            children.append(child)

        if children:
            self.stats["n_expansions"] += 1
        return children

    def evaluate(self, node: MCTSNode, model, context=None, spectra=None) -> float:
        if self.config.max_diffusion_steps is not None:
            estimated_steps = self.current_total_steps + self.config.fast_rollout_steps
            if estimated_steps > self.config.max_diffusion_steps:
                if self.config.verbose:
                    print(
                        f"      Skip evaluate: estimated steps {estimated_steps} "
                        f"> max {self.config.max_diffusion_steps}"
                    )
                return -float("inf")

        final_state, pred_t = self.checkpoint_sampler.rollout(
            model=model,
            state=node.state,
            context=context,
            spectra=spectra,
            default_seed=self.config.rollout_seed,
            default_guidance_scale=self.config.rollout_guidance_scale,
            fast_steps=self.config.fast_rollout_steps,
        )

        self.current_total_steps += self.config.fast_rollout_steps
        node.total_steps_to_node = self.current_total_steps
        node.rollout_pred_t = pred_t
        node.rollout_final_state = final_state

        reward = self.reward_scorer.score(pred_t, node.state.node_mask)
        reward = reward.mean().item() if getattr(reward, "dim", lambda: 0)() > 0 else reward.item()

        edge_x = final_state.edge_x if getattr(final_state, "edge_x", None) is not None else None
        is_valid_2d, smiles_2d = self._rebuild_validity(
            pred_t,
            final_state.node_mask,
            edge_x,
            self.mol_rebuild_fn,
            invalid_key="n_invalid_mols",
        )

        if self.mol_rebuild_fn_3d is None:
            is_valid_3d, smiles_3d = is_valid_2d, smiles_2d
        else:
            is_valid_3d, smiles_3d = self._rebuild_validity(
                pred_t,
                final_state.node_mask,
                edge_x,
                self.mol_rebuild_fn_3d,
                invalid_key="n_invalid_mols_3d",
            )

        if not is_valid_2d:
            reward = 0.0
            mol = None
        else:
            if self.mol_rebuild_fn_3d is not None and not is_valid_3d:
                reward *= self.invalid_3d_penalty
            mol = self._pred_t_to_mol(pred_t, final_state.node_mask)

        if mol is not None and self.substructure_scorer is not None and self.substructure_weight > 0:
            try:
                motif_reward = self.substructure_scorer.score(mol, batch_idx=0)
                if self.substructure_combine_mode == "weighted_average":
                    reward = (reward + self.substructure_weight * motif_reward) / (
                        1.0 + self.substructure_weight
                    )
                else:
                    reward = reward + self.substructure_weight * motif_reward
            except Exception:
                pass

        if self.use_3d_validity and smiles_3d is not None:
            node.pred_smiles = smiles_3d
        else:
            node.pred_smiles = smiles_2d

        node.rollout_reward = reward
        self.stats["n_rollouts"] += 1
        return reward

    def _rebuild_validity(self, pred_t, node_mask, edge_x, rebuild_fn, invalid_key: str) -> Tuple[bool, Optional[str]]:
        if rebuild_fn is None:
            return self._check_mol_validity_strict(pred_t, node_mask, edge_x), None
        try:
            smiles = rebuild_fn(pred_t, edge_x, node_mask)
        except Exception:
            smiles = None
        if smiles is None:
            self.stats[invalid_key] = self.stats.get(invalid_key, 0) + 1
            return False, None
        return True, smiles

    def _check_mol_validity_strict(self, pred_t, node_mask, edge_x=None) -> bool:
        try:
            node_mask_squeezed = node_mask.squeeze(-1)
            n_valid = node_mask_squeezed[0].sum().int().item()
            if n_valid == 0:
                return False

            atom_pred = pred_t[0, :n_valid, 3:]
            if atom_pred.shape[-1] > 5:
                atom_pred = atom_pred[:, :-1]
            atom_types = atom_pred[:, :5].argmax(dim=-1)
            if (atom_types == 0).all():
                return False

            if edge_x is not None:
                edge_types = edge_x[0, :n_valid, :n_valid]
                if edge_types.dim() > 2:
                    edge_types = edge_types.argmax(dim=-1)
                n_bonds = (edge_types > 0).sum().item()
                if n_bonds == 0:
                    return False
            return True
        except Exception:
            return False

    def _pred_t_to_mol(self, pred_t, node_mask):
        from rdkit import Chem
        from rdkit.Geometry import Point3D

        node_mask_squeezed = node_mask.squeeze(-1)
        n_valid = node_mask_squeezed[0].sum().int().item()
        if n_valid == 0:
            return None

        pos = pred_t[0, :n_valid, :3].cpu().numpy()
        atom_pred = pred_t[0, :n_valid, 3:]
        if atom_pred.shape[-1] > 5:
            atom_pred = atom_pred[:, :-1]
        atom_types_idx = atom_pred[:, :5].argmax(dim=-1).cpu().numpy()
        atom_map = {0: 1, 1: 6, 2: 7, 3: 8, 4: 9}

        try:
            mol = Chem.RWMol()
            conf = Chem.Conformer(n_valid)
            for idx in range(n_valid):
                atom = Chem.Atom(atom_map.get(int(atom_types_idx[idx]), 6))
                mol.AddAtom(atom)
                conf.SetAtomPosition(idx, Point3D(float(pos[idx, 0]), float(pos[idx, 1]), float(pos[idx, 2])))
            mol.AddConformer(conf, assignId=True)
            return mol.GetMol()
        except Exception:
            return None

    def backpropagate(self, node: MCTSNode, reward: float):
        current = node
        while current is not None:
            current.update(reward)
            current = current.parent

    def run_simulation(self, root: MCTSNode, model, context=None, spectra=None):
        leaf = self.select(root)
        expanded = False
        if not leaf.is_terminal(self.n_segments) and leaf.visit_count > 0:
            new_children = self.expand(leaf, model, context, spectra)
            if new_children:
                leaf = new_children[0]
                expanded = True

        reward = self.evaluate(leaf, model, context, spectra)
        self._last_sim_reward = reward
        self._last_sim_expanded = expanded
        self._last_sim_depth = leaf.depth
        self.backpropagate(leaf, reward)
        self.stats["n_simulations"] += 1

        if reward > self.stats["best_reward"]:
            self.stats["best_reward"] = reward
            self.stats["best_reward_sim"] = self.stats["n_simulations"]

    def get_best_reward_node(self, root: MCTSNode) -> MCTSNode:
        best_node = root
        best_reward = root.rollout_reward if root.rollout_reward is not None else -float("inf")
        best_depth = root.depth

        def traverse(node: MCTSNode):
            nonlocal best_depth, best_node, best_reward
            if node.rollout_reward is not None:
                if node.rollout_reward > best_reward or (
                    node.rollout_reward == best_reward and node.depth > best_depth
                ):
                    best_reward = node.rollout_reward
                    best_depth = node.depth
                    best_node = node
            for child in node.children:
                traverse(child)

        traverse(root)
        return best_node

    def get_topk_reward_nodes(
        self,
        root: MCTSNode,
        k: int = 4,
        threshold: Optional[float] = None,
    ) -> List[MCTSNode]:
        all_nodes: List[MCTSNode] = []

        def traverse(node: MCTSNode):
            if node.rollout_reward is not None:
                if threshold is None or node.rollout_reward >= threshold:
                    all_nodes.append(node)
            for child in node.children:
                traverse(child)

        traverse(root)
        all_nodes.sort(key=lambda node: (node.rollout_reward, node.depth), reverse=True)
        return all_nodes[:k]

    def count_nodes_above_threshold(self, root: MCTSNode, threshold: float) -> int:
        count = 0

        def traverse(node: MCTSNode):
            nonlocal count
            if node.rollout_reward is not None and node.rollout_reward >= threshold:
                count += 1
            for child in node.children:
                traverse(child)

        traverse(root)
        return count

    def search(
        self,
        model,
        initial_state: MCTSState,
        context=None,
        spectra=None,
        reward_spectra=None,
        topk: int = 1,
        topk_threshold: Optional[float] = None,
    ) -> Tuple[MCTSNode, Dict[str, Any], List[MCTSNode]]:
        start_time = time.time()
        self.current_total_steps = 0
        self.stats = self._fresh_stats()

        root = MCTSNode(state=initial_state)
        score_spectra = reward_spectra if reward_spectra is not None else spectra
        if score_spectra is not None:
            self.reward_scorer.cache_spectra_features(score_spectra)
            if self.config.use_guidance:
                self.checkpoint_sampler.cache_spectra_features(score_spectra)

        early_stop_reason = None
        for sim_idx in range(self.config.n_simulations):
            if (
                self.config.max_diffusion_steps is not None
                and self.current_total_steps >= self.config.max_diffusion_steps
            ):
                early_stop_reason = "max_diffusion_steps"
                if self.config.verbose:
                    print(
                        f"  Early stop: current_total_steps {self.current_total_steps} "
                        f">= max_diffusion_steps {self.config.max_diffusion_steps}"
                    )
                break

            self.run_simulation(root, model, context, spectra)

            if (sim_idx + 1) % 5 == 0:
                try:
                    import torch

                    torch.cuda.empty_cache()
                except Exception:
                    pass

            print_every = 5 if self.config.n_simulations > 10 else 1
            if self.config.verbose and (sim_idx + 1) % print_every == 0:
                last_reward = getattr(self, "_last_sim_reward", 0.0)
                last_depth = getattr(self, "_last_sim_depth", 0)
                expanded = "expand" if getattr(self, "_last_sim_expanded", False) else "eval"
                marker = "★" if last_reward >= self.stats["best_reward"] else ""
                steps_info = (
                    f", steps={self.current_total_steps}"
                    if self.config.max_diffusion_steps is not None
                    else ""
                )
                print(
                    f"      Sim {sim_idx + 1}/{self.config.n_simulations}: "
                    f"this={last_reward:.4f}{marker}, best={self.stats['best_reward']:.4f}, "
                    f"depth={last_depth}, {expanded}{steps_info}"
                )

            if self.stats["best_reward"] >= self.config.value_threshold:
                early_stop_reason = "value_threshold"
                if self.config.verbose:
                    print(
                        f"  Early stop: reward {self.stats['best_reward']:.4f} "
                        f">= threshold {self.config.value_threshold}"
                    )
                break

            if topk > 1 and topk_threshold is not None:
                n_above = self.count_nodes_above_threshold(root, topk_threshold)
                if n_above >= topk:
                    early_stop_reason = "topk_threshold"
                    if self.config.verbose:
                        print(f"  Top-k early stop: found {n_above} nodes >= {topk_threshold}")
                    break

        self.stats["total_time"] = time.time() - start_time
        self.stats["early_stop_reason"] = early_stop_reason

        best_node = self.get_best_reward_node(root)
        topk_nodes = self.get_topk_reward_nodes(root, k=topk, threshold=topk_threshold)
        self.stats["topk_rewards"] = [node.rollout_reward for node in topk_nodes]
        self.stats["n_above_threshold"] = len(topk_nodes)
        self.stats["total_diffusion_steps"] = int(getattr(best_node, "total_steps_to_node", 0))

        if self.config.verbose:
            self._print_search_summary(best_node, topk_nodes)
        return best_node, self.stats, topk_nodes

    def _print_search_summary(self, best_node: MCTSNode, topk_nodes: List[MCTSNode]):
        print("\n" + "=" * 60)
        print("MCTS Search Summary")
        print("=" * 60)
        print(f"Total simulations: {self.stats['n_simulations']}")
        print(f"Total expansions: {self.stats['n_expansions']}")
        print(f"Total rollouts: {self.stats['n_rollouts']}")
        print(f"Total time: {self.stats['total_time']:.2f}s")
        print(f"Best reward: {self.stats['best_reward']:.4f}")
        print(f"Best node: {best_node}")
        print(f"Best depth: {best_node.depth}")
        print(f"Best action sequence: {best_node.state.seed_history}")
        if len(topk_nodes) > 1:
            print(f"\nTop-{len(topk_nodes)} nodes:")
            for rank, node in enumerate(topk_nodes, start=1):
                print(f"  #{rank}: reward={node.rollout_reward:.4f}, depth={node.depth}")
        if self.stats.get("early_stop_reason"):
            print(f"\nEarly stop reason: {self.stats['early_stop_reason']}")
        print("=" * 60 + "\n")
