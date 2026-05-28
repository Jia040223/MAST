#!/usr/bin/env python
"""
Checkpoint Schedule for MCTS-based Diffusion Sampling

将1000步采样压缩为若干checkpoint，每个checkpoint之间为一个segment（宏动作）
"""
import numpy as np
import torch


def make_checkpoints(n_steps=1000, n_ckpt=20, kind="geom"):
    """生成checkpoint索引列表
    
    Args:
        n_steps: 总采样步数
        n_ckpt: checkpoint数量（不含端点可能会多2个）
        kind: 调度类型
            - "geom": 几何级数，早期密集（噪声大时更频繁决策）
            - "linear": 等间距
            - "cosine": 余弦调度
            - "manual": 手动定义的关键点
            
    Returns:
        checkpoints: List[int]，checkpoint的step索引（0-based）
    """
    # 兼容 torch.Tensor 传入（可能在 GPU 上）
    if isinstance(n_steps, torch.Tensor):
        n_steps = int(n_steps.shape[0]) if n_steps.dim() > 0 else int(n_steps.cpu().item())
    if isinstance(n_ckpt, torch.Tensor):
        n_ckpt = int(n_ckpt.cpu().item())
    if kind == "geom":
        # 几何级数：早期密集，后期稀疏
        # 因为早期噪声大，决策更重要
        xs = np.geomspace(1, n_steps, num=n_ckpt).astype(int) - 1
        cp = sorted(set(xs.tolist() + [0, n_steps - 1]))
        return cp
        
    elif kind == "linear":
        # 等间距
        xs = np.linspace(0, n_steps - 1, num=n_ckpt + 1).astype(int)
        cp = sorted(set(xs.tolist()))
        return cp
        
    elif kind == "cosine":
        # 余弦调度：两端密集，中间稀疏
        # 适合diffusion的噪声调度特性
        t = np.linspace(0, np.pi, n_ckpt + 1)
        xs = ((1 - np.cos(t)) / 2 * (n_steps - 1)).astype(int)
        cp = sorted(set(xs.tolist()))
        return cp
        
    elif kind == "manual":
        # 手动定义：针对1000步的经验值
        return [0, 20, 45, 70, 100, 140, 190, 250, 320, 400, 
                490, 590, 700, 820, 940, 999]
                
    elif kind == "adaptive":
        # 自适应：根据噪声调度的特性动态调整
        # 早期（高噪声）密集，后期稀疏
        # 使用平方根调度
        xs = np.sqrt(np.linspace(0, 1, n_ckpt + 1)) * (n_steps - 1)
        xs = xs.astype(int)
        cp = sorted(set(xs.tolist() + [n_steps - 1]))
        return cp
        
    else:
        raise ValueError(f"Unknown checkpoint kind: {kind}")


def get_segment_ranges(checkpoints):
    """获取每个segment的起止索引
    
    Args:
        checkpoints: checkpoint索引列表
        
    Returns:
        segments: List[Tuple[int, int]]，每个tuple为(start_idx, end_idx)
                  表示从checkpoints[start_idx]到checkpoints[end_idx]的segment
    """
    segments = []
    for i in range(len(checkpoints) - 1):
        segments.append((checkpoints[i], checkpoints[i + 1]))
    return segments


def checkpoint_to_time_range(checkpoints, time_steps):
    """将checkpoint索引转换为时间步范围
    
    Args:
        checkpoints: checkpoint索引列表
        time_steps: [n_steps] 时间步tensor
        
    Returns:
        time_ranges: List[Tuple[float, float]]，每个tuple为(t_start, t_end)
    """
    time_ranges = []
    for i in range(len(checkpoints) - 1):
        t_start = time_steps[checkpoints[i]].item()
        t_end = time_steps[checkpoints[i + 1]].item()
        time_ranges.append((t_start, t_end))
    return time_ranges


def visualize_checkpoints(checkpoints, n_steps=1000, time_steps=None):
    """可视化checkpoint分布
    
    Args:
        checkpoints: checkpoint索引列表
        n_steps: 总步数
        time_steps: 可选，时间步tensor
    """
    print(f"\n{'='*60}")
    print(f"Checkpoint Schedule Visualization")
    print(f"Total steps: {n_steps}, Checkpoints: {len(checkpoints)}")
    print(f"{'='*60}")
    
    # 打印checkpoint索引
    print(f"\nCheckpoint indices: {checkpoints}")
    
    # 打印segment信息
    segments = get_segment_ranges(checkpoints)
    print(f"\nSegments ({len(segments)} total):")
    for i, (start, end) in enumerate(segments):
        steps_in_seg = end - start
        if time_steps is not None:
            t_start = time_steps[start].item()
            t_end = time_steps[end].item() if end < len(time_steps) else 0
            print(f"  Segment {i:2d}: steps [{start:4d}, {end:4d}] "
                  f"({steps_in_seg:3d} steps), t: [{t_start:.4f}, {t_end:.4f}]")
        else:
            print(f"  Segment {i:2d}: steps [{start:4d}, {end:4d}] ({steps_in_seg:3d} steps)")
    
    # 可视化分布（简单ASCII图）
    print(f"\nDistribution (| = checkpoint):")
    bar = ['.'] * 50
    for cp in checkpoints:
        idx = int(cp / n_steps * 49)
        bar[idx] = '|'
    print("  " + ''.join(bar))
    print(f"  0{' '*23}500{' '*22}1000")
    
    print(f"{'='*60}\n")


class CheckpointManager:
    """Checkpoint管理器，用于MCTS采样"""
    
    def __init__(self, n_steps=1000, n_ckpt=20, kind="geom", time_steps=None):
        """
        Args:
            n_steps: 总采样步数
            n_ckpt: checkpoint数量
            kind: 调度类型
            time_steps: 时间步tensor（可选）
        """
        self.n_steps = n_steps
        self.n_ckpt = n_ckpt
        self.kind = kind
        
        # 生成checkpoints
        self.checkpoints = make_checkpoints(n_steps, n_ckpt, kind)
        self.segments = get_segment_ranges(self.checkpoints)
        
        # 存储时间步（如果提供）
        self.time_steps = time_steps
        
    def get_segment(self, seg_idx):
        """获取指定segment的范围
        
        Args:
            seg_idx: segment索引
            
        Returns:
            (start_step, end_step)
        """
        if seg_idx < 0 or seg_idx >= len(self.segments):
            raise IndexError(f"Segment index {seg_idx} out of range [0, {len(self.segments)})")
        return self.segments[seg_idx]
    
    def get_time_range(self, seg_idx):
        """获取指定segment的时间范围
        
        Args:
            seg_idx: segment索引
            
        Returns:
            (t_start, t_end)
        """
        if self.time_steps is None:
            raise ValueError("time_steps not provided")
        start, end = self.get_segment(seg_idx)
        return (self.time_steps[start].item(), 
                self.time_steps[end].item() if end < len(self.time_steps) else 0)
    
    @property
    def n_segments(self):
        """segment数量（树的最大深度）"""
        return len(self.segments)
    
    @property
    def depth(self):
        """别名：树深度"""
        return self.n_segments
    
    def __len__(self):
        return self.n_segments
    
    def __repr__(self):
        return (f"CheckpointManager(n_steps={self.n_steps}, n_ckpt={self.n_ckpt}, "
                f"kind='{self.kind}', n_segments={self.n_segments})")


# 测试和可视化
if __name__ == "__main__":
    print("测试不同的Checkpoint调度策略:")
    
    n_steps = 1000
    
    for kind in ["geom", "linear", "cosine", "manual", "adaptive"]:
        print(f"\n{'='*40}")
        print(f"Kind: {kind}")
        cp = make_checkpoints(n_steps, n_ckpt=15, kind=kind)
        print(f"Checkpoints ({len(cp)}): {cp}")
        
        # 计算segment长度分布
        segments = get_segment_ranges(cp)
        lengths = [e - s for s, e in segments]
        print(f"Segment lengths: min={min(lengths)}, max={max(lengths)}, "
              f"mean={np.mean(lengths):.1f}")
    
    # 详细可视化一个
    print("\n" + "="*60)
    print("详细可视化 'geom' 调度:")
    time_steps = torch.linspace(1.0, 1e-3, n_steps)
    cp = make_checkpoints(n_steps, n_ckpt=20, kind="geom")
    visualize_checkpoints(cp, n_steps, time_steps)
    
    # 测试CheckpointManager
    print("测试CheckpointManager:")
    manager = CheckpointManager(n_steps=1000, n_ckpt=20, kind="geom", time_steps=time_steps)
    print(manager)
    print(f"Number of segments: {manager.n_segments}")
    print(f"First segment: {manager.get_segment(0)}")
    print(f"First segment time range: {manager.get_time_range(0)}")
    
    print("\n✅ Checkpoint Schedule测试通过！")


