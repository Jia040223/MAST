import torch
import os
import logging
import numpy as np


def _load_model_state_flex(model, state_dict, strict=True):
    try:
        return model.load_state_dict(state_dict, strict=strict)
    except RuntimeError:
        model_state = model.state_dict()
        remapped = {}
        for key, value in state_dict.items():
            if key in model_state:
                remapped[key] = value
            elif key.startswith('module.') and key[7:] in model_state:
                remapped[key[7:]] = value
            elif ('module.' + key) in model_state:
                remapped['module.' + key] = value
        return model.load_state_dict(remapped, strict=False)


def restore_checkpoint(ckpt_dir, state, device):
    if not os.path.exists(ckpt_dir):
        if not os.path.exists(os.path.dirname(ckpt_dir)):
            os.makedirs(os.path.dirname(ckpt_dir))
        logging.warning(f"No checkpoint found at {ckpt_dir}. "
                        f"Returned the same state as input")
        return state
    else:
        loaded_state = torch.load(ckpt_dir, map_location=device, weights_only=False)
        state['optimizer'].load_state_dict(loaded_state['optimizer'])
        _load_model_state_flex(state['model'], loaded_state['model'], strict=True)
        state['ema'].load_state_dict(loaded_state['ema'])
        state['step'] = loaded_state['step']
        return state


def load_pretrained_checkpoint(ckpt_path, state, device, reset_optimizer=True):
    """
    加载预训练检查点用于迁移学习/fine-tuning
    
    Args:
        ckpt_path: 预训练检查点路径
        state: 当前state字典
        device: 设备
        reset_optimizer: 是否重置优化器状态（默认True，用于fine-tuning）
    
    Returns:
        state: 加载权重后的state（step保持为0）
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"预训练检查点不存在: {ckpt_path}")
    
    logging.info(f"🔄 加载预训练检查点: {ckpt_path}")
    loaded_state = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    # 加载模型权重（处理DataParallel的情况）
    model_dict = state['model'].state_dict()
    pretrained_dict = loaded_state['model']
    
    # 统计参数匹配情况
    total_params = len(model_dict)
    pretrained_params = len(pretrained_dict)
    matched_params = 0
    shape_mismatched = 0
    
    try:
        # 尝试直接加载
        missing_keys, unexpected_keys = _load_model_state_flex(
            state['model'],
            loaded_state['model'],
            strict=False,
        )
        
        # 统计成功加载的参数
        matched_params = pretrained_params - len(unexpected_keys)
        
        logging.info(f"✅ 模型权重加载成功")
        logging.info(f"   匹配参数: {matched_params}/{total_params} ({matched_params/total_params*100:.1f}%)")
        
        if len(missing_keys) > 0:
            logging.info(f"   新增参数: {len(missing_keys)}个（将从随机初始化开始）")
            if len(missing_keys) <= 5:
                for k in missing_keys:
                    logging.info(f"      - {k}")
        
        if len(unexpected_keys) > 0:
            logging.info(f"   未使用的预训练参数: {len(unexpected_keys)}个")
            
    except Exception as e:
        # 如果失败，尝试处理module前缀不匹配的情况
        logging.warning(f"⚠️  直接加载失败: {e}")
        logging.info("尝试处理DataParallel前缀...")
        
        # 处理键名不匹配（添加或移除module.前缀）
        new_pretrained_dict = {}
        for k, v in pretrained_dict.items():
            # 当前模型有module前缀，但预训练没有
            if 'module.' + k in model_dict:
                new_pretrained_dict['module.' + k] = v
            # 当前模型没有module前缀，但预训练有
            elif k.startswith('module.') and k[7:] in model_dict:
                new_pretrained_dict[k[7:]] = v
            # 键名相同
            elif k in model_dict:
                new_pretrained_dict[k] = v
        
        # 检查形状匹配
        for k, v in new_pretrained_dict.items():
            if k in model_dict:
                if model_dict[k].shape == v.shape:
                    matched_params += 1
                else:
                    shape_mismatched += 1
        
        missing_keys, unexpected_keys = _load_model_state_flex(
            state['model'],
            new_pretrained_dict,
            strict=False,
        )
        
        logging.info(f"✅ 模型权重加载成功（已处理前缀）")
        logging.info(f"   匹配参数: {matched_params}/{total_params} ({matched_params/total_params*100:.1f}%)")
        
        if shape_mismatched > 0:
            logging.warning(f"   ⚠️  形状不匹配: {shape_mismatched}个参数")
        
        if len(missing_keys) > 0:
            logging.info(f"   新增参数: {len(missing_keys)}个（将从随机初始化开始）")
    
    # 加载EMA权重
    if 'ema' in loaded_state:
        try:
            state['ema'].load_state_dict(loaded_state['ema'])
            logging.info(f"✅ EMA权重加载成功")
        except Exception as e:
            logging.warning(f"⚠️  EMA权重加载失败: {e}")
    
    # 是否重置优化器
    if not reset_optimizer and 'optimizer' in loaded_state:
        try:
            state['optimizer'].load_state_dict(loaded_state['optimizer'])
            logging.info(f"✅ 优化器状态加载成功")
        except Exception as e:
            logging.warning(f"⚠️  优化器状态加载失败: {e}")
    else:
        logging.info(f"📝 优化器状态已重置（用于fine-tuning）")
    
    # step保持为0，开始新的训练
    state['step'] = 0
    
    original_step = loaded_state.get('step', 'unknown')
    logging.info(f"✅ 预训练检查点加载完成（原训练步数: {original_step}，新训练从步数0开始）")
    
    return state


def save_checkpoint(ckpt_dir, state):
    saved_state = {
        'optimizer': state['optimizer'].state_dict(),
        'model': state['model'].state_dict(),
        'ema': state['ema'].state_dict(),
        'step': state['step']
    }
    torch.save(saved_state, ckpt_dir)


def get_data_scaler(config):
    """Data normalizer"""
    # not consider bias here
    if isinstance(config.model.normalize_factors, str):
        normalize_factors = config.model.normalize_factors.split(',')
        normalize_factors = [int(normalize_factor) for normalize_factor in normalize_factors]
    else:
        normalize_factors = config.model.normalize_factors

    if len(normalize_factors) == 3:
        pos_norm, atom_type_norm, fc_charge_norm = normalize_factors
        edge_norm = 1
    else:
        pos_norm, atom_type_norm, fc_charge_norm, edge_norm = normalize_factors

    centered = config.data.centered

    def scale_fn(pos, atom_type, fc_charge, node_mask, edge_type=None, edge_mask=None):
        if centered:
            atom_type = atom_type * 2. - 1.

        if pos is not None:
            pos = pos / pos_norm * node_mask
        atom_type = atom_type / atom_type_norm * node_mask
        fc_charge = fc_charge / fc_charge_norm * node_mask

        if edge_type is not None:
            if centered:
                edge_type = edge_type * 2. - 1.
            edge_type = edge_type / edge_norm
            edge_type = edge_type * edge_mask.reshape(node_mask.size(0), node_mask.size(1), node_mask.size(1), 1)
            return pos, atom_type, fc_charge, edge_type

        return pos, atom_type, fc_charge

    return scale_fn


def get_data_inverse_scaler(config):
    """Inverse data normalizer."""
    # not consider bias here
    if isinstance(config.model.normalize_factors, str):
        normalize_factors = config.model.normalize_factors.split(',')
        normalize_factors = [int(normalize_factor) for normalize_factor in normalize_factors]
    else:
        normalize_factors = config.model.normalize_factors

    if len(normalize_factors) == 3:
        pos_norm, atom_type_norm, fc_charge_norm = normalize_factors
        edge_norm = 1
    else:
        pos_norm, atom_type_norm, fc_charge_norm, edge_norm = normalize_factors

    centered = config.data.centered

    def inverse_scale_fn(pos, atom_type, fc_charge, node_mask, edge_type=None, edge_mask=None):
        if pos is not None:
            pos = pos * pos_norm * node_mask
        atom_type = atom_type * atom_type_norm
        fc_charge = fc_charge * fc_charge_norm * node_mask
        if centered:
            atom_type = (atom_type + 1.) / 2. * node_mask

        if edge_type is not None:
            edge_type = edge_type * edge_norm
            if centered:
                edge_type = (edge_type + 1.) / 2.
            edge_type = edge_type * edge_mask.reshape(node_mask.size(0), node_mask.size(1), node_mask.size(1), 1)
            return pos, atom_type, fc_charge, edge_type

        return pos, atom_type, fc_charge

    return inverse_scale_fn


def get_self_cond_fn(config):
    # To simplify: directly return

    process_type = config.model.self_cond_type  # 'ori', 'clamp'
    compress_edge = config.data.compress_edge
    atom_types = config.data.atom_types
    include_fc = config.model.include_fc_charge
    atom_type_scale = np.array([0., 1.])
    fc_scale = np.array(config.data.fc_scale)
    edge_type_scale = np.array([0., 1.])
    if isinstance(config.model.normalize_factors, str):
        normalize_factors = config.model.normalize_factors.split(',')
        normalize_factors = [int(normalize_factor) for normalize_factor in normalize_factors]
    else:
        normalize_factors = config.model.normalize_factors
    _, atom_type_norm, fc_norm, edge_norm = normalize_factors

    # get the value scale
    centered = config.data.centered
    if centered:
        atom_type_scale = atom_type_scale * 2. - 1.
        edge_type_scale = edge_type_scale * 2. - 1.
    atom_type_scale = atom_type_scale / atom_type_norm
    fc_scale = fc_scale / fc_norm
    edge_type_scale = edge_type_scale / edge_norm

    def process_self_cond(cond_x, cond_edge_x):
        if process_type == 'ori':
            return cond_x, cond_edge_x
        elif process_type == 'clamp':
            atom_x = cond_x[:, :, 3:3+atom_types]
            atom_x = atom_x.clamp(atom_type_scale[0], atom_type_scale[1])
            cond_x[:, :, 3:3+atom_types] = atom_x
            if include_fc:
                atom_fc = cond_x[:, :, -1:]
                atom_fc = atom_fc.clamp(fc_scale[0], fc_scale[1])
                cond_x[:, :, -1:] = atom_fc
            cond_edge_x = cond_edge_x.clamp(edge_type_scale[0], edge_type_scale[1])
            return cond_x, cond_edge_x
        else:
            raise ValueError("Self-condition data process error.")

    return process_self_cond


def expand_dims(v, dims):
    """
    Expand the tensor `v` to the dim `dims`.

    Args:
        `v`: a PyTorch tensor with shape [N].
        `dim`: a `int`.
    Returns:
        a PyTorch tensor with shape [N, 1, 1, ..., 1] and the total dimension is `dims`.
    """
    return v[(...,) + (None,) * (dims - 1)]
