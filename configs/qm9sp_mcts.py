import os

import ml_collections
import torch


def get_config():
    config = ml_collections.ConfigDict()

    config.exp_type = "vpsde_edge"
    config.pred_edge = True
    config.only_2D = False
    config.seed = 42
    config.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    config.data = data = ml_collections.ConfigDict()
    data.root = os.environ.get("MAST_DATA_ROOT", "./data/qm9sp")
    data.name = "QM9SP"
    data.dataset_arg = "homo"
    data.transform = "QM9SPTransform"
    data.collate = "collate_qm9sp"
    data.info_name = "qm9_with_h"
    data.num_workers = 4
    data.split_file = os.environ.get("MAST_SPLIT_FILE", "")
    data.compress_edge = True
    data.centered = True
    data.include_aromatic = False
    data.atom_types = 5
    data.bond_types = 4
    data.fc_scale = [-1.0, 1.0]
    data.max_node = 29
    data.spectra_modalities = ("uv", "ir", "raman")
    data.spectra_normalize = True

    config.sde = sde = ml_collections.ConfigDict()
    sde.schedule = "cosine"
    sde.continuous_beta_0 = 0.1
    sde.continuous_beta_1 = 20.0

    config.model = model = ml_collections.ConfigDict()
    model.name = "MASTDenoiser"
    model.pred_data = True
    model.include_fc_charge = True
    model.normalize_factors = "1, 4, 4, 1"
    model.ema_decay = 0.999
    model.edge_ch = 2
    model.nf = 256
    model.n_layers = 8
    model.n_heads = 16
    model.dropout = 0.1
    model.cond_time = True
    model.dist_gbf = True
    model.gbf_name = "CondGaussianLayer"
    model.self_cond = True
    model.self_cond_type = "ori"
    model.edge_quan_th = 0.0
    model.n_extra_heads = 2
    model.CoM = True
    model.mlp_ratio = 2
    model.spatial_cut_off = 2.0
    model.softmax_inf = True
    model.trans_name = "TransMixLayer"
    model.use_spectra = True
    model.spectra_modalities = ("uv", "ir", "raman")
    model.spectra_output_dim = 128
    model.spectra_d_model = 256
    model.spectra_depth = 4
    model.spectra_n_heads = 8
    model.spectra_fusion = "concat"
    model.spectra_patch_lens = (32, 64, 64)
    model.spectra_strides = (16, 32, 32)
    model.use_substructure = True
    model.substructure_checkpoint_path = os.environ.get("MAST_MOTIF_CKPT", "")
    model.freeze_substructure_predictor = True
    model.substructure_embedding_dim = 128
    model.substructure_hidden_dim = 256
    model.substructure_dropout = 0.1
    model.substructure_fusion = "gate"

    config.training = training = ml_collections.ConfigDict()
    training.reduce_mean = False
    training.batch_size = 1
    training.eval_batch_size = 1
    training.eval_samples = 1
    training.n_iters = 1
    training.snapshot_freq = 1
    training.snapshot_freq_for_preemption = 1
    training.log_freq = 1
    training.pretrained_checkpoint_path = ""
    training.reset_optimizer = True

    config.optim = optim = ml_collections.ConfigDict()
    optim.optimizer = "AdamW"
    optim.lr = 2e-4
    optim.beta1 = 0.9
    optim.eps = 1e-8
    optim.weight_decay = 1e-12
    optim.grad_clip = 1.0
    optim.warmup = 5000
    optim.disable_grad_log = True

    config.sampling = sampling = ml_collections.ConfigDict()
    sampling.method = "mcts"
    sampling.steps = 1000
    sampling.predictor = "euler_maruyama"
    sampling.corrector = "none"
    sampling.n_steps = 1000
    sampling.snr = 0.16
    sampling.scale_eps = 1.0
    sampling.n_steps_each = 1

    config.mcts = mcts = ml_collections.ConfigDict()
    mcts.n_checkpoints = 50
    mcts.checkpoint_kind = "linear"
    mcts.n_simulations = 100
    mcts.c_param = 1.0
    mcts.expand_width = 4
    mcts.n_actions = 20
    mcts.horizon = 3
    mcts.rollout_seed = 42
    mcts.fast_rollout_steps = 10
    mcts.reinit_noise_at_first_segment = True
    mcts.use_guidance = True
    mcts.guidance_scales = [0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4]
    mcts.guidance_as_action = True
    mcts.apply_to_pos_only = True
    mcts.time_scale_guidance = True
    mcts.value_threshold = 0.92
    mcts.top_k = 4
    mcts.verbose = True
    mcts.backup_strategy = "softmax"
    mcts.softmax_temperature = 0.5
    mcts.max_diffusion_steps = 3000
    mcts.max_tree_depth = None

    config.reward = reward = ml_collections.ConfigDict()
    reward.ckpt_path = os.environ.get("MAST_REWARD_CKPT", "")
    reward.use_cosine_similarity = True
    reward.use_substructure_reward = True
    reward.substructure_predictor_path = os.environ.get("MAST_MOTIF_CKPT", "")
    reward.substructure_weight = 0.1
    reward.use_3d_validity = True
    reward.invalid_3d_penalty = 0.5

    config.eval = evaluate = ml_collections.ConfigDict()
    evaluate.batch_size = 1
    evaluate.num_samples = 128
    evaluate.begin_ckpt = 1
    evaluate.end_ckpt = 1
    evaluate.ckpts = ""

    return config
