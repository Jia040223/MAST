import logging
import os
import pickle
import random
from pathlib import Path

import numpy as np
import torch
from rdkit import RDLogger

import losses
from datasets import get_dataset, get_dataloader, inf_iterator
from diffusion import NoiseScheduleVP
from models import create_model, get_node_dist
from models.ema import ExponentialMovingAverage
from sampling import get_sampling_fn
from utils import (
    get_data_inverse_scaler,
    get_data_scaler,
    load_pretrained_checkpoint,
    restore_checkpoint,
    save_checkpoint,
)


def set_random_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _setup_logging(log_path: Path):
    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(levelname)s - %(asctime)s - %(message)s")
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)


def _maybe_load_pretrained_checkpoint(config, state):
    ckpt_path = getattr(config.training, "pretrained_checkpoint_path", "")
    if ckpt_path:
        logging.info("Loading pretrained diffusion checkpoint from %s", ckpt_path)
        state = load_pretrained_checkpoint(
            ckpt_path,
            state,
            config.device,
            reset_optimizer=getattr(config.training, "reset_optimizer", True),
        )
    return state


def train(config, workdir: str):
    RDLogger.DisableLog("rdApp.*")
    set_random_seed(int(config.seed))

    workdir = Path(workdir)
    _setup_logging(workdir / "train.log")

    train_ds, val_ds, test_ds, dataset_info = get_dataset(config)
    train_loader, _, _ = get_dataloader(train_ds, val_ds, test_ds, config)
    train_iter = inf_iterator(train_loader)

    model = create_model(config)
    ema = ExponentialMovingAverage(model.parameters(), decay=config.model.ema_decay)
    optimizer = losses.get_optimizer(config, model.parameters())
    state = {"optimizer": optimizer, "model": model, "ema": ema, "step": 0}

    checkpoint_dir = workdir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_meta = workdir / "checkpoints-meta" / "checkpoint.pth"
    checkpoint_meta.parent.mkdir(parents=True, exist_ok=True)

    state = _maybe_load_pretrained_checkpoint(config, state)
    state = restore_checkpoint(str(checkpoint_meta), state, config.device)
    initial_step = int(state["step"])

    model_size_mb = sum(p.numel() for p in model.parameters()) * 4 / 2 ** 20
    logging.info("Model size: %.1f MB", model_size_mb)
    logging.info("Dataset sizes: train=%d, val=%d, test=%d", len(train_ds), len(val_ds), len(test_ds))
    logging.info(config)

    noise_scheduler = NoiseScheduleVP(
        config.sde.schedule,
        continuous_beta_0=config.sde.continuous_beta_0,
        continuous_beta_1=config.sde.continuous_beta_1,
    )
    scaler = get_data_scaler(config)
    optimize_fn = losses.optimization_manager(config)
    train_step_fn = losses.get_step_fn(noise_scheduler, True, optimize_fn, scaler, config)

    for step in range(initial_step, int(config.training.n_iters) + 1):
        batch = next(train_iter)
        loss = train_step_fn(state, batch)

        if step % int(config.training.log_freq) == 0:
            logging.info("step=%d loss=%.6e", step, loss.item())

        if step != 0 and step % int(config.training.snapshot_freq_for_preemption) == 0:
            save_checkpoint(str(checkpoint_meta), state)

        if step != 0 and step % int(config.training.snapshot_freq) == 0:
            save_step = step // int(config.training.snapshot_freq)
            save_checkpoint(str(checkpoint_dir / f"checkpoint_{save_step}.pth"), state)

        del batch
        del loss

    save_checkpoint(str(checkpoint_meta), state)


def sample(config, workdir: str, checkpoint_path: str = None, output_path: str = None):
    RDLogger.DisableLog("rdApp.*")
    set_random_seed(int(config.seed))

    workdir = Path(workdir)
    _setup_logging(workdir / "sample.log")

    train_ds, _, test_ds, dataset_info = get_dataset(config, transform=False)
    nodes_dist = get_node_dist(dataset_info)
    model = create_model(config)
    ema = ExponentialMovingAverage(model.parameters(), decay=config.model.ema_decay)
    optimizer = losses.get_optimizer(config, model.parameters())
    state = {"optimizer": optimizer, "model": model, "ema": ema, "step": 0}

    if checkpoint_path is None:
        ckpts = getattr(config.eval, "ckpts", "")
        if ckpts:
            checkpoint_id = ckpts.split(",")[0].strip()
            checkpoint_path = workdir / "checkpoints" / f"checkpoint_{checkpoint_id}.pth"
        else:
            checkpoint_path = workdir / "checkpoints" / f"checkpoint_{int(config.eval.end_ckpt)}.pth"

    checkpoint_path = str(checkpoint_path)
    logging.info("Loading checkpoint from %s", checkpoint_path)
    state = restore_checkpoint(checkpoint_path, state, config.device)
    ema.copy_to(model.parameters())
    model.eval()

    noise_scheduler = NoiseScheduleVP(
        config.sde.schedule,
        continuous_beta_0=config.sde.continuous_beta_0,
        continuous_beta_1=config.sde.continuous_beta_1,
    )
    inverse_scaler = get_data_inverse_scaler(config)
    sampling_fn = get_sampling_fn(
        config,
        noise_scheduler,
        nodes_dist,
        int(config.eval.batch_size),
        int(config.eval.num_samples),
        inverse_scaler,
    )
    samples = sampling_fn(model)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(samples, f)
    logging.info("Saved %d samples to %s", len(samples), output_path)
