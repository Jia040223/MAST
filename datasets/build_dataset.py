from torch.utils.data import DataLoader

from .datasets_config import get_dataset_info
from .qm9sp_dataset import QM9SP, QM9SPTransform, collate_qm9sp


def get_dataset(config, transform=True):
    dataset_info = get_dataset_info(config.data.info_name)
    dataset_transform = None
    if transform:
        dataset_transform = QM9SPTransform(
            atom_type_list=[0, 1, 2, 3, 4],
            include_aromatic=config.data.include_aromatic,
            spectra_normalize=getattr(config.data, "spectra_normalize", True),
        )

    dataset = QM9SP(
        config.data.root,
        dataset_arg=getattr(config.data, "dataset_arg", "homo"),
        modalities=getattr(config.data, "spectra_modalities", ("uv", "ir", "raman")),
        transform=dataset_transform,
    )

    split_file = getattr(config.data, "split_file", "")
    split_idx = dataset.get_idx_split(split_file=split_file or None)
    train_dataset = dataset.index_select(split_idx["train"])
    val_dataset = dataset.index_select(split_idx["valid"])
    test_dataset = dataset.index_select(split_idx["test"])
    return train_dataset, val_dataset, test_dataset, dataset_info


def inf_iterator(iterable):
    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable)


def get_dataloader(train_ds, val_ds, test_ds, config):
    train_loader = DataLoader(
        train_ds,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        collate_fn=collate_qm9sp,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.training.eval_batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        collate_fn=collate_qm9sp,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config.training.eval_batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        collate_fn=collate_qm9sp,
    )
    return train_loader, val_loader, test_loader
