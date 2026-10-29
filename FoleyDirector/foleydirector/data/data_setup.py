# Copyright 2024 Sony Research Inc.
# Modifications Copyright 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import logging
import random

import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
from torch.utils.data.distributed import DistributedSampler

from foleydirector.data.mm_dataset import MultiModalDatasetSFT
from foleydirector.data.training_dataset import FoleyTrainingDataset
from foleydirector.utils.dist_utils import local_rank

log = logging.getLogger(__name__)


def worker_init_fn(worker_id: int):
    worker_seed = torch.initial_seed() % (2**31) + worker_id + local_rank * 1000
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def setup_training_datasets(cfg: DictConfig):
    datasets = []
    for data_cfg in cfg.data.datasets:
        if not data_cfg.get("enabled", True):
            continue
        datasets.append(
            FoleyTrainingDataset(
                manifest=data_cfg.manifest,
                memmap_dir=data_cfg.memmap_dir,
                annotations=data_cfg.annotations,
                data_dim=cfg.data_dim,
                fgc_pt=cfg.fgc_pt,
                drop_ratio=cfg.drop_ratio,
                video_drop_ratio=cfg.video_drop_ratio,
                text_drop_ratio=cfg.text_drop_ratio,
                with_details=cfg.with_details,
            ))
    if not datasets:
        raise ValueError("At least one enabled data.datasets entry is required")
    dataset = MultiModalDatasetSFT(datasets)
    sampler, loader = construct_loader(
        dataset,
        cfg.batch_size,
        cfg.num_workers,
        shuffle=True,
        drop_last=True,
        pin_memory=cfg.pin_memory,
    )
    return dataset, sampler, loader


def update_threshold(cfg, dataset: Dataset, new_threshold: float):
    log.warning("dynamic_threshold is ignored by the portable dataset format")
    sampler, loader = construct_loader(
        dataset,
        cfg.batch_size,
        cfg.num_workers,
        shuffle=True,
        drop_last=True,
        pin_memory=cfg.pin_memory,
    )
    return dataset, sampler, loader


def error_avoidance_collate(batch):
    batch = [sample for sample in batch if sample is not None]
    if not batch:
        raise ValueError("All samples in this batch failed to load")
    return default_collate(batch)


def construct_loader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    *,
    shuffle: bool = True,
    drop_last: bool = True,
    pin_memory: bool = False,
    error_avoidance: bool = False,
):
    sampler = DistributedSampler(
        dataset, rank=local_rank, shuffle=shuffle, drop_last=drop_last)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
        pin_memory=pin_memory,
        collate_fn=error_avoidance_collate if error_avoidance else None,
    )
    return sampler, loader
