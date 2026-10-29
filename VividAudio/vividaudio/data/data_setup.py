# Copyright 2024 Sony Research Inc.
# Modifications Copyright 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import random

import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
from torch.utils.data.distributed import DistributedSampler

from vividaudio.data.mm_dataset import MultiModalDatasetSFT
from vividaudio.data.training_dataset import VividTrainingDataset
from vividaudio.utils.dist_utils import local_rank


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
            VividTrainingDataset(
                manifest=data_cfg.manifest,
                memmap_dir=data_cfg.memmap_dir,
                annotations_dir=data_cfg.annotations_dir,
                audio_dir=data_cfg.audio_dir,
                feature_dir=data_cfg.get("feature_dir"),
                data_dim=cfg.data_dim,
                w_prefetch=cfg.w_prefetch,
                max_events=data_cfg.get("max_events", 10),
                video_drop_ratio=cfg.video_drop_ratio,
                text_drop_ratio=cfg.text_drop_ratio,
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
