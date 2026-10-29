# Copyright 2024 Sony Research Inc.
# Modifications Copyright 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from argparse import ArgumentParser
from datetime import timedelta
from pathlib import Path

import pandas as pd
import tensordict as td
import torch
import torch.distributed as distributed
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from foleydirector.data.data_setup import error_avoidance_collate
from foleydirector.data.extraction.vgg_sound import VGGSound
from foleydirector.model.utils.features_utils import FeaturesUtils
from foleydirector.utils.dist_utils import local_rank


def parse_args():
    parser = ArgumentParser(description="Extract audiovisual training latents")
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--variant", choices=("16k", "44k"), default="44k")
    parser.add_argument("--vae-checkpoint", type=Path, required=True)
    parser.add_argument("--vocoder-checkpoint", type=Path)
    parser.add_argument("--synchformer-checkpoint", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--normalize-audio", action="store_true")
    return parser.parse_args()


@torch.inference_mode()
def main():
    args = parse_args()
    distributed.init_process_group(backend="nccl", timeout=timedelta(hours=1))
    torch.cuda.set_device(local_rank)
    sample_rate = 16000 if args.variant == "16k" else 44100
    num_samples = 128000 if args.variant == "16k" else 353280
    if args.variant == "16k" and args.vocoder_checkpoint is None:
        raise ValueError("--vocoder-checkpoint is required for the 16k variant")

    dataset = VGGSound(
        args.video_dir,
        tsv_path=args.manifest,
        sample_rate=sample_rate,
        duration_sec=8.0,
        audio_samples=num_samples,
        normalize_audio=args.normalize_audio,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=DistributedSampler(dataset, rank=local_rank, shuffle=False),
        num_workers=args.num_workers,
        drop_last=False,
        collate_fn=error_avoidance_collate,
    )
    extractor = FeaturesUtils(
        tod_vae_ckpt=args.vae_checkpoint,
        bigvgan_vocoder_ckpt=args.vocoder_checkpoint,
        synchformer_ckpt=args.synchformer_checkpoint,
        enable_conditions=True,
        mode=args.variant,
    ).eval().cuda()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    if any(args.work_dir.glob("*.pth")):
        raise FileExistsError(
            f"{args.work_dir} contains old shards; use an empty --work-dir")
    for batch_index, batch in enumerate(tqdm(loader)):
        distribution = extractor.encode_audio(batch["audio"].cuda())
        output = {
            "id": list(batch["id"]),
            "caption": list(batch["caption"]),
            "mean": distribution.mean.cpu().transpose(1, 2),
            "std": distribution.std.cpu().transpose(1, 2),
            "clip_features": extractor.encode_video_with_clip(
                batch["clip_video"].cuda()).cpu(),
            "sync_features": extractor.encode_video_with_sync(
                batch["sync_video"].cuda()).cpu(),
            "text_features": extractor.encode_text(list(batch["caption"])).cpu(),
        }
        torch.save(output, args.work_dir / f"r{local_rank}_{batch_index:05d}.pth")

    distributed.barrier()
    if local_rank == 0:
        records = []
        keys = ("mean", "std", "clip_features", "sync_features", "text_features")
        tensors = {key: [] for key in keys}
        seen = set()
        for path in sorted(args.work_dir.glob("*.pth")):
            shard = torch.load(path, map_location="cpu", weights_only=True)
            for index, sample_id in enumerate(shard["id"]):
                if sample_id in seen:
                    continue
                seen.add(sample_id)
                records.append({"id": sample_id, "label": shard["caption"][index]})
                for key in keys:
                    tensors[key].append(shard[key][index])
        args.output_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(records).to_csv(
            args.output_dir.with_suffix(".tsv"), sep="\t", index=False)
        td.TensorDict({key: torch.stack(value) for key, value in tensors.items()}).memmap_(
            args.output_dir)
    distributed.barrier()
    distributed.destroy_process_group()


if __name__ == "__main__":
    main()
