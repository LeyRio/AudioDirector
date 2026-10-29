# Copyright 2024 Sony Research Inc.
# Modifications Copyright 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import logging
from argparse import ArgumentParser
from datetime import timedelta
from pathlib import Path

import pandas as pd
import tensordict as td
import torch
import torch.distributed as distributed
import torch.nn.functional as F
from open_clip import create_model_from_pretrained
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from foleydirector.data.data_setup import error_avoidance_collate
from foleydirector.data.extraction.wav_dataset import WavTextClipsDataset
from foleydirector.ext.autoencoder import AutoEncoderModule
from foleydirector.ext.mel_converter import get_mel_converter
from foleydirector.utils.dist_utils import local_rank

log = logging.getLogger(__name__)


def parse_args():
    parser = ArgumentParser(description="Extract audio latent and text features")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--captions-tsv", type=Path, required=True)
    parser.add_argument("--clips-tsv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--variant", choices=("16k", "44k"), default="44k")
    parser.add_argument("--vae-checkpoint", type=Path, required=True)
    parser.add_argument("--vocoder-checkpoint", type=Path)
    parser.add_argument("--clip-model-dir", type=Path, required=True)
    parser.add_argument("--clip-tokenizer-dir", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    return parser.parse_args()


def setup_distributed():
    distributed.init_process_group(backend="nccl", timeout=timedelta(hours=1))
    torch.cuda.set_device(local_rank)


@torch.inference_mode()
def main():
    args = parse_args()
    setup_distributed()
    sample_rate = 16000 if args.variant == "16k" else 44100
    num_samples = 128000 if args.variant == "16k" else 353280
    if args.variant == "16k" and args.vocoder_checkpoint is None:
        raise ValueError("--vocoder-checkpoint is required for the 16k variant")

    clip_model = create_model_from_pretrained(
        f"local-dir:{args.clip_model_dir}", return_transform=False).eval().cuda()

    def encode_text(self, tokens, normalize=False):
        cast_dtype = self.transformer.get_cast_dtype()
        hidden = self.token_embedding(tokens).to(cast_dtype)
        hidden = hidden + self.positional_embedding.to(cast_dtype)
        hidden = self.transformer(hidden, attn_mask=self.attn_mask)
        hidden = self.ln_final(hidden)
        return F.normalize(hidden, dim=-1) if normalize else hidden

    clip_model.encode_text = encode_text.__get__(clip_model)
    autoencoder = AutoEncoderModule(
        vae_ckpt_path=args.vae_checkpoint,
        vocoder_ckpt_path=args.vocoder_checkpoint,
        mode=args.variant,
    ).eval().cuda()
    mel_converter = get_mel_converter(args.variant).eval().cuda()
    dataset = WavTextClipsDataset(
        args.data_dir,
        captions_tsv=args.captions_tsv,
        clips_tsv=args.clips_tsv,
        sample_rate=sample_rate,
        num_samples=num_samples,
        normalize_audio=True,
        reject_silent=True,
        tokenizer_id=args.clip_tokenizer_dir,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=DistributedSampler(dataset, rank=local_rank, shuffle=False),
        num_workers=args.num_workers,
        drop_last=False,
        collate_fn=error_avoidance_collate,
    )
    args.work_dir.mkdir(exist_ok=True, parents=True)
    for batch_index, batch in tqdm(enumerate(loader), total=len(loader)):
        waveforms = batch["waveform"].cuda()
        distribution = autoencoder.encode(mel_converter(waveforms))
        torch.save(
            {
                "id": list(batch["id"]),
                "caption": list(batch["caption"]),
                "mean": distribution.mean.cpu().transpose(1, 2),
                "std": distribution.std.cpu().transpose(1, 2),
                "text_features": clip_model.encode_text(
                    batch["tokens"].cuda(), normalize=True).cpu(),
            },
            args.work_dir / f"r{local_rank}_{batch_index:05d}.pth",
        )

    distributed.barrier()
    if local_rank == 0:
        records = []
        tensors = {"mean": [], "std": [], "text_features": []}
        for path in sorted(args.work_dir.glob("*.pth")):
            shard = torch.load(path, map_location="cpu", weights_only=True)
            for index, sample_id in enumerate(shard["id"]):
                records.append({"id": sample_id, "caption": shard["caption"][index]})
                for key in tensors:
                    tensors[key].append(shard[key][index])
        args.output_dir.mkdir(exist_ok=True, parents=True)
        pd.DataFrame(records).to_csv(
            args.output_dir.with_suffix(".tsv"), sep="\t", index=False)
        td.TensorDict({key: torch.stack(value) for key, value in tensors.items()}).memmap_(
            args.output_dir)
    distributed.barrier()
    distributed.destroy_process_group()


if __name__ == "__main__":
    main()
