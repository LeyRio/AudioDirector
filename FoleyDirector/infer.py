# Copyright 2024 Sony Research Inc.
# Modifications Copyright 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import json
import logging
from argparse import ArgumentParser
from pathlib import Path

import torch
import torchaudio

from foleydirector.eval_utils import (
    ModelConfig,
    all_model_cfg,
    generate_tracks,
    load_video,
    make_video,
    setup_eval_logging,
)
from foleydirector.model.flow_matching import FlowMatching
from foleydirector.model.networks import get_my_mmaudio
from foleydirector.model.utils.features_utils import FeaturesUtils

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

log = logging.getLogger(__name__)


def load_tracks(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        payload = json.load(file)
    tracks = payload["tracks"] if isinstance(payload, dict) else payload
    if not isinstance(tracks, list) or not tracks:
        raise ValueError("tracks JSON must contain a non-empty 'tracks' list")

    for index, track in enumerate(tracks):
        if not isinstance(track, dict) or not isinstance(track.get("prompt"), str):
            raise ValueError(f"tracks[{index}].prompt must be a string")
        events = track.get("events")
        weights = track.get("weights")
        if not isinstance(events, list) or len(events) != 8:
            raise ValueError(f"tracks[{index}].events must contain exactly 8 strings")
        if not all(isinstance(event, str) for event in events):
            raise ValueError(f"tracks[{index}].events must contain only strings")
        if not isinstance(weights, list) or len(weights) != 8:
            raise ValueError(f"tracks[{index}].weights must contain exactly 8 numbers")
        if not all(isinstance(weight, (int, float)) and weight >= 0 for weight in weights):
            raise ValueError(f"tracks[{index}].weights must be non-negative numbers")

    for segment in range(8):
        if sum(float(track["weights"][segment]) for track in tracks) <= 0:
            raise ValueError(f"segment {segment} has zero weight across every track")
    return tracks


def load_checkpoint(path: Path, device: str) -> dict:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must be a state dict or contain a 'weights' state dict")
    return checkpoint.get("weights", checkpoint)


@torch.inference_mode()
def main() -> None:
    setup_eval_logging()
    parser = ArgumentParser(description="FoleyDirector multi-track inference")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--tracks", type=Path, required=True, help="Structured track JSON")
    parser.add_argument("--output", type=Path, default=Path("./output"))
    parser.add_argument("--variant", default="medium_44k", choices=tuple(all_model_cfg))
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--cfg-strength", type=float, default=4.5)
    parser.add_argument("--num-steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fgc-pt", type=int, default=8)
    parser.add_argument("--full-precision", action="store_true")
    parser.add_argument("--audio-only", action="store_true")
    args = parser.parse_args()

    for path in (args.model_path, args.video, args.tracks):
        if not path.expanduser().is_file():
            parser.error(f"file does not exist: {path}")
    tracks = load_tracks(args.tracks.expanduser())

    model: ModelConfig = all_model_cfg[args.variant]
    seq_cfg = model.seq_cfg
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32 if args.full_precision or device == "cpu" else torch.bfloat16

    net = get_my_mmaudio(
        model.model_name,
        sep_fgc=False,
        joint_fgc=False,
        delay_fusion=True,
        interleaved_rope=True,
        fusion_fgc_clip=False,
        fgc_pt=args.fgc_pt,
    ).to(device, dtype).eval()
    net.load_weights(load_checkpoint(args.model_path.expanduser(), device))
    log.info("Loaded weights from %s", args.model_path)

    feature_utils = FeaturesUtils(
        tod_vae_ckpt=model.vae_path,
        synchformer_ckpt=model.synchformer_ckpt,
        enable_conditions=True,
        mode=model.mode,
        bigvgan_vocoder_ckpt=model.bigvgan_16k_path,
        need_vae_encoder=False,
    ).to(device, dtype).eval()

    video_path = args.video.expanduser()
    video_info = load_video(video_path, args.duration)
    seq_cfg.duration = video_info.duration_sec
    net.update_seq_lengths(
        seq_cfg.latent_seq_len,
        seq_cfg.clip_seq_len,
        seq_cfg.sync_seq_len,
    )
    rng = torch.Generator(device=device).manual_seed(args.seed)
    fm = FlowMatching(min_sigma=0, inference_mode="euler", num_steps=args.num_steps)
    audio = generate_tracks(
        video_info.clip_frames.unsqueeze(0),
        video_info.sync_frames.unsqueeze(0),
        tracks,
        negative_text=args.negative_prompt,
        feature_utils=feature_utils,
        net=net,
        fm=fm,
        rng=rng,
        cfg_strength=args.cfg_strength,
        fgc_pt=args.fgc_pt,
        fgc_pool=True,
    ).float().cpu()[0]

    args.output.mkdir(parents=True, exist_ok=True)
    audio_path = args.output / f"{video_path.stem}.flac"
    torchaudio.save(audio_path, audio, seq_cfg.sampling_rate)
    log.info("Audio saved to %s", audio_path)
    if not args.audio_only:
        video_output = args.output / f"{video_path.stem}.mp4"
        make_video(video_info, video_output, audio, sampling_rate=seq_cfg.sampling_rate)
        log.info("Video saved to %s", video_output)


if __name__ == "__main__":
    main()
