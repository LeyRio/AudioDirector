# Copyright 2024 Sony Research Inc.
# Modifications Copyright 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from colorlog import ColoredFormatter
from PIL import Image
from torchvision.transforms import v2

from foleydirector.data.av_utils import ImageInfo, VideoInfo, read_frames, reencode_with_audio
from foleydirector.model.flow_matching import FlowMatching
from foleydirector.model.networks import MMAudio
from foleydirector.model.sequence_config import CONFIG_16K, CONFIG_44K, SequenceConfig
from foleydirector.model.utils.features_utils import FeaturesUtils
from foleydirector.utils.download_utils import download_model_if_needed
import os

mnt_path = os.environ.get("DATA_PREFIX", "./data_root")

log = logging.getLogger()


@dataclasses.dataclass
class ModelConfig:
    model_name: str
    model_path: Path
    vae_path: Path
    bigvgan_16k_path: Optional[Path]
    mode: str
    synchformer_ckpt: Path = Path(f'{mnt_path}/checkpoints/synchformer_state_dict.pth')

    @property
    def seq_cfg(self) -> SequenceConfig:
        if self.mode == '16k':
            return CONFIG_16K
        elif self.mode == '44k':
            return CONFIG_44K

    def download_if_needed(self):
        download_model_if_needed(self.model_path)
        download_model_if_needed(self.vae_path)
        if self.bigvgan_16k_path is not None:
            download_model_if_needed(self.bigvgan_16k_path)
        download_model_if_needed(self.synchformer_ckpt)


small_16k = ModelConfig(model_name='small_16k',
                        model_path=Path(f'{mnt_path}/checkpoints/mmaudio_small_16k.pth'),
                        vae_path=Path(f'{mnt_path}/checkpoints/v1-16.pth'),
                        bigvgan_16k_path=Path(f'{mnt_path}/checkpoints/best_netG.pt'),
                        mode='16k')
small_44k = ModelConfig(model_name='small_44k',
                        model_path=Path(f'{mnt_path}/checkpoints/mmaudio_small_44k.pth'),
                        vae_path=Path(f'{mnt_path}/checkpoints/v1-44.pth'),
                        bigvgan_16k_path=None,
                        mode='44k')
medium_44k = ModelConfig(model_name='medium_44k',
                         model_path=Path(f'{mnt_path}/checkpoints/mmaudio_medium_44k.pth'),
                         vae_path=Path(f'{mnt_path}/checkpoints/v1-44.pth'),
                         bigvgan_16k_path=None,
                         mode='44k')
large_44k = ModelConfig(model_name='large_44k',
                        model_path=Path(f'{mnt_path}/checkpoints/mmaudio_large_44k.pth'),
                        vae_path=Path(f'{mnt_path}/checkpoints/v1-44.pth'),
                        bigvgan_16k_path=None,
                        mode='44k')
large_44k_v2 = ModelConfig(model_name='large_44k_v2',
                           model_path=Path(f'{mnt_path}/checkpoints/mmaudio_large_44k_v2.pth'),
                           vae_path=Path(f'{mnt_path}/checkpoints/v1-44.pth'),
                           bigvgan_16k_path=None,
                           mode='44k')
all_model_cfg: dict[str, ModelConfig] = {
    'small_16k': small_16k,
    'small_44k': small_44k,
    'medium_44k': medium_44k,
    'large_44k': large_44k,
    'large_44k_v2': large_44k_v2,
}


def generate(
    clip_video: Optional[torch.Tensor],
    sync_video: Optional[torch.Tensor],
    text: Optional[list[str]],
    *,
    negative_text: Optional[list[str]] = None,
    feature_utils: FeaturesUtils,
    net: MMAudio,
    fm: FlowMatching,
    rng: torch.Generator,
    cfg_strength: float,
    clip_batch_size_multiplier: int = 40,
    sync_batch_size_multiplier: int = 40,
    image_input: bool = False,
) -> torch.Tensor:
    device = feature_utils.device
    dtype = feature_utils.dtype

    bs = len(text)
    if clip_video is not None:
        clip_video = clip_video.to(device, dtype, non_blocking=True)
        clip_features = feature_utils.encode_video_with_clip(clip_video,
                                                             batch_size=bs *
                                                             clip_batch_size_multiplier)
        if image_input:
            clip_features = clip_features.expand(-1, net.clip_seq_len, -1)
    else:
        clip_features = net.get_empty_clip_sequence(bs)

    if sync_video is not None and not image_input:
        sync_video = sync_video.to(device, dtype, non_blocking=True)
        sync_features = feature_utils.encode_video_with_sync(sync_video,
                                                             batch_size=bs *
                                                             sync_batch_size_multiplier)
    else:
        sync_features = net.get_empty_sync_sequence(bs)

    if text is not None:
        text_features = feature_utils.encode_text(text)
    else:
        text_features = net.get_empty_string_sequence(bs)


    if negative_text is not None:
        assert len(negative_text) == bs
        negative_text_features = feature_utils.encode_text(negative_text)
    else:
        negative_text_features = net.get_empty_string_sequence(bs)

    x0 = torch.randn(bs,
                     net.latent_seq_len,
                     net.latent_dim,
                     device=device,
                     dtype=dtype,
                     generator=rng)
    preprocessed_conditions = net.preprocess_conditions(clip_features, sync_features, text_features)
    empty_conditions = net.get_empty_conditions(
        bs, negative_text_features=negative_text_features if negative_text is not None else None)

    cfg_ode_wrapper = lambda t, x: net.ode_wrapper(t, x, preprocessed_conditions, empty_conditions,
                                                   cfg_strength)
    x1 = fm.to_data(cfg_ode_wrapper, x0)
    x1 = net.unnormalize(x1)
    spec = feature_utils.decode(x1)
    audio = feature_utils.vocode(spec)
    return audio


def extract_fgc_features(fgc_text: Optional[list[str]],
                         feature_utils: FeaturesUtils,
                         device: torch.device,
                         dtype: torch.dtype,
                         fgc_pt: int = 10,
                         fgc_pool: bool = False) -> torch.Tensor:
    b = len(fgc_text[0])   # batch size
    all_batch_feats = []

    for bi in range(b):
        texts_8 = [fgc_text[t][bi] for t in range(8)]

        feats = feature_utils.encode_text(texts_8)  # (8, T, D) or (8, D)
        if feats.ndim == 2:
            feats = feats.unsqueeze(1)  # 变成 (8, 1, D) for consistency

        if fgc_pool:
            pooled_feats = feats.mean(dim=1)  # (8, D)
            feats_10 = pooled_feats.unsqueeze(1).expand(-1, fgc_pt, -1)  # (8, fgc_pt, D)
        else:
            feats_10 = feats[:, :fgc_pt, :]  # (8, fgc_pt, D)

        feats_10 = feats_10.reshape(8 * fgc_pt, -1)
        all_batch_feats.append(feats_10)

    fgc_text_feature = torch.stack(all_batch_feats, dim=0).to(device, dtype, non_blocking=True)
    return fgc_text_feature

def generate_fgc(
    clip_video: Optional[torch.Tensor],
    sync_video: Optional[torch.Tensor],
    text: Optional[list[str]],
    fgc_text: Optional[list[str]],
    *,
    negative_text: Optional[list[str]] = None,
    feature_utils: FeaturesUtils,
    net: MMAudio,
    fm: FlowMatching,
    rng: torch.Generator,
    cfg_strength: float,
    clip_batch_size_multiplier: int = 40,
    sync_batch_size_multiplier: int = 40,
    image_input: bool = False,
    fgc_pt: int = 10,
    fgc_pool: bool = False,
    fgc_v2: bool = False,
) -> torch.Tensor:
    device = feature_utils.device
    dtype = feature_utils.dtype

    bs = len(text)
    if clip_video is not None:
        clip_video = clip_video.to(device, dtype, non_blocking=True)
        clip_features = feature_utils.encode_video_with_clip(clip_video,
                                                             batch_size=bs *
                                                             clip_batch_size_multiplier)
        if image_input:
            clip_features = clip_features.expand(-1, net.clip_seq_len, -1)
    else:
        clip_features = net.get_empty_clip_sequence(bs)

    if sync_video is not None and not image_input:
        sync_video = sync_video.to(device, dtype, non_blocking=True)
        sync_features = feature_utils.encode_video_with_sync(sync_video,
                                                             batch_size=bs *
                                                             sync_batch_size_multiplier)
    else:
        sync_features = net.get_empty_sync_sequence(bs)

    if text is not None:
        text_features = feature_utils.encode_text(text)
    else:
        text_features = net.get_empty_string_sequence(bs)

    # import pdb;pdb.set_trace()
    if fgc_text is not None:
        if not fgc_v2:
            fgc_text_feature = []
            for fgc_t in fgc_text:
                fgc_text_feature_in = extract_fgc_features(fgc_t, feature_utils, device, dtype, fgc_pt, fgc_pool)
                fgc_text_feature.append(fgc_text_feature_in)
            fgc_text_feature = torch.cat(fgc_text_feature, dim=0)
        # else:
        #     total_tokens = 80
        #     token_duration = 0.125

        #     bs = v2_tr.shape[0]
        #     all_batch_feats = []
        #     valid = [1 for i in range(bs)]
        #     for b in range(bs):
        #         categories = [i[b] for i in v2_cap]
        #         tr_mask = v2_tr[b]
        #         all_empty = all(all(not cap.strip() for cap in cap_tuple) for cap_tuple in categories)
        #         valid[b] = int(not all_empty)

        #         valid_texts = categories
        #         all_categories = valid_texts + [""]
        #         category_feats = encode_text_pool(all_categories, tokenizer, clip_model)  # b 77 1024
        #         category_embeddings = {}
        #         for i, category in enumerate(all_categories):
        #             category_embeddings[category] = category_feats[i]

        #         token_features = torch.zeros(total_tokens, category_feats.shape[-1]).to(category_feats[0].device)  # (80, D)
        #         token_features[:] = category_embeddings[""]  # 初始化为背景

        #         for i in range(len(valid_texts)):
        #             text = valid_texts[i]
        #             mask = tr_mask[i]  # (80,) bool tensor
        #             if text:
        #                 new_positions = mask

        #                 if new_positions.any():
        #                     token_features[new_positions] = category_embeddings[text]

        #         all_batch_feats.append(token_features)

        #     fgc_text_feature = torch.stack(all_batch_feats, dim=0).to(device, dtype, non_blocking=True)  # (b, 80, D)

    if negative_text is not None:
        assert len(negative_text) == bs
        negative_text_features = feature_utils.encode_text(negative_text)
    else:
        negative_text_features = net.get_empty_string_sequence(bs)

    x0 = torch.randn(bs,
                     net.latent_seq_len,
                     net.latent_dim,
                     device=device,
                     dtype=dtype,
                     generator=rng)
    preprocessed_conditions = net.preprocess_conditions(clip_features, sync_features, text_features, fgc_text_feature)
    empty_conditions = net.get_empty_conditions(
        bs, negative_text_features=negative_text_features if negative_text is not None else None)

    cfg_ode_wrapper = lambda t, x: net.ode_wrapper(t, x, preprocessed_conditions, empty_conditions,
                                                   cfg_strength)
    x1 = fm.to_data(cfg_ode_wrapper, x0)
    x1 = net.unnormalize(x1)
    spec = feature_utils.decode(x1)
    audio = feature_utils.vocode(spec)
    return audio


def generate_tracks(
    clip_video: Optional[torch.Tensor],
    sync_video: Optional[torch.Tensor],
    tracks: list[dict],
    *,
    negative_text: str,
    feature_utils: FeaturesUtils,
    net: MMAudio,
    fm: FlowMatching,
    rng: torch.Generator,
    cfg_strength: float,
    fgc_pt: int = 8,
    fgc_pool: bool = True,
) -> torch.Tensor:
    """Render structured tracks independently and fuse them on eight time slots."""
    if not tracks:
        raise ValueError("At least one track is required")

    num_tracks = len(tracks)
    device = feature_utils.device
    dtype = feature_utils.dtype

    if clip_video is None:
        clip_features = net.get_empty_clip_sequence(num_tracks)
    else:
        visual = feature_utils.encode_video_with_clip(
            clip_video.to(device, dtype, non_blocking=True),
            batch_size=40,
        )
        clip_features = torch.cat(
            [visual, net.get_empty_clip_sequence(num_tracks - 1)],
            dim=0,
        )

    if sync_video is None:
        sync_features = net.get_empty_sync_sequence(num_tracks)
    else:
        sync = feature_utils.encode_video_with_sync(
            sync_video.to(device, dtype, non_blocking=True),
            batch_size=40,
        )
        sync_features = sync.expand(num_tracks, -1, -1)

    prompts = [track["prompt"] for track in tracks]
    text_features = feature_utils.encode_text(prompts)
    fgc_features = []
    for track in tracks:
        event_batch = [[event] for event in track["events"]]
        fgc_features.append(
            extract_fgc_features(
                event_batch,
                feature_utils,
                device,
                dtype,
                fgc_pt,
                fgc_pool,
            )
        )
    fgc_features = torch.cat(fgc_features, dim=0)

    negative_features = feature_utils.encode_text([negative_text] * num_tracks)
    x0 = torch.randn(
        1,
        net.latent_seq_len,
        net.latent_dim,
        device=device,
        dtype=dtype,
        generator=rng,
    )
    conditions = net.preprocess_conditions(
        clip_features,
        sync_features,
        text_features,
        fgc_features,
    )
    empty_conditions = net.get_empty_conditions(
        num_tracks,
        negative_text_features=negative_features,
    )
    track_weights = [track["weights"] for track in tracks]

    def cfg_ode_wrapper(t, x):
        return net.ode_wrapper(
            t,
            x,
            conditions,
            empty_conditions,
            cfg_strength,
            track_weights,
        )

    x1 = fm.to_data(cfg_ode_wrapper, x0)
    spec = feature_utils.decode(net.unnormalize(x1))
    return feature_utils.vocode(spec)


LOGFORMAT = "[%(log_color)s%(levelname)-8s%(reset)s]: %(log_color)s%(message)s%(reset)s"


def setup_eval_logging(log_level: int = logging.INFO):
    logging.root.setLevel(log_level)
    formatter = ColoredFormatter(LOGFORMAT)
    stream = logging.StreamHandler()
    stream.setLevel(log_level)
    stream.setFormatter(formatter)
    log = logging.getLogger()
    log.setLevel(log_level)
    log.addHandler(stream)


_CLIP_SIZE = 384
_CLIP_FPS = 8.0

_SYNC_SIZE = 224
_SYNC_FPS = 25.0


def load_video(video_path: Path, duration_sec: float, load_all_frames: bool = True) -> VideoInfo:

    clip_transform = v2.Compose([
        v2.Resize((_CLIP_SIZE, _CLIP_SIZE), interpolation=v2.InterpolationMode.BICUBIC),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ])

    sync_transform = v2.Compose([
        v2.Resize(_SYNC_SIZE, interpolation=v2.InterpolationMode.BICUBIC),
        v2.CenterCrop(_SYNC_SIZE),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    output_frames, all_frames, orig_fps = read_frames(video_path,
                                                      list_of_fps=[_CLIP_FPS, _SYNC_FPS],
                                                      start_sec=0,
                                                      end_sec=duration_sec,
                                                      need_all_frames=load_all_frames)

    clip_chunk, sync_chunk = output_frames
    clip_chunk = torch.from_numpy(clip_chunk).permute(0, 3, 1, 2)
    sync_chunk = torch.from_numpy(sync_chunk).permute(0, 3, 1, 2)

    clip_frames = clip_transform(clip_chunk)
    sync_frames = sync_transform(sync_chunk)

    clip_length_sec = clip_frames.shape[0] / _CLIP_FPS
    sync_length_sec = sync_frames.shape[0] / _SYNC_FPS

    if clip_length_sec < duration_sec:
        log.warning(f'Clip video is too short: {clip_length_sec:.2f} < {duration_sec:.2f}')
        log.warning(f'Truncating to {clip_length_sec:.2f} sec')
        duration_sec = clip_length_sec

    if sync_length_sec < duration_sec:
        log.warning(f'Sync video is too short: {sync_length_sec:.2f} < {duration_sec:.2f}')
        log.warning(f'Truncating to {sync_length_sec:.2f} sec')
        duration_sec = sync_length_sec

    clip_frames = clip_frames[:int(_CLIP_FPS * duration_sec)]
    sync_frames = sync_frames[:int(_SYNC_FPS * duration_sec)]

    video_info = VideoInfo(
        duration_sec=duration_sec,
        fps=orig_fps,
        clip_frames=clip_frames,
        sync_frames=sync_frames,
        all_frames=all_frames if load_all_frames else None,
    )
    return video_info


def load_image(image_path: Path) -> VideoInfo:
    clip_transform = v2.Compose([
        v2.Resize((_CLIP_SIZE, _CLIP_SIZE), interpolation=v2.InterpolationMode.BICUBIC),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ])

    sync_transform = v2.Compose([
        v2.Resize(_SYNC_SIZE, interpolation=v2.InterpolationMode.BICUBIC),
        v2.CenterCrop(_SYNC_SIZE),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    frame = np.array(Image.open(image_path))

    clip_chunk = torch.from_numpy(frame).unsqueeze(0).permute(0, 3, 1, 2)
    sync_chunk = torch.from_numpy(frame).unsqueeze(0).permute(0, 3, 1, 2)

    clip_frames = clip_transform(clip_chunk)
    sync_frames = sync_transform(sync_chunk)

    video_info = ImageInfo(
        clip_frames=clip_frames,
        sync_frames=sync_frames,
        original_frame=frame,
    )
    return video_info


def make_video(video_info: VideoInfo, output_path: Path, audio: torch.Tensor, sampling_rate: int):
    reencode_with_audio(video_info, output_path, audio, sampling_rate)

def apply_camera_motion(frames, camera_motion):
    """对一组视频帧应用动态相机运动（平移/缩放/混合）"""
    if camera_motion is None:
        return frames

    start_bbox = np.array(camera_motion.get("start_bbox", (0, 0, 1, 1)), dtype=np.float32)
    end_bbox = np.array(camera_motion.get("end_bbox", (0, 0, 1, 1)), dtype=np.float32)
    ease = camera_motion.get("ease", "linear")
    T, C, H, W = frames.shape

    out_frames = []
    for i in range(T):
        t = i / (T - 1)
        # 插值曲线
        if ease == "ease_in":
            t = t * t
        elif ease == "ease_out":
            t = 1 - (1 - t) ** 2
        # 线性插值 bbox
        bbox = start_bbox * (1 - t) + end_bbox * t
        x1, y1, x2, y2 = bbox
        # 计算像素坐标
        px1, py1, px2, py2 = int(x1 * W), int(y1 * H), int(x2 * W), int(y2 * H)
        px1, py1 = max(0, px1), max(0, py1)
        px2, py2 = min(W, px2), min(H, py2)
        crop = frames[i, :, py1:py2, px1:px2]
        # 缩放回原分辨率
        crop_resized = torch.nn.functional.interpolate(
            crop.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False
        ).squeeze(0)
        out_frames.append(crop_resized)

    return torch.stack(out_frames)

def load_video_cinematic(video_path: Path, duration_sec: float, load_all_frames: bool = True, camera_motion=None) -> VideoInfo:
    clip_transform = v2.Compose([
        v2.Resize((_CLIP_SIZE, _CLIP_SIZE), interpolation=v2.InterpolationMode.BICUBIC),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ])

    sync_transform = v2.Compose([
        v2.Resize(_SYNC_SIZE, interpolation=v2.InterpolationMode.BICUBIC),
        v2.CenterCrop(_SYNC_SIZE),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    output_frames, all_frames, orig_fps = read_frames(
        video_path,
        list_of_fps=[_CLIP_FPS, _SYNC_FPS],
        start_sec=0,
        end_sec=duration_sec,
        need_all_frames=load_all_frames
    )

    clip_chunk, sync_chunk = output_frames
    clip_chunk = torch.from_numpy(clip_chunk).permute(0, 3, 1, 2)
    sync_chunk = torch.from_numpy(sync_chunk).permute(0, 3, 1, 2)

    # 应用相机运动
    if camera_motion:
        clip_chunk = apply_camera_motion(clip_chunk, camera_motion)
        sync_chunk = apply_camera_motion(sync_chunk, camera_motion)

    clip_frames = clip_transform(clip_chunk)
    sync_frames = sync_transform(sync_chunk)

    clip_length_sec = clip_frames.shape[0] / _CLIP_FPS
    sync_length_sec = sync_frames.shape[0] / _SYNC_FPS

    if clip_length_sec < duration_sec:
        log.warning(f'Clip video is too short: {clip_length_sec:.2f} < {duration_sec:.2f}')
        log.warning(f'Truncating to {clip_length_sec:.2f} sec')
        duration_sec = clip_length_sec

    if sync_length_sec < duration_sec:
        log.warning(f'Sync video is too short: {sync_length_sec:.2f} < {duration_sec:.2f}')
        log.warning(f'Truncating to {sync_length_sec:.2f} sec')
        duration_sec = sync_length_sec

    clip_frames = clip_frames[:int(_CLIP_FPS * duration_sec)]
    sync_frames = sync_frames[:int(_SYNC_FPS * duration_sec)]

    video_info = VideoInfo(
        duration_sec=duration_sec,
        fps=orig_fps,
        clip_frames=clip_frames,
        sync_frames=sync_frames,
        all_frames=all_frames if load_all_frames else None,
    )
    return video_info
