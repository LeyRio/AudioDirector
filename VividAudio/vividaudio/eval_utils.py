# Copyright 2024 Sony Research Inc.
# Modifications Copyright 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import dataclasses
import logging
import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from colorlog import ColoredFormatter
from PIL import Image
from torchvision.transforms import v2

from vividaudio.data.av_utils import ImageInfo, VideoInfo, read_frames, reencode_with_audio
from vividaudio.model.flow_matching import FlowMatching
from vividaudio.model.networks import MMAudio
from vividaudio.model.sequence_config import CONFIG_16K, CONFIG_44K, SequenceConfig
from vividaudio.model.utils.features_utils import FeaturesUtils
from vividaudio.utils.download_utils import download_model_if_needed
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


def generate_rag(
    clip_video: Optional[torch.Tensor],
    sync_video: Optional[torch.Tensor],
    text: Optional[list[str]],
    ref_audio,
    *,
    negative_text: Optional[list[str]] = None,
    feature_utils: FeaturesUtils,
    net: MMAudio,
    fm: FlowMatching,
    rng: torch.Generator,
    cfg_strength: float,
    cfg_rag_weight: float = 0.5,
    cfg_schedule_mode: str = 'base',
    clip_batch_size_multiplier: int = 40,
    sync_batch_size_multiplier: int = 40,
    image_input: bool = False,
    fgc_pt: int = 10,
    fgc_pool: bool = False,
    fgc_v2: bool = False,
    masked_x1: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    negative_rag_audio_emb: Optional[torch.Tensor] = None,
    latent_unmasked: Optional[torch.Tensor] = None,
    rag_vae_emb: Optional[torch.Tensor] = None,
    negative_rag_vae_emb: Optional[torch.Tensor] = None,
    grag_audio_emb: Optional[torch.Tensor] = None,
    negative_grag_audio_emb: Optional[torch.Tensor] = None,
    cfg_t_lo: float = 0.0,
    cfg_t_hi: float = 1.0,
    cfg_rescale_std: float = 0.0,
) -> torch.Tensor:
    device = feature_utils.device
    # dtype = feature_utils.dtype
    dtype = clip_video.dtype

    cfg_schedule_mode_norm = (cfg_schedule_mode or 'base').strip().lower()
    if cfg_schedule_mode_norm in {'0', 'base', 'default'}:
        cfg_schedule_mode_norm = 'base'
    elif cfg_schedule_mode_norm in {'1', 'scheme1', 'plan1', 's_decay', 'strength_decay'}:
        cfg_schedule_mode_norm = 'scheme1'
    elif cfg_schedule_mode_norm in {'2', 'scheme2', 'plan2', 's_decay_rag_ramp', 'strength_decay_rag_ramp'}:
        cfg_schedule_mode_norm = 'scheme2'
    else:
        raise ValueError(f'Unknown cfg_schedule_mode: {cfg_schedule_mode!r}')

    def _cfg_values_at_t(t_in) -> tuple[float, float]:
        if cfg_schedule_mode_norm == 'base':
            s_val, a_val = float(cfg_strength), float(cfg_rag_weight)
        else:
            t_val = t_in.item() if isinstance(t_in, torch.Tensor) else float(t_in)
            if getattr(fm, 'inference_mode', 'euler') == 'euler':
                denom = 1.0 - float(getattr(fm, 'min_sigma', 0.0))
                if denom > 0:
                    t_val = t_val / denom
            p = max(0.0, min(1.0, t_val))

            s_max = float(cfg_strength)
            gamma = 3.0
            s_t = 1.0 + (s_max - 1.0) * ((1.0 - p) ** gamma)

            if cfg_schedule_mode_norm == 'scheme1':
                s_val, a_val = float(s_t), float(cfg_rag_weight)
            else:
                alpha_max = float(cfg_rag_weight)
                alpha_min = alpha_max * 0.20
                m = 0.6
                tau = 0.12
                z = (p - m) / tau
                sig = 1.0 / (1.0 + math.exp(-z))
                alpha_t = alpha_min + (alpha_max - alpha_min) * sig
                s_val, a_val = float(s_t), float(alpha_t)

        # Plan B interval gating: 在 [cfg_t_lo, cfg_t_hi] 之外关闭 CFG (s_t=1)
        if cfg_t_lo > 0.0 or cfg_t_hi < 1.0:
            t_scalar = t_in.item() if isinstance(t_in, torch.Tensor) else float(t_in)
            if getattr(fm, 'inference_mode', 'euler') == 'euler':
                denom = 1.0 - float(getattr(fm, 'min_sigma', 0.0))
                if denom > 0:
                    t_scalar = t_scalar / denom
            if not (cfg_t_lo <= t_scalar <= cfg_t_hi):
                s_val = 1.0
        return s_val, a_val

    def _rescale(flow_guided: torch.Tensor, flow_cond: torch.Tensor) -> torch.Tensor:
        # Plan B rescale-CFG: 用 conditional 的 std 缩回 guided 输出, 抑制过饱和
        if cfg_rescale_std <= 0.0:
            return flow_guided
        dims = list(range(1, flow_guided.ndim))
        std_g = flow_guided.float().std(dim=dims, keepdim=True).clamp_min(1e-8)
        std_c = flow_cond.float().std(dim=dims, keepdim=True).clamp_min(1e-8)
        rescaled = flow_guided * (std_c / std_g).to(flow_guided.dtype)
        phi = float(cfg_rescale_std)
        return phi * rescaled + (1.0 - phi) * flow_guided

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
    preprocessed_conditions = net.preprocess_conditions(
        clip_features, sync_features, text_features, ref_audio, rag_vae_emb, grag_f=grag_audio_emb
    )
    neg_preprocessed_conditions = net.preprocess_conditions(
        clip_features, sync_features, text_features, negative_rag_audio_emb, negative_rag_vae_emb,
        grag_f=negative_grag_audio_emb
    )
    empty_conditions = net.get_empty_conditions(bs, negative_text_features=negative_text_features if negative_text is not None else None)

    # empty_conditions.rag_f = neg_preprocessed_conditions.rag_f

    if masked_x1 is not None and mask is not None:
        def cfg_ode_wrapper(t, x):
            s_t, alpha_t = _cfg_values_at_t(t)
            x_full = torch.cat([x, masked_x1, mask], dim=-1)
            flow_guided = net.ode_wrapper(
                t,
                x_full,
                preprocessed_conditions,
                empty_conditions,
                s_t,
                neg_conditions=neg_preprocessed_conditions,
                cfg_rag_weight=alpha_t,
            )
            if cfg_rescale_std > 0.0 and s_t >= 1.0:
                t_bcast = t * torch.ones(len(x_full), device=x_full.device, dtype=x_full.dtype)
                flow_cond = net.predict_flow(x_full, t_bcast, preprocessed_conditions)
                flow_guided = _rescale(flow_guided, flow_cond)
            return flow_guided
    else:
        def cfg_ode_wrapper(t, x):
            s_t, alpha_t = _cfg_values_at_t(t)
            flow_guided = net.ode_wrapper(
                t,
                x,
                preprocessed_conditions,
                empty_conditions,
                s_t,
                neg_conditions=neg_preprocessed_conditions,
                cfg_rag_weight=alpha_t,
            )
            if cfg_rescale_std > 0.0 and s_t >= 1.0:
                t_bcast = t * torch.ones(len(x), device=x.device, dtype=x.dtype)
                flow_cond = net.predict_flow(x, t_bcast, preprocessed_conditions)
                flow_guided = _rescale(flow_guided, flow_cond)
            return flow_guided
    x1 = fm.to_data(cfg_ode_wrapper, x0)
    x1 = net.unnormalize(x1)
    spec = feature_utils.decode(x1)
    audio = feature_utils.vocode(spec)
    return audio


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
        log.warning(f'Padding to {duration_sec:.2f} sec')
        target_len = int(_CLIP_FPS * duration_sec)
        pad_len = target_len - clip_frames.shape[0]
        if pad_len > 0:
            clip_frames = torch.cat([clip_frames, clip_frames[-1:].repeat(pad_len, 1, 1, 1)], dim=0)

    if sync_length_sec < duration_sec:
        log.warning(f'Sync video is too short: {sync_length_sec:.2f} < {duration_sec:.2f}')
        log.warning(f'Padding to {duration_sec:.2f} sec')
        target_len = int(_SYNC_FPS * duration_sec)
        pad_len = target_len - sync_frames.shape[0]
        if pad_len > 0:
            sync_frames = torch.cat([sync_frames, sync_frames[-1:].repeat(pad_len, 1, 1, 1)], dim=0)

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
        log.warning(f'Padding to {duration_sec:.2f} sec')
        target_len = int(_CLIP_FPS * duration_sec)
        pad_len = target_len - clip_frames.shape[0]
        if pad_len > 0:
            clip_frames = torch.cat([clip_frames, clip_frames[-1:].repeat(pad_len, 1, 1, 1)], dim=0)

    if sync_length_sec < duration_sec:
        log.warning(f'Sync video is too short: {sync_length_sec:.2f} < {duration_sec:.2f}')
        log.warning(f'Padding to {duration_sec:.2f} sec')
        target_len = int(_SYNC_FPS * duration_sec)
        pad_len = target_len - sync_frames.shape[0]
        if pad_len > 0:
            sync_frames = torch.cat([sync_frames, sync_frames[-1:].repeat(pad_len, 1, 1, 1)], dim=0)

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