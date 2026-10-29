# Copyright 2024 Sony Research Inc.
# Modifications Copyright 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import logging
import os
from pathlib import Path

import hydra
import torch
import torch.distributed as distributed
import torchaudio
from omegaconf import DictConfig
from tqdm import tqdm
import random

from vividaudio.eval_utils import ModelConfig, all_model_cfg, generate_rag
from vividaudio.model.flow_matching import FlowMatching
from vividaudio.model.local_rag_router import (
    LocalRagRouterConfig,
    wrap_with_local_rag_router,
)
from vividaudio.model.networks import MMAudio, get_my_mmaudio
from vividaudio.model.utils.features_utils import FeaturesUtils
from vividaudio.timbre_extractor import TimbreExtractor

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

local_rank = int(os.environ.get('LOCAL_RANK', 0))
world_size = int(os.environ.get('WORLD_SIZE', 1))
log = logging.getLogger()


import json
import math
import torch.nn.functional as F
import torchaudio.functional as Fa

def match_audio_segment_length(ref_audio_segment: torch.Tensor, seg_length: int, mode: str) -> torch.Tensor:
    current_length = ref_audio_segment.shape[1]
    if current_length == seg_length:
        return ref_audio_segment

    if current_length <= 0 or seg_length <= 0:
        return torch.zeros((1, max(seg_length, 0)), dtype=ref_audio_segment.dtype, device=ref_audio_segment.device)

    if current_length > seg_length:
        return ref_audio_segment[:, :seg_length]

    if mode == 'repeat':
        repeat_factor = math.ceil(seg_length / current_length)
        tiled = ref_audio_segment.repeat(1, repeat_factor)
        return tiled[:, :seg_length]

    if mode == 'silence_pad':
        pad_width = seg_length - current_length
        return F.pad(ref_audio_segment, (0, pad_width))

    return F.interpolate(
        ref_audio_segment.unsqueeze(0),
        size=seg_length,
        mode='linear',
        align_corners=False
    ).squeeze(0)


def create_composite_audio(ref_audio_list, seg_in_ref_audio, seg_in_gen_audio, target_duration=8.0, sample_rate=16000, short_ref_mode='stretch'):
    total_samples = int(target_duration * sample_rate)
    composite_audio = torch.zeros((1, total_samples), dtype=torch.float32)

    for audio_idx, (audio_path, ref_seg, gen_seg) in enumerate(zip(ref_audio_list, seg_in_ref_audio, seg_in_gen_audio)):
        if not os.path.exists(audio_path):
            continue
        waveform, sr = torchaudio.load(audio_path)
        waveform = waveform.mean(dim=0, keepdim=True) if waveform.size(0) > 1 else waveform
        if sr != sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, sample_rate)

        ref_start_sample = int(ref_seg[0] * sample_rate)
        ref_end_sample = int(ref_seg[1] * sample_rate)
        ref_start_sample = max(0, ref_start_sample)
        ref_end_sample = min(waveform.shape[1], ref_end_sample)

        if ref_start_sample >= ref_end_sample:
            continue

        ref_audio_segment = waveform[:, ref_start_sample:ref_end_sample]

        gen_start_sample = int(gen_seg[0] * sample_rate)
        gen_end_sample = int(gen_seg[1] * sample_rate)
        gen_start_sample = max(0, gen_start_sample)
        gen_end_sample = min(total_samples, gen_end_sample)

        if gen_start_sample >= gen_end_sample:
            continue

        seg_length = gen_end_sample - gen_start_sample
        if ref_audio_segment.shape[1] != seg_length:
            ref_audio_segment = match_audio_segment_length(ref_audio_segment, seg_length, short_ref_mode)

        composite_audio[:, gen_start_sample:gen_end_sample] = ref_audio_segment

    return composite_audio


# --- Per-event grag (B口径) helpers -------------------------------------------
# For each event k we build a stand-alone 8 s canvas that contains ONLY the
# ref clip of event k (other slots stay zeroed out). Compared to the training
# canvas — where multiple refs coexist — this feeds the timbre encoder with a
# clean, single-event style so downstream AdaLN can adapt each latent slot
# independently.
#
# Inputs (per case):
#   ref_paths, ref_segs, gen_segs  : parallel lists over K events
# Outputs:
#   per_event_canvases : list of Tensor (1, target_samples) length K
#   slot_ranges        : list of (gen_start_sec, gen_end_sec) length K, used to
#                        build the soft slot mask later.
def build_per_event_canvases(
    ref_paths, ref_segs, gen_segs,
    target_duration=8.0, sample_rate=16000, short_ref_mode='stretch',
):
    canvases = []
    slot_ranges = []
    for k, (ap, rs, gs) in enumerate(zip(ref_paths, ref_segs, gen_segs)):
        canvas = create_composite_audio(
            [ap], [rs], [gs],
            target_duration=target_duration,
            sample_rate=sample_rate,
            short_ref_mode=short_ref_mode,
        )
        canvases.append(canvas)
        slot_ranges.append((float(gs[0]), float(gs[1])))
    return canvases, slot_ranges


def build_slot_mask(
    slot_ranges_list, N_latent, target_duration=8.0,
    crossfade_frac=0.05, gap_style='mean',
    device='cpu', dtype=torch.float32,
):
    """Build a (B, K, N) soft slot mask.

    slot_ranges_list : list of length B; each element is list of K (start, end)
                       in seconds. Different B entries may have different K, so
                       we right-pad with zero-mask events (they get zero style
                       contribution).
    crossfade_frac   : fraction of a slot length used for raised-cosine edges.
    gap_style        : how to fill regions where no slot covers this token.
                       'mean' -> renormalize masks to sum to 1 per token (in
                                 the gap this yields a uniform K-mixture, i.e.
                                 the classic mean-over-events style).
                       'zero' -> leave gap as no style (nograg-style).
    """
    B = len(slot_ranges_list)
    K_max = max(len(r) for r in slot_ranges_list) if slot_ranges_list else 0
    mask = torch.zeros(B, K_max, N_latent, dtype=dtype, device=device)
    if K_max == 0 or N_latent <= 0:
        return mask
    step_per_sec = N_latent / float(target_duration)
    t_axis = torch.arange(N_latent, device=device, dtype=dtype)  # (N,)
    for b, ranges in enumerate(slot_ranges_list):
        for k, (s_sec, e_sec) in enumerate(ranges):
            if e_sec <= s_sec:
                continue
            s_idx = s_sec * step_per_sec
            e_idx = e_sec * step_per_sec
            width = max(1.0, (e_idx - s_idx) * crossfade_frac)
            # smoothstep on both edges
            left = torch.clamp((t_axis - s_idx + width) / (2.0 * width), 0.0, 1.0)
            right = torch.clamp((e_idx + width - t_axis) / (2.0 * width), 0.0, 1.0)
            m = torch.minimum(left, right)
            # cos-shaped fade
            m = 0.5 - 0.5 * torch.cos(m * torch.pi)
            mask[b, k] = m
    if gap_style == 'mean':
        # In gap tokens the column sum can be tiny; the classic mean-over-K style is
        # achieved by uniformly filling those columns with 1/K before renorm.
        gap = (mask.sum(dim=1, keepdim=True) < 1e-4).to(dtype)  # (B, 1, N)
        uniform = gap * (1.0 / max(1, K_max))
        mask = mask + uniform  # gap tokens now uniform-K
        mask = mask / mask.sum(dim=1, keepdim=True).clamp_min(1e-6)
    elif gap_style == 'zero':
        # No renormalization; slots partially cover, gap tokens stay near 0.
        pass
    return mask


# --- Plan A: degraded RAG as CFG negative ------------------------------------
# 在 rag_composite 波形上做 noise / lowpass / reverb 退化, 用退化后的 timbre 特征
# 当作 negative_rag_audio_emb. 默认关闭 (cfg_neg_source='zero'), 与旧行为一致.

def _batch_active_mask(wave: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return wave.abs() > eps


def _perturb_anchor_latent(masked_x1: torch.Tensor, mask: torch.Tensor,
                           mode: str, strength: float,
                           seed: int = 0) -> torch.Tensor:
    """Apply a content-ablating perturbation to the anchor latent WITHIN the
    slot region (mask==1). All modes are training-free: they mutate ``masked_x1``
    just before it is fed to ``generate_rag``, so timbre-adjacent statistics
    survive while temporally-localized content cues are damaged.

    Modes
    -----
    - ``none``      : return input unchanged (default; must be a no-op).
    - ``tshuffle``  : within each contiguous slot on the time axis, permute
                      the anchor's frames. Keeps per-channel distribution,
                      destroys onset/rhythm.
    - ``tile``      : within each slot, tile a random short window of length
                      ``max(1, round(slot_len * strength))`` to cover the slot.
                      strength=0.25 → 4x repetition; strength=1.0 == identity.
    - ``awgn``      : add Gaussian noise whose std = strength * per-channel
                      std of the slot content. Preserves spectral shape,
                      damages phase / transient.
    - ``lowpass``   : temporal moving-average smoothing with kernel size
                      max(1, round(1 + strength * (slot_len - 1))). strength=0
                      → identity; strength=1 → full-slot average per channel.

    ``mask`` is (B, N, 1) with 1.0 marking anchor slots. Gap regions are always
    left untouched, matching how ``inpainting_mlp`` reads them.
    """
    if masked_x1 is None or mode is None:
        return masked_x1
    mode = str(mode).lower()
    if mode in ('none', '', 'off'):
        return masked_x1
    if strength <= 0.0 and mode != 'tshuffle':
        # tshuffle is deterministic-in-strength; others degenerate to identity.
        return masked_x1

    B, N, D = masked_x1.shape
    gen = torch.Generator(device='cpu')
    gen.manual_seed(int(seed))
    out = masked_x1.clone()
    mask_1d = (mask[..., 0] > 0.5)  # (B, N) bool

    for b in range(B):
        # Split slots into contiguous [start, end) runs on the time axis.
        m = mask_1d[b]
        if not m.any():
            continue
        idxs = torch.nonzero(m, as_tuple=False).squeeze(-1).tolist()
        runs = []
        cur_start = idxs[0]
        prev = idxs[0]
        for j in idxs[1:]:
            if j == prev + 1:
                prev = j
            else:
                runs.append((cur_start, prev + 1))
                cur_start = j
                prev = j
        runs.append((cur_start, prev + 1))

        for (s, e) in runs:
            L = e - s
            if L <= 0:
                continue
            seg = out[b, s:e, :]  # (L, D)
            if mode == 'tshuffle':
                perm = torch.randperm(L, generator=gen).to(seg.device)
                out[b, s:e, :] = seg[perm]
            elif mode == 'tile':
                win = max(1, int(round(L * max(1e-6, min(1.0, strength)))))
                if win >= L:
                    continue
                start = int(torch.randint(0, L - win + 1, (1,), generator=gen).item())
                block = seg[start:start + win, :]  # (win, D)
                reps = (L + win - 1) // win
                tiled = block.repeat(reps, 1)[:L, :]
                out[b, s:e, :] = tiled
            elif mode == 'awgn':
                # per-channel std over the slot; guard against all-zero slots.
                std = seg.float().std(dim=0, keepdim=True).clamp_min(1e-6)
                noise = torch.randn(seg.shape, generator=gen).to(seg.device, seg.dtype)
                out[b, s:e, :] = seg + (strength * std).to(seg.dtype) * noise
            elif mode == 'lowpass':
                k = max(1, int(round(1 + strength * (L - 1))))
                if k <= 1:
                    continue
                if k >= L:
                    avg = seg.mean(dim=0, keepdim=True)
                    out[b, s:e, :] = avg.expand_as(seg)
                    continue
                # 1-D moving average along the time axis, channel-wise.
                x = seg.transpose(0, 1).unsqueeze(0)  # (1, D, L)
                weight = torch.ones(D, 1, k, device=x.device, dtype=x.dtype) / float(k)
                pad = k // 2
                smoothed = torch.nn.functional.conv1d(
                    torch.nn.functional.pad(x, (pad, pad), mode='replicate'),
                    weight, groups=D,
                )
                out[b, s:e, :] = smoothed.squeeze(0).transpose(0, 1)[:L]
            else:
                # Unknown mode -> no-op; caller is responsible for validation.
                pass
    return out


def _bottleneck_anchor_latent(masked_x1: torch.Tensor,
                              mask: torch.Tensor,
                              gen_seg_lists,
                              target_duration: float,
                              mode: str,
                              mix: float = 1.0) -> torch.Tensor:
    """Collapse the inpainting anchor into a lower-bandwidth style signal.

    This is a runtime-only inference knob: it never touches the trained
    weights and defaults to a strict no-op.

    Modes
    -----
    - ``none``        : return input unchanged.
    - ``global_mean`` : pool all mask==1 tokens into a single style vector and
                        broadcast it back to the masked region.
    - ``slot_mean``   : pool each generation slot separately and broadcast the
                        per-slot mean inside that slot. This keeps coarse
                        per-event timbre while removing within-slot temporal
                        details that are easy to copy.

    ``mix`` interpolates between the original anchor and the bottlenecked one:
        out = (1 - mix) * original + mix * bottlenecked
    so mix=1.0 is a full bottleneck, mix=0.0 is identity.
    """
    if masked_x1 is None or mode is None:
        return masked_x1
    mode = str(mode).lower()
    if mode in ("none", "", "off"):
        return masked_x1
    mix = float(max(0.0, min(1.0, mix)))
    if mix <= 0.0:
        return masked_x1

    out = masked_x1.clone()
    B, N, D = out.shape
    mask_1d = (mask[..., 0] > 0.5)

    if mode == "global_mean":
        for b in range(B):
            idx = torch.nonzero(mask_1d[b], as_tuple=False).squeeze(-1)
            if idx.numel() == 0:
                continue
            pooled = masked_x1[b, idx, :].mean(dim=0, keepdim=True)
            stylized = pooled.expand(idx.numel(), D)
            out[b, idx, :] = (1.0 - mix) * masked_x1[b, idx, :] + mix * stylized
        return out

    if mode == "slot_mean":
        for b in range(B):
            segs = gen_seg_lists[b] if gen_seg_lists is not None and b < len(gen_seg_lists) else []
            if not segs:
                idx = torch.nonzero(mask_1d[b], as_tuple=False).squeeze(-1)
                if idx.numel() == 0:
                    continue
                pooled = masked_x1[b, idx, :].mean(dim=0, keepdim=True)
                stylized = pooled.expand(idx.numel(), D)
                out[b, idx, :] = (1.0 - mix) * masked_x1[b, idx, :] + mix * stylized
                continue
            for seg in segs:
                if seg is None or len(seg) < 2:
                    continue
                s_sec, e_sec = float(seg[0]), float(seg[1])
                s_tok = max(0, min(N, int((s_sec / float(target_duration)) * N)))
                e_tok = max(0, min(N, int((e_sec / float(target_duration)) * N)))
                if s_tok >= e_tok:
                    continue
                idx = torch.nonzero(mask_1d[b, s_tok:e_tok], as_tuple=False).squeeze(-1)
                if idx.numel() == 0:
                    continue
                idx = idx + s_tok
                pooled = masked_x1[b, idx, :].mean(dim=0, keepdim=True)
                stylized = pooled.expand(idx.numel(), D)
                out[b, idx, :] = (1.0 - mix) * masked_x1[b, idx, :] + mix * stylized
        return out

    return masked_x1


def _parse_channel_beta_map(raw, latent_dim: int, default_beta: float) -> list[float] | None:
    """Parse a comma-separated per-channel β list. ``latent_dim`` is the audio
    latent channel count (D). Extra entries are truncated; missing entries are
    padded with ``default_beta`` so the caller always gets a length-D list.
    Returns None when the raw value is empty / None so downstream code can
    skip the branch entirely."""
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)):
        parts = [float(x) for x in raw]
    else:
        s = str(raw).strip()
        if not s:
            return None
        parts = [float(x) for x in s.split(',') if x.strip()]
    if not parts:
        return None
    if len(parts) < latent_dim:
        parts = parts + [float(default_beta)] * (latent_dim - len(parts))
    else:
        parts = parts[:latent_dim]
    return parts


def degrade_rag_composite(wave: torch.Tensor, sample_rate: int,
                          noise_snr_db: float = 0.0,
                          lowpass_cutoff_hz: float = 0.0,
                          reverb_rt60_sec: float = 0.0,
                          reverb_wet: float = 0.3) -> torch.Tensor:
    """Apply noise / lowpass / reverb only on active regions of ``wave``.
    ``wave`` is (B, T) or (1, T); returns same shape."""
    out = wave.clone()
    active = _batch_active_mask(out)
    if not active.any():
        return out

    if noise_snr_db and noise_snr_db > 0:
        power = (out.pow(2) * active.float()).sum(dim=-1, keepdim=True) / active.float().sum(dim=-1, keepdim=True).clamp_min(1.0)
        noise_power = power / (10.0 ** (noise_snr_db / 10.0))
        noise = torch.randn_like(out) * noise_power.sqrt()
        out = out + noise * active.float()

    if lowpass_cutoff_hz and lowpass_cutoff_hz > 0:
        rc = 1.0 / (2.0 * math.pi * lowpass_cutoff_hz)
        dt = 1.0 / sample_rate
        alpha = dt / (rc + dt)
        try:
            a = torch.tensor([1.0, -(1.0 - alpha)], device=out.device, dtype=out.dtype)
            b = torch.tensor([alpha, 0.0], device=out.device, dtype=out.dtype)
            filt = Fa.lfilter(out, a_coeffs=a, b_coeffs=b, clamp=False)
            out = torch.where(active, filt, out)
        except Exception:
            pass

    if reverb_rt60_sec and reverb_rt60_sec > 0:
        ir_len = max(int(sample_rate * reverb_rt60_sec), 32)
        t = torch.arange(ir_len, device=out.device, dtype=out.dtype)
        decay = torch.exp(-6.9078 * t / max(ir_len - 1, 1))
        ir = torch.randn(ir_len, device=out.device, dtype=out.dtype) * decay
        ir = ir / ir.abs().max().clamp_min(1e-6)
        wet = F.conv1d(out.unsqueeze(1), ir.view(1, 1, -1).flip(-1), padding=ir_len - 1)
        wet = wet.squeeze(1)[:, :out.shape[-1]]
        mixed = (1.0 - reverb_wet) * out + reverb_wet * wet
        out = torch.where(active, mixed, out)

    return out


class RAGEvalDataset(torch.utils.data.Dataset):
    def __init__(self, jsonl_path: str, duration_sec: float = 8.0):
        self.data = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))
        self.duration_sec = duration_sec

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        item = self.data[idx]
        video_path = item.get('video_path', None)

        clip_chunk, sync_chunk = None, None
        if video_path is not None and os.path.exists(video_path):
            from vividaudio.eval_utils import load_video
            video_info = load_video(Path(video_path), self.duration_sec)
            clip_chunk = video_info.clip_frames
            sync_chunk = video_info.sync_frames

        return {
            'name': item.get('name', f'sample_{idx}'),
            'caption': item.get('prompt', ''),
            'negative_caption': item.get('negative_prompt', ''),
            'clip_video': clip_chunk,
            'sync_video': sync_chunk,
            'rag_audio_list': item.get('rag_audio_list', []),
            'rag_seg_in_ref': item.get('rag_seg_in_ref', []),
            'rag_seg_in_gen': item.get('rag_seg_in_gen', []),
            'neg_audio_list': item.get('neg_audio_list', []),
            'neg_seg_in_ref': item.get('neg_seg_in_ref', []),
            'neg_seg_in_gen': item.get('neg_seg_in_gen', []),
            'inp_audio_list': item.get('inp_audio_list', []),
            'inp_seg_in_ref': item.get('inp_seg_in_ref', []),
            'inp_seg_in_gen': item.get('inp_seg_in_gen', []),
            # path-swap: alternative composite routed only to grag (global)
            'swap_rag_audio_list': item.get('swap_rag_audio_list', []),
            'swap_rag_seg_in_ref': item.get('swap_rag_seg_in_ref', []),
            'swap_rag_seg_in_gen': item.get('swap_rag_seg_in_gen', []),
            # reverse path-swap: alternative composite routed only to rag (local)
            'local_swap_audio_list': item.get('local_swap_audio_list', []),
            'local_swap_seg_in_ref': item.get('local_swap_seg_in_ref', []),
            'local_swap_seg_in_gen': item.get('local_swap_seg_in_gen', []),
            # grag raw slice: load a raw wav slice and route straight to grag,
            # bypassing composite construction. Useful for feeding tail-2s of
            # the 10s reference into the global path.
            'grag_raw_audio_path': item.get('grag_raw_audio_path', None),
            'grag_raw_slice_sec': item.get('grag_raw_slice_sec', None),
            # Original clip used by train_grag_source='tail'. This is kept
            # separate because rag_audio_list may contain retrieved clips.
            'grag_tail_audio_path': item.get(
                'grag_tail_audio_path',
                item.get('source_audio_path', item.get('grag_raw_audio_path', None))
            ),
            'grag_tail_slice_sec': item.get('grag_tail_slice_sec', None),
        }

def custom_collate(batch):
    collated = {}
    for key in batch[0].keys():
        if key in ['clip_video', 'sync_video']:
            if batch[0][key] is not None:
                collated[key] = torch.stack([item[key] for item in batch])
            else:
                collated[key] = None
        else:
            collated[key] = [item[key] for item in batch]
    return collated

@torch.inference_mode()
@hydra.main(version_base='1.3.2', config_path='configs', config_name='eval_config.yaml')
def main(cfg: DictConfig):
    # Allow arbitrary CLI overrides (`+key=value`) without struct errors so
    # this script can be invoked with configs whose schema does not yet
    # declare every eval key (e.g., path-swap orchestration).
    from omegaconf import OmegaConf
    OmegaConf.set_struct(cfg, False)
    device = 'cuda'
    torch.cuda.set_device(local_rank)
    model_inpainting = cfg.get('inpainting', False)
    use_inpainting_input = cfg.get('use_inpainting_input', model_inpainting)

    if cfg.model not in all_model_cfg:
        raise ValueError(f'Unknown model variant: {cfg.model}')
    model: ModelConfig = all_model_cfg[cfg.model]
    # model.download_if_needed()
    seq_cfg = model.seq_cfg

    output_dir = Path(cfg.output_dir)
    output_name = Path(cfg.model_path).stem
    if not hasattr(cfg, "flag"):
        flag = 'base'
    else:
        flag = cfg.flag
    if not cfg.get('dataset_jsonl', ''):
        output_dir = output_dir / f'{cfg.dataset}-{output_name}-{flag}'
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_feat_type = cfg.get('audio_feat_type', 'beats')
    timbre_extractor = TimbreExtractor(model_type=audio_feat_type, device=device)
    rag_dim = timbre_extractor.out_dim
    repa_dim = timbre_extractor.out_dim

    with torch.no_grad():
        dummy_wave = torch.zeros(1, 16000 * 8, device=device)
        dummy_feat = timbre_extractor(dummy_wave)
        native_T = dummy_feat.shape[1]

    fgc_pt = native_T / 8.0
    time_per_step = 8.0 / native_T
    total_seq_len = int(cfg.duration_s / time_per_step)
    log.info(f"Auto-configured rag_dim={rag_dim}, repa_dim={repa_dim}, fgc_pt={fgc_pt}, native_T={native_T}")

    # load a pretrained model
    seq_cfg.duration = cfg.duration_s
    net: MMAudio = get_my_mmaudio(cfg.model,
                                  sep_fgc=cfg.get('sep_fgc', False),
                                  joint_fgc=cfg.get('joint_fgc', False),
                                  delay_fusion=cfg.get('delay_fusion', False),
                                  fusion_fgc_clip=cfg.get('fusion_fgc_clip', False),
                                  fgc_pt=fgc_pt,
                                  interleaved_rope=cfg.get('interleaved_rope', False),
                                  sync_rope=cfg.get('sync_rope', False),
                                  cross_attn_delay=cfg.get('cross_attn_delay', False),
                                  local_window_size=cfg.get('local_window_size', 0),
                                  hybrid_cross_attn=cfg.get('hybrid_cross_attn', False),
                                  zero_init_gate=cfg.get('zero_init_gate', True),
                                  mask_rag_ratio=cfg.get('mask_rag_ratio', 0.0),
                                  mask_rag_block_size=cfg.get('mask_rag_block_size', 10),
                                  global_rag_cond=cfg.get('global_rag_cond', False),
                                  global_emb_type=cfg.get('global_emb_type', 'beats'),
                                  rag_dim=rag_dim,
                                  repa_dim=repa_dim,
                                  inpainting=model_inpainting,
                                  bypass_empty_inpainting=cfg.get('bypass_empty_inpainting', False),
                                  use_decoupled_grag=cfg.get('use_decoupled_grag', False),
                                  grag_extractor_type=cfg.get('grag_extractor_type', 'mean'),
                                  grag_qformer_n_q=cfg.get('grag_qformer_n_q', 8),
                                  grag_qformer_n_layers=cfg.get('grag_qformer_n_layers', 2)).to(device).eval()
    weights = torch.load(cfg.model_path, map_location=device, weights_only=True)
    if 'weights' in weights:
        net.load_weights(weights['weights'])
    else:
        net.load_weights(weights)
    # net.load_weights(torch.load(model.model_path, map_location=device, weights_only=True))
    log.info(f'Loaded weights from {cfg.model_path}')

    router_mode = bool(cfg.get('router_mode', False))
    if router_mode:
        if not cfg.get('delay_fusion', False):
            raise ValueError('router_mode requires delay_fusion=True')
        if not cfg.get('cross_attn_delay', False):
            raise ValueError('router_mode requires cross_attn_delay=True')
        if not cfg.get('sync_rope', False):
            raise ValueError('router_mode requires sync_rope=True')
        net = wrap_with_local_rag_router(
            net,
            cfg=LocalRagRouterConfig(
                enabled=True,
                alpha=float(cfg.get('router_alpha', 1.0)),
                start_joint_block=int(cfg.get('router_start_block', 0)),
                end_joint_block=int(cfg.get('router_end_block', -1)),
            ),
        )
        log.info(
            '[router] enabled: alpha=%s blocks=[%s, %s)',
            cfg.get('router_alpha', 1.0),
            cfg.get('router_start_block', 0),
            cfg.get('router_end_block', -1),
        )
    net.update_seq_lengths(seq_cfg.latent_seq_len, seq_cfg.clip_seq_len, seq_cfg.sync_seq_len)
    # Optional inference-time scaling of the grag global condition. 1.0 keeps
    # trained behaviour; smaller values dial down grag's influence on AdaLN.
    _grag_scale = float(cfg.get('grag_scale', 1.0))
    net.grag_scale = _grag_scale
    log.info(f'[grag] grag_scale={_grag_scale}')
    # Optional inference-time scaling of the inpainting residual (beta). 1.0
    # keeps trained behaviour; 0.0 disables inpainting anchor entirely.
    _inpainting_scale = float(cfg.get('inpainting_scale', 1.0))
    net.inpainting_scale = _inpainting_scale
    log.info(f'[inpainting] inpainting_scale={_inpainting_scale}')
    # -- Channel-wise β (thought-2) -----------------------------------------
    # Optional per-channel scaling of the inpainting residual. Accepts a
    # comma-separated float list via ``inpainting_channel_scale``; length is
    # padded / truncated to the residual hidden dim (the output width of
    # ``inpainting_mlp``), not ``net.latent_dim``. None / empty => residual
    # keeps its trained per-channel gain (identity multiplier). Never touches
    # non-inpainting sampling paths (predict_flow's ``self.inpainting=False``
    # branch).
    _chan_scale_raw = cfg.get('inpainting_channel_scale', None)
    _chan_scale_default = float(cfg.get('inpainting_channel_scale_default', _inpainting_scale))
    _chan_scale_dim = net.latent_dim
    if getattr(net, 'inpainting', False) and getattr(net, 'inpainting_mlp', None) is not None:
        _last_inp_layer = net.inpainting_mlp[-1]
        _chan_scale_dim = int(getattr(_last_inp_layer, 'out_features', _chan_scale_dim))
    _chan_scale_list = _parse_channel_beta_map(_chan_scale_raw, _chan_scale_dim, _chan_scale_default)
    if _chan_scale_list is not None:
        net.inpainting_channel_scale = torch.tensor(
            _chan_scale_list, device=device, dtype=torch.float32,
        )
        # For long lists log only the summary to keep stdout readable.
        _cs_preview = ', '.join(f'{v:.3f}' for v in _chan_scale_list[:8])
        _cs_more = '' if len(_chan_scale_list) <= 8 else f', ...(+{len(_chan_scale_list) - 8})'
        log.info(
            f"[inpainting] channel-wise β active: D={_chan_scale_dim} "
            f"default={_chan_scale_default} preview=[{_cs_preview}{_cs_more}]"
        )
    else:
        net.inpainting_channel_scale = None
    # Optional t-scheduled anchor gate (F1). Modes:
    #   'none'       -> no schedule (default trained behaviour)
    #   'early_only' -> gate = 1 when t > cutoff else 0 (hard step)
    #   'linear'     -> gate = clip((t - t_off) / (t_on - t_off), 0, 1)
    #                   ramps 0 -> 1 as t goes from t_off to t_on (t=1 is noise)
    #   'cosine'     -> smoothstep-shaped variant of linear
    # ``t`` here is the flow-matching time from generate_rag / fm; convention
    # in this repo is t=1 for pure noise, t=0 for clean sample.
    _inp_schedule = str(cfg.get('inpainting_schedule', 'none')).lower()
    _inp_sched_t_on = float(cfg.get('inpainting_schedule_t_on', 0.75))
    _inp_sched_t_off = float(cfg.get('inpainting_schedule_t_off', 0.25))
    _inp_sched_low = float(cfg.get('inpainting_schedule_low', 0.0))
    _inp_sched_high = float(cfg.get('inpainting_schedule_high', 1.0))
    # step-mode extras: piecewise-constant β with a single switch time.
    # t >= t_switch (noise-side, early sampling steps) -> β = before
    # t <  t_switch (clean-side, late sampling steps)  -> β = after
    _inp_sched_t_switch = float(cfg.get('inpainting_schedule_t_switch', 0.5))
    _inp_sched_before = float(cfg.get('inpainting_schedule_before', 1.0))
    _inp_sched_after = float(cfg.get('inpainting_schedule_after', 0.0))
    _inp_style_bottleneck = str(cfg.get('inpainting_style_bottleneck', 'none')).lower()
    _inp_style_mix = float(cfg.get('inpainting_style_mix', 1.0))

    def _make_inpainting_scale_fn(mode, t_on, t_off, low, high,
                                  t_switch, before, after):
        if mode in ('none', '', 'off'):
            return None
        # normalize order: t_on is the noise side (usually > t_off)
        if t_on < t_off:
            t_on, t_off = t_off, t_on

        def _fn(t):
            # t is a torch tensor (B,) or scalar; normalize to (B, 1, 1)
            if not isinstance(t, torch.Tensor):
                t_val = torch.tensor(float(t))
            else:
                t_val = t.float()
            if mode == 'step':
                # piecewise-constant β: t>=t_switch -> before, else after
                return torch.where(t_val >= t_switch,
                                   torch.full_like(t_val, float(before)),
                                   torch.full_like(t_val, float(after)))
            if mode == 'early_only':
                cutoff = 0.5 * (t_on + t_off)
                gate = (t_val >= cutoff).float()
            elif mode == 'late_only':
                cutoff = 0.5 * (t_on + t_off)
                gate = (t_val <= cutoff).float()
            elif mode == 'linear':
                if t_on <= t_off:
                    gate = torch.ones_like(t_val)
                else:
                    gate = ((t_val - t_off) / (t_on - t_off)).clamp(0.0, 1.0)
            elif mode == 'cosine':
                if t_on <= t_off:
                    gate = torch.ones_like(t_val)
                else:
                    x = ((t_val - t_off) / (t_on - t_off)).clamp(0.0, 1.0)
                    gate = 0.5 - 0.5 * torch.cos(x * torch.pi)
            else:
                return None
            return low + (high - low) * gate

        return _fn

    net.inpainting_scale_fn = _make_inpainting_scale_fn(
        _inp_schedule, _inp_sched_t_on, _inp_sched_t_off,
        _inp_sched_low, _inp_sched_high,
        _inp_sched_t_switch, _inp_sched_before, _inp_sched_after,
    )
    if net.inpainting_scale_fn is not None:
        if _inp_schedule == 'step':
            log.info(
                f"[inpainting] schedule=step t_switch={_inp_sched_t_switch} "
                f"before(t>=t_switch)={_inp_sched_before} "
                f"after(t<t_switch)={_inp_sched_after}"
            )
        else:
            log.info(
                f"[inpainting] schedule={_inp_schedule} t_on={_inp_sched_t_on} "
                f"t_off={_inp_sched_t_off} low={_inp_sched_low} high={_inp_sched_high}"
            )
    if _inp_style_bottleneck not in ('none', '', 'off'):
        log.info(
            f"[inpainting] style bottleneck active: mode={_inp_style_bottleneck} "
            f"mix={_inp_style_mix}"
        )
    log.info(f'Latent seq len: {seq_cfg.latent_seq_len}')
    log.info(f'Clip seq len: {seq_cfg.clip_seq_len}')
    log.info(f'Sync seq len: {seq_cfg.sync_seq_len}')
    random_seed = cfg.get("random_seed", False)

    # misc setup
    rng = torch.Generator(device=device)
    rng.manual_seed(cfg.seed)
    fm = FlowMatching(cfg.sampling.min_sigma,
                      inference_mode=cfg.sampling.method,
                      num_steps=cfg.sampling.num_steps)

    need_vae_encoder = use_inpainting_input or cfg.get('global_emb_type', 'beats') in ['vae', 'both']
    feature_utils = FeaturesUtils(tod_vae_ckpt=model.vae_path,
                                  synchformer_ckpt=model.synchformer_ckpt,
                                  enable_conditions=True,
                                  mode=model.mode,
                                  bigvgan_vocoder_ckpt=model.bigvgan_16k_path,
                                  need_vae_encoder=need_vae_encoder)
    feature_utils = feature_utils.to(device).eval()

    audio_feat_type = cfg.get('audio_feat_type', 'beats')  # demo里用 args.audio_feat_type

    if cfg.compile:
        net.preprocess_conditions = torch.compile(net.preprocess_conditions)
        net.predict_flow = torch.compile(net.predict_flow)
        feature_utils.compile()

    dataset_jsonl = cfg.get('dataset_jsonl', '')
    if not dataset_jsonl or not os.path.isfile(dataset_jsonl):
        raise ValueError(f'dataset_jsonl does not exist: {dataset_jsonl!r}')
    dataset = RAGEvalDataset(dataset_jsonl, duration_sec=cfg.duration_s)
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=local_rank,
        shuffle=False,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        sampler=sampler,
        collate_fn=custom_collate,
    )

    dtype = torch.bfloat16 if cfg.amp else torch.float32
    log.info(f'model_inpainting={model_inpainting}, use_inpainting_input={use_inpainting_input}')

    _cfg_neg_source = str(cfg.get('cfg_neg_source', 'zero')).lower()
    _cfg_t_lo = float(cfg.get('cfg_t_lo', 0.0))
    _cfg_t_hi = float(cfg.get('cfg_t_hi', 1.0))
    _cfg_rescale_std = float(cfg.get('cfg_rescale_std', 0.0))
    _short_ref_mode = str(cfg.get('short_ref_mode', 'stretch')).lower()
    _train_grag_source = str(cfg.get('train_grag_source', 'shared')).lower()
    if _train_grag_source not in ('shared', 'tail', 'mix'):
        raise ValueError(
            "train_grag_source must be one of 'shared', 'tail', or 'mix', "
            f"got {_train_grag_source!r}"
        )
    _train_grag_source_prob_tail = float(
        cfg.get('train_grag_source_prob_tail', 0.5)
    )
    if not 0.0 <= _train_grag_source_prob_tail <= 1.0:
        raise ValueError('train_grag_source_prob_tail must be in [0, 1]')
    _grag_tail_start_sec = cfg.get('grag_tail_start_sec', None)
    _grag_tail_start_sec = (
        float(cfg.duration_s)
        if _grag_tail_start_sec is None
        else float(_grag_tail_start_sec)
    )
    _grag_tail_duration_sec = float(cfg.get('grag_tail_duration_sec', 2.0))
    if _grag_tail_duration_sec <= 0:
        raise ValueError('grag_tail_duration_sec must be positive')
    if _train_grag_source != 'shared' and not cfg.get('use_decoupled_grag', False):
        log.warning(
            "train_grag_source=%s has no effect because use_decoupled_grag=False",
            _train_grag_source,
        )
    _plan_a_on = (_cfg_neg_source != 'zero')
    _plan_b_on = (_cfg_t_lo > 0.0 or _cfg_t_hi < 1.0 or _cfg_rescale_std > 0.0)
    log.info(f"[ref] short_ref_mode={_short_ref_mode}")
    if _plan_a_on or _plan_b_on:
        log.info(
            f"[CFG-opt] plan_a={_plan_a_on} (source={_cfg_neg_source}, "
            f"noise_snr={cfg.get('cfg_neg_noise_snr_db', 0.0)}, "
            f"lp={cfg.get('cfg_neg_lowpass_cutoff_hz', 0.0)}, "
            f"rt60={cfg.get('cfg_neg_reverb_rt60_sec', 0.0)}), "
            f"plan_b={_plan_b_on} (t_window=[{_cfg_t_lo},{_cfg_t_hi}], "
            f"rescale_std={_cfg_rescale_std})"
        )
    else:
        log.info("[CFG-opt] plan_a=OFF, plan_b=OFF (fully reverted to baseline CFG)")
    log.info(
        "[grag] train_grag_source=%s, tail=[%.3f, %.3f]s, p_tail=%.3f",
        _train_grag_source,
        _grag_tail_start_sec,
        _grag_tail_start_sec + _grag_tail_duration_sec,
        _train_grag_source_prob_tail,
    )

    with torch.amp.autocast(enabled=cfg.amp, dtype=torch.bfloat16, device_type=device):
        for batch in tqdm(loader):
            if random_seed:
                seed = random.randint(0, 2**32 - 1)
                rng.manual_seed(seed)


            # Use 'name' if available, otherwise try to infer batch size
            bs = len(batch.get('name', batch.get('caption', [0])))

            # --- RAG AUDIO ---
            rag_composite_list = []
            neg_composite_list = []
            for i in range(bs):
                comp = create_composite_audio(
                    batch.get('rag_audio_list', [[]]*bs)[i],
                    batch.get('rag_seg_in_ref', [[]]*bs)[i],
                    batch.get('rag_seg_in_gen', [[]]*bs)[i],
                    target_duration=cfg.duration_s,
                    sample_rate=16000,
                    short_ref_mode=_short_ref_mode,
                )
                rag_composite_list.append(comp)

                ncomp = create_composite_audio(
                    batch.get('neg_audio_list', [[]]*bs)[i],
                    batch.get('neg_seg_in_ref', [[]]*bs)[i],
                    batch.get('neg_seg_in_gen', [[]]*bs)[i],
                    target_duration=cfg.duration_s,
                    sample_rate=16000,
                    short_ref_mode=_short_ref_mode,
                )
                neg_composite_list.append(ncomp)

            rag_composite_audio = torch.cat(rag_composite_list, dim=0).to(device)
            negative_rag_audio = torch.cat(neg_composite_list, dim=0).to(device)

            # --- Path-swap: build an alternative composite that will be routed
            # to the global (grag) path only, keeping rag_composite unchanged. ---
            swap_composite_audio = None
            swap_lists = batch.get("swap_rag_audio_list", [[]] * bs)
            has_swap = any(len(lst) > 0 for lst in swap_lists)
            if has_swap:
                swap_composite_list = []
                for i in range(bs):
                    swap_comp = create_composite_audio(
                        batch.get('swap_rag_audio_list', [[]]*bs)[i],
                        batch.get('swap_rag_seg_in_ref', [[]]*bs)[i],
                        batch.get('swap_rag_seg_in_gen', [[]]*bs)[i],
                        target_duration=cfg.duration_s,
                        sample_rate=16000,
                        short_ref_mode=_short_ref_mode,
                    )
                    swap_composite_list.append(swap_comp)
                swap_composite_audio = torch.cat(swap_composite_list, dim=0).to(device)

            # --- Reverse Path-swap: alternative composite routed to LOCAL (rag)
            # only. Global (grag) keeps rag_composite. Used to test whether
            # local path independently drives event timbre. ---
            local_swap_composite_audio = None
            local_swap_lists = batch.get("local_swap_audio_list", [[]] * bs)
            has_local_swap = any(len(lst) > 0 for lst in local_swap_lists)
            if has_local_swap:
                local_swap_list = []
                for i in range(bs):
                    ls_comp = create_composite_audio(
                        batch.get('local_swap_audio_list', [[]]*bs)[i],
                        batch.get('local_swap_seg_in_ref', [[]]*bs)[i],
                        batch.get('local_swap_seg_in_gen', [[]]*bs)[i],
                        target_duration=cfg.duration_s,
                        sample_rate=16000,
                        short_ref_mode=_short_ref_mode,
                    )
                    local_swap_list.append(ls_comp)
                local_swap_composite_audio = torch.cat(local_swap_list, dim=0).to(device)

            # --- Optional raw source for the global (grag) path. ---
            # In tail mode this mirrors train_group.py: use audio after the
            # target window rather than the RAG composite.
            grag_raw_audio = None
            if _train_grag_source in ('tail', 'mix'):
                grag_raw_paths = batch.get("grag_tail_audio_path", [None] * bs)
                grag_raw_slices = batch.get("grag_tail_slice_sec", [None] * bs)
            else:
                grag_raw_paths = batch.get("grag_raw_audio_path", [None] * bs)
                grag_raw_slices = batch.get("grag_raw_slice_sec", [None] * bs)
            has_grag_raw = (
                cfg.get('use_decoupled_grag', False)
                and any(p is not None for p in grag_raw_paths)
            )
            if has_grag_raw:
                grag_raw_list = []
                for i in range(bs):
                    p = grag_raw_paths[i]
                    slc = grag_raw_slices[i]
                    if p is None or not os.path.exists(p):
                        # Mixed batches may omit an optional source.
                        grag_raw_list.append(None)
                        continue
                    wav, sr = torchaudio.load(p)
                    if wav.shape[0] > 1:
                        wav = wav.mean(dim=0, keepdim=True)
                    if sr != 16000:
                        wav = torchaudio.functional.resample(wav, sr, 16000)
                    if slc is not None:
                        a, b = slc
                        i0, i1 = int(a * 16000), int(b * 16000)
                        wav = wav[:, i0:i1]
                    elif _train_grag_source in ('tail', 'mix'):
                        i0 = max(0, int(_grag_tail_start_sec * 16000))
                        i1 = max(i0, int(
                            (_grag_tail_start_sec + _grag_tail_duration_sec)
                            * 16000
                        ))
                        wav = wav[:, i0:i1]
                    if _train_grag_source in ('tail', 'mix'):
                        target_samples = int(_grag_tail_duration_sec * 16000)
                        if wav.shape[1] < target_samples:
                            wav = F.pad(wav, (0, target_samples - wav.shape[1]))
                        else:
                            wav = wav[:, :target_samples]
                    grag_raw_list.append(wav)
                grag_raw_audio = [
                    w.to(device) if w is not None else None for w in grag_raw_list
                ]

            # --- Plan A: 用退化的 rag_composite 覆盖 negative_rag_audio ---
            # 默认 cfg_neg_source='zero' → 保持旧行为 (由 neg_audio_list 生成)
            cfg_neg_source = str(cfg.get('cfg_neg_source', 'zero')).lower()
            if cfg_neg_source == 'degrade':
                negative_rag_audio = degrade_rag_composite(
                    rag_composite_audio.clone(),
                    sample_rate=16000,
                    noise_snr_db=float(cfg.get('cfg_neg_noise_snr_db', 0.0)),
                    lowpass_cutoff_hz=float(cfg.get('cfg_neg_lowpass_cutoff_hz', 0.0)),
                    reverb_rt60_sec=float(cfg.get('cfg_neg_reverb_rt60_sec', 0.0)),
                    reverb_wet=float(cfg.get('cfg_neg_reverb_wet', 0.3)),
                )

            # with torch.no_grad():
            #     with torch.amp.autocast(enabled=False, device_type=device):
            #         padding_mask = torch.zeros_like(rag_composite_audio, dtype=torch.bool, device=device)
            #         rag_audio_emb = BEATs_model.extract_features(rag_composite_audio, padding_mask=padding_mask)[0]

            #         neg_padding_mask = torch.zeros_like(negative_rag_audio, dtype=torch.bool, device=device)
            #         negative_rag_audio_emb = BEATs_model.extract_features(negative_rag_audio, padding_mask=neg_padding_mask)[0]
            with torch.no_grad():
                with torch.amp.autocast(enabled=False, device_type=device):
                    rag_audio_emb = timbre_extractor(rag_composite_audio)
                    negative_rag_audio_emb = timbre_extractor(negative_rag_audio)


            rag_audio_emb = rag_audio_emb.to(dtype)
            if native_T > 1 and rag_audio_emb.shape[1] != total_seq_len:
                rag_audio_emb = F.interpolate(rag_audio_emb.permute(0, 2, 1), size=total_seq_len, mode='nearest-exact').permute(0, 2, 1)

            negative_rag_audio_emb = negative_rag_audio_emb.to(dtype)
            if native_T > 1 and negative_rag_audio_emb.shape[1] != total_seq_len:
                negative_rag_audio_emb = F.interpolate(negative_rag_audio_emb.permute(0, 2, 1), size=total_seq_len, mode='nearest-exact').permute(0, 2, 1)
            grag_audio_emb = rag_audio_emb.clone() if cfg.get('use_decoupled_grag', False) else None
            negative_grag_audio_emb = negative_rag_audio_emb.clone() if cfg.get('use_decoupled_grag', False) else None

            # --- Path-swap: if a separate swap composite exists, route its BEATs
            # embedding to the global (grag) path only. Local path (rag) stays.
            if grag_audio_emb is not None and swap_composite_audio is not None:
                with torch.no_grad():
                    with torch.amp.autocast(enabled=False, device_type=device):
                        swap_emb = timbre_extractor(swap_composite_audio)
                swap_emb = swap_emb.to(dtype)
                if native_T > 1 and swap_emb.shape[1] != total_seq_len:
                    swap_emb = F.interpolate(
                        swap_emb.permute(0, 2, 1),
                        size=total_seq_len, mode='nearest-exact',
                    ).permute(0, 2, 1)
                grag_audio_emb = swap_emb
                log.info(
                    f"[path-swap] grag_audio_emb <- swap composite; "
                    f"local unchanged. shapes rag={rag_audio_emb.shape} "
                    f"grag={grag_audio_emb.shape}"
                )

            # --- Grag raw slice: bypass composite entirely for global path.
            # Per-sample lengths may differ; compute embeddings one at a time
            # then stack after interpolating to total_seq_len.
            if grag_audio_emb is not None and grag_raw_audio is not None:
                emb_list = []
                for i, w in enumerate(grag_raw_audio):
                    if w is None or (
                        _train_grag_source == 'mix'
                        and random.random() >= _train_grag_source_prob_tail
                    ):
                        emb_list.append(grag_audio_emb[i:i + 1])
                        continue
                    with torch.no_grad():
                        with torch.amp.autocast(enabled=False, device_type=device):
                            e = timbre_extractor(w)
                    e = e.to(dtype)
                    if native_T > 1 and e.shape[1] != total_seq_len:
                        e = F.interpolate(
                            e.permute(0, 2, 1),
                            size=total_seq_len, mode='nearest-exact',
                        ).permute(0, 2, 1)
                    emb_list.append(e)
                grag_audio_emb = torch.cat(emb_list, dim=0)
                log.info(
                    f"[grag-{_train_grag_source}] global embedding source applied; "
                    f"local unchanged. shapes rag={rag_audio_emb.shape} "
                    f"grag={grag_audio_emb.shape}"
                )

            # --- Reverse path-swap: route local_swap composite to rag (local).
            # Global (grag) keeps the original rag_composite. Ensures local
            # path is fed the swap while global sees the anchor's own audio.
            if local_swap_composite_audio is not None:
                with torch.no_grad():
                    with torch.amp.autocast(enabled=False, device_type=device):
                        ls_emb = timbre_extractor(local_swap_composite_audio)
                ls_emb = ls_emb.to(dtype)
                if native_T > 1 and ls_emb.shape[1] != total_seq_len:
                    ls_emb = F.interpolate(
                        ls_emb.permute(0, 2, 1),
                        size=total_seq_len, mode='nearest-exact',
                    ).permute(0, 2, 1)
                rag_audio_emb = ls_emb
                log.info(
                    f"[local-swap] rag_audio_emb <- local_swap composite; "
                    f"global unchanged. shapes rag={rag_audio_emb.shape} "
                    f"grag={grag_audio_emb.shape if grag_audio_emb is not None else None}"
                )

            # --- VAE EMBEDDINGS (if global_emb_type is 'vae' or 'both') ---
            rag_vae_emb = None
            negative_rag_vae_emb = None
            if cfg.get('global_rag_cond', False) and cfg.get('global_emb_type', 'beats') in ['vae', 'both']:
                with torch.no_grad():
                    with torch.amp.autocast(enabled=False, device_type=device):
                        feature_utils.tod.to(torch.float32)
                        dist = feature_utils.encode_audio(rag_composite_audio.to(torch.float32))
                        dist_neg = feature_utils.encode_audio(negative_rag_audio.to(torch.float32))
                        feature_utils.tod.to(dtype)

                    latent = dist.mode().transpose(1, 2)
                    if latent.shape[1] != net.latent_seq_len:
                        latent = F.interpolate(latent.permute(0, 2, 1), size=net.latent_seq_len, mode='nearest-exact').permute(0, 2, 1)
                    # rag_vae_emb = net.normalize(latent.to(dtype))
                    rag_vae_emb = latent.to(dtype)


                    latent_neg = dist_neg.mode().transpose(1, 2)
                    if latent_neg.shape[1] != net.latent_seq_len:
                        latent_neg = F.interpolate(latent_neg.permute(0, 2, 1), size=net.latent_seq_len, mode='nearest-exact').permute(0, 2, 1)
                    # negative_rag_vae_emb = net.normalize(latent_neg.to(dtype))
                    negative_rag_vae_emb = latent_neg.to(dtype)

            # --- INPAINTING ---
            masked_x1 = None
            mask = None
            latent_unmasked = None

            if model_inpainting:
                latent_unmasked = torch.zeros((bs, net.latent_seq_len, net.latent_dim), device=device, dtype=dtype)
                mask = torch.zeros((bs, net.latent_seq_len, 1), device=device, dtype=dtype)
                masked_x1 = latent_unmasked

            if model_inpainting and use_inpainting_input:
                # Fallback: if the JSONL doesn't provide dedicated inpainting
                # slots (topic_bench builds them empty), reuse the rag slots
                # so the inpainting path has something to anchor to. Opt-in
                # via cfg.inp_fallback_to_rag (default True when
                # use_inpainting_input is on); disable to keep the strict
                # "empty means no anchor" behaviour of the original script.
                _inp_fallback = bool(cfg.get('inp_fallback_to_rag', True))
                inp_audio_lists = list(batch.get('inp_audio_list', [[]] * bs))
                inp_seg_ref_lists = list(batch.get('inp_seg_in_ref', [[]] * bs))
                inp_seg_gen_lists = list(batch.get('inp_seg_in_gen', [[]] * bs))
                rag_audio_lists = list(batch.get('rag_audio_list', [[]] * bs))
                rag_seg_ref_lists = list(batch.get('rag_seg_in_ref', [[]] * bs))
                rag_seg_gen_lists = list(batch.get('rag_seg_in_gen', [[]] * bs))
                if _inp_fallback:
                    for i in range(bs):
                        if not inp_audio_lists[i] and rag_audio_lists[i]:
                            inp_audio_lists[i] = rag_audio_lists[i]
                            inp_seg_ref_lists[i] = rag_seg_ref_lists[i]
                            inp_seg_gen_lists[i] = rag_seg_gen_lists[i]
                inp_composite_list = []
                for i in range(bs):
                    icomp = create_composite_audio(
                        inp_audio_lists[i],
                        inp_seg_ref_lists[i],
                        inp_seg_gen_lists[i],
                        target_duration=cfg.duration_s,
                        sample_rate=seq_cfg.sampling_rate,
                        short_ref_mode=_short_ref_mode,
                    )
                    inp_composite_list.append(icomp)
                inp_composite_audio = torch.cat(inp_composite_list, dim=0).to(device)

                with torch.no_grad():
                    with torch.amp.autocast(enabled=False, device_type=device):
                        feature_utils.tod.to(torch.float32)
                        dist = feature_utils.encode_audio(inp_composite_audio.to(torch.float32))
                        feature_utils.tod.to(dtype)

                    latent = dist.mode().transpose(1, 2)
                    if latent.shape[1] != net.latent_seq_len:
                        latent = F.interpolate(latent.permute(0, 2, 1), size=net.latent_seq_len, mode='nearest-exact').permute(0, 2, 1)
                    latent_unmasked = net.normalize(latent.to(dtype))

                mask = torch.zeros((bs, net.latent_seq_len, 1), device=device, dtype=dtype)
                for i in range(bs):
                    gen_segs = inp_seg_gen_lists[i]
                    for gen_seg in gen_segs:
                        start_sec, end_sec = gen_seg
                        start_token = int((start_sec / cfg.duration_s) * net.latent_seq_len)
                        end_token = int((end_sec / cfg.duration_s) * net.latent_seq_len)
                        start_token = max(0, min(start_token, net.latent_seq_len))
                        end_token = max(0, min(end_token, net.latent_seq_len))
                        if start_token < end_token:
                            mask[i, start_token:end_token, :] = 1.0

                masked_x1 = latent_unmasked * mask

                # -- Anchor content-ablation (thought-1) ------------------
                # Default is 'none' so this loop is a no-op for legacy runs.
                # Modes: none|tshuffle|tile|awgn|lowpass. Strength in [0, 1].
                _anchor_perturb_mode = str(cfg.get('anchor_perturb', 'none')).lower()
                _anchor_perturb_strength = float(cfg.get('anchor_perturb_strength', 0.0))
                if _anchor_perturb_mode not in ('none', '', 'off'):
                    try:
                        masked_x1 = _perturb_anchor_latent(
                            masked_x1, mask,
                            mode=_anchor_perturb_mode,
                            strength=_anchor_perturb_strength,
                            seed=int(cfg.get('seed', 0)) + (id(batch) & 0xFFFF),
                        )
                        log.info(
                            f"[anchor-perturb] mode={_anchor_perturb_mode} "
                            f"strength={_anchor_perturb_strength}"
                        )
                    except Exception as _ap_err:
                        log.warning(f"[anchor-perturb] failed ({_ap_err}); "
                                    f"using untouched masked_x1")

                # -- Style-only bottleneck (idea-3) ----------------------
                # Collapse the anchor into a lower-bandwidth style signal so
                # timbre can survive while fine-grained content/onset cues are
                # harder to copy. Default 'none' keeps legacy behaviour.
                if _inp_style_bottleneck not in ('none', '', 'off'):
                    try:
                        masked_x1 = _bottleneck_anchor_latent(
                            masked_x1,
                            mask,
                            inp_seg_gen_lists,
                            float(cfg.duration_s),
                            mode=_inp_style_bottleneck,
                            mix=_inp_style_mix,
                        )
                    except Exception as _isb_err:
                        log.warning(f"[inpainting-style-bottleneck] failed ({_isb_err}); "
                                    f"using untouched masked_x1")

            # ------------------------------------------------------------------
            # Optional per-event grag stash. See notes at the top of the file
            # for the B口径: for each of the K events we encode its ref clip in
            # isolation and let networks.py blend the K style vecs onto the
            # latent time axis with a soft slot mask. Defaults to OFF so this
            # branch never triggers in production runs.
            _per_event_grag_on = bool(cfg.get('per_event_grag', False))
            _per_event_local_rag_on = bool(cfg.get('per_event_local_rag', False))
            _pe_stash_active = False
            # -- Per-slot inpainting scale map (independent from per-event grag) --
            # Accepts a comma-separated list of floats via
            # ``per_slot_inpainting_scale``. Element k applies to event k's
            # slot; ``per_slot_inpainting_scale_out`` (scalar) applies to gap
            # tokens. Both default to None => use the scalar inpainting_scale
            # from above. Never triggers unless the list is non-empty.
            _pslot_ips_active = False
            _pslot_ips_raw = cfg.get('per_slot_inpainting_scale', None)
            _pslot_ips_out_val = cfg.get('per_slot_inpainting_scale_out', None)
            _pslot_ips_list = None
            if _pslot_ips_raw is not None:
                if isinstance(_pslot_ips_raw, str):
                    _pslot_ips_list = [float(s) for s in _pslot_ips_raw.split(',') if s.strip()]
                elif isinstance(_pslot_ips_raw, (list, tuple)):
                    _pslot_ips_list = [float(s) for s in _pslot_ips_raw]
                else:
                    _pslot_ips_list = [float(_pslot_ips_raw)]
            if _pslot_ips_list and model_inpainting:
                try:
                    _N = net.latent_seq_len
                    _dur = float(cfg.duration_s)
                    _map = torch.full((bs, _N, 1),
                                      float(_pslot_ips_out_val) if _pslot_ips_out_val is not None
                                      else float(_inpainting_scale),
                                      device=device, dtype=dtype)
                    for i in range(bs):
                        gen_segs = batch.get('rag_seg_in_gen', [[]] * bs)[i]
                        for k, gen_seg in enumerate(gen_segs):
                            if k >= len(_pslot_ips_list):
                                break
                            beta_k = float(_pslot_ips_list[k])
                            s_sec, e_sec = float(gen_seg[0]), float(gen_seg[1])
                            s_tok = max(0, min(_N, int((s_sec / _dur) * _N)))
                            e_tok = max(0, min(_N, int((e_sec / _dur) * _N)))
                            if s_tok < e_tok:
                                _map[i, s_tok:e_tok, 0] = beta_k
                    net.inpainting_scale_map = _map
                    _pslot_ips_active = True
                    log.info(
                        f"[inpainting] per-slot scale map active: slot_betas={_pslot_ips_list} "
                        f"gap_beta={_pslot_ips_out_val if _pslot_ips_out_val is not None else _inpainting_scale}"
                    )
                except Exception as _pslot_err:
                    log.warning(f"[inpainting] failed to build per-slot scale map: {_pslot_err}")
                    net.inpainting_scale_map = None
                    _pslot_ips_active = False
            if (
                (_per_event_grag_on and cfg.get('use_decoupled_grag', False))
                or _per_event_local_rag_on
            ):
                try:
                    _pe_target = float(cfg.duration_s)
                    _pe_sr = 16000
                    _pe_canvases = []
                    _pe_slot_ranges = []
                    _pe_K_max = 0
                    for i in range(bs):
                        canvases_i, slots_i = build_per_event_canvases(
                            batch.get('rag_audio_list', [[]] * bs)[i],
                            batch.get('rag_seg_in_ref', [[]] * bs)[i],
                            batch.get('rag_seg_in_gen', [[]] * bs)[i],
                            target_duration=_pe_target,
                            sample_rate=_pe_sr,
                            short_ref_mode=_short_ref_mode,
                        )
                        _pe_canvases.append(canvases_i)
                        _pe_slot_ranges.append(slots_i)
                        _pe_K_max = max(_pe_K_max, len(canvases_i))
                    if _pe_K_max > 0:
                        # Pad each case up to K_max with zero canvases so the
                        # stack is regular.
                        _pe_samples = int(_pe_target * _pe_sr)
                        stack_rows = []
                        for canvases_i in _pe_canvases:
                            padded = list(canvases_i)
                            while len(padded) < _pe_K_max:
                                padded.append(torch.zeros((1, _pe_samples), dtype=torch.float32))
                            stack_rows.append(torch.stack([c.squeeze(0) for c in padded], dim=0))
                        _pe_wave_stack = torch.stack(stack_rows, dim=0).to(device)  # (B, K, samples)
                        # Batched timbre encode: fold K into batch.
                        B_pe, K_pe, S_pe = _pe_wave_stack.shape
                        with torch.no_grad():
                            with torch.amp.autocast(enabled=False, device_type=device):
                                _pe_flat_emb = timbre_extractor(
                                    _pe_wave_stack.reshape(B_pe * K_pe, S_pe)
                                )
                        _pe_emb = _pe_flat_emb.reshape(B_pe, K_pe, _pe_flat_emb.shape[1], _pe_flat_emb.shape[2]).to(dtype)
                        # Slot mask on the latent grid.
                        _pe_N = net.latent_seq_len
                        _pe_mask = build_slot_mask(
                            _pe_slot_ranges,
                            N_latent=_pe_N,
                            target_duration=_pe_target,
                            crossfade_frac=float(cfg.get('per_event_crossfade_frac', 0.05)),
                            gap_style=str(cfg.get('per_event_gap_style', 'mean')),
                            device=device,
                            dtype=dtype,
                        )
                        if _per_event_grag_on and cfg.get('use_decoupled_grag', False):
                            net.per_event_grag_stack = _pe_emb
                            net.per_event_slot_mask = _pe_mask
                        if _per_event_local_rag_on:
                            net.per_event_local_rag_stack = _pe_emb
                            net.per_event_local_slot_mask = _pe_mask
                        _pe_stash_active = True
                        log.info(
                            f"[per-event-grag] enabled: stack={_pe_emb.shape} "
                            f"mask={_pe_mask.shape} gap_style={cfg.get('per_event_gap_style', 'mean')} "
                            f"crossfade_frac={cfg.get('per_event_crossfade_frac', 0.05)}"
                        )
                except Exception as _pe_err:
                    log.warning(f"[per-event-grag] failed to build stash, falling back to classic grag: {_pe_err}")
                    net.per_event_grag_stack = None
                    net.per_event_slot_mask = None
                    net.per_event_local_rag_stack = None
                    net.per_event_local_slot_mask = None
                    _pe_stash_active = False

            try:
                audios = generate_rag(batch.get('clip_video', None),
                              batch.get('sync_video', None),
                              batch.get('caption', None),
                              ref_audio=rag_audio_emb,
                              negative_text=batch.get('negative_caption', None),
                              feature_utils=feature_utils,
                              net=net,
                              fm=fm,
                              rng=rng,
                              cfg_strength=cfg.cfg_strength,
                              cfg_rag_weight=cfg.get('cfg_rag_weight', 0.5),
                              cfg_schedule_mode=cfg.get('cfg_schedule_mode', 'base'),
                              clip_batch_size_multiplier=64,
                              sync_batch_size_multiplier=64,
                              fgc_pt=fgc_pt,
                              fgc_pool=cfg.fgc_pool,
                              fgc_v2=cfg.fgc_v2,
                              masked_x1=masked_x1,
                              mask=mask,
                              negative_rag_audio_emb=negative_rag_audio_emb,
                              latent_unmasked=latent_unmasked,
                              rag_vae_emb=rag_vae_emb,
                              negative_rag_vae_emb=negative_rag_vae_emb,
                              grag_audio_emb=grag_audio_emb,
                              negative_grag_audio_emb=negative_grag_audio_emb,
                              cfg_t_lo=float(cfg.get('cfg_t_lo', 0.0)),
                              cfg_t_hi=float(cfg.get('cfg_t_hi', 1.0)),
                              cfg_rescale_std=float(cfg.get('cfg_rescale_std', 0.0)),
                              )
            finally:
                # Always drop the per-event stash so the classic pipeline (any
                # subsequent case / eval script call) reverts to trained behaviour.
                if _pe_stash_active:
                    net.per_event_grag_stack = None
                    net.per_event_slot_mask = None
                    net.per_event_local_rag_stack = None
                    net.per_event_local_slot_mask = None
                if _pslot_ips_active:
                    net.inpainting_scale_map = None
            audios = audios.float().cpu()
            names = batch['name']
            for audio, name in zip(audios, names):
                torchaudio.save(output_dir / f'{name}.mp3', audio, seq_cfg.sampling_rate)


def distributed_setup():
    global local_rank, world_size
    if world_size > 1 and not distributed.is_initialized():
        distributed.init_process_group(backend="nccl")
        local_rank = distributed.get_rank()
        world_size_dist = distributed.get_world_size()
        log.info(f'Initialized: local_rank={local_rank}, world_size={world_size_dist}')
        return local_rank, world_size_dist
    return local_rank, world_size

if __name__ == '__main__':
    distributed_setup()

    main()

    # clean-up
    if distributed.is_initialized():
        distributed.destroy_process_group()
