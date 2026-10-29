# Copyright 2024 Sony Research Inc.
# Modifications Copyright 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import logging
import math
import random
from datetime import timedelta

import hydra
import numpy as np
import torch
import torch.distributed as distributed
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict
from torch.distributed.elastic.multiprocessing.errors import record

from vividaudio.data.data_setup import setup_training_datasets
from vividaudio.model.sequence_config import CONFIG_16K, CONFIG_44K
from vividaudio.runner import Runner
from vividaudio.utils.dist_utils import info_if_rank_zero, local_rank, world_size
from vividaudio.utils.logger import TensorboardLogger

import torch.nn.functional as F
import torchaudio
import torchaudio.functional as Fa
from vividaudio.timbre_extractor import TimbreExtractor

# 变速倍数范围（快放>1，慢放<1）
SPEED_FACTORS = [0.8, 1.2, 1.5, 0.7]  # 慢放0.7/0.8，快放1.2/1.5
SAMPLE_RATE = 16000
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

log = logging.getLogger()

def distributed_setup():
    distributed.init_process_group(backend="nccl", timeout=timedelta(hours=2))
    log.info(f'Initialized: local_rank={local_rank}, world_size={world_size}')
    return local_rank, world_size

def calculate_rms(waveform, frame_size=1024, hop_size=512):
    rms_values = []
    for i in range(0, waveform.size(1) - frame_size + 1, hop_size):
        frame = waveform[:, i:i + frame_size]
        rms = torch.sqrt(torch.mean(frame ** 2))  # 计算RMS
        rms_values.append(rms.item())  # 将RMS值添加到列表中
    return torch.tensor(rms_values)

def calculate_rms_per_token(waveform, sr, num_tokens, token_duration=0.02, frame_size=1024, hop_size=512):
    rms_values = []

    for token_idx in range(num_tokens):
        start_sample = int(token_idx * token_duration * sr)
        end_sample = int((token_idx + 1) * token_duration * sr)

        token_waveform = waveform[:, start_sample:end_sample]
        if token_waveform.sum() == 0:
            rms_values.append(0.0)
        else:
            rms = torch.sqrt(torch.mean(token_waveform ** 2))
            rms_values.append(rms.item())

    return torch.tensor(rms_values)

def time_stretch_audio_vectorized(audio, speed_factors, sample_rate=SAMPLE_RATE):
    B, T = audio.shape
    device = audio.device

    need_stretch = speed_factors != 1.0
    if not need_stretch.any():
        return audio

    stretched_audio = audio.clone()
    audio_to_stretch = audio[need_stretch]  # [B', T]
    factors_to_apply = speed_factors[need_stretch]  # [B']

    new_sample_rates = (sample_rate * factors_to_apply).int()
    for i, (audio_clip, new_sr, factor) in enumerate(zip(audio_to_stretch, new_sample_rates, factors_to_apply)):
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate,
            new_freq=new_sr.item(),
            dtype=audio_clip.dtype
        ).to(device)

        stretched = resampler(audio_clip.unsqueeze(0)).squeeze(0)
        orig_len = len(stretched)

        if orig_len > T:
            stretched = stretched[:T]
        else:
            padding_len = T - orig_len
            addition_times = (padding_len // orig_len) + 1  # 至少额外padding1个
            padded_stretched = torch.cat([stretched] * (addition_times + 1), dim=0)[:T]

            # （可选）方案2：仅重复开头的一小段（比如前10%）
            # head_len = max(1, int(orig_len * 0.1))  # 取开头10%，至少1个样本
            # head_audio = stretched[:head_len]
            # repeat_times = (padding_len // head_len) + 1
            # padded_head = torch.cat([head_audio] * repeat_times, dim=0)[:padding_len]
            # padded_stretched = torch.cat([stretched, padded_head], dim=0)

            stretched = padded_stretched

        stretched_audio[need_stretch][i] = stretched

    return stretched_audio

def extract_segment_features_vectorized(
    waveforms,
    segments,
    time_per_step,
    timbre_extractor,  # 参数名从 beat_model 改为更通用的 timbre_extractor
    speed_factors=None,
    map_method='mean'
):
    B, N, _ = segments.shape
    D = timbre_extractor.out_dim  # 使用统一定义的特征维度
    T_feat = int(8.0 / time_per_step)
    device = waveforms.device

    if speed_factors is None:
        speed_factors = torch.ones(B, N, device=device, dtype=torch.float32)

    start_samples = (segments[..., 0] * SAMPLE_RATE).long()
    end_samples = (segments[..., 1] * SAMPLE_RATE).long()

    start_samples = start_samples.clamp(0, waveforms.shape[1] - 1)
    end_samples = end_samples.clamp(0, waveforms.shape[1] - 1)

    seg_feats = torch.zeros(B, N, D, device=device, dtype=torch.float32)

    valid_seg_mask = start_samples < end_samples
    if not valid_seg_mask.any():
        return seg_feats

    for b in range(B):
        for n in range(N):
            if not valid_seg_mask[b, n]:
                continue

            start = start_samples[b, n]
            end = end_samples[b, n]
            audio_seg = waveforms[b, start:end]

            # 变速
            if speed_factors[b, n] != 1.0:
                audio_seg = time_stretch_audio_vectorized(
                    audio_seg.unsqueeze(0),
                    torch.tensor([speed_factors[b, n]], device=device)
                ).squeeze(0)

            seg_len = end - start
            if len(audio_seg) < seg_len:
                padding = torch.zeros(seg_len - len(audio_seg), device=device)
                audio_seg = torch.cat([audio_seg, padding])

            temp_waveform = torch.zeros_like(waveforms[b])
            temp_waveform[start:end] = audio_seg[:end-start]

            with torch.no_grad():
                # 统一调用 forward, 提取器内部处理 mask 等逻辑
                features = timbre_extractor(temp_waveform.unsqueeze(0))

            if features.shape[1] != T_feat and features.shape[1] > 1 and T_feat > 1:
                features = F.interpolate(
                    features.transpose(1, 2),
                    size=T_feat,
                    mode='linear',
                    align_corners=False
                ).transpose(1, 2)

            # 兼容全局特征 (如 PaSST 返回形状为 (1, 1, D))
            if features.shape[1] == 1:
                seg_feats[b, n] = features[0, 0]
                continue

            start_step = int(segments[b, n, 0].item() / time_per_step)
            end_step = int(segments[b, n, 1].item() / time_per_step)
            start_step = max(0, start_step)
            end_step = min(T_feat - 1, end_step)

            if start_step <= end_step:
                # 增加一层安全防护，防止模型实际返回特征比预期 T_feat 短
                end_step_safe = min(features.shape[1] - 1, end_step)
                if start_step <= end_step_safe:
                    seg_seq = features[0, start_step:end_step_safe + 1]  # (T_seg, D)

                    if map_method == 'mean':
                        seg_feats[b, n] = seg_seq.mean(dim=0)
                    elif map_method == 'copy':
                        center_idx = (seg_seq.shape[0] - 1) // 2
                        seg_feats[b, n] = seg_seq[center_idx]
                    else:
                        raise ValueError(f"Unknown map_method: {map_method}")
                else:
                    # 如果步长溢出，退而求其次取最后一个 token
                    seg_feats[b, n] = features[0, -1]

    return seg_feats


def fit_audio_to_length(audio_seg: torch.Tensor, target_len: int) -> torch.Tensor:
    if target_len <= 0:
        return audio_seg.new_zeros((0,))
    if audio_seg.numel() == 0:
        return audio_seg.new_zeros((target_len,))
    if audio_seg.shape[0] > target_len:
        return audio_seg[:target_len]
    if audio_seg.shape[0] < target_len:
        return F.pad(audio_seg, (0, target_len - audio_seg.shape[0]))
    return audio_seg


# ============================================================
# Reference-side augmentations (operate on the composite waveform
# that will be re-encoded by BEATs/timbre_extractor to form grag_audio_emb).
#
# 设计原则:
#   * 只作用于 grag_composite_waveform (ref 一侧),target/GT 保持 clean
#   * 每条样本独立采样(防止 batch 内信号同向)
#   * 仅在 ref 时段(非零区域)生效,空白处保持 0,避免引入伪事件
#   * 各项 probability 独立可调,默认全 0 = 关闭
# ============================================================


def _ref_active_mask(ref_wave: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """返回长度等于 ref_wave 的 bool mask,标识非静音区域。
    用于把 augmentation 限制在 ref 的有效区段(其它位置保持原 0)。"""
    return ref_wave.abs() > eps


def _apply_only_in_active(ref_wave: torch.Tensor,
                          processed: torch.Tensor,
                          active: torch.Tensor) -> torch.Tensor:
    """processed 只覆盖 active=True 的位置,其它保留 ref_wave 原值(通常是 0)。"""
    out = ref_wave.clone()
    if processed.shape[0] != ref_wave.shape[0]:
        processed = fit_audio_to_length(processed, ref_wave.shape[0])
    out[active] = processed[active]
    return out


def mix_distractor_into_ref(ref_wave: torch.Tensor,
                            distractor_wave: torch.Tensor,
                            snr_db: float) -> torch.Tensor:
    """
    方案 2: 在 ref 的非零区段叠加一段无关 distractor 波形,按指定 SNR(单位 dB) 控制混合比例。
    SNR 越小 distractor 越响。distractor 比 ref 长则随机裁剪,短则循环填充。
    """
    if ref_wave.numel() == 0:
        return ref_wave
    active = _ref_active_mask(ref_wave)
    if not active.any():
        return ref_wave

    T = ref_wave.shape[0]
    if distractor_wave.numel() == 0:
        return ref_wave
    if distractor_wave.shape[0] >= T:
        start = random.randint(0, distractor_wave.shape[0] - T)
        dist = distractor_wave[start:start + T]
    else:
        repeats = (T // distractor_wave.shape[0]) + 1
        dist = distractor_wave.repeat(repeats)[:T]
    dist = dist.to(ref_wave.device).to(ref_wave.dtype)

    ref_active = ref_wave[active]
    dist_active = dist[active]
    ref_power = ref_active.pow(2).mean().clamp_min(1e-10)
    dist_power = dist_active.pow(2).mean().clamp_min(1e-10)
    target_dist_power = ref_power / (10.0 ** (snr_db / 10.0))
    scale = torch.sqrt(target_dist_power / dist_power)
    mixed = ref_wave.clone()
    mixed[active] = ref_wave[active] + scale * dist[active]
    # 防爆: 若混合后峰值溢出,等比缩回到与 ref 同 peak
    peak = mixed.abs().max().clamp_min(1e-6)
    ref_peak = ref_wave.abs().max().clamp_min(1e-6)
    if peak > ref_peak * 1.5:
        mixed = mixed * (ref_peak * 1.5 / peak)
    return mixed


def add_gaussian_noise_snr(ref_wave: torch.Tensor, snr_db: float) -> torch.Tensor:
    """方案 6.a: 给 ref 加宽带高斯噪声,SNR 控制噪声强度。"""
    active = _ref_active_mask(ref_wave)
    if not active.any():
        return ref_wave
    ref_power = ref_wave[active].pow(2).mean().clamp_min(1e-10)
    noise_power = ref_power / (10.0 ** (snr_db / 10.0))
    noise = torch.randn_like(ref_wave) * torch.sqrt(noise_power)
    out = ref_wave.clone()
    out[active] = ref_wave[active] + noise[active]
    return out


def lowpass_biquad_simple(ref_wave: torch.Tensor,
                          sample_rate: int,
                          cutoff_hz: float) -> torch.Tensor:
    """方案 6.b: 一阶 IIR low-pass(等价 RC 低通),用张量原生算子,避免任何 CPU 跳出。"""
    if ref_wave.numel() == 0:
        return ref_wave
    active = _ref_active_mask(ref_wave)
    if not active.any():
        return ref_wave
    rc = 1.0 / (2.0 * math.pi * cutoff_hz)
    dt = 1.0 / sample_rate
    alpha = dt / (rc + dt)
    # 串行 IIR; 一段事件通常 < 2s = 32k samples,for-loop 太慢
    # 用 torchaudio biquad: a0=1, a1=-(1-alpha), b0=alpha, b1=0
    try:
        filtered = Fa.lfilter(
            ref_wave.unsqueeze(0),
            a_coeffs=torch.tensor([1.0, -(1.0 - alpha)], device=ref_wave.device, dtype=ref_wave.dtype),
            b_coeffs=torch.tensor([alpha, 0.0], device=ref_wave.device, dtype=ref_wave.dtype),
            clamp=False,
        ).squeeze(0)
    except Exception:
        return ref_wave
    return _apply_only_in_active(ref_wave, filtered, active)


def short_reverb(ref_wave: torch.Tensor,
                 sample_rate: int,
                 rt60_sec: float,
                 wet: float = 0.3) -> torch.Tensor:
    """方案 6.c: 用一个指数衰减脉冲响应做卷积模拟短混响。rt60_sec 控制混响时长。"""
    if ref_wave.numel() == 0:
        return ref_wave
    active = _ref_active_mask(ref_wave)
    if not active.any():
        return ref_wave
    ir_len = max(int(sample_rate * rt60_sec), 32)
    t = torch.arange(ir_len, device=ref_wave.device, dtype=ref_wave.dtype)
    decay = torch.exp(-6.9078 * t / max(ir_len - 1, 1))  # ~-60 dB at end
    noise = torch.randn(ir_len, device=ref_wave.device, dtype=ref_wave.dtype)
    ir = noise * decay
    ir = ir / ir.abs().max().clamp_min(1e-6)
    # 卷积
    wet_signal = F.conv1d(
        ref_wave.view(1, 1, -1),
        ir.view(1, 1, -1).flip(-1),
        padding=ir_len - 1,
    ).view(-1)[:ref_wave.shape[0]]
    mixed = (1.0 - wet) * ref_wave + wet * wet_signal
    return _apply_only_in_active(ref_wave, mixed, active)


def event_mix_with_distractor(event_wave: torch.Tensor,
                              distractor_pool: torch.Tensor,
                              cfg) -> torch.Tensor:
    """事件级 mix:在该事件波形上叠加一段 distractor(从 batch 内其它 sample 取整条波形,
    内部会随机裁剪/重复到事件长度)。返回与 event_wave 等长的波形。"""
    if distractor_pool is None or distractor_pool.numel() == 0:
        return event_wave
    snr_lo = float(getattr(cfg, 'aug_mix_snr_db_low', -5.0))
    snr_hi = float(getattr(cfg, 'aug_mix_snr_db_high', 15.0))
    snr_db = random.uniform(snr_lo, snr_hi)
    d_idx = random.randint(0, distractor_pool.shape[0] - 1)
    return mix_distractor_into_ref(event_wave, distractor_pool[d_idx], snr_db)


def event_degradation(event_wave: torch.Tensor, cfg,
                      sample_rate: int = SAMPLE_RATE) -> torch.Tensor:
    """事件级 degradation:在 noise/lowpass/reverb 三选一(各自 1/3 概率),
    range 仍由 aug_*_low/high 控制。"""
    choice = random.choice(('noise', 'lowpass', 'reverb'))
    if choice == 'noise':
        snr_lo = float(getattr(cfg, 'aug_noise_snr_db_low', 10.0))
        snr_hi = float(getattr(cfg, 'aug_noise_snr_db_high', 30.0))
        return add_gaussian_noise_snr(event_wave, random.uniform(snr_lo, snr_hi))
    if choice == 'lowpass':
        cf_lo = float(getattr(cfg, 'aug_lowpass_cutoff_low', 2000.0))
        cf_hi = float(getattr(cfg, 'aug_lowpass_cutoff_high', 8000.0))
        return lowpass_biquad_simple(event_wave, sample_rate, random.uniform(cf_lo, cf_hi))
    rt_lo = float(getattr(cfg, 'aug_reverb_rt60_low', 0.1))
    rt_hi = float(getattr(cfg, 'aug_reverb_rt60_high', 0.5))
    wet_lo = float(getattr(cfg, 'aug_reverb_wet_low', 0.1))
    wet_hi = float(getattr(cfg, 'aug_reverb_wet_high', 0.4))
    return short_reverb(event_wave, sample_rate,
                        random.uniform(rt_lo, rt_hi),
                        wet=random.uniform(wet_lo, wet_hi))



@record
@hydra.main(version_base='1.3.2', config_path='configs', config_name='train.yaml')
def train(cfg: DictConfig):
    # initial setup
    torch.cuda.set_device(local_rank)
    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
    distributed_setup()
    num_gpus = world_size
    run_dir = HydraConfig.get().run.dir

    # patch data dim
    if cfg.model.endswith('16k'):
        seq_cfg = CONFIG_16K
    elif cfg.model.endswith('44k'):
        seq_cfg = CONFIG_44K
    else:
        raise ValueError(f'Unknown model: {cfg.model}')

    with open_dict(cfg):
        cfg.data_dim.latent_seq_len = seq_cfg.latent_seq_len
        cfg.data_dim.clip_seq_len = seq_cfg.clip_seq_len
        cfg.data_dim.sync_seq_len = seq_cfg.sync_seq_len

    # wrap python logger with a tensorboard logger
    log = TensorboardLogger(cfg.exp_id,
                            run_dir,
                            logging.getLogger(),
                            is_rank0=(local_rank == 0),
                            enable_email=cfg.enable_email and not cfg.debug)

    info_if_rank_zero(log, f'All configuration: {cfg}')
    info_if_rank_zero(log, f'Number of GPUs detected: {num_gpus}')

    # number of dataloader workers
    info_if_rank_zero(log, f'Number of dataloader workers (per GPU): {cfg.num_workers}')

    # Set seeds to ensure the same initialization
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    # setting up configurations
    info_if_rank_zero(log, f'Training configuration: {cfg}')
    cfg.batch_size //= num_gpus
    info_if_rank_zero(log, f'Batch size (per GPU): {cfg.batch_size}')

    prob_same_group = float(cfg.prob_same_group)
    prob_speed_change = float(cfg.prob_speed_change)
    prob_mix = float(cfg.get('prob_mix', 0.0))
    prob_degradation = float(cfg.get('prob_degradation', 0.0))
    prob_empty = float(cfg.prob_empty)

    _prob_sum = prob_same_group + prob_speed_change + prob_mix + prob_degradation + prob_empty
    if abs(_prob_sum - 1.0) >= 1e-6:
        raise ValueError(
            "prob_same_group + prob_speed_change + prob_mix + prob_degradation + prob_empty "
            f"must sum to 1.0, got {_prob_sum:.6f}"
        )

    # determine time to change max skip
    total_iterations = cfg['num_iterations']

    # setup datasets
    dataset, sampler, loader = setup_training_datasets(cfg)
    info_if_rank_zero(log, f'Number of training samples: {len(dataset)}')
    info_if_rank_zero(log, f'Number of training batches: {len(loader)}')

    # compute and set mean and std
    latent_mean, latent_std = dataset.compute_latent_stats()
    timbre_type = getattr(cfg, 'audio_feat_type', 'beats')
    info_if_rank_zero(log, f"Initializing TimbreExtractor with type: {timbre_type}")
    # 提前初始化 extractor
    timbre_extractor = TimbreExtractor(model_type=timbre_type, device=f'cuda:{local_rank}')

    with torch.no_grad():
        dummy_wave = torch.zeros(1, 16000 * 8, device=f'cuda:{local_rank}')
        dummy_feat = timbre_extractor(dummy_wave)
        native_T = dummy_feat.shape[1]

    # 将真实的特征维度写入 cfg 中，供 Runner 和 Network 初始化使用
    cfg.rag_dim = dummy_feat.shape[-1]
    cfg.repa_dim = dummy_feat.shape[-1]
    cfg.fgc_pt = native_T / 8.0

    info_if_rank_zero(log, f"Auto-configured rag_dim={cfg.rag_dim}, repa_dim={cfg.repa_dim}, fgc_pt={cfg.fgc_pt} (native {native_T} frames) based on '{timbre_type}'")

    # construct the trainer
    trainer = Runner(cfg,
                     log=log,
                     run_path=run_dir,
                     for_training=True,
                     latent_mean=latent_mean,
                     latent_std=latent_std).enter_train()
    # load previous checkpoint if needed
    if cfg['checkpoint'] is not None:
        curr_iter = trainer.load_checkpoint(cfg['checkpoint'])
        cfg['checkpoint'] = None
        info_if_rank_zero(log, 'Model checkpoint loaded!')
    else:
        # if run_dir exists, load the latest checkpoint
        checkpoint = trainer.get_latest_checkpoint_path()
        if checkpoint is not None:
            curr_iter = trainer.load_checkpoint(checkpoint)
            info_if_rank_zero(log, 'Latest checkpoint loaded!')
        else:
            # load previous network weights if needed
            curr_iter = 0
            if cfg['weights'] is not None:
                info_if_rank_zero(log, 'Loading weights from the disk')
                trainer.load_weights(cfg['weights'])
                cfg['weights'] = None

    # determine max epoch
    trainable_params = []
    total_epoch = math.ceil(total_iterations / len(loader))
    current_epoch = curr_iter // len(loader)
    info_if_rank_zero(log, f'We will approximately use {total_epoch} epochs.')
    if not cfg.train_full:
        for name, param in trainer.network.named_parameters():
            if "rag" in name:
                param.requires_grad = True
            elif ("clip" in name and not cfg.sep_fgc) or ("clip" in name and cfg.clip_train): # attn的情况需要进一步调一下clip
                param.requires_grad = True
            elif "fgc_block" in name:
                param.requires_grad = True
            elif "delay" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    for name, param in trainer.network.named_parameters():
        if param.requires_grad:
            trainable_params.append({
                "name": name,
                "shape": param.shape,
                "num_params": param.numel(),  # Number of parameters
                "requires_grad": param.requires_grad
            })
    print("="*80)
    print(f"Total trainable parameters: {len(trainable_params)}")
    print("="*80)
    for idx, param_info in enumerate(trainable_params, 1):
        print(f"{idx:3d} | Name: {param_info['name']:<60} | Shape: {str(param_info['shape']):<20} | Params: {param_info['num_params']:,}")

    # Optional: Print total number of trainable parameters (count)
    total_trainable_num = sum(p["num_params"] for p in trainable_params)
    print("="*80)
    print(f"Total trainable parameter count: {total_trainable_num:,}")
    print("="*80)

    # training loop
    np.random.seed(np.random.randint(2**30 - 1) + local_rank * 1000)

    # 预生成速度因子张量，避免循环中重复创建
    speed_factors_tensor = torch.tensor(SPEED_FACTORS, device=f'cuda:{local_rank}', dtype=torch.float32)
    train_grag_mode = cfg.get('train_grag_mode', 'full')
    if cfg.w_prefetch:
        raise ValueError(
            "The portable data pipeline uses waveform RAG extraction; set w_prefetch=false.")

    while curr_iter < total_iterations:
        # Crucial for randomness!
        sampler.set_epoch(current_epoch)
        current_epoch += 1
        log.debug(f'Current epoch: {current_epoch}')

        trainer.enter_train()
        trainer.log.data_timer.start()

        for data in loader:
            rag_vae_emb = None
            if cfg.w_prefetch:
                _prefetched_features = data['features'].squeeze(1)
                _prefetched_rms = data['rms'].squeeze(1)
            else:
                # 修复：确保waveform移到正确的GPU
                waveform = data['waveform'].squeeze(1).cuda(local_rank, non_blocking=True)  # [B, T]
                B, T = waveform.shape
                use_rag_vae_emb = getattr(cfg, 'use_vae_latent_style_loss', False) or getattr(
                    cfg, 'global_emb_type', 'beats'
                ) in ['vae', 'both']

                with torch.no_grad():
                    # 1. 批量提取完整音频的BEATS特征
                    # padding_mask = torch.zeros_like(waveform, dtype=torch.bool, device=waveform.device)
                    # full_features = BEATs_model.extract_features(waveform, padding_mask=padding_mask)[0]  # [B, 496, 768]
                    # waveform: (B, T)
                    full_features = timbre_extractor(waveform)  # [B, T_feat, D]

                    target_len = int(8 * cfg.fgc_pt)
                    if full_features.shape[1] != target_len and full_features.shape[1] > 1 and target_len > 1:
                        full_features = F.interpolate(
                            full_features.transpose(1, 2),
                            size=target_len,
                            mode='linear',
                            align_corners=False
                        ).transpose(1, 2)

                    time_per_step = 8.0 / full_features.shape[1]
                    rag_audio_emb = torch.zeros_like(full_features, device=waveform.device)
                    grag_composite_waveform = (
                        torch.zeros_like(waveform, device=waveform.device)
                        if cfg.get('use_decoupled_grag', False) and train_grag_mode == 'composite'
                        else None
                    )
                    full_latent = data['a_mean'].cuda(local_rank, non_blocking=True) if use_rag_vae_emb else None
                    rag_vae_emb = torch.zeros_like(full_latent, device=waveform.device) if use_rag_vae_emb else None
                    latent_time_per_step = 8.0 / full_latent.shape[1] if use_rag_vae_emb else None

                    segments = data['segments'].cuda(local_rank, non_blocking=True)  # [B, N, 2]
                    group_idx = data['group_idx'].cuda(local_rank, non_blocking=True)  # [B, N]
                    B, N, _ = segments.shape
                    valid_mask = (segments[..., 0] >= 0) & (segments[..., 1] >= 0) & (group_idx >= 0)  # [B, N]
                    rand_probs = torch.rand(B, N, device=waveform.device, dtype=torch.float32)  # 批量生成随机数
                    # 5 类事件按累积区间划分:
                    # [0, p_sg)               → same_group
                    # [p_sg, p_sg+p_sc)       → speed_change
                    # [..., +p_mix)           → mix (event-level distractor mixing)
                    # [..., +p_deg)           → degradation (noise/lowpass/reverb)
                    # [..., 1.0)              → empty
                    th_sg = prob_same_group
                    th_sc = th_sg + prob_speed_change
                    th_mix = th_sc + prob_mix
                    th_deg = th_mix + prob_degradation
                    same_group_mask = rand_probs < th_sg
                    speed_change_mask = (rand_probs >= th_sg) & (rand_probs < th_sc)
                    mix_mask = (rand_probs >= th_sc) & (rand_probs < th_mix)
                    degradation_mask = (rand_probs >= th_mix) & (rand_probs < th_deg)
                    empty_mask = rand_probs >= th_deg

                    # 7. 应用有效掩码
                    same_group_mask = same_group_mask & valid_mask
                    speed_change_mask = speed_change_mask & valid_mask
                    mix_mask = mix_mask & valid_mask
                    degradation_mask = degradation_mask & valid_mask
                    empty_mask = empty_mask & valid_mask

                    # 8. 批量处理同组特征
                    if same_group_mask.any():
                        # 为每个事件找到同组的其他事件
                        for b in range(B):
                            for n in range(N):
                                if same_group_mask[b, n]:
                                    gid = group_idx[b, n]
                                    # 找到同组的其他有效事件
                                    group_events = (group_idx[b] == gid) & valid_mask[b] & (torch.arange(N, device=waveform.device) != n)
                                    if group_events.any():
                                        selected_n = torch.multinomial(group_events.float(), 1).item()  # 从同组的中抽出一个来。 只会从1的那些项中选择一个抽出来
                                        start_time = segments[b, selected_n, 0].item()
                                        end_time = segments[b, selected_n, 1].item()
                                        start_step = int(start_time / time_per_step)
                                        end_step = int(end_time / time_per_step)
                                        start_step = max(0, start_step)
                                        end_step = min(full_features.shape[1]-1, end_step)

                                        if start_step <= end_step:
                                            target_start = int(segments[b, n, 0].item() / time_per_step)
                                            target_end = int(segments[b, n, 1].item() / time_per_step)
                                            target_start = max(0, target_start)
                                            target_end = min(full_features.shape[1]-1, target_end)
                                            if target_start <= target_end:
                                                if cfg.map_method == 'mean':
                                                    rag_audio_emb[b, target_start:target_end+1] = full_features[b, start_step:end_step+1].mean(dim=0)
                                                elif cfg.map_method == 'copy':
                                                    if end_step - start_step == target_end - target_start:
                                                        rag_audio_emb[b, target_start:target_end+1] = full_features[b, start_step:end_step+1]
                                                    else:
                                                        idx = torch.linspace(
                                                            0, end_step - start_step, steps=target_end - target_start + 1, device=full_features.device
                                                        ).long()
                                                        rag_audio_emb[b, target_start:target_end+1] = full_features[b, start_step:end_step+1][idx]

                                                if grag_composite_waveform is not None:
                                                    src_start_sample = int(start_time * SAMPLE_RATE)
                                                    src_end_sample = int(end_time * SAMPLE_RATE)
                                                    tgt_start_sample = int(segments[b, n, 0].item() * SAMPLE_RATE)
                                                    tgt_end_sample = int(segments[b, n, 1].item() * SAMPLE_RATE)
                                                    src_start_sample = max(0, min(src_start_sample, waveform.shape[1]))
                                                    src_end_sample = max(0, min(src_end_sample, waveform.shape[1]))
                                                    tgt_start_sample = max(0, min(tgt_start_sample, waveform.shape[1]))
                                                    tgt_end_sample = max(0, min(tgt_end_sample, waveform.shape[1]))
                                                    if src_start_sample < src_end_sample and tgt_start_sample < tgt_end_sample:
                                                        target_len_samples = tgt_end_sample - tgt_start_sample
                                                        audio_seg = waveform[b, src_start_sample:src_end_sample]
                                                        grag_composite_waveform[b, tgt_start_sample:tgt_end_sample] = fit_audio_to_length(
                                                            audio_seg, target_len_samples
                                                        )

                                                if use_rag_vae_emb:
                                                    latent_start = int(start_time / latent_time_per_step)
                                                    latent_end = int(end_time / latent_time_per_step)
                                                    latent_start = max(0, latent_start)
                                                    latent_end = min(full_latent.shape[1] - 1, latent_end)
                                                    latent_target_start = int(segments[b, n, 0].item() / latent_time_per_step)
                                                    latent_target_end = int(segments[b, n, 1].item() / latent_time_per_step)
                                                    latent_target_start = max(0, latent_target_start)
                                                    latent_target_end = min(full_latent.shape[1] - 1, latent_target_end)
                                                    if latent_start <= latent_end and latent_target_start <= latent_target_end:
                                                        src_latent = full_latent[b, latent_start:latent_end+1]
                                                        tgt_latent_len = latent_target_end - latent_target_start + 1
                                                        if cfg.map_method == 'mean':
                                                            rag_vae_emb[b, latent_target_start:latent_target_end+1] = src_latent.mean(dim=0)
                                                        elif src_latent.shape[0] == tgt_latent_len:
                                                            rag_vae_emb[b, latent_target_start:latent_target_end+1] = src_latent
                                                        else:
                                                            resized_latent = F.interpolate(
                                                                src_latent.transpose(0, 1).unsqueeze(0),
                                                                size=tgt_latent_len,
                                                                mode='linear',
                                                                align_corners=False,
                                                            ).squeeze(0).transpose(0, 1)
                                                            rag_vae_emb[b, latent_target_start:latent_target_end+1] = resized_latent
                                    else:
                                        speed_change_mask[b, n] = True
                                        same_group_mask[b, n] = False

                    if speed_change_mask.any():
                        rand_speed_idx = torch.randint(0, len(SPEED_FACTORS), (B, N), device=waveform.device)
                        event_speed_factors = speed_factors_tensor[rand_speed_idx]  # [B, N]

                        seg_feats = extract_segment_features_vectorized(
                            waveform, segments, time_per_step, timbre_extractor, event_speed_factors, map_method = cfg.map_method
                        )

                        for b in range(B):
                            for n in range(N):
                                if speed_change_mask[b, n]:
                                    start_time = segments[b, n, 0].item()
                                    end_time = segments[b, n, 1].item()
                                    start_step = int(start_time / time_per_step)
                                    end_step = int(end_time / time_per_step)
                                    start_step = max(0, start_step)
                                    end_step = min(full_features.shape[1]-1, end_step)

                                    if start_step <= end_step:
                                        rag_audio_emb[b, start_step:end_step+1] = seg_feats[b, n]
                                        if grag_composite_waveform is not None:
                                            start_sample = int(start_time * SAMPLE_RATE)
                                            end_sample = int(end_time * SAMPLE_RATE)
                                            start_sample = max(0, min(start_sample, waveform.shape[1]))
                                            end_sample = max(0, min(end_sample, waveform.shape[1]))
                                            if start_sample < end_sample:
                                                audio_seg = waveform[b, start_sample:end_sample]
                                                speed_tensor = event_speed_factors[b, n].view(1)
                                                stretched = time_stretch_audio_vectorized(
                                                    audio_seg.unsqueeze(0), speed_tensor
                                                ).squeeze(0)
                                                target_len_samples = end_sample - start_sample
                                                grag_composite_waveform[b, start_sample:end_sample] = fit_audio_to_length(
                                                    stretched, target_len_samples
                                                )
                                        if use_rag_vae_emb:
                                            latent_start = int(start_time / latent_time_per_step)
                                            latent_end = int(end_time / latent_time_per_step)
                                            latent_start = max(0, latent_start)
                                            latent_end = min(full_latent.shape[1] - 1, latent_end)
                                            if latent_start <= latent_end:
                                                src_latent = full_latent[b, latent_start:latent_end+1]
                                                tgt_latent_len = latent_end - latent_start + 1
                                                if cfg.map_method == 'mean':
                                                    rag_vae_emb[b, latent_start:latent_end+1] = src_latent.mean(dim=0)
                                                elif src_latent.shape[0] == tgt_latent_len:
                                                    rag_vae_emb[b, latent_start:latent_end+1] = src_latent
                                                else:
                                                    resized_latent = F.interpolate(
                                                        src_latent.transpose(0, 1).unsqueeze(0),
                                                        size=tgt_latent_len,
                                                        mode='linear',
                                                        align_corners=False,
                                                    ).squeeze(0).transpose(0, 1)
                                                    rag_vae_emb[b, latent_start:latent_end+1] = resized_latent

                    # ===== 事件级 mix:把 distractor 混进当前事件波形,写入 composite =====
                    # rag_audio_emb 在对应 step 先用 clean full_features 占位
                    # (后面 composite forward 会覆盖,占位仅为 rag_written_mask 识别),
                    # 必须保证 grag_composite_waveform 不为 None(即 use_decoupled_grag=composite)。
                    if mix_mask.any() and grag_composite_waveform is not None:
                        B_cur = waveform.shape[0]
                        for b in range(B):
                            distractor_pool = None
                            if B_cur > 1:
                                other_indices = [j for j in range(B_cur) if j != b]
                                distractor_pool = waveform[other_indices]
                            for n in range(N):
                                if not mix_mask[b, n]:
                                    continue
                                start_time = segments[b, n, 0].item()
                                end_time = segments[b, n, 1].item()
                                start_step = int(start_time / time_per_step)
                                end_step = int(end_time / time_per_step)
                                start_step = max(0, start_step)
                                end_step = min(full_features.shape[1] - 1, end_step)
                                start_sample = max(0, min(int(start_time * SAMPLE_RATE), waveform.shape[1]))
                                end_sample = max(0, min(int(end_time * SAMPLE_RATE), waveform.shape[1]))
                                if start_step > end_step or start_sample >= end_sample:
                                    continue
                                event_wave = waveform[b, start_sample:end_sample]
                                mixed_wave = event_mix_with_distractor(event_wave, distractor_pool, cfg)
                                grag_composite_waveform[b, start_sample:end_sample] = fit_audio_to_length(
                                    mixed_wave, end_sample - start_sample
                                )
                                # rag_audio_emb 先用 clean 切片占位,后面 composite forward 会覆盖
                                rag_audio_emb[b, start_step:end_step+1] = full_features[b, start_step:end_step+1]
                                if use_rag_vae_emb:
                                    latent_start = max(0, int(start_time / latent_time_per_step))
                                    latent_end = min(full_latent.shape[1] - 1, int(end_time / latent_time_per_step))
                                    if latent_start <= latent_end:
                                        rag_vae_emb[b, latent_start:latent_end+1] = full_latent[b, latent_start:latent_end+1]

                    # ===== 事件级 degradation:对当前事件波形施加 noise/lowpass/reverb 之一 =====
                    if degradation_mask.any() and grag_composite_waveform is not None:
                        for b in range(B):
                            for n in range(N):
                                if not degradation_mask[b, n]:
                                    continue
                                start_time = segments[b, n, 0].item()
                                end_time = segments[b, n, 1].item()
                                start_step = int(start_time / time_per_step)
                                end_step = int(end_time / time_per_step)
                                start_step = max(0, start_step)
                                end_step = min(full_features.shape[1] - 1, end_step)
                                start_sample = max(0, min(int(start_time * SAMPLE_RATE), waveform.shape[1]))
                                end_sample = max(0, min(int(end_time * SAMPLE_RATE), waveform.shape[1]))
                                if start_step > end_step or start_sample >= end_sample:
                                    continue
                                event_wave = waveform[b, start_sample:end_sample]
                                degraded_wave = event_degradation(event_wave, cfg, sample_rate=SAMPLE_RATE)
                                grag_composite_waveform[b, start_sample:end_sample] = fit_audio_to_length(
                                    degraded_wave, end_sample - start_sample
                                )
                                rag_audio_emb[b, start_step:end_step+1] = full_features[b, start_step:end_step+1]
                                if use_rag_vae_emb:
                                    latent_start = max(0, int(start_time / latent_time_per_step))
                                    latent_end = min(full_latent.shape[1] - 1, int(end_time / latent_time_per_step))
                                    if latent_start <= latent_end:
                                        rag_vae_emb[b, latent_start:latent_end+1] = full_latent[b, latent_start:latent_end+1]

                    # 空特征保持0，无需处理

                    # ============================================================
                    # rag_audio_emb / grag_audio_emb 与 inference 对齐:
                    #   * same_group / speed_change 走 token-level 切片(已在上面写好)
                    #   * mix / degradation 走 composite 波形(已在上面写好) →
                    #     这里只要对 grag_composite_waveform 做一次 timbre_extractor,
                    #     就能拿到 rag 和 grag 都用得上的统一特征。
                    #   * rag/grag 共用同一份(可能含 aug) composite,
                    #     与 batch_eval_rag.py 推理一致。
                    # rag_emb_source 控制 same_group/speed_change 段的 rag 特征来源:
                    #   * 'full_features' (默认/legacy): 保留上面切片得到的 GT 特征
                    #   * 'composite': 覆盖为 composite forward 结果(完全对齐推理)
                    # 触发 composite forward 的条件:
                    #   - 用户显式选 composite,或
                    #   - 当前 batch 出现了 mix/degradation 事件(必须用 composite 才能拿到对应特征)。
                    # ============================================================
                    rag_emb_source = str(cfg.get('rag_emb_source', 'full_features'))
                    has_event_aug = bool(mix_mask.any().item() or degradation_mask.any().item())
                    need_composite_forward = grag_composite_waveform is not None and (
                        rag_emb_source == 'composite' or has_event_aug
                    )

                    if need_composite_forward:
                        with torch.no_grad():
                            composite_features = timbre_extractor(grag_composite_waveform)
                            if (
                                composite_features.shape[1] != full_features.shape[1]
                                and composite_features.shape[1] > 1
                                and full_features.shape[1] > 1
                            ):
                                composite_features = F.interpolate(
                                    composite_features.transpose(1, 2),
                                    size=full_features.shape[1],
                                    mode='linear',
                                    align_corners=False,
                                ).transpose(1, 2)
                            # 只覆盖 rag_audio_emb 中已写入 ref 的 step;empty 区域保持 0
                            rag_written_mask = (rag_audio_emb.abs().sum(dim=-1) > 0)
                            rag_audio_emb = torch.where(
                                rag_written_mask.unsqueeze(-1),
                                composite_features,
                                rag_audio_emb,
                            )


                    # Innovation C: Masked RAG Modeling (Block Masking)
                    # Force the model to "inpaint" the missing RAG features based on text/video context
                    mask_rag_ratio = getattr(cfg, 'mask_rag_ratio', 0.25)
                    if mask_rag_ratio > 0.0:
                        block_size = getattr(cfg, 'mask_rag_block_size', 10)  # default 10 frames (~160ms)
                        for b in range(B):
                            num_blocks = int((full_features.shape[1] * mask_rag_ratio) / block_size)
                            if num_blocks > 0:
                                starts = torch.randint(0, full_features.shape[1] - block_size + 1, (num_blocks,))
                                for start in starts:
                                    rag_audio_emb[b, start:start+block_size] = 0.0
                                    if rag_vae_emb is not None:
                                        latent_block_size = max(1, int(round(block_size * full_latent.shape[1] / full_features.shape[1])))
                                        latent_start = min(int(round(start.item() * full_latent.shape[1] / full_features.shape[1])), full_latent.shape[1] - 1)
                                        latent_end = min(latent_start + latent_block_size, full_latent.shape[1])
                                        rag_vae_emb[b, latent_start:latent_end] = 0.0

            data['rag_audio_emb'] = rag_audio_emb
            data['beats_features'] = full_features
            if rag_vae_emb is not None:
                data['rag_vae_emb'] = rag_vae_emb
            if cfg.get('use_decoupled_grag', False):
                if train_grag_mode == 'composite':
                    # grag_composite_waveform 在上面的事件循环中已经写好了
                    # (same_group/speed_change 直接放波形, mix/degradation 经过增强后放波形),
                    # 这里把它一次性 forward 成 grag_audio_emb。
                    # rag 一侧若 rag_emb_source='composite' 或有 mix/degradation 事件,
                    # 也已从同一份 composite_features 覆盖,与 inference 完全对齐。
                    with torch.no_grad():
                        grag_audio_emb = timbre_extractor(grag_composite_waveform)
                    if grag_audio_emb.shape[1] != target_len and grag_audio_emb.shape[1] > 1 and target_len > 1:
                        grag_audio_emb = F.interpolate(
                            grag_audio_emb.transpose(1, 2),
                            size=target_len,
                            mode='linear',
                            align_corners=False
                        ).transpose(1, 2)
                    data['grag_audio_emb'] = grag_audio_emb
                else:
                    data['grag_audio_emb'] = full_features.clone()

                # -------- Optional override: alternative source for grag ---------
                # `train_grag_source` controls what feeds the global path:
                #   * 'shared' (default) : keep the value assigned above (composite
                #                          or full_features), i.e. grag shares
                #                          the local path's information.
                #   * 'tail'             : take `tail_waveform` from the dataloader
                #                          (the 8-10s region of the original 10s
                #                          VGGSound clip, which is *outside* the
                #                          8s target and hence leak-free),
                #                          BEATs-encode it, interpolate up to
                #                          target_len, and use it as grag input.
                #   * 'mix'              : for each sample, randomly pick 'shared'
                #                          or 'tail' according to
                #                          `train_grag_source_prob_tail`.
                grag_source = str(cfg.get('train_grag_source', 'shared')).lower()
                if grag_source in ('tail', 'mix'):
                    tail_waveform = data.get('tail_waveform', None)
                    if tail_waveform is None:
                        # dataloader did not provide it; skip override so
                        # training proceeds with 'shared' behaviour
                        pass
                    else:
                        tail_waveform = tail_waveform.squeeze(1).cuda(
                            local_rank, non_blocking=True
                        )
                        with torch.no_grad():
                            tail_features = timbre_extractor(tail_waveform)
                        if (tail_features.shape[1] != target_len
                            and tail_features.shape[1] > 1 and target_len > 1):
                            tail_features = F.interpolate(
                                tail_features.transpose(1, 2),
                                size=target_len,
                                mode='linear',
                                align_corners=False,
                            ).transpose(1, 2)

                        if grag_source == 'tail':
                            data['grag_audio_emb'] = tail_features
                        else:  # 'mix' -- per-sample random switch
                            import random as _random
                            p_tail = float(cfg.get(
                                'train_grag_source_prob_tail', 0.5))
                            B_ = tail_features.shape[0]
                            device_ = tail_features.device
                            mask = torch.tensor(
                                [_random.random() < p_tail for _ in range(B_)],
                                device=device_,
                            )
                            # broadcast mask over (T, D)
                            m = mask.view(B_, 1, 1)
                            shared_features = data['grag_audio_emb']
                            data['grag_audio_emb'] = torch.where(
                                m, tail_features, shared_features
                            )
            trainer.train_pass(data, curr_iter)

            curr_iter += 1
            if curr_iter >= total_iterations:
                break

    trainer.save_checkpoint(curr_iter)
    trainer.save_weights(curr_iter)
    del trainer
    torch.cuda.empty_cache()

    distributed.barrier()
    log.complete()
    distributed.barrier()
    distributed.destroy_process_group()

if __name__ == '__main__':
    train()
