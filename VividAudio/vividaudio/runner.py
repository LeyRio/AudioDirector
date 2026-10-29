# Copyright 2024 Sony Research Inc.
# Modifications Copyright 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

"""
trainer.py - wrapper and utility functions for network training
Compute loss, back-prop, update parameters, logging, etc.
"""
import os
from pathlib import Path
from typing import Optional, Union

import torch
import torch.distributed
import torch.optim as optim
from nitrous_ema import PostHocEMA
from omegaconf import DictConfig
from torch.nn.parallel import DistributedDataParallel as DDP

from vividaudio.model.flow_matching import FlowMatching
from vividaudio.model.networks import get_my_mmaudio
from vividaudio.model.sequence_config import CONFIG_16K, CONFIG_44K
from vividaudio.model.utils.features_utils import FeaturesUtils
from vividaudio.model.utils.parameter_groups import get_parameter_groups
from vividaudio.model.utils.sample_utils import log_normal_sample
from vividaudio.utils.dist_utils import (info_if_rank_zero, local_rank, string_if_rank_zero)
from vividaudio.utils.log_integrator import Integrator
from vividaudio.utils.logger import TensorboardLogger
from vividaudio.utils.time_estimator import PartialTimeEstimator, TimeEstimator
import torch.nn.functional as F
import random


class Runner:

    def __init__(self,
                 cfg: DictConfig,
                 log: TensorboardLogger,
                 run_path: Union[str, Path],
                 for_training: bool = True,
                 latent_mean: Optional[torch.Tensor] = None,
                 latent_std: Optional[torch.Tensor] = None):
        self.exp_id = cfg.exp_id
        self.use_amp = cfg.amp
        self.enable_grad_scaler = cfg.enable_grad_scaler
        self.for_training = for_training
        self.cfg = cfg

        if cfg.model.endswith('16k'):
            self.seq_cfg = CONFIG_16K
            mode = '16k'
        elif cfg.model.endswith('44k'):
            self.seq_cfg = CONFIG_44K
            mode = '44k'
        else:
            raise ValueError(f'Unknown model: {cfg.model}')

        self.sample_rate = self.seq_cfg.sampling_rate
        self.duration_sec = self.seq_cfg.duration
        self.inpainting = cfg.get('inpainting', False)
        self.zero_mask_ratio = cfg.get('zero_mask_ratio', 0.0)

        # setting up the model
        empty_string_feat = torch.load(
            cfg.empty_string_feat, weights_only=True, map_location='cpu')[0]
        self.network = DDP(get_my_mmaudio(cfg.model,
                                          latent_mean=latent_mean,
                                          latent_std=latent_std,
                                          sep_fgc=cfg.sep_fgc,
                                          joint_fgc=cfg.joint_fgc,
                                          delay_fusion=cfg.delay_fusion,
                                          interleaved_rope=cfg.interleaved_rope,
                                          fusion_fgc_clip=cfg.fusion_fgc_clip,
                                          fgc_pt=cfg.fgc_pt,
                                          qformer=cfg.qformer,
                                          empty_string_feat=empty_string_feat,
                                          inpainting=self.inpainting,
                                          sync_rope=cfg.get('sync_rope', False),
                                          cross_attn_delay=cfg.get('cross_attn_delay', False),
                                          local_window_size=cfg.get('local_window_size', 0),
                                          hybrid_cross_attn=cfg.get('hybrid_cross_attn', False),
                                          zero_init_gate=cfg.get('zero_init_gate', True),
                                          mask_rag_ratio=cfg.get('mask_rag_ratio', 0.0),
                                          mask_rag_block_size=cfg.get('mask_rag_block_size', 10),
                                          global_rag_cond=cfg.get('global_rag_cond', False),
                                          rag_dim=cfg.get('rag_dim', 768),
                                          repa_dim=cfg.get('repa_dim', 768),
                                          global_emb_type=cfg.get('global_emb_type', 'beats'),
                                          bypass_empty_inpainting=cfg.get('bypass_empty_inpainting', False),
                                          use_decoupled_grag=cfg.get('use_decoupled_grag', False),
                                          grag_extractor_type=cfg.get('grag_extractor_type', 'mean'),
                                          grag_qformer_n_q=cfg.get('grag_qformer_n_q', 8),
                                          grag_qformer_n_layers=cfg.get('grag_qformer_n_layers', 2)).cuda(),
                           device_ids=[local_rank],
                           find_unused_parameters=True,
                           broadcast_buffers=False)
        if cfg.compile:
            # NOTE: though train_fn and val_fn are very similar
            # (early on they are implemented as a single function)
            # keeping them separate and compiling them separately are CRUCIAL for high performance
            self.train_fn = torch.compile(self.train_fn)
            self.val_fn = torch.compile(self.val_fn)

        self.fm = FlowMatching(cfg.sampling.min_sigma,
                               inference_mode=cfg.sampling.method,
                               num_steps=cfg.sampling.num_steps)

        # ema profile
        if for_training and cfg.ema.enable and local_rank == 0:
            self.ema = PostHocEMA(self.network.module,
                                  sigma_rels=cfg.ema.sigma_rels,
                                  update_every=cfg.ema.update_every,
                                  checkpoint_every_num_steps=cfg.ema.checkpoint_every,
                                  checkpoint_folder=cfg.ema.checkpoint_folder,
                                  step_size_correction=True).cuda()
            self.ema_start = cfg.ema.start
        else:
            self.ema = None

        self.rng = torch.Generator(device='cuda')
        self.rng.manual_seed(cfg['seed'] + local_rank)

        # setting up feature extractors and VAEs
        if mode == '16k':
            self.features = FeaturesUtils(
                tod_vae_ckpt=cfg['vae_16k_ckpt'],
                bigvgan_vocoder_ckpt=cfg['bigvgan_vocoder_ckpt'],
                synchformer_ckpt=cfg['synchformer_ckpt'],
                enable_conditions=True,
                mode=mode,
                need_vae_encoder=False,
            )
        elif mode == '44k':
            self.features = FeaturesUtils(
                tod_vae_ckpt=cfg['vae_44k_ckpt'],
                synchformer_ckpt=cfg['synchformer_ckpt'],
                enable_conditions=True,
                mode=mode,
                need_vae_encoder=False,
            )
        self.features = self.features.cuda().eval()

        if cfg.compile:
            self.features.compile()

        # hyperparameters
        self.log_normal_sampling_mean = cfg.sampling.mean
        self.log_normal_sampling_scale = cfg.sampling.scale
        self.null_condition_probability = cfg.null_condition_probability
        self.cfg_strength = cfg.cfg_strength

        # setting up logging
        self.log = log
        self.run_path = Path(run_path)
        string_if_rank_zero(self.log, 'model_size',
                            f'{sum([param.nelement() for param in self.network.parameters()])}')
        string_if_rank_zero(
            self.log, 'number_of_parameters_that_require_gradient: ',
            str(
                sum([
                    param.nelement()
                    for param in filter(lambda p: p.requires_grad, self.network.parameters())
                ])))
        info_if_rank_zero(self.log, 'torch version: ' + torch.__version__)
        self.train_integrator = Integrator(self.log, distributed=True)
        self.val_integrator = Integrator(self.log, distributed=True)

        # setting up optimizer and loss
        if for_training:
            self.enter_train()
            parameter_groups = get_parameter_groups(self.network, cfg, print_log=(local_rank == 0))

            if self.cfg.use_muon:
                hidden_matrix_params = [p for n, p in self.network.named_parameters() if p.ndim >= 2 and "embed" not in n]
                embed_params = [p for n, p in self.network.named_parameters() if "embed" in n]
                scalar_params = [p for n, p in self.network.named_parameters() if p.ndim < 2 and "embed" not in n]
                # head_params = [self.network.lm_head.weight]

                # init the optimizer(s)
                # small adam epsilon by @YouJiacheng. this is an alternate method of fixing the world_size dependence
                # discovered by @fernbear.bsky.social https://x.com/hi_tysam/status/1879692937589875094
                from muon import MuonWithAuxAdam
                adam_groups = [dict(params=embed_params, lr=2e-5), dict(params=scalar_params, lr=2e-5)]
                adam_groups = [dict(**g, betas=(0.9, 0.95), eps=1e-8, use_muon=False) for g in adam_groups]
                muon_group = dict(params=hidden_matrix_params, lr=2e-5, momentum=0.95, use_muon=True)
                param_groups = [*adam_groups, muon_group]
                optimizer = MuonWithAuxAdam(param_groups)
                self.optimizer = optimizer
            else:
                self.optimizer = optim.AdamW(parameter_groups,
                                            lr=cfg['learning_rate'],
                                            weight_decay=cfg['weight_decay'],
                                            betas=[0.9, 0.95],
                                            eps=1e-6 if self.use_amp else 1e-8,
                                            fused=True)
            if self.enable_grad_scaler:
                self.scaler = torch.amp.GradScaler(init_scale=2048)
            self.clip_grad_norm = cfg['clip_grad_norm']

            # linearly warmup learning rate
            linear_warmup_steps = cfg['linear_warmup_steps']

            def warmup(currrent_step: int):
                return (currrent_step + 1) / (linear_warmup_steps + 1)

            warmup_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=warmup)

            # setting up learning rate scheduler
            if cfg['lr_schedule'] == 'constant':
                next_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda _: 1)
            elif cfg['lr_schedule'] == 'poly':
                total_num_iter = cfg['num_iterations']
                next_scheduler = optim.lr_scheduler.LambdaLR(self.optimizer,
                                                             lr_lambda=lambda x:
                                                             (1 - (x / total_num_iter))**0.9)
            elif cfg['lr_schedule'] == 'step':
                next_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                                cfg['lr_schedule_steps'],
                                                                cfg['lr_schedule_gamma'])
            elif cfg['lr_schedule'] == 'cosine':
                total_num_iter = cfg['num_iterations']
                next_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                                      T_max = total_num_iter - linear_warmup_steps,
                                                                      eta_min = 1e-6)
            else:
                raise NotImplementedError

            self.scheduler = optim.lr_scheduler.SequentialLR(self.optimizer,
                                                             [warmup_scheduler, next_scheduler],
                                                             [linear_warmup_steps])

            # Logging info
            self.log_text_interval = cfg['log_text_interval']
            self.log_extra_interval = cfg['log_extra_interval']
            self.save_weights_interval = cfg['save_weights_interval']
            self.save_checkpoint_interval = cfg['save_checkpoint_interval']
            self.save_copy_iterations = cfg['save_copy_iterations']
            self.num_iterations = cfg['num_iterations']
            if cfg['debug']:
                self.log_text_interval = self.log_extra_interval = 1

            # update() is called when we log metrics, within the logger
            self.log.batch_timer = TimeEstimator(self.num_iterations, self.log_text_interval)
            # update() is called every iteration, in this script
            self.log.data_timer = PartialTimeEstimator(self.num_iterations, 1, ema_alpha=0.9)
        else:
            self.enter_val()

    # def _generate_inpainting_mask(self, x1):
    #     bs, seq_len, _ = x1.shape
    #     mask = torch.ones((bs, seq_len, 1), device=x1.device)
    #     for i in range(bs):
    #         num_segments = torch.randint(1, 4, (1,)).item()
    #         for _ in range(num_segments):
    #             length = torch.randint(seq_len // 10, max(2, seq_len // 2), (1,)).item()
    #             start = torch.randint(0, max(1, seq_len - length), (1,)).item()
    #             mask[i, start:start+length] = 0.0
    #     return mask

    def _generate_inpainting_mask(
        self,
        x1
    ):
        bs, seq_len, _ = x1.shape
        mask = torch.ones((bs, seq_len, 1), device=x1.device)
        mask_pos_ratio = [0.0, 1.0]
        mask_seg_num = [self.cfg.mask_seg_num_low, self.cfg.mask_seg_num_high]  # 默认[1, 3]
        min_mask_len_ratio = self.cfg.min_mask_len_ratio
        max_mask_len_ratio = self.cfg.max_mask_len_ratio

        start_ratio, end_ratio = mask_pos_ratio

        for i in range(bs):
            num_segments = torch.randint(mask_seg_num[0], mask_seg_num[1], (1,)).item()

            for _ in range(num_segments):
                min_len = max(1, int(seq_len * min_mask_len_ratio))  # 最小长度（至少1）
                max_len = min(seq_len, int(seq_len * max_mask_len_ratio))  # 最大长度（不超过序列）
                max_len = max(min_len + 1, max_len)  # 避免min和max相等
                length = torch.randint(min_len, max_len, (1,)).item()

                pos_start = int(seq_len * start_ratio)
                pos_end = int(seq_len * end_ratio) - length
                pos_end = max(pos_start, pos_end)
                if pos_start >= pos_end:
                    continue
                start = torch.randint(pos_start, pos_end + 1, (1,)).item()
                mask[i, start:start+length] = 0.0

        return mask

    def train_fn(
        self,
        clip_f: torch.Tensor,
        sync_f: torch.Tensor,
        text_f: torch.Tensor,
        a_mean: torch.Tensor,
        a_std: torch.Tensor,
        rag_f: torch.Tensor,
        beats_f: torch.Tensor = None,
        rag_vae_f: torch.Tensor = None,
        grag_f: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # sample
        a_randn = torch.empty_like(a_mean).normal_(generator=self.rng)
        x1 = a_mean + a_std * a_randn
        bs = x1.shape[0]  # batch_size * seq_len * num_channels

        # normalize the latents
        x1 = self.network.module.normalize(x1)

        t = log_normal_sample(x1,
                              generator=self.rng,
                              m=self.log_normal_sampling_mean,
                              s=self.log_normal_sampling_scale)
        x0, x1, xt, (clip_f, sync_f, text_f, rag_f) = self.fm.get_x0_xt_c(x1,
                                                                   t,
                                                                   Cs=[clip_f, sync_f, text_f, rag_f],
                                                                   generator=self.rng)

        if self.inpainting:
            mask = self._generate_inpainting_mask(x1)
            if random.random() < self.zero_mask_ratio:
                mask = torch.zeros_like(mask)
            masked_x1 = x1 * mask
            xt_in = torch.cat([xt, masked_x1, mask], dim=-1)
        else:
            xt_in = xt

        # classifier-free training
        samples = torch.rand(bs, device=x1.device, generator=self.rng)
        null_video = (samples < self.null_condition_probability)
        clip_f[null_video] = self.network.module.empty_clip_feat
        sync_f[null_video] = self.network.module.empty_sync_feat


        samples = torch.rand(bs, device=x1.device, generator=self.rng)
        null_text = (samples < self.null_condition_probability)
        text_f[null_text] = self.network.module.empty_string_feat

        pred_v, inter_feat_proj = self.network(
            xt_in, clip_f, sync_f, text_f, t, rag_f, return_inter=True, rag_vae_f=rag_vae_f, grag_f=grag_f
        )

        def align_seq(x: torch.Tensor, target_len: int) -> torch.Tensor:
            if x.shape[1] == target_len:
                return x
            if x.shape[1] == 1:
                return x.expand(-1, target_len, -1)
            return F.interpolate(
                x.transpose(1, 2),
                size=target_len,
                mode='linear',
                align_corners=False,
            ).transpose(1, 2)

        def gram_matrix(x: torch.Tensor) -> torch.Tensor:
            # x: (B, T, D). Normalize first so Gram focuses on correlation/style instead of scale.
            x = F.normalize(x, p=2, dim=-1)
            features = x.transpose(1, 2)  # (B, D, T)
            return torch.bmm(features, features.transpose(1, 2)).div(x.shape[1])

        # Estimate the clean latent in normalized VAE space from the predicted flow.
        x1_hat = pred_v + (1 - self.fm.min_sigma) * x0

        # compute REPA loss
        repa_loss = pred_v.new_zeros(())
        vae_style_loss = pred_v.new_zeros(())
        semantic_anchor_loss = pred_v.new_zeros(())
        global_grag_cosine_loss = pred_v.new_zeros(())
        if beats_f is not None:
            B, T_beats, D_beats = beats_f.shape
            # inter_feat_proj: (B, N, 768)
            if T_beats == 1:
                # è‹¥ä¸ºå…¨å±€ç‰¹å¾(å¦‚PaSST)ï¼Œç›´æŽ¥å¯¹ç½‘ç»œä¸­é—´ç‰¹å¾æ±‚å…¨å±€å‡å€¼
                inter_feat_aligned = inter_feat_proj.mean(dim=1, keepdim=True)
            else:
                # è‹¥ä¸ºå¸§çº§ç‰¹å¾ï¼Œåˆ™çº¿æ€§æ’å€¼å¯¹é½
                inter_feat_proj_t = inter_feat_proj.transpose(1, 2)  # (B, D, N)
                inter_feat_aligned = F.interpolate(inter_feat_proj_t, size=T_beats, mode='linear', align_corners=False)
                inter_feat_aligned = inter_feat_aligned.transpose(1, 2)  # (B, T_beats, D)

            D_inter = inter_feat_proj.shape[-1]
            if D_beats != D_inter:
                if D_beats > D_inter:
                    target_f = beats_f[:, :, :D_inter] # ç®€å•æˆªæ–­
                else:
                    target_f = F.pad(beats_f, (0, D_inter - D_beats)) # è¡¥é›¶
            else:
                target_f = beats_f

            # cosine similarity
            cos_sim = F.cosine_similarity(inter_feat_aligned, target_f, dim=-1)
            frame_cos_loss = 1.0 - cos_sim.mean()
            frame_cos_weight = self.cfg.get('frame_cos_weight', 1.0)

            gram_gen = gram_matrix(inter_feat_aligned)
            gram_tgt = gram_matrix(target_f)

            style_loss = F.l1_loss(gram_gen, gram_tgt)
            style_weight = self.cfg.get('style_weight', 50.0) # å¦‚æžœç”¨ L1ï¼Œæƒé‡å¯ä»¥è®¾åœ¨ 10~50 ä¹‹é—´
            repa_loss = frame_cos_weight * frame_cos_loss + style_weight * style_loss

        if self.cfg.get('use_vae_latent_style_loss', False) and rag_vae_f is not None:
            low_noise_mask = t >= self.cfg.get('vae_latent_style_t_min', 0.7)
            if low_noise_mask.any():
                x1_hat_style = x1_hat[low_noise_mask]
                rag_vae_style = rag_vae_f[low_noise_mask]
                rag_vae_style = self.network.module.normalize(rag_vae_style)
                rag_vae_style = align_seq(rag_vae_style, x1_hat_style.shape[1])

                vae_gram_loss = F.l1_loss(gram_matrix(x1_hat_style), gram_matrix(rag_vae_style))
                x1_hat_mu = x1_hat_style.mean(dim=1)
                rag_vae_mu = rag_vae_style.mean(dim=1)
                x1_hat_std = x1_hat_style.std(dim=1, unbiased=False)
                rag_vae_std = rag_vae_style.std(dim=1, unbiased=False)
                vae_stat_loss = F.l1_loss(x1_hat_mu, rag_vae_mu) + F.l1_loss(x1_hat_std, rag_vae_std)
                vae_style_loss = vae_gram_loss + self.cfg.get('vae_latent_style_stat_weight', 0.5) * vae_stat_loss

        if self.cfg.get('use_semantic_anchor_loss', False):
            semantic_mask = t >= self.cfg.get('semantic_anchor_t_min', 0.3)
            if semantic_mask.any():
                x1_hat_sem = x1_hat[semantic_mask]
                x1_target = x1[semantic_mask]
                semantic_cos_loss = 1.0 - F.cosine_similarity(
                    x1_hat_sem.mean(dim=1), x1_target.mean(dim=1), dim=-1
                ).mean()

                if x1_hat_sem.shape[1] > 1:
                    delta_hat = x1_hat_sem[:, 1:] - x1_hat_sem[:, :-1]
                    delta_target = x1_target[:, 1:] - x1_target[:, :-1]
                    semantic_delta_loss = F.l1_loss(delta_hat, delta_target)
                else:
                    semantic_delta_loss = semantic_cos_loss.new_zeros(())

                semantic_anchor_loss = semantic_cos_loss + self.cfg.get(
                    'semantic_anchor_delta_weight', 0.5
                ) * semantic_delta_loss

        if (
            self.cfg.get('use_global_grag_cosine_loss', False)
            and grag_f is not None
            and getattr(self.network.module, 'use_decoupled_grag', False)
            and getattr(self.network.module, 'global_rag_cond', False)
        ):
            gen_global = inter_feat_proj.mean(dim=1)
            with torch.no_grad():
                if getattr(self.network.module, 'grag_extractor_type', 'mean') == 'qformer':
                    target_global = self.network.module.grag_qformer(grag_f).mean(dim=1)
                else:
                    target_global = self.network.module.rag_input_proj(grag_f).mean(dim=1)

            if target_global.shape[-1] != gen_global.shape[-1]:
                if target_global.shape[-1] > gen_global.shape[-1]:
                    target_global = target_global[:, :gen_global.shape[-1]]
                else:
                    target_global = F.pad(target_global, (0, gen_global.shape[-1] - target_global.shape[-1]))

            global_grag_cosine_loss = 1.0 - F.cosine_similarity(
                gen_global, target_global.detach(), dim=-1
            ).mean()

        loss = self.fm.loss(pred_v, x0, x1)
        mean_loss = loss.mean()

        repa_weight = self.cfg.get('repa_weight', 0.1)
        total_loss = mean_loss + repa_weight * repa_loss
        total_loss = total_loss + self.cfg.get('vae_latent_style_weight', 0.02) * vae_style_loss
        total_loss = total_loss + self.cfg.get('semantic_anchor_weight', 0.02) * semantic_anchor_loss
        total_loss = total_loss + self.cfg.get('global_grag_cosine_weight', 0.02) * global_grag_cosine_loss

        return x1, loss, total_loss, t, repa_loss, vae_style_loss, semantic_anchor_loss, global_grag_cosine_loss

    def val_fn(
        self,
        clip_f: torch.Tensor,
        sync_f: torch.Tensor,
        text_f: torch.Tensor,
        rag_f: torch.Tensor,
        x1: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bs = x1.shape[0]  # batch_size * seq_len * num_channels
        # normalize the latents
        x1 = self.network.module.normalize(x1)
        t = log_normal_sample(x1,
                              generator=self.rng,
                              m=self.log_normal_sampling_mean,
                              s=self.log_normal_sampling_scale)
        x0, x1, xt, (clip_f, sync_f, text_f, rag_f) = self.fm.get_x0_xt_c(x1,
                                                                   t,
                                                                   Cs=[clip_f, sync_f, text_f, rag_f],
                                                                   generator=self.rng)

        if self.inpainting:
            mask = self._generate_inpainting_mask(x1)
            masked_x1 = x1 * mask
            xt_in = torch.cat([xt, masked_x1, mask], dim=-1)
        else:
            xt_in = xt

        # classifier-free training
        samples = torch.rand(bs, device=x1.device, generator=self.rng)
        # null mask is for when a video is provided but we decided to ignore it
        null_video = (samples < self.null_condition_probability)
        # complete mask is for when a video is not provided or we decided to ignore it
        clip_f[null_video] = self.network.module.empty_clip_feat
        sync_f[null_video] = self.network.module.empty_sync_feat

        samples = torch.rand(bs, device=x1.device, generator=self.rng)
        null_text = (samples < self.null_condition_probability)
        text_f[null_text] = self.network.module.empty_string_feat

        pred_v = self.network(xt_in, clip_f, sync_f, text_f, t, rag_f)

        loss = self.fm.loss(pred_v, x0, x1)
        mean_loss = loss.mean()
        return loss, mean_loss, t

    def train_pass(self, data, it: int = 0):

        if not self.for_training:
            raise ValueError('train_pass() should not be called when not training.')

        self.enter_train()
        with torch.amp.autocast('cuda', enabled=self.use_amp, dtype=torch.bfloat16):
            clip_f = data['clip_features'].cuda(non_blocking=True)
            sync_f = data['sync_features'].cuda(non_blocking=True)
            text_f = data['text_features'].cuda(non_blocking=True)
            video_exist = data['video_exist'].cuda(non_blocking=True)
            text_exist = data['text_exist'].cuda(non_blocking=True)
            a_mean = data['a_mean'].cuda(non_blocking=True)
            a_std = data['a_std'].cuda(non_blocking=True)
            rag_f = data['rag_audio_emb'].cuda(non_blocking=True)
            beats_f = data.get('beats_features', None)
            rag_vae_f = data.get('rag_vae_emb', None)
            grag_f = data.get('grag_audio_emb', None)

            if beats_f is not None:
                beats_f = beats_f.cuda(non_blocking=True)
            if rag_vae_f is not None:
                rag_vae_f = rag_vae_f.cuda(non_blocking=True)
            if grag_f is not None:
                grag_f = grag_f.cuda(non_blocking=True)

            # these masks are for non-existent data; masking for CFG training is in train_fn
            clip_f[~video_exist] = self.network.module.empty_clip_feat
            sync_f[~video_exist] = self.network.module.empty_sync_feat
            text_f[~text_exist] = self.network.module.empty_string_feat

            self.log.data_timer.end()
            if it % self.log_extra_interval == 0:
                unmasked_clip_f = clip_f.clone()
                unmasked_sync_f = sync_f.clone()
                unmasked_text_f = text_f.clone()
                unmasked_rag_f = rag_f.clone()
                unmasked_rag_vae_f = rag_vae_f.clone() if rag_vae_f is not None else None
                unmasked_grag_f = grag_f.clone() if grag_f is not None else None

            x1, loss, mean_loss, t, repa_loss, vae_style_loss, semantic_anchor_loss, global_grag_cosine_loss = self.train_fn(
                clip_f, sync_f, text_f, a_mean, a_std, rag_f, beats_f, rag_vae_f, grag_f
            )

            log_dict = {'loss': mean_loss}
            if beats_f is not None:
                log_dict['repa_loss'] = repa_loss
            if self.cfg.get('use_vae_latent_style_loss', False):
                log_dict['vae_style_loss'] = vae_style_loss
            if self.cfg.get('use_semantic_anchor_loss', False):
                log_dict['semantic_anchor_loss'] = semantic_anchor_loss
            if self.cfg.get('use_global_grag_cosine_loss', False):
                log_dict['global_grag_cosine_loss'] = global_grag_cosine_loss
            self.train_integrator.add_dict(log_dict)

        if it % self.log_text_interval == 0 and it != 0:
            self.train_integrator.add_scalar('lr', self.scheduler.get_last_lr()[0])
            self.train_integrator.add_binned_tensor('binned_loss', loss, t)
            self.train_integrator.finalize('train', it)
            self.train_integrator.reset_except_hooks()

        # Backward pass
        self.optimizer.zero_grad(set_to_none=True)
        if self.enable_grad_scaler:
            self.scaler.scale(mean_loss).backward()
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.network.parameters(),
                                                       self.clip_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            mean_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.network.parameters(),
                                                       self.clip_grad_norm)
            self.optimizer.step()

        if self.ema is not None and it >= self.ema_start:
            self.ema.update()
        self.scheduler.step()
        self.integrator.add_scalar('grad_norm', grad_norm)

        self.enter_val()
        with torch.amp.autocast('cuda', enabled=self.use_amp,
                                dtype=torch.bfloat16), torch.inference_mode():
            try:
                if it % self.log_extra_interval == 0:
                    # save GT audio
                    # unnormalize the latents
                    x1 = self.network.module.unnormalize(x1[0:1])
                    mel = self.features.decode(x1)
                    audio = self.features.vocode(mel).cpu()[0]  # 1 * num_samples
                    self.log.log_spectrogram('train', f'spec-gt-r{local_rank}', mel.cpu()[0], it)
                    self.log.log_audio('train',
                                       f'audio-gt-r{local_rank}',
                                       audio,
                                       it,
                                       sample_rate=self.sample_rate)

                    # save audio from sampling
                    x0 = torch.empty_like(x1[0:1]).normal_(generator=self.rng)
                    clip_f = unmasked_clip_f[0:1]
                    sync_f = unmasked_sync_f[0:1]
                    text_f = unmasked_text_f[0:1]
                    rag_f = unmasked_rag_f[0:1]
                    rag_vae_f_log = unmasked_rag_vae_f[0:1] if unmasked_rag_vae_f is not None else None
                    grag_f_log = unmasked_grag_f[0:1] if unmasked_grag_f is not None else None
                    conditions = self.network.module.preprocess_conditions(
                        clip_f, sync_f, text_f, rag_f, rag_vae_f_log, grag_f=grag_f_log
                    )
                    empty_conditions = self.network.module.get_empty_conditions(x0.shape[0])

                    if self.inpainting:
                        mask = self._generate_inpainting_mask(x1[0:1])
                        masked_x1 = x1[0:1] * mask
                        def cfg_ode_wrapper(t, x):
                            x_in = torch.cat([x, masked_x1, mask], dim=-1)
                            return self.network.module.ode_wrapper(t, x_in, conditions, empty_conditions, self.cfg_strength)
                    else:
                        cfg_ode_wrapper = lambda t, x: self.network.module.ode_wrapper(
                            t, x, conditions, empty_conditions, self.cfg_strength)

                    x1_hat = self.fm.to_data(cfg_ode_wrapper, x0)
                    x1_hat = self.network.module.unnormalize(x1_hat)
                    mel = self.features.decode(x1_hat)
                    audio = self.features.vocode(mel).cpu()[0]
                    self.log.log_spectrogram('train', f'spec-r{local_rank}', mel.cpu()[0], it)
                    self.log.log_audio('train',
                                       f'audio-r{local_rank}',
                                       audio,
                                       it,
                                       sample_rate=self.sample_rate)
            except Exception as e:
                self.log.warning(f'Error in extra logging: {e}')
                if self.cfg.debug:
                    raise

        # Save network weights and checkpoint if needed
        save_copy = it in self.save_copy_iterations

        if (it % self.save_weights_interval == 0 and it != 0) or save_copy:
            self.save_weights(it)

        if it % self.save_checkpoint_interval == 0 and it != 0:
            self.save_checkpoint(it, save_copy=save_copy)

        self.log.data_timer.start()

    @torch.inference_mode()
    def validation_pass(self, data, it: int = 0):
        self.enter_val()
        with torch.amp.autocast('cuda', enabled=self.use_amp, dtype=torch.bfloat16):
            clip_f = data['clip_features'].cuda(non_blocking=True)
            sync_f = data['sync_features'].cuda(non_blocking=True)
            text_f = data['text_features'].cuda(non_blocking=True)
            video_exist = data['video_exist'].cuda(non_blocking=True)
            text_exist = data['text_exist'].cuda(non_blocking=True)
            a_mean = data['a_mean'].cuda(non_blocking=True)
            a_std = data['a_std'].cuda(non_blocking=True)
            rag_f = data['rag_feature'].cuda(non_blocking=True)

            clip_f[~video_exist] = self.network.module.empty_clip_feat
            sync_f[~video_exist] = self.network.module.empty_sync_feat
            text_f[~text_exist] = self.network.module.empty_string_feat
            a_randn = torch.empty_like(a_mean).normal_(generator=self.rng)
            x1 = a_mean + a_std * a_randn

            self.log.data_timer.end()
            loss, mean_loss, t = self.val_fn(clip_f.clone(), sync_f.clone(), text_f.clone(), rag_f.clone(), x1)

            self.val_integrator.add_binned_tensor('binned_loss', loss, t)
            self.val_integrator.add_dict({'loss': mean_loss})

        self.log.data_timer.start()

    @torch.inference_mode()
    def inference_pass(self,
                       data,
                       it: int,
                       data_cfg: DictConfig,
                       *,
                       save_eval: bool = True) -> Path:
        self.enter_val()
        with torch.amp.autocast('cuda', enabled=self.use_amp, dtype=torch.bfloat16):
            clip_f = data['clip_features'].cuda(non_blocking=True)
            sync_f = data['sync_features'].cuda(non_blocking=True)
            text_f = data['text_features'].cuda(non_blocking=True)
            video_exist = data['video_exist'].cuda(non_blocking=True)
            text_exist = data['text_exist'].cuda(non_blocking=True)
            a_mean = data['a_mean'].cuda(non_blocking=True)  # for the shape only
            rag_f = data['rag_feature'].cuda(non_blocking=True)

            clip_f[~video_exist] = self.network.module.empty_clip_feat
            sync_f[~video_exist] = self.network.module.empty_sync_feat
            text_f[~text_exist] = self.network.module.empty_string_feat

            # sample
            x0 = torch.empty_like(a_mean).normal_(generator=self.rng)
            conditions = self.network.module.preprocess_conditions(clip_f, sync_f, text_f, rag_f)
            empty_conditions = self.network.module.get_empty_conditions(x0.shape[0])

            if self.inpainting:
                # Use a fully generated mode (all masked) for general inference if mask is not provided
                mask = torch.zeros((x0.shape[0], x0.shape[1], 1), device=x0.device)
                masked_x1 = torch.zeros_like(x0)
                def cfg_ode_wrapper(t, x):
                    x_in = torch.cat([x, masked_x1, mask], dim=-1)
                    return self.network.module.ode_wrapper(t, x_in, conditions, empty_conditions, self.cfg_strength)
            else:
                cfg_ode_wrapper = lambda t, x: self.network.module.ode_wrapper(
                    t, x, conditions, empty_conditions, self.cfg_strength)

            x1_hat = self.fm.to_data(cfg_ode_wrapper, x0)
            x1_hat = self.network.module.unnormalize(x1_hat)
            mel = self.features.decode(x1_hat)
            audio = self.features.vocode(mel).cpu()
            for i in range(audio.shape[0]):
                video_id = data['id'][i]
                if (not self.for_training) and i == 0:
                    # save very few videos
                    self.test_video_joiner.join(video_id, f'{video_id}', audio[i].transpose(0, 1))

                if data_cfg.output_subdir is not None:
                    # validation
                    if save_eval:
                        iter_naming = f'{it:09d}'
                    else:
                        iter_naming = 'val-cache'
                    audio_dir = self.log.log_audio(iter_naming,
                                                   f'{video_id}',
                                                   audio[i],
                                                   it=None,
                                                   sample_rate=self.sample_rate,
                                                   subdir=Path(data_cfg.output_subdir))
                    if save_eval and i == 0:
                        self.val_video_joiner.join(video_id, f'{iter_naming}-{video_id}',
                                                   audio[i].transpose(0, 1))
                else:
                    # full test set, usually
                    audio_dir = self.log.log_audio(f'{data_cfg.tag}-sampled',
                                                   f'{video_id}',
                                                   audio[i],
                                                   it=None,
                                                   sample_rate=self.sample_rate)

        return Path(audio_dir)

    @torch.inference_mode()
    def eval(self, audio_dir: Path, it: int, data_cfg: DictConfig) -> dict[str, float]:
        from av_bench.evaluate import evaluate
        from av_bench.extract import extract

        with torch.amp.autocast('cuda', enabled=False):
            if local_rank == 0:
                extract(audio_path=audio_dir,
                        output_path=audio_dir / 'cache',
                        device='cuda',
                        batch_size=32,
                        audio_length=8)
                output_metrics = evaluate(gt_audio_cache=Path(data_cfg.gt_cache),
                                          pred_audio_cache=audio_dir / 'cache')
                for k, v in output_metrics.items():
                    # pad k to 10 characters
                    # pad v to 10 decimal places
                    self.log.log_scalar(f'{data_cfg.tag}/{k}', v, it)
                    self.log.info(f'{data_cfg.tag}/{k:<10}: {v:.10f}')
            else:
                output_metrics = None

        return output_metrics

    def save_weights(self, it, save_copy=False):
        if local_rank != 0:
            return

        os.makedirs(self.run_path, exist_ok=True)
        if save_copy:
            model_path = self.run_path / f'{self.exp_id}_{it}.pth'
            torch.save(self.network.module.state_dict(), model_path)
            self.log.info(f'Network weights saved to {model_path}.')

        # if last exists, move it to a shadow copy
        model_path = self.run_path / f'{self.exp_id}_last.pth'
        if model_path.exists():
            shadow_path = model_path.with_name(model_path.name.replace('last', 'shadow'))
            model_path.replace(shadow_path)
            self.log.info(f'Network weights shadowed to {shadow_path}.')

        torch.save(self.network.module.state_dict(), model_path)
        self.log.info(f'Network weights saved to {model_path}.')

    def save_checkpoint(self, it, save_copy=False):
        if local_rank != 0:
            return

        checkpoint = {
            'it': it,
            'weights': self.network.module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'ema': self.ema.state_dict() if self.ema is not None else None,
        }

        os.makedirs(self.run_path, exist_ok=True)
        if save_copy:
            model_path = self.run_path / f'{self.exp_id}_ckpt_{it}.pth'
            torch.save(checkpoint, model_path)
            self.log.info(f'Checkpoint saved to {model_path}.')

        # if ckpt_last exists, move it to a shadow copy
        model_path = self.run_path / f'{self.exp_id}_ckpt_last.pth'
        if model_path.exists():
            shadow_path = model_path.with_name(model_path.name.replace('last', 'shadow'))
            model_path.replace(shadow_path)  # moves the file
            self.log.info(f'Checkpoint shadowed to {shadow_path}.')

        torch.save(checkpoint, model_path)
        self.log.info(f'Checkpoint saved to {model_path}.')

    def get_latest_checkpoint_path(self):
        ckpt_path = self.run_path / f'{self.exp_id}_ckpt_last.pth'
        if not ckpt_path.exists():
            info_if_rank_zero(self.log, f'No checkpoint found at {ckpt_path}.')
            return None
        return ckpt_path

    def get_latest_weight_path(self):
        weight_path = self.run_path / f'{self.exp_id}_last.pth'
        if not weight_path.exists():
            self.log.info(f'No weight found at {weight_path}.')
            return None
        return weight_path

    def get_final_ema_weight_path(self):
        weight_path = self.run_path / f'{self.exp_id}_ema_final.pth'
        if not weight_path.exists():
            self.log.info(f'No weight found at {weight_path}.')
            return None
        return weight_path

    def load_checkpoint(self, path):
        # This method loads everything and should be used to resume training
        map_location = 'cuda:%d' % local_rank
        checkpoint = torch.load(path, map_location={'cuda:0': map_location}, weights_only=True)

        it = checkpoint['it']
        weights = checkpoint['weights']
        optimizer = checkpoint['optimizer']
        scheduler = checkpoint['scheduler']
        if self.ema is not None:
            self.ema.load_state_dict(checkpoint['ema'])
            self.log.info(f'EMA states loaded from step {self.ema.step}')

        map_location = 'cuda:%d' % local_rank
        self.network.module.load_state_dict(weights, strict=False)
        self.optimizer.load_state_dict(optimizer)
        self.scheduler.load_state_dict(scheduler)

        self.log.info(f'Global iteration {it} loaded.')
        self.log.info('Network weights, optimizer states, and scheduler states loaded.')

        return it

    def load_weights_in_memory(self, src_dict):
        self.network.module.load_weights(src_dict)
        self.log.info('Network weights loaded from memory.')

    def load_weights(self, path):
        # This method loads only the network weight and should be used to load a pretrained model
        map_location = 'cuda:%d' % local_rank
        src_dict = torch.load(path, map_location={'cuda:0': map_location}, weights_only=True)
        if "weights" in src_dict and isinstance(src_dict["weights"], dict):
            src_dict = src_dict["weights"]

        self.log.info(f'Importing network weights from {path}...')
        self.load_weights_in_memory(src_dict)

    def weights(self):
        return self.network.module.state_dict()

    def enter_train(self):
        self.integrator = self.train_integrator
        self.network.train()
        return self

    def enter_val(self):
        self.network.eval()
        return self
