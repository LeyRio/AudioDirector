# Copyright 2024 Sony Research Inc.
# Modifications Copyright 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import logging
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from vividaudio.ext.rotary_embeddings import compute_rope_rotations
from vividaudio.model.embeddings import TimestepEmbedder
from vividaudio.model.low_level import MLP, ChannelLastConv1d, ConvMLP
from vividaudio.model.transformer_layers import (FinalBlock, JointBlock, MMDitSingleBlock)

log = logging.getLogger()


@dataclass
class PreprocessedConditions:
    clip_f: torch.Tensor
    sync_f: torch.Tensor
    text_f: torch.Tensor
    clip_f_c: torch.Tensor
    text_f_c: torch.Tensor
    rag_f: torch.Tensor
    rag_f_c: torch.Tensor = None
    # Per-event grag style-vecs (B, K, D). Populated only when per-event grag
    # is active (see MMAudio.per_event_grag_stack), else None.
    rag_f_c_per_event: torch.Tensor = None
    # Slot mask (B, K, N_latent) used to blend the per-event style vecs onto
    # the latent time axis inside predict_flow. Only present when
    # rag_f_c_per_event is set.
    slot_mask: torch.Tensor = None


class AudioQFormer(nn.Module):
    def __init__(self, dim_in=768, dim_q=256, n_q=8, n_layers=4):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(n_q, dim_q))
        self.proj_kv = nn.Linear(dim_in, dim_q)

        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=dim_q,
                nhead=8,
                batch_first=True
            ) for _ in range(n_layers)
        ])

    def forward(self, feats):  # feats: (B, T, 768)
        B = feats.size(0)
        q = self.queries.unsqueeze(0).expand(B, -1, -1)
        kv = self.proj_kv(feats)

        for layer in self.layers:
            q = layer(q, kv)

        return q   # (B, Nq, Dq)


class SlotAttention(nn.Module):
    def __init__(self, dim, num_slots=1, iters=3):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters

        self.slots_mu = nn.Parameter(torch.randn(1, num_slots, dim))
        self.slots_logsigma = nn.Parameter(torch.zeros(1, num_slots, dim))

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        self.gru = nn.GRUCell(dim, dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, inputs):
        B, T, D = inputs.shape
        slots = self.slots_mu + torch.exp(self.slots_logsigma) * torch.randn_like(self.slots_mu)
        slots = slots.expand(B, -1, -1)

        k, v = self.to_k(inputs), self.to_v(inputs)

        for _ in range(self.iters):
            q = self.to_q(slots)
            attn = torch.softmax(
                torch.einsum('bid,bjd->bij', q, k),
                dim=1
            )
            updates = torch.einsum('bij,bjd->bid', attn, v)
            slots = self.gru(
                updates.reshape(-1, D),
                slots.reshape(-1, D)
            ).view(B, -1, D)
            slots = slots + self.mlp(slots)

        return slots


# Partially from https://github.com/facebookresearch/DiT
class MMAudio(nn.Module):

    def __init__(self,
                 *,
                 latent_dim: int,
                 clip_dim: int,
                 sync_dim: int,
                 text_dim: int,
                 rag_dim: int,
                 repa_dim: int,
                 hidden_dim: int,
                 depth: int,
                 fused_depth: int,
                 num_heads: int,
                 mlp_ratio: float = 4.0,
                 latent_seq_len: int,
                 clip_seq_len: int,
                 sync_seq_len: int,
                 text_seq_len: int = 77,
                 latent_mean: Optional[torch.Tensor] = None,
                 latent_std: Optional[torch.Tensor] = None,
                 empty_string_feat: Optional[torch.Tensor] = None,
                 inpainting: bool = False,
                 sep_fgc: bool = True,
                 joint_fgc: bool = False,
                 delay_fusion: bool = False,
                 interleaved_rope: bool = False,
                 fusion_fgc_clip: bool = False,
                 fgc_pt: int = 10,
                 qformer: bool = False,
                 v2: bool = False,
                 sync_rope: bool = False,
                 cross_attn_delay: bool = False,
                 local_window_size: int = 0,
                 hybrid_cross_attn: bool = False,
                 zero_init_gate: bool = True,
                 mask_rag_ratio: float = 0.0,
                 mask_rag_block_size: int = 10,
                 global_rag_cond: bool = False,
                 global_emb_type: str = 'beats',
                 bypass_empty_inpainting: bool = False,
                 use_decoupled_grag: bool = False,
                 grag_extractor_type: str = 'mean',
                 grag_qformer_n_q: int = 8,
                 grag_qformer_n_layers: int = 2,
                 **kwargs) -> None:
        super().__init__()

        self.v2 = v2
        self.latent_dim = latent_dim
        self._latent_seq_len = latent_seq_len
        self._fgc_seq_len = int(8 * fgc_pt)  # <--- 加一个 int()
        self.fgc_pt = fgc_pt
        self._clip_seq_len = clip_seq_len
        self._sync_seq_len = sync_seq_len
        self._text_seq_len = text_seq_len
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.sep_fgc = sep_fgc
        self.inpainting = inpainting
        self.joint_fgc = joint_fgc
        self.delay_fusion = delay_fusion
        self.interleaved_rope = interleaved_rope
        self.fusion_fgc_clip = fusion_fgc_clip
        self.sync_rope = sync_rope
        self.cross_attn_delay = cross_attn_delay
        self.local_window_size = local_window_size
        self.hybrid_cross_attn = hybrid_cross_attn
        self.zero_init_gate = zero_init_gate
        self.mask_rag_ratio = mask_rag_ratio
        self.mask_rag_block_size = mask_rag_block_size
        self.global_rag_cond = global_rag_cond
        self.global_emb_type = global_emb_type
        self.bypass_empty_inpainting = bypass_empty_inpainting
        self.use_decoupled_grag = use_decoupled_grag
        self.grag_extractor_type = grag_extractor_type
        # Runtime-only inference knob: scales the grag global condition before
        # it feeds AdaLN. Set via cfg.grag_scale in eval configs. 1.0 keeps the
        # trained behaviour, 0.0 == turning off global rag entirely.
        self.grag_scale = 1.0
        # Runtime-only inference knob: scales the ``inpainting_mlp`` residual
        # that anchors latent tokens to the inpainted ref latent. 1.0 keeps
        # the trained behaviour, 0.0 == turning off the inpainting anchor
        # entirely (equivalent to using audio_input_proj on xt only). Values
        # in between let you trade "stick to ref" (high beta -> high timbre
        # similarity) for "video-time alignment" (low beta).
        self.inpainting_scale = 1.0
        # Optional per-slot inpainting scaling. When set, this (B, N, 1)
        # tensor supersedes ``inpainting_scale`` per latent token. Populated
        # by batch_eval_rag when ``--per-slot-inpainting-scale`` is passed.
        # None => fall back to the scalar above. Cleared right after
        # predict_flow returns.
        self.inpainting_scale_map = None
        # Optional t-dependent gate on the inpainting residual (F1: time
        # scheduled anchor). When set, this callable takes the current flow-
        # matching time ``t`` (float tensor in [0, 1]; 1 = pure noise, 0 =
        # clean) and returns a scalar (or (B,) tensor broadcastable to (B, 1,
        # 1)) that is multiplied on top of inpainting_scale (or per-slot map).
        # Semantics: high early-t weight anchors the "coarse timbre" phase
        # then releases the "fine-grained sync" phase to the video branch.
        # None => no schedule (default trained behaviour).
        self.inpainting_scale_fn = None
        # Optional per-channel β on the inpainting residual (thought-2:
        # channel-wise β). Expected shape (D,) or (1, 1, D) with D ==
        # latent_dim. When set, it is broadcast-multiplied INSIDE the residual,
        # stacking on top of scalar / per-slot / scheduled β. Defaults to None
        # so the trained residual is untouched.
        self.inpainting_channel_scale = None
        # Runtime-only "per-event grag" plumbing. When None (default) the model
        # behaves EXACTLY as trained: a single style vec broadcast to all
        # latent steps. When set from batch_eval_rag, preprocess_conditions
        # will run K parallel grag paths and predict_flow will interpolate
        # them onto the latent time axis with a smooth slot mask, so each slot
        # gets its own timbre. Only touched during inference.
        # Shape contract:
        #   per_event_grag_stack : (B, K, T_grag_native, C_grag) or None
        #   per_event_slot_mask  : (B, K, N_latent) or None
        # Both are set to None right after predict_flow returns.
        self.per_event_grag_stack = None
        self.per_event_slot_mask = None
        if self.joint_fgc:
            self.sep_fgc = True
        # if not self.sep_fgc or self.joint_fgc:
        #     self.fusion = nn.MultiheadAttention(embed_dim=896, num_heads=8, batch_first=True)

        if v2:
            self.audio_input_proj = nn.Sequential(
                ChannelLastConv1d(latent_dim, hidden_dim, kernel_size=7, padding=3),
                nn.SiLU(),
                ConvMLP(hidden_dim, hidden_dim * 4, kernel_size=7, padding=3),
            )
            if self.inpainting:
                self.inpainting_mlp = nn.Sequential(
                    nn.Linear(latent_dim * 2 + 1, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, hidden_dim)
                )

            self.clip_input_proj = nn.Sequential(
                nn.Linear(clip_dim, hidden_dim),
                nn.SiLU(),
                ConvMLP(hidden_dim, hidden_dim * 4, kernel_size=3, padding=1),
            )

            self.sync_input_proj = nn.Sequential(
                ChannelLastConv1d(sync_dim, hidden_dim, kernel_size=7, padding=3),
                nn.SiLU(),
                ConvMLP(hidden_dim, hidden_dim * 4, kernel_size=3, padding=1),
            )

            self.text_input_proj = nn.Sequential(
                nn.Linear(text_dim, hidden_dim),
                nn.SiLU(),
                MLP(hidden_dim, hidden_dim * 4),
            )


            self.rag_input_proj = nn.Sequential(
                nn.Linear(rag_dim, hidden_dim),
                nn.SiLU(),
                MLP(hidden_dim, hidden_dim * 4),
            )

        else:
            self.audio_input_proj = nn.Sequential(
                ChannelLastConv1d(latent_dim, hidden_dim, kernel_size=7, padding=3),
                nn.SELU(),
                ConvMLP(hidden_dim, hidden_dim * 4, kernel_size=7, padding=3),
            )
            if self.inpainting:
                self.inpainting_mlp = nn.Sequential(
                    nn.Linear(latent_dim * 2 + 1, hidden_dim),
                    nn.SELU(),
                    nn.Linear(hidden_dim, hidden_dim)
                )

            self.clip_input_proj = nn.Sequential(
                nn.Linear(clip_dim, hidden_dim),
                ConvMLP(hidden_dim, hidden_dim * 4, kernel_size=3, padding=1),
            )

            self.sync_input_proj = nn.Sequential(
                ChannelLastConv1d(sync_dim, hidden_dim, kernel_size=7, padding=3),
                nn.SELU(),
                ConvMLP(hidden_dim, hidden_dim * 4, kernel_size=3, padding=1),
            )

            self.text_input_proj = nn.Sequential(
                nn.Linear(text_dim, hidden_dim),
                MLP(hidden_dim, hidden_dim * 4),
            )


            self.rag_input_proj = nn.Sequential(
                nn.Linear(rag_dim, hidden_dim),
                MLP(hidden_dim, hidden_dim * 4),
            )

        self.qformer = qformer
        if qformer:
            self.audio_qformer = AudioQFormer(dim_in=768, dim_q=768, n_q=8, n_layers=4)

        self.clip_cond_proj = nn.Linear(hidden_dim, hidden_dim)
        self.text_cond_proj = nn.Linear(hidden_dim, hidden_dim)
        if self.global_rag_cond:
            if self.global_emb_type in ['beats', 'both']:
                self.rag_cond_proj = nn.Linear(hidden_dim, hidden_dim)
            if self.global_emb_type in ['vae', 'both']:
                self.rag_vae_cond_proj = nn.Linear(latent_dim, hidden_dim)
            if self.use_decoupled_grag and self.grag_extractor_type == 'qformer':
                self.grag_qformer = AudioQFormer(
                    dim_in=rag_dim,
                    dim_q=hidden_dim,
                    n_q=grag_qformer_n_q,
                    n_layers=grag_qformer_n_layers,
                )
        self.global_cond_mlp = MLP(hidden_dim, hidden_dim * 4)
        self.inpainting = inpainting
        # each synchformer output segment has 8 feature frames
        self.sync_pos_emb = nn.Parameter(torch.zeros((1, 1, 8, sync_dim)))

        self.final_layer = FinalBlock(hidden_dim, latent_dim)

        if v2:
            self.t_embed = TimestepEmbedder(hidden_dim,
                                            frequency_embedding_size=hidden_dim,
                                            max_period=1)
        else:
            self.t_embed = TimestepEmbedder(hidden_dim,
                                            frequency_embedding_size=256,
                                            max_period=10000)
        self.joint_blocks = nn.ModuleList([
            JointBlock(hidden_dim,
                       num_heads,
                       mlp_ratio=mlp_ratio,
                       sep_fgc=self.sep_fgc,
                       delay_fusion=self.delay_fusion,
                       joint_fgc=self.joint_fgc,
                       interleaved_rope = self.interleaved_rope,
                       fusion_fgc_clip=self.fusion_fgc_clip,
                       sync_rope=self.sync_rope,
                       cross_attn_delay=self.cross_attn_delay,
                       local_window_size=self.local_window_size,
                       hybrid_cross_attn=self.hybrid_cross_attn,
                       zero_init_gate=self.zero_init_gate,
                       pre_only=(i == depth - fused_depth - 1)) for i in range(depth - fused_depth)
        ])

        if self.inpainting:
            self.input_proj = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.SiLU(),
                MLP(hidden_dim, hidden_dim * 4),
            )

        self.fused_blocks = nn.ModuleList([
            MMDitSingleBlock(hidden_dim, num_heads, mlp_ratio=mlp_ratio, kernel_size=3, padding=1)
            for i in range(fused_depth)
        ])

        if latent_mean is None:
            # these values are not meant to be used
            # if you don't provide mean/std here, we should load them later from a checkpoint
            assert latent_std is None
            latent_mean = torch.ones(latent_dim).view(1, 1, -1).fill_(float('nan'))
            latent_std = torch.ones(latent_dim).view(1, 1, -1).fill_(float('nan'))
        else:
            assert latent_std is not None
            assert latent_mean.numel() == latent_dim, f'{latent_mean.numel()=} != {latent_dim=}'
        if empty_string_feat is None:
            empty_string_feat = torch.zeros((text_seq_len, text_dim))
        self.latent_mean = nn.Parameter(latent_mean.view(1, 1, -1), requires_grad=False)
        self.latent_std = nn.Parameter(latent_std.view(1, 1, -1), requires_grad=False)

        self.empty_string_feat = nn.Parameter(empty_string_feat, requires_grad=False)
        self.empty_clip_feat = nn.Parameter(torch.zeros(1, clip_dim), requires_grad=True)
        self.empty_sync_feat = nn.Parameter(torch.zeros(1, sync_dim), requires_grad=True)
        self.empty_rag_feat = nn.Parameter(torch.zeros(1, rag_dim), requires_grad=False)

        self.repa_proj = nn.Linear(hidden_dim, repa_dim)

        # print('initialize weights')
        # self.initialize_weights()
        self.initialize_rotations()

    # def build_rope_for_interleaved_audio_visual(self, total_len):
    #     target_ndim = 1  # n-d RoPE
    #     rope_sizes = [total_len]
    #     head_dim = self.hidden_dim // self.num_heads
    #     rope_dim_list = [head_dim // target_ndim for _ in range(target_ndim)]
    #     assert sum(rope_dim_list) == head_dim, "sum(rope_dim_list) should equal to head_dim of attention layer"
    #     freqs_cos, freqs_sin = get_nd_rotary_pos_embed(
    #         rope_dim_list=rope_dim_list,
    #         start=rope_sizes,
    #         theta=10000,
    #         use_real=True,
    #         theta_rescale_factor=1.0,
    #     )
    #     return freqs_cos, freqs_sin

    def build_rope_for_interleaved_audio_visual(self, total_len, device='cpu'):
        head_dim = self.hidden_dim // self.num_heads

        # 调 compute_rope_rotations
        rot = compute_rope_rotations(
            length=total_len,
            dim=head_dim,
            theta=10000,
            freq_scaling=1.0,
            device=device
        )
        return rot  # shape: [1, total_len, head_dim/2, 2, 2]


    def initialize_rotations(self):
        base_freq = 1.0
        latent_rot = compute_rope_rotations(self._latent_seq_len,
                                            self.hidden_dim // self.num_heads,
                                            10000,
                                            freq_scaling=base_freq,
                                            device=self.device)
        clip_rot = compute_rope_rotations(self._clip_seq_len,
                                          self.hidden_dim // self.num_heads,
                                          10000,
                                          freq_scaling=base_freq * self._latent_seq_len /
                                          self._clip_seq_len,
                                          device=self.device)


        self.latent_rot = nn.Buffer(latent_rot, persistent=False)
        self.clip_rot = nn.Buffer(clip_rot, persistent=False)

        fgc_rot = compute_rope_rotations(self._fgc_seq_len,
                                        self.hidden_dim // self.num_heads,
                                        10000,
                                        freq_scaling=base_freq * self._latent_seq_len / self._fgc_seq_len,
                                        device=self.device)
        self.fgc_rot = nn.Buffer(fgc_rot, persistent=False)

        inter_rot = self.build_rope_for_interleaved_audio_visual(self._fgc_seq_len * 2)
        self.inter_rot = nn.Buffer(inter_rot, persistent=False)
        inter_fusion_rot = self.build_rope_for_interleaved_audio_visual(self._fgc_seq_len * 2)
        self.inter_fusion_rot = nn.Buffer(inter_fusion_rot, persistent=False)


    def update_seq_lengths(self, latent_seq_len: int, clip_seq_len: int, sync_seq_len: int) -> None:
        self._latent_seq_len = latent_seq_len
        self._clip_seq_len = clip_seq_len
        self._sync_seq_len = sync_seq_len
        self.initialize_rotations()

    def initialize_weights(self):

        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        if self.inpainting:
            nn.init.constant_(self.inpainting_mlp[-1].weight, 0)
            if self.inpainting_mlp[-1].bias is not None:
                nn.init.constant_(self.inpainting_mlp[-1].bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embed.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.joint_blocks:
            nn.init.constant_(block.latent_block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.latent_block.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(block.clip_block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.clip_block.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(block.text_block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.text_block.adaLN_modulation[-1].bias, 0)
        for block in self.fused_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.conv.weight, 0)
        nn.init.constant_(self.final_layer.conv.bias, 0)

        if self.global_rag_cond:
            if self.global_emb_type in ['beats', 'both']:
                nn.init.constant_(self.rag_cond_proj.weight, 0)
                nn.init.constant_(self.rag_cond_proj.bias, 0)
            if self.global_emb_type in ['vae', 'both']:
                nn.init.constant_(self.rag_vae_cond_proj.weight, 0)
                nn.init.constant_(self.rag_vae_cond_proj.bias, 0)

        # empty string feat shall be initialized by a CLIP encoder
        nn.init.constant_(self.sync_pos_emb, 0)
        nn.init.constant_(self.empty_clip_feat, 0)
        nn.init.constant_(self.empty_sync_feat, 0)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        # return (x - self.latent_mean) / self.latent_std
        return x.sub_(self.latent_mean).div_(self.latent_std)

    def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        # return x * self.latent_std + self.latent_mean
        return x.mul_(self.latent_std).add_(self.latent_mean)

    def preprocess_conditions(self, clip_f: torch.Tensor, sync_f: torch.Tensor,
                              text_f: torch.Tensor, rag_f: torch.Tensor, rag_vae_f: torch.Tensor = None,
                              grag_f: torch.Tensor = None) -> PreprocessedConditions:
        """
        cache computations that do not depend on the latent/time step
        i.e., the features are reused over steps during inference
        """
        assert clip_f.shape[1] == self._clip_seq_len, f'{clip_f.shape=} {self._clip_seq_len=}'
        assert sync_f.shape[1] == self._sync_seq_len, f'{sync_f.shape=} {self._sync_seq_len=}'
        assert text_f.shape[1] == self._text_seq_len, f'{text_f.shape=} {self._text_seq_len=}'

        bs = clip_f.shape[0]

        # B * num_segments (24) * 8 * 768
        num_sync_segments = self._sync_seq_len // 8
        sync_f = sync_f.view(bs, num_sync_segments, 8, -1) + self.sync_pos_emb
        sync_f = sync_f.flatten(1, 2)  # (B, VN, D)

        # extend vf to match x
        clip_f = self.clip_input_proj(clip_f)  # (B, VN, D)
        sync_f = self.sync_input_proj(sync_f)  # (B, VN, D)
        text_f = self.text_input_proj(text_f)  # (B, VN, D)
        rag_f = self.rag_input_proj(rag_f)  # (B, VN, D)

        # upsample the sync features to match the audio
        sync_f = sync_f.transpose(1, 2)  # (B, D, VN)
        sync_f = F.interpolate(sync_f, size=self._latent_seq_len, mode='nearest-exact')
        sync_f = sync_f.transpose(1, 2)  # (B, N, D)

        # get conditional features from the clip side
        clip_f_c = self.clip_cond_proj(clip_f.mean(dim=1))  # (B, D)
        text_f_c = self.text_cond_proj(text_f.mean(dim=1))  # (B, D)

        if self.global_rag_cond:
            rag_f_c = None
            if self.use_decoupled_grag and grag_f is not None:
                if self.grag_extractor_type == 'qformer':
                    rag_f_c = self.rag_cond_proj(self.grag_qformer(grag_f).mean(dim=1))
                else:
                    grag_f_proj = self.rag_input_proj(grag_f)
                    rag_f_c = self.rag_cond_proj(grag_f_proj.mean(dim=1))
            elif self.global_emb_type in ['beats', 'both']:
                rag_f_c = self.rag_cond_proj(rag_f.mean(dim=1))  # (B, D)
            if self.global_emb_type in ['vae', 'both'] and rag_vae_f is not None:
                vae_c = self.rag_vae_cond_proj(rag_vae_f.mean(dim=1))
                rag_f_c = vae_c if rag_f_c is None else (rag_f_c + vae_c)
            if rag_f_c is not None and self.grag_scale != 1.0:
                rag_f_c = rag_f_c * self.grag_scale
        else:
            rag_f_c = None

        # ------------------------------------------------------------------
        # Optional per-event grag path. Only runs when
        #   MMAudio.per_event_grag_stack is set (by batch_eval_rag with
        #   ``per_event_grag=True``). Never triggers by default.
        #
        # Inputs (stashed on ``self`` to keep forward() signatures untouched):
        #   self.per_event_grag_stack : (B, K, T_grag, C_grag)
        #     -- one grag input per event (B口径: single ref clip padded/
        #        repeated to the same T_grag; other slots ignored).
        #   self.per_event_slot_mask  : (B, K, N_latent)
        #     -- soft mask blending per-event styles onto the latent axis.
        # Output:
        #   rag_f_c_per_event : (B, K, D)   used by predict_flow
        #   slot_mask         : (B, K, N)   kept alongside for blending
        # ``rag_f_c`` (the classic (B, D) vec) is preserved as the "mean-over-K
        # style" so that clip/text/fgc AdaLN branches see the same distribution
        # they were trained with.
        rag_f_c_per_event = None
        slot_mask = None
        pe_stack = self.per_event_grag_stack
        pe_mask = self.per_event_slot_mask
        if (
            pe_stack is not None
            and pe_mask is not None
            and self.global_rag_cond
            and self.use_decoupled_grag
            and self.grag_extractor_type == 'qformer'
        ):
            B, K, T_g, C_g = pe_stack.shape
            flat = pe_stack.reshape(B * K, T_g, C_g)
            rag_f_c_per_event = self.rag_cond_proj(
                self.grag_qformer(flat).mean(dim=1)
            ).reshape(B, K, -1)  # (B, K, D)
            if self.grag_scale != 1.0:
                rag_f_c_per_event = rag_f_c_per_event * self.grag_scale
            # Override the classic (B, D) rag_f_c so that non-latent AdaLN
            # branches use the SAME mean-of-events style as the training-time
            # single-composite path (whose grag input contained all events).
            rag_f_c = rag_f_c_per_event.mean(dim=1)
            slot_mask = pe_mask

        return PreprocessedConditions(clip_f=clip_f,
                                      sync_f=sync_f,
                                      text_f=text_f,
                                      rag_f=rag_f,
                                      clip_f_c=clip_f_c,
                                      text_f_c=text_f_c,
                                      rag_f_c=rag_f_c,
                                      rag_f_c_per_event=rag_f_c_per_event,
                                      slot_mask=slot_mask)

    def predict_flow(self, latent: torch.Tensor, t: torch.Tensor,
                     conditions: PreprocessedConditions, return_inter: bool = False) -> torch.Tensor:
        """
        for non-cacheable computations
        """
        if self.inpainting:
            xt = latent[..., :self.latent_dim]
            assert xt.shape[1] == self._latent_seq_len, f'{xt.shape=} {self._latent_seq_len=}'
            # Runtime scaling factor for the inpainting residual. Default 1.0
            # keeps the trained behaviour; smaller values weaken the ref
            # anchor (audio_input_proj(xt) alone survives at scale=0). A
            # per-slot map (B, N, 1) takes priority over the scalar when set.
            _inp_scale_map = getattr(self, 'inpainting_scale_map', None)
            if _inp_scale_map is not None:
                _inp_scale = _inp_scale_map.to(xt.dtype)
            else:
                _inp_scale = getattr(self, 'inpainting_scale', 1.0)
            # Optional t-scheduled gate on the anchor. Applied *on top of*
            # the scalar / per-slot scale so all three knobs stack.
            _inp_fn = getattr(self, 'inpainting_scale_fn', None)
            if _inp_fn is not None:
                _gate = _inp_fn(t)
                if isinstance(_gate, torch.Tensor):
                    while _gate.dim() < 3:
                        _gate = _gate.unsqueeze(-1)
                    _gate = _gate.to(xt.dtype)
                _inp_scale = _inp_scale * _gate
            # Optional per-channel β vector on the residual. When None, the
            # residual keeps its trained per-channel gain (== identity mult).
            _chan_scale = getattr(self, 'inpainting_channel_scale', None)
            if _chan_scale is not None:
                if not isinstance(_chan_scale, torch.Tensor):
                    _chan_scale = torch.as_tensor(_chan_scale, device=xt.device, dtype=xt.dtype)
                while _chan_scale.dim() < 3:
                    _chan_scale = _chan_scale.unsqueeze(0)
                _chan_scale = _chan_scale.to(xt.device, xt.dtype)
            # When the inpainting mask is entirely zero, optionally bypass the inpainting branch
            # so the model behaves closer to the non-inpainting path for ablations.
            if self.bypass_empty_inpainting:
                mask = latent[..., -1:]
                has_inpainting = (mask.amax(dim=1, keepdim=True) > 0).to(xt.dtype)
                _resid = self.inpainting_mlp(latent)
                if _chan_scale is not None:
                    _resid = _resid * _chan_scale
                latent = self.audio_input_proj(xt) + _inp_scale * has_inpainting * _resid
            else:
                # Apply audio_input_proj to xt (pretrained behavior) and add inpainting_mlp.
                _resid = self.inpainting_mlp(latent)
                if _chan_scale is not None:
                    _resid = _resid * _chan_scale
                latent = self.audio_input_proj(xt) + _inp_scale * _resid
        else:
            assert latent.shape[1] == self._latent_seq_len, f'{latent.shape=} {self._latent_seq_len=}'
            latent = self.audio_input_proj(latent)  # (B, N, D)

        clip_f = conditions.clip_f
        sync_f = conditions.sync_f
        text_f = conditions.text_f
        clip_f_c = conditions.clip_f_c
        text_f_c = conditions.text_f_c
        rag_f = conditions.rag_f
        rag_f_c = conditions.rag_f_c

        global_c_input = clip_f_c + text_f_c
        if self.global_rag_cond and rag_f_c is not None:
            global_c_input = global_c_input + rag_f_c

        global_c = self.global_cond_mlp(global_c_input)  # (B, D)

        global_c = self.t_embed(t).unsqueeze(1) + global_c.unsqueeze(1)  # (B, D)
        extended_c = global_c + sync_f

        # ------------------------------------------------------------------
        # Per-event grag → per-token extended_c override. Only activated when
        # preprocess_conditions received a per-event stack + slot mask. In
        # every other case ``extended_c`` stays the same (B, 1|N, D) tensor
        # the model was trained with.
        pe_style = conditions.rag_f_c_per_event      # (B, K, D) or None
        pe_mask = conditions.slot_mask               # (B, K, N) or None
        if pe_style is not None and pe_mask is not None and self.global_rag_cond:
            # Blend per-event styles onto the latent time axis.
            # style_at_t: (B, N, D)  =  sum_k mask[b,k,n] * style[b,k,:]
            style_at_t = torch.einsum('bkd,bkn->bnd', pe_style, pe_mask)
            # Recompute global_c on the (clip+text+per-token-style) base.
            base_ct = (clip_f_c + text_f_c).unsqueeze(1)  # (B, 1, D)
            global_c_t = self.global_cond_mlp(base_ct + style_at_t)  # (B, N, D)
            global_c_t = self.t_embed(t).unsqueeze(1) + global_c_t   # (B, N, D)
            # Only the latent AdaLN path receives the time-varying c; clip /
            # text / fgc branches keep the classic (B, D) ``global_c`` (which
            # already carries the mean-over-events style via rag_f_c above).
            extended_c = global_c_t + sync_f  # (B, N, D)

        for i, block in enumerate(self.joint_blocks):
            latent, clip_f, text_f, rag_f = block(latent, clip_f, text_f, rag_f, global_c, extended_c,
                                           self.latent_rot, self.clip_rot, self.fgc_rot, self.sep_fgc, self.delay_fusion, self.inter_rot, self.inter_fusion_rot)  # (B, N, D)
            if i == len(self.joint_blocks) // 2:
                inter_latent = latent

        for block in self.fused_blocks:
            latent = block(latent, extended_c, self.latent_rot)

        # should be extended_c; this is a minor implementation error #55
        flow = self.final_layer(latent, extended_c)  # (B, N, out_dim), remove t
        if return_inter:
            inter_latent_proj = self.repa_proj(inter_latent)
            return flow, inter_latent_proj
        return flow

    def forward(self, latent: torch.Tensor, clip_f: torch.Tensor, sync_f: torch.Tensor,
                text_f: torch.Tensor, t: torch.Tensor, rag_f: torch.Tensor, return_inter: bool = False,
                rag_vae_f: torch.Tensor = None, grag_f: torch.Tensor = None) -> torch.Tensor:
        """
        latent: (B, N, C)
        vf: (B, T, C_V)
        t: (B,)
        """
        conditions = self.preprocess_conditions(clip_f, sync_f, text_f, rag_f, rag_vae_f, grag_f=grag_f)
        return self.predict_flow(latent, t, conditions, return_inter=return_inter)

    def get_empty_string_sequence(self, bs: int) -> torch.Tensor:
        return self.empty_string_feat.unsqueeze(0).expand(bs, -1, -1)

    def get_empty_clip_sequence(self, bs: int) -> torch.Tensor:
        return self.empty_clip_feat.unsqueeze(0).expand(bs, self._clip_seq_len, -1)

    def get_empty_sync_sequence(self, bs: int) -> torch.Tensor:
        return self.empty_sync_feat.unsqueeze(0).expand(bs, self._sync_seq_len, -1)

    def get_empty_fgc_sequence(self, bs: int) -> torch.Tensor:
        return self.empty_rag_feat.unsqueeze(0).expand(bs, self._fgc_seq_len, -1)

    def get_empty_conditions(
            self,
            bs: int,
            *,
            negative_text_features: Optional[torch.Tensor] = None) -> PreprocessedConditions:
        if negative_text_features is not None:
            empty_text = negative_text_features
        else:
            empty_text = self.get_empty_string_sequence(1)

        empty_clip = self.get_empty_clip_sequence(1)
        empty_sync = self.get_empty_sync_sequence(1)
        empty_fgc = self.get_empty_fgc_sequence(1)

        conditions = self.preprocess_conditions(empty_clip, empty_sync, empty_text, empty_fgc)
        conditions.clip_f = conditions.clip_f.expand(bs, -1, -1)
        conditions.sync_f = conditions.sync_f.expand(bs, -1, -1)
        conditions.rag_f = conditions.rag_f.expand(bs, -1, -1)
        conditions.clip_f_c = conditions.clip_f_c.expand(bs, -1)
        if conditions.rag_f_c is not None:
            conditions.rag_f_c = conditions.rag_f_c.expand(bs, -1)
        if negative_text_features is None:
            conditions.text_f = conditions.text_f.expand(bs, -1, -1)
            conditions.text_f_c = conditions.text_f_c.expand(bs, -1)

        return conditions

    def ode_wrapper(self, t: torch.Tensor, latent: torch.Tensor, conditions: PreprocessedConditions,
                    empty_conditions: PreprocessedConditions, cfg_strength: float, neg_conditions: torch.Tensor = None, cfg_rag_weight: float = 0.5) -> torch.Tensor:

        t = t * torch.ones(len(latent), device=latent.device, dtype=latent.dtype)

        if cfg_strength < 1.0:
            return self.predict_flow(latent, t, conditions)
        else:
            # return (cfg_strength * self.predict_flow(latent, t, conditions) +
            #         (1 - cfg_strength) * self.predict_flow(latent, t, empty_conditions))
            flow_cond = self.predict_flow(latent, t, conditions)
            new_conditions = PreprocessedConditions(
                clip_f=empty_conditions.clip_f,    # 示例形状
                sync_f=empty_conditions.sync_f,
                text_f=empty_conditions.text_f,
                clip_f_c=empty_conditions.clip_f_c,
                text_f_c=empty_conditions.text_f_c,
                rag_f=conditions.rag_f,
            )
            # empty_conditions.rag_f = conditions.rag_f
            empty_conditions.rag_f = neg_conditions.rag_f
            flow_rag = self.predict_flow(latent, t, new_conditions) # 视频 + 文本 + 空rag
            flow_empty = self.predict_flow(latent, t, empty_conditions) # 空视频 + 空文本 + 静音rag
            # return cfg_strength * flow_cond - cfg_rag * flow_rag + (1 - cfg_strength + cfg_rag) * flow_empty
            # return cfg_strength * flow_cond + (1 - cfg_strength) * flow_empty
            return cfg_rag_weight * (cfg_strength * flow_cond + (1 - cfg_strength) * flow_rag) + (1.0 - cfg_rag_weight) * (cfg_strength * flow_cond + (1 - cfg_strength) * flow_empty)

    def load_weights(self, src_dict) -> None:
        if 't_embed.freqs' in src_dict:
            del src_dict['t_embed.freqs']
        if 'latent_rot' in src_dict:
            del src_dict['latent_rot']
        if 'clip_rot' in src_dict:
            del src_dict['clip_rot']

        self.load_state_dict(src_dict, strict=False)

    def load_weights_fgc(self, src_dict) -> None:
        if 't_embed.freqs' in src_dict:
            del src_dict['t_embed.freqs']
        if 'latent_rot' in src_dict:
            del src_dict['latent_rot']
        if 'clip_rot' in src_dict:
            del src_dict['clip_rot']

        self.load_state_dict(src_dict, strict=True)

    @property
    def device(self) -> torch.device:
        return self.latent_mean.device

    @property
    def latent_seq_len(self) -> int:
        return self._latent_seq_len

    @property
    def clip_seq_len(self) -> int:
        return self._clip_seq_len

    @property
    def sync_seq_len(self) -> int:
        return self._sync_seq_len


def small_16k(**kwargs) -> MMAudio:
    num_heads = 7
    return MMAudio(latent_dim=20,
                   clip_dim=1024,
                   sync_dim=768,
                   text_dim=1024,
                   hidden_dim=64 * num_heads,
                   depth=12,
                   fused_depth=8,
                   num_heads=num_heads,
                   latent_seq_len=250,
                   clip_seq_len=64,
                   sync_seq_len=192,
                   **kwargs)


def small_44k(**kwargs) -> MMAudio:
    num_heads = 7
    return MMAudio(latent_dim=40,
                   clip_dim=1024,
                   sync_dim=768,
                   text_dim=1024,
                   hidden_dim=64 * num_heads,
                   depth=12,
                   fused_depth=8,
                   num_heads=num_heads,
                   latent_seq_len=345,
                   clip_seq_len=64,
                   sync_seq_len=192,
                   **kwargs)


def medium_44k(**kwargs) -> MMAudio:
    num_heads = 14
    return MMAudio(latent_dim=40,
                   clip_dim=1024,
                   sync_dim=768,
                   text_dim=1024,
                   hidden_dim=64 * num_heads,
                   depth=12,
                   fused_depth=8,
                   num_heads=num_heads,
                   latent_seq_len=345,
                   clip_seq_len=64,
                   sync_seq_len=192,
                   **kwargs)


def large_44k(**kwargs) -> MMAudio:
    num_heads = 14
    return MMAudio(latent_dim=40,
                   clip_dim=1024,
                   sync_dim=768,
                   text_dim=1024,
                   hidden_dim=64 * num_heads,
                   depth=21,
                   fused_depth=14,
                   num_heads=num_heads,
                   latent_seq_len=345,
                   clip_seq_len=64,
                   sync_seq_len=192,
                   **kwargs)


def large_44k_v2(**kwargs) -> MMAudio:
    num_heads = 14
    return MMAudio(latent_dim=40,
                   clip_dim=1024,
                   sync_dim=768,
                   text_dim=1024,
                   hidden_dim=64 * num_heads,
                   depth=21,
                   fused_depth=14,
                   num_heads=num_heads,
                   latent_seq_len=345,
                   clip_seq_len=64,
                   sync_seq_len=192,
                   v2=True,
                   **kwargs)


def get_my_mmaudio(name: str, **kwargs) -> MMAudio:
    if name == 'small_16k':
        return small_16k(**kwargs)
    if name == 'small_44k':
        return small_44k(**kwargs)
    if name == 'medium_44k':
        return medium_44k(**kwargs)
    if name == 'large_44k':
        return large_44k(**kwargs)
    if name == 'large_44k_v2':
        return large_44k_v2(**kwargs)

    raise ValueError(f'Unknown model name: {name}')


if __name__ == '__main__':
    network = get_my_mmaudio('small_16k')

    # print the number of parameters in terms of millions
    num_params = sum(p.numel() for p in network.parameters()) / 1e6
    print(f'Number of parameters: {num_params:.2f}M')
