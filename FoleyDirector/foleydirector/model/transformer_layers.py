# Copyright 2024 Sony Research Inc.
# Modifications Copyright 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange

from foleydirector.ext.rotary_embeddings import apply_rope
from foleydirector.model.low_level import MLP, ChannelLastConv1d, ConvMLP


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return x * (1 + scale) + shift


def fuse_tracks_by_segments(
    features: torch.Tensor,
    track_weights: Optional[list[list[float]]],
) -> torch.Tensor:
    """Fuse independently rendered tracks with piecewise-constant weights."""
    if track_weights is None:
        return features
    if features.ndim != 3:
        raise ValueError(f"Expected [tracks, tokens, channels], got {features.shape}")

    weights = torch.as_tensor(
        track_weights,
        device=features.device,
        dtype=features.dtype,
    )
    if weights.ndim != 2 or weights.shape[0] != features.shape[0]:
        raise ValueError(
            "track_weights must have shape [num_tracks, num_segments], "
            f"got {tuple(weights.shape)} for {features.shape[0]} tracks"
        )
    if weights.shape[1] == 0:
        raise ValueError("track_weights must contain at least one segment")

    segments = torch.tensor_split(features, weights.shape[1], dim=1)
    fused = []
    for segment_index, segment in enumerate(segments):
        segment_weights = weights[:, segment_index]
        weight_sum = segment_weights.sum()
        if weight_sum.abs() < 1e-8:
            segment_weights = torch.full_like(segment_weights, 1.0 / features.shape[0])
        else:
            segment_weights = segment_weights / weight_sum
        fused.append(
            (segment * segment_weights[:, None, None]).sum(dim=0, keepdim=True)
        )
    return torch.cat(fused, dim=1)


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    # training will crash without these contiguous calls and the CUDNN limitation
    # I believe this is related to https://github.com/pytorch/pytorch/issues/133974
    # unresolved at the time of writing
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    out = F.scaled_dot_product_attention(q, k, v)
    out = rearrange(out, 'b h n d -> b n (h d)').contiguous()
    return out

# Interleaved sequence
def interleave_two_sequences(x1: torch.Tensor, x2: torch.Tensor):
    # [B, N1, H, C] & [B, N2, H, C]
    B, H, N1, C = x1.shape
    B, H, N2, C = x2.shape
    assert x1.ndim == x2.ndim == 4
    x1 = x1.reshape(B, N1, H, C)
    x2 = x2.reshape(B, N2, H, C)

    if N1 != N2:
        x2 = x2.view(B, N2, -1).transpose(1, 2)
        x2 = F.interpolate(x2, size=(N1), mode="nearest-exact")
        x2 = x2.transpose(1, 2).view(B, N1, H, C)
    x = torch.stack((x1, x2), dim=2)
    x = x.reshape(B, N1 * 2, H, C)
    x = x.reshape(B, H, N1 * 2, C)
    return x

def decouple_interleaved_two_sequences(x: torch.Tensor, len1: int, len2: int):
    B, H, N, C = x.shape
    x = x.reshape(B, N, H, C)

    assert N % 2 == 0 and N // 2 == len1

    x = x.reshape(B, -1, 2, H, C)
    x1 = x[:, :, 0]
    x2 = x[:, :, 1]
    if x2.shape[1] != len2:
        x2 = x2.view(B, len1, H * C).transpose(1, 2)
        x2 = F.interpolate(x2, size=(len2), mode="nearest-exact")
        x2 = x2.transpose(1, 2).view(B, len2, H, C)
    x1 = x1.reshape(B, H, len1, C)
    x2 = x2.reshape(B, H, len2, C)
    return x1, x2



class SelfAttention(nn.Module):

    def __init__(self, dim: int, nheads: int):
        super().__init__()
        self.dim = dim
        self.nheads = nheads

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.q_norm = nn.RMSNorm(dim // nheads)
        self.k_norm = nn.RMSNorm(dim // nheads)

        self.split_into_heads = Rearrange('b n (h d j) -> b h n d j',
                                          h=nheads,
                                          d=dim // nheads,
                                          j=3)

    def pre_attention(
            self, x: torch.Tensor,
            rot: Optional[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: batch_size * n_tokens * n_channels
        qkv = self.qkv(x)
        q, k, v = self.split_into_heads(qkv).chunk(3, dim=-1)
        q = q.squeeze(-1)
        k = k.squeeze(-1)
        v = v.squeeze(-1)
        q = self.q_norm(q)
        k = self.k_norm(k)

        if rot is not None:
            q = apply_rope(q, rot)
            k = apply_rope(k, rot)

        return q, k, v

    def forward(
            self,
            x: torch.Tensor,  # batch_size * n_tokens * n_channels
    ) -> torch.Tensor:
        q, v, k = self.pre_attention(x)
        out = attention(q, k, v)
        return out


class MMDitSingleBlock(nn.Module):

    def __init__(self,
                 dim: int,
                 nhead: int,
                 mlp_ratio: float = 4.0,
                 pre_only: bool = False,
                 kernel_size: int = 7,
                 padding: int = 3):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = SelfAttention(dim, nhead)

        self.pre_only = pre_only
        if pre_only:
            self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim, bias=True))
        else:
            if kernel_size == 1:
                self.linear1 = nn.Linear(dim, dim)
            else:
                self.linear1 = ChannelLastConv1d(dim, dim, kernel_size=kernel_size, padding=padding)
            self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)

            if kernel_size == 1:
                self.ffn = MLP(dim, int(dim * mlp_ratio))
            else:
                self.ffn = ConvMLP(dim,
                                   int(dim * mlp_ratio),
                                   kernel_size=kernel_size,
                                   padding=padding)

            self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))

    def pre_attention(self, x: torch.Tensor, c: torch.Tensor, rot: Optional[torch.Tensor]):
        # x: BS * N * D
        # cond: BS * D
        modulation = self.adaLN_modulation(c)
        if self.pre_only:
            (shift_msa, scale_msa) = modulation.chunk(2, dim=-1)
            gate_msa = shift_mlp = scale_mlp = gate_mlp = None
        else:
            (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp,
             gate_mlp) = modulation.chunk(6, dim=-1)

        x = modulate(self.norm1(x), shift_msa, scale_msa)
        q, k, v = self.attn.pre_attention(x, rot)
        return (q, k, v), (gate_msa, shift_mlp, scale_mlp, gate_mlp)

    def post_attention(self, x: torch.Tensor, attn_out: torch.Tensor, c: tuple[torch.Tensor]):
        if self.pre_only:
            return x

        (gate_msa, shift_mlp, scale_mlp, gate_mlp) = c
        x = x + self.linear1(attn_out) * gate_msa
        r = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + self.ffn(r) * gate_mlp

        return x

    def forward(self, x: torch.Tensor, cond: torch.Tensor,
                rot: Optional[torch.Tensor],
                track_weights: Optional[list[list[float]]] = None) -> torch.Tensor:
        # x: BS * N * D
        # cond: BS * D
        if track_weights is not None and x.shape[0] == 1 and cond.shape[0] > 1:
            x = x.expand(cond.shape[0], -1, -1)
        x_qkv, x_conditions = self.pre_attention(x, cond, rot)
        attn_out = attention(*x_qkv)
        x = self.post_attention(x, attn_out, x_conditions)
        return fuse_tracks_by_segments(x, track_weights)


# class JointBlock(nn.Module):

#     def __init__(self, dim: int, nhead: int, mlp_ratio: float = 4.0, sep_fgc: bool = False, delay_fusion: bool = False, pre_only: bool = False, interleaved_rope: bool = False, joint_fgc: bool = False):
#         super().__init__()
#         self.pre_only = pre_only
#         self.latent_block = MMDitSingleBlock(dim,
#                                              nhead,
#                                              mlp_ratio,
#                                              pre_only=False,
#                                              kernel_size=3,
#                                              padding=1)
#         self.clip_block = MMDitSingleBlock(dim,
#                                            nhead,
#                                            mlp_ratio,
#                                            pre_only=pre_only,
#                                            kernel_size=3,
#                                            padding=1)

#         self.text_block = MMDitSingleBlock(dim, nhead, mlp_ratio, pre_only=pre_only, kernel_size=1)
#         self.sep_fgc = sep_fgc
#         self.delay_fusion = delay_fusion
#         self.interleaved_rope = interleaved_rope
#         self.joint_fgc = joint_fgc
#         # if self.sep_fgc or self.delay_fusion:
#         self.fgc_block = MMDitSingleBlock(dim, nhead, mlp_ratio, pre_only=pre_only, kernel_size=1)
#         if self.delay_fusion:
#             self.delay_block = MMDitSingleBlock(dim, nhead, mlp_ratio, pre_only=pre_only, kernel_size=1)


#     def forward(self, latent: torch.Tensor, clip_f: torch.Tensor, text_f: torch.Tensor, fgc_f: torch.Tensor,
#                 global_c: torch.Tensor, extended_c: torch.Tensor, latent_rot: torch.Tensor,
#                 clip_rot: torch.Tensor, fgc_rot: torch.Tensor, sep_fgc: bool, delay_fusion: bool, inter_rot: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
#         # latent: BS * N1 * D
#         # clip_f: BS * N2 * D
#         # c: BS * (1/N) * D
#         x_qkv, x_mod = self.latent_block.pre_attention(latent, extended_c, latent_rot)
#         c_qkv, c_mod = self.clip_block.pre_attention(clip_f, global_c, clip_rot)
#         t_qkv, t_mod = self.text_block.pre_attention(text_f, global_c, rot=None)


#         latent_len = latent.shape[1]
#         clip_len = clip_f.shape[1]
#         text_len = text_f.shape[1]


#         if not delay_fusion:


#             if sep_fgc:
#                 fgc_qkv, fgc_mod = self.fgc_block.pre_attention(fgc_f, global_c, fgc_rot)
#                 fgc_len = fgc_f.shape[1]
#                 joint_qkv = [torch.cat([x_qkv[i], c_qkv[i], t_qkv[i], fgc_qkv[i]], dim=2) for i in range(3)]
#             else:
#                 fgc_qkv, fgc_mod = self.fgc_block.pre_attention(fgc_f, global_c, fgc_rot)
#                 joint_qkv = [torch.cat([x_qkv[i], c_qkv[i], t_qkv[i]], dim=2) for i in range(3)]

#             attn_out = attention(*joint_qkv)
#             x_attn_out = attn_out[:, :latent_len]
#             c_attn_out = attn_out[:, latent_len:latent_len + clip_len]
#             t_attn_out = attn_out[:, latent_len + clip_len:latent_len + clip_len + text_len]

#             if sep_fgc:
#                 fgc_attn_out = attn_out[:, latent_len + clip_len + text_len:]

#             latent = self.latent_block.post_attention(latent, x_attn_out, x_mod)
#             if not self.pre_only:
#                 clip_f = self.clip_block.post_attention(clip_f, c_attn_out, c_mod)
#                 text_f = self.text_block.post_attention(text_f, t_attn_out, t_mod)
#                 if sep_fgc:
#                     fgc_f = self.fgc_block.post_attention(fgc_f, fgc_attn_out, fgc_mod)
#         else:
#             joint_qkv = [torch.cat([x_qkv[i], c_qkv[i], t_qkv[i]], dim=2) for i in range(3)]

#             attn_out = attention(*joint_qkv)
#             x_attn_out = attn_out[:, :latent_len]
#             c_attn_out = attn_out[:, latent_len:latent_len + clip_len]
#             t_attn_out = attn_out[:, latent_len + clip_len:latent_len + clip_len + text_len]



#             latent = self.latent_block.post_attention(latent, x_attn_out, x_mod)
#             if not self.pre_only:
#                 clip_f = self.clip_block.post_attention(clip_f, c_attn_out, c_mod)
#                 text_f = self.text_block.post_attention(text_f, t_attn_out, t_mod)

#             # delay fusion
#             if self.interleaved_rope:
#                 arotx = None
#                 frotx = None
#             else:
#                 arotx = latent_rot
#                 frotx = fgc_rot

#             x_qkv, x_mod = self.delay_block.pre_attention(latent, extended_c, arotx)
#             fgc_qkv, fgc_mod = self.fgc_block.pre_attention(fgc_f, global_c, frotx)

#             if self.interleaved_rope:
#                 ori_audio_len = x_qkv[0].shape[2]
#                 ori_fgc_len = fgc_qkv[0].shape[2]
#                 interleaved_audio_visual_q = interleave_two_sequences(x_qkv[0], fgc_qkv[0])
#                 interleaved_audio_visual_k = interleave_two_sequences(x_qkv[1], fgc_qkv[1])
#                 interleaved_audio_visual_qq = apply_rope(interleaved_audio_visual_q, inter_rot.to(interleaved_audio_visual_q.device))
#                 interleaved_audio_visual_kk = apply_rope(interleaved_audio_visual_k, inter_rot.to(interleaved_audio_visual_q.device))
#                 audio_qq, v_cond_qq = decouple_interleaved_two_sequences(
#                     interleaved_audio_visual_qq, ori_audio_len, ori_fgc_len
#                 )
#                 audio_kk, v_cond_kk = decouple_interleaved_two_sequences(
#                     interleaved_audio_visual_kk, ori_audio_len, ori_fgc_len
#                 )
#                 x_qkv = (audio_qq, audio_kk, x_qkv[2])
#                 fgc_qkv = (v_cond_qq, v_cond_kk, fgc_qkv[2])

#             joint_qkv = [torch.cat([x_qkv[i], fgc_qkv[i]], dim=2) for i in range(3)]
#             attn_out = attention(*joint_qkv)
#             x_attn_out = attn_out[:, :latent_len]
#             fgc_attn_out = attn_out[:, latent_len:]

#             latent = self.delay_block.post_attention(latent, x_attn_out, x_mod)
#             if not self.pre_only:
#                 fgc_f = self.fgc_block.post_attention(fgc_f, fgc_attn_out, fgc_mod)

#         return latent, clip_f, text_f, fgc_f


class JointBlock(nn.Module):

    def __init__(self, dim: int, nhead: int, mlp_ratio: float = 4.0, sep_fgc: bool = False, delay_fusion: bool = False, pre_only: bool = False, interleaved_rope: bool = False, joint_fgc: bool = False, fusion_fgc_clip: bool = False):
        super().__init__()
        self.pre_only = pre_only
        self.latent_block = MMDitSingleBlock(dim,
                                             nhead,
                                             mlp_ratio,
                                             pre_only=False,
                                             kernel_size=3,
                                             padding=1)
        self.clip_block = MMDitSingleBlock(dim,
                                           nhead,
                                           mlp_ratio,
                                           pre_only=pre_only,
                                           kernel_size=3,
                                           padding=1)

        self.text_block = MMDitSingleBlock(dim, nhead, mlp_ratio, pre_only=pre_only, kernel_size=1)
        self.sep_fgc = sep_fgc
        self.delay_fusion = delay_fusion
        self.interleaved_rope = interleaved_rope
        self.joint_fgc = joint_fgc
        # if self.sep_fgc or self.delay_fusion:
        self.fgc_block = MMDitSingleBlock(dim, nhead, mlp_ratio, pre_only=pre_only, kernel_size=1)
        if self.delay_fusion:
            self.delay_block = MMDitSingleBlock(dim, nhead, mlp_ratio, pre_only=pre_only, kernel_size=1)
        if not self.sep_fgc or self.joint_fgc:
            self.fusion_clip_block = MMDitSingleBlock(dim, nhead, mlp_ratio, pre_only=pre_only, kernel_size=3, padding=1)
        if not self.sep_fgc or self.joint_fgc or fusion_fgc_clip:
            self.fusion_fgc_block = MMDitSingleBlock(dim, nhead, mlp_ratio, pre_only=pre_only, kernel_size=1)
        self.fusion_fgc_clip = fusion_fgc_clip


    def forward(self, latent: torch.Tensor, clip_f: torch.Tensor, text_f: torch.Tensor, fgc_f: torch.Tensor,
                global_c: torch.Tensor, extended_c: torch.Tensor, latent_rot: torch.Tensor,
                clip_rot: torch.Tensor, fgc_rot: torch.Tensor, sep_fgc: bool, delay_fusion: bool, inter_rot: torch.Tensor, inter_fusion_rot: torch.Tensor = None, skip_delay_fusion: bool = False,
                track_weights: Optional[list[list[float]]] = None) -> tuple[torch.Tensor, torch.Tensor]:
        # latent: BS * N1 * D
        # clip_f: BS * N2 * D
        # c: BS * (1/N) * D
        if track_weights is not None and latent.shape[0] == 1 and extended_c.shape[0] > 1:
            latent = latent.expand(extended_c.shape[0], -1, -1)
        x_qkv, x_mod = self.latent_block.pre_attention(latent, extended_c, latent_rot)
        t_qkv, t_mod = self.text_block.pre_attention(text_f, global_c, rot=None)


        latent_len = latent.shape[1]
        clip_len = clip_f.shape[1]
        text_len = text_f.shape[1]


        if not delay_fusion:

            if sep_fgc or self.joint_fgc:

                if self.joint_fgc:
                    c_qkv, c_mod = self.fusion_clip_block.pre_attention(clip_f, global_c, clip_rot)
                    fgc_qkv, fgc_mod = self.fusion_fgc_block.pre_attention(fgc_f, global_c, fgc_rot)
                    joint_qkv = [torch.cat([c_qkv[i], fgc_qkv[i]], dim=2) for i in range(3)]
                    attn_out = attention(*joint_qkv)
                    c_attn_out = attn_out[:, :clip_len]
                    fgc_attn_out = attn_out[:, clip_len:]
                    clip_f = self.fusion_clip_block.post_attention(clip_f, c_attn_out, c_mod)
                    fgc_f = self.fgc_block.post_attention(fgc_f, fgc_attn_out, fgc_mod)

                c_qkv, c_mod = self.clip_block.pre_attention(clip_f, global_c, clip_rot)
                fgc_qkv, fgc_mod = self.fgc_block.pre_attention(fgc_f, global_c, fgc_rot)
                fgc_f.shape[1]
                joint_qkv = [torch.cat([x_qkv[i], c_qkv[i], t_qkv[i], fgc_qkv[i]], dim=2) for i in range(3)]
            else:
                fgc_qkv, fgc_mod = self.fgc_block.pre_attention(fgc_f, global_c, None)
                c_qkv, c_mod = self.fusion_clip_block.pre_attention(clip_f, global_c, None)

                ori_audio_len = c_qkv[0].shape[2]
                ori_fgc_len = fgc_qkv[0].shape[2]
                interleaved_audio_visual_q = interleave_two_sequences(c_qkv[0], fgc_qkv[0])
                interleaved_audio_visual_k = interleave_two_sequences(c_qkv[1], fgc_qkv[1])
                interleaved_audio_visual_qq = apply_rope(interleaved_audio_visual_q, inter_fusion_rot.to(interleaved_audio_visual_q.device))
                interleaved_audio_visual_kk = apply_rope(interleaved_audio_visual_k, inter_fusion_rot.to(interleaved_audio_visual_q.device))
                audio_qq, v_cond_qq = decouple_interleaved_two_sequences(
                    interleaved_audio_visual_qq, ori_audio_len, ori_fgc_len
                )
                audio_kk, v_cond_kk = decouple_interleaved_two_sequences(
                    interleaved_audio_visual_kk, ori_audio_len, ori_fgc_len
                )
                c_qkv = (audio_qq, audio_kk, c_qkv[2])
                fgc_qkv = (v_cond_qq, v_cond_kk, fgc_qkv[2])
                joint_qkv = [torch.cat([c_qkv[i], fgc_qkv[i]], dim=2) for i in range(3)]
                attn_out = attention(*joint_qkv)
                c_attn_out = attn_out[:, :clip_len]
                fgc_attn_out = attn_out[:, clip_len:]
                clip_f = self.fusion_clip_block.post_attention(clip_f, c_attn_out, c_mod)
                fgc_f = self.fgc_block.post_attention(fgc_f, fgc_attn_out, fgc_mod)
                c_qkv, c_mod = self.clip_block.pre_attention(clip_f, global_c, clip_rot)
                joint_qkv = [torch.cat([x_qkv[i], c_qkv[i], t_qkv[i]], dim=2) for i in range(3)]

            attn_out = attention(*joint_qkv)
            x_attn_out = attn_out[:, :latent_len]
            c_attn_out = attn_out[:, latent_len:latent_len + clip_len]
            t_attn_out = attn_out[:, latent_len + clip_len:latent_len + clip_len + text_len]

            if sep_fgc:
                fgc_attn_out = attn_out[:, latent_len + clip_len + text_len:]

            latent = self.latent_block.post_attention(latent, x_attn_out, x_mod)
            if not self.pre_only:
                clip_f = self.clip_block.post_attention(clip_f, c_attn_out, c_mod)
                text_f = self.text_block.post_attention(text_f, t_attn_out, t_mod)
                if sep_fgc:
                    fgc_f = self.fgc_block.post_attention(fgc_f, fgc_attn_out, fgc_mod)
        else:
            c_qkv, c_mod = self.clip_block.pre_attention(clip_f, global_c, clip_rot)
            if self.fusion_fgc_clip:
                fgc_qkv, fgc_mod = self.fusion_fgc_block.pre_attention(fgc_f, global_c, fgc_rot)
                joint_qkv = [torch.cat([c_qkv[i], fgc_qkv[i]], dim=2) for i in range(3)]
                c_attn_out = attn_out[:, :clip_len]
                fgc_attn_out = attn_out[:, clip_len:]
                fgc_f = self.fusion_fgc_block.post_attention(fgc_f, fgc_attn_out, fgc_mod)

            joint_qkv = [torch.cat([x_qkv[i], c_qkv[i], t_qkv[i]], dim=2) for i in range(3)]
            attn_out = attention(*joint_qkv)
            x_attn_out = attn_out[:, :latent_len]
            c_attn_out = attn_out[:, latent_len:latent_len + clip_len]
            t_attn_out = attn_out[:, latent_len + clip_len:latent_len + clip_len + text_len]

            latent = self.latent_block.post_attention(latent, x_attn_out, x_mod)
            if not self.pre_only:
                clip_f = self.clip_block.post_attention(clip_f, c_attn_out, c_mod)
                text_f = self.text_block.post_attention(text_f, t_attn_out, t_mod)

            # delay fusion
            if not skip_delay_fusion:
                if self.interleaved_rope:
                    arotx = None
                    frotx = None
                else:
                    arotx = latent_rot
                    frotx = fgc_rot

                x_qkv, x_mod = self.delay_block.pre_attention(latent, extended_c, arotx)
                fgc_qkv, fgc_mod = self.fgc_block.pre_attention(fgc_f, global_c, frotx)

                if self.interleaved_rope:
                    ori_audio_len = x_qkv[0].shape[2]
                    ori_fgc_len = fgc_qkv[0].shape[2]
                    interleaved_audio_visual_q = interleave_two_sequences(x_qkv[0], fgc_qkv[0])
                    interleaved_audio_visual_k = interleave_two_sequences(x_qkv[1], fgc_qkv[1])
                    interleaved_audio_visual_qq = apply_rope(interleaved_audio_visual_q, inter_rot.to(interleaved_audio_visual_q.device))
                    interleaved_audio_visual_kk = apply_rope(interleaved_audio_visual_k, inter_rot.to(interleaved_audio_visual_q.device))
                    audio_qq, v_cond_qq = decouple_interleaved_two_sequences(
                        interleaved_audio_visual_qq, ori_audio_len, ori_fgc_len
                    )
                    audio_kk, v_cond_kk = decouple_interleaved_two_sequences(
                        interleaved_audio_visual_kk, ori_audio_len, ori_fgc_len
                    )
                    x_qkv = (audio_qq, audio_kk, x_qkv[2])
                    fgc_qkv = (v_cond_qq, v_cond_kk, fgc_qkv[2])

                joint_qkv = [torch.cat([x_qkv[i], fgc_qkv[i]], dim=2) for i in range(3)]
                attn_out = attention(*joint_qkv)
                x_attn_out = attn_out[:, :latent_len]
                fgc_attn_out = attn_out[:, latent_len:]

                latent = self.delay_block.post_attention(latent, x_attn_out, x_mod)
                if not self.pre_only:
                    fgc_f = self.fgc_block.post_attention(fgc_f, fgc_attn_out, fgc_mod)

        latent = fuse_tracks_by_segments(latent, track_weights)
        return latent, clip_f, text_f, fgc_f


class FinalBlock(nn.Module):

    def __init__(self, dim, out_dim):
        super().__init__()
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim, bias=True))
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.conv = ChannelLastConv1d(dim, out_dim, kernel_size=7, padding=3)

    def forward(self, latent, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        latent = modulate(self.norm(latent), shift, scale)
        latent = self.conv(latent)
        return latent
