# Copyright 2024 Sony Research Inc.
# Modifications Copyright 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange

from vividaudio.ext.rotary_embeddings import apply_rope
from vividaudio.model.low_level import MLP, ChannelLastConv1d, ConvMLP


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return x * (1 + scale) + shift


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None):
    # training will crash without these contiguous calls and the CUDNN limitation
    # I believe this is related to https://github.com/pytorch/pytorch/issues/133974
    # unresolved at the time of writing
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    out = rearrange(out, 'b h n d -> b n (h d)').contiguous()
    return out

# Interleaved sequence
# def interleave_two_sequences(x1: torch.Tensor, x2: torch.Tensor):
#     # [B, N1, H, C] & [B, N2, H, C]
#     import pdb;pdb.set_trace()
#     B, H, N1, C = x1.shape
#     B, H, N2, C = x2.shape
#     assert x1.ndim == x2.ndim == 4
#     x1 = x1.reshape(B, N1, H, C)
#     x2 = x2.reshape(B, N2, H, C)

#     if N1 != N2:
#         x2 = x2.view(B, N2, -1).transpose(1, 2)
#         x2 = F.interpolate(x2, size=(N1), mode="nearest-exact")
#         x2 = x2.transpose(1, 2).view(B, N1, H, C)
#     x = torch.stack((x1, x2), dim=2)
#     x = x.reshape(B, N1 * 2, H, C)
#     x = x.reshape(B, H, N1 * 2, C)
#     return x


def interleave_two_sequences(x1: torch.Tensor, x2: torch.Tensor):
    # 1. 校验输入维度和批次一致性
    assert x1.ndim == x2.ndim == 4, "输入必须是4维张量 [B, H, N, C]"
    assert x1.shape[0] == x2.shape[0], "两个张量的批次维度B必须一致"
    assert x1.shape[1] == x2.shape[1] and x1.shape[3] == x2.shape[3], "H和C维度必须一致"

    # 2. 提取原始维度（B, H, N, C）
    B, H, N1, C = x1.shape
    B, H, N2, C = x2.shape

    # 3. 确定最长序列长度，并将短序列插值对齐到最长长度
    N_max = max(N1, N2)
    if N1 != N_max:
        # x1更短，插值到N_max
        x1 = x1.reshape(B, H, N1, C).transpose(2, 1)  # [B, N1, H, C] → [B, H, N1, C] → [B, H, N1, C].transpose(2,1) → [B, N1, H, C]
        x1 = x1.reshape(B, N1, -1).transpose(1, 2)     # [B, N1, H*C] → [B, H*C, N1]
        x1 = F.interpolate(x1, size=N_max, mode="nearest-exact")  # 插值到N_max
        x1 = x1.transpose(1, 2).reshape(B, N_max, H, C)  # [B, N_max, H*C] → [B, N_max, H, C]
        x1 = x1.transpose(1, 2)  # 恢复为 [B, H, N_max, C]
    elif N2 != N_max:
        # x2更短，插值到N_max
        x2 = x2.reshape(B, H, N2, C).transpose(2, 1)  # [B, H, N2, C] → [B, N2, H, C]
        x2 = x2.reshape(B, N2, -1).transpose(1, 2)     # [B, N2, H*C] → [B, H*C, N2]
        x2 = F.interpolate(x2, size=N_max, mode="nearest-exact")  # 插值到N_max
        x2 = x2.transpose(1, 2).reshape(B, N_max, H, C)  # [B, N_max, H*C] → [B, N_max, H, C]
        x2 = x2.transpose(1, 2)  # 恢复为 [B, H, N_max, C]

    # 4. 交替穿插拼接（核心逻辑）
    # 先将张量reshape为 [B, N_max, H, C]，方便按序列维度穿插
    x1_reshaped = x1.transpose(1, 2)  # [B, N_max, H, C]
    x2_reshaped = x2.transpose(1, 2)  # [B, N_max, H, C]

    # 堆叠后维度：[B, N_max, 2, H, C] → 穿插后reshape为 [B, N_max*2, H, C]
    x_interleaved = torch.stack((x1_reshaped, x2_reshaped), dim=2)  # [B, N_max, 2, H, C]
    x_interleaved = x_interleaved.reshape(B, N_max*2, H, C)         # [B, N_max*2, H, C]

    # 5. 恢复原始维度顺序 [B, H, N_max*2, C]
    x_out = x_interleaved.transpose(1, 2)  # [B, H, N_max*2, C]
    # x_out = x_interleaved.reshape(B, H, N_max * 2, C)

    return x_out

# def decouple_interleaved_two_sequences(x: torch.Tensor, len1: int, len2: int):
#     B, H, N, C = x.shape
#     x = x.reshape(B, N, H, C)

#     assert N % 2 == 0 and N // 2 == len1

#     x = x.reshape(B, -1, 2, H, C)
#     x1 = x[:, :, 0]
#     x2 = x[:, :, 1]
#     if x2.shape[1] != len2:
#         x2 = x2.view(B, len1, H * C).transpose(1, 2)
#         x2 = F.interpolate(x2, size=(len2), mode="nearest-exact")
#         x2 = x2.transpose(1, 2).view(B, len2, H, C)
#     x1 = x1.reshape(B, H, len1, C)
#     x2 = x2.reshape(B, H, len2, C)
#     return x1, x2

def decouple_interleaved_two_sequences(x: torch.Tensor, len1: int, len2: int):
    B, H, N, C = x.shape
    # 1. 确定最大长度，校验拼接后的长度是否符合 max_len*2
    max_len = max(len1, len2)
    assert N == max_len * 2, f"拼接后的长度N={N}必须等于max(len1={len1}, len2={len2})*2={max_len*2}"

    # 2. 调整维度，准备解耦
    x = x.reshape(B, N, H, C)  # [B, N, H, C]
    x = x.reshape(B, max_len, 2, H, C)  # [B, max_len, 2, H, C]

    # 3. 解耦出两个序列（此时都是max_len长度）
    x1_max = x[:, :, 0]  # [B, max_len, H, C]
    x2_max = x[:, :, 1]  # [B, max_len, H, C]

    if x1_max.shape[1] != len1:
        x1_max = x1_max.view(B, max_len, H * C).transpose(1, 2)  # [B, H*C, max_len]
        x1 = F.interpolate(x1_max, size=len1, mode="nearest-exact")  # 插值到原始len1
        x1 = x1.transpose(1, 2).view(B, len1, H, C)  # [B, len1, H, C]
    else:
        x1 = x1_max  # 长度一致，无需插值

    if x2_max.shape[1] != len2:
        x2_max = x2_max.view(B, max_len, H * C).transpose(1, 2)  # [B, H*C, max_len]
        x2 = F.interpolate(x2_max, size=len2, mode="nearest-exact")  # 插值到原始len2
        x2 = x2.transpose(1, 2).view(B, len2, H, C)  # [B, len2, H, C]
    else:
        x2 = x2_max  # 长度一致，无需插值

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
            rot: Optional[torch.Tensor] = None,
            *,
            return_unrotated_k: bool = False):
        qkv = self.qkv(x)
        q, k, v = self.split_into_heads(qkv).chunk(3, dim=-1)
        q = q.squeeze(-1)
        k = k.squeeze(-1)
        v = v.squeeze(-1)
        q = self.q_norm(q)
        k = self.k_norm(k)

        k_unrot = k if return_unrotated_k else None

        if rot is not None:
            q = apply_rope(q, rot)
            k = apply_rope(k, rot)

        if return_unrotated_k:
            return q, k, v, k_unrot
        return q, k, v

    def forward(
            self,
            x: torch.Tensor,  # batch_size * n_tokens * n_channels
    ) -> torch.Tensor:
        q, k, v = self.pre_attention(x)
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

    def pre_attention(self, x: torch.Tensor, c: torch.Tensor, rot: Optional[torch.Tensor] = None, *, return_unrotated_k: bool = False):
        modulation = self.adaLN_modulation(c)
        if self.pre_only:
            (shift_msa, scale_msa) = modulation.chunk(2, dim=-1)
            gate_msa = shift_mlp = scale_mlp = gate_mlp = None
        else:
            (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp,
             gate_mlp) = modulation.chunk(6, dim=-1)

        x = modulate(self.norm1(x), shift_msa, scale_msa)
        if return_unrotated_k:
            q, k, v, k_unrot = self.attn.pre_attention(x, rot, return_unrotated_k=True)
            return (q, k, v), (gate_msa, shift_mlp, scale_mlp, gate_mlp), k_unrot
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
                rot: Optional[torch.Tensor]) -> torch.Tensor:
        # x: BS * N * D
        # cond: BS * D
        x_qkv, x_conditions = self.pre_attention(x, cond, rot)
        attn_out = attention(*x_qkv)
        x = self.post_attention(x, attn_out, x_conditions)

        return x



class JointBlock(nn.Module):

    def __init__(self, dim: int, nhead: int, mlp_ratio: float = 4.0, sep_fgc: bool = False, delay_fusion: bool = False, pre_only: bool = False, interleaved_rope: bool = False, joint_fgc: bool = False, fusion_fgc_clip: bool = False, sync_rope: bool = False, cross_attn_delay: bool = False, local_window_size: int = 0, hybrid_cross_attn: bool = True, zero_init_gate: bool = True):
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
        self.sync_rope = sync_rope
        self.cross_attn_delay = cross_attn_delay
        self.local_window_size = local_window_size
        self.hybrid_cross_attn = hybrid_cross_attn

        # Innovation: Adaptive Modality Gating.
        # zero_init_gate=True (default): gate = 0 -> pretrained DiT sees no RAG
        # signal at step 0, so it can only smoothly ramp up during training.
        # zero_init_gate=False: gate = 1 from step 0, i.e. RAG cross-attention
        # is injected at full strength immediately (ablates the safety valve).
        if self.cross_attn_delay:
            init_val = 0.0 if zero_init_gate else 1.0
            self.rag_gate = nn.Parameter(torch.full((1,), init_val))


    def forward(self, latent: torch.Tensor, clip_f: torch.Tensor, text_f: torch.Tensor, fgc_f: torch.Tensor,
                global_c: torch.Tensor, extended_c: torch.Tensor, latent_rot: torch.Tensor,
                clip_rot: torch.Tensor, fgc_rot: torch.Tensor, sep_fgc: bool, delay_fusion: bool, inter_rot: torch.Tensor, inter_fusion_rot: torch.Tensor = None, skip_delay_fusion: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        # latent: BS * N1 * D
        # clip_f: BS * N2 * D
        # c: BS * (1/N) * D
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
            # import pdb;pdb.set_trace()
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
                if self.sync_rope:
                    arotx = latent_rot
                    frotx = fgc_rot
                elif self.interleaved_rope:
                    arotx = None
                    frotx = None
                else:
                    arotx = latent_rot
                    frotx = fgc_rot

                x_qkv, x_mod = self.delay_block.pre_attention(latent, extended_c, arotx)
                if self.cross_attn_delay and self.hybrid_cross_attn and self.local_window_size > 0:
                    fgc_qkv, fgc_mod, fgc_k_unrot = self.fgc_block.pre_attention(fgc_f, global_c, frotx, return_unrotated_k=True)
                else:
                    fgc_qkv, fgc_mod = self.fgc_block.pre_attention(fgc_f, global_c, frotx)
                    fgc_k_unrot = None

                if self.interleaved_rope and not self.sync_rope:
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


                if self.cross_attn_delay:
                    # X queries FGC (Cross Attention)
                    q = x_qkv[0]
                    k = fgc_qkv[1]
                    v = fgc_qkv[2]

                    attn_mask = None
                    if self.local_window_size > 0:
                        B, H, N_q, D = q.shape
                        _, _, N_k, _ = k.shape

                        idx_q = torch.arange(N_q, device=q.device).float()
                        idx_k = torch.arange(N_k, device=k.device).float()

                        # Map Q indices to K scale in case lengths differ
                        ratio = N_k / N_q if N_q > 0 else 1.0
                        mapped_idx_q = idx_q * ratio

                        dist = torch.abs(mapped_idx_q.unsqueeze(1) - idx_k.unsqueeze(0))

                        # Boolean Mask: True means allowed to attend
                        local_mask = dist <= self.local_window_size

                        if self.hybrid_cross_attn:
                            if fgc_k_unrot is None:
                                k_global = k.mean(dim=2, keepdim=True)
                            else:
                                k_global_raw = fgc_k_unrot.mean(dim=2, keepdim=True)
                                if self.interleaved_rope and not self.sync_rope:
                                    k_global = apply_rope(k_global_raw, inter_rot[:, :1].to(k_global_raw.device))
                                elif frotx is not None:
                                    k_global = apply_rope(k_global_raw, frotx[:, :1].to(k_global_raw.device))
                                else:
                                    k_global = k_global_raw
                            v_global = v.mean(dim=2, keepdim=True)

                            k = torch.cat([k_global, k], dim=2)
                            v = torch.cat([v_global, v], dim=2)

                            global_allow = torch.ones((N_q, 1), dtype=torch.bool, device=q.device)
                            combined_mask = torch.cat([global_allow, local_mask], dim=1)
                            attn_mask = combined_mask.unsqueeze(0).unsqueeze(0).expand(B, H, -1, -1).contiguous()
                        else:
                            attn_mask = local_mask.unsqueeze(0).unsqueeze(0).expand(B, H, -1, -1).contiguous()

                    x_attn_out = attention(q, k, v, mask=attn_mask)

                    # Apply Adaptive Gating (Smooth transition from pretrained DiT)
                    x_attn_out = x_attn_out * self.rag_gate

                    latent = self.delay_block.post_attention(latent, x_attn_out, x_mod)
                    # Note: fgc_f is not updated here as it's purely a condition
                else:
                    joint_qkv = [torch.cat([x_qkv[i], fgc_qkv[i]], dim=2) for i in range(3)]
                    attn_out = attention(*joint_qkv)
                    x_attn_out = attn_out[:, :latent_len]
                    fgc_attn_out = attn_out[:, latent_len:]

                    latent = self.delay_block.post_attention(latent, x_attn_out, x_mod)
                    if not self.pre_only:
                        fgc_f = self.fgc_block.post_attention(fgc_f, fgc_attn_out, fgc_mod)

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
