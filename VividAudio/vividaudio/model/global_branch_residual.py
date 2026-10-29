# Copyright 2024 Sony Research Inc.
# Modifications Copyright 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from vividaudio.model.networks import PreprocessedConditions


@dataclass
class GlobalBranchResidualConfig:
    """Inference-time wrapper config for global style branching.

    This wrapper is intentionally non-invasive:
    - no new trainable parameters
    - no checkpoint format changes
    - default behavior is identical to the wrapped network

    The actual branch-residual algorithm is guarded by ``enabled`` and can be
    filled in iteratively without affecting the original model path.
    """

    enabled: bool = False
    fusion_mode: str = "residual"
    alpha: float = 1.0
    start_joint_block: int = 0
    end_joint_block: int = -1
    apply_fused_blocks: bool = False
    schedule: str = "constant"


class GlobalBranchResidualWrapper(nn.Module):
    """Transparent wrapper around ``MMAudio`` for experimental branch routing.

    Design goal:
    keep the original ``MMAudio`` untouched and let experiments opt into a new
    inference path through composition instead of patching ``networks.py``.
    """

    def __init__(self, base_net: nn.Module, cfg: Optional[GlobalBranchResidualConfig] = None) -> None:
        super().__init__()
        self.base_net = base_net
        self.branch_cfg = cfg or GlobalBranchResidualConfig()

    def __setattr__(self, name: str, value) -> None:
        # batch_eval_rag stashes per-event tensors on ``net`` before calling
        # preprocess_conditions(). Once wrapped, that write must reach the
        # underlying MMAudio instance, which owns and consumes these fields.
        if name in {"per_event_grag_stack", "per_event_slot_mask"}:
            base_net = self.__dict__.get("_modules", {}).get("base_net")
            if base_net is not None:
                setattr(base_net, name, value)
                return
        super().__setattr__(name, value)

    def preprocess_conditions(self, *args, **kwargs) -> PreprocessedConditions:
        return self.base_net.preprocess_conditions(*args, **kwargs)

    def get_empty_conditions(self, *args, **kwargs) -> PreprocessedConditions:
        return self.base_net.get_empty_conditions(*args, **kwargs)

    def ode_wrapper(
        self,
        t: torch.Tensor,
        latent: torch.Tensor,
        conditions: PreprocessedConditions,
        empty_conditions: PreprocessedConditions,
        cfg_strength: float,
        neg_conditions: PreprocessedConditions = None,
        cfg_rag_weight: float = 0.5,
    ) -> torch.Tensor:
        t = t * torch.ones(len(latent), device=latent.device, dtype=latent.dtype)

        if cfg_strength < 1.0:
            return self.predict_flow(latent, t, conditions)

        flow_cond = self.predict_flow(latent, t, conditions)
        new_conditions = PreprocessedConditions(
            clip_f=empty_conditions.clip_f,
            sync_f=empty_conditions.sync_f,
            text_f=empty_conditions.text_f,
            clip_f_c=empty_conditions.clip_f_c,
            text_f_c=empty_conditions.text_f_c,
            rag_f=conditions.rag_f,
        )
        empty_conditions.rag_f = neg_conditions.rag_f
        flow_rag = self.predict_flow(latent, t, new_conditions)
        flow_empty = self.predict_flow(latent, t, empty_conditions)
        return cfg_rag_weight * (
            cfg_strength * flow_cond + (1 - cfg_strength) * flow_rag
        ) + (1.0 - cfg_rag_weight) * (
            cfg_strength * flow_cond + (1 - cfg_strength) * flow_empty
        )

    def normalize(self, *args, **kwargs):
        return self.base_net.normalize(*args, **kwargs)

    def unnormalize(self, *args, **kwargs):
        return self.base_net.unnormalize(*args, **kwargs)

    def load_weights(self, *args, **kwargs):
        return self.base_net.load_weights(*args, **kwargs)

    def load_weights_fgc(self, *args, **kwargs):
        return self.base_net.load_weights_fgc(*args, **kwargs)

    def forward(
        self,
        latent: torch.Tensor,
        clip_f: torch.Tensor,
        sync_f: torch.Tensor,
        text_f: torch.Tensor,
        t: torch.Tensor,
        rag_f: torch.Tensor,
        return_inter: bool = False,
        rag_vae_f: torch.Tensor = None,
        grag_f: torch.Tensor = None,
    ):
        conditions = self.preprocess_conditions(
            clip_f, sync_f, text_f, rag_f, rag_vae_f, grag_f=grag_f
        )
        return self.predict_flow(latent, t, conditions, return_inter=return_inter)

    def enable(self, enabled: bool = True) -> None:
        self.branch_cfg.enabled = bool(enabled)

    def _branch_ready(self, conditions: PreprocessedConditions) -> bool:
        return (
            self.branch_cfg.enabled
            and getattr(self.base_net, "global_rag_cond", False)
            and getattr(conditions, "rag_f_c_per_event", None) is not None
            and getattr(conditions, "slot_mask", None) is not None
        )

    def _alpha_at_t(self, t: torch.Tensor, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        alpha = float(self.branch_cfg.alpha)
        mode = str(self.branch_cfg.schedule or "constant").lower()
        if mode in {"constant", "const", "off"}:
            return torch.as_tensor(alpha, device=device, dtype=dtype)

        t_scalar = t.float().mean().clamp(0.0, 1.0)
        if mode in {"early_decay", "linear_decay", "t"}:
            return torch.as_tensor(alpha, device=device, dtype=dtype) * t_scalar.to(device=device, dtype=dtype)
        if mode in {"late_ramp", "1-t", "reverse"}:
            return torch.as_tensor(alpha, device=device, dtype=dtype) * (1.0 - t_scalar).to(device=device, dtype=dtype)
        return torch.as_tensor(alpha, device=device, dtype=dtype)

    def _joint_block_selected(self, idx: int, total_blocks: int) -> bool:
        start = max(0, int(self.branch_cfg.start_joint_block))
        end = int(self.branch_cfg.end_joint_block)
        if end < 0:
            end = total_blocks
        end = min(total_blocks, end)
        return start <= idx < end

    def _compute_global_and_extended(
        self,
        conditions: PreprocessedConditions,
        t: torch.Tensor,
        rag_style_override: Optional[torch.Tensor] = None,
        use_per_event_latent_override: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        clip_f_c = conditions.clip_f_c
        text_f_c = conditions.text_f_c
        sync_f = conditions.sync_f
        rag_f_c = conditions.rag_f_c

        global_c_input = clip_f_c + text_f_c
        if getattr(self.base_net, "global_rag_cond", False):
            if rag_style_override is not None:
                global_c_input = global_c_input + rag_style_override
            elif rag_f_c is not None:
                global_c_input = global_c_input + rag_f_c

        global_c = self.base_net.global_cond_mlp(global_c_input)
        global_c = self.base_net.t_embed(t).unsqueeze(1) + global_c.unsqueeze(1)
        extended_c = global_c + sync_f

        pe_style = conditions.rag_f_c_per_event
        pe_mask = conditions.slot_mask
        if (
            use_per_event_latent_override
            and rag_style_override is None
            and pe_style is not None
            and pe_mask is not None
            and getattr(self.base_net, "global_rag_cond", False)
        ):
            style_at_t = torch.einsum("bkd,bkn->bnd", pe_style, pe_mask)
            base_ct = (clip_f_c + text_f_c).unsqueeze(1)
            global_c_t = self.base_net.global_cond_mlp(base_ct + style_at_t)
            global_c_t = self.base_net.t_embed(t).unsqueeze(1) + global_c_t
            extended_c = global_c_t + sync_f

        return global_c, extended_c

    def _apply_branch_residual_delta(
        self,
        latent_base: torch.Tensor,
        fused_delta: Optional[torch.Tensor],
        alpha_t: torch.Tensor,
    ) -> torch.Tensor:
        if fused_delta is None:
            return latent_base
        while alpha_t.dim() < fused_delta.dim():
            alpha_t = alpha_t.unsqueeze(-1)
        return latent_base + alpha_t.to(fused_delta.dtype) * fused_delta

    def _prepare_latent_input(self, latent: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if self.base_net.inpainting:
            xt = latent[..., :self.base_net.latent_dim]
            assert xt.shape[1] == self.base_net._latent_seq_len, (
                f"{xt.shape=} {self.base_net._latent_seq_len=}"
            )
            inp_scale_map = getattr(self.base_net, "inpainting_scale_map", None)
            if inp_scale_map is not None:
                inp_scale = inp_scale_map.to(xt.dtype)
            else:
                inp_scale = getattr(self.base_net, "inpainting_scale", 1.0)

            inp_fn = getattr(self.base_net, "inpainting_scale_fn", None)
            if inp_fn is not None:
                gate = inp_fn(t)
                if isinstance(gate, torch.Tensor):
                    while gate.dim() < 3:
                        gate = gate.unsqueeze(-1)
                    gate = gate.to(xt.dtype)
                inp_scale = inp_scale * gate

            chan_scale = getattr(self.base_net, "inpainting_channel_scale", None)
            if chan_scale is not None:
                if not isinstance(chan_scale, torch.Tensor):
                    chan_scale = torch.as_tensor(chan_scale, device=xt.device, dtype=xt.dtype)
                while chan_scale.dim() < 3:
                    chan_scale = chan_scale.unsqueeze(0)
                chan_scale = chan_scale.to(xt.device, xt.dtype)

            resid = self.base_net.inpainting_mlp(latent)
            if chan_scale is not None:
                resid = resid * chan_scale
            if self.base_net.bypass_empty_inpainting:
                mask = latent[..., -1:]
                has_inpainting = (mask.amax(dim=1, keepdim=True) > 0).to(xt.dtype)
                latent = self.base_net.audio_input_proj(xt) + inp_scale * has_inpainting * resid
            else:
                latent = self.base_net.audio_input_proj(xt) + inp_scale * resid
            return latent

        assert latent.shape[1] == self.base_net._latent_seq_len, (
            f"{latent.shape=} {self.base_net._latent_seq_len=}"
        )
        return self.base_net.audio_input_proj(latent)

    def _predict_flow_branch_residual(
        self,
        latent: torch.Tensor,
        t: torch.Tensor,
        conditions: PreprocessedConditions,
        return_inter: bool = False,
    ) -> torch.Tensor:
        latent = self._prepare_latent_input(latent, t)

        clip_f = conditions.clip_f
        text_f = conditions.text_f
        rag_f = conditions.rag_f
        pe_style = conditions.rag_f_c_per_event
        pe_mask = conditions.slot_mask

        global_c, extended_c = self._compute_global_and_extended(
            conditions, t, use_per_event_latent_override=True
        )
        alpha_t = self._alpha_at_t(t, device=latent.device, dtype=latent.dtype)

        inter_latent = None
        total_joint_blocks = len(self.base_net.joint_blocks)
        for i, block in enumerate(self.base_net.joint_blocks):
            latent_base, clip_f_base, text_f_base, rag_f_base = block(
                latent,
                clip_f,
                text_f,
                rag_f,
                global_c,
                extended_c,
                self.base_net.latent_rot,
                self.base_net.clip_rot,
                self.base_net.fgc_rot,
                self.base_net.sep_fgc,
                self.base_net.delay_fusion,
                self.base_net.inter_rot,
                self.base_net.inter_fusion_rot,
            )

            if self._joint_block_selected(i, total_joint_blocks):
                fused_delta = None
                for k in range(pe_style.shape[1]):
                    global_c_k, extended_c_k = self._compute_global_and_extended(
                        conditions,
                        t,
                        rag_style_override=pe_style[:, k, :],
                        use_per_event_latent_override=False,
                    )
                    latent_k, _, _, _ = block(
                        latent,
                        clip_f,
                        text_f,
                        rag_f,
                        global_c_k,
                        extended_c_k,
                        self.base_net.latent_rot,
                        self.base_net.clip_rot,
                        self.base_net.fgc_rot,
                        self.base_net.sep_fgc,
                        self.base_net.delay_fusion,
                        self.base_net.inter_rot,
                        self.base_net.inter_fusion_rot,
                    )
                    delta_k = latent_k - latent_base
                    weight_k = pe_mask[:, k, :].unsqueeze(-1).to(delta_k.dtype)
                    contrib_k = weight_k * delta_k
                    fused_delta = contrib_k if fused_delta is None else fused_delta + contrib_k
                latent = self._apply_branch_residual_delta(latent_base, fused_delta, alpha_t)
                clip_f, text_f, rag_f = clip_f_base, text_f_base, rag_f_base
            else:
                latent, clip_f, text_f, rag_f = latent_base, clip_f_base, text_f_base, rag_f_base

            if i == total_joint_blocks // 2:
                inter_latent = latent

        if inter_latent is None:
            inter_latent = latent

        for block in self.base_net.fused_blocks:
            latent_base = block(latent, extended_c, self.base_net.latent_rot)
            if self.branch_cfg.apply_fused_blocks:
                fused_delta = None
                for k in range(pe_style.shape[1]):
                    _, extended_c_k = self._compute_global_and_extended(
                        conditions,
                        t,
                        rag_style_override=pe_style[:, k, :],
                        use_per_event_latent_override=False,
                    )
                    latent_k = block(latent, extended_c_k, self.base_net.latent_rot)
                    delta_k = latent_k - latent_base
                    weight_k = pe_mask[:, k, :].unsqueeze(-1).to(delta_k.dtype)
                    contrib_k = weight_k * delta_k
                    fused_delta = contrib_k if fused_delta is None else fused_delta + contrib_k
                latent = self._apply_branch_residual_delta(latent_base, fused_delta, alpha_t)
            else:
                latent = latent_base

        flow = self.base_net.final_layer(latent, extended_c)
        if return_inter:
            inter_latent_proj = self.base_net.repa_proj(inter_latent)
            return flow, inter_latent_proj
        return flow

    def predict_flow(
        self,
        latent: torch.Tensor,
        t: torch.Tensor,
        conditions: PreprocessedConditions,
        return_inter: bool = False,
    ) -> torch.Tensor:
        if not self._branch_ready(conditions):
            return self.base_net.predict_flow(latent, t, conditions, return_inter=return_inter)
        return self._predict_flow_branch_residual(latent, t, conditions, return_inter=return_inter)

    def __getattr__(self, name: str):
        if name in {"base_net", "branch_cfg"}:
            return super().__getattr__(name)
        return getattr(self.base_net, name)


def wrap_with_global_branch_residual(
    base_net: nn.Module,
    cfg: Optional[GlobalBranchResidualConfig] = None,
    **kwargs,
) -> GlobalBranchResidualWrapper:
    if cfg is None:
        cfg = GlobalBranchResidualConfig(**kwargs)
    return GlobalBranchResidualWrapper(base_net, cfg=cfg)
