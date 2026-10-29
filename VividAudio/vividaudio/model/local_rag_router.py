# Copyright 2024 Sony Research Inc.
# Modifications Copyright 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from vividaudio.model.networks import PreprocessedConditions
from vividaudio.model.transformer_layers import attention


@dataclass
class LocalRagRouterConfig:
    """Inference-only slot-wise router for the local RAG cross-attention path."""

    enabled: bool = False
    alpha: float = 1.0
    start_joint_block: int = 0
    end_joint_block: int = -1


class LocalRagRouterWrapper(nn.Module):
    """Route each latent slot to its own event-local RAG token bank.

    This wrapper supports the delayed cross-attention architecture only. It
    keeps the latent/clip/text trunk shared, skips the original all-event local
    injection in selected joint blocks, then fuses event-specific local
    attention outputs before one shared ``delay_block.post_attention`` call.
    """

    def __init__(
        self,
        base_net: nn.Module,
        cfg: Optional[LocalRagRouterConfig] = None,
    ) -> None:
        super().__init__()
        self.base_net = base_net
        self.router_cfg = cfg or LocalRagRouterConfig()
        self.per_event_local_rag_stack = None
        self.per_event_local_slot_mask = None

    def preprocess_conditions(self, *args, **kwargs) -> PreprocessedConditions:
        conditions = self.base_net.preprocess_conditions(*args, **kwargs)
        stack = self.per_event_local_rag_stack
        slot_mask = self.per_event_local_slot_mask
        if stack is None or slot_mask is None:
            return conditions

        batch, events, token_len, channels = stack.shape
        target_len = conditions.rag_f.shape[1]
        local_raw = stack.reshape(batch * events, token_len, channels)
        if token_len != target_len:
            local_raw = F.interpolate(
                local_raw.transpose(1, 2),
                size=target_len,
                mode="nearest-exact",
            ).transpose(1, 2)
        local_tokens = self.base_net.rag_input_proj(local_raw).reshape(
            batch,
            events,
            target_len,
            -1,
        )
        # Keep this inference-only data outside the baseline model's condition
        # schema so the unwrapped network remains unchanged.
        conditions.local_rag_f_per_event = local_tokens
        conditions.local_slot_mask = slot_mask
        return conditions

    def get_empty_conditions(self, *args, **kwargs) -> PreprocessedConditions:
        return self.base_net.get_empty_conditions(*args, **kwargs)

    def normalize(self, *args, **kwargs):
        return self.base_net.normalize(*args, **kwargs)

    def unnormalize(self, *args, **kwargs):
        return self.base_net.unnormalize(*args, **kwargs)

    def load_weights(self, *args, **kwargs):
        return self.base_net.load_weights(*args, **kwargs)

    def load_weights_fgc(self, *args, **kwargs):
        return self.base_net.load_weights_fgc(*args, **kwargs)

    def _joint_block_selected(self, index: int, total_blocks: int) -> bool:
        start = max(0, int(self.router_cfg.start_joint_block))
        end = int(self.router_cfg.end_joint_block)
        if end < 0:
            end = total_blocks
        return start <= index < min(total_blocks, end)

    def _router_ready(self, conditions: PreprocessedConditions) -> bool:
        return (
            self.router_cfg.enabled
            and getattr(conditions, "local_rag_f_per_event", None) is not None
            and getattr(conditions, "local_slot_mask", None) is not None
            and bool(getattr(self.base_net, "delay_fusion", False))
            and bool(getattr(self.base_net, "cross_attn_delay", False))
        )

    def _prepare_latent_input(self, latent: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if not self.base_net.inpainting:
            return self.base_net.audio_input_proj(latent)

        xt = latent[..., :self.base_net.latent_dim]
        inp_scale = getattr(self.base_net, "inpainting_scale", 1.0)
        inp_scale_map = getattr(self.base_net, "inpainting_scale_map", None)
        if inp_scale_map is not None:
            inp_scale = inp_scale_map.to(xt.dtype)

        inp_fn = getattr(self.base_net, "inpainting_scale_fn", None)
        if inp_fn is not None:
            gate = inp_fn(t)
            if isinstance(gate, torch.Tensor):
                while gate.dim() < 3:
                    gate = gate.unsqueeze(-1)
                inp_scale = inp_scale * gate.to(xt.dtype)

        residual = self.base_net.inpainting_mlp(latent)
        channel_scale = getattr(self.base_net, "inpainting_channel_scale", None)
        if channel_scale is not None:
            channel_scale = torch.as_tensor(
                channel_scale,
                device=xt.device,
                dtype=xt.dtype,
            )
            while channel_scale.dim() < 3:
                channel_scale = channel_scale.unsqueeze(0)
            residual = residual * channel_scale

        if self.base_net.bypass_empty_inpainting:
            mask = latent[..., -1:]
            has_inpainting = (mask.amax(dim=1, keepdim=True) > 0).to(xt.dtype)
            residual = residual * has_inpainting
        return self.base_net.audio_input_proj(xt) + inp_scale * residual

    def _compute_conditions(
        self,
        conditions: PreprocessedConditions,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        global_input = conditions.clip_f_c + conditions.text_f_c
        if self.base_net.global_rag_cond and conditions.rag_f_c is not None:
            global_input = global_input + conditions.rag_f_c

        global_c = self.base_net.global_cond_mlp(global_input)
        global_c = self.base_net.t_embed(t).unsqueeze(1) + global_c.unsqueeze(1)
        extended_c = global_c + conditions.sync_f

        # Preserve existing per-event global routing when it is independently
        # enabled, while this wrapper routes the local condition.
        global_styles = getattr(conditions, "rag_f_c_per_event", None)
        global_mask = getattr(conditions, "slot_mask", None)
        if global_styles is not None and global_mask is not None and self.base_net.global_rag_cond:
            style_at_t = torch.einsum("bkd,bkn->bnd", global_styles, global_mask)
            base_ct = (conditions.clip_f_c + conditions.text_f_c).unsqueeze(1)
            global_c_t = self.base_net.global_cond_mlp(base_ct + style_at_t)
            global_c_t = self.base_net.t_embed(t).unsqueeze(1) + global_c_t
            extended_c = global_c_t + conditions.sync_f
        return global_c, extended_c

    def _local_attention_mask(
        self,
        slot_mask: torch.Tensor,
        event_index: int,
        query_len: int,
        key_len: int,
        heads: int,
        device: torch.device,
    ) -> torch.Tensor:
        batch = slot_mask.shape[0]
        source_mask = F.interpolate(
            slot_mask[:, event_index:event_index + 1, :],
            size=key_len,
            mode="nearest-exact",
        ).squeeze(1) > 1e-6
        # Padded event slots would otherwise produce an all-False attention row.
        empty_sources = ~source_mask.any(dim=1)
        source_mask[empty_sources, 0] = True
        allowed = source_mask[:, None, :].expand(batch, query_len, key_len)

        local_window = int(getattr(self.base_net, "local_window_size", 0))
        if local_window > 0:
            query_pos = torch.arange(query_len, device=device).float()
            key_pos = torch.arange(key_len, device=device).float()
            distance = torch.abs(
                (query_pos * (key_len / query_len)).unsqueeze(1)
                - key_pos.unsqueeze(0)
            )
            allowed = allowed & (distance <= local_window).unsqueeze(0)
        return allowed.unsqueeze(1).expand(batch, heads, -1, -1).contiguous()

    def _inject_slotwise_local_rag(
        self,
        block: nn.Module,
        latent: torch.Tensor,
        global_c: torch.Tensor,
        extended_c: torch.Tensor,
        local_tokens: torch.Tensor,
        slot_mask: torch.Tensor,
    ) -> torch.Tensor:
        if not getattr(block, "cross_attn_delay", False):
            raise RuntimeError("LocalRagRouterWrapper requires cross_attn_delay=True")
        if not getattr(block, "sync_rope", False):
            raise RuntimeError("LocalRagRouterWrapper currently requires sync_rope=True")

        latent_qkv, latent_mod = block.delay_block.pre_attention(
            latent,
            extended_c,
            self.base_net.latent_rot,
        )
        query, _, _ = latent_qkv
        routed_attention = torch.zeros_like(latent)
        events = local_tokens.shape[1]
        for event_index in range(events):
            local_qkv, _ = block.fgc_block.pre_attention(
                local_tokens[:, event_index],
                global_c,
                self.base_net.fgc_rot,
            )
            local_mask = self._local_attention_mask(
                slot_mask,
                event_index,
                query.shape[2],
                local_qkv[1].shape[2],
                query.shape[1],
                query.device,
            )
            event_attention = attention(
                query,
                local_qkv[1],
                local_qkv[2],
                mask=local_mask,
            )
            event_weight = slot_mask[:, event_index, :].unsqueeze(-1).to(event_attention.dtype)
            routed_attention = routed_attention + event_weight * event_attention

        routed_attention = routed_attention * block.rag_gate * float(self.router_cfg.alpha)
        return block.delay_block.post_attention(latent, routed_attention, latent_mod)

    def _predict_flow_slotwise_local(
        self,
        latent: torch.Tensor,
        t: torch.Tensor,
        conditions: PreprocessedConditions,
        return_inter: bool = False,
    ) -> torch.Tensor:
        latent = self._prepare_latent_input(latent, t)
        global_c, extended_c = self._compute_conditions(conditions, t)
        clip_f, text_f, rag_f = conditions.clip_f, conditions.text_f, conditions.rag_f
        local_tokens = conditions.local_rag_f_per_event
        slot_mask = conditions.local_slot_mask
        total_blocks = len(self.base_net.joint_blocks)
        inter_latent = None

        for index, block in enumerate(self.base_net.joint_blocks):
            selected = self._joint_block_selected(index, total_blocks)
            latent, clip_f, text_f, rag_f = block(
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
                skip_delay_fusion=selected,
            )
            if selected:
                latent = self._inject_slotwise_local_rag(
                    block,
                    latent,
                    global_c,
                    extended_c,
                    local_tokens,
                    slot_mask,
                )
            if index == total_blocks // 2:
                inter_latent = latent

        if inter_latent is None:
            inter_latent = latent
        for block in self.base_net.fused_blocks:
            latent = block(latent, extended_c, self.base_net.latent_rot)

        flow = self.base_net.final_layer(latent, extended_c)
        if return_inter:
            return flow, self.base_net.repa_proj(inter_latent)
        return flow

    def predict_flow(
        self,
        latent: torch.Tensor,
        t: torch.Tensor,
        conditions: PreprocessedConditions,
        return_inter: bool = False,
    ) -> torch.Tensor:
        if not self._router_ready(conditions):
            return self.base_net.predict_flow(latent, t, conditions, return_inter=return_inter)
        return self._predict_flow_slotwise_local(latent, t, conditions, return_inter=return_inter)

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
        rag_only_conditions = PreprocessedConditions(
            clip_f=empty_conditions.clip_f,
            sync_f=empty_conditions.sync_f,
            text_f=empty_conditions.text_f,
            clip_f_c=empty_conditions.clip_f_c,
            text_f_c=empty_conditions.text_f_c,
            rag_f=conditions.rag_f,
        )
        empty_conditions.rag_f = neg_conditions.rag_f
        flow_rag = self.predict_flow(latent, t, rag_only_conditions)
        flow_empty = self.predict_flow(latent, t, empty_conditions)
        return cfg_rag_weight * (
            cfg_strength * flow_cond + (1.0 - cfg_strength) * flow_rag
        ) + (1.0 - cfg_rag_weight) * (
            cfg_strength * flow_cond + (1.0 - cfg_strength) * flow_empty
        )

    def forward(self, *args, **kwargs):
        return self.base_net.forward(*args, **kwargs)

    def __getattr__(self, name: str):
        if name in {"base_net", "router_cfg"}:
            return super().__getattr__(name)
        return getattr(self.base_net, name)


def wrap_with_local_rag_router(
    base_net: nn.Module,
    cfg: Optional[LocalRagRouterConfig] = None,
    **kwargs,
) -> LocalRagRouterWrapper:
    if cfg is None:
        cfg = LocalRagRouterConfig(**kwargs)
    return LocalRagRouterWrapper(base_net, cfg=cfg)
