# Copyright 2024 Sony Research Inc.
# Modifications Copyright 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

class DummyCfg:
    """伪装 BEATs 的 cfg 对象，兼容原有代码对 beat_model.cfg.encoder_embed_dim 的调用"""
    def __init__(self, dim):
        self.encoder_embed_dim = dim

class TimbreExtractor(nn.Module):
    def __init__(self, model_type="beats", device="cuda", beats_ckpt_path=None):
        super().__init__()
        self.model_type = model_type
        self.device = device
        checkpoint_root = os.path.join(
            os.environ.get("DATA_PREFIX", "./data_root"), "checkpoints"
        )
        beats_ckpt_path = beats_ckpt_path or os.path.join(
            checkpoint_root, "BEATs_iter3_plus_AS2M.pt"
        )

        self.model_types = [t.strip() for t in model_type.split('+') if t.strip()]
        self.models = nn.ModuleDict()
        self.out_dim = 0

        for m_type in self.model_types:
            if m_type == "beats":
                from beats.BEATs import BEATs, BEATsConfig
                checkpoint = torch.load(beats_ckpt_path, map_location='cpu')
                beat_cfg = BEATsConfig(checkpoint['cfg'])
                model = BEATs(beat_cfg)
                model.load_state_dict(checkpoint['model'])
                self.models[m_type] = model
                self.out_dim += 768

            elif m_type == "wavlm":
                from transformers import WavLMModel
                model = WavLMModel.from_pretrained(
                    os.path.join(checkpoint_root, "wavlm-base-plus")
                )
                self.models[m_type] = model
                self.out_dim += model.config.hidden_size

            elif m_type == "wavlm_large":
                from transformers import WavLMModel
                model = WavLMModel.from_pretrained(
                    os.path.join(checkpoint_root, "wavlm-large")
                )
                self.models[m_type] = model
                self.out_dim += model.config.hidden_size

            elif m_type == "audiomae":
                from transformers import ASTModel, ASTFeatureExtractor
                audiomae_path = os.path.join(
                    checkpoint_root, "ast-finetuned-audioset-10-10-0.4593"
                )
                model = ASTModel.from_pretrained(audiomae_path)
                self.processor = ASTFeatureExtractor.from_pretrained(audiomae_path)
                self.models[m_type] = model
                self.out_dim += model.config.hidden_size

            elif m_type == "passt":
                from hear21passt.base import get_basic_model
                model = get_basic_model(mode="embed_only")
                self.models[m_type] = model
                self.out_dim += 768

            else:
                raise ValueError(f"Unsupported model_type: {m_type}")

        # 伪装原有的 BEATs Config，以便下游直接读取维度
        self.cfg = DummyCfg(self.out_dim)

        self.models = self.models.to(device)
        self.models.eval()

        # 冻结特征提取器的所有参数
        for param in self.models.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, waveform):
        feats = []
        for m_type in self.model_types:
            model = self.models[m_type]
            if m_type == "beats":
                padding_mask = torch.zeros_like(waveform, dtype=torch.bool, device=waveform.device)
                feat = model.extract_features(waveform, padding_mask=padding_mask)[0]

            elif m_type == "wavlm" or m_type == "wavlm_large":
                feat = model(waveform).last_hidden_state

            elif m_type == "audiomae":
                wav_np = waveform.cpu().numpy()
                inputs = self.processor(wav_np, sampling_rate=16000, return_tensors="pt")
                input_values = inputs.input_values.to(self.device)
                feat = model(input_values).last_hidden_state

            elif m_type == "passt":
                # PaSST 建议的输入采样率为 32k
                waveform_32k = F.interpolate(
                    waveform.unsqueeze(1),
                    scale_factor=2,
                    mode='linear',
                    align_corners=False
                ).squeeze(1)
                embeds = model.get_scene_embeddings(waveform_32k)
                feat = embeds.unsqueeze(1)  # (B, 1, D) 拓展时间维以兼容后续逻辑

            feats.append(feat)

        if len(feats) == 1:
            return feats[0]

        # ============= 多编码器 Early Fusion (按特征维拼接) =============
        max_T = max(f.shape[1] for f in feats)
        aligned_feats = []

        for f in feats:
            if f.shape[1] != max_T:
                if f.shape[1] == 1:
                    # 全局特征（如 passt），直接在时间维度上 expand
                    f_aligned = f.expand(-1, max_T, -1)
                else:
                    # 帧级特征如果长度存在细微差异（如 CNN 降采样截断），通过插值对齐
                    f_aligned = F.interpolate(
                        f.transpose(1, 2),
                        size=max_T,
                        mode='nearest-exact'
                    ).transpose(1, 2)
                aligned_feats.append(f_aligned)
            else:
                aligned_feats.append(f)

        # 沿特征维度拼接: (B, max_T, D_1 + D_2 + ...)
        return torch.cat(aligned_feats, dim=-1)

    def extract_features(self, waveform, padding_mask=None):
        """
        完全模拟 BEATs 的原始接口，让下游代码（如 extract_segment_features_vectorized）无感切换
        原代码调用: beat_model.extract_features(waveform)[0]
        """
        # 我们用列表包装一下，这样下游取 [0] 时刚好拿到 forward 返回的 Tensor
        return [self.forward(waveform)]
