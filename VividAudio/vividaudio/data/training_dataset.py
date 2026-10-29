# Copyright 2024 Sony Research Inc.
# Modifications Copyright 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import json
from pathlib import Path
from typing import Union

import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
from tensordict import TensorDict
from torch.utils.data import Dataset


class VividTrainingDataset(Dataset):
    """Memory-mapped audiovisual latents paired with event-level RAG data."""

    def __init__(
        self,
        manifest: Union[str, Path],
        memmap_dir: Union[str, Path],
        annotations_dir: Union[str, Path],
        audio_dir: Union[str, Path],
        data_dim: dict[str, int],
        *,
        feature_dir: Union[str, Path, None] = None,
        w_prefetch: bool = False,
        max_events: int = 10,
        video_drop_ratio: float = 0.0,
        text_drop_ratio: float = 0.0,
    ):
        self.rows = pd.read_csv(manifest, sep="\t").to_dict("records")
        self.ids = [str(row["id"]) for row in self.rows]
        self.row_index = {sample_id: i for i, sample_id in enumerate(self.ids)}
        self.annotations_dir = Path(annotations_dir)
        self.audio_dir = Path(audio_dir)
        self.feature_dir = Path(feature_dir) if feature_dir else None
        self.w_prefetch = bool(w_prefetch)
        self.max_events = int(max_events)
        self.video_drop_ratio = float(video_drop_ratio)
        self.text_drop_ratio = float(text_drop_ratio)
        if self.w_prefetch and self.feature_dir is None:
            raise ValueError("feature_dir is required when w_prefetch=true")

        tensors = TensorDict.load_memmap(memmap_dir)
        self.mean = tensors["mean"]
        self.std = tensors["std"]
        self.clip_features = tensors["clip_features"]
        self.sync_features = tensors["sync_features"]
        self.text_features = tensors["text_features"]
        self._validate_shapes(data_dim)

    def _validate_shapes(self, data_dim):
        expected = {
            "mean": (self.mean, "latent_seq_len"),
            "std": (self.std, "latent_seq_len"),
            "clip_features": (self.clip_features, "clip_seq_len"),
            "sync_features": (self.sync_features, "sync_seq_len"),
            "text_features": (self.text_features, "text_seq_len"),
        }
        for name, (tensor, key) in expected.items():
            if tensor.shape[1] != data_dim[key]:
                raise ValueError(f"{name} length {tensor.shape[1]} != {key} {data_dim[key]}")

    def compute_latent_stats(self):
        return self.mean.mean(dim=(0, 1)), self.mean.std(dim=(0, 1))

    def __len__(self):
        return len(self.ids)

    def _events(self, sample_id):
        path = self.annotations_dir / f"{sample_id}_events.json"
        with path.open() as handle:
            events = json.load(handle).get("events", [])[:self.max_events]
        events.extend(
            {
                "start": -1,
                "end": -1,
                "energy": -1,
                "desc": "",
                "group": -1,
            }
            for _ in range(self.max_events - len(events))
        )
        return path, events

    def _waveform(self, sample_id):
        path = self.audio_dir / f"{sample_id}.wav"
        waveform, sample_rate = torchaudio.load(path)
        waveform = waveform.mean(dim=0, keepdim=True)
        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
        target_samples = 8 * 16000
        tail_samples = 2 * 16000
        tail = waveform[:, target_samples:target_samples + tail_samples]
        if tail.shape[1] == 0:
            tail = waveform[:, -tail_samples:]
        tail = F.pad(tail, (0, max(0, tail_samples - tail.shape[1])))[:, :tail_samples]
        waveform = F.pad(
            waveform, (0, max(0, target_samples - waveform.shape[1])))[:, :target_samples]
        return waveform, tail

    def __getitem__(self, index):
        sample_id = self.ids[index]
        annotation_path, events = self._events(sample_id)
        starts = torch.tensor([event.get("start", -1) for event in events], dtype=torch.float32)
        ends = torch.tensor([event.get("end", -1) for event in events], dtype=torch.float32)
        row = self.rows[index]
        data = {
            "id": sample_id,
            "a_mean": self.mean[index],
            "a_std": self.std[index],
            "clip_features": self.clip_features[index],
            "sync_features": self.sync_features[index],
            "text_features": self.text_features[index],
            "caption": str(row.get("label", row.get("caption", ""))),
            "video_exist": torch.rand(()) >= self.video_drop_ratio,
            "text_exist": torch.rand(()) >= self.text_drop_ratio,
            "segments": torch.stack((starts, ends), dim=1),
            "energy": torch.tensor(
                [event.get("energy", -1) for event in events], dtype=torch.float32),
            "valid_mask": (starts >= 0) & (ends > starts),
            "rag_desc": [str(event.get("desc", "")) for event in events],
            "audio_dir": str(self.audio_dir),
            "json_path": str(annotation_path),
            "group_idx": torch.tensor(
                [
                    -1 if event.get("group", -1) is None else event.get("group", -1)
                    for event in events
                ],
                dtype=torch.int64,
            ),
        }
        if self.w_prefetch:
            data["features"] = torch.load(
                self.feature_dir / f"{sample_id}_beats.pt",
                map_location="cpu",
                weights_only=True,
            )
            data["rms"] = torch.load(
                self.feature_dir / f"{sample_id}_rms.pt",
                map_location="cpu",
                weights_only=True,
            )
        else:
            data["waveform"], data["tail_waveform"] = self._waveform(sample_id)
        return data
