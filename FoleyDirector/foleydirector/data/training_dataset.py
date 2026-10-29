# Copyright 2024 Sony Research Inc.
# Modifications Copyright 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from typing import Union

import pandas as pd
import torch
from tensordict import TensorDict
from torch.utils.data import Dataset


class FoleyTrainingDataset(Dataset):
    """Memory-mapped audiovisual latents paired with eight-slot FGC labels."""

    def __init__(
        self,
        manifest: Union[str, Path],
        memmap_dir: Union[str, Path],
        annotations: Union[str, Path],
        data_dim: dict[str, int],
        *,
        fgc_pt: int = 10,
        drop_ratio: float = 0.0,
        video_drop_ratio: float = 0.0,
        text_drop_ratio: float = 0.0,
        with_details: bool = True,
    ):
        self.rows = pd.read_csv(manifest, sep="\t").to_dict("records")
        self.ids = [str(row["id"]) for row in self.rows]
        self.row_index = {sample_id: i for i, sample_id in enumerate(self.ids)}
        self.annotations = self._load_annotations(Path(annotations))
        self.ids = [sample_id for sample_id in self.ids if sample_id in self.annotations]
        if not self.ids:
            raise ValueError("No manifest IDs have matching FGC annotations")

        tensors = TensorDict.load_memmap(memmap_dir)
        self.mean = tensors["mean"]
        self.std = tensors["std"]
        self.clip_features = tensors["clip_features"]
        self.sync_features = tensors["sync_features"]
        self.text_features = tensors["text_features"]
        self.fgc_pt = int(fgc_pt)
        self.drop_ratio = float(drop_ratio)
        self.video_drop_ratio = float(video_drop_ratio)
        self.text_drop_ratio = float(text_drop_ratio)
        self.with_details = bool(with_details)
        self._validate_shapes(data_dim)

    @staticmethod
    def _load_annotations(path: Path) -> dict[str, dict]:
        records = []
        with path.open() as handle:
            for line in handle:
                if line.strip():
                    records.append(json.loads(line))
        output = {}
        for record in records:
            sample_id = str(record.get("id") or record.get("audio_name", "")).rsplit(".", 1)[0]
            if sample_id:
                output[sample_id] = record
        return output

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
        indices = torch.tensor([self.row_index[sample_id] for sample_id in self.ids])
        latents = self.mean[indices]
        return latents.mean(dim=(0, 1)), latents.std(dim=(0, 1))

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def _clean(value) -> str:
        if value is None:
            return ""
        value = str(value).strip()
        return "" if value.lower() in {"none", "no", "null"} else value

    def _slots(self, record):
        if "slots" in record:
            return list(record["slots"])[:8]
        result = record.get("result", {})
        return [result.get(str(i)) or {} for i in range(8)]

    def __getitem__(self, index):
        sample_id = self.ids[index]
        tensor_index = self.row_index[sample_id]
        record = self.annotations[sample_id]
        slots = self._slots(record)
        slots.extend({} for _ in range(8 - len(slots)))

        fgc_feature = []
        for slot in slots:
            event = self._clean(slot.get("event", slot.get("Event")))
            loudness = self._clean(slot.get("loudness", slot.get("Loudness")))
            timbre = self._clean(slot.get("timbre", slot.get("Timbre")))
            details = [value for value in (event, loudness, timbre) if value]
            text = ", ".join(details if self.with_details else details[:1])
            if torch.rand(()) < self.drop_ratio:
                text = ""
            fgc_feature.append(text)

        v2_cap = [""] * 8
        v2_tr = torch.zeros(8, 8 * self.fgc_pt, dtype=torch.bool)
        for event_index, event in enumerate(record.get("events", [])[:8]):
            v2_cap[event_index] = self._clean(event.get("label", event.get("event")))
            for start, end in event.get("segments", []):
                start_idx = max(0, int(float(start) * self.fgc_pt))
                end_idx = min(8 * self.fgc_pt, int(float(end) * self.fgc_pt))
                if start_idx < end_idx:
                    v2_tr[event_index, start_idx:end_idx] = True

        return {
            "id": sample_id,
            "a_mean": self.mean[tensor_index],
            "a_std": self.std[tensor_index],
            "clip_features": self.clip_features[tensor_index],
            "sync_features": self.sync_features[tensor_index],
            "text_features": self.text_features[tensor_index],
            "video_exist": torch.rand(()) >= self.video_drop_ratio,
            "text_exist": torch.rand(()) >= self.text_drop_ratio,
            "fgc_feature": fgc_feature,
            "v2_cap": v2_cap,
            "v2_tr": v2_tr,
        }
