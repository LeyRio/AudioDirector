#!/usr/bin/env bash
# Copyright 2024 Sony Research Inc.
# Modifications Copyright 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

set -euo pipefail

if [[ $# -lt 5 || $# -gt 6 ]]; then
  echo "Usage: $0 ASSET_ROOT VIDEO_DIR MANIFEST_TSV EVENTS_JSONL OUTPUT_ROOT [NUM_GPUS]" >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DATA_PREFIX="$1"
VIDEO_DIR="$2"
MANIFEST="$3"
EVENTS="$4"
OUTPUT_ROOT="$5"
NUM_GPUS="${6:-1}"

mkdir -p "$OUTPUT_ROOT/memmap" "$OUTPUT_ROOT/annotations" "$OUTPUT_ROOT/work"
cd "$ROOT_DIR"

torchrun --standalone --nproc_per_node="$NUM_GPUS" \
  data_pipeline/extract_video_training_latents.py \
  --video-dir "$VIDEO_DIR" \
  --manifest "$MANIFEST" \
  --output-dir "$OUTPUT_ROOT/memmap/train" \
  --work-dir "$OUTPUT_ROOT/work/video-latents" \
  --variant 44k \
  --vae-checkpoint "$DATA_PREFIX/checkpoints/v1-44.pth" \
  --synchformer-checkpoint "$DATA_PREFIX/checkpoints/synchformer_state_dict.pth"

python data_pipeline/extract_audio_tracks.py \
  --video-dir "$VIDEO_DIR" \
  --manifest "$MANIFEST" \
  --output-dir "$OUTPUT_ROOT/audio"

python data_pipeline/build_annotations.py \
  --input "$EVENTS" \
  --output-dir "$OUTPUT_ROOT/annotations/rag"
