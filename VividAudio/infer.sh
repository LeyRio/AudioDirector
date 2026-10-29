#!/usr/bin/env bash
# Copyright 2024 Sony Research Inc.
# Modifications Copyright 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

set -euo pipefail

if [[ $# -lt 3 || $# -gt 5 ]]; then
  echo "Usage: $0 MODEL_PATH REQUESTS_JSONL OUTPUT_DIR [normal|router] [DATA_ROOT]" >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH="$1"
REQUESTS_JSONL="$2"
OUTPUT_DIR="$3"
MODE="${4:-normal}"
MODEL_VARIANT="${MODEL_VARIANT:-medium_44k}"

if [[ "${MODE}" != "normal" && "${MODE}" != "router" ]]; then
  echo "Mode must be 'normal' or 'router'." >&2
  exit 2
fi
if [[ ! -f "${MODEL_PATH}" ]]; then
  echo "Checkpoint does not exist: ${MODEL_PATH}" >&2
  exit 2
fi
if [[ ! -f "${REQUESTS_JSONL}" ]]; then
  echo "Request file does not exist: ${REQUESTS_JSONL}" >&2
  exit 2
fi
if [[ $# -eq 5 ]]; then
  export DATA_PREFIX="$5"
fi

ROUTER=false
if [[ "${MODE}" == "router" ]]; then
  ROUTER=true
fi

cd "${ROOT_DIR}"
exec python3 inference.py \
  "model=${MODEL_VARIANT}" \
  "model_path=${MODEL_PATH}" \
  "dataset_jsonl=${REQUESTS_JSONL}" \
  "output_dir=${OUTPUT_DIR}" \
  batch_size=1 \
  num_workers=0 \
  compile=false \
  delay_fusion=true \
  interleaved_rope=true \
  sync_rope=true \
  cross_attn_delay=true \
  global_rag_cond=true \
  use_decoupled_grag=true \
  grag_extractor_type=qformer \
  grag_qformer_n_q=32 \
  grag_qformer_n_layers=8 \
  "router_mode=${ROUTER}" \
  "per_event_local_rag=${ROUTER}"
