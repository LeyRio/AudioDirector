#!/usr/bin/env bash
# Copyright 2024 Sony Research Inc.
# Modifications Copyright 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

if [[ $# -lt 3 || $# -gt 5 ]]; then
  echo "Usage: $0 MODEL_PATH VIDEO_PATH TRACKS_JSON [OUTPUT_DIR] [DATA_ROOT]" >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH="$1"
VIDEO_PATH="$2"
TRACKS_JSON="$3"
OUTPUT_DIR="${4:-${ROOT_DIR}/output}"
MODEL_VARIANT="${MODEL_VARIANT:-medium_44k}"

if [[ $# -eq 5 ]]; then
  export DATA_PREFIX="$5"
fi

cd "${ROOT_DIR}"
exec python3 infer.py \
  --model-path "${MODEL_PATH}" \
  --video "${VIDEO_PATH}" \
  --tracks "${TRACKS_JSON}" \
  --output "${OUTPUT_DIR}" \
  --variant "${MODEL_VARIANT}"
