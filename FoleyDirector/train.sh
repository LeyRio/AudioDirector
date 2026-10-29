#!/usr/bin/env bash
# Copyright 2024 Sony Research Inc.
# Modifications Copyright 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 ASSET_ROOT DATA_ROOT [INITIAL_WEIGHTS] [NUM_GPUS] [HYDRA_OVERRIDES...]" >&2
  exit 2
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DATA_PREFIX="$1"
export DATA_ROOT="$2"
WEIGHTS="${3:-}"
NUM_GPUS="${4:-1}"
ORIGINAL_ARGC=$#
if [[ $ORIGINAL_ARGC -ge 4 ]]; then
  shift 4
else
  shift "$ORIGINAL_ARGC"
fi

cd "$ROOT_DIR"
ARGS=("$@")
if [[ -n "$WEIGHTS" ]]; then
  ARGS+=("weights=$WEIGHTS")
fi
torchrun --standalone --nproc_per_node="$NUM_GPUS" train.py "${ARGS[@]}"
