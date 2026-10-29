# Copyright 2024 Sony Research Inc.
# Modifications Copyright 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

def get_parameter_groups(model, cfg, print_log=False):
    """Return the trainable parameters in an optimizer-compatible group."""
    del print_log  # Reserved for compatibility with existing callers.
    other_params = []
    memo = set()
    for param in model.parameters():
        if not param.requires_grad or param in memo:
            continue
        memo.add(param)
        other_params.append(param)

    return [
        {
            'params': other_params,
            'lr': cfg.learning_rate,
            'weight_decay': cfg.weight_decay,
        },
    ]
