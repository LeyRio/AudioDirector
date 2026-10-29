# Copyright 2024 Sony Research Inc.
# Modifications Copyright 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import logging
import math
import random
from datetime import timedelta

import hydra
import numpy as np
import torch
import torch.distributed as distributed
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict
from torch.distributed.elastic.multiprocessing.errors import record

from foleydirector.data.data_setup import setup_training_datasets, update_threshold
from foleydirector.model.sequence_config import CONFIG_16K, CONFIG_44K
from foleydirector.runner import Runner
from foleydirector.utils.dist_utils import info_if_rank_zero, local_rank, world_size
from foleydirector.utils.logger import TensorboardLogger

from open_clip import create_model_from_pretrained
import open_clip
import torch.nn.functional as F

def patch_clip(clip_model):
    # a hack to make it output last hidden states
    # https://github.com/mlfoundations/open_clip/blob/fc5a37b72d705f760ebbc7915b84729816ed471f/src/open_clip/model.py#L269
    def new_encode_text(self, text, normalize: bool = False):
        cast_dtype = self.transformer.get_cast_dtype()

        x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.to(cast_dtype)
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = self.ln_final(x)  # [batch_size, n_ctx, transformer.width]
        return F.normalize(x, dim=-1) if normalize else x

    clip_model.encode_text = new_encode_text.__get__(clip_model)
    return clip_model

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

log = logging.getLogger()


def distributed_setup():
    distributed.init_process_group(backend="nccl", timeout=timedelta(hours=2))
    log.info(f'Initialized: local_rank={local_rank}, world_size={world_size}')
    return local_rank, world_size

@torch.inference_mode()
def encode_text(text: list[str], tokenizer, clip_model) -> torch.Tensor:
    tokens = tokenizer(text).cuda()
    return clip_model.encode_text(tokens, normalize=True)

@torch.inference_mode()
def encode_text_pool(text: list[str], tokenizer, clip_model) -> torch.Tensor:
    tokens = tokenizer(text).cuda()
    sequence_output = clip_model.encode_text(tokens, normalize=True)  # (b, 77, 1024)
    attention_mask = (tokens != 0).float()  # (b, 77)
    attention_mask = attention_mask.unsqueeze(-1)  # (b, 77, 1)
    sum_embeddings = torch.sum(sequence_output * attention_mask, dim=1)  # (b, 1024)
    sum_mask = torch.sum(attention_mask, dim=1)  # (b, 1)
    sum_mask = torch.clamp(sum_mask, min=1e-9)
    pooled_output = sum_embeddings / sum_mask  # (b, 1024)

    return pooled_output.unsqueeze(1)  # (b, 1, 1024)



# def process_fgc_feature(fgc_feature, tokenizer, clip_model, fgc_pool=False, fgc_pt=10):
#     b = len(fgc_feature[0])   # batch size
#     all_batch_feats = []

#     for bi in range(b):
#         texts_8 = []
#         for t in range(8):
#             texts_8.append(fgc_feature[t][bi])

#         feats = encode_text(texts_8, tokenizer, clip_model)  # (8, D)
#         feats_10 = feats[:, :fgc_pt, :]      # (8, 10, D)
#         feats_10 = feats_10.reshape(8 * fgc_pt, -1)  # (80, D)
#         all_batch_feats.append(feats_10)

#     return torch.stack(all_batch_feats, dim=0)   # (b, 80, D)


def process_fgc_feature(fgc_feature, tokenizer, clip_model, fgc_pool=False, fgc_pt=10):

    b = len(fgc_feature[0])   # batch size
    all_batch_feats = []

    for bi in range(b):
        texts_8 = [fgc_feature[t][bi] for t in range(8)]

        feats = encode_text(texts_8, tokenizer, clip_model)  # (8, T, D) or (8, D)
        if feats.ndim == 2:
            feats = feats.unsqueeze(1)  # 变成 (8, 1, D) for consistency

        if fgc_pool:
            pooled_feats = feats.mean(dim=1)  # (8, D)
            feats_10 = pooled_feats.unsqueeze(1).expand(-1, fgc_pt, -1)  # (8, fgc_pt, D)
        else:
            feats_10 = feats[:, :fgc_pt, :]  # (8, fgc_pt, D)

        feats_10 = feats_10.reshape(8 * fgc_pt, -1)
        all_batch_feats.append(feats_10)

    return torch.stack(all_batch_feats, dim=0)  # (b, 8*fgc_pt, D)


def process_fgc_feature_v2(v2_cap, v2_tr, tokenizer, clip_model, fgc_pt=10):
    total_tokens = 8 * fgc_pt

    bs = v2_tr.shape[0]
    all_batch_feats = []
    valid = [1 for i in range(bs)]
    for b in range(bs):
        categories = [i[b] for i in v2_cap]
        tr_mask = v2_tr[b]
        all_empty = all(all(not cap.strip() for cap in cap_tuple) for cap_tuple in categories)
        valid[b] = int(not all_empty)

        valid_texts = categories
        all_categories = valid_texts + [""]
        category_feats = encode_text_pool(all_categories, tokenizer, clip_model)  # b 77 1024
        category_embeddings = {}
        for i, category in enumerate(all_categories):
            category_embeddings[category] = category_feats[i]

        token_features = torch.zeros(total_tokens, category_feats.shape[-1]).to(category_feats[0].device)  # (80, D)
        token_features[:] = category_embeddings[""]  # 初始化为背景

        for i in range(len(valid_texts)):
            text = valid_texts[i]
            mask = tr_mask[i]  # (80,) bool tensor

            if text:
                new_positions = mask

                if new_positions.any():
                    token_features[new_positions] = category_embeddings[text]

        all_batch_feats.append(token_features)

    return torch.stack(all_batch_feats, dim=0), valid

@record
@hydra.main(version_base='1.3.2', config_path='configs', config_name='train.yaml')
def train(cfg: DictConfig):
    # initial setup
    torch.cuda.set_device(local_rank)
    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
    distributed_setup()
    num_gpus = world_size
    run_dir = HydraConfig.get().run.dir

    # patch data dim
    if cfg.model.endswith('16k'):
        seq_cfg = CONFIG_16K
    elif cfg.model.endswith('44k'):
        seq_cfg = CONFIG_44K
    else:
        raise ValueError(f'Unknown model: {cfg.model}')

    with open_dict(cfg):
        cfg.data_dim.latent_seq_len = seq_cfg.latent_seq_len
        cfg.data_dim.clip_seq_len = seq_cfg.clip_seq_len
        cfg.data_dim.sync_seq_len = seq_cfg.sync_seq_len

    # wrap python logger with a tensorboard logger
    log = TensorboardLogger(cfg.exp_id,
                            run_dir,
                            logging.getLogger(),
                            is_rank0=(local_rank == 0),
                            enable_email=cfg.enable_email and not cfg.debug)

    info_if_rank_zero(log, f'All configuration: {cfg}')
    info_if_rank_zero(log, f'Number of GPUs detected: {num_gpus}')

    # number of dataloader workers
    info_if_rank_zero(log, f'Number of dataloader workers (per GPU): {cfg.num_workers}')

    # Set seeds to ensure the same initialization
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    # setting up configurations
    info_if_rank_zero(log, f'Training configuration: {cfg}')
    cfg.batch_size //= num_gpus
    info_if_rank_zero(log, f'Batch size (per GPU): {cfg.batch_size}')

    # determine time to change max skip
    total_iterations = cfg['num_iterations']

    # setup datasets
    dataset, sampler, loader = setup_training_datasets(cfg)
    info_if_rank_zero(log, f'Number of training samples: {len(dataset)}')
    info_if_rank_zero(log, f'Number of training batches: {len(loader)}')

    # val_dataset, val_loader, eval_loader = setup_val_datasets(cfg)
    # info_if_rank_zero(log, f'Number of val samples: {len(val_dataset)}')
    # val_cfg = cfg.data.ExtractedVGG_valfgc

    # compute and set mean and std
    latent_mean, latent_std = dataset.compute_latent_stats()

    # construct the trainer
    trainer = Runner(cfg,
                     log=log,
                     run_path=run_dir,
                     for_training=True,
                     latent_mean=latent_mean,
                     latent_std=latent_std).enter_train()
    # load previous checkpoint if needed
    if cfg['checkpoint'] is not None:
        curr_iter = trainer.load_checkpoint(cfg['checkpoint'])
        cfg['checkpoint'] = None
        info_if_rank_zero(log, 'Model checkpoint loaded!')
    else:
        # if run_dir exists, load the latest checkpoint
        checkpoint = trainer.get_latest_checkpoint_path()
        if checkpoint is not None:
            curr_iter = trainer.load_checkpoint(checkpoint)
            info_if_rank_zero(log, 'Latest checkpoint loaded!')
        else:
            # load previous network weights if needed
            curr_iter = 0
            if cfg['weights'] is not None:
                info_if_rank_zero(log, 'Loading weights from the disk')
                trainer.load_weights(cfg['weights'])
                cfg['weights'] = None

    # determine max epoch
    total_epoch = math.ceil(total_iterations / len(loader))
    current_epoch = curr_iter // len(loader)
    info_if_rank_zero(log, f'We will approximately use {total_epoch} epochs.')
    if not cfg.train_full:
        for name, param in trainer.network.named_parameters():
            if "fgc" in name:
                param.requires_grad = True
            elif ("clip" in name and not cfg.sep_fgc) or ("clip" in name and cfg.clip_train): # attn的情况需要进一步调一下clip
                param.requires_grad = True
            elif "fusion" in name and not cfg.sep_fgc:
                param.requires_grad = True
            elif "delay" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    # 加载CLIP模型
    clip_model = create_model_from_pretrained(
        f'local-dir:{cfg.clip_model_dir}', return_transform=False).cuda()
    clip_model = patch_clip(clip_model)
    tokenizer = open_clip.get_tokenizer(cfg.clip_tokenizer_dir)

    # training loop
    # try:
        # Need this to select random bases in different workers
    dynamic_iters = cfg.dynamic_iterations
    np.random.seed(np.random.randint(2**30 - 1) + local_rank * 1000)
    while curr_iter < total_iterations:
        # Crucial for randomness!
        sampler.set_epoch(current_epoch)
        current_epoch += 1
        log.debug(f'Current epoch: {current_epoch}')
        if cfg.dynamic_threshold:
            new_threshold = cfg.dt_list[min(current_epoch % dynamic_iters, len(cfg.dt_list) - 1)]
            dataset, sampler, loader = update_threshold(cfg, dataset, new_threshold)

        # print(len(dataset))

        trainer.enter_train()
        trainer.log.data_timer.start()
        for data in loader:
            fgc_feature = process_fgc_feature(data['fgc_feature'], tokenizer, clip_model, cfg.fgc_pool, fgc_pt=cfg.fgc_pt).detach().cpu()
            fgc_feature_v2, valid = process_fgc_feature_v2(data['v2_cap'], data['v2_tr'], tokenizer, clip_model, fgc_pt=cfg.fgc_pt)
            fgc_feature_v2 = fgc_feature_v2.detach().cpu()
            valid_tensor = torch.tensor(valid, dtype=torch.bool, device=fgc_feature.device)
            fgc_feature_input = fgc_feature
            if random.random() < cfg.v2_ratio:
                fgc_feature_input[valid_tensor] = fgc_feature_v2[valid_tensor]
            data['fgc_feature'] = fgc_feature_input

            trainer.train_pass(data, curr_iter)

            # if (curr_iter + 1) % cfg.val_interval == 0:
            #     # swap into a eval rng state, i.e., use the same seed for every validation pass
            #     train_rng_snapshot = trainer.rng.graphsafe_get_state()
            #     trainer.rng.graphsafe_set_state(eval_rng_clone)
            #     info_if_rank_zero(log, f'Iteration {curr_iter} val: validating')
            #     for data in val_loader:
            #         fgc_feature = process_fgc_feature(data['fgc_feature'], tokenizer, clip_model).detach().cpu()
            #         data['fgc_feature'] = fgc_feature
            #         trainer.validation_pass(data, curr_iter)
            #     distributed.barrier()
            #     trainer.val_integrator.finalize('val', curr_iter, ignore_timer=True)
            #     trainer.rng.graphsafe_set_state(train_rng_snapshot)

            # if (curr_iter + 1) % cfg.eval_interval == 0:
            #     save_eval = (curr_iter + 1) % cfg.save_eval_interval == 0
            #     train_rng_snapshot = trainer.rng.graphsafe_get_state()
            #     trainer.rng.graphsafe_set_state(eval_rng_clone)
            #     info_if_rank_zero(log, f'Iteration {curr_iter} eval: validating')
            #     for data in eval_loader:
            #         fgc_feature = process_fgc_feature(data['fgc_feature'], tokenizer, clip_model).detach().cpu()
            #         data['fgc_feature'] = fgc_feature
            #         audio_path = trainer.inference_pass(data,
            #                                             curr_iter,
            #                                             val_cfg,
            #                                             save_eval=save_eval)
            #     distributed.barrier()
            #     trainer.rng.graphsafe_set_state(train_rng_snapshot)
            #     trainer.eval(audio_path, curr_iter, val_cfg)

            curr_iter += 1

            if curr_iter >= total_iterations:
                break
    # except Exception as e:
    #     log.error(f'Error occurred at iteration {curr_iter}!')
    #     log.critical(e.message if hasattr(e, 'message') else str(e))
    #     raise
    # finally:
    #     if not cfg.debug:
    #         trainer.save_checkpoint(curr_iter)
    #         trainer.save_weights(curr_iter)

    trainer.save_checkpoint(curr_iter)
    trainer.save_weights(curr_iter)
    del trainer
    torch.cuda.empty_cache()

    distributed.barrier()

    # clean-up
    log.complete()
    distributed.barrier()
    distributed.destroy_process_group()


if __name__ == '__main__':
    train()
