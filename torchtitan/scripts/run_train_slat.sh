#!/usr/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

# number of GPUs per node
NGPU=${NGPU:-"8"}
# ranks to log
LOG_RANK=${LOG_RANK:-0}
# training config file
CONFIG_FILE=${CONFIG_FILE:-"exp/full/slat_tripo3_r64_13B_siglip_cc_dino.toml"}
# resume the newest checkpoint if exists
PRETRAIN_PATH=${PRETRAIN_PATH:-""}
# experiment name
NAME=${NAME:-"slat-13B-tripo3_458w-c32d8r64v0703patch2-shift2-rope4d-siglip2n20_cc_dinov2l-bs8px36"}
# log path
LOG_PATH=${LOG_PATH:-"./outputs/${NAME}"}

# checkpoint save path
CHECKPOINT_PATH=${CHECKPOINT_PATH:-"./outputs/${NAME}/ckpts"}
# uncomment to following the save checkpoints to jfs
# juicefs mount redis://:Jy7kgxm08k@redis.itjdzririeic.scs.bj.baidubce.com:24092/23 /jfs &
# CHECKPOINT_PATH=${CHECKPOINT_PATH:-"/jfs/YOUR_USER_NAME/torchtitan/outputs/${NAME}/ckpts"}

overrides=""
if [ $# -ne 0 ]; then
    overrides="$*"
fi

# network proxy
export http_proxy=http://192.168.48.17:18000 https_proxy=http://192.168.48.17:18000
# for huggingface model loading
export HF_HOME=/mnt/pfs/share/pretrained_model/.cache/huggingface/ HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DIFFUSERS_OFFLINE=1

# for BOS data loading, set your own BOS_ACCESS_KEY and BOS_SECRET_KEY
export BOS_ENDPOINT=bj.bcebos.com
export BOS_ACCESS_KEY=${BOS_ACCESS_KEY:-"YOUR_BOS_AK"}
export BOS_SECRET_KEY=${BOS_SECRET_KEY:-"YOUR_BOS_SK"}

# wandb logging, set your own WANDB_API_KEY and WANDB_PROJECT
export WANDB_API_KEY=${WANDB_API_KEY:-"YOUR_WANDB_API_KEY"}
export WANDB_PROJECT=${WANDB_PROJECT:-"YOUR_WANDB_PROJECT_NAME"}
export WANDB_ENTITY=${WANDB_ENTITY:-"vast-ai-research"}
export WANDB_NAME=$NAME

# fix flash attention3 torch.compile error
cp scripts/flash_attn_interface.py /usr/local/lib/python3.12/dist-packages/


PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE} \
torchrun --nproc_per_node=${NGPU} --rdzv_backend c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
--nnodes=${WORLD_SIZE} --node_rank=${RANK} --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} \
--local-ranks-filter ${LOG_RANK} --role rank --tee 3 \
-m torchtitan.experiments.trellis.train_slat --job.config_file ./torchtitan/experiments/trellis/configs/${CONFIG_FILE} \
--job.dump_folder=${LOG_PATH} \
--checkpoint.path_override=${CHECKPOINT_PATH} \
--checkpoint.load_path_override=${PRETRAIN_PATH} \
$overrides 
