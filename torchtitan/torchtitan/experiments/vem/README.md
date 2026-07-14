# Torchtitan x Trellis

This folder contains the Trellis training code based on torchtitan.

## Configuration

Configuration files under `configs/` have fields matching the following two specs:

- `JobConfig` defined in `torchtitan/config_manager.py`
- `torchtitan/experiments/trellis/extra_args.py`

You can extend `extra_args.py` to support more options.

Network structure is specified by the `[model]` field in the config file, and it looks up `torchtitan/experiments/trellis/__init__.py` to find the corresponding network definition.

## Training

Example training scripts: `scripts/run_train_structure.sh` and `scripts/run_train_slat.sh`.

A basic training loop for diffusion models is implemented by the `DiffusionTrainer` class in `torchtitan/experiments/trellis/diffusion_trainer.py`.Most of the time we don't need to modify this file.

For structure and slat diffusion model training (a.k.a the 1st and 2rd stage of the Trellis model), there're `torchtitan/experiments/trellis/train_structure.py` and `torchtitan/experiments/trellis/train_slat.py` which extend the `DiffusionTrainer` class and implement the following necessary functions:

- `prepare_extra_modules`: initialize modules besides the transformer here, typically condition encoders
- `get_latents`: get the latents from the data batch (note that in this framework we assume pre-cached latents for training)
- `prepare_conditions`: get the conditions from the data batch and encode them
- `forward_model`: given noisy latents and conditions, call the transformer model to get the prediction

Several things to pay attention to:

- Initialize all weights and buffers (for example the RoPE module) in the `init_weights` functions of the model. See `torchtitan/experiments/trellis/model/transformer.py` for an example. If you do not initialize here, for example you do the initialization in `__init__` you'll have zero-initialized everything because the tensors are all in meta device after `__init__` which do not have meaningful values.
- Set `data_parallel_replicate_degree=1` and `data_parallel_shard_degree=-1` in the config to use FSDP. You can also set `data_parallel_replicate_degree=NUM_NODES` and `data_parallel_shard_degree=NUM_GPUS_PER_NODE` to use HSDP (FSDP in node and DDP across nodes).
- If you resume from pretrained checkpoints, there're certain circumstances where you should exclude certain components or keys from loading. If you add/change/remove parameters, you should exclude `optimizer` from loading by setting `exclude_from_loading=["optimizer"]` and exclude related model keys from loading by setting (for example) `load_model_exclude_keys=["image_embedder", "attn2.to_k", "attn2.to_v"]`. If you resume using a different number of GPUs or change the data list, you should exclude `dataloader` from loading by setting `exclude_from_loading=["dataloader"]`. Note that it's better to use a different shuffle seed when excluding loading `dataloader` to prevent training on the exact same data.
- The checkpoints can be huge for large models. The training script provides the option to save checkpoints to BOS using juicefs. There's a BOS-based file system created using juicefs and you can mount it inside the pod. After mounted just treat this file system as any regular file system. But note that resuming checkpoints from this file system is problematic, so you need to copy the checkpoint to local directory and resume from there.
- Running the training script using a single GPU (setting `NGPU=1`) uses float32, not mixed precision (of course flash attention still runs in bfloat16).
- Monitor the MFU during training, which indicates how well you're utilizing the GPU compute. You should have >40% MFU all the time otherwise you need to find a way to improve training efficiency. This criteria applies to large models only since small models (<1B) can hardly achieve 40% MFU.

Example training command:
```sh
# debug
NGPU=1 BOS_ACCESS_KEY=YOUR_BOS_AK BOS_SECRET_KEY=YOUR_BOS_SK WANDB_API_KEY= WANDB_PROJECT=SOME_PROJECT_NAME sh scripts/run_train_structure.sh
# real training
BOS_ACCESS_KEY=YOUR_BOS_AK BOS_SECRET_KEY=YOUR_BOS_SK WANDB_API_KEY=YOUR_WANDB_API_KEY WANDB_PROJECT=SOME_PROJECT_NAME sh scripts/run_train_structure.sh
```

# Testing
This framework DOES NOT do testing during training. You can test manually using the provided testing scripts. See `scripts/test_trellis_ss_diffusion.py` and `scripts/test_trellis_slat_diffusion.py` for details.

# Data
This framework uses `IterableDataset` instead of regular fixed length dataset. To support multi-worker loading, we need to assign the data to each worker before training. We first shard the data to each rank, and shard the rank data to each worker. See `torchtitan/experiments/trellis/dataset/structure.py` for details.
