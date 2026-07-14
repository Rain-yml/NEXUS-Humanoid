export BOS_ACCESS_KEY=""
export BOS_SECRET_KEY=""
export WANDB_API_KEY=""

NAME=spacediff_debug_bs256 CONFIG_FILE=smdiff/default.toml \
    NGPU=8 sh launch_script/run_train_smdiff_singlenode.sh \
    --training.batch_size=32 \
    --optimizer.lr=1e-4 \
    --metrics.no-enable-wandb \
    --activation_checkpoint.mode=none \
    --scheduler.logit_mean=0.0 \
    --training.steps=200000 \
    --checkpoint.persistent_steps '100000' \
    --ema.enabled --ema.update_after_step=10000 --ema.beta=0.9998 --ema.update_every=5
