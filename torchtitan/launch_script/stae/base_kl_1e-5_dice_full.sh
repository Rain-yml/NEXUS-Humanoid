export BOS_ACCESS_KEY=""
export BOS_SECRET_KEY=""
export WANDB_API_KEY=""

NAME=mesh_stae_pred_kl_1e-5_dice_gate_full2 NGPU=8 CONFIG_FILE=stae/default.toml \
    sh launch_script/run_train_stae_singlenode.sh \
    --model.flavor=base-kl-1e-5-gate-dice-full-2 \
    --training.batch_size=8 \
    --training.dataset_kwargs.no_random_negative \
    --optimizer.lr=1e-4 \
    --training.steps=80000 \
    --training.mixed_precision_param=float32 \
    --metrics.no-enable-wandb