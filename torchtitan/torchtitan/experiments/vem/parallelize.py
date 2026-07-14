from collections import defaultdict
import torch
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
    CheckpointWrapper
)
from torch.distributed._composable.replicate import replicate
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import CPUOffloadPolicy, fully_shard, MixedPrecisionPolicy
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
)

from torchtitan.config_manager import JobConfig, TORCH_DTYPE_MAP
from torchtitan.distributed import ParallelDims
from torchtitan.tools.logging import logger


def model_layers_iterator(model: nn.Module):
    for name, module in model.named_modules():
        if name.endswith(".layers") or name == "layers":
            yield name, module
        elif name.endswith(".pre_layers") or name == "pre_layers":
            yield name, module
        elif name.endswith(".post_layers") or name == "post_layers":
            yield name, module
        elif name.endswith(".dec_blocks") or name == "dec_blocks":
            yield name, module

def apply_compile(model: nn.Module):
    """
    Apply torch.compile to each TransformerBlock, which makes compilation efficient due to
    repeated structure. Alternatively one can compile the whole model (after applying DP).
    """

    for name, module in model_layers_iterator(model):
        for layer_id, layer in module.named_children():
            layer = torch.compile(layer, fullgraph=True)
            module.register_module(layer_id, layer)
        logger.info(f"Compiling each layer in {name} with torch.compile")

def parallelize(
    model: nn.Module,
    world_mesh: DeviceMesh,
    parallel_dims: ParallelDims,
    job_config: JobConfig,
):
    if parallel_dims.tp_enabled:
        if (
            job_config.parallelism.enable_async_tensor_parallel
            and not job_config.training.compile
        ):
            raise RuntimeError("Async TP requires --training.compile")

        apply_tp(
            model,
            world_mesh["tp"],
            loss_parallel=parallel_dims.loss_parallel_enabled,
            enable_async_tp=job_config.parallelism.enable_async_tensor_parallel,
        )

    if job_config.activation_checkpoint.mode != "none":
        apply_ac(model, job_config.activation_checkpoint)


    if job_config.training.compile:
        apply_compile(model)
        # NOTE: needed for torch.compile to work with dynamic shapes in token-choice MoE
        # Avoid unsupported data dependent operation in torch.compile  
        torch._dynamo.config.cache_size_limit = 100
        torch._dynamo.config.capture_dynamic_output_shape_ops = True
        torch._dynamo.config.capture_scalar_outputs = True


    dp_mesh: DeviceMesh | None = None
    if (
        parallel_dims.dp_shard_enabled or parallel_dims.cp_enabled
    ):  # apply FSDP or HSDP, potentially with Context Parallel
        if parallel_dims.dp_replicate_enabled:
            dp_mesh_dim_names = ("dp_replicate", "dp_shard_cp")
        else:
            dp_mesh_dim_names = ("dp_shard_cp",)
        dp_mesh = world_mesh[tuple(dp_mesh_dim_names)]

        apply_fsdp(
            model,
            dp_mesh,
            param_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_reduce],
            pp_enabled=parallel_dims.pp_enabled,
            cpu_offload=job_config.training.enable_cpu_offload,
            reshard_after_forward_policy=job_config.parallelism.fsdp_reshard_after_forward,
            cast_forward_inputs=job_config.training.fsdp_cast_forward_inputs,
        )

        if parallel_dims.dp_replicate_enabled:
            logger.info("Applied HSDP to the model")
        else:
            logger.info("Applied FSDP to the model")

        if parallel_dims.cp_enabled:
            logger.info("Applied Context Parallel to the model")

        if job_config.training.enable_cpu_offload:
            logger.info("Applied CPU Offloading to the model")
    elif parallel_dims.dp_replicate_enabled:
        if world_mesh.ndim > 1:
            raise RuntimeError("DDP has not supported > 1D parallelism")
        dp_mesh = world_mesh
        apply_ddp(
            model,
            dp_mesh,
            enable_compile=job_config.training.compile,
            enable_compiled_autograd=job_config.parallelism.enable_compiled_autograd,
        )

    
    if parallel_dims.pp_enabled:
        raise NotImplementedError

    return model


def parallelize_dmd(
    model: nn.Module,
    world_mesh: DeviceMesh,
    parallel_dims: ParallelDims,
    job_config: JobConfig,
):
    """
    Parallelize DMDModel - handles nested teacher/student/fake_score models.
    
    Each nested OctreeDiffusionModel is parallelized individually using the
    standard procedure, so they become FSDPModules like in normal training.
    This ensures that parameters are properly managed during forward pass
    (appear as regular Parameters, not raw DTensors).
    
    Args:
        model: DMDModel containing teacher, student, and fake_score submodules
        world_mesh: Device mesh for parallelism
        parallel_dims: Parallel dimension configuration
        job_config: Job configuration
    
    Returns:
        The parallelized model
    """
    # Apply full parallelization to each nested model
    # This gives them the same treatment as OctreeDiffusionWrapper in normal training
    for name in ["teacher", "student", "fake_score"]:
        nested_model = getattr(model, name)
        parallelize(nested_model, world_mesh, parallel_dims, job_config)
        logger.info(f"Applied parallelization to DMD {name} model")
    
    # NOTE: We intentionally do NOT wrap the parent DMDModel with FSDP.
    # Wrapping both parent and children causes issues with DCP's optimizer state_dict
    # because the FQN mapping gets confused with nested FSDP modules.
    # The children are already properly parallelized, which is sufficient for training.
    # For checkpointing, use model_weights_only=true in config, or implement custom
    # checkpoint handling for DMD models.
    
    return model


def apply_tp(
    model: nn.Module,
    tp_mesh: DeviceMesh,
    loss_parallel: bool,
    enable_async_tp: bool,
):
    """Apply tensor parallelism."""
    raise NotImplementedError


def apply_fsdp(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    pp_enabled: bool,
    cpu_offload: bool = False,
    cast_forward_inputs: bool = True,
    reshard_after_forward_policy: str = "default",
):
    """
    Apply data parallelism (via FSDP2) to the model.

    Args:
        model (nn.Module): The model to apply data parallelism to.
        dp_mesh (DeviceMesh): The device mesh to use for data parallelism.
        param_dtype (torch.dtype): The data type to use for model parameters.
        reduce_dtype (torch.dtype): The data type to use for reduction operations.
        pp_enabled (bool): Whether pipeline parallelism is enabled.
        cpu_offload (bool, optional): Whether to offload model parameters to CPU. Defaults to False.
        reshard_after_forward_policy (str, optional): The policy to use for resharding after forward pass. Defaults to "default".
            Other options: "never", "always".
            - "default" applies default resharding behavior, implementing "smart defaults" for known optimal scenarios.
            - "always" will enable `reshard_after_forward` for all forward passes.
            - "never" will disable `reshard_after_forward` for all forward passes.

    """
    mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype, cast_forward_inputs=cast_forward_inputs)
    fsdp_config = {"mesh": dp_mesh, "mp_policy": mp_policy}

    if cpu_offload:
        fsdp_config["offload_policy"] = CPUOffloadPolicy()

    # TODO: change model.blocks if needed
    for name, module in model_layers_iterator(model):
        for block_id, transformer_block in module.named_children():
            if reshard_after_forward_policy == "always":
                reshard_after_forward = True
            elif reshard_after_forward_policy == "never":
                reshard_after_forward = False
            elif reshard_after_forward_policy == "default":
                if pp_enabled:
                    # For PP, do not reshard after forward to avoid per-microbatch
                    # all-gathers, which can be expensive and non-overlapped
                    reshard_after_forward = False
                else:
                    # As an optimization, do not reshard after forward for the last
                    # transformer block since FSDP would prefetch it immediately
                    reshard_after_forward = int(block_id) < len(module) - 1
            else:
                raise ValueError(
                    f"Invalid reshard_after_forward_policy: {reshard_after_forward_policy}."
                )
            fully_shard(
                transformer_block,
                **fsdp_config,
                reshard_after_forward=reshard_after_forward,
            )

    fully_shard(model, **fsdp_config, reshard_after_forward=not pp_enabled)


def apply_ddp(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    enable_compile: bool,
    enable_compiled_autograd: bool,
):
    if enable_compile:
        if enable_compiled_autograd:
            torch._dynamo.config.optimize_ddp = (
                "python_reducer_without_compiled_forward"
            )
        else:
            torch._dynamo.config.optimize_ddp = "ddp_optimizer"

    replicate(model, device_mesh=dp_mesh, bucket_cap_mb=100)

    logger.info("Applied DDP to the model")


_save_list = {
    torch.ops.aten.mm.default,
    torch.ops.aten._scaled_dot_product_efficient_attention.default,
    torch.ops.aten._scaled_dot_product_flash_attention.default,
    torch.ops._c10d_functional.reduce_scatter_tensor.default,
    # for low precision training, it's useful to always save
    # the result of max, since the absolute maximum is
    # used to compute the scaling factor for quantization.
    torch.ops.aten.max.default,
    torch.ops.flash_attn,
    torch.ops.flash_attn3
}

def is_float(x: str):
    try:
        float(x)
        return True
    except ValueError:
        return False

def _apply_ac_to_transformer_block(num_blocks: int, module: nn.Module, ac_config):
    valid_ac_modes = ("full", "selective")
    if ac_config.mode not in valid_ac_modes:
        raise ValueError(
            f"Invalid AC mode: {ac_config.mode}. Valid modes: {valid_ac_modes}"
        )

    if ac_config.mode == "full":
        return ptd_checkpoint_wrapper(module, preserve_rng_state=False)

    assert ac_config.mode == "selective", f"{ac_config.mode}"
    use_op_sac = ac_config.selective_ac_option == "op"
    if isinstance(ac_config.selective_ac_option, str):
        if ac_config.selective_ac_option.isdigit():
            ac_config.selective_ac_option = int(ac_config.selective_ac_option)
        elif is_float(ac_config.selective_ac_option):
            ac_config.selective_ac_option = float(ac_config.selective_ac_option)
    use_layer_sac = isinstance(ac_config.selective_ac_option, int) or isinstance(ac_config.selective_ac_option, float)
    if not use_op_sac and not use_layer_sac:
        raise ValueError(
            f"Invalid selective AC option: {ac_config.selective_ac_option}. "
            f"Valid options: 'op' or a positive int representing layer frequency"
        )
    if use_op_sac:
        from torch.utils.checkpoint import (
            CheckpointPolicy,
            create_selective_checkpoint_contexts,
        )

        def _get_custom_policy(meta):
            def _custom_policy(ctx, func, *args, **kwargs):
                mode = "recompute" if ctx.is_recompute else "forward"
                mm_count_key = f"{mode}_mm_count"
                if func == torch.ops.aten.mm.default:
                    meta[mm_count_key] += 1
                # Saves output of all compute ops, except every second mm
                to_save = func in _save_list and not (
                    func == torch.ops.aten.mm.default and meta[mm_count_key] % 2 == 0
                )
                return (
                    CheckpointPolicy.MUST_SAVE
                    if to_save
                    else CheckpointPolicy.PREFER_RECOMPUTE
                )

            return _custom_policy

        def selective_checkpointing_context_fn():
            meta = defaultdict(int)
            return create_selective_checkpoint_contexts(_get_custom_policy(meta))

        return ptd_checkpoint_wrapper(
            module,
            context_fn=selective_checkpointing_context_fn,
            preserve_rng_state=False,
        )
    elif use_layer_sac:   
        # Checkpoint every `ac_freq` of the modules passed to this function
        if isinstance(ac_config.selective_ac_option, int):
            ac_freq = int(ac_config.selective_ac_option)
            ptd_checkpoint_wrapper.__dict__.setdefault("_count", 0)
            ptd_checkpoint_wrapper._count += 1
            if not ac_freq or ptd_checkpoint_wrapper._count % ac_freq == 0:
                return ptd_checkpoint_wrapper(module, preserve_rng_state=False)
            else:
                return module
        elif isinstance(ac_config.selective_ac_option, float):
            ac_count = int(num_blocks * ac_config.selective_ac_option)
            ptd_checkpoint_wrapper.__dict__.setdefault("_count", 0)
            ptd_checkpoint_wrapper._count += 1
            if ptd_checkpoint_wrapper._count <= ac_count:
                return ptd_checkpoint_wrapper(module, preserve_rng_state=False)
            else:
                return module


def apply_ac(model: nn.Module, ac_config):
    """Apply activation checkpointing to the model."""
    for name, module in model_layers_iterator(model):
        logger.info(f"Applied {ac_config.mode} to {name} with {len(module)} layers")
        for layer_id, transformer_block in module.named_children():
            transformer_block = _apply_ac_to_transformer_block(len(module), transformer_block, ac_config)
            module.register_module(layer_id, transformer_block)

    logger.info(f"Applied {ac_config.mode} activation checkpointing to the model")


def is_ac_applied(module):
    return isinstance(module, CheckpointWrapper)


def apply_ac_to_module(module_parent: nn.Module, layer_id: int, module: nn.Module):
    if is_ac_applied(module):
        return
    module = ptd_checkpoint_wrapper(module, preserve_rng_state=False)
    module_parent.register_module(layer_id, module)


def disable_ac_on_module(module_parent: nn.Module, layer_id: int, module: nn.Module):
    if not is_ac_applied(module):
        return
    module = module._checkpoint_wrapped_module
    module_parent.register_module(layer_id, module)


def apply_partial_ac_on_the_fly(model: nn.Module, ac_ratio: float):
    for name, module in model_layers_iterator(model):
        num_layers = len(module)
        num_ac_layers = int(ac_ratio * num_layers)
        for li, (layer_id, transformer_block) in enumerate(module.named_children()):
            if li < num_ac_layers:
                apply_ac_to_module(module, layer_id, transformer_block)
            else:
                disable_ac_on_module(module, layer_id, transformer_block)
