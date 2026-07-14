from collections.abc import Mapping
import torch
import torch.distributed as dist


def _to_compare_tensor(x):
    """
    Convert DTensor / ShardedTensor / Tensor -> local comparable tensor
    """

    # DTensor
    if hasattr(x, "full_tensor"):
        try:
            x = x.full_tensor()
        except Exception:
            pass

    # ShardedTensor
    if hasattr(x, "local_shards"):
        try:
            local_shards = x.local_shards()

            if len(local_shards) == 1:
                x = local_shards[0].tensor
        except Exception:
            pass

    if torch.is_tensor(x):
        x = x.detach().cpu()

    return x


def compare_state_dict(sd1, sd2, prefix=""):
    """
    Recursively compare two FSDP state_dicts.
    """

    rank = dist.get_rank()

    keys1 = set(sd1.keys())
    keys2 = set(sd2.keys())

    only1 = sorted(keys1 - keys2)
    only2 = sorted(keys2 - keys1)

    if only1:
        print(f"\n[ONLY IN SD1] ({len(only1)})")
        for k in only1:
            print(f"  {k}")

    if only2:
        print(f"\n[ONLY IN SD2] ({len(only2)})")
        for k in only2:
            print(f"  {k}")

    common_keys = sorted(keys1 & keys2)

    mismatch_count = 0

    for k in common_keys:
        v1 = sd1[k]
        v2 = sd2[k]

        name = f"{prefix}.{k}" if prefix else k

        # nested dict
        if isinstance(v1, Mapping) and isinstance(v2, Mapping):
            mismatch_count += compare_state_dict(v1, v2, name)
            continue

        # normalize tensor
        v1 = _to_compare_tensor(v1)
        v2 = _to_compare_tensor(v2)

        # tensor compare
        if torch.is_tensor(v1) and torch.is_tensor(v2):

            if v1.shape != v2.shape:
                print(
                    f"[RANK {rank}] [SHAPE MISMATCH] "
                    f"{name}: {v1.shape} vs {v2.shape}"
                )
                mismatch_count += 1
                continue

            if v1.dtype != v2.dtype:
                print(
                    f"[RANK {rank}] [DTYPE MISMATCH] "
                    f"{name}: {v1.dtype} vs {v2.dtype}"
                )

            equal = torch.equal(v1, v2)

            if not equal:
                diff = (v1.float() - v2.float()).abs()

                max_diff = diff.max().item()
                mean_diff = diff.mean().item()

                # 找到最大diff位置
                max_idx = diff.argmax().item()

                print(
                    f"[RANK {rank}] [DIFF] {name}\n"
                    f"    shape      = {tuple(v1.shape)}\n"
                    f"    dtype      = {v1.dtype}\n"
                    f"    max_diff   = {max_diff:.10e}\n"
                    f"    mean_diff  = {mean_diff:.10e}\n"
                    f"    max_idx    = {max_idx}\n"
                    f"    v1[max]    = {v1.flatten()[max_idx].item():.10e}\n"
                    f"    v2[max]    = {v2.flatten()[max_idx].item():.10e}"
                )

                mismatch_count += 1

        else:
            # non tensor
            if v1 != v2:
                print(
                    f"[RANK {rank}] [VALUE MISMATCH] "
                    f"{name}: {v1} vs {v2}"
                )
                mismatch_count += 1

    return mismatch_count