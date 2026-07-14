"""Humanoid dataset registration."""

from torchtitan.experiments.humanoid.data.dataset import RiggedHumanoidJointOctreeDataset
from torchtitan.experiments.vem.dataloader import ParallelAwareDataloader


def build_humanoid_dataloader(dp_world_size, dp_rank, job_config, **kwargs):
    if job_config.training.dataset != "rigged-humanoid-joint-octree":
        raise ValueError(f"Unsupported humanoid dataset: {job_config.training.dataset}")
    dataset = RiggedHumanoidJointOctreeDataset(
        **job_config.training.dataset_kwargs,
        force_divisible_by=max(
            1,
            job_config.training.batch_size
            * max(1, job_config.training.num_workers)
            * dp_world_size,
        ),
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
    )
    return ParallelAwareDataloader(
        dataset=dataset,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        batch_size=job_config.training.batch_size,
        collate_fn=dataset.collate_fn,
        num_workers=job_config.training.num_workers,
        drop_last=job_config.training.drop_last,
        pin_memory=job_config.training.pin_memory,
    )


__all__ = ["build_humanoid_dataloader", "RiggedHumanoidJointOctreeDataset"]
