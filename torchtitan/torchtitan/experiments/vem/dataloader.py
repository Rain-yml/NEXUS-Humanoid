# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.

import pickle
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any
import math

import numpy as np
import torch
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset
from torchdata.stateful_dataloader import StatefulDataLoader
from torchtitan.tools.logging import logger
import gc


class BaseDataLoader(Stateful, ABC):
    """Base class for all dataloaders.

    This is used to enforce that all dataloaders have the methods defined in ``Stateful``,
    ``state_dict()`` and ``load_state_dict()``.
    """

    @abstractmethod
    def __iter__(self):
        ...


def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset  # the dataset copy in this worker process
    np.random.seed(worker_id)
    for data_key_to_shard in dataset.worker_shard_data:
        data_to_shard = getattr(dataset, data_key_to_shard)
        assert isinstance(data_to_shard, list)
        data_size = len(data_to_shard)
        per_worker = int(math.ceil(data_size / float(worker_info.num_workers)))
        worker_id = worker_info.id
        data_to_shard = data_to_shard[worker_id * per_worker:(worker_id + 1) * per_worker]
        setattr(dataset, data_key_to_shard, data_to_shard)
        logger.info(f"Worker {worker_id} has {len(data_to_shard)} {data_key_to_shard}")
    
    if hasattr(dataset, 'worker_shard_nested_data'):
        for nested_data_key_to_shard in dataset.worker_shard_nested_data:
            nested_data_to_shard = getattr(dataset, nested_data_key_to_shard)
            assert isinstance(nested_data_to_shard, list)
            for i in range(len(nested_data_to_shard)):
                data_to_shard = nested_data_to_shard[i]
                assert isinstance(data_to_shard, list)
                data_size = len(data_to_shard)
                start = int(worker_id * data_size / worker_info.num_workers)
                end = int((worker_id + 1) * data_size / worker_info.num_workers)
                data_to_shard = data_to_shard[start:end]
                nested_data_to_shard[i] = data_to_shard
            setattr(dataset, nested_data_key_to_shard, nested_data_to_shard)
            logger.info(f"Worker {worker_id} has nested {nested_data_key_to_shard} with lengths {[len(nested_data_to_shard[i]) for i in range(len(nested_data_to_shard))]}")
    
    if not gc.isenabled():
        logger.info(f"Enabling GC in dataloader worker {worker_id}")
        gc.enable()


class ParallelAwareDataloader(StatefulDataLoader, BaseDataLoader):
    """Dataloader that is aware of distributed data parallelism.

    This dataloader is used to load data in a distributed data parallel fashion. It also
    utilizes ``torchdata.stateful_dataloader.StatefulDataLoader`` to implement the necessary
    methods such as ``__iter__``.

    Args:
        dataset (IterableDataset): The dataset to iterate over.
        dp_rank: Data parallelism rank for this dataloader.
        dp_world_size: The world size of the data parallelism.
        batch_size: The batch size to use for each iteration.
        collate_fn: Optional function to collate samples in a batch.
    """

    dp_rank: int
    dp_world_size: int
    batch_size: int

    def __init__(
        self,
        dataset: IterableDataset,
        dp_rank: int,
        dp_world_size: int,
        batch_size: int,
        num_workers: int = 0,
        drop_last: bool = False,
        pin_memory: bool = False,
        collate_fn: Callable | None = None,
        **kwargs
    ):
        self.dp_world_size = dp_world_size
        self.dp_rank = dp_rank
        self.batch_size = batch_size
        super().__init__(dataset, batch_size, collate_fn=collate_fn, num_workers=num_workers, drop_last=drop_last, pin_memory=pin_memory, worker_init_fn=worker_init_fn, **kwargs)
        self._rank_id = f"dp_rank_{dp_rank}"

    def state_dict(self) -> dict[str, Any]:
        # Store state only for dp rank to avoid replicating the same state across other dimensions.
        return {
            # We don't have to use pickle as DCP will serialize the state_dict. However,
            # we have to keep this for backward compatibility.
            self._rank_id: pickle.dumps(super().state_dict()),
            "world_size": self.dp_world_size,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        # State being empty is valid.
        if not state_dict:
            return

        if self._rank_id not in state_dict:
            logger.warning(
                f"DataLoader state is empty for dp rank {self.dp_rank}, "
                "expected key {self._rank_id}"
            )
            return

        assert self.dp_world_size == state_dict["world_size"], (
            "dp_degree is inconsistent before and after checkpoint, "
            "dataloader resharding is not supported yet."
        )
        # We don't have to use pickle as DCP will serialize the state_dict. However, we have to
        # keep this for backward compatibility.
        super().load_state_dict(pickle.loads(state_dict[self._rank_id]))
