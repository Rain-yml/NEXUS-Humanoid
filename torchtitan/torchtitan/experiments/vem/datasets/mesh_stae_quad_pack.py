from typing import List, Optional, Tuple
import random
import traceback

from torchtitan.tools.logging import logger
from torchtitan.experiments.vem.datasets.bos import BOSClient
from torchtitan.experiments.vem.datasets.json_utils import load_json
from torchtitan.experiments.vem.datasets.mesh_utils import MeshProcessor
from torchtitan.experiments.vem.datasets.mesh_stae_quad import (
    SpaceTimeQuadAEDataset,
    shard_interleave,
)

from torchtitan.experiments.vem.datasets.path_io import DatasetPathIO


class SpaceTimeQuadAEPackDataset(SpaceTimeQuadAEDataset):
    worker_shard_data = ["batches"]

    def __init__(
        self,
        batches_packed: str,
        repeats: int = 1,
        shuffle_seed: int = 0,
        aug_flip: bool = True,
        aug_rotate_all: bool = False,
        aug_rotate_z: bool = True,
        aug_scale: bool = False,
        aug_scale_range: List[float] = [0.8, 1.2],
        yup_to_zup: bool = False,
        vertex_noise: float = 0.0,
        extra_feat: str = "none",
        include_face: bool = False,
        include_face_orient: bool = False,
        vertex_resolutions: List[int] = [-1],
        vertex_position_type: str = "none",
        face_negative: str = "random",
        mode: str = "tri_connect",
        diag_as_edge: bool = True,
        return_mesh_mixed: bool = False,
        # auto assigned args
        infinite: bool = True,
        force_divisible_by: int = 1,
        dp_rank: int = 0,
        dp_world_size: int = 1,
    ) -> None:
        self.repeats = repeats
        self.shuffle_seed = shuffle_seed
        self.infinite = infinite
        self.dp_rank = dp_rank
        self.dp_world_size = dp_world_size
        self.use_bos = True
        assert vertex_position_type in ["none", "int"]
        assert face_negative in ["none", "random"]
        assert mode in ["tri", "tri_connect", "native_quad", "native_quad_wireframe", "tri_bi_connect"]
        if not include_face:
            raise ValueError("SpaceTimeQuadAEPackDataset requires include_face=True")
        if vertex_position_type == "int" and not any(v > 0 for v in vertex_resolutions):
            raise ValueError("vertex_position_type='int' requires a positive vertex_resolution")
        self.vertex_position_type = vertex_position_type

        self.path_io = DatasetPathIO()
        self.sample_idx = 0

        batches_packed_json = load_json(batches_packed)
        batches = batches_packed_json["batches"]
        self._validate_batches(batches)
        batches = batches * repeats
        if force_divisible_by > 1:
            batches = batches[: len(batches) // force_divisible_by * force_divisible_by]
            logger.info(f"Batch number after dropping: {len(batches)}")

        batches = shard_interleave(batches, dp_world_size, dp_rank)

        rng = random.Random(shuffle_seed)
        rng.shuffle(batches)

        print("rank", dp_rank, batches[:10])

        self.batches = batches
        self.packed = True

        self.mp = MeshProcessor()
        self.aug_flip = aug_flip
        self.aug_rotate_all = aug_rotate_all
        self.aug_rotate_z = aug_rotate_z
        self.aug_scale = aug_scale
        self.aug_scale_range = aug_scale_range
        self.yup_to_zup = yup_to_zup
        self.vertex_noise = vertex_noise
        self.include_face = include_face
        self.vertex_resolutions = vertex_resolutions
        self.include_face_orient = include_face_orient
        self.extra_feat = extra_feat
        self.face_negative = face_negative
        self.mode = mode
        self.diag_as_edge = diag_as_edge
        self.return_mesh_mixed = return_mesh_mixed
        assert self.extra_feat in ["none", "normal", "face_normal"]
        print(
            "Augmentations",
            "flip",
            aug_flip,
            "rotate_all",
            aug_rotate_all,
            "rotate_z",
            aug_rotate_z,
            "scale",
            aug_scale,
            "scale_range",
            aug_scale_range,
        )
        print("dp_rank", dp_rank, "dp_world_size", dp_world_size)

    @staticmethod
    def _validate_batches(batches: List[List[Tuple[str, str, str]]]) -> None:
        for batch_idx, batch in enumerate(batches):
            if not isinstance(batch, list):
                raise ValueError(f"Packed batch {batch_idx} must be a list")
            for record_idx, record in enumerate(batch):
                if not isinstance(record, (list, tuple)) or len(record) != 3:
                    raise ValueError(
                        f"Packed batch {batch_idx} record {record_idx} must be "
                        "a (uuid, bucket, bos_path) triple"
                    )
                if not all(isinstance(value, str) for value in record):
                    raise ValueError(
                        f"Packed batch {batch_idx} record {record_idx} triple values must be strings"
                    )

    def __iter__(self):
        while True:
            sample_idx = self.sample_idx

            while True:
                try:
                    yield [self.get_data(record) for record in self.batches[sample_idx]]
                    break
                except GeneratorExit:
                    raise
                except:
                    logger.warning(
                        f"Failed to load data for batch {self.batches[sample_idx]}: "
                        f"{traceback.format_exc()}"
                    )
                    sample_idx = random.randint(0, len(self.batches) - 1)

            self.sample_idx += 1

            if self.sample_idx >= len(self.batches):
                if not self.infinite:
                    logger.warning("Dataset has run out of data.")
                    break
                else:
                    self.sample_idx = 0
                    logger.warning("Dataset is being re-looped.")

    def load_state_dict(self, state_dict):
        self.sample_idx = state_dict["sample_idx"]
        self.batches = state_dict["batches"]

    def state_dict(self):
        return {
            "batches": self.batches,
            "sample_idx": self.sample_idx,
        }
