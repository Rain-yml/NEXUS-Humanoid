from types import SimpleNamespace

import torch

from torchtitan.experiments.humanoid.pipelines.image_mesh_to_joint_octree import (
    ImageMeshToJointOctreePipeline,
    TeacherForcedMeshLayer,
)
from torchtitan.experiments.humanoid.pipelines.image_mesh_to_single_stream_joint_octree import (
    ImageMeshToSingleStreamJointOctreePipeline,
    SingleStreamTeacherForcedMeshLayer,
)


class _Scheduler:
    config = SimpleNamespace(num_train_timesteps=1000)

    def set_timesteps(self, _steps, device):
        self.timesteps = torch.tensor([750.0, 250.0], device=device)

    def step(self, prediction, _timestep, _sample, **_kwargs):
        return (prediction,)


class _Model(torch.nn.Module):
    def __init__(self, mesh_prediction_value: float):
        super().__init__()
        self.mesh_prediction_value = mesh_prediction_value
        self.calls = []

    def forward(self, **kwargs):
        self.calls.append(
            {
                "mesh_x_t": kwargs["mesh_x_t"].detach().clone(),
                "mesh_centers": kwargs["mesh_centers"].detach().clone(),
                "mesh_t": kwargs["mesh_t"].detach().clone(),
            }
        )
        mesh_prediction = torch.full_like(
            kwargs["mesh_x_t"], self.mesh_prediction_value
        )
        joint_prediction = torch.zeros_like(kwargs["joint_x_t"])
        joint_prediction[:, 5] = 1.0
        return mesh_prediction, joint_prediction


def _run(mesh_prediction_value: float):
    model = _Model(mesh_prediction_value)
    pipeline = ImageMeshToJointOctreePipeline(
        image_encoder=torch.nn.Identity(),
        octree_dit=model,
        scheduler=None,
    )
    mesh_layer = TeacherForcedMeshLayer(
        centers=torch.tensor([[8, 8, 8], [4, 4, 4]]),
        occupancy=torch.tensor(
            [[1, -1, 1, -1, 1, -1, 1, -1], [-1, 1, -1, 1, -1, 1, -1, 1]],
            dtype=torch.float32,
        ),
        depth=0,
    )
    joints = pipeline.sample_layer(
        scheduler=_Scheduler(),
        mesh_layer=mesh_layer,
        joint_centers=torch.full((3, 3), 8, dtype=torch.long),
        joint_ids=torch.arange(3),
        device=torch.device("cpu"),
        dtype=torch.float32,
        generator=torch.Generator().manual_seed(7),
        sub_voxel_size=8,
        num_inference_steps=2,
        image_embeds=torch.zeros((1, 1, 4)),
        guidance_scale=1.0,
        num_vertices=2,
        prediction="v",
    )
    return mesh_layer, model, joints


def test_gt_mesh_is_teacher_forced_and_mesh_prediction_is_ignored():
    mesh_layer, first_model, first_joints = _run(-1000.0)
    _, second_model, second_joints = _run(1000.0)

    torch.testing.assert_close(first_joints, second_joints)
    for call in first_model.calls + second_model.calls:
        torch.testing.assert_close(call["mesh_centers"], mesh_layer.centers)

    inferred_noise = []
    for call in first_model.calls:
        sigma = call["mesh_t"][0] / 1000.0
        inferred_noise.append(
            (call["mesh_x_t"] - (1.0 - sigma) * mesh_layer.occupancy) / sigma
        )
    torch.testing.assert_close(inferred_noise[0], inferred_noise[1])


class _ImageEncoder(torch.nn.Module):
    def preprocess(self, image):
        return image

    def forward(self, image):
        return torch.zeros((image.shape[0], 1, 4), dtype=image.dtype)


class _SingleStreamModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.joint_ids = []

    def forward(self, **kwargs):
        self.joint_ids.append(kwargs["joint_ids"].detach().clone())
        prediction = torch.zeros_like(kwargs["x_t"])
        prediction[kwargs["joint_mask"], 5] = 1.0
        return prediction


def test_single_stream_pipeline_preserves_explicit_joint_ids():
    model = _SingleStreamModel()
    pipeline = ImageMeshToSingleStreamJointOctreePipeline(
        image_encoder=_ImageEncoder(),
        octree_dit=model,
        scheduler=None,
    )
    mesh_layer = SingleStreamTeacherForcedMeshLayer(
        centers=torch.tensor([[8, 8, 8], [4, 4, 4]]),
        occupancy=torch.tensor(
            [[1, -1, 1, -1, 1, -1, 1, -1], [-1, 1, -1, 1, -1, 1, -1, 1]],
            dtype=torch.float32,
        ),
        depth=0,
    )
    requested_joint_ids = torch.tensor([0, 5, 27])
    result = pipeline(
        image=torch.zeros((1, 3, 4, 4)),
        mesh_layers=[mesh_layer],
        scheduler=_Scheduler(),
        device=torch.device("cpu"),
        num_inference_steps=2,
        generator=torch.Generator().manual_seed(7),
        num_vertices=2,
        enable_progress=False,
        grid_size=16,
        dtype=torch.float32,
        prediction="v",
        joint_ids=requested_joint_ids,
    )

    torch.testing.assert_close(result.joint_ids, requested_joint_ids)
    assert result.joints.shape == (3, 3)
    expected_token_ids = torch.tensor([-1, -1, 0, 5, 27])
    for token_ids in model.joint_ids:
        torch.testing.assert_close(token_ids, expected_token_ids)
