from types import SimpleNamespace

import torch

from torchtitan.experiments.humanoid.pipelines.image_mesh_to_joint_octree import (
    ImageMeshToJointOctreePipeline,
    TeacherForcedMeshLayer,
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
