from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple, Union

import torch
from diffusers.schedulers import DPMSolverMultistepScheduler, FlowMatchEulerDiscreteScheduler
from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteSchedulerOutput,
)


@dataclass
class RolloutBatch:
    x0: torch.Tensor
    noise: torch.Tensor
    latents: torch.Tensor
    timesteps: torch.Tensor
    next_timesteps: torch.Tensor


class FlowMatchEulerStochasticDiscreteScheduler(FlowMatchEulerDiscreteScheduler):
    """FlowMatch Euler scheduler with the stochastic update used by V2F scripts."""

    stochastic_sampling = True

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        sample: torch.FloatTensor,
        s_churn: float = 0.0,
        s_tmin: float = 0.0,
        s_tmax: float = float("inf"),
        s_noise: float = 1.0,
        generator: torch.Generator | None = None,
        return_dict: bool = True,
    ) -> Union[FlowMatchEulerDiscreteSchedulerOutput, Tuple]:
        del s_churn, s_tmin, s_tmax, s_noise

        if self.step_index is None:
            self._init_step_index(timestep)

        sample = sample.to(torch.float32)
        sigma = self.sigmas[self.step_index]
        sigma_next = self.sigmas[self.step_index + 1]
        x0 = sample - sigma * model_output
        noise = torch.empty_like(sample).normal_(generator=generator)
        prev_sample = ((1.0 - sigma_next) * x0 + sigma_next * noise).to(model_output.dtype)

        self._step_index += 1

        if not return_dict:
            return (prev_sample,)
        return FlowMatchEulerDiscreteSchedulerOutput(prev_sample=prev_sample)


def build_rollout_scheduler(
    sampler_type: Literal["euler", "dpm", "euler_sto"] | str,
    num_train_timesteps: int,
    shift: float = 1.0,
):
    if sampler_type == "euler":
        return FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=num_train_timesteps,
            shift=shift,
        )
    if sampler_type == "dpm":
        return DPMSolverMultistepScheduler(
            num_train_timesteps=num_train_timesteps,
            prediction_type="flow_prediction",
            use_flow_sigmas=True,
            flow_shift=shift,
        )
    if sampler_type == "euler_sto":
        return FlowMatchEulerStochasticDiscreteScheduler(
            num_train_timesteps=num_train_timesteps,
            shift=shift,
        )
    raise ValueError(
        f"Unknown RL sampler type: {sampler_type}. "
        "Supported rollout schedulers: euler, dpm, euler_sto"
    )


def build_rollout_sampler(
    sampler_type: str,
    num_train_timesteps: int,
    noise_std: float,
    eta: float,
    shift: float = 1.0,
):
    del noise_std, eta
    return build_rollout_scheduler(
        sampler_type=sampler_type,
        num_train_timesteps=num_train_timesteps,
        shift=shift,
    )
