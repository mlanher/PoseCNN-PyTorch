from typing import Tuple

import numpy as np
import torch
from torch import nn


class DSM:
    @staticmethod
    def marginal_prob_std(t: torch.Tensor, sigma=0.5):
        """
        Represents "shaped" version of standard deviation.
        """
        return torch.sqrt((sigma ** (2 * t) - 1.) / (2. * np.log(sigma)))

    @staticmethod
    def marginal_prob_std_np(t, sigma=0.5):
        return np.sqrt((sigma ** (2 * t) - 1.) / (2. * np.log(sigma)))

    @staticmethod
    def perturb(tensor: torch.Tensor, eps=1e-5) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        noise_scale = torch.rand_like(tensor[..., 0], device=tensor.device) * (1. - eps) + eps
        z = torch.randn_like(tensor)
        std = DSM.marginal_prob_std(noise_scale)
        perturbed_t = tensor + z * std[..., None]
        return perturbed_t, noise_scale, z, std

    @staticmethod
    def annealed_langevin(model: nn.Module, input_features: torch.Tensor, poses_noisy: torch.Tensor, step: int,
                          eps=1e-3,
                          alpha=1e-4, noise_std=.5, horizon=100, noise_off=False) -> torch.Tensor:
        """
        Args:
             noise_off: If true, particles directly follow gradient.
        """
        # This prevents a torch.cuda.OutOfMemoryError in for-loop of sample method
        poses_noisy_ = poses_noisy.clone().detach()

        phase = (horizon - step) / horizon + eps
        sigma_T = DSM.marginal_prob_std_np(eps)
        sigma_i = DSM.marginal_prob_std_np(phase)
        ratio = sigma_i ** 2 / sigma_T ** 2
        c_lr = 1e-2 if noise_off else alpha * ratio

        noise_scale = phase * torch.ones_like(poses_noisy_[..., 0], device=poses_noisy_.device)
        z_pred = model.rotation_model(input_features, poses_noisy_, noise_scale.unsqueeze(1))

        noise = torch.zeros_like(poses_noisy_) if noise_off else torch.randn_like(
            poses_noisy_) * noise_std

        delta = -c_lr / 2 * z_pred + np.sqrt(c_lr) * noise
        return poses_noisy_ + delta

    @staticmethod
    def _sample_annealed_langevin(model: nn.Module, input_features: torch.Tensor, poses_init: torch.Tensor, horizon=100,
                                  horizon_noise_off=100):
        poses_next = poses_init
        hist = [poses_init.detach().clone().cpu().numpy()]
        print("Performing {} steps of annealed langevin with noise.".format(horizon))
        for step in range(horizon):
            poses_next = DSM.annealed_langevin(model, input_features, poses_next, step, horizon=horizon)
            hist.append(poses_next.detach().clone().cpu().numpy())

        print("Performing {} steps of annealed langevin without noise.".format(horizon))
        for step in range(horizon_noise_off):
            poses_next = DSM.annealed_langevin(model, input_features, poses_next, step, horizon=horizon, noise_off=True)
            hist.append(poses_next.detach().clone().cpu().numpy())
        poses_last = poses_next

        return poses_last, hist

    @staticmethod
    def sample(device: str, model: nn.Module, input_features: torch.Tensor, n_dim_poses: int, n_poses_init=100,
               horizon=100, horizon_noise_off=100):
        poses_init = torch.rand(n_poses_init, n_dim_poses, device=torch.device(device)) * 2 - 1
        return DSM._sample_annealed_langevin(model, input_features, poses_init, horizon, horizon_noise_off)
