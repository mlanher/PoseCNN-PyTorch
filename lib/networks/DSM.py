import numpy as np
import torch
from networks.SO3_R3 import SO3_R3


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
    def perturb_in_vector_space(H: SO3_R3, eps=1e-5):
        """
        Perturbs SE(3) pose H in vector space and returns noisy pose in vector space.
        """
        tw = H.log_map()

        noise_scale = torch.rand_like(tw[..., 0], device=tw.device) * (1. - eps) + eps
        z = torch.randn_like(tw)
        std = DSM.marginal_prob_std(noise_scale)
        noise = z * std[..., None]

        tw_noisy = tw + noise
        return tw_noisy, noise_scale, z, std
