import torch
from torch import nn


class RotationDiffusion(nn.Module):
    """ Diffusion model to predict rotations as quaternion. """

    def __init__(self, num_classes):
        super(RotationDiffusion, self).__init__()

        # Reduce dimensionality
        self.fc_dim_reduction = nn.Linear(512 * 7 * 7, 1024)

        self.attention = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.Softmax(dim=1)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(1024 + 88 + 1, 512),  # 1024 from feature map, 88 from target pose, 1 from noise scale
            nn.BatchNorm1d(512),
            nn.Tanh(),
            nn.Dropout(0.5)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.Tanh(),
            nn.Dropout(0.5)
        )

        self.fc3 = nn.Linear(256, 4 * num_classes)

    def adaptive_tanh(self, x: torch.Tensor, noise_scale: torch.Tensor):
        """
        Ensures range (-1, 1) while preserving the correlation of the added noise and the noise scale
        """
        return torch.tanh(x * noise_scale)

    def forward(self, roi_pool, noisy_poses_target, noise_scale):
        roi_pool_reduced = self.fc_dim_reduction(roi_pool)

        attention_weights = self.attention(roi_pool_reduced)
        roi_pool_attention = attention_weights * roi_pool_reduced

        noisy_poses_ensured_range = self.adaptive_tanh(noisy_poses_target, noise_scale)
        out = torch.cat((roi_pool_attention, noisy_poses_ensured_range, noise_scale), dim=1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out