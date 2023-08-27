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
            nn.Linear(1024 + 154 + 1, 512),  # 1024 from feature map, 154 from target pose, 1 from noise scale
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

        self.fc3 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.Tanh(),
            nn.Dropout(0.5)
        )

        self.fc4 = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.Tanh(),
            nn.Dropout(0.5)
        )

        self.fc5 = nn.Linear(64, 6)

    def forward(self, roi_pool, poses, noise_scale):
        roi_pool_reduced = self.fc_dim_reduction(roi_pool)

        attention_weights = self.attention(roi_pool_reduced)
        roi_pool_attention = attention_weights * roi_pool_reduced

        out = torch.cat((roi_pool_attention, poses, noise_scale), dim=1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc4(out)
        out = self.fc5(out)
        return out