import torch
import torchvision.models as models

__all__ = [
    'posecnn_diffusion_rot_poc',
]

from torch import nn
from torch.nn.init import kaiming_normal_
from networks.RotationDiffusion import RotationDiffusion
from networks.DSM import DSM
from torchvision.ops import roi_pool

vgg16 = models.vgg16(pretrained=False)


class DiffusionPoseCNNRotPoC(nn.Module):
    """
    Proof of concept of rotation prediction by diffusion model on YCB Video dataset.
    """

    def __init__(self, num_classes):
        super(DiffusionPoseCNNRotPoC, self).__init__()
        self.num_classes = num_classes

        # Only use feature extraction stage
        features = list(vgg16.features)[:30]
        self.features = nn.ModuleList(features)

        self.rotation_model = RotationDiffusion(num_classes)
        self.rot_loss = nn.L1Loss()

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, poses, bboxes):
        """
        Forward pass expects RGB image as well as gt segmentation, bounding boxes and rotation as quaternion.

        Args:
            x: RGB images.
            bboxes_coords: Ground truth coordinates of bounding boxes of shape (batch_size, 22, 4)
            labels: Ground truth segmentation of shape (batch_size, img_height, img_width) for hough voting.
            vertex_targets: For hough voting.
            meta_data: For hough voting.
            extents: For hough voting.
            rotations: Ground truth rotation of shape (batch_size, num_classes * 4) as quaternion representation.
        """
        device = next(self.parameters()).device

        # conv features
        for i, model in enumerate(self.features):
            x = model(x)
            if i == 22:
                out_conv4_3 = x
            if i == 29:
                out_conv5_3 = x

        bboxes_coords, bboxes_labels = bboxes[:, :, :-1], bboxes[:, :, -1]
        batch_size, num_classes, poses_dim = poses.size()

        poses_target = torch.zeros(batch_size * num_classes, poses_dim * num_classes, device=poses.device)

        bboxes_labels_flatten = bboxes_labels.view(-1)
        poses_flatten = poses.view(batch_size * num_classes, -1)

        mask = bboxes_labels_flatten > 0
        label_indices = ((bboxes_labels_flatten[mask] - 1) * poses_dim).long()
        indices = (torch.arange(batch_size, device=poses.device).unsqueeze(1) * num_classes) + \
                  torch.arange(num_classes, device=poses.device)
        for idx, label_idx in zip(indices.view(-1)[mask], label_indices):
            poses_target[idx, label_idx:label_idx + poses_dim] = poses_flatten[idx]

        # Ensure RoI pooling format
        batch_indices = torch.arange(bboxes_coords.size(0)).view(-1, 1, 1).expand(-1, bboxes_coords.size(1), -1).cuda()
        rois = torch.cat([batch_indices.reshape(-1, 1), bboxes_coords.reshape(-1, 4)], dim=1)

        out_roi_conv4 = roi_pool(out_conv4_3, rois, output_size=(7, 7), spatial_scale=1.0 / 8.0)
        out_roi_conv5 = roi_pool(out_conv5_3, rois, output_size=(7, 7), spatial_scale=1.0 / 16.0)
        out_roi = out_roi_conv4 + out_roi_conv5
        out_roi_flatten = out_roi.view(out_roi.size(0), -1)

        # TODO: Inference
        if self.training:
            with torch.set_grad_enabled(True):
                poses_target_noisy, noise_scale, z_target, std = DSM.perturb(poses_target)
                noise_scale = noise_scale.unsqueeze(1)
                rot_target_noisy = poses_target_noisy.detach().requires_grad_(True)
        z_pred = self.rotation_model(out_roi_flatten, rot_target_noisy, noise_scale)

        return self.rot_loss(z_pred * std[..., None], z_target)

    def weight_parameters(self):
        parameters = [param for name, param in self.named_parameters() if 'weight' in name]
        return list(filter(lambda p: p.requires_grad, parameters))

    def bias_parameters(self):
        parameters = [param for name, param in self.named_parameters() if 'bias' in name]
        return list(filter(lambda p: p.requires_grad, parameters))


def posecnn_diffusion_rot_poc(num_classes, data=None):
    model = DiffusionPoseCNNRotPoC(num_classes)

    if data is not None:
        model_dict = model.state_dict()
        print('model keys')
        print('=================================================')
        for k, v in model_dict.items():
            print(k)
        print('=================================================')

        print('data keys')
        print('=================================================')
        for k, v in data.items():
            print(k)
        print('=================================================')

        pretrained_dict = {k: v for k, v in data.items() if k in model_dict and v.size() == model_dict[k].size()}
        print('load the following keys from the pretrained model')
        print('=================================================')
        for k, v in pretrained_dict.items():
            print(k)
        print('=================================================')
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze all parameters not loaded
        for name, param in model.named_parameters():
            if name not in pretrained_dict:
                param.requires_grad = True

    return model
