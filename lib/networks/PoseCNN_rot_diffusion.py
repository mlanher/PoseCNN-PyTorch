# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from fcn.config import cfg
from layers.hard_label import HardLabel
from layers.hough_voting import HoughVoting
from layers.pose_target_layer import pose_target_layer
from layers.roi_pooling import RoIPool
from layers.roi_target_layer import roi_target_layer
from torch.nn.init import kaiming_normal_

__all__ = [
    'posecnn_rot_diffusion',
]

vgg16 = models.vgg16(pretrained=False)


def log_softmax_high_dimension(input):
    num_classes = input.size()[1]
    m = torch.max(input, dim=1, keepdim=True)[0]
    if input.dim() == 4:
        d = input - m.repeat(1, num_classes, 1, 1)
    else:
        d = input - m.repeat(1, num_classes)
    e = torch.exp(d)
    s = torch.sum(e, dim=1, keepdim=True)
    if input.dim() == 4:
        output = d - torch.log(s.repeat(1, num_classes, 1, 1))
    else:
        output = d - torch.log(s.repeat(1, num_classes))
    return output


def softmax_high_dimension(input):
    num_classes = input.size()[1]
    m = torch.max(input, dim=1, keepdim=True)[0]
    if input.dim() == 4:
        e = torch.exp(input - m.repeat(1, num_classes, 1, 1))
    else:
        e = torch.exp(input - m.repeat(1, num_classes))
    s = torch.sum(e, dim=1, keepdim=True)
    if input.dim() == 4:
        output = torch.div(e, s.repeat(1, num_classes, 1, 1))
    else:
        output = torch.div(e, s.repeat(1, num_classes))
    return output


def conv(in_planes, out_planes, kernel_size=3, stride=1, relu=True):
    if relu:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2,
                      bias=True),
            nn.ReLU(inplace=True))
    else:
        return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2,
                         bias=True)


def fc(in_planes, out_planes, relu=True):
    if relu:
        return nn.Sequential(
            nn.Linear(in_planes, out_planes),
            nn.LeakyReLU(0.1, inplace=True))
    else:
        return nn.Linear(in_planes, out_planes)


def upsample(scale_factor):
    return nn.Upsample(scale_factor=scale_factor, mode='bilinear')


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
        poses_target_noisy_ = poses_noisy.clone().detach()

        phase = (horizon - step) / horizon + eps
        sigma_T = DSM.marginal_prob_std_np(eps)
        sigma_i = DSM.marginal_prob_std_np(phase)
        ratio = sigma_i ** 2 / sigma_T ** 2
        c_lr = 1e-2 if noise_off else alpha * ratio

        noise_scale = phase * torch.ones_like(poses_target_noisy_[..., 0], device=poses_target_noisy_.device)

        z_pred = model.rotation_model(input_features, poses_target_noisy_, noise_scale.unsqueeze(1))

        noise = torch.zeros_like(poses_target_noisy_) if noise_off else torch.randn_like(
            poses_target_noisy_) * noise_std

        delta = -c_lr / 2 * z_pred + np.sqrt(c_lr) * noise
        return poses_target_noisy_ + delta

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


class RotationDiffusion(nn.Module):
    """ Diffusion model to predict rotations as quaternion. """

    def __init__(self, num_classes):
        super(RotationDiffusion, self).__init__()

        # Results RoI pooling, noisy rotation targets, noise scale
        self.fc1 = nn.Linear(512 * 7 * 7 + num_classes * 4 + 1, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 4 * num_classes)

        self.tanh = nn.Tanh()

    def forward(self, roi_pool, noisy_poses_target, noise_scale):
        out = torch.cat((roi_pool, noisy_poses_target, noise_scale), dim=1)
        out = self.tanh(self.fc1(out))
        out = self.tanh(self.fc2(out))
        out = self.tanh(self.fc3(out))
        return out


class PoseCNNRotDiffusion(nn.Module):
    """
    Uses diffusion model for rotation prediction. Only works on RGB images and YCB Video dataset with config
    published on original repository.
    """

    def __init__(self, num_classes, num_units):
        super(PoseCNNRotDiffusion, self).__init__()
        self.num_classes = num_classes

        # conv features
        features = list(vgg16.features)[:30]
        self.features = nn.ModuleList(features)
        self.classifier = vgg16.classifier[:-1]

        # Diffusion model for rotation prediction
        self.rotation_model = RotationDiffusion(num_classes)
        self.rot_loss = nn.L1Loss()

        # semantic labeling branch
        self.conv4_embed = conv(512, num_units, kernel_size=1)
        self.conv5_embed = conv(512, num_units, kernel_size=1)
        self.upsample_conv5_embed = upsample(2.0)
        self.upsample_embed = upsample(8.0)
        self.conv_score = conv(num_units, num_classes, kernel_size=1)
        self.hard_label = HardLabel(threshold=cfg.TRAIN.HARD_LABEL_THRESHOLD,
                                    sample_percentage=cfg.TRAIN.HARD_LABEL_SAMPLING)
        self.dropout = nn.Dropout()

        # center regression branch
        self.conv4_vertex_embed = conv(512, 2 * num_units, kernel_size=1, relu=False)
        self.conv5_vertex_embed = conv(512, 2 * num_units, kernel_size=1, relu=False)
        self.upsample_conv5_vertex_embed = upsample(2.0)
        self.upsample_vertex_embed = upsample(8.0)
        self.conv_vertex_score = conv(2 * num_units, 3 * num_classes, kernel_size=1, relu=False)
        # hough voting
        self.hough_voting = HoughVoting(is_train=0, skip_pixels=10, label_threshold=100, inlier_threshold=0.9,
                                        voting_threshold=-1, per_threshold=0.01)

        self.roi_pool_conv4 = RoIPool(pool_height=7, pool_width=7, spatial_scale=1.0 / 8.0)
        self.roi_pool_conv5 = RoIPool(pool_height=7, pool_width=7, spatial_scale=1.0 / 16.0)

        dim_fc = 4096
        self.fc8 = fc(dim_fc, num_classes)
        self.fc9 = fc(dim_fc, 4 * num_classes, relu=False)

        self.device = next(self.parameters()).device

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, label_gt, meta_data, extents, gt_boxes, poses, points, symmetry):

        # conv features
        for i, model in enumerate(self.features):
            x = model(x)
            if i == 22:
                out_conv4_3 = x
            if i == 29:
                out_conv5_3 = x

        # semantic labeling branch
        out_conv4_embed = self.conv4_embed(out_conv4_3)
        out_conv5_embed = self.conv5_embed(out_conv5_3)
        out_conv5_embed_up = self.upsample_conv5_embed(out_conv5_embed)
        out_embed = self.dropout(out_conv4_embed + out_conv5_embed_up)
        out_embed_up = self.upsample_embed(out_embed)
        out_score = self.conv_score(out_embed_up)
        out_logsoftmax = log_softmax_high_dimension(out_score)
        out_prob = softmax_high_dimension(out_score)
        out_label = torch.max(out_prob, dim=1)[1].type(torch.IntTensor).cuda()
        out_weight = self.hard_label(out_prob, label_gt, torch.rand(out_prob.size()).cuda())

        # center regression branch
        out_conv4_vertex_embed = self.conv4_vertex_embed(out_conv4_3)
        out_conv5_vertex_embed = self.conv5_vertex_embed(out_conv5_3)
        out_conv5_vertex_embed_up = self.upsample_conv5_vertex_embed(out_conv5_vertex_embed)
        out_vertex_embed = self.dropout(out_conv4_vertex_embed + out_conv5_vertex_embed_up)
        out_vertex_embed_up = self.upsample_vertex_embed(out_vertex_embed)
        out_vertex = self.conv_vertex_score(out_vertex_embed_up)

        # hough voting
        if self.training:
            self.hough_voting.is_train = 1
            self.hough_voting.label_threshold = cfg.TRAIN.HOUGH_LABEL_THRESHOLD
            self.hough_voting.voting_threshold = cfg.TRAIN.HOUGH_VOTING_THRESHOLD
            self.hough_voting.skip_pixels = cfg.TRAIN.HOUGH_SKIP_PIXELS
            self.hough_voting.inlier_threshold = cfg.TRAIN.HOUGH_INLIER_THRESHOLD
        else:
            self.hough_voting.is_train = 0
            self.hough_voting.label_threshold = cfg.TEST.HOUGH_LABEL_THRESHOLD
            self.hough_voting.voting_threshold = cfg.TEST.HOUGH_VOTING_THRESHOLD
            self.hough_voting.skip_pixels = cfg.TEST.HOUGH_SKIP_PIXELS
            self.hough_voting.inlier_threshold = cfg.TEST.HOUGH_INLIER_THRESHOLD
        out_box, out_pose = self.hough_voting(out_label, out_vertex, meta_data, extents)

        # bounding box classification and regression branch
        bbox_labels, bbox_targets, bbox_inside_weights, bbox_outside_weights = roi_target_layer(out_box, gt_boxes)
        out_roi_conv4 = self.roi_pool_conv4(out_conv4_3, out_box)
        out_roi_conv5 = self.roi_pool_conv5(out_conv5_3, out_box)
        out_roi = out_roi_conv4 + out_roi_conv5
        out_roi_flatten = out_roi.view(out_roi.size(0), -1)
        out_fc7 = self.classifier(out_roi_flatten)
        out_fc8 = self.fc8(out_fc7)
        out_logsoftmax_box = log_softmax_high_dimension(out_fc8)
        bbox_prob = softmax_high_dimension(out_fc8)
        bbox_label_weights = self.hard_label(bbox_prob, bbox_labels, torch.rand(bbox_prob.size()).cuda())
        bbox_pred = self.fc9(out_fc7)

        # rotation regression branch
        # poses target values are in (-1, 1)
        rois, poses_target, poses_weight = pose_target_layer(out_box, bbox_prob, bbox_pred, gt_boxes, poses,
                                                             self.training)
        poses_target_noisy, noise_scale, z_target, std = DSM.perturb(poses_target)
        noise_scale = noise_scale.unsqueeze(1)
        poses_target_noisy = poses_target_noisy.detach().requires_grad_(True)

        out_qt_conv4 = self.roi_pool_conv4(out_conv4_3, rois)
        out_qt_conv5 = self.roi_pool_conv5(out_conv5_3, rois)
        out_qt = out_qt_conv4 + out_qt_conv5
        out_qt_flatten = out_qt.view(out_qt.size(0), -1)

        if self.training:
            with torch.set_grad_enabled(True):
                z_pred = self.rotation_model(out_qt_flatten, poses_target_noisy, noise_scale)
                z_pred_weighted = nn.functional.normalize(torch.mul(z_pred, poses_weight))

            loss_pose = self.rot_loss(z_pred_weighted * std[..., None], z_target)

            return out_logsoftmax, out_weight, out_vertex, out_logsoftmax_box, bbox_label_weights, bbox_pred, \
                bbox_targets, bbox_inside_weights, loss_pose, poses_weight
        else:
            # TODO: Return annealed langevin dynamics history to visualize it
            out_quaternion, _ = DSM.sample(self.device, self, input_features=out_qt_flatten,
                                           n_dim_poses=4 * self.num_classes, n_poses_init=100, horizon=100,
                                           horizon_noise_off=100)
            return out_label, out_vertex, rois, out_pose, out_quaternion

    def weight_parameters(self):
        parameters = [param for name, param in self.named_parameters() if 'weight' in name]
        return list(filter(lambda p: p.requires_grad, parameters))

    def bias_parameters(self):
        parameters = [param for name, param in self.named_parameters() if 'bias' in name]
        return list(filter(lambda p: p.requires_grad, parameters))


def posecnn_rot_diffusion(num_classes, num_units, data=None):
    model = PoseCNNRotDiffusion(num_classes, num_units)

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
