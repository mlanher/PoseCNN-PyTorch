import time

import numpy as np
from fcn.config import cfg
from utils.nms import *
from utils.se3 import *


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return '{:.3f} ({:.3f})'.format(self.val, self.avg)


def train_diffusion_rot_poc(train_loader, background_loader, network, optimizer, epoch, writer):
    batch_time = AverageMeter()
    losses = AverageMeter()

    epoch_size = len(train_loader)
    writer.add_scalar("Epoch size", epoch_size, epoch)

    enum_background = enumerate(background_loader)

    network.train()

    epoch_loss_pose = 0.
    for i, sample in enumerate(train_loader):
        end = time.time()

        imgs = sample["image_color"].cuda()
        im_info = sample["im_info"]
        masks = sample["mask"].cuda()
        # Shape: (batch_size, num_classes, 5) where bboxes[:, :, 0:4] represents [min_x, min_y, max_x, max_y] and
        # bboxes[:, :, 4] represents class label
        bboxes = sample["gt_boxes"].cuda()
        # Shape: (batch_size, num_classes, 9) where poses[:, :, 2:6] is for rotation, poses[6:9] is translation
        poses = sample["poses"].cuda()

        try:
            _, background = next(enum_background)
        except:
            enum_background = enumerate(background_loader)
            _, background = next(enum_background)

        if imgs.size(0) != background['background_color'].size(0):
            enum_background = enumerate(background_loader)
            _, background = next(enum_background)

        background_color = background['background_color'].cuda()
        for j in range(imgs.size(0)):
            is_syn = im_info[j, -1]
            if is_syn or np.random.rand(1) > 0.5:
                imgs[j] = masks[j] * imgs[j] + (1 - masks[j]) * background_color[j]

        # Only give rotations
        loss_pose = network(imgs, poses[:, :, 2:6], bboxes)
        loss = loss_pose

        epoch_loss_pose += loss_pose.item()

        writer.add_scalar("Loss Pose (Sample)", loss_pose.item(), epoch * i)

        fname_loss = "losses.txt"
        loss_to_save = loss_pose.item()
        with open(fname_loss, "a") as f:
            f.write(str(loss_to_save) + "\n")

        # record loss
        losses.update(loss.data, imgs.size(0))

        # compute gradient and do optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)

        print(
            '[%d/%d][%d/%d], %.4f, pose %.4f, lr %.6f, time %.2f' % (
                epoch,
                cfg.epochs,
                i,
                epoch_size,
                loss.data,
                loss_pose.data,
                optimizer.param_groups[0]['lr'],
                batch_time.val))
        cfg.TRAIN.ITERS += 1

    epoch_loss_pose /= epoch_size
    writer.add_scalar("Loss Pose (Epoch)", epoch_loss_pose, epoch)
    return
