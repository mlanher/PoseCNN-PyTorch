#!/bin/bash

set -x
set -e

time ./tools/train_net_diffusion_rot_poc.py \
  --network posecnn_diffusion_rot_poc \
  --pretrained /data/checkpoints/ycb_video/vgg16_ycb_video_epoch_16.checkpoint.pth \
  --dataset ycb_video_train \
  --cfg experiments/cfgs/ycb_video.yml \
  --solver sgd \
  --epochs 16
