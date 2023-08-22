#!/bin/bash

set -x
set -e

time ./tools/train_net_diffusion_rot_poc.py \
  --network posecnn_diffusion_rot_poc \
  --pretrained data/checkpoints/vgg16-397923af.pth \
  --dataset ycb_video_train \
  --cfg experiments/cfgs/ycb_video.yml \
  --solver sgd \
  --epochs 16
