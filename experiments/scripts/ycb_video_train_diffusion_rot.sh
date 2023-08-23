#!/bin/bash

set -x
set -e

time ./tools/train_net_diffusion_rot.py \
  --network diffusion_posenn_rot \
  --pretrained data/checkpoints/vgg16-397923af.pth \
  --dataset ycb_video_train \
  --cfg experiments/cfgs/ycb_video.yml \
  --solver sgd \
  --epochs 16
