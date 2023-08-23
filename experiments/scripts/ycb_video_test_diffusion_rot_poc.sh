#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=$1

time ./tools/test_net.py --gpu $1 \
  --network posecnn_diffusion_rot_poc \
  --pretrained /data/checkpoints/ycb_video/vgg16_ycb_video_epoch_16.checkpoint.pth \
  --dataset ycb_video_keyframe \
  --cfg experiments/cfgs/ycb_video.yml
