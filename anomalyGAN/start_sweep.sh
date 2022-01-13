#!/usr/bin/env sh

docker run -d -it\
  --gpu=all\
  --name GANSweep \
  --mount type=bind,source="$(pwd)"/code/,target=/opt1/program \
  --mount type=bind,source="$(pwd)"/results/,target=/opt1/out \
  --mount type=bind,source="$(pwd)"/../data/,target=/opt1/data/ \
  --mount type=bind,source="$(pwd)"/pretrained-detectors/,target=/opt1/pretrained-detectors/ \
  anomalygan "$@"

