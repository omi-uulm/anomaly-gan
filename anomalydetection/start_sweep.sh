#!/usr/bin/env sh

docker run -d -it\
  --gpu=all\
  --name AnomalyDetectorSweep \
  --mount type=bind,source="$(pwd)"/code/,target=/opt1/program \
  --mount type=bind,source="$(pwd)"/results/,target=/opt1/out \
  --mount type=bind,source="$(pwd)"/../data/,target=/opt1/data/ \
  anomalygan "$@"

