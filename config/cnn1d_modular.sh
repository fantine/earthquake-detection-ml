#! /usr/bin/env bash

export MODULE_ARGS=" \
  --model=CNN1DModular \
  --height=512 \
  --width=1 \
  --channels=6 \
  --tfrecord_height=695\
  --tfrecord_width=1 \
  --num_epochs=100 \
"
