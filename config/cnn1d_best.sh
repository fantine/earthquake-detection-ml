! /usr/bin/env bash

export MODULE_ARGS=" \
  --model=CNN1DModular \
  --height=512 \
  --width=1 \
  --channels=6 \
  --tfrecord_height=695\
  --tfrecord_width=1 \
  --num_epochs=100 \
  --learning_rate=0.0001 \
  --batch_size=64 \
  --network_depth=4 \
  --num_filters=8 \
  --filter_increase_mode=3 \
  --filter_multiplier=4 \
  --activation=0 \
  --downsampling=1 \
  --batchnorm=0 \
  --conv_dropout=0.0 \
  --dense_dropout=0.6 \
"
