#! /usr/bin/env bash

export MODULE_ARGS=" \
  --model=CNN2DModular \
  --height=288 \
  --width=512 \
  --tfrecord_height=288 \
  --tfrecord_width=695 \
  --num_epochs=100 \
  --learning_rate=0.0001 \
  --batch_size=16 \
  --network_depth=4 \
  --num_filters=32 \
  --filter_increase_mode=3 \
  --filter_multiplier=4 \
  --downsampling=1 \
  --dense_dropout=0.6 \
"
