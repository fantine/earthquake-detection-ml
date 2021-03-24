#! /usr/bin/env bash

export MODULE_ARGS=" \
  --model=CNN2DModular \
  --height=288 \
  --width=512 \
  --channels=1 \
  --tfrecord_height=288 \
  --tfrecord_width=696 \
  --num_epochs=20 \
"