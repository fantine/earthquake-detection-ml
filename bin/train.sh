#!/bin/bash

# Run a ML model training job
#
# e.g. bin/train.sh model_config dataset label
#
# @param {model_config} Name of ML model configuration to use.
#            This should correspond to a configuration file named as follows:
#            config/${model_config}.sh.
# @param {dataset} Dataset identifier.
#            Check the variables `datapath`, `train_file`, and `eval_file`,
#            to ensure that this maps to the correct input data.
# @param {label} Optional label to add to the job name.

# Get arguments
model_config=$1
dataset=$2
label=$3

# Set path to input data
datapath="/scr1/fantine/sdasa"
train_file="${datapath}/tfrecords/${dataset}-train-*.tfrecord.gz"
eval_file="${datapath}/tfrecords/${dataset}-eval-*.tfrecord.gz"

# Check the ML model config file
config_file=config/$model_config.sh
if [ ! -f "$config_file" ]; then
  echo "Config file not found: $config_file";
  exit 1;
fi

# Read the ML model config file
. "$config_file"

# Define the job name
now=$(date +%Y%m%d_%H%M%S)
job_name=job_${now}_${model_config}_${dataset}_${label}
job_dir="${datapath}/models/${job_name}"
log_file="log/${job_name}.log"

# Set package and module name
package_path=trainer/
module_name=trainer.task

# Run the job
if [ "$label" != "hptuning" ]; then
  echo 'Running ML job in the background.'
  echo "Logging to file: $log_file"
  python -m $module_name \
  --job_dir=$job_dir \
  $MODULE_ARGS \
  --train_file=$train_file \
  --eval_file=$eval_file \
  > $log_file 2>&1 &
else # if this is a hyperparameter tuning job, run it in the foreground
  echo "Logging to file: $log_file"
  python -m $module_name \
  --job_dir=$job_dir \
  $MODULE_ARGS \
  --train_file=$train_file \
  --eval_file=$eval_file \
  > $log_file 2>&1
fi