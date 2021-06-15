import datetime
import logging
import subprocess

from preprocessing import parameters


def run_detections(model_config, ckpt, starttime, endtime):
  start_year = starttime.year
  end_year = endtime.year
  for year in range(start_year, end_year + 1):
    for month in range(1, 13):
      dataset = 'data_{:d}{:02d}'.format(year, month)
      cmd = 'bin/predict.sh {} {} {}'.format(model_config, dataset, ckpt)
      logging.info(cmd)
      output = subprocess.run(
          cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True,
          check=False)


def main():
  starttime = parameters.continuous_starttime
  endtime = parameters.continuous_endtime
  model_config = 'cnn2d_modular'
  ckpt = 'pretrained_th40'
  run_detections(model_config, ckpt, starttime, endtime)


if __name__ == "__main__":
  main()
