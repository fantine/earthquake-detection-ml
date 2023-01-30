import datetime as dt
import logging
import os

import numpy as np

from das_reader.reader import Reader
from preprocessing import parameters
from processing_utils import processing_utils as processing


def process(raw_data, low_freq, high_freq, dt, q, clip_val, norm_val):
  data = processing.get_strain_rate(raw_data)
  data = processing.remove_median(data)
  data = processing.bandpass(data, low_freq, high_freq, dt)
  data = processing.decimate(data, q)
  data = np.clip(data, -clip_val, clip_val) / norm_val
  data = np.float32(data)
  return data


def pull_das_data(reader, starttime, window, low_freq, high_freq, dt, q,
    clip_val, norm_val):
  raw_data = reader.readData(starttime, window)
  if raw_data is not None:
    data = process(raw_data, low_freq, high_freq, dt, q, clip_val, norm_val)
    return data
  return None


def pull_continuous_data(reader, datapath, starttime, endtime, window, low_freq,
    high_freq, dt, q, clip_val, norm_val,):
  logging.info('Downloading continuous data...')
  datapath = os.path.join(datapath, 'continuous')
  os.makedirs(datapath, exist_ok=True)
  logging.info('Writing to %s', datapath)

  starttimes = [starttime, ]
  while starttimes[-1] < endtime:
    starttimes.append(starttimes[-1] + dt.timedelta(seconds=window))

  for timestamp in starttimes:
    data, headers = pull_das_data(reader, timestamp, window, low_freq,
        high_freq, dt, q, clip_val, norm_val)
    if data is not None:
      filename = os.path.join(
          datapath, 'data_{}.npy'.format(timestamp.strftime('%Y%m%d_%H%M%S')))
      np.save(filename, data)
    else:
      logging.info('No available DAS data for %s.', timestamp)


def main():
  channels = parameters.channel_subset1

  reader = Reader(channels=channels, sampling=parameters.passive_das_sampling)

  logging.info('Pulling continuous DAS data...')
  pull_continuous_data(
      reader=reader,
      datapath=os.path.join(parameters.raw_datapath, 'das'),
      starttime=parameters.continuous_starttime,
      endtime=parameters.continuous_endtime,
      window=parameters.continuous_window,
      low_freq=parameters.low_freq,
      high_freq=parameters.high_freq,
      dt=parameters.das_dt,
      q=parameters.das_downsampling_factor,
      clip_val=parameters.das_clip_val, 
      norm_val=parameters.das_norm_val,
  )


if __name__ == '__main__':
  main()
