import datetime as dt
import logging
import os

import h5py
import numpy as np

from das_reader.reader import Reader
from preprocessing import parameters
from processing_utils import processing_utils as processing


def process(raw_data):
  low_freq, high_freq, dt, q = 1.0, 12.0, 0.02, 2
  data = processing.get_strain_rate(raw_data)
  data = processing.remove_median(data)
  data = processing.bandpass(data, low_freq, high_freq, dt)
  data = processing.decimate(data, q)
  data = np.clip(data, -0.024, 0.024) / 0.014088576
  data = np.float32(data)
  return data


def make_headers(starttime, window, channels, sampling):
  headers = {}
  headers['starttime'] = str(starttime)
  headers['window_length'] = window
  headers['start_channel'] = channels[0]
  headers['end_channel'] = channels[-1]
  headers['num_channels'] = len(channels)
  headers['sampling_rate_Hz'] = sampling
  return headers


def write_to_hdf5(data, headers, filename):
  with h5py.File(filename, 'w') as f:
    f.create_dataset('data', data=data)
    for key in headers:
      f.attrs[key] = headers[key]


def pull_das_data(reader, starttime, window):
  raw_data = reader.readData(starttime, window)
  headers = make_headers(starttime, window, reader.channels, reader.sampling)
  if raw_data is not None:
    data = process(raw_data)
    return data, headers
  return None, headers


def pull_continuous_data(reader, datapath, starttime, endtime, window):
  logging.info('Downloading continuous data...')
  datapath = os.path.join(datapath, 'continuous')
  os.makedirs(datapath, exist_ok=True)
  logging.info('Writing to %s', datapath)

  starttimes = [starttime, ]
  while starttimes[-1] < endtime:
    starttimes.append(starttimes[-1] + dt.timedelta(seconds=window))

  for timestamp in starttimes:
    data, headers = pull_das_data(reader, timestamp, window)
    if data is not None:
      filename = os.path.join(
          datapath, 'data_{}.h5'.format(timestamp.strftime('%Y%m%d_%H%M%S')))
      write_to_hdf5(data, headers, filename)
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
  )


if __name__ == '__main__':
  main()
