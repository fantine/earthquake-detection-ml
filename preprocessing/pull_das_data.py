import datetime as dt
import logging
import os
import sys

import h5py
import numpy as np
import pandas as pd

from das_reader.reader import Reader
from earthquakes import parameters


def process(raw_data):
  data = raw_data.astype(np.float32)
  return data


def make_headers(eventtime, window, channels, sampling):
  headers = {}
  headers['arrival_time'] = str(eventtime)
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


def pull_eventtime_data(reader, eventtime, window):
  starttime = eventtime - dt.timedelta(seconds=window / 2)
  raw_data = reader.readData(starttime, window)
  headers = make_headers(eventtime, window, reader.channels, reader.sampling)
  if raw_data is not None:
    data = process(raw_data)
    return data, headers
  return None, headers


def pull_catalog_data(catalog_file, reader, window, batch, datapath,
                      file_prefix):
  catalog = pd.read_pickle(catalog_file)
  logging.info('Saving to %s', datapath)

  for i, eventtime in enumerate(catalog.datetime):
    if i % batch == 0:
      logging.info('Pulled %s files.', i)
      subdir = os.path.join(datapath, '{:05d}'.format(i))
      os.makedirs(subdir, exist_ok=True)
    data, headers = pull_eventtime_data(reader, eventtime, window)
    if data is not None:
      filename = os.path.join(subdir, '{}_{:05d}.hdf5'.format(file_prefix, i))
      write_to_hdf5(data, headers, filename)
      catalog.at[i, 'has_das_data'] = True
    else:
      logging.info('No available DAS data for %s %s.', file_prefix, i)
      catalog.at[i, 'has_das_data'] = False

  catalog.to_pickle(catalog_file)


def parse_args():
  available_options = ['all', 'event', 'noise', 'continuous']

  option = 'all'
  if len(sys.argv) > 1:
    option = sys.argv[1]
    if option not in available_options:
      print("Argument should be 'all', 'event', 'noise', or 'continuous'")
      sys.exit()
  return option


def main():
  channels = np.arange(parameters.start_channel, parameters.end_channel + 1)

  reader = Reader(channels=channels, sampling=parameters.passive_das_sampling)

  option = parse_args()
  if option in ['all', 'event']:
    logging.info('Pulling seismic event DAS data...')
    pull_catalog_data(
        catalog_file=parameters.event_catalog,
        reader=reader,
        window=parameters.raw_window_length,
        batch=parameters.batch,
        datapath=os.path.join(parameters.raw_datapath, 'das/event'),
        file_prefix='event')

  if option in ['all', 'noise']:
    logging.info('Pulling background noise DAS data...')
    pull_catalog_data(
        catalog_file=parameters.noise_catalog,
        reader=reader,
        window=parameters.raw_window_length,
        batch=parameters.batch,
        datapath=os.path.join(parameters.raw_datapath, 'das/noise'),
        file_prefix='noise')


if __name__ == '__main__':
  main()
