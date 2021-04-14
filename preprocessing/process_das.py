"""DAS data processing."""

import logging
import os

import h5py
import numpy as np
from processing_utils import processing_utils as processing

from preprocessing import parameters


logging.basicConfig(level=logging.INFO)


def _process(data, low_freq, high_freq, dt, clip_percentile, q):
  data = processing.get_strain_rate(data)
  data = processing.remove_median(data)
  # data = processing.clip(data, clip_percentile)
  data = processing.bandpass(data, low_freq, high_freq, dt)
  data = processing.decimate(data, q)
  # data = processing.normalize(data)
  return data


def _crop(data, raw_window, detect_window, event_duration, dt):
  sps = 1 // dt
  start_sample = int((raw_window / 2 - (detect_window - event_duration)) * sps)
  end_sample = int((raw_window / 2 + detect_window) * sps)
  return data[:, start_sample:end_sample]


def _get_label(filename):
  basename = os.path.basename(filename)
  if 'noise' in basename:
    return np.zeros((1,), dtype=np.float32)
  if 'event' in basename:
    return np.ones((1,), dtype=np.float32)


def write_hdf5(out_file, data, label):
  with h5py.File(out_file, 'w') as f:
    f.create_dataset('input', data=data)
    f.create_dataset('label', data=label)


def read_hdf5(filename):
  with h5py.File(filename, 'r') as f:
    return f.get('data')[()]


def process(file_pattern, in_dir, out_dir, raw_window, detect_window,
            event_duration, low_freq, high_freq, dt, clip_percentile, q,
            channel_subset1, channel_subset2):
  filenames = processing.get_filenames(file_pattern)

  for i, filename in enumerate(filenames):
    if i % 1000 == 0:
      logging.info('Processed %s files.', i)
    data = read_hdf5(filename)
    data = _process(data, low_freq, high_freq, dt, clip_percentile, q)
    data = _crop(data, raw_window, detect_window, event_duration, dt * q)
    label = _get_label(filename)
    out_file = filename.replace(in_dir, out_dir)
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    data1 = data[channel_subset1]
    data2 = data[channel_subset2]
    data2 = data2[::-1]
    out_file1 = out_file.replace('.hdf5', '_1.h5')
    out_file2 = out_file.replace('.hdf5', '_2.h5')
    write_hdf5(out_file1, data1, label)
    write_hdf5(out_file2, data2, label)


def main():
  datatype = 'das'
  datapath = os.path.join(parameters.raw_datapath, datatype)
  file_pattern = os.path.join(datapath, '*/*/*')
  # file_pattern = os.path.join(datapath, 'event/00000/event_000*')
  process(
      file_pattern,
      in_dir=parameters.raw_datapath,
      out_dir=parameters.processed_datapath,
      raw_window=parameters.raw_window_length,
      detect_window=parameters.detect_window_length,
      event_duration=parameters.event_duration,
      low_freq=parameters.low_freq,
      high_freq=parameters.high_freq,
      dt=parameters.das_dt,
      clip_percentile=parameters.clip_percentile,
      q=parameters.das_downsampling_factor,
      channel_subset1=parameters.channel_subset1,
      channel_subset2=parameters.channel_subset2
  )


if __name__ == '__main__':
  main()
