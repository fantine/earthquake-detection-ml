"""DAS data processing."""

import logging
import os

import h5py
import numpy as np
from processing_utils import processing_utils as processing

from preprocessing import parameters


logging.basicConfig(level=logging.INFO)


def _process(data, low_freq, high_freq, dt, q):
  data = processing.bandpass(data, low_freq, high_freq, dt)
  data = processing.decimate(data, q)
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
  channels = []
  with h5py.File(filename, 'r') as f:
    if len(f.keys()) == 6:
      channels.append(f.get('JRSC_HNE')[()])
      channels.append(f.get('JRSC_HNN')[()])
      channels.append(f.get('JRSC_HNZ')[()])
      channels.append(f.get('JSFB_HNE')[()])
      channels.append(f.get('JSFB_HNN')[()])
      channels.append(f.get('JSFB_HNZ')[()])
      return np.stack(channels, axis=0)
  return None


def process(file_pattern, in_dir, out_dir, raw_window, detect_window,
            event_duration, low_freq, high_freq, dt, q
            ):
  filenames = processing.get_filenames(file_pattern)

  for i, filename in enumerate(filenames):
    if i % 1000 == 0:
      logging.info('Processed %s files.', i)
    data = read_hdf5(filename)
    if data is not None:
      data = _process(data, low_freq, high_freq, dt, q)
      data = _crop(data, raw_window, detect_window, event_duration, dt * q)
      label = _get_label(filename)
      out_file = filename.replace(in_dir, out_dir)
      os.makedirs(os.path.dirname(out_file), exist_ok=True)
      out_file = out_file.replace('.hdf5', '.h5')
      write_hdf5(out_file, data, label)


def main():
  datatype = 'geophone'
  datapath = os.path.join(parameters.raw_datapath, datatype)
  file_pattern = os.path.join(datapath, '*/*/*')
  process(
      file_pattern,
      in_dir=parameters.raw_datapath,
      out_dir=parameters.processed_datapath,
      raw_window=parameters.raw_window_length,
      detect_window=parameters.detect_window_length,
      event_duration=parameters.event_duration,
      low_freq=parameters.low_freq,
      high_freq=parameters.high_freq,
      dt=parameters.geophone_dt,
      q=parameters.geophone_downsampling_factor,
  )


if __name__ == '__main__':
  main()
