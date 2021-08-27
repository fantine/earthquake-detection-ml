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


def write_hdf5(out_file, data, label=None):
  with h5py.File(out_file, 'w') as f:
    f.create_dataset('input', data=data)
    if label is not None:
      f.create_dataset('label', data=label)


def read_hdf5(filename):
  channels = []
 # f = h5py.File(filename, 'a')
 # f.close()
  with h5py.File(filename, 'r') as f:
    if len(f.keys()) == 6:
      channels.append(f.get('JRSC.HNE')[()])
      channels.append(f.get('JRSC.HNN')[()])
      channels.append(f.get('JRSC.HNZ')[()])
      channels.append(f.get('JSFB.HNE')[()])
      channels.append(f.get('JSFB.HNN')[()])
      channels.append(f.get('JSFB.HNZ')[()])
      try:
        return np.stack(channels, axis=0)
      except:
        return None
  return None


def process_windows(file_pattern, in_dir, out_dir, raw_window, detect_window,
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


def process_continuous(file_pattern, in_dir, out_dir, raw_window, low_freq,
                       high_freq, dt, q):
  filenames = processing.get_filenames(file_pattern)

  for i, filename in enumerate(filenames):
    if i % 1000 == 0:
      logging.info('Processed %s files.', i)
    out_file = filename.replace(in_dir, out_dir)
    if  not os.path.exists(out_file):
      data = read_hdf5(filename)
      if data is not None and data.shape[1] == 8640000:
        data = _process(data, low_freq, high_freq, dt, q)
     # clip_values = np.array([ 31.26569397,  26.99611814,  24.20824539, 166.83790821, 156.57493343, 170.47219654], dtype=np.float32)
     # clip_values = np.expand_dims(clip_values, axis=1)
     # std_values = np.array(
     #     [ 4.199149 ,  3.9638348,  3.3899522, 23.58608  , 22.261927 ,
     #  24.283094 ],
     #     dtype=np.float32
     # )
     # std_values = np.expand_dims(std_values, axis=1)
     # data = np.clip(data, -clip_values, clip_values) / std_values
        data = data.T
     # out_file = filename.replace(in_dir, out_dir)
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        write_hdf5(out_file, data)


def main():
  datatype = 'seismometer'
  datapath = os.path.join(parameters.raw_datapath, datatype)
  # file_pattern = os.path.join(datapath, '*/*/*')
  # process_windows(
  #     file_pattern,
  #     in_dir=parameters.raw_datapath,
  #     out_dir=parameters.processed_datapath,
  #     raw_window=parameters.raw_window_length,
  #     detect_window=parameters.detect_window_length,
  #     event_duration=parameters.event_duration,
  #     low_freq=parameters.low_freq,
  #     high_freq=parameters.high_freq,
  #     dt=parameters.seismometer_dt,
  #     q=parameters.seismometer_downsampling_factor,
  # )
  file_pattern = os.path.join(datapath, 'continuous/*')
  print(file_pattern)
  process_continuous(
      file_pattern,
      in_dir=parameters.raw_datapath,
      out_dir=parameters.processed_datapath,
      raw_window=parameters.continuous_window,
      low_freq=parameters.low_freq,
      high_freq=parameters.high_freq,
      dt=parameters.seismometer_dt,
      q=parameters.seismometer_downsampling_factor,
  )


if __name__ == '__main__':
  main()
