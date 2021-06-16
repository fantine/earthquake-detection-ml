import os
import datetime

import h5py
import pandas as pd

from preprocessing import parameters


def write(filename, data):
  with h5py.File(filename, 'w') as f:
    f.create_dataset('data', data=data)


def extract_windows(df, in_path, out_path, window_length, dt):
  for i, timestamp in enumerate(df.datetime):
    starttime = datetime.datetime(
        timestamp.year, timestamp.month, timestamp.day)
    filename = os.path.join(
        in_path, 'data_{}.h5'.format(starttime.strftime('%Y%m%d_%H%M%S')))
    with h5py.File(filename, 'r') as f:
      data = f.get('data')[()]
      timesample = (timestamp - starttime).total_seconds() // dt
      startsample = timesample - window_length // (2 * dt)
      endsample = timesample + window_length // (2 * dt)
      out_file = os.path.join(out_path, 'fp_{:05d}.h5'.format(i))
      write(out_file, data[:, startsample:endsample])


def main():
  catalog_file = 'fp.h5'
  in_path = '/data/biondo/fantine/earthquake-detection-ml/continuous/'
  out_path = '/scratch/fantine/earthquake-detection-ml/false_positives/'
  window_length = 30
  dt = parameters.das_dt
  catalog = pd.read_hdf(catalog_file)
  catalog = catalog.sort_values(by=['datetime'])

  extract_windows(catalog, in_path, out_path, window_length, dt)


if __name__ == "__main__":
  main()
