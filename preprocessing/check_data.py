import os
import re

import pandas as pd

from preprocessing import parameters
from processing_utils import processing_utils as processing


def check_data(catalog, file_pattern, key):
  df = pd.read_hdf(catalog, 'df')
  df[key] = False
  filenames = processing.get_filenames(file_pattern)
  regex_pattern = r'[a-z]+_(\d+)'
  for filename in filenames:
    index = int(re.match(regex_pattern, os.path.basename(filename)).group(1))
    df.at[index, key] = True
  df.to_hdf(catalog, key='df', mode='w')


def main():
  check_data(
      parameters.event_catalog,
      os.path.join(parameters.processed_datapath, 'das', 'event/*/*'),
      'has_das_data')
  check_data(
      parameters.noise_catalog,
      os.path.join(parameters.processed_datapath, 'das', 'noise/*/*'),
      'has_das_data')
  check_data(
      parameters.event_catalog,
      os.path.join(parameters.processed_datapath, 'geophone', 'event/*/*'),
      'has_geophone_data')
  check_data(
      parameters.noise_catalog,
      os.path.join(parameters.processed_datapath, 'geophone', 'noise/*/*'),
      'has_geophone_data')


if __name__ == '__main__':
  main()
