import argparse
import enum
import logging
import os
import random
import re
import sys

import h5py
import numpy as np
import tensorflow as tf
import yaml


random.seed(42)


_DATAPATH_FILE = 'config/datapath.sh'


class CompressionType(enum.Enum):
  GZIP = 'GZIP'
  NONE = ''


_FILE_EXTENSION = {
    CompressionType.GZIP: '.gz',
    CompressionType.NONE: '',
}


logging.basicConfig(level=logging.INFO)


# def _float_feature(data):
#   return tf.train.Feature(float_list=tf.train.FloatList(value=data.reshape(-1)))

def _bytes_feature(data):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[data]))


def create_tf_example(inputs, labels):
  feature_dict = {
      'inputs': _bytes_feature(inputs.tobytes()),
      'labels': _bytes_feature(labels.tobytes()),
  }
  return tf.train.Example(features=tf.train.Features(feature=feature_dict))


class DataLoader():
  def __init__(self, min_val, max_val):
    self.min_val = min_val
    self.max_val = max_val

  def _clip_and_rescale(self, data):
    data = np.clip(data, self.min_val, self.max_val)
    return np.divide((data - self.min_val), (self.max_val - self.min_val))

  def read(self, filename):
    with h5py.File(filename, 'r') as f:
      data = f.get('input')[()]
      labels = f.get('label')[()]
    # if self.min_val != 0.0 or self.max_val != 1.0:
    #   inputs = self._clip_and_rescale(inputs)
      # min_clip = np.array(
      #     [-0.81,  -4.38,  -2.24, -14.0, -12.4, -12.7], dtype=np.float32)
      # max_clip = np.array(
      #     [0.81,  4.38,  2.24, 14.0, 12.4, 12.7], dtype=np.float32)
      # min_clip = np.expand_dims(min_clip, axis=1)
      # max_clip = np.expand_dims(max_clip, axis=1)
      # inputs = np.clip(inputs, min_clip, max_clip)
      # inputs = np.divide(inputs, max_clip - min_clip)
    # clip_values = np.array([6.0, 7.5, 5.6, 38.0, 37.0, 42.0], dtype=np.float32)
    # std_values = np.array(
    #     [1.5769932,  2.2115157,  1.618729, 11.568308, 10.987169, 12.1504755],
    #     dtype=np.float32
    # )

    # 99.9th
    clip_values = np.array(
        [178.15227144,  149.67709143, 124.48483422,
         873.76586145, 827.1183634, 857.89569763], dtype=np.float32)
    std_values = np.array(
        [8.12889331,  7.08172751,  5.95884035, 41.0858224, 38.99603783,
         41.18142846], dtype=np.float32)
    # 99.5th
    clip_values = np.array([38.73090591,  33.27859217,  29.42056465, 204.55128136,
                            192.66809975, 207.06877602], dtype=np.float32)
    std_values = np.array([3.79600604,  3.61591117,  3.02815053, 21.11769033, 19.97995761,
                           21.58822516], dtype=np.float32)
    # 99th
    clip_values = np.array([19.12533085,  16.99479103,  14.8097368, 100.73499916,
                            97.68273155, 106.57915833], dtype=np.float32)
    std_values = np.array([2.60649068,  2.75844068,  2.20224507, 15.27593551, 14.63964137,
                           15.97061868], dtype=np.float32)
    # 98th
    clip_values = np.array([9.29433527,  9.75987179,  7.43926465, 50.09132858, 48.78753105,
                            54.2594487], dtype=np.float32)
    std_values = np.array([1.75401556,  2.2324876,  1.63497632, 11.47794848, 10.901093,
                           11.95739858], dtype=np.float32)
    clip_values = np.expand_dims(clip_values, axis=1)
    std_values = np.expand_dims(std_values, axis=1)
    data = np.clip(data, -clip_values, clip_values) / std_values
    data = np.float32(data)
    data = data.T
    return data, labels


def _get_file_suffix(compression_type):
  return '.tfrecord{}'.format(_FILE_EXTENSION[compression_type])


def read_manifest(manifest_file):
  with open(manifest_file, 'r') as f:
    file_list = [line.rstrip() for line in f]
  logging.info('Converting %s files into TFRecords.', len(file_list))
  return file_list


def _glob(file_pattern):
  return sorted(tf.io.gfile.glob(file_pattern))


def create_manifest(manifest_file, file_pattern, shuffle=True):
  file_list = _glob(file_pattern)
  if shuffle:
    random.shuffle(file_list)
  os.makedirs(os.path.dirname(manifest_file), exist_ok=True)
  with open(manifest_file, 'w') as f:
    for filename in file_list:
      f.write(filename + '\n')


def _get_datapath():
  regex_pattern = r'DATAPATH="(\S+)"'
  with open(_DATAPATH_FILE, 'r') as f:
    datapath_text = f.read()
  regex_match = re.search(regex_pattern, datapath_text)
  if regex_match:
    return regex_match.group(1)
  raise ValueError(
      'Please set a correct datapath in {}'.format(_DATAPATH_FILE))


def convert_to_tfrecords(params):
  datapath = _get_datapath()
  manifest_file = os.path.join(datapath, params.manifest_file)
  if not os.path.exists(manifest_file):
    logging.info('Creating manifest file: %s', manifest_file)
    create_manifest(manifest_file, os.path.join(
        datapath, params.input_file_pattern))
  else:
    logging.info('Using the existing manifest file: %s', manifest_file)

  file_list = read_manifest(manifest_file)
  file_shards = np.array_split(file_list, params.num_shards)
  file_suffix = _get_file_suffix(params.compression_type)
  options = tf.io.TFRecordOptions(
      compression_type=params.compression_type.value)
  data_loader = DataLoader(params.min_val, params.max_val)
  output_file_prefix = os.path.join(datapath, params.output_file_prefix)

  os.makedirs(os.path.dirname(output_file_prefix), exist_ok=True)
  for i, file_shard in enumerate(file_shards):
    tfrecord_file = '{}-{:04d}-of-{:04d}{}'.format(
        output_file_prefix, i, params.num_shards, file_suffix)
    logging.info('Writing %s', tfrecord_file)
    with tf.io.TFRecordWriter(tfrecord_file, options=options) as writer:
      for filename in file_shard:
        filename = filename.replace('das', 'geophone')
        filename = filename.replace('_1.h5', '.h5')
        if os.path.isfile(filename):
          inputs, outputs = data_loader.read(filename)
          tf_example = create_tf_example(inputs, outputs)
          writer.write(tf_example.SerializeToString())


class ArgumentParser():

  def __init__(self):
    config_parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False)

    config_parser.add_argument(
        '-c', '--config-file',
        help='Parse script arguments from config file.',
        default=None,
        metavar='FILE')

    self._config_parser = config_parser

    self._parser = argparse.ArgumentParser(parents=[config_parser])

  @ staticmethod
  def _parse_config(items):
    argv = []
    for k, v in items:
      argv.append('--{}'.format(k))
      argv.append(v)
    return argv

  def _add_arguments(self, defaults=None):
    parser = self._parser

    parser.add_argument(
        '--input_file_pattern',
        help='Input data files.',
        default='',
    )
    parser.add_argument(
        '--output_file_prefix',
        help='Output file prefix.',
        default='tfrecords/',
    )
    parser.add_argument(
        '--input_height',
        help='Input data height.',
        type=int,
    )
    parser.add_argument(
        '--input_width',
        help='Input data width.',
        type=int,
        default=1,
    )
    parser.add_argument(
        '--input_depth',
        help='Input data depth.',
        type=int,
        default=1,
    )
    parser.add_argument(
        '--input_channels',
        help='Input data channels.',
        type=int,
        default=1,
    )
    parser.add_argument(
        '--num_shards',
        help='Number of shards to generate.',
        type=int,
        default=0,
    )
    parser.add_argument(
        '--compression_type',
        help='File compression type.',
        type=CompressionType,
        choices=list(CompressionType),
        default=CompressionType.GZIP,
    )
    parser.add_argument(
        '--manifest_file',
        help='Manifest file.',
        default='tfrecords/manifests/manifest.txt',
    )
    parser.add_argument(
        '--min_val',
        help='Minimum value.',
        type=float,
        default=0.0,
    )
    parser.add_argument(
        '--max_val',
        help='Maximum value.',
        type=float,
        default=1.0,
    )

  def parse_known_args(self, argv):
    args, remaining_argv = self._config_parser.parse_known_args(argv)
    if args.config_file:
      with open(args.config_file, 'r') as config:
        defaults = yaml.safe_load(config)
      defaults['config_file'] = args.config_file
    else:
      defaults = dict()
    self._add_arguments(defaults=defaults)
    self._parser.set_defaults(**defaults)

    return self._parser.parse_known_args(remaining_argv)


def main():
  params, _ = ArgumentParser().parse_known_args(sys.argv[1:])
  convert_to_tfrecords(params)


if __name__ == '__main__':
  main()
