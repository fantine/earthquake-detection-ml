"""DAS data processing."""

import logging
import os

import h5py
import numpy as np
from cwt import helper
from cwt import cwt
from processing_utils import processing_utils as processing

from preprocessing import parameters


logging.basicConfig(level=logging.INFO)


def _process(data):
  # x = data.T
  # dx = 1
  # dt = 1/25
  # # Find the next power of two
  # npad = helper.next_p2(x.shape[0] + 1)
  # # Padding parameters
  # tdiff = (npad - x.shape[0]) // 2
  # bdiff = npad - x.shape[0] - tdiff
  # # Find the next power of two
  # npad = helper.next_p2(x.shape[1] + 1)
  # # Padding parameters
  # ldiff = (npad - x.shape[1]) // 2
  # rdiff = npad - x.shape[1] - ldiff
  # # Pad data
  # x_pad = np.pad(x, ((tdiff, bdiff), (ldiff, rdiff)), 'constant')
  # dj = 1  # voices per octave
  # # Wavelet scaling factors
  # scales = cwt.dyadic_scales(x.shape[0], dj)
  # # This function computes the dyadic scales over the full space
  # # but we actually do not need the largest scales
  # scales = scales[:3]
  # # Wavelet rotation angles
  # nthetas = 6
  # thetas = np.pi * np.linspace(0, 1, nthetas + 1)
  # thetas = thetas[3:4]
  # X = cwt.cwt2d(x_pad, scales, thetas, dt, dx)
  # X = X[:, :, tdiff:-bdiff, ldiff:-rdiff]
  # return np.abs(np.squeeze(X).T)

  x = data.T
  dx = 1
  dt = 0.04
  # Find the next power of two
  npad = helper.next_p2(x.shape[0] + 1)
  # Padding parameters
  tdiff = (npad - x.shape[0]) // 2
  bdiff = npad - x.shape[0] - tdiff
  # Find the next power of two
  npad = helper.next_p2(x.shape[1] + 1)
  # Padding parameters
  ldiff = (npad - x.shape[1]) // 2
  rdiff = npad - x.shape[1] - ldiff
  # Pad data
  x_pad = np.pad(x, ((tdiff, bdiff), (ldiff, rdiff)), 'constant')
  dj = 1  # voices per octave
  # Wavelet scaling factors
  scales = cwt.dyadic_scales(x.shape[0], dj)
  # This function computes the dyadic scales over the full space
  # but we actually do not need the largest scales
  scales = scales
  # Wavelet rotation angles
  nthetas = 6
  thetas = np.pi * np.linspace(0, 1, nthetas + 1)
  thetas = thetas[1:]
  X = cwt.cwt2d(x_pad, scales, thetas, dt, dx)
  X = X[:, :, tdiff:-bdiff, ldiff:-rdiff]
  labels = np.zeros_like(X[:, :, 0, 0])
  labels[0] = 1
  labels[0, 2] = 0
  labels[1] = 1
  labels[1, 2] = 0
  labels[2] = 1
  labels[2, 2] = 0
  labels[3] = 1
  labels[3:2] = 0
  mask = np.zeros_like(X)
  clip = 98
  for i in range(labels.shape[0]):
    for j in range(labels.shape[1]):
      if labels[i, j] == 1:
        threshold = np.percentile(X[i, j], clip)
        mask[i, j] = np.where(np.abs(X[i, j]) > threshold, 1, 0)
  mute = mask == 1
  Y = np.copy(X)
  Y[mute] = 0
  ix_mute = cwt.icwt2d(Y, scales, thetas, dt, dx)
  return ix_mute


def append_hdf5(out_file, data):
  with h5py.File(out_file, 'a') as f:
    f.create_dataset('icwt', data=data)


def read_hdf5(filename):
  with h5py.File(filename, 'r') as f:
    return f.get('input')[()]


def process(file_pattern):
  filenames = processing.get_filenames(file_pattern)

  for i, filename in enumerate(filenames):
    if i % 1000 == 0:
      logging.info('Processed %s files.', i)
    data = read_hdf5(filename)
    data = _process(data)
    append_hdf5(filename, data)


def main():
  datatype = 'das'
  datapath = os.path.join(parameters.processed_datapath, datatype)
  # file_pattern = os.path.join(datapath, '*/*/*')
  file_pattern = os.path.join(datapath, 'event/00000/event_000*')
  process(
      file_pattern,
  )


if __name__ == '__main__':
  main()
