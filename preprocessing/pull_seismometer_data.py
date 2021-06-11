"""Downloads waveforms from the USGS network."""

import datetime as dt
from functools import partial
import logging
import multiprocessing
import os
import sys
from typing import List, Text

import h5py
import numpy as np
import pandas as pd
from obspy.clients.fdsn import Client

from preprocessing import parameters


logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


class WaveformDownloader():
  """Class for downloading waveforms from the seismic network.

  Attr:
    client: Seismic network client.
    network: Network code.
    stations: Seismic stations for which to download the data.
    channels: Channels for which to download the data.
    location: Location for which to download the data.
    datapath: Path to where to write the data.
    """

  def __init__(
      self, clientcode: Text, network: Text, stations: List[Text],
      channels: List[Text], location: Text, datapath: Text,
  ):
    """Initialization."""
    # Seismic network parameters
    self.client = Client(clientcode)
    self.network = network
    self.stations = ','.join(stations)
    self.channels = ','.join(channels)
    self.location = location

    # Writing parameters
    self.datapath = datapath

  def get_catalog_waveforms(self, catalog: Text, prefix: Text, window: int,
                            batch: int = 1000, n_threads: int = 8):
    """Gets the waveforms corresponding to the time stamps in the catalog.

    The waveforms are downloaded from the seismic network client.

    Args:
      catalog: Event catalog file.
        Must have a column named 'datetime' indicating the event time.
      prefix: Label for identifying the type of waveforms.
      window: Length of the data window (seconds).
      batch: Organize the files into subdirectories with `batch` number of
        files.
      n_threads: Number of threads to use.
    """
    logging.info('Downloading %s waveforms...', prefix)
    datapath = os.path.join(self.datapath, 'seismometer/{}'.format(prefix))
    logging.info('Writing waveforms to %s', datapath)
    df = pd.read_hdf(catalog, 'df')
    with multiprocessing.Pool(n_threads) as pool:
      pool.starmap(
          partial(
              _get_catalog_waveform, client=self.client, network=self.network,
              stations=self.stations, channels=self.channels,
              location=self.location, window=window, datapath=datapath,
              prefix=prefix, batch=batch,
          ),
          enumerate(df['datetime']))

  def get_continuous_waveforms(
      self, starttime: dt.datetime, endtime: dt.datetime, window: int,
      n_threads: int = 8
  ):
    """Gets continuous data between `starttime` and `endtime`.

    The waveforms are downloaded from the seismic network client.

    Args:
      starttime: Start time for the continuous data.
      endtime: End time for the continuous data.
      window: Length of the data windows (seconds).
        The continuous data is divided into files of `window` length.
      n_threads: Number of threads to use.
    """
    logging.info('Downloading continuous waveforms...')
    datapath = os.path.join(self.datapath, 'seismometer/continuous')
    os.makedirs(datapath, exist_ok=True)
    logging.info('Writing waveforms to %s', datapath)

    starttimes = [starttime, ]
    while starttimes[-1] < endtime:
      starttimes.append(starttimes[-1] + dt.timedelta(seconds=window))
    filenames = []
    for timestamp in starttimes:
      filenames.append(os.path.join(
          datapath, 'data_{}.h5'.format(timestamp.strftime('%Y%m%d_%H%M%S'))))
    n_threads = min(n_threads, len(starttimes))
    with multiprocessing.Pool(n_threads) as pool:
      pool.starmap(
          partial(
              _get_waveform, window=window, client=self.client,
              network=self.network, stations=self.stations,
              channels=self.channels, location=self.location,
          ),
          zip(filenames, starttimes)
      )


def _get_catalog_waveform(
        i: int, eventtime: dt.datetime, client: Client, network: Text,
        stations: Text, channels: Text, location: Text, window: int,
        datapath: Text, prefix: Text, batch: int,
):
  """Gets a catalog waveform from the seismic network.

  This function is implemented outside of the class to avoid conflict
  between multiprocessing and pickled objects.

  Args:
    i: Event ID number.
    eventtime: Time of the event.
    client: Seismic network client.
    network: Seismic network code.
    stations: Seismic stations for which to download the data.
    channels: Channels for which to download the data.
    location: Location for which to download the data.
    window: Length of the data window (seconds).
    datapath: Path to where to write the data.
    prefix: Label for identifying the type of waveforms.
    batch: Divide the files into subdirectories with `batch` number of files.
  """
  subfolder = os.path.join(datapath, '{:05d}'.format(i // batch * batch))
  os.makedirs(subfolder, exist_ok=True)
  starttime = eventtime - dt.timedelta(seconds=window//2)
  filename = os.path.join(subfolder, '{}_{:05d}.h5'.format(prefix, i))
  _get_waveform(
      filename=filename, starttime=starttime, window=window, client=client,
      network=network, stations=stations, channels=channels, location=location,
  )


def _get_waveform(
        filename: Text, starttime: dt.datetime, window: int,
        client: Client, network: Text, stations: Text, channels: Text,
        location: Text,
):
  """Downloads a waveform from the seismic network.

  This function is implemented outside of the class to avoid conflict
  between multiprocessing and pickled objects.

  Args:
    filename: File to which to write the downloaded waveform.
    starttime: Start of the time window to download.
    window: Length of the data window (seconds).
    client: Seismic network client.
    network: Seismic network code.
    stations: Seismic stations for which to download the data.
    channels: Channels for which to download the data.
    location: Location for which to download the data.
  """
  endtime = starttime + dt.timedelta(seconds=window)
  try:
    st = client.get_waveforms(
        network=network, station=stations, channel=channels,
        location=location, starttime=starttime, endtime=endtime)
  except Exception as e:
    print(e)
    st = None

  if st is not None:
    if not os.path.isfile(filename):
      f = h5py.File(filename, 'w')
    else:
      f = h5py.File(filename, 'a')
    for tr in st:
      dataset_name = '{}.{}'.format(tr.stats.station, tr.stats.channel)
      if dataset_name not in f.keys():
        waveform = np.asarray(tr, dtype=np.float32)
        f.create_dataset(dataset_name, data=waveform)
  else:
    logging.warning('Could not download data for %s.',
                    os.path.basename(filename))


def parse_args():
  """Parse arguments."""
  available_options = ['all', 'event', 'noise', 'continuous']

  option = 'all'
  if len(sys.argv) > 1:
    option = sys.argv[1]
    if option not in available_options:
      print("Argument should be 'all', 'event', 'noise', or 'continuous'")
      sys.exit()
  return option


def main():
  """Download waveforms from the seismic network."""
  waveform_downloader = WaveformDownloader(
      clientcode=parameters.clientcode, network=parameters.network,
      stations=parameters.stations, channels=parameters.channels,
      location=parameters.location, datapath=parameters.raw_datapath,
  )
  option = parse_args()
  if option in ['all', 'event']:
    waveform_downloader.get_catalog_waveforms(
        parameters.event_catalog, 'event', window=parameters.raw_window_length,
        batch=parameters.batch, n_threads=parameters.n_threads
    )
  if option in ['all', 'noise']:
    waveform_downloader.get_catalog_waveforms(
        parameters.noise_catalog, 'noise', window=parameters.raw_window_length,
        batch=parameters.batch, n_threads=parameters.n_threads
    )
  if option in ['all', 'continuous']:
    waveform_downloader.get_continuous_waveforms(
        starttime=parameters.continuous_starttime,
        endtime=parameters.continuous_endtime,
        window=parameters.continuous_window,
        n_threads=parameters.n_threads
    )


if __name__ == "__main__":
  main()
