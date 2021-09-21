"""Creates seismic event catalog using the Obspy client."""
import datetime as dt
import logging
import re
from typing import Text, Tuple

import numpy as np
from obspy.clients.fdsn import Client
from obspy.geodetics.base import gps2dist_azimuth
import pandas as pd
from scipy.optimize import curve_fit

from preprocessing import parameters


logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


class EventCatalogBuilder():
  """Class for creating a catalog of local seismic events."""

  def __init__(
      self,
      clientcode: Text,
      reference_coordinates: Tuple[float, float, float],
      reference_station: Text,
      reference_channel: Text,
  ):
    """Initialization.

    Args:
        clientcode: Obspy client code.
        reference_coordinates: (longitude, latitude, altitude)
            coordinates to be used for the radius search.
        reference_station: reference station code.
        reference_channel: reference channel code.
    """
    self.client = Client(clientcode)
    self.catalog = pd.DataFrame(
        columns=['id', 'type',
                 'focal_time', 'datetime',
                 'magnitude', 'distance', 'azimuth',
                 'latitude', 'longitude', 'depth',
                 'arrival_time', 'velocity_estimate', 'duration',
                 'onset', 'evaluation_mode', 'evaluation_status',
                 ])
    self.reference_coordinates = reference_coordinates
    self.station = reference_station
    self.channel = reference_channel

  def _get_preferred_pick(self, event):
    """Wrapper to parse event picks."""
    selected_pick = None
    for pick in event.picks:
      # only select picks from the reference station and channel
      if pick.waveform_id.station_code == self.station:
        if pick.waveform_id.channel_code == self.channel:
          # select pick with earliest arrival time
          if selected_pick is None or pick.time < selected_pick.time:
            selected_pick = pick
    return selected_pick

  def _get_duration(self, event):
    """Wrapper to parse event duration."""
    for amplitude in event.amplitudes:
      # only select picks from the reference station and channel
      if amplitude.waveform_id.station_code == self.station:
        if amplitude.waveform_id.channel_code == self.channel:
          if amplitude.category == 'duration':
            return amplitude.generic_amplitude
    return None

  def append_events_to_catalog(self, obspy_catalog):
    """Append the events from obspy_catalog to the catalog.

    Args:
        obspy_catalog (obspy.core.event.Catalog): Obspy container for events.
    """
    for event in obspy_catalog:
      origin = event.preferred_origin()
      pick = self._get_preferred_pick(event)
      self.catalog = self.catalog.append(
          {
              'id': _get_id(event),
              'type': event.event_type,
              'focal_time': origin.time.datetime,
              'magnitude': _get_magnitude(event),
              'latitude': origin.latitude,
              'longitude': origin.longitude,
              'depth': origin.depth,
              'arrival_time': pick.time.datetime if pick else None,
              'duration': self._get_duration(event),
              'onset': pick.onset if pick else None,
              'evaluation_mode': pick.evaluation_mode if pick else None,
              'evaluation_status': pick.evaluation_status if pick else None,
          },
          ignore_index=True)

  def _get_distance_azimuth(self, latitude, longitude, depth):
    """Compute distance and azimuth."""
    ref_latitude, ref_longitude, ref_elevation = self.reference_coordinates
    flat_distance, azimuth, _ = gps2dist_azimuth(
        ref_latitude, ref_longitude, latitude, longitude)
    distance = np.sqrt(flat_distance**2 + (-depth - ref_elevation)**2)
    return distance, azimuth

  def _fill_datetime(self):
    """Fill catalog with computed arrival times."""
    # for earthquakes, we approximate velocity as a linear function of distance
    # pylint: disable=invalid-name
    def _func(x, a, b):
      return a * x + b

    earthquakes = self.catalog[self.catalog.type == 'earthquake']
    earthquakes = earthquakes[earthquakes.velocity_estimate > 0]
    coef, _ = curve_fit(_func, earthquakes.distance,
                        earthquakes.velocity_estimate)

    # for quarry blasts, we take the mean velocity
    quarry_blasts = self.catalog[self.catalog.type == 'quarry blast']
    quarry_blast_velocity = np.mean(quarry_blasts.velocity_estimate)

    for i in range(len(self.catalog)):
      if self.catalog.at[i, 'type'] == 'quarry blast':
        time_lag = self.catalog.at[i, 'distance'] / quarry_blast_velocity
      else:
        velocity = _func(self.catalog.at[i, 'distance'], *coef)
        time_lag = self.catalog.at[i, 'distance'] / velocity
      self.catalog.at[i, 'datetime'] = (
          self.catalog.at[i, 'focal_time'] + dt.timedelta(seconds=time_lag))

  def fill_computed_columns(self):
    """Fills in computed values."""
    for i in range(len(self.catalog)):
      distance, azimuth = self._get_distance_azimuth(
          self.catalog.at[i, 'latitude'],
          self.catalog.at[i, 'longitude'],
          self.catalog.at[i, 'depth'])
      self.catalog.at[i, 'distance'] = distance
      self.catalog.at[i, 'azimuth'] = azimuth

      focal_time = self.catalog.at[i, 'focal_time']
      arrival_time = self.catalog.at[i, 'arrival_time']
      velocity = _get_velocity(distance, focal_time, arrival_time)
      self.catalog.at[i, 'velocity_estimate'] = velocity

    self._fill_datetime()

  def create_event_catalog(self, starttime, endtime, maxradius):
    """Calls the obspy client to create a seismic event catalog.
    The events are pulled within maxradius from the reference point.
    The catalog is stored in a dataframe.

    Args:
        starttime (datetime): beginning of query time window.
        endtime (datetime): end of query time window.
        maxradius (float): limit query to events within the specified maximum
            number of longitude/latitude degrees from the reference point.
    """
    # The obspy client times out when pulling too many events at the same time,
    # so we divide the server calls into shorter time windows.
    time_windows = _split_into_time_windows(starttime, endtime)

    reference_latitude, reference_longitude, _ = self.reference_coordinates
    for (window_starttime, window_endtime) in time_windows:
      logging.info('Pulling events from %s to %s...',
                   window_starttime, window_endtime)
      obspy_catalog = self.client.get_events(
          starttime=window_starttime,
          endtime=window_endtime,
          includearrivals=True,
          latitude=reference_latitude,
          longitude=reference_longitude,
          minradius=0,
          maxradius=maxradius,
          orderby='time-asc')
      self.append_events_to_catalog(obspy_catalog)

    self.fill_computed_columns()

    self.catalog.sort_values(by=['focal_time'])

    logging.info('Found %s events.', len(self.catalog))


def _split_into_time_windows(starttime, endtime, interval=6 * 30):
  """Splits the time window defined by starttime and endtime into 6-month time
  windows.

  Args:
      starttime (datetime): start of the time window.
      endtime (datetime): end of the time window.

  Raises:
      ValueError: starttime should be before endtime.

  Returns:
      list of datetime tuples: a list of time windows, each defined by a
      (start time, end time) tuple, subdividing the input time window into
      6-month chunks.
  """
  if starttime > endtime:
    raise ValueError('starttime should be before endtime.')

  windows = []
  window_starttime = starttime
  num_days = (endtime - window_starttime).days
  while num_days > interval:
    window_endtime = window_starttime + dt.timedelta(days=interval)
    windows.append((window_starttime, window_endtime))
    window_starttime = window_endtime
    num_days = (endtime - window_starttime).days
  windows.append((window_starttime, endtime))
  return windows


def _get_id(event):
  """Wrapper to parse event id."""
  resource_id = str(event.resource_id)
  # pylint: disable=anomalous-backslash-in-string
  match = re.search('(\d+)$', resource_id)
  return int(match.group(1)) if match else resource_id


def _get_magnitude(event):
  """Wrapper to get event magnitude.
  Magnitude is set to 'nan' when unavailable."""
  if event.preferred_magnitude() is None:
    return float('nan')
  return event.preferred_magnitude().mag


def _get_velocity(distance, focal_time, arrival_time):
  """Computes travel velocity from the available recorded arrival times."""
  if arrival_time is not None:
    time_lag = (arrival_time - focal_time).total_seconds()
    return distance / time_lag
  return None


def main():
  """Creates a catalog of seismic events close to Stanford."""
  logging.info('Creating seismic event catalog...')
  catalog_builder = EventCatalogBuilder(
      clientcode=parameters.clientcode,
      reference_coordinates=(
          parameters.stanford_latitude,
          parameters.stanford_longitude,
          parameters.stanford_elevation
      ),
      reference_station=parameters.closest_station,
      reference_channel=parameters.reference_channel
  )
  catalog_builder.create_event_catalog(
      starttime=parameters.starttime,
      endtime=parameters.endtime,
      maxradius=parameters.max_radius
  )

  logging.info('Saving catalog to file: %s', parameters.event_catalog)
  catalog_builder.catalog.to_hdf(parameters.event_catalog, key='df')


if __name__ == '__main__':
  main()
