"""Create background noise catalog."""
import datetime as dt
import logging
import random

import pandas as pd

from earthquakes import parameters


random.seed(1337)


class NoiseCatalogBuilder():
  """Class for creating a catalog of background noise."""

  def __init__(self, event_catalog, num_entries):
    """Initialization.

    Args:
        event_catalog (str): path to the event catalog.
        num_entries (int): number of entries to create.
    """
    self.catalog = pd.DataFrame(index=range(num_entries), columns=['datetime'])
    self.num_entries = num_entries
    self.event_catalog = pd.read_pickle(event_catalog)
    self.event_catalog = self.event_catalog.sort_values(by=['datetime'])

  def create_noise_catalog(self):
    """Creates a background noise catalog.
    The background noise windows are randomly selected within the time span of
    the event catalog, while ensuring they are at least 5 minutes away from any
    recorded event.
    """
    num_events = len(self.event_catalog)
    catalog_starttime = self.event_catalog.at[0, 'datetime']
    catalog_endtime = self.event_catalog.at[num_events - 1, 'datetime']

    buffer = dt.timedelta(minutes=5)
    timespan = int((catalog_endtime - catalog_starttime).total_seconds())

    count = 0
    while count < self.num_entries:
      random_time_shift = dt.timedelta(seconds=random.randrange(timespan))
      random_time = catalog_starttime + random_time_shift
      if self._check_interval(random_time - buffer, random_time + buffer):
        self.catalog.at[count, 'datetime'] = random_time
        count += 1

  def _check_interval(self, starttime, endtime):
    """Checks whether there are no recorded events in the time interval.

    Args:
        starttime (datetime): start of time interval.
        endtime (datetime): end of time interval.

    Returns:
        bool: True if there are no events in the time interval.
    """
    # search is faster on a sorted dataframe
    left_count = self.event_catalog['datetime'].searchsorted(
        starttime, 'right')
    right_count = self.event_catalog['datetime'].searchsorted(
        endtime, 'right')
    return (right_count - left_count) == 0


def main():
  """Create a catalog of background noise time stamps."""
  logging.info('Creating background noise catalog...')
  catalog_builder = NoiseCatalogBuilder(
      event_catalog=parameters.event_catalog,
      num_entries=parameters.num_noise_examples
  )

  catalog_builder.create_noise_catalog()

  logging.info('Saving catalog to file: %s', parameters.noise_catalog)
  catalog_builder.catalog.to_pickle(parameters.noise_catalog)


if __name__ == "__main__":
  main()
