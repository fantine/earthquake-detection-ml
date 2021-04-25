"""Parameters for data preprocessing."""

import datetime
import os

from config import get_datapath

# pylint: disable=invalid-name

# ---- catalog parameters -----------------------------------------------------
num_noise_examples = 30000
event_catalog = 'catalog/earthquake_catalog.h5'
noise_catalog = 'catalog/noise_catalog.h5'

# ---- datapath ---------------------------------------------------------------
datapath = get_datapath.get_datapath()

# ---- USGS seismic network parameters ----------------------------------------
clientcode = "NCEDC"

starttime = datetime.datetime(year=2016, month=9, day=2)
endtime = datetime.datetime(year=2019, month=12, day=10)

# Coordinates of the Bay Area
min_latitude = 37.257  # degrees
max_latitude = 38.260  # degrees
min_longitude = -122.577  # degrees
max_longitude = -121.702  # degrees

min_radius = 0.0  # degrees
max_radius = 1.0  # degrees
stanford_latitude = 37.429354  # positive because northern
stanford_longitude = -122.175839  # negative because western
stanford_elevation = 29.0  # meters

closest_station = 'JSFB'
reference_channel = 'EHZ'

network = 'NC'
stations = 'JSFB,'
channels = '*'
location = '*'

# ---- DAS processing parameters ----------------------------------------------
raw_datapath = os.path.join(datapath, 'raw_data')
processed_datapath = os.path.join(datapath, 'processed_data')

start_channel = 14
end_channel = 310

batch = 1000
low_freq = 1.0  # Hz
high_freq = 12.0  # Hz
# we save time windows from event_time - window//2 to event_time + window//2
raw_window_length = 60  # seconds
passive_das_sampling = 50  # samples per second
clip_percentile = 99.5
detect_window_length = 20.48
event_duration = 12
das_dt = 0.02
das_downsampling_factor = 2
geophone_dt = 0.01
geophone_downsampling_factor = 4
channel_subset1 = list(range(13, 301))
channel_subset2 = list(range(328, 616))
