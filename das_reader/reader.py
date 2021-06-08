import datetime as dt
import numpy as np
import sqlite3
from das_reader import regularFileSet as rfs
import obspy

from das_reader import parameters


class Reader:
  """Reader to access the DAS data"""

  def __init__(self, channels, sampling,
               dataPath=parameters.datapath,
               fileset_database=parameters.fileset_database,
               nTxtFileHeader=3200, nBinFileHeader=400, nTraceHeader=240):
    """Intialize the reader

    Args:
        dataPath (str): path to the data files
        rfsCursor (SQL cursor): cursor to the SQL database of regular file sets
        sampling (int): number of samples per second
        channels (list): list of channel numbers
    """
    self.dataPath = dataPath
    self.nameParts = [dataPath, 'year4', '/', 'month', '/', 'day', '/cbt_processed_', 'year4',
                      'month', 'day', '_', 'hour24start0', 'minute', 'second', '.', 'millisecond', '+0000.sgy']
    rfs_connect = sqlite3.connect(
        fileset_database, detect_types=sqlite3.PARSE_DECLTYPES)
    rfs_cursor = rfs_connect.cursor()
    self.RFSC = rfs_cursor
    self.sampling = sampling
    self.channels = channels
    self.nTxtFileHeader = nTxtFileHeader
    self.nBinFileHeader = nBinFileHeader
    self.nTraceHeader = nTraceHeader

  def readTrace(self, infile, nSamples, dataLen, traceNumber, endian, startSample, nSamplesToRead):
    """Read one trace

    Args:
        infile (str): SEGY file to read
        nSamples (int): number of samples per sensor
        traceNumber (int): sensor number (start with 1)
        dataLen (int): number of bytes per data sample

    Returns:
        1d numpy array"""

    with open(infile, 'rb') as fin:  # open file for reading in binary mode
      startData = self.nTxtFileHeader + self.nBinFileHeader + self.nTraceHeader + \
          (traceNumber - 1) * (self.nTraceHeader +
                               dataLen * nSamples) + startSample * dataLen
      fin.seek(startData)
      data = np.fromfile(fin, dtype=endian+'f', count=nSamplesToRead)
    return data

  def addLeapSecond(self, time):
    """Take leap second into an account"""

    timeOfLeapSecond = obspy.UTCDateTime(2016, 12, 31, 23, 59, 59)
    leapSecond = dt.timedelta(seconds=1)
    if obspy.UTCDateTime(time) > timeOfLeapSecond:
      time += leapSecond
    return time

  def locateData(self, startTime, windowLength):
    """Locates the DAS data for the time window starting at |startTime| and
    of length |windowLength| (in seconds), and returns the list of regular
    file sets covering the specified interval
    """

    endTime = startTime + dt.timedelta(seconds=windowLength)
    print(endTime)
    self.RFSC.execute(
        "SELECT * FROM regularFileSets WHERE NOT (startTime >= ? OR endTime <= ?)", (str(endTime), str(startTime)))
    rows = self.RFSC.fetchall()

    fileSetList = []
    for row in rows:
      thisSetStartTime = row[0]
      thisSetEndTime = row[1]
      secondsPerFile = row[2]
      nFiles = row[3]
      fileSetList.append(rfs.regularFileSet(self.nameParts, thisSetStartTime.year,
                                            thisSetStartTime.month, thisSetStartTime.day, thisSetStartTime.hour,
                                            thisSetStartTime.minute, thisSetStartTime.second, int(
                                                0.001*thisSetStartTime.microsecond),
                                            secondsPerFile, nFiles))

      fileSetsDisjoint = thisSetStartTime > fileSetList[-1].endTime
      if fileSetsDisjoint:
        print("Warning: No data for window %s (Disjoint file sets)" % startTime)
        return []

    if fileSetList:
      eventIncluded = ((fileSetList[0].startTime < startTime) and (
          fileSetList[-1].endTime > endTime))
      if not eventIncluded:
        print("Warning: No data for window %s (No file set)" % startTime)
        fileSetList = []
    else:
      print("Warning: No data for window %s (No file set)" % startTime)

    return fileSetList

  def readData(self, startTime, windowLength):
    fileSetList = self.locateData(startTime, windowLength)
    samplesPerWindow = windowLength * self.sampling
    nChannels = len(self.channels)
    data = np.zeros((nChannels, samplesPerWindow), dtype=np.float32)
    endTime = startTime + dt.timedelta(seconds=windowLength)

    startIdx = 0
    if not fileSetList:
      return None

    for fileSet in fileSetList:
      for fileName in fileSet.getFileNamesInRange(startTime, endTime):
        print(fileName)
        st = obspy.read(fileName, format='segy', headonly=True)
        thisSamplingRate = st.traces[0].stats.sampling_rate
        if thisSamplingRate != self.sampling:
          print("Warning: No data for window %s (Active data)" % startTime)
          return None
        fileStartTime = fileSet.getTimeFromFilename(fileName)
        secondsBetweenFiles = fileSet.secondsBetweenFiles
        fileEndTime = fileStartTime + dt.timedelta(seconds=secondsBetweenFiles)
        startIdxReading = 0
        if startTime > fileStartTime:
          secondsAfterStart = (startTime - fileStartTime).total_seconds()
          startIdxReading = int(self.sampling * secondsAfterStart)
        samplesPerFile = int(secondsBetweenFiles * self.sampling)
        endIdxReading = samplesPerFile
        if endTime < fileEndTime:
          secondsAfterStart = (endTime - fileStartTime).total_seconds()
          endIdxReading = int(self.sampling * secondsAfterStart)
        nIdxToRead = endIdxReading - startIdxReading
        endIdx = startIdx + nIdxToRead
        for chIdx, ch in enumerate(self.channels):
          data[chIdx, startIdx:endIdx] = self.readTrace(
              fileName, samplesPerFile, 4, ch, '>', startIdxReading, nIdxToRead)
        startIdx = endIdx  # index in data array
    return data
