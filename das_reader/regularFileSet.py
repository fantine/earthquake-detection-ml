from das_reader.namePart import *
from das_reader.fileSet import *
import math
import datetime


class regularFileSet(fileSet):
  '''This is a set of files that starts with a particular file time and includes many files that are evenly spaced by some number of seconds. If files in set are still being recorded nFiles -1 is the flag for that'''

  def __init__(self, listOfPartTypes, startYear, startMonth, startDay, startHour, startMin, startSec, startMillisec, secondsPerFile, nFiles=-1):
    fileSet.__init__(self, listOfPartTypes)
    self.startTime = datetime.datetime(
        int(startYear), int(startMonth), int(startDay), int(startHour), int(startMin), int(startSec), int(startMillisec*1000))
    self.secondsBetweenFiles = secondsPerFile
    # if nFiles = -1 it treats it as though the number of files is unknown and goes up until the current time possibly (this can be risky)
    self.nFiles = nFiles
    # typical case with a specified number of files in this range
    self.endTime = self.startTime + \
        datetime.timedelta(seconds=self.nFiles*self.secondsBetweenFiles)
    if(self.nFiles == 0):  # if number of files=-1 assume it goes until now
      self.endTime = datetime.now()

  def nameOfLastFile(self):
    '''Return the name of the last file in this file set'''
    startOfLastFileTime = self.endTime - \
        datetime.timedelta(seconds=self.secondsBetweenFiles)
    return self.generateFilenameDTObj(startOfLastFileTime)

  def addFilesToEnd(self, nFilesToAdd=1):
    '''Add a file to the end of the regularFileSet by adjusting the endTime and nFiles'''
    self.nFiles = self.nFiles + nFilesToAdd
    self.endTime = self.startTime + \
        datetime.timedelta(seconds=self.nFiles*self.secondsBetweenFiles)

  def addFilesToStart(self, nFilesToAdd=1):
    '''Add a file to the start of the regularFileSet by adjusting the startTime and nFiles'''
    self.nFiles = self.nFiles + nFilesToAdd
    self.startTime = self.startTime - \
        datetime.timedelta(seconds=self.nFilesToAdd*self.secondsBetweenFiles)

  def getFileNameIncludingTime(self, dateTimeObjectRequested):
    '''Based on the datetime.datetime object representing the time you want to grab, this generates the file name in this file set'''
    if((dateTimeObjectRequested < self.startTime) or (dateTimeObjectRequested > self.endTime)):
      return "ERROR file not in this fileSet"
    else:
      timeIntoFileSet = dateTimeObjectRequested - self.startTime  # timedelta object
      numberOfFilesIntoFileSet = math.floor(
          timeIntoFileSet.total_seconds()/self.secondsBetweenFiles)
      fileTime = self.startTime + \
          datetime.timedelta(
              seconds=self.secondsBetweenFiles*numberOfFilesIntoFileSet)
      return self.generateFilenameDTObj(fileTime)

  def getFileNamesInRange(self, dateTimeObjectStart, dateTimeObjectEnd):
    '''Returns a list in order of all files in this fileSet that are between the start and end datetime datetime.datetime objects.'''
    if(dateTimeObjectStart > dateTimeObjectEnd):
      print("ERROR in fileSet.py getFileNamesInRange: make sure start is before end")
      return ([], [], [])
    else:
      # return empty set of there's definitely no overlap of these times and the fileset
      if((dateTimeObjectStart > self.endTime) or (dateTimeObjectEnd < self.startTime)):
        return []
      else:
        fileList = []
        # calculate the first time these ranges have in common
        firstTime = self.startTime
        print(self.startTime, self.endTime)
        if(self.startTime < dateTimeObjectStart):  # if this file set starts before the requested range
          timeDiff = dateTimeObjectStart - self.startTime
          firstTime = self.startTime + datetime.timedelta(seconds=self.secondsBetweenFiles*(
              int(timeDiff.total_seconds())//int(self.secondsBetweenFiles)))

        # until you get to the last time these ranges have in common, continue generating the file names
        currentTime = firstTime
        timeToNextFile = datetime.timedelta(seconds=self.secondsBetweenFiles)
        lastCommonTime = min(dateTimeObjectEnd, self.endTime)
        while(currentTime < lastCommonTime):
          fileList.append(self.generateFilenameDTObj(currentTime))
          currentTime = currentTime + timeToNextFile
        return fileList

  def getFirstFileStartingThisDay(self, dateObject):
    '''Returns the first filename that contains data starting after 12:00 am on that date given a datetime.date object'''
    midnightDate = datetime.datetime(
        dateObject.year, dateObject.month, dateObject.day, 0, 0, 0, 0)  # start of the day
    # start of the next day and end of this day
    nextMidnightDate = datetime.timedelta(days=1)+midnightDate
    # if this day doesn't overlap with the file set, return an empty string
    if((midnightDate > self.endTime) or (nextMidnightDate < self.startTime)):
      return 0
    else:
      # if day starts before and extends into file set, use first file in file set
      firstTime = self.startTime
      # if day starts during file set, get file starting just after midnight
      if(self.startTime < midnightDate):
        timeDiff = midnightDate - self.startTime
        firstTime = self.startTime + datetime.timedelta(seconds=self.secondsBetweenFiles*(
            1+int(timeDiff.total_seconds())//int(self.secondsBetweenFiles)))
      firstFileName = self.generateFilenameDTObj(firstTime)
      return firstFileName

  def getLastFileStartingThisDay(self, dateObject):
    '''Returns the last filename that contains data starting before 11:59:99.9999 pm on that date given a datetime.date object'''
    midnightDate = datetime.datetime(
        dateObject.year, dateObject.month, dateObject.day, 0, 0, 0, 0)  # start of the day
    # start of the next day and end of this day
    nextMidnightDate = datetime.timedelta(days=1)+midnightDate
    # if this day doesn't overlap with this file set
    if((midnightDate > self.endTime) or (nextMidnightDate < self.startTime)):
      return 0
    else:
      # if day starts in and extends beyond file set, use last file in file set
      lastTime = self.endTime - \
          datetime.timedelta(seconds=self.secondsBetweenFiles)
      if(self.endTime > nextMidnightDate):  # if day ends during file set, get last file before midnight
        timeDiff = nextMidnightDate - self.startTime
        lastTime = self.startTime + datetime.timedelta(seconds=self.secondsBetweenFiles*(
            int(timeDiff.total_seconds())//int(self.secondsBetweenFiles)))
      lastFileName = self.generateFilenameDTObj(lastTime)
      return lastFileName

  def checkUnion(self, otherRFS):
    '''Checks if this RFS and another RFS are contiguous neighbors or overlapping. Returns False if they are not, and a regular file set made up of their union if they are.'''
    # Check if their start and end times line up and they have the same number of seconds per file
    theUnion = False
    timesOverlap = ((self.startTime <= otherRFS.startTime <= self.endTime) or (
        self.startTime <= otherRFS.endTime <= self.endTime))
    sameFileLengths = (self.secondsBetweenFiles ==
                       otherRFS.secondsBetweenFiles)
    if timesOverlap and sameFileLengths:
      theUnion = True

    # if they should be combined, do it!
    if theUnion:
      minStartTime = min(self.startTime, otherRFS.startTime)
      maxEndTime = max(self.endTime, otherRFS.endTime)
      nFilesTotal = int(
          (maxEndTime-minStartTime).total_seconds()//self.secondsBetweenFiles)
      theUnion = regularFileSet(self.listOfPartTypes, minStartTime.year, minStartTime.month, minStartTime.day, minStartTime.hour,
                                minStartTime.minute, minStartTime.second, int(0.001*minStartTime.microsecond), self.secondsBetweenFiles, nFilesTotal)

    return theUnion
