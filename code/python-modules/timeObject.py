'''
module: timeObject.py 
use: contains functions and class associated with datetime objects and converting .txt strings to usable times 
'''

import datetime
import string

class timeObject(object):
	def __init__(self,date_str,time_str):
		self.time = self.convertToDatetime(date_str,time_str)

	def parseTime(self,date_str,time_str):

		'''
		Purpose: 
		Converts MM/DD and hh:mm:ss.ms timestamps individual components MM, DD, hh, mm, ss, us of int outputs

		Inputs: 
		date_str: 			unparsed date string in MM/DD format
		time_str: 			unparsed time string of the hh:mm:ss.ms format

		Outputs: 
		MM:					int month
		DD:					int day
		hh: 				int hours
		mm: 				int minutes
		ss:					int seconds
		us: 				int microseconds
		'''
		MM,DD = string.split(date_str,'/')
		hh, mm, ss = string.split(time_str,':')
		ss, ms = string.split(ss,'.')
		MM = int(MM)
		DD = int(DD)
		if '.' in hh: 
			hh = string.split(hh,'.')[0]
		hh = int(hh)
		mm = int(mm)
		
		ss = int(ss)
		if len(ms) > 3: 
			ms = ms[:3]
		ms = int(ms)
		if hh > 24: 
			hh = 0

		return MM, DD, hh, mm, ss, ms

	def convertToDatetime(self,date_str,time_str):
		MM, DD, hh, mm, ss, ms = self.parseTime(date_str,time_str)
		us = int(ms*1000)
		time = datetime.datetime(1950,MM,DD,hh,mm,ss,us) #generic year
		return time
