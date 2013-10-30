#!/usr/bin/env python

import math
import sys,csv

import QSTK.qstkutil.qsdateutil as du
import QSTK.qstkutil.tsutil as tsu
import QSTK.qstkutil.DataAccess as da

import numpy as np
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt


def main(argv)	:
	reader = csv.reader(open("../data/values.csv", 'rU'), delimiter=',')
	dates = []
	fundvalue = [0]
	tradeCount = 0
	print "***************Portfolio Analyzer******************************"
	for row in reader:
		date = dt.datetime(int(row[0]), int(row[1]), int(row[2]), 16)
		dates.append(date)
		fundvalue.append(int(row[3]))
#	print fundvalue
	startdate = dates[0]
	enddate = dates[-1]
	dt_timeofday = dt.timedelta(hours=16)
	print "Fetching Data..."
	ldt_timestamps = du.getNYSEdays(startdate, enddate, dt_timeofday)
	c_dataobj = da.DataAccess('Yahoo', cachestalltime=0)
	ls_keys = ['close']
	ldf_data = c_dataobj.get_data(ldt_timestamps, [argv[1]], ls_keys)
	d_data = dict(zip(ls_keys, ldf_data))
	print "Calculating Portfolio Values..."
	na_price = d_data['close'].values
	na_normalized_price = na_price / na_price[0]
	na_dailyrets = na_normalized_price.copy()
	tsu.returnize0(na_dailyrets)
	na_dailyret = np.average(na_dailyrets)
	na_stdev = np.std(na_dailyrets)
	na_sharpe = math.sqrt(252) * na_dailyret / na_stdev
	na_cumret = na_normalized_price[-1]
	print 
	print 	
	print "***************************************************************"
	print argv[1]
	print "Daily Return         :    " + str(na_dailyret)
	print "Standard Deviation   :    " + str(na_stdev)
	print "Sharpe               :    " + str(na_sharpe)
	print "Cumulative Return    :    " + str(na_cumret)
	print "***************************************************************"
	
	na_price1 = np.array(fundvalue)
	plt.plot(na_price1)
	plt.show()
	na_price1 = na_price1[1:] * 1.0
	#print na_price1
	na_normalized_price1 = na_price1 / na_price1[0]
	#print na_normalized_price1
	na_dailyrets1 = na_normalized_price1.copy()
	tsu.returnize0(na_dailyrets1)
	#print na_dailyrets1
	na_dailyret1 = np.average(na_dailyrets1)
	na_stdev1 = np.std(na_dailyrets1)
	na_sharpe1 = math.sqrt(252) * na_dailyret1 / na_stdev1
	na_cumret1 = na_normalized_price1[-1]
	print 
	print 	
	print "***************************************************************"
	print "Fund Value"
	print "Daily Return         :    " + str(na_dailyret1)
	print "Standard Deviation   :    " + str(na_stdev1)
	print "Sharpe               :    " + str(na_sharpe1)
	print "Cumulative Return    :    " + str(na_cumret1)
	print "***************************************************************"
	print 
	print 	
	print "Done."
	
if __name__ == "__main__":
    main(sys.argv)
