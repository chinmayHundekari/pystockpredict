#!/usr/bin/env python

import csv
import math as math

import QSTK.qstkutil.qsdateutil as du
import QSTK.qstkutil.tsutil as tsu
import QSTK.qstkutil.DataAccess as da

import numpy as np
import datetime as dt
import pandas as pd

def _csv_read_sym_dates(filename):
	print "Reading Symbols..."
	reader = csv.reader(open(filename, 'rU'), delimiter=',')
	symbolList = []
	dateList = []
	for row in reader:
		if not(row[3] in symbolList):
			symbolList.append(row[3])
		date = dt.datetime(int(row[0]), int(row[1]), int(row[2]), 16)
		if not(date in dateList):
			dateList.append(date)
	dateList = sorted(dateList)
	return symbolList, dateList

def _csv_read_trades(filename):
	print "Reading Trades..."
	reader = csv.reader(open(filename, 'rU'), delimiter=',')
	symbolList = []
	dateList = []
	dates = []
	symbols = []
	orders = []
	volume = []
	tradeCount = 0
	for row in reader:
		#print row
		date = dt.datetime(int(row[0]), int(row[1]), int(row[2]), 16)
		if not(date in dateList):
			dateList.append(date)
		if not(row[3] in symbolList):
			symbolList.append(row[3])
		dates.append(date)
		symbols.append(row[3])
		orders.append(row[4])
		volume.append(int(row[5]))
		tradeCount = tradeCount +1
	dateList = sorted(dateList)
	return dates,symbols,orders,volume,tradeCount,symbolList, dateList
	
def main() :
	initial_cash = 1000000
	orders_file = "../data/orders.csv"
	values_file = "../data/values.csv"
	netCash = initial_cash 
	netValue = []
	resultFile = open(values_file,'wb')
	writer = csv.writer(resultFile, dialect='excel')
	
	print "***************Market Simulator*********************************"
	dates,symbols,orders,volume,tradeCount,symbolList,dateList = _csv_read_trades(orders_file)
	df = pd.read_csv(orders_file, parse_dates=True, names=['year','month','day', 'symbol', 'order', 'size', 'empty'], header=0 )
	del df['empty']
	df = df.sort(columns=['year','month','day'],ascending=1)
	#print df
	startdate = dateList[0]
	enddate = dateList[-1]
	dt_timeofday = dt.timedelta(hours=16)
	print "Fetching Data..."
	ldt_timestamps = du.getNYSEdays(startdate, enddate, dt_timeofday)
	c_dataobj = da.DataAccess('Yahoo')#, cachestalltime=0)
	ls_keys = ['close']
	ldf_data = c_dataobj.get_data(ldt_timestamps, symbolList, ls_keys)
	d_data = dict(zip(ls_keys, ldf_data))
	for s_key in ls_keys:
		d_data[s_key] = d_data[s_key].fillna(method = 'ffill')
		d_data[s_key] = d_data[s_key].fillna(method = 'bfill')
		d_data[s_key] = d_data[s_key].fillna(1.0)
	na_price = d_data['close'].values
	#print na_price.shape
	ownedStocks = np.zeros(na_price.shape[1])
	date_index = 0
	print "Processing Orders..."
	for ldt_ts in ldt_timestamps:
		order_count = 0	
		for order_date in dates:
			if ldt_ts == order_date:
				#print "New Comp:"
				#print symbols[order_count]
				#print order_date,ldt_ts
				symbol_index = 0
				for order_symbols in symbolList:
					if order_symbols == symbols[order_count]:
						cash = na_price[date_index][symbol_index] * volume[order_count]
						if orders[order_count] == "Buy":
							ownedStocks[symbol_index] = ownedStocks[symbol_index] + volume[order_count]
#							print ownedStocks
							cash = -cash
						else:
							ownedStocks[symbol_index] = ownedStocks[symbol_index] - volume[order_count]
#							print ownedStocks
						#ownedValue
						netCash = netCash + cash	
					symbol_index = symbol_index + 1
			order_count = order_count + 1
		#
		sym_idx = 0
		owned_value = 0
		for volume_own in ownedStocks:
			#print volume_own
			#print ownedStocks
			#print na_price[date_index][sym_idx]
			owned_value = owned_value + volume_own * na_price[date_index][sym_idx]
			#print owned_value
			sym_idx = sym_idx + 1
		#print netCash
		#print owned_value+netCash
		append1 = owned_value+netCash
		netValue.append(append1)
		results = [str(ldt_ts.year),str(ldt_ts.month),str(ldt_ts.day),str(int(append1))]
		writer.writerow(results)
		date_index = date_index + 1
	#print netValue
	print "Generating Output file..."
	print "Done."
	print "****************************************************************"
	
	
if __name__ == "__main__":
    main()