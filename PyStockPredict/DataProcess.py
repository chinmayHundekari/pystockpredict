#!/usr/bin/env python

import bp,containers
import pandas as pd
import numpy as np
import scipy.io as sio
import math, copy, time, sys, csv
import QSTK.qstkutil.qsdateutil as du
import datetime as dt
import QSTK.qstkutil.DataAccess as da
import QSTK.qstkutil.tsutil as tsu
import QSTK.qstkstudy.EventProfiler as ep

inputRows = []
outputRows = []
rowCount = 0

def printClass(quotes, Bollinger, sym, writer):
    global rowCount, inputRows, outputRows
    t_beg = -3 
    t_end = 3
    t_endl = t_end+1
    f_symprice = np.zeros(t_endl-t_beg)
    f_bolval = np.zeros(t_endl-t_beg)
    ldt_timestamps = Bollinger.index
    for x in range(20, len(ldt_timestamps)-t_end+t_beg): # 3...len-1
        i = x - t_beg
        input = np.zeros(t_end-t_beg-1)
        for t in range(t_beg,t_endl): # -3 -2 -1 0 1 2 3
            t1 = t-t_beg
            f_symprice[t1] = quotes.ix[ldt_timestamps[i + t]] #   -3 -2 -1 0 1 2 3   -3  
            f_bolval[t1] = Bollinger.ix[ldt_timestamps[i + t]]
            #print t1, f_bolval[t1]
            if ((t1 <= 1-t_beg) and (t1 > 0)):
                input[t1] = f_bolval[t1]
        input[0] = (f_symprice[-t_beg] - f_symprice[0])*10.0/f_symprice[0]
        if (np.isnan(sum(input))):
            continue
        input = input.reshape(1,input.size)
#       print input.reshape(input.size,1).shape
        y = bp.bpExec(input)
        if y == 2:
#            if quotes.ix[ldt_timestamps[i + t]]
            ldt_ts = ldt_timestamps[x]
            results = [str(ldt_ts.year),str(ldt_ts.month),str(ldt_ts.day),str(sym),"Buy",str(100),""]
            writer.writerow(results)
            i1 = i+3
            if i1 >= len(ldt_timestamps):
                i1 = len(ldt_timestamps)-1
            ldt_ts = ldt_timestamps[i1]
            results = [str(ldt_ts.year),str(ldt_ts.month),str(ldt_ts.day),str(sym),"Sell",str(100),""]
            writer.writerow(results)
            
              

 
def generateOrders(d_data):
    df_close = d_data['close']
    ls_symbols=list(d_data['close'].columns)
    resultFile = open("../data/orders.csv",'wb')
    writer = csv.writer(resultFile, dialect='excel')

    Bollinger = initBollinger(d_data, 20)

    for s_sym in ls_symbols:
        printClass(df_close[s_sym], Bollinger[s_sym], s_sym, writer)
    return 

def initBollinger(data, period):
    BOLLINGER = period 
    print "Calculating Bollinger Band Values..."
    means = pd.rolling_mean(data['close'],BOLLINGER,min_periods=BOLLINGER)
    rolling_std = pd.rolling_std(data['close'],BOLLINGER,min_periods=BOLLINGER)
    max_bol = means + rolling_std
    min_bol = means - rolling_std
    Bollinger = (d_data['close'] - means) / (rolling_std)
    return Bollinger


def classifyBollinger(quotes, Bollinger):
    global rowCount, inputRows, outputRows
#    quotes = quotes[sym]
#    Bollinger = Bollinger[sym]
    t_beg = -3 
    t_end = 3
    t_endl = t_end+1
    f_symprice = np.zeros(t_endl-t_beg)
    f_bolval = np.zeros(t_endl-t_beg)
    ldt_timestamps = Bollinger.index
    for x in range(20, len(ldt_timestamps)-t_end+t_beg): # 3...len-1
        i = x - t_beg
        input = np.zeros(t_end-t_beg-1)
        for t in range(t_beg,t_endl): # -3 -2 -1 0 1 2 3
            t1 = t-t_beg
            f_symprice[t1] = quotes.ix[ldt_timestamps[i + t]] #   -3 -2 -1 0 1 2 3   -3  
            f_bolval[t1] = Bollinger.ix[ldt_timestamps[i + t]]
            #print t1, f_bolval[t1]
            if ((t1 <= 1-t_beg) and (t1 > 0)):
                input[t1] = f_bolval[t1]
        input[0] = (f_symprice[-t_beg] - f_symprice[0])*10.0/f_symprice[0]
        if (np.isnan(sum(input))):
            continue
        if (rowCount == 0):
            inputRows = input
        else:
            inputRows = np.vstack((inputRows,input))
        outputval  = 2 * (f_symprice[t_end] > (f_symprice[0] * 1.04)) + 1 * (f_symprice[t_end] < (f_symprice[0] *0.96))
        outputRows.append(outputval)
        rowCount = rowCount + 1

def processData(d_data):
    global inputRows, outputRows
    inputRows = []
    outputRows = []
    df_close = d_data['close']
    ls_symbols=list(d_data['close'].columns)

    Bollinger = initBollinger(d_data, 20)

    for s_sym in ls_symbols:
        print s_sym
        classifyBollinger(df_close[s_sym], Bollinger[s_sym])
    inputRows =np.array(inputRows)
    outputRows = np.array(outputRows) 
    print inputRows.shape
    return inputRows, outputRows 

def getData(startDate, endDate, symbols, cache=1):
    ldt_timestamps = du.getNYSEdays(startDate, endDate, dt.timedelta(hours=16))
    if (cache  == 1):
        dataobj = da.DataAccess('Yahoo')#, cachestalltime=0)
    else:
        dataobj = da.DataAccess(('Yahoo'), cachestalltime=0)
    symbols.append('SPY')
    ls_keys = ['open', 'high', 'low', 'close', 'volume', 'actual_close']
    print "Obtaining data"
    ldf_data = dataobj.get_data(ldt_timestamps, symbols, ls_keys)
    d_data = dict(zip(ls_keys, ldf_data))
    for s_key in ls_keys:
        d_data[s_key] = d_data[s_key].fillna(method = 'ffill')
        d_data[s_key] = d_data[s_key].fillna(method = 'bfill')
        d_data[s_key] = d_data[s_key].fillna(1.0)
	return d_data

if __name__ == '__main__':
#    values_file = sys.argv[1]
    dt_start = dt.datetime(2005, 1, 1)
    dt_end = dt.datetime(2008, 12, 31)
    ls_symbols = da.DataAccess('Yahoo').get_symbols_from_list('sp5002012')[:100]
    d_data = getData(dt_start,dt_end,ls_symbols,1)
    X,y = processData(d_data)
    sio.savemat('../data/bollinger_inputs.mat', {'X':X, 'y':y} )
    bp.bpTrain()

    generateOrders(d_data)        
