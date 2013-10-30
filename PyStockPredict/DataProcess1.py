#!/usr/bin/env python
#python Dataprocess1.py
# latest version
import bp
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import scipy.io as sio
import time, csv
import QSTK.qstkutil.qsdateutil as du
import datetime as dt
import QSTK.qstkutil.DataAccess as da
#import QSTK.qstkutil.tsutil as tsu
#import QSTK.qstkstudy.EventProfiler as ep

inputRows = []
outputRows = []
rowCount = 0

def printGraph(s_sym,quotes, Bollinger, sym):
    t_beg = -3 
    t_end = 3
    t_endl = t_end+1
    f_symprice = np.zeros(t_endl-t_beg)
    ldt_timestamps = Bollinger[0][s_sym].index
    buy = pd.Series(np.zeros(len(Bollinger[0])), index=Bollinger[0].index) 
    sell = pd.Series(np.zeros(len(Bollinger[0])), index=Bollinger[0].index) 
    succ = pd.Series(np.zeros(len(Bollinger[0])), index=Bollinger[0].index) 
	
    for x in range(20, len(ldt_timestamps)-t_end+t_beg): # 3...len-1
        i = x - t_beg
        input = np.zeros((t_endl-t_beg)*len(Bollinger))
        for t in range(t_beg,t_endl): # -3 -2 -1 0 1 2 3
            t1 = t-t_beg
            f_symprice[t1] = quotes.ix[ldt_timestamps[i + t]] #   -3 -2 -1 0 1 2 3   -3  
            for j in range(0, len(Bollinger)):
                input[t1+j*(t_endl-t_beg)] = Bollinger[j][s_sym].ix[ldt_timestamps[i + t]]
        if (np.isnan(sum(input))):
            continue
        input = input.reshape(1,input.size)
        y = bp.bpExec(input)
        if y == 2:
            buy[i] = quotes[i] * 0.70
            i1 = i+3
            if i1 >= len(ldt_timestamps):
                i1 = len(ldt_timestamps)-1
            sell[i1] = quotes[i1] * 1.30
            if (quotes[i1] > quotes[i]):
                succ[i1] = quotes[i1] * 0.50
			
    print "Plotting"
    fig = plt.figure()
    fig.set_size_inches(10,7)
    ax = fig.add_subplot(111)
    ax.plot(ldt_timestamps, list(quotes), 'b-')
    ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')
    fig.autofmt_xdate()
    #print quotes
    plt.plot(ldt_timestamps, buy, 'g^') 
    plt.plot(ldt_timestamps, sell, 'rv') 
    plt.plot(ldt_timestamps, succ, 'b*') 
    plt.title(s_sym)
    str = "../data/" + s_sym + ".png"
    fig.savefig(str)
    fig.clf()

def showGraph(stock, d_data):
    df_close = d_data['close']
    ls_symbols=list(d_data['close'].columns)[0]
    Bollinger = initBollinger(d_data)
    printGraph(stock, df_close[ls_symbols], Bollinger, ls_symbols)
    return 

def printClass(quotes, Bollinger, s_sym, writer):
    global rowCount, inputRows, outputRows
    t_beg = -3 
    t_end = 3
    t_endl = t_end+1
    f_symprice = np.zeros(t_endl-t_beg)
    ldt_timestamps = Bollinger[0][s_sym].index
    for x in range(20, len(ldt_timestamps)-t_end+t_beg): # 3...len-1
        i = x - t_beg
        input = np.zeros((t_endl-t_beg)*len(Bollinger))
        for t in range(t_beg,t_endl): # -3 -2 -1 0 1 2 3
            t1 = t-t_beg
            f_symprice[t1] = quotes.ix[ldt_timestamps[i + t]] #   -3 -2 -1 0 1 2 3   -3  
            for j in range(0, len(Bollinger)):
                input[t1+j*(t_endl-t_beg)] = Bollinger[j][s_sym].ix[ldt_timestamps[i + t]]
        if (np.isnan(sum(input))):
            continue
        input = input.reshape(1,input.size)
#       print input.reshape(input.size,1).shape
        y = bp.bpExec(input)
        if y == 2:
#            if quotes.ix[ldt_timestamps[i + t]]
            ldt_ts = ldt_timestamps[x]
            results = [str(ldt_ts.year),str(ldt_ts.month),str(ldt_ts.day),str(s_sym),"Buy",str(100),""]
            writer.writerow(results)
            i1 = i+3
            if i1 >= len(ldt_timestamps):
                i1 = len(ldt_timestamps)-1
            ldt_ts = ldt_timestamps[i1]
            results = [str(ldt_ts.year),str(ldt_ts.month),str(ldt_ts.day),str(s_sym),"Sell",str(100),""]
            writer.writerow(results)
 
def generateOrders(d_data):
    df_close = d_data['close']
    ls_symbols=list(d_data['close'].columns)
    resultFile = open("../data/orders.csv",'wb')
    writer = csv.writer(resultFile, dialect='excel')

    Bollinger= initBollinger(d_data)

    for s_sym in ls_symbols:
        printClass(df_close[s_sym], Bollinger, s_sym, writer)
    return 

def initBollinger(data):
    BOLLINGER = 20
#    print "Calculating Bollinger Band Values..."
    means = pd.rolling_mean(data['close'],BOLLINGER,min_periods=BOLLINGER)
    rolling_std = pd.rolling_std(data['close'],BOLLINGER,min_periods=BOLLINGER)
    max_bol = means + rolling_std
    min_bol = means - rolling_std
    Bollinger = (data['close'] - means) / (rolling_std)
    Bollinger2 = Bollinger*Bollinger
    return [Bollinger, Bollinger2]

def classifyBollinger(f_symprice):
    ## 2 - Buy Class, 1 - Sell Class, 0 - Hold Class
    return (2 * (f_symprice[-1] > (f_symprice[0] * 1.04)) + 1 * (f_symprice[-1] < (f_symprice[0] *0.96)))
    
def classifyValues(s_sym, quotes, Bollinger):
    global rowCount, inputRows, outputRows
    t_beg = -3 
    t_end = 3
    t_endl = t_end+1
    f_symprice = np.zeros(t_endl-t_beg)
    ldt_timestamps = Bollinger[0][s_sym].index
    for x in range(20, len(ldt_timestamps)-t_end+t_beg): # 3...len-1
        i = x - t_beg
        input = np.zeros((t_endl-t_beg)*len(Bollinger))
        for t in range(t_beg,t_endl): # -3 -2 -1 0 1 2 3
            t1 = t-t_beg
            f_symprice[t1] = quotes.ix[ldt_timestamps[i + t]] #   -3 -2 -1 0 1 2 3   -3  
            for j in range(0, len(Bollinger)):
                input[t1+j*(t_endl-t_beg)] = Bollinger[j][s_sym].ix[ldt_timestamps[i + t]]
        if (np.isnan(sum(input))):
            continue
        if (rowCount == 0):
            inputRows = input
        else:
            inputRows = np.vstack((inputRows,input))
        outputval  = classifyBollinger(f_symprice)
        outputRows.append(outputval)
        rowCount = rowCount + 1

def processData(d_data, strategy):
    global inputRows, outputRows
    inputRows = []
    outputRows = []
    df_close = d_data['close']
    ls_symbols=list(d_data['close'].columns)

    stratValues= strategy(d_data)

    for s_sym in ls_symbols:
        print s_sym
        classifyValues(s_sym, df_close[s_sym], stratValues)
    inputRows = np.array(inputRows)
    outputRows = np.array(outputRows) 
    print inputRows.shape
    return inputRows, outputRows 

def getData(startDate, endDate, symbols, cache=1):
    ldt_timestamps = du.getNYSEdays(startDate, endDate, dt.timedelta(hours=16))
    if (cache  == 1):
        dataobj = da.DataAccess('Yahoo')#, cachestalltime=0)
    else:
        dataobj = da.DataAccess(('Yahoo'), cachestalltime=0)
    ls_keys = ['open', 'high', 'low', 'close', 'volume', 'actual_close']
    ldf_data = dataobj.get_data(ldt_timestamps, symbols, ls_keys)
    d_data = dict(zip(ls_keys, ldf_data))
    for s_key in ls_keys:
        d_data[s_key] = d_data[s_key].fillna(method = 'ffill')
        d_data[s_key] = d_data[s_key].fillna(method = 'bfill')
        d_data[s_key] = d_data[s_key].fillna(1.0)
    return d_data

def trainBpNet():
    dt_start = dt.datetime(2011, 1, 1)
    dt_end = dt.datetime(2011, 12, 31)
    ls_symbols = da.DataAccess('Yahoo').get_symbols_from_list('sp5002012')[0:100]
    d_data = getData(dt_start,dt_end,ls_symbols,1)
    X,y = processData(d_data)
    sio.savemat('../data/bollinger_inputs.mat', {'X':X, 'y':y} )
    bp.bpTrain()

def predictStock(stock, startDate, endDate):
    d_data = getData(startDate,endDate,[stock],1)
    generateOrders(d_data)
    showGraph(stock, d_data)

#import QSTK.qstktools.YahooDataPull	

import urllib2
import urllib
import os

def get_data(data_path, ls_symbols):

    # Create path if it doesn't exist
    if not (os.access(data_path, os.F_OK)):
        os.makedirs(data_path)

    # utils.clean_paths(data_path)   

    _now =dt.datetime.now();
    miss_ctr=0; #Counts how many symbols we could not get
    for symbol in ls_symbols:
        # Preserve original symbol since it might
        # get manipulated if it starts with a "$"
        symbol_name = symbol
        if symbol[0] == '$':
            symbol = '^' + symbol[1:]

        symbol_data=list()
        # print "Getting {0}".format(symbol)
        
        try:
            params= urllib.urlencode ({'a':0, 'b':1, 'c':2000, 'd':_now.month, 'e':_now.day, 'f':_now.year, 's': symbol})
            url = "http://ichart.finance.yahoo.com/table.csv?%s" % params
            url_get= urllib2.urlopen(url)
            
            header= url_get.readline()
            symbol_data.append (url_get.readline())
            while (len(symbol_data[-1]) > 0):
                symbol_data.append(url_get.readline())
            
            symbol_data.pop(-1) #The last element is going to be the string of length zero. We don't want to write that to file.
            #now writing data to file
            f= open (data_path + symbol_name + ".csv", 'w')
            
            #Writing the header
            f.write (header)
            
            while (len(symbol_data) > 0):
                f.write (symbol_data.pop(0))
             
            f.close();    
                        
        except urllib2.HTTPError:
            miss_ctr += 1
            print "Unable to fetch data for stock: {0} at {1}".format(symbol_name, url)
        except urllib2.URLError:
            miss_ctr += 1
            print "URL Error for stock: {0} at {1}".format(symbol_name, url)
            
    print "All done. Got {0} stocks. Could not get {1}".format(len(ls_symbols) - miss_ctr, miss_ctr)

def read_symbols(s_symbols_file):

    ls_symbols=[]
    file = open(s_symbols_file, 'r')
    for line in file.readlines():
        str_line = str(line)
        if str_line.strip(): 
            ls_symbols.append(str_line.strip())
    file.close()
    
    return ls_symbols  

if __name__ == '__main__':
#   start = time.clock()
#   trainBpNet()
#   print "Elapsed time :", (time.clock() - start)
    dt_start = dt.datetime.now()-dt.timedelta(720)#dt.datetime(2012, 1, 1)
    dt_end = dt.datetime.now()
    ls_symbols = read_symbols('../data/symbols.txt')
    get_data("C:\Python27\Lib\site-packages\QSTK\QSData\Yahoo\\", ls_symbols)
    for sym in ls_symbols:
	    predictStock(sym,dt_start,dt_end)
        
    
