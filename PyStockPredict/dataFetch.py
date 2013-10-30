import pandas as pd
import numpy as np
import scipy.io as sio
import math, copy, time, sys, csv
import QSTK.qstkutil.qsdateutil as du
import datetime as dt
import QSTK.qstkutil.DataAccess as da
import QSTK.qstkutil.tsutil as tsu
import QSTK.qstkstudy.EventProfiler as ep
import urllib2
import urllib
import os

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

def fetchNSE():
    dt_start = dt.datetime.now()-dt.timedelta(720)#dt.datetime(2012, 1, 1)
    dt_end = dt.datetime.now()
    ls_symbols = read_symbols('../data/symbols.txt')
    get_data("C:\Python27\Lib\site-packages\QSTK\QSData\Yahoo\\", ls_symbols)

if __name__ == '__main__':
    fetchNSE()