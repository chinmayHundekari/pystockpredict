import PyStockPredict as psp
import datetime

def trainBpNet():
    dt_start = dt.datetime(2011, 1, 1)
    dt_end = dt.datetime(2011, 12, 31)
    ls_symbols = da.DataAccess('Yahoo').get_symbols_from_list('sp5002012')[0:100]
    d_data = psp.getData(dt_start,dt_end,ls_symbols,1)
    X,y = psp.processData(d_data)
    sio.savemat('../data/bollinger_inputs.mat', {'X':X, 'y':y} )
    bp.bpTrain()

def main():
    trainBpNet()
    
if __name__ == '__main__':
    main()

