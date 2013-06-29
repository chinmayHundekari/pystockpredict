import PyStockPredict
import datetime

def main():
    stock = ["MSFT"]
    strat = PyStockPredict.bollinger    # Strat can be external to project
    #train(strat)   # training not required for internal strategies
    begDate = datetime.date(2012,6,1)
    endDate = datetime.date(2013,6,30)
    dispChart(stock,strat,begDate,endDate)

if __name__ == '__main__':
    main()

