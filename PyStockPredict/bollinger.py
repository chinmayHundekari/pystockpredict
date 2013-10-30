class bollinger:

    def bollinger(data):
        BOLLINGER = 20
    #    print "Calculating Bollinger Band Values..."
        means = pd.rolling_mean(data['close'],BOLLINGER,min_periods=BOLLINGER)
        rolling_std = pd.rolling_std(data['close'],BOLLINGER,min_periods=BOLLINGER)
        max_bol = means + rolling_std
        min_bol = means - rolling_std
        Bollinger = (data['close'] - means) / (rolling_std)
        Bollinger2 = Bollinger*Bollinger
        return [Bollinger, Bollinger2]

    def classify(price):
        ## 2 - Buy Class, 1 - Sell Class, 0 - Hold Class
        return (2 * (price[-1] > (price[0] * 1.04)) + 1 * (price[-1] < (price[0] *0.96)))
