#from ib_insync import *
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date,timedelta
import pickle as pkl
import os

class YahooData:
    def __init__(self, start_date = '2015-01-01') -> None:
        self.start_date = start_date
        self.load_data()

    def load_data(self, cache=True):
        end_offset = 1
        self.end_date = (date.today()-timedelta(end_offset)).isoformat()

        cache_file = f'etf_prices_{self.end_date}.pkl'
        if cache:
            try:
                self.prices = pd.read_pickle(cache_file)  
                return
            except:
                print(f"Cache data not available for {self.end_date}. Downloading...")

        sheet='usetf.xlsx'
        etfs = pd.read_excel(sheet)

        pulled = []
        data = yf.download(list(etfs['Ticker']),self.start_date,self.end_date)
        while missing_col := sum(data.isnull().all()) > 0:
            if 0:
                end_offset += 1
                self.end_date = (date.today()-timedelta(end_offset)).isoformat()
            pulled = yf.download(list(etfs['Ticker']),self.start_date,self.end_date)
            null_col = data.isnull().all()
            nc = null_col[null_col>0].index.values
            data[nc] = pulled[nc]

            #raise AttributeError(f"Failed to fetch prices for {missing_col} tickers.")

        self.yahoo_to_prices(data)

        self.prices.to_pickle(cache_file)

    def yahoo_to_prices(self, data):
        # get adj close, after div tax
        #set(data.columns.get_level_values(0)) # {'Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume'}
        whtax = 0.
        self.prices = data['Adj Close'] * (1-whtax) + data['Close'] * whtax
        return self.prices

if __name__ == '__main__':
    f = YahooData()
    