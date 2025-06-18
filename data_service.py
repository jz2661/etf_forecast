#from ib_insync import *
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date,timedelta
import pickle as pkl
import os
from pdb import set_trace
from util_ib import ETF_TARGETS

class YahooData:
    def __init__(self, start_date='2020-01-01', padding_days=4) -> None:
        self.start_date = start_date
        self.max_retries = 2
        self.delisted = 2

        # duplicate last rows
        self.tail_padding_days = padding_days

    def labels_batch(self, targets):
        end_offset = 0
        self.end_date = (date.today()-timedelta(end_offset)).isoformat()

        cache_file = f'label_batch_{self.end_date}.parquet'

        data = yf.download(targets,self.start_date,self.end_date,auto_adjust=False)
        # set_trace()

        retry_cnt = 0
        while (missing_col := sum(data.isnull().all())) > 0 and retry_cnt < self.max_retries:
            if 0:
                end_offset += 1
                self.end_date = (date.today()-timedelta(end_offset)).isoformat()
            pulled = yf.download(targets,self.start_date,self.end_date,auto_adjust=False)
            null_col = data.isnull().all()
            nc = null_col[null_col>0].index.values
            data[nc] = pulled[nc]

            retry_cnt += 1
            print(f"Failed to fetch prices for {missing_col} tickers.")
            #raise AttributeError(f"Failed to fetch prices for {missing_col} tickers.")

        self.yahoo_to_prices(data)

        self.prices.to_parquet(cache_file)

    def load_data(self, cache=True):
        end_offset = 0
        self.end_date = (date.today()-timedelta(end_offset)).isoformat()

        cache_file = f'etf_prices_{self.end_date}.pkl'
        if cache:
            try:
                self.prices = pd.read_pickle(cache_file)  
                return
            except:
                print(f"Cache data not available for {self.end_date}. Downloading...")

        sheet='usetf2.xlsx'
        etfs = pd.read_excel(sheet)

        pulled = []
        data = yf.download(list(etfs['Ticker']),self.start_date,self.end_date,auto_adjust=False)
        #set_trace()

        retry_cnt = 0
        while (missing_col := sum(data.isnull().all())) > self.delisted and retry_cnt < self.max_retries:
            if 0:
                end_offset += 1
                self.end_date = (date.today()-timedelta(end_offset)).isoformat()
            pulled = yf.download(list(etfs['Ticker']),self.start_date,self.end_date)
            null_col = data.isnull().all()
            nc = null_col[null_col>0].index.values
            data[nc] = pulled[nc]

            retry_cnt += 1
            print(f"Failed to fetch prices for {missing_col} tickers.")
            #raise AttributeError(f"Failed to fetch prices for {missing_col} tickers.")

        self.yahoo_to_prices(data)

        self.tail_padding()

        self.prices.to_pickle(cache_file)

    def yahoo_to_prices(self, data):
        # get adj close, after div tax
        #set(data.columns.get_level_values(0)) # {'Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume'}
        whtax = 0.15
        
        # set_trace()
        self.prices = data['Adj Close'] * (1-whtax) + data['Close'] * whtax
        return self.prices

    def tail_padding(self):
        last_row = self.prices.iloc[-1]  # Get the last row
        self.prices = pd.concat([self.prices, pd.DataFrame([last_row] * self.tail_padding_days)], ignore_index=False)  # Concatenate the last row four times

if __name__ == '__main__':
    if 0:
        f = YahooData(start_date = '2023-01-01')
        f.load_data(cache=False)

    if 1:
        f = YahooData(start_date = '2024-01-01')
        f.labels_batch(ETF_TARGETS)

