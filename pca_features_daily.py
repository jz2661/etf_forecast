#from ib_insync import *
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date,timedelta,datetime
import os
import pickle as pkl
from sklearn.decomposition import SparsePCA
from util_ib import SUBMODEL_PATH, PCA_FEATURES_LATEST
from data_service import YahooData

class PCA_Features:
    model_name = 'pca_20250616.pkl'

    def __init__(self) -> None:
        self.lags = [1,5,21,126]
    
    def prepare_data(self, prices):
        # all data, output contain na

        self.rets = {}
        for lag in self.lags:
            self.rets[lag] = prices.pct_change(lag)

        # 1200 columns 300 ticker x 4 returns
        self.X = pd.concat([self.rets[lag] for lag in self.lags], axis=1)
        self.X.index = pd.to_datetime(self.X.index)
        self.X = self.X.fillna(0)

    def train_model(self):
        pca = SparsePCA(n_components=20, random_state=42)
        pca.fit(self.X)

        self.pca = pca

    def dump_model(self):
        filename = os.path.join(SUBMODEL_PATH, f"pca_{datetime.today().strftime('%Y%m%d')}.pkl")
        with open(filename, 'wb') as file:
            pkl.dump(self.pca, file)

    @classmethod
    def load_model(cls):
        filename = os.path.join(SUBMODEL_PATH, cls.model_name)
        with open(filename, 'rb') as file:
            loaded_pca = pkl.load(file)
        return loaded_pca

    def get_raw_data(self, cache_data):
        self.data_service = YahooData(start_date='2024-01-01', padding_days=0)
        self.data_service.load_data(cache=cache_data)

        self.prepare_data(self.data_service.prices)        

    def train(self, dump=True, cache_data=True):
        self.get_raw_data(cache_data=cache_data)

        self.train_model()
        
        if dump:
            self.dump_model()

    def model_predict(self):
        # predict on self prepared data
        self.pca = self.load_model()

        X_transformed = self.pca.transform(self.X)
        df_features = pd.DataFrame(X_transformed)
        df_features.index = self.X.index

        return df_features
    
    def predict(self):
        self.get_raw_data(cache_data=False)
        return self.model_predict()

def features_daily():
    m = PCA_Features()
    df = m.predict()
    df.to_parquet(PCA_FEATURES_LATEST)

if __name__ == '__main__':
    if 0:
        # train a new model
        m = PCA_Features()
        m.train(dump=True, cache_data=False)

    if 1:
        # predict
        features_daily()
