#from ib_insync import *
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date,timedelta,datetime
import pickle as pkl
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, GaussianNoise
from tensorflow.keras import regularizers
from sklearn.model_selection import train_test_split
from data_service import YahooData
from pdb import set_trace
import matplotlib.pyplot as plt
from kerastuner.tuners import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters
from keras_tuner import Objective
from model_dnn import ModelDNN
from util_ib import PCA_FEATURES_LATEST, ETF_TARGETS_DNN_MODELS, ETF_TARGETS

SUBMODEL_PATH = 'models'

def train_multi(label_file):
    for target in ETF_TARGETS:
        m = DNNTarget(target=target)
        m.train(label_file=label_file, dump=True, warm_model=False, epochs=100)        

def predict_multi():
    dfs = []
    for target in ETF_TARGETS:
        # predict
        m = DNNTarget(target=target)
        m.load_target_model()
        pp = m.predict()
        dfs.append(pp)
    df = pd.concat(dfs, axis=1)
    last_row_sorted = df.iloc[-1].sort_values(ascending=False)
    return df, last_row_sorted

class DNNTarget(ModelDNN):
    def __init__(self, target, prev_model=None) -> None:
        self.lags = [1,5,21,126]
        if 1:
            self.dropout = 0.1
            self.l1_loss = 1e-2
            self.activation = 'tanh'
            self.learning_rate = 1e-5
            # the higher, less over-fitting but slower learning
            self.augmentation_noise_multiple = 1.
            self.X_noise = 0.000
            self.augmentation_records = 1e4
        self.target = target
        self.prev_model = prev_model
        self.this_model = f"dnn_{target}_{datetime.today().strftime('%Y%m%d')}.keras"

    def load_target_model(self):
        filename = os.path.join(SUBMODEL_PATH, ETF_TARGETS_DNN_MODELS.get(self.target))
        customs = {
            'weighted_binary_crossentropy': self.weighted_binary_crossentropy,
            'custom_pnl': self.custom_pnl,
            }
        self.model = tf.keras.models.load_model(filename, custom_objects=customs)

    def specify_model(self, warm=False):
        if warm:
            self.load_target_model()
            return

        # Define the model architecture
        self.model = Sequential([
            GaussianNoise(self.X_noise),

            Dense(16, activation=self.activation),
            Dropout(self.dropout),  # Dropout layer to prevent overfitting
            BatchNormalization(),
            
            Dense(16, activation=self.activation, kernel_regularizer=regularizers.l1(self.l1_loss)),
            Dropout(self.dropout),
            BatchNormalization(),
            
            Dense(4, activation=self.activation, kernel_regularizer=regularizers.l1(self.l1_loss)),
            BatchNormalization(),
            Dense(1, activation='sigmoid')  # Output layer with a single neuron and sigmoid activation function
        ])

        # Compile the model
        opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.model.compile(optimizer=opt, loss=self.weighted_binary_crossentropy, metrics=[self.custom_pnl])
    
    def load_features(self):
        self.X = pd.read_parquet(PCA_FEATURES_LATEST)

    def get_returns(self, prices):
        self.rets = {}
        for lag in self.lags:
            self.rets[lag] = prices.pct_change(lag)

    def join_label(self, label_file):  
        prices = pd.read_parquet(label_file)

        self.get_returns(prices)
        # predict weekly
        self.X['y'] = self.rets[5][self.target].shift(-5)

    def dump_model(self):
        self.model.save(os.path.join(SUBMODEL_PATH, self.this_model))
    
    def train(self, label_file, dump=True, warm_model=True, epochs=10):
        self.load_features()
        self.join_label(label_file)

        # print(f"epochs train: {epochs}")
        self.train_model(warm_model=warm_model, epochs=epochs)
        if dump:
            self.dump_model()

    def model_predict(self):
        # predict on self prepared data
        X = self.X.fillna(0)

        probs = [x[0] for x in self.model.predict(X.drop(['y'], axis=1, errors='ignore').values)]
        X[self.target] = probs
        return X[[self.target]]
    
    def predict(self):
        self.load_features()

        return self.model_predict()

if __name__ == '__main__':
    # run data_service to refresh
    label_file='label_batch_2025-06-18.parquet'

    if 0:
        # train a new model
        m = DNNTarget(target='QQQ')
        m.train(label_file=label_file, dump=True, warm_model=False, epochs=100)
        # m.train(label_file=label_file, dump=True, warm_model=True, epochs=1)

    if 1:
        train_multi(label_file)

    if 0:
        # predict
        m = DNNTarget(target='QQQ')
        m.load_target_model()
        pp = m.predict()
        print(pp)
