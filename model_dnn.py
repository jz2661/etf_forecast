#from ib_insync import *
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date,timedelta,datetime
import pickle as pkl
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras import regularizers
from sklearn.model_selection import train_test_split
from data_service import YahooData
from pdb import set_trace
import matplotlib.pyplot as plt

SUBMODEL_PATH = 'models'

class ModelDNN:
    def __init__(self) -> None:
        self.lags = [1,5,21,126]
        self.dropout = 0.2
        self.l1_loss = 0.0

    # Define custom loss function weighted by abs(price return)
    @staticmethod
    def weighted_binary_crossentropy(y_true, y_pred):
        weights = tf.abs(y_true)
        y_binary = tf.sign(y_true)
        return tf.keras.losses.binary_crossentropy(y_binary, y_pred) * weights

    def specify_model(self, warm=True):
        if warm:
            self.model = self.load_model().model
            return

        # Define the model architecture
        self.model = Sequential([
            Dense(512, activation='relu', input_shape=(1200,)),
            Dropout(self.dropout),  # Dropout layer to prevent overfitting
            BatchNormalization(),  # Batch normalization layer
            Dense(256, activation='relu', kernel_regularizer=regularizers.l1(self.l1_loss)),
            Dropout(self.dropout),
            BatchNormalization(),
            Dense(128, activation='relu', kernel_regularizer=regularizers.l1(self.l1_loss)),
            Dropout(self.dropout),
            BatchNormalization(),
            Dense(1, activation='sigmoid')  # Output layer with a single neuron and sigmoid activation function
        ])

        # Compile the model
        self.model.compile(optimizer='adam', loss=self.weighted_binary_crossentropy, metrics=[])

    def train_epochs(self, epochs=10):
        # Train the model
        history = self.model.fit(self.X_train, self.y_train, epochs=epochs, validation_data=(self.X_test, self.y_test), verbose=1)
        self.plot_history(history)
        
        self.train_history = history
        return history

    @staticmethod
    def plot_history(history):
        # Plot loss history
        plt.figure()
        plt.plot(history.history['loss'], 'r-')
        plt.plot(history.history['val_loss'], 'b-')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        #plt.show()
        plt.savefig('train_loss.png', dpi=900)

    def test_model(self):
        # Evaluate the model
        loss = self.model.evaluate(self.X_test, self.y_test)
        print(f"Test Loss: {loss}")

    def train_model(self):
        # valid rows
        valids = (self.X.notnull().sum(axis=1) / self.X.shape[1]) > 0.8
        X = self.X[valids].fillna(0)

        # Split the data into training and testing sets
        self.X_train, self.X_test, self.y_train,self.y_test = train_test_split(X.drop(['y'], axis=1).values, X['y'].values, test_size=0.2, shuffle=False)

        self.specify_model(warm=True)
        self.train_epochs(200)
        self.test_model()

    def model_predict(self):
        # predict on self prepared data
        X = self.X.fillna(0)

        probs = [x[0] for x in self.model.predict(X.drop(['y'], axis=1).values)]
        outdf = self.X[['y']]
        outdf['y_prob'] = probs
        return outdf
    
    def prepare_data(self, prices):
        # all data, output contain na

        self.rets = {}
        for lag in self.lags:
            self.rets[lag] = prices.pct_change(lag)

        # 1200 columns 300 ticker x 4 returns
        self.X = pd.concat([self.rets[lag] for lag in self.lags], axis=1)
        self.X.index = pd.to_datetime(self.X.index)
        
        # predict weekly
        self.X['y'] = self.rets[5]['QQQ'].shift(-5)

    def dump_model(self):
        self.model.save(os.path.join(SUBMODEL_PATH, f"dnn_{datetime.today().strftime('%Y%m%d')}.keras"))

    @classmethod
    def load_model(cls, filename=os.path.join(SUBMODEL_PATH, "dnn_20240415.keras")):
        this = ModelDNN()
        this.model = tf.keras.models.load_model(filename, custom_objects={'weighted_binary_crossentropy': cls.weighted_binary_crossentropy})
        return this
    
    def train(self, dump=True):
        self.data_service = YahooData()
        self.prepare_data(self.data_service.prices)
        self.train_model()
        if dump:
            self.dump_model()

    def predict(self):
        self.data_service = YahooData(start_date = '2023-01-01')
        self.prepare_data(self.data_service.prices)
        return self.model_predict()

if __name__ == '__main__':
    if 0:
        m = ModelDNN()
        m.train()
    
    if 1:
        # predict
        m = ModelDNN.load_model()
        pp = m.model_predict()
