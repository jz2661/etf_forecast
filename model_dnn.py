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

SUBMODEL_PATH = 'models'

# Define a model-building function
def build_model(hp):    
    hp.Choice('activation', values=['relu','tanh'])
    # Define the model architecture
    self.model = Sequential([
        GaussianNoise(hp.Choice('x_noise', values=[1e-2, 1e-3, 0e-4])),
        Dense(512, activation=hp.Choice('activation', values=['relu','tanh']), input_shape=(1200,)),
        Dropout(self.dropout),  # Dropout layer to prevent overfitting
        BatchNormalization(),  # Batch normalization layer
        Dense(512, activation=self.activation, kernel_regularizer=regularizers.l1(self.l1_loss)),
        Dropout(self.dropout),
        #BatchNormalization(),
        Dense(256, activation=self.activation, kernel_regularizer=regularizers.l1(self.l1_loss)),
        Dropout(self.dropout),
        #BatchNormalization(),
        Dense(128, activation=self.activation, kernel_regularizer=regularizers.l1(self.l1_loss)),
        Dropout(self.dropout),
        BatchNormalization(),
        Dense(1, activation='sigmoid')  # Output layer with a single neuron and sigmoid activation function
    ])

    # Compile the model
    opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
    self.model.compile(optimizer=opt, loss=self.weighted_binary_crossentropy, metrics=[self.custom_pnl])
    return model

class ModelDNN:
    def __init__(self) -> None:
        self.lags = [1,5,21,126]
        if 0:
            self.dropout = 0.2
            self.l1_loss = 1e-3
            self.activation = 'relu'
            self.learning_rate = 1e-3
            # the higher, less over-fitting but slower learning
            self.augmentation_noise_multiple = 1.
            self.X_noise = 0.001
            self.augmentation_records = 1e4
        if 1:
            self.dropout = 0.2
            self.l1_loss = 1e-3
            self.activation = 'relu'
            self.learning_rate = 1e-3
            # the higher, less over-fitting but slower learning
            self.augmentation_noise_multiple = 1.
            self.X_noise = 0.001
            self.augmentation_records = 1e4

    # Define custom loss function weighted by abs(price return)
    @staticmethod
    def weighted_binary_crossentropy(y_true, y_pred):
        # around 1
        weights = tf.pow(tf.abs(y_true) * 100, 1)

        y_binary = tf.round((tf.sign(y_true)+1) / 2)

        loss = -tf.reduce_mean((y_binary * tf.math.log(y_pred) + (1 - y_binary) * tf.math.log(1 - y_pred)) )

        return loss

    # Define custom accuracy metric function
    @staticmethod
    def custom_pnl_loss(y_true, y_pred):
        return -ModelDNN.custom_pnl(y_true, y_pred)

    # Define custom accuracy metric function
    @staticmethod
    def custom_pnl(y_true, y_pred):
        # Convert predicted probabilities to binary predictions
        #y_pred_binary = tf.round(y_pred)
        posi = tf.round(y_pred * 2 - 1.)
        pnlx = y_true * posi
        gamma = 0.1
        # Compare binary predictions with true labels
        pnl = (tf.reduce_mean(pnlx) - gamma * tf.math.reduce_std(pnlx)) * 52
        
        return pnl

    def specify_model(self, warm=True):
        if warm:
            self.model = self.load_model().model
            return

        # Define the model architecture
        self.model = Sequential([
            GaussianNoise(self.X_noise),
            Dense(512, activation=self.activation, input_shape=(1200,)),
            Dropout(self.dropout),  # Dropout layer to prevent overfitting
            BatchNormalization(),  # Batch normalization layer
            Dense(512, activation=self.activation, kernel_regularizer=regularizers.l1(self.l1_loss)),
            Dropout(self.dropout),
            #BatchNormalization(),
            Dense(256, activation=self.activation, kernel_regularizer=regularizers.l1(self.l1_loss)),
            Dropout(self.dropout),
            #BatchNormalization(),
            Dense(128, activation=self.activation, kernel_regularizer=regularizers.l1(self.l1_loss)),
            Dropout(self.dropout),
            BatchNormalization(),
            Dense(1, activation='sigmoid')  # Output layer with a single neuron and sigmoid activation function
        ])

        # Compile the model
        opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.model.compile(optimizer=opt, loss=self.weighted_binary_crossentropy, metrics=[self.custom_pnl])

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

    def data_augmentation(self):
        times = int(self.augmentation_records) // self.X_train.shape[0] + 1

        rng = np.std(self.y_train) * self.augmentation_noise_multiple
        np.random.seed(42)
        random_array = np.random.normal(0, rng, size=(self.X_train.shape[0], times))

        #set_trace()
        self.X_train = np.vstack([self.X_train for _ in range(times)])
        self.y_train = np.concatenate([(self.y_train + random_array[:, i]) for i in range(times)])
        
    def train_model(self):
        # valid rows
        valids = (self.X.notnull().sum(axis=1) / self.X.shape[1]) > 0.8
        X = self.X[valids].fillna(0)

        # Split the data into training and testing sets
        self.X_train, self.X_test, self.y_train,self.y_test = train_test_split(X.drop(['y'], axis=1).values, X['y'].values, test_size=0.2, shuffle=False)
        self.data_augmentation()

        self.specify_model(warm=False)
        self.train_epochs(50)
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
    def load_model(cls, filename=os.path.join(SUBMODEL_PATH, "dnn_20240416.keras")):
        this = ModelDNN()
        customs = {
            'weighted_binary_crossentropy': cls.weighted_binary_crossentropy,
            'custom_pnl': cls.custom_pnl,
            }
        this.model = tf.keras.models.load_model(filename, custom_objects=customs)
        return this
    
    def train(self, dump=True):
        self.data_service = YahooData()
        self.data_service.load_data(cache=True)
        self.prepare_data(self.data_service.prices)
        self.train_model()
        if dump:
            self.dump_model()

    def predict(self):
        self.data_service = YahooData(start_date = '2023-01-01')
        self.data_service.load_data(cache=False)
        self.prepare_data(self.data_service.prices)
        return self.model_predict()

if __name__ == '__main__':
    if 1:
        m = ModelDNN()
        m.train(dump=False)
    
    if 0:
        # predict
        m = ModelDNN.load_model()
        pp = m.model_predict()
