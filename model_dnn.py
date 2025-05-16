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

SUBMODEL_PATH = 'models'

# Define a model-building function
def build_model(hp):    
    dropout = 0.2
    l1_loss = hp.Choice('l1_loss', values=[1e-2,1e-3,1e-5,])
    activation = hp.Choice('activation', values=['relu','tanh'])
    learning_rate = hp.Choice('learning_rate', values=[1e-2,1e-3,1e-4,])
    X_noise = 0
    
    # Define the model architecture
    model = Sequential([
        GaussianNoise(X_noise),
        Dense(512, activation=activation, input_shape=(1200,)),
        Dropout(dropout),  # Dropout layer to prevent overfitting
        BatchNormalization(),  # Batch normalization layer
        Dense(512, activation=activation, input_shape=(1200,)),
        Dropout(dropout),  # Dropout layer to prevent overfitting
        Dense(256, activation=activation, kernel_regularizer=regularizers.l1(l1_loss)),
        Dropout(dropout),
        #BatchNormalization(),
        Dense(128, activation=activation, kernel_regularizer=regularizers.l1(l1_loss)),
        Dropout(dropout),
        BatchNormalization(),
        Dense(1, activation='sigmoid')  # Output layer with a single neuron and sigmoid activation function
    ])

    # Compile the model
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss=ModelDNN.weighted_binary_crossentropy, metrics=[ModelDNN.custom_pnl])
    return model

class ModelDNN:
    def __init__(self) -> None:
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

    # Define custom loss function weighted by abs(price return)
    @staticmethod
    def weighted_binary_crossentropy(y_true, y_pred):
        # around 1
        weights = tf.pow(tf.abs(y_true) * 100, 1)

        y_binary = tf.round((tf.sign(y_true)+1) / 2)

        y_std = 0.01
        y_prob = tf.sigmoid(y_true / y_std)

        #loss = -tf.reduce_mean((y_binary * tf.math.log(y_pred) + (1 - y_binary) * tf.math.log(1 - y_pred)) * weights)
        loss = -tf.reduce_mean((y_prob * tf.math.log(y_pred) + (1 - y_prob) * tf.math.log(1 - y_pred)) )

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
        posi = (y_pred * 2 - 1.)
        pnlx = y_true * posi
        gamma = 0.1
        epsv = 0.0001
        # Compare binary predictions with true labels
        pnl = (tf.reduce_mean(pnlx) / (tf.math.reduce_std(pnlx)+epsv) ) * np.sqrt(52)
        
        return pnl

    def tune_train_epochs(self, epochs=10):
        # Instantiate the tuner
        print(f"Model Tuning:")

        tuner = RandomSearch(
            build_model,
            objective=Objective("val_custom_pnl", direction="max"),
            max_trials=20,  # Number of hyperparameter combinations to try
            executions_per_trial=2,  # Number of models to train per trial
            directory='my_tuning',  # Directory to store the tuning results
            project_name='my_tuning_project')  # Name of the tuning project

        # Start the hyperparameter search
        tuner.search(self.X_train, self.y_train, epochs=epochs, validation_data=(self.X_test, self.y_test))

        # Get the best hyperparameters
        self.best_hp = tuner.get_best_hyperparameters(num_trials=2)[0]
        print()
        print(self.best_hp.values)

        # Build the best model
        self.best_model = tuner.hypermodel.build(self.best_hp)

        # Train the best model
        history = self.best_model.fit(self.X_train, self.y_train, epochs=epochs, validation_data=(self.X_test, self.y_test))
        self.plot_history(history)
        
        self.train_history = history
        return history
    
    def specify_model(self, warm=True):
        if warm:
            self.model = self.load_model().model
            return

        # Define the model architecture
        self.model = Sequential([
            GaussianNoise(self.X_noise),
            Dense(512, activation=self.activation),
            Dropout(self.dropout),  # Dropout layer to prevent overfitting
            BatchNormalization(),  # Batch normalization layer

            Dense(512, activation=self.activation),
            Dropout(self.dropout),  # Dropout layer to prevent overfitting
            BatchNormalization(),
            
            Dense(256, activation=self.activation, kernel_regularizer=regularizers.l1(self.l1_loss)),
            Dropout(self.dropout),
            BatchNormalization(),
            
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
        
    def train_model(self, warm_model=True, epochs=10):
        # valid rows
        valids = (self.X.notnull().sum(axis=1) / self.X.shape[1]) > 0.8
        X = self.X[valids].fillna(0)

        # Split the data into training and testing sets
        self.X_train, self.X_test, self.y_train,self.y_test = train_test_split(X.drop(['y'], axis=1).values, X['y'].values, test_size=0.2, shuffle=True)
        # check training data
        if 1:
            print(f"Training y: {self.y_train}")
            print(f"Test y: {self.y_test}")
            print()

        self.data_augmentation()

        self.specify_model(warm=warm_model)
        
        self.train_epochs(epochs)
        #self.tune_train_epochs(10)

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
    def load_model(cls, filename=os.path.join(SUBMODEL_PATH, "dnn_20240422.keras")):
        this = ModelDNN()
        customs = {
            'weighted_binary_crossentropy': cls.weighted_binary_crossentropy,
            'custom_pnl': cls.custom_pnl,
            }
        this.model = tf.keras.models.load_model(filename, custom_objects=customs)
        return this
    
    def train(self, dump=True, cache_data=True, warm_model=True, epochs=10):
        self.data_service = YahooData(start_date = '2023-01-01')
        self.data_service.load_data(cache=cache_data)
        self.prepare_data(self.data_service.prices)
        self.train_model(warm_model=warm_model, epochs=epochs)
        if dump:
            self.dump_model()

    def predict(self):
        self.data_service = YahooData(start_date = '2023-01-01')
        self.data_service.load_data(cache=False)
        self.prepare_data(self.data_service.prices)
        return self.model_predict()

if __name__ == '__main__':
    if 1:
        # train a new model
        m = ModelDNN()
        # m.train(dump=True, cache_data=False, warm_model=True, epochs=10)
        m.train(dump=True, cache_data=True, warm_model=True, epochs=20)

    if 0:
        # predict
        m = ModelDNN.load_model()
        pp = m.model_predict()
