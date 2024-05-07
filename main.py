#from ib_insync import *
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date,timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import ShuffleSplit,GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
import pickle as pkl
import os
from data_service import YahooData
from model_dnn import ModelDNN
from util_ib import send_mail

class Forecast:
    def __init__(self) -> None:
        pass
    
    def enter_price(self):
        eps = []
        for lst,margin in [(['SRLN',],.001), (['JEPI','DXJ','XMHQ','IWY','DFAU','GLDM','QYLD','JEPQ'],.01), ]:
            eps.append(self.model.data_service.prices[lst].iloc[-1] * (1-margin))
        endf = pd.DataFrame(pd.concat(eps))
        
        send_mail(df=endf.loc[['JEPI','SRLN','QYLD','GLDM','DFAU','DXJ','IWY','XMHQ','JEPQ']])

    def run(self):
        self.model = ModelDNN.load_model()
        self.outdf = self.model.predict()

        maildf = self.outdf.tail(21)
        print(maildf)
        send_mail(df=maildf)

        self.enter_price()
        
if __name__ == '__main__':
    f = Forecast()
    f.run()
    