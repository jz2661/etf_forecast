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
from model_dnn import ModelDNN, SUBMODEL_PATH
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

        self.model_new = ModelDNN.load_model(filename=os.path.join(SUBMODEL_PATH, "dnn_20241230.keras"))
        self.outdf_new = self.model_new.predict()

        maildf = self.outdf.tail(21)
        maildf_new = self.outdf_new.tail(21)
        maildf['y_prob_new'] = maildf_new['y_prob']
        maildf['y_prob_avg'] = (maildf['y_prob'] + maildf['y_prob_new'])/2
        print(maildf)
        send_mail(df=maildf)

        self.enter_price()
        
if __name__ == '__main__':
    f = Forecast()
    f.run()
    