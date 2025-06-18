from dnn_multi import predict_multi
from util_ib import send_mail
import pandas as pd

def run():
    df, last_row_sorted = predict_multi()

    maildf = last_row_sorted.to_frame()
    print(maildf)
    send_mail(df=maildf, subject='DNN Multi Daily')

    return df, last_row_sorted
        
if __name__ == '__main__':
    df, last_row_sorted = run()
    