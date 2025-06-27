from dnn_multi import predict_multi
from util_ib import send_mail
import pandas as pd

def prob_to_dollar(p):
    pup, dup = .7, 100
    plow = .58
    return max(0, (p-plow)/(pup-plow) * dup)

def qqq_to_lever(pqqq):
    return max(1, (pqqq-.5)/0.1)

def run():
    df, last_row_sorted = predict_multi()

    maildf = last_row_sorted.to_frame()
    
    maildf.iloc[:, 0] = maildf.iloc[:, 0].round(3)
    
    maildf['$size'] = maildf.iloc[:, 0].apply(prob_to_dollar)
    # QQQ leverage
    maildf['$size'] *= qqq_to_lever(maildf.iloc[:, 0]['QQQ'])
    maildf['$size'] = maildf['$size'].round().astype(int)

    print(maildf)
    send_mail(df=maildf, subject='DNN Multi Daily')

    return df, last_row_sorted
        
if __name__ == '__main__':
    df, last_row_sorted = run()
    