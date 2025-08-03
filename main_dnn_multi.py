from dnn_multi import predict_multi
from util_ib import send_mail
import pandas as pd

def prob_to_dollar(p):
    pup, dup = .7, 100
    plow = .58
    return max(0, (p-plow)/(pup-plow) * dup)

def qqq_to_lever(pqqq):
    return max(1, (pqqq-.5)/0.1)

# from data_service labels 
# f.prices.pct_change().std()
vol_df = pd.read_csv('etf_vol.csv').drop_duplicates(subset=['Ticker'], keep='last').set_index('Ticker')['0']
def vol_scaler(ticker):
    return vol_df['QQQ'] / vol_df[ticker]

def run():
    df, last_row_sorted = predict_multi()

    maildf = last_row_sorted.to_frame()
    
    maildf.iloc[:, 0] = maildf.iloc[:, 0].round(3)
    
    maildf['$size'] = maildf.iloc[:, 0].apply(prob_to_dollar)
    # QQQ leverage
    maildf['$size'] *= qqq_to_lever(maildf.iloc[:, 0]['QQQ'])
    
    maildf['$vol_adj_size'] = maildf['$size'] * maildf.index.map(vol_scaler)
    
    for c in ['$size','$vol_adj_size']:
        maildf[c] = maildf[c].round().astype(int)

    print(maildf)
    send_mail(df=maildf, subject='DNN Multi Daily')

    return df, last_row_sorted
        
if __name__ == '__main__':
    df, last_row_sorted = run()
    