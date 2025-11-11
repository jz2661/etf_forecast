import logging,os
import pandas as pd
import numpy as np
from datetime import datetime,date
import pytz
import asyncio
from collections import defaultdict

from email.mime.text import MIMEText

# Import smtplib for the actual sending function
import smtplib
import mimetypes

# Here are the email package modules we'll need
from email.message import EmailMessage

#__all__ = ['expand_data','black','rank','remove_seen','send_mail','today_str','tz_now','tz_min']

SUBMODEL_PATH = 'models'
PCA_FEATURES_LATEST = 'pca_features_latest.parquet'
ETF_TARGETS = ['USO','SVXY','GLDM','IWY','XMHQ','QQQ','FXI','SPLG','JEPI','HYG','VTIP','BOXX',
               'SVOL','SQQQ','EFAV','UVXY','VPL','SMH',]
# 'IBIT',

MODEL_VERSION = defaultdict(lambda: '20251104')
# for t in ['USO','XMHQ']:
#     MODEL_VERSION[t] = '20250814'
# for t in ['SVOL','BOXX','JEPI','IBIT']:
#     MODEL_VERSION[t] = '20250618'

ETF_TARGETS_DNN_MODELS = {t: f"dnn_{t}_{MODEL_VERSION[t]}.keras" for t in ETF_TARGETS}
MODEL_START_DATE = '2024-01-01'

async def run_periodically(interval, periodic_function):
    while True:
        periodic_function()
        try:
            await asyncio.wait_for( asyncio.sleep(interval), interval+3)
        except:
            print("Loop got a timeout. Resuming..")

class WZException(Exception):
    pass

def today_str(tz='US/Eastern'):
    if tz:
        return datetime.now(pytz.timezone(tz)).date().isoformat()
    return date.today().isoformat()

def tz_now(tz='US/Eastern'):
    if tz:
        return datetime.now(pytz.timezone(tz))
    return datetime.now()

def tz_combine(time, tz='US/Eastern'):
    if type(time) == str:
        time = datetime.strptime(time, '%H:%M').time()
    dn = tz_now(tz)
    return datetime.combine(dn.date(), time)

def last_tindex(df, t):
    try:
        return df.index[df.index < datetime.combine(t, datetime.min.time())][-1]
    except:
        return df.index[df.index < t][-1]

def dtstr_add_tz(dtstr, tz):
    dt = datetime.fromisoformat(dtstr)
    if tz:
        return pytz.timezone(tz).localize(dt)
    return dt

def tz_min():
    return datetime.min.replace(tzinfo=pytz.timezone('US/Eastern'))

def expand_data(data):
    return (data.date, data.title, data.company, data.apply_link, data.link, len(data.description), data.place)

def black(df):
    bl = [x.upper() for x in ['C++','Java','Sale','contract','summer','compliance','graduate','middle', \
            'intern','junior','control','RELATION','legal','student','human','operations','marketing', \
            'governance','account',]]
    mask = df['title'].apply(lambda x: any(kw in x.upper() for kw in bl))

    bl = [x.upper() for x in ['Argyll Scott','HSBC','DBS','Manulife','Selby','EY','HKIP','Hang Seng','AXA', \
            'McKinley','AIA','deloitte','Societe','prudential','kpmg','junan','consulting','agency','acca', \
            'Standard Chartered','agoda','wells','Recruitment',]]
    mask |= df['company'].apply(lambda x: any(kw in x.upper() for kw in bl))

    try:
        mask |= df['place'].apply(lambda x: all(lc not in x.upper() for lc in ['HONG KONG','SINGAPORE']))
    except:
        pass

    return df.drop(df.index[mask])

def rank(df):
    return df.sort_values('des',ascending=False).drop_duplicates(subset=['title'])

def remove_seen(df):
    try:
        cachedf = pd.read_excel('res.xlsx',index_col=0)
        cachedf['ap'] = cachedf['ap'].replace(np.nan, '')
        
        comb = pd.concat([df,cachedf]).drop_duplicates()
        comb.to_excel(f'res_{datetime.now().isoformat()[:-13]}.xlsx')
        comb.to_excel('res.xlsx')

        mask = df['title'].apply(lambda x: x not in cachedf['title'].values)
        return df[mask]
    except:
        raise Exception("Remove seen failed.")
        return df

def send_mail(files=[], df=None, subject='ib_forecast'):
    # Create the container email message.
    msg = EmailMessage()
    msg['Subject'] = subject
    # me == the sender's email address
    # family = the list of all recipients' email addresses
    msg['From'] = 'zjzzjz2010@gmail.com'
    msg['To'] = ['zjzzjz2010@gmail.com','wangycthu@gmail.com']
    msg.preamble = 'You will not see this in a MIME-aware mail reader.\n'
    
    if not df is None:
        html = """\
        <html>
          <head></head>
          <body>
            {0}
          </body>
        </html>
        """.format(df.round(3).to_html())
        
        msg.set_content(html, subtype='html')

    # Open the files in binary mode.  Use imghdr to figure out the
    # MIME subtype for each specific image.
    for file in files:
        with open(file, 'rb') as fp:
            data = fp.read()
        ctype, encoding = mimetypes.guess_type(file)
        maintype, subtype = ctype.split('/', 1)
        msg.add_attachment(data,
                               maintype=maintype,
                               subtype=subtype,
                               filename=file)        

    # Send the email via our own SMTP server.
    with smtplib.SMTP(host="smtp.gmail.com", port="587") as smtp:  # 設定SMTP伺服器
        smtp.ehlo()  # 驗證SMTP伺服器
        smtp.starttls()  # 建立加密傳輸
        smtp.login("zjzzjz2010@gmail.com", "yrdxebidsnczndvq")  # 登入寄件者gmail
        smtp.send_message(msg)  # 寄送郵件
        print("Complete!")

INPUT_TICKERS = ['ACWI',
 'ACWV',
 'ACWX',
 'AGG',
 'AMLP',
 'ARKK',
 'AVDE',
 'AVDV',
 'AVEM',
 'AVUS',
 'AVUV',
 'BBAX',
 'BBCA',
 'BBEU',
 'BBJP',
 'BIL',
 'BIV',
 'BKLN',
 'BLV',
 'BND',
 'BNDX',
 'BSV',
 'CALF',
 'CGDV',
 'CGGR',
 'CIBR',
 'COWZ',
 'DBEF',
 'DFAC',
 'DFAI',
 'DFAS',
 'DFAT',
 'DFAU',
 'DFAX',
 'DFCF',
 'DFIC',
 'DFIV',
 'DFUS',
 'DFUV',
 'DGRO',
 'DGRW',
 'DIA',
 'DSI',
 'DUHP',
 'DVY',
 'DXJ',
 'DYNF',
 'EEM',
 'EEMV',
 'EFA',
 'EFAV',
 'EFG',
 'EFV',
 'EMB',
 'EMXC',
 'ESGD',
 'ESGU',
 'ESGV',
 'EWJ',
 'EWT',
 'EWY',
 'EWZ',
 'EZU',
 'FBND',
 'FBTC',
 'FDN',
 'FIXD',
 'FLOT',
 'FNDA',
 'FNDE',
 'FNDF',
 'FNDX',
 'FNGU',
 'FPE',
 'FTCS',
 'FTEC',
 'FTSM',
 'FVD',
 'FXI',
 'GBIL',
 'GBTC',
 'GDX',
 'GDXJ',
 'GLD',
 'GLDM',
 'GOVT',
 'GSLC',
 'GUNR',
 'HDV',
 'HEFA',
 'HYG',
 'IAGG',
 'IAU',
 'IBB',
 'IBIT',
 'ICSH',
 'IDEV',
 'IEF',
 'IEFA',
 'IEI',
 'IEMG',
 'IEUR',
 'IGIB',
 'IGM',
 'IGSB',
 'IGV',
 'IHI',
 'IJH',
 'IJJ',
 'IJK',
 'IJR',
 'IJS',
 'IJT',
 'INDA',
 'IOO',
 'IQLT',
 'ITA',
 'ITOT',
 'IUSB',
 'IUSG',
 'IUSV',
 'IVE',
 'IVV',
 'IVW',
 'IWB',
 'IWD',
 'IWF',
 'IWM',
 'IWN',
 'IWO',
 'IWP',
 'IWR',
 'IWS',
 'IWV',
 'IWY',
 'IXN',
 'IXUS',
 'IYW',
 'JAAA',
 'JEPI',
 'JEPQ',
 'JIRE',
 'JNK',
 'JPST',
 'KWEB',
 'LQD',
 'MBB',
 'MCHI',
 'MDY',
 'MGC',
 'MGK',
 'MGV',
 'MINT',
 'MOAT',
 'MTUM',
 'MUB',
 'NOBL',
 'OEF',
 'OMFL',
 'ONEQ',
 'PAVE',
 'PBUS',
 'PDBC',
 'PFF',
 'PGX',
 'PHYS',
 'PRF',
 'PULS',
 'QLD',
 'QQQ',
 'QQQM',
 'QUAL',
 'QYLD',
 'RDVY',
 'RSP',
 'SCHA',
 'SCHB',
 'SCHD',
 'SCHE',
 'SCHF',
 'SCHG',
 'SCHH',
 'SCHI',
 'SCHM',
 'SCHO',
 'SCHP',
 'SCHR',
 'SCHV',
 'SCHX',
 'SCHZ',
 'SCZ',
 'SDY',
 'SGOV',
 'SHV',
 'SHY',
 'SHYG',
 'SJNK',
 'SLV',
 'SMH',
 'SOXL',
 'SOXS',
 'SOXX',
 'SPAB',
 'SPDW',
 'SPEM',
 'SPGP',
 'SPHQ',
 'SPIB',
 'SPLV',
 'SPMB',
 'SPMD',
 'SPSB',
 'SPSM',
 'SPTI',
 'SPTL',
 'SPTM',
 'SPTS',
 'SPY',
 'SPYD',
 'SPYG',
 'SPYM',
 'SPYV',
 'SRLN',
 'SSO',
 'STIP',
 'SUB',
 'TFLO',
 'TIP',
 'TLH',
 'TLT',
 'TMF',
 'TQQQ',
 'USFR',
 'USHY',
 'USIG',
 'USMV',
 'VB',
 'VBK',
 'VBR',
 'VCIT',
 'VCLT',
 'VCR',
 'VCSH',
 'VDC',
 'VDE',
 'VEA',
 'VEU',
 'VFH',
 'VGIT',
 'VGK',
 'VGLT',
 'VGSH',
 'VGT',
 'VHT',
 'VIG',
 'VIGI',
 'VIS',
 'VLUE',
 'VMBS',
 'VNQ',
 'VO',
 'VOE',
 'VONE',
 'VONG',
 'VONV',
 'VOO',
 'VOOG',
 'VOOV',
 'VOT',
 'VPL',
 'VPU',
 'VSS',
 'VT',
 'VTEB',
 'VTI',
 'VTIP',
 'VTV',
 'VTWO',
 'VUG',
 'VV',
 'VWO',
 'VWOB',
 'VXF',
 'VXUS',
 'VYM',
 'VYMI',
 'XBI',
 'XLB',
 'XLC',
 'XLE',
 'XLF',
 'XLI',
 'XLK',
 'XLP',
 'XLRE',
 'XLU',
 'XLV',
 'XLY',
 'XMHQ',
 'XOP']