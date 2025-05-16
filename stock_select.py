import numpy as np
import pandas as pd

ps = f.model.data_service.prices
ret = ps.pct_change()[-252:]
rety = ret.mean()*252

retneg = ret.copy()
retneg[retneg>0] = 0

retstd = retneg.std()*np.sqrt(252)

for i in range(10):
    rsk = i*0.01+0.02

    lowrisk = pd.DataFrame(retstd[retstd < rsk])
    rety.name = 'rety'
    # Merge lowrisk with rety, keeping only tickers that are in lowrisk
    lowret = lowrisk.merge(rety, left_index=True, right_index=True, how='left')

    # Sort by return
    lowret = lowret.sort_values(by='rety', ascending=False)

    print(rsk)
    print(lowret.head(2))
