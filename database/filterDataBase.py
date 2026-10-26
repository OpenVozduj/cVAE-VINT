import numpy as np
import pandas as pd
from scipy.stats import spearmanr

dataDir = 'OpenVINT6_DB_2p_V2/'
cases = np.load(dataDir+'cases.npy')

PG = np.empty([0,15])
WTG = np.empty([0,4])
VG = np.array([])
Treq = np.array([])

#%%
for t in range(700):
    pg = np.load(dataDir+'Pg_test_'+str(t)+'.npy')
    wtg = np.load(dataDir+'WTg_test_'+str(t)+'.npy') 
    for i in range(20):
        if wtg[i,2] >= 0.60 and wtg[i,1] >= cases[t,1] and wtg[i,1] <= cases[t,1]*1.5:
            PG = np.vstack((PG, pg[i]))
            WTG = np.vstack((WTG, wtg[i]))
            VG = np.append(VG, cases[t,0])
            Treq = np.append(Treq, cases[t,1])
            
Data = np.concatenate((PG, WTG), axis=1)
Data = np.column_stack((Data, VG))
# Data = np.column_stack((Data, Treq))

np.save('database2b.npy', Data)

param = ['c_d_r', 
         'c_d_m',
         'c_d_t',
         'r_cdm',
         'alpha_r',
         'alpha_m',
         'alpha_t',
         'r_alpham',
         'yt_r',
         'yt_m',
         'yt_t',
         'r_yt',
         'n_min',
         'B',
         'd',
         'W',
         'T',
         'eta_d',
         'ets_s',
         'V_inf']

df = pd.DataFrame(Data, columns=param)

matrix_Corr = df.corr(method='spearman')

cols = df.columns
n = len(cols)

pvals = pd.DataFrame(np.zeros((n,n)), index=cols, columns=cols)

for i in range(n):
    for j in range(i+1, n):
        _, p = spearmanr(df[cols[i]], df[cols[j]])
        pvals.iloc[i,j] = p
        pvals.iloc[j,i] = p
        
np.fill_diagonal(pvals.values, np.nan)