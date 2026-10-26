import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle

db2b = np.load('database2b.npy')
db3b = np.load('database3b.npy')
db4b = np.load('database4b.npy')

dbG = np.concatenate((db2b, db3b, db4b), axis=0)

names_columns = ['c_d_r', 
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
                 'eta_s',
                 'V_inf']

df = pd.DataFrame(dbG, columns=names_columns)

#%%
custom_percentiles = [0.05, 0.25, 0.50, 0.75, 0.95]

desc_stats = df.describe(percentiles=custom_percentiles).round(4)
desc_stats.loc['skewness'] = df.skew()
desc_stats.loc['kurtosis'] = df.kurtosis()

#%%
df['n_min_log'] = np.log10(df['n_min'])
ops_heavy_tail = ['n_min_log', 'W']

geom_bounded = ['c_d_r', 
                'c_d_m', 
                'c_d_t', 
                'r_cdm', 
                'r_alpham', 
                'r_yt',
                'alpha_r', 
                'alpha_m', 
                'alpha_t', 
                'yt_r', 
                'yt_m', 
                'yt_t',
                'eta_d', 
                'eta_s']

cond_continuous = ['d', 'T', 'V_inf']

df = pd.get_dummies(df, columns=['B'], prefix='B', drop_first=False)
cat_columns = [col for col in df.columns if col.startswith('B_')]

scaler_Y_cont = MinMaxScaler(feature_range=(0, 1))
scaler_Y_cont.fit(df[cond_continuous])

scaler_X_bounded = MinMaxScaler(feature_range=(0, 1))
scaler_X_bounded.fit(df[geom_bounded])

scaler_X_ops = StandardScaler()
scaler_X_ops.fit(df[ops_heavy_tail])

X_bounded_scaled = scaler_X_bounded.transform(df[geom_bounded])
X_ops_scaled = scaler_X_ops.transform(df[ops_heavy_tail])

X_scaled = np.hstack((X_bounded_scaled, X_ops_scaled))

Y_cont_scaled = scaler_Y_cont.transform(df[cond_continuous])
Y_cat = df[cat_columns].values

Y_scaled = np.hstack((Y_cont_scaled, Y_cat))

scalers = {
    'X_bounded': scaler_X_bounded,
    'X_ops': scaler_X_ops,
    'Y_cont': scaler_Y_cont,  
    'cat_columns': cat_columns,
    'geom_bounded': geom_bounded,
    'ops_heavy_tail': ops_heavy_tail,
    'cond_continuous': cond_continuous
}

with open('scalers_cvae_hybrid.pkl', 'wb') as f:
    pickle.dump(scalers, f)

np.save('X_scaled.npy', X_scaled)
np.save('Y_scaled.npy', Y_scaled)

#%%
import seaborn as sns
import matplotlib.pyplot as plt

# Configuración de estilo para publicación
sns.set_theme(style="whitegrid", context="paper", font="sans-serif")
plt.rcParams['figure.dpi'] = 200
plt.rcParams['savefig.dpi'] = 200
plt.rcParams['font.size'] = 10

# --- A. HISTOGRAMAS DE DISTRIBUCIÓN (Densidad) ---
fig, axes = plt.subplots(2, 1, figsize=(6, 4))
axes = axes.flatten()

# 1. n_min original (para mostrar la asimetría)
sns.histplot(df['n_min'], kde=True, color='darkred', ax=axes[0], bins=50)
axes[0].set_title('RPM ($n_{min}$) - Original data')
axes[0].set_xlabel('RPM')

# 2. n_min_log (para mostrar la corrección)
sns.histplot(df['n_min_log'], kde=True, color='darkgreen', ax=axes[1], bins=50)
axes[1].set_title('RPM ($n_{min}$) - With logarithmic transformation log$_{10}$')
axes[1].set_xlabel('log$_{10}$(RPM)')

# # 3. Eficiencia Dinámica
# sns.histplot(df['eta_d'], kde=True, color='royalblue', ax=axes[2], bins=50)
# axes[2].set_title('Eficiencia Dinámica ($\eta_d$)')
# axes[2].set_xlabel('$\eta_d$')

# # 4. Eficiencia Estática
# sns.histplot(df['ets_s'], kde=True, color='darkorange', ax=axes[3], bins=50)
# axes[3].set_title('Eficiencia Estática ($\eta_s$)')
# axes[3].set_xlabel('$\eta_s$')

# # 5. Cuerda media (Ejemplo geométrico)
# sns.histplot(df['c_d_m'], kde=True, color='purple', ax=axes[4], bins=50)
# axes[4].set_title('Relación Cuerda/Diámetro Media')
# axes[4].set_xlabel('$(c/d)_m$')

# # 6. Espesor raíz (Ejemplo geométrico)
# sns.histplot(df['yt_r'], kde=True, color='teal', ax=axes[5], bins=50)
# axes[5].set_title('Espesor Relativo en Raíz')
# axes[5].set_xlabel('$y_{t,r}$')

plt.tight_layout()
plt.savefig('figura_1_distribuciones_variables.png', bbox_inches='tight')
print("Figura 1 (Distribuciones) guardada.")
plt.show()

#%%
corr_vars = ['d', 'T', 'V_inf', 'n_min_log', 'W', 'eta_d', 'eta_s']
corr_matrix = df[corr_vars].corr(method='pearson')

plt.figure(figsize=(5, 5))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
            fmt=".2f", linewidths=0.5, cbar_kws={'shrink': 0.8}, 
            annot_kws={"size": 8})
plt.tight_layout()
plt.savefig('matriz_correlacion_3.png', bbox_inches='tight')
plt.show()
