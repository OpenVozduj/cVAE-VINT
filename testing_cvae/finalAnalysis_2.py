import numpy as np
import json
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl

with open('scalers_cvae_hybrid.pkl', 'rb') as f:
    scalers = pickle.load(f)
C = np.load('generated_designs_multi_seed/seed_42/c_conditions.npy')
c_physical = scalers['Y_cont'].inverse_transform(C[0,:,:3])

X = np.load('generated_designs_multi_seed/seed_42/X_physical.npy')

performanceB2 = np.empty((0,9))
B = 2
for i in range(30):
    T_target_i = c_physical[i,1]
    case = 'validationBEMT_seed_42/case_'+str(i)+'/blades_'+str(B)
    data = np.load(case+'/compDataOper.npy')
    performanceB2 = np.vstack((performanceB2, data))
    
performanceB3 = np.empty((0,9))
B = 3
for i in range(30):
    T_target_i = c_physical[i,1]
    case = 'validationBEMT_seed_42/case_'+str(i)+'/blades_'+str(B)
    data = np.load(case+'/compDataOper.npy')
    performanceB3 = np.vstack((performanceB3, data))

performanceB4 = np.empty((0,9))
B = 4
for i in range(30):
    T_target_i = c_physical[i,1]
    case = 'validationBEMT_seed_42/case_'+str(i)+'/blades_'+str(B)
    data = np.load(case+'/compDataOper.npy')
    performanceB4 = np.vstack((performanceB4, data))
    
#%%
font = {'family' : 'Liberation Serif',
        'weight' : 'normal',
        'size'   : 10}
cm=1/2.54
mpl.rc('font', **font)
mpl.rc('axes', linewidth=1)
mpl.rc('lines', lw=1)

val_ref_eta = np.array([min(performanceB2[:,6]), max(performanceB2[:,6])])
fig = plt.figure(1, figsize=(36*cm, 12*cm))
ax = fig.add_subplot(131)
ax.scatter(performanceB2[:,6], performanceB2[:,7], s=4, color='blue', label='Data')
ax.plot(val_ref_eta, val_ref_eta, color='black', linestyle='-', 
        label='$\eta_{d,pred}=\eta_{d,calc}$')
ax.set_xlabel('$\eta_{d,pred}$')
ax.set_ylabel('$\eta_{d,calc}$')
ax.set_title('a)')
ax.set_ylim(bottom=0.60)
ax.grid()
ax.legend(loc='best')

val_ref_eta = np.array([min(performanceB3[:,6]), max(performanceB3[:,6])])
ax = fig.add_subplot(132)
ax.scatter(performanceB3[:,6], performanceB3[:,7], s=4, color='blue', label='Data')
ax.plot(val_ref_eta, val_ref_eta, color='black', linestyle='-', 
        label='$\eta_{d,pred}=\eta_{d,calc}$')
ax.set_xlabel('$\eta_{d,pred}$')
ax.set_ylabel('$\eta_{d,calc}$')
ax.set_title('b)')
ax.grid()
ax.legend(loc='best')

val_ref_eta = np.array([min(performanceB4[:,6]), max(performanceB4[:,6])])
ax = fig.add_subplot(133)
ax.scatter(performanceB4[:,6], performanceB4[:,7], s=4, color='blue', label='Data')
ax.plot(val_ref_eta, val_ref_eta, color='black', linestyle='-', 
        label='$\eta_{d,pred}=\eta_{d,calc}$')
ax.set_xlabel('$\eta_{d,pred}$')
ax.set_ylabel('$\eta_{d,calc}$')
ax.set_title('c)')
ax.grid()
ax.legend(loc='best')

fig.tight_layout()
plt.show()
plt.savefig('eta_predvseta_calc.png', dpi=600)

#%%
fig = plt.figure(2, figsize=(36*cm, 12*cm))
ax = fig.add_subplot(131)
ax.scatter(performanceB2[:,2], performanceB2[:,7], s=4, color='blue', label='Data')
ax.axhline(y=0.60, color='red', linestyle='--', linewidth=1.5, label='Minimum criterion (60%)')
ax.set_xlabel('$Error_T$ [%]')
ax.set_ylabel('$\eta_{d,calc}$')
ax.set_ylim(bottom=0.595)
ax.set_title('a)')
ax.grid()
ax.legend(loc='best')

ax = fig.add_subplot(132)
ax.scatter(performanceB3[:,2], performanceB3[:,7], s=4, color='blue', label='Data')
ax.axhline(y=0.60, color='red', linestyle='--', linewidth=1.5, label='Minimum criterion (60%)')
ax.set_xlabel('$Error_T$ [%]')
ax.set_ylabel('$\eta_{d,calc}$')
ax.set_title('b)')
ax.grid()
ax.legend(loc='best')

ax = fig.add_subplot(133)
ax.scatter(performanceB4[:,2], performanceB4[:,7], s=4, color='blue', label='Data')
ax.axhline(y=0.60, color='red', linestyle='--', linewidth=1.5, label='Minimum criterion (60%)')
ax.set_xlabel('$Error_T$ [%]')
ax.set_ylabel('$\eta_{d,calc}$')
ax.set_title('c)')
ax.grid()
ax.legend(loc='best')

fig.tight_layout()
plt.show()
plt.savefig('ErrorTvseta_calc.png', dpi=600)

#%%
eta2_Avg = np.array([])
eta3_Avg = np.array([])
eta4_Avg = np.array([])

eta2_Std = np.array([])
eta3_Std = np.array([])
eta4_Std = np.array([])

for i in range(30):
    blade2_avg = np.mean(performanceB2[50*i:50*(i+1),7])
    eta2_Avg = np.append(eta2_Avg, blade2_avg)
    blade3_avg = np.mean(performanceB3[50*i:50*(i+1),7])
    eta3_Avg = np.append(eta3_Avg, blade3_avg)
    blade4_avg = np.mean(performanceB4[50*i:50*(i+1),7])
    eta4_Avg = np.append(eta4_Avg, blade4_avg)
    
    blade2_std = np.std(performanceB2[50*i:50*(i+1),7])
    eta2_Std = np.append(eta2_Std, blade2_std)
    blade3_std = np.std(performanceB3[50*i:50*(i+1),7])
    eta3_Std = np.append(eta3_Std, blade3_std)
    blade4_std = np.std(performanceB4[50*i:50*(i+1),7])
    eta4_Std = np.append(eta4_Std, blade4_std)
    
eta_anal2 = np.column_stack((eta2_Avg, eta2_Std))
eta_anal3 = np.column_stack((eta3_Avg, eta3_Std))
eta_anal4 = np.column_stack((eta4_Avg, eta4_Std))

fig = plt.figure(3, figsize=(12*cm, 12*cm))
ax = fig.add_subplot(111)
ax.scatter(eta_anal2[:,0], eta_anal2[:,1], c='blue', marker='o', 
           s=60, alpha=0.7, edgecolors='black', linewidth=0.5,
           label='Propellers with 2 blades')
ax.scatter(eta_anal3[:,0], eta_anal3[:,1], c='green', marker='s', 
           s=60, alpha=0.7, edgecolors='black', linewidth=0.5,
           label='Propellers with 3 blades')
ax.scatter(eta_anal4[:,0], eta_anal4[:,1], c='red', marker='^', 
           s=60, alpha=0.7, edgecolors='black', linewidth=0.5,
           label='Propellers with 4 blades')
ax.set_xlabel('Calculated Average Efficiency')
ax.set_ylabel('Standard Deviation of Efficiency')
ax.grid()
ax.legend(loc='best')

plt.tight_layout()
plt.savefig('eficiencia_vs_varianza.png', dpi=600, bbox_inches='tight')
plt.show()

#%%
cvValB2 = np.empty((0,13))
B = 2
for i in range(30):
    case = 'validationBEMT_seed_42/case_'+str(i)+'/blades_'+str(B)
    dataOper = np.load(case+'/compDataOper.npy')
    dataGeo = np.load(case+'/geoData.npy')
    dataGeoVal = np.empty((0,13))
    for j in range(50):
        if dataOper[j,2] <= 5:
            dataGeoVal = np.vstack((dataGeoVal, dataGeo[j]))
    muGeo = np.mean(dataGeoVal, axis=0)
    sigmaGeo = np.std(dataGeoVal, axis=0)
    CVgeo = (sigmaGeo/muGeo) *100        
    cvValB2 = np.vstack((cvValB2, CVgeo))
    
cvValB3 = np.empty((0,13))
B = 3
for i in range(30):
    case = 'validationBEMT_seed_42/case_'+str(i)+'/blades_'+str(B)
    dataOper = np.load(case+'/compDataOper.npy')
    dataGeo = np.load(case+'/geoData.npy')
    dataGeoVal = np.empty((0,13))
    for j in range(50):
        if dataOper[j,2] <= 5:
            dataGeoVal = np.vstack((dataGeoVal, dataGeo[j]))
    muGeo = np.mean(dataGeoVal, axis=0)
    sigmaGeo = np.std(dataGeoVal, axis=0)
    CVgeo = (sigmaGeo/muGeo) *100        
    cvValB3 = np.vstack((cvValB3, CVgeo))
    
cvValB4 = np.empty((0,13))
B = 4
for i in range(30):
    case = 'validationBEMT_seed_42/case_'+str(i)+'/blades_'+str(B)
    dataOper = np.load(case+'/compDataOper.npy')
    dataGeo = np.load(case+'/geoData.npy')
    dataGeoVal = np.empty((0,13))
    for j in range(50):
        if dataOper[j,2] <= 5:
            dataGeoVal = np.vstack((dataGeoVal, dataGeo[j]))
    muGeo = np.mean(dataGeoVal, axis=0)
    sigmaGeo = np.std(dataGeoVal, axis=0)
    CVgeo = (sigmaGeo/muGeo) *100        
    cvValB4 = np.vstack((cvValB4, CVgeo))
    
#%%
import pandas as pd

font = {'family' : 'Liberation Serif',
        'weight' : 'normal',
        'size'   : 12}
cm=1/2.54
mpl.rc('font', **font)
mpl.rc('axes', linewidth=1)
mpl.rc('lines', lw=1)

param_groups = ['($c/d$)', '($c/d$)', '($c/d$)', '($r$)', '($r$)', '($r$)',
    '($\\alpha$)', '($\\alpha$)', '($\\alpha$)', '($y_{t,max}$)', '($y_{t,max}$)', 
    '($y_{t,max}$)', '($n_{min}$)']

df_2 = pd.DataFrame(cvValB2, columns=param_groups)
df_3 = pd.DataFrame(cvValB3, columns=param_groups)
df_4 = pd.DataFrame(cvValB4, columns=param_groups)

fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

data_to_plot = [df_2['($c/d$)'].values.flatten(), 
                df_2['($r$)'].values.flatten(),
                df_2['($\\alpha$)'].values.flatten(),
                df_2['($y_{t,max}$)'].values.flatten(),
                df_2['($n_{min}$)'].values.flatten()]

labels = ['chords', 'positions', 'AoA', 'airfoil_thickness ', '$n_{min}$']
bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True,
                boxprops=dict(facecolor='#2E86AB', alpha=0.7),
                medianprops=dict(color='red', linewidth=2))

ax.set_ylabel('CV [%]', fontweight='bold')
ax.grid()
plt.tight_layout()
plt.savefig('boxplot_diversidad_gruposB2.png', dpi=600, bbox_inches='tight')
plt.show()

fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

data_to_plot = [df_3['($c/d$)'].values.flatten(), 
                df_3['($r$)'].values.flatten(),
                df_3['($\\alpha$)'].values.flatten(),
                df_3['($y_{t,max}$)'].values.flatten(),
                df_3['($n_{min}$)'].values.flatten()]

labels = ['chords', 'positions', 'AoA', 'airfoil_thickness ', '$n_{min}$']
bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True,
                boxprops=dict(facecolor='#2E86AB', alpha=0.7),
                medianprops=dict(color='red', linewidth=2))

ax.set_ylabel('CV [%]', fontweight='bold')
ax.grid()
plt.tight_layout()
plt.savefig('boxplot_diversidad_gruposB3.png', dpi=600, bbox_inches='tight')
plt.show()

fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

data_to_plot = [df_4['($c/d$)'].values.flatten(), 
                df_4['($r$)'].values.flatten(),
                df_4['($\\alpha$)'].values.flatten(),
                df_4['($y_{t,max}$)'].values.flatten(),
                df_4['($n_{min}$)'].values.flatten()]

labels = ['chords', 'positions', 'AoA', 'airfoil_thickness ', '$n_{min}$']
bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True,
                boxprops=dict(facecolor='#2E86AB', alpha=0.7),
                medianprops=dict(color='red', linewidth=2))

ax.set_ylabel('CV [%]', fontweight='bold')
ax.grid()
plt.tight_layout()
plt.savefig('boxplot_diversidad_gruposB4.png', dpi=600, bbox_inches='tight')
plt.show()

#%%

import seaborn as sns

indiv_mean_2 = df_2.mean()
indiv_mean_3 = df_3.mean()
indiv_mean_4 = df_4.mean()

heatmap_data = pd.DataFrame({
    '2 Blades': indiv_mean_2.values,
    '3 Blades': indiv_mean_3.values,
    '4 Blades': indiv_mean_4.values
}, index=[f'Par {i+1}' for i in range(13)])

yticks_labels = ['$(c/d)_r$', '$(c/d)_m$', '$(c/d)_t$', 
                 '$r_{cdm}$', '$r_{\\alpha m}$', '$r_{ytm}$', 
                 '$\\alpha_r$', '$\\alpha_m$', '$\\alpha_t$', 
                 '$y_{t,max,r}$', '$y_{t,max,m}$', '$y_{t,max,t}$', 
                 '$n_{min}$']

fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='YlOrRd', 
            xticklabels=['2 Blades', '3 Blades', '4 Blades'],
            yticklabels=yticks_labels, ax=ax, cbar_kws={'label': 'Mena CV (%)'})

plt.tight_layout()
plt.savefig('heatmap_cvs_parametros.png', dpi=600, bbox_inches='tight')
plt.show()