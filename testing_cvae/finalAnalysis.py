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

#%%
T_target = np.array([])
T_calc2 = np.array([])

B = 2
for i in range(30):
    T_target_i = c_physical[i,1]
    case = 'validationBEMT_seed_42/case_'+str(i)+'/blades_'+str(B)
    for j in range(50):
        sample = case+'/sample_'+str(j)
        data_ij = json.load(open(sample+'/info_real_prop.json'))
        T_calc2 = np.append(T_calc2, data_ij['Thrust'])
        T_target = np.append(T_target, T_target_i)
Error_T2 = 100*abs(T_calc2 - T_target)/T_target

T_calc3 = np.array([])      
B = 3
for i in range(30):
    case = 'validationBEMT_seed_42/case_'+str(i)+'/blades_'+str(B)
    for j in range(50):
        sample = case+'/sample_'+str(j)
        data_ij = json.load(open(sample+'/info_real_prop.json'))
        T_calc3 = np.append(T_calc3, data_ij['Thrust'])
Error_T3 = 100*abs(T_calc3 - T_target)/T_target

T_calc4 = np.array([])   
B = 4
for i in range(30):
    case = 'validationBEMT_seed_42/case_'+str(i)+'/blades_'+str(B)
    for j in range(50):
        sample = case+'/sample_'+str(j)
        data_ij = json.load(open(sample+'/info_real_prop.json'))
        T_calc4 = np.append(T_calc4, data_ij['Thrust'])
Error_T4 = 100*abs(T_calc4 - T_target)/T_target

#%%
font = {'family' : 'Liberation Serif',
        'weight' : 'normal',
        'size'   : 10}
cm=1/2.54
mpl.rc('font', **font)
mpl.rc('axes', linewidth=1)
mpl.rc('lines', lw=1)

val_ref_T = np.array([min(T_target), max(T_target)])

fig = plt.figure(1, figsize=(36*cm, 12*cm))
ax = fig.add_subplot(131)
ax.scatter(T_target, T_calc2, s=4, color='blue', label='Data')
ax.plot(val_ref_T, val_ref_T, color='black', linestyle='-', label='$T_{target}=T_{calc}$')
ax.fill_between(val_ref_T, val_ref_T*0.95, val_ref_T*1.05, color='green', alpha=0.5,
                label='Valid region $\pm 5\%$')
ax.fill_between(val_ref_T, val_ref_T*1.05, val_ref_T*1.10, color='orange', alpha=0.5,
                label='Partially valid region $\pm [5, 10]\%$')
ax.fill_between(val_ref_T, val_ref_T*0.95, val_ref_T*0.90, color='orange', alpha=0.5)
ax.set_xlabel('$T_{target}$ [N]')
ax.set_ylabel('$T_{calc}$ [N]')
ax.set_title('a)')
ax.grid()
ax.legend(loc='best')

ax = fig.add_subplot(132)
ax.scatter(T_target, T_calc3, s=4, color='blue', label='Data')
ax.plot(val_ref_T, val_ref_T, color='black', linestyle='-', label='$T_{target}=T_{calc}$')
ax.fill_between(val_ref_T, val_ref_T*0.95, val_ref_T*1.05, color='green', alpha=0.5,
                label='Valid region $\pm 5\%$')
ax.fill_between(val_ref_T, val_ref_T*1.05, val_ref_T*1.10, color='orange', alpha=0.5,
                label='Partially valid region $\pm [5, 10]\%$')
ax.fill_between(val_ref_T, val_ref_T*0.95, val_ref_T*0.90, color='orange', alpha=0.5)
ax.set_xlabel('$T_{target}$ [N]')
ax.set_ylabel('$T_{calc}$ [N]')
ax.set_title('b)')
ax.grid()
ax.legend(loc='best')

ax = fig.add_subplot(133)
ax.scatter(T_target, T_calc4, s=4, color='blue', label='Data')
ax.plot(val_ref_T, val_ref_T, color='black', linestyle='-', label='$T_{target}=T_{calc}$')
ax.fill_between(val_ref_T, val_ref_T*0.95, val_ref_T*1.05, color='green', alpha=0.5,
                label='Valid region $\pm 5\%$')
ax.fill_between(val_ref_T, val_ref_T*1.05, val_ref_T*1.10, color='orange', alpha=0.5,
                label='Partially valid region $\pm [5, 10]\%$')
ax.fill_between(val_ref_T, val_ref_T*0.95, val_ref_T*0.90, color='orange', alpha=0.5)
ax.set_xlabel('$T_{target}$ [N]')
ax.set_ylabel('$T_{calc}$ [N]')
ax.set_title('c)')
ax.grid()
ax.legend(loc='best')


fig.tight_layout()
plt.show()
plt.savefig('T_targetvsT_calc.png', dpi=600)

#%%

valid_2 = np.sum(Error_T2<=5.0)
valid_3 = np.sum(Error_T3<=5.0)
valid_4 = np.sum(Error_T4<=5.0)

porc_valid_2 = valid_2/1500
porc_valid_3 = valid_3/1500
porc_valid_4 = valid_4/1500

part_valid_2 = np.sum((Error_T2 >= 5) & (Error_T2 <= 10))
part_valid_3 = np.sum((Error_T3 >= 5) & (Error_T3 <= 10))
part_valid_4 = np.sum((Error_T4 >= 5) & (Error_T4 <= 10))

porc_part_valid_2 = part_valid_2/1500
porc_part_valid_3 = part_valid_3/1500
porc_part_valid_4 = part_valid_4/1500

fig = plt.figure(3, figsize=(16*cm, 8*cm))
ax = plt.subplot(121)
binsDTest = np.linspace(0, 40, 40)
ax.hist(np.concatenate((Error_T2, Error_T3, Error_T4)), histtype='step', color='blue', 
        bins=binsDTest)
ax.set_xlabel('$Error_T$')
ax.set_ylabel('Propellers')
ax.legend(loc='best')
ax.set_title('a)')

ax = plt.subplot(122)
binsDTest = np.linspace(0, 40, 40)
ax.hist(Error_T2, histtype='step', color='blue', bins=binsDTest, label='2 blades')
ax.hist(Error_T3, histtype='step', color='green', bins=binsDTest, label='3 blades')
ax.hist(Error_T4, histtype='step', color='red', bins=binsDTest, label='4 blades')
ax.set_xlabel('$Error_T$')
ax.set_ylabel('Propellers')
ax.legend(loc='best')
ax.set_title('b)')

fig.tight_layout()
plt.show()
plt.savefig('Error_T_hist.png', dpi=600)

#%%
import pandas as pd

Error_T2_Avg = np.array([])
Error_T3_Avg = np.array([])
Error_T4_Avg = np.array([])

Error_T2_Std = np.array([])
Error_T3_Std = np.array([])
Error_T4_Std = np.array([])

for i in range(30):
    blade2_avg = np.mean(Error_T2[50*i:50*(i+1)])
    Error_T2_Avg = np.append(Error_T2_Avg, blade2_avg)
    blade3_avg = np.mean(Error_T3[50*i:50*(i+1)])
    Error_T3_Avg = np.append(Error_T3_Avg, blade3_avg)
    blade4_avg = np.mean(Error_T4[50*i:50*(i+1)])
    Error_T4_Avg = np.append(Error_T4_Avg, blade4_avg)
    
    blade2_std = np.std(Error_T2[50*i:50*(i+1)])
    Error_T2_Std = np.append(Error_T2_Std, blade2_std)
    blade3_std = np.std(Error_T3[50*i:50*(i+1)])
    Error_T3_Std = np.append(Error_T3_Std, blade3_std)
    blade4_std = np.std(Error_T4[50*i:50*(i+1)])
    Error_T4_Std = np.append(Error_T4_Std, blade4_std)
    
#%%
best5_B2 = np.argsort(Error_T2_Avg)[:5]
best5_B3 = np.argsort(Error_T3_Avg)[:5]
best5_B4 = np.argsort(Error_T4_Avg)[:5]

best5_best3_2 = np.empty((0,3))
for iB in best5_B2:
    best5_best3_2 = np.vstack((best5_best3_2, np.argsort(Error_T2[50*iB:50*(iB+1)])[:3]))

best5_best3_3 = np.empty((0,3))
for iB in best5_B3:
    best5_best3_3 = np.vstack((best5_best3_3, np.argsort(Error_T2[50*iB:50*(iB+1)])[:3]))
    
best5_best3_4 = np.empty((0,3))
for iB in best5_B4:
    best5_best3_4 = np.vstack((best5_best3_4, np.argsort(Error_T2[50*iB:50*(iB+1)])[:3]))
#%%    
Error_data2 = np.column_stack((c_physical, Error_T2_Avg, Error_T2_Std))
Error_data3 = np.column_stack((c_physical, Error_T3_Avg, Error_T3_Std))
Error_data4 = np.column_stack((c_physical, Error_T4_Avg, Error_T4_Std))

nameColumns = ['d', 'T', 'Vinf', 'Error_avg', 'Error_std']

#Blades 2
df2 = pd.DataFrame(Error_data2, columns=nameColumns)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Subplot 1: Error vs Diámetro
axes[0].errorbar(df2['d'], df2['Error_avg'], yerr=df2['Error_std'],
                 fmt='o', color='#2E86AB', ecolor='#2E86AB', 
                 capsize=5, markersize=8, linewidth=2)
axes[0].axhline(y=5.0, color='green', linestyle='--', linewidth=1.5, alpha=0.7)
axes[0].axhline(y=10.0, color='orange', linestyle='--', linewidth=1.5, alpha=0.7)
axes[0].set_xlabel('$d$ [m]', fontweight='bold')
axes[0].set_ylabel('$Error_T$ (%)', fontweight='bold')
axes[0].set_title('a)', fontweight='bold')
axes[0].set_ylim(0, max(df2['Error_avg'] + df2['Error_std']) + 2)

# Subplot 2: Error vs Empuje
axes[1].errorbar(df2['T'], df2['Error_avg'], yerr=df2['Error_std'],
                 fmt='s', color='#A23B72', ecolor='#A23B72',
                 capsize=5, markersize=8, linewidth=2)
axes[1].axhline(y=5.0, color='green', linestyle='--', linewidth=1.5, alpha=0.7)
axes[1].axhline(y=10.0, color='orange', linestyle='--', linewidth=1.5, alpha=0.7)
axes[1].set_xlabel('$T$ [N]', fontweight='bold')
axes[1].set_ylabel('$Error_T$ (%)', fontweight='bold')
axes[1].set_title('b)', fontweight='bold')
axes[1].set_ylim(0, max(df2['Error_avg'] + df2['Error_std']) + 2)

# Subplot 3: Error vs Velocidad
axes[2].errorbar(df2['Vinf'], df2['Error_avg'], yerr=df2['Error_std'],
                 fmt='^', color='#F18F01', ecolor='#F18F01',
                 capsize=5, markersize=8, linewidth=2)
axes[2].axhline(y=5.0, color='green', linestyle='--', linewidth=1.5, alpha=0.7)
axes[2].axhline(y=10.0, color='orange', linestyle='--', linewidth=1.5, alpha=0.7)
axes[2].set_xlabel('$V$∞ [m/s]', fontweight='bold')
axes[2].set_ylabel('$Error_T$ (%)', fontweight='bold')
axes[2].set_title('c)', fontweight='bold')
axes[2].set_ylim(0, max(df2['Error_avg'] + df2['Error_std']) + 2)

plt.tight_layout()
plt.savefig('error_vs_variables_diseno_B2.png', dpi=600, bbox_inches='tight')
plt.show()

#Blades 3
df3 = pd.DataFrame(Error_data3, columns=nameColumns)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Subplot 1: Error vs Diámetro
axes[0].errorbar(df3['d'], df3['Error_avg'], yerr=df3['Error_std'],
                 fmt='o', color='#2E86AB', ecolor='#2E86AB', 
                 capsize=5, markersize=8, linewidth=2)
axes[0].axhline(y=5.0, color='green', linestyle='--', linewidth=1.5, alpha=0.7)
axes[0].axhline(y=10.0, color='orange', linestyle='--', linewidth=1.5, alpha=0.7)
axes[0].set_xlabel('$d$ [m]', fontweight='bold')
axes[0].set_ylabel('$Error_T$ (%)', fontweight='bold')
axes[0].set_title('a)', fontweight='bold')
axes[0].set_ylim(0, max(df3['Error_avg'] + df3['Error_std']) + 2)

# Subplot 2: Error vs Empuje
axes[1].errorbar(df3['T'], df3['Error_avg'], yerr=df3['Error_std'],
                 fmt='s', color='#A23B72', ecolor='#A23B72',
                 capsize=5, markersize=8, linewidth=2)
axes[1].axhline(y=5.0, color='green', linestyle='--', linewidth=1.5, alpha=0.7)
axes[1].axhline(y=10.0, color='orange', linestyle='--', linewidth=1.5, alpha=0.7)
axes[1].set_xlabel('$T$ [N]', fontweight='bold')
axes[1].set_ylabel('$Error_T$ (%)', fontweight='bold')
axes[1].set_title('b)', fontweight='bold')
axes[1].set_ylim(0, max(df3['Error_avg'] + df3['Error_std']) + 2)

# Subplot 3: Error vs Velocidad
axes[2].errorbar(df3['Vinf'], df3['Error_avg'], yerr=df3['Error_std'],
                 fmt='^', color='#F18F01', ecolor='#F18F01',
                 capsize=5, markersize=8, linewidth=2)
axes[2].axhline(y=5.0, color='green', linestyle='--', linewidth=1.5, alpha=0.7)
axes[2].axhline(y=10.0, color='orange', linestyle='--', linewidth=1.5, alpha=0.7)
axes[2].set_xlabel('$V$∞ [m/s]', fontweight='bold')
axes[2].set_ylabel('$Error_T$ (%)', fontweight='bold')
axes[2].set_title('c)', fontweight='bold')
axes[2].set_ylim(0, max(df3['Error_avg'] + df3['Error_std']) + 2)

plt.tight_layout()
plt.savefig('error_vs_variables_diseno_B3.png', dpi=600, bbox_inches='tight')
plt.show()

#Blades 4
df4 = pd.DataFrame(Error_data4, columns=nameColumns)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Subplot 1: Error vs Diámetro
axes[0].errorbar(df4['d'], df4['Error_avg'], yerr=df4['Error_std'],
                 fmt='o', color='#2E86AB', ecolor='#2E86AB', 
                 capsize=5, markersize=8, linewidth=2)
axes[0].axhline(y=5.0, color='green', linestyle='--', linewidth=1.5, alpha=0.7)
axes[0].axhline(y=10.0, color='orange', linestyle='--', linewidth=1.5, alpha=0.7)
axes[0].set_xlabel('$d$ [m]', fontweight='bold')
axes[0].set_ylabel('$Error_T$ (%)', fontweight='bold')
axes[0].set_title('a)', fontweight='bold')
axes[0].set_ylim(0, max(df4['Error_avg'] + df4['Error_std']) + 2)

# Subplot 2: Error vs Empuje
axes[1].errorbar(df4['T'], df4['Error_avg'], yerr=df4['Error_std'],
                 fmt='s', color='#A23B72', ecolor='#A23B72',
                 capsize=5, markersize=8, linewidth=2)
axes[1].axhline(y=5.0, color='green', linestyle='--', linewidth=1.5, alpha=0.7)
axes[1].axhline(y=10.0, color='orange', linestyle='--', linewidth=1.5, alpha=0.7)
axes[1].set_xlabel('$T$ [N]', fontweight='bold')
axes[1].set_ylabel('$Error_T$ (%)', fontweight='bold')
axes[1].set_title('b)', fontweight='bold')
axes[1].set_ylim(0, max(df4['Error_avg'] + df4['Error_std']) + 2)

# Subplot 3: Error vs Velocidad
axes[2].errorbar(df4['Vinf'], df4['Error_avg'], yerr=df4['Error_std'],
                 fmt='^', color='#F18F01', ecolor='#F18F01',
                 capsize=5, markersize=8, linewidth=2)
axes[2].axhline(y=5.0, color='green', linestyle='--', linewidth=1.5, alpha=0.7)
axes[2].axhline(y=10.0, color='orange', linestyle='--', linewidth=1.5, alpha=0.7)
axes[2].set_xlabel('$V$∞ [m/s]', fontweight='bold')
axes[2].set_ylabel('$Error_T$ (%)', fontweight='bold')
axes[2].set_title('c)', fontweight='bold')
axes[2].set_ylim(0, max(df4['Error_avg'] + df4['Error_std']) + 2)

plt.tight_layout()
plt.savefig('error_vs_variables_diseno_B4.png', dpi=600, bbox_inches='tight')
plt.show()
