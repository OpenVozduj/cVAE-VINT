import numpy as np
import json
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl

with open('scalers_cvae_hybrid.pkl', 'rb') as f:
    scalers = pickle.load(f)

#%% L4
C = np.load('generated_designs_multi_seed/seed_42/c_conditions.npy')
c_physical = scalers['Y_cont'].inverse_transform(C[0,:,:3])

performanceB2_4 = np.empty((0,9))
B = 2
for i in range(30):
    T_target_i = c_physical[i,1]
    case = 'validationBEMT_L4/case_'+str(i)+'/blades_'+str(B)
    data = np.load(case+'/compDataOper.npy')
    performanceB2_4 = np.vstack((performanceB2_4, data))
    
performanceB3_4 = np.empty((0,9))
B = 3
for i in range(30):
    T_target_i = c_physical[i,1]
    case = 'validationBEMT_L4/case_'+str(i)+'/blades_'+str(B)
    data = np.load(case+'/compDataOper.npy')
    performanceB3_4 = np.vstack((performanceB3_4, data))

performanceB4_4 = np.empty((0,9))
B = 4
for i in range(30):
    T_target_i = c_physical[i,1]
    case = 'validationBEMT_L4/case_'+str(i)+'/blades_'+str(B)
    data = np.load(case+'/compDataOper.npy')
    performanceB4_4 = np.vstack((performanceB4_4, data))

error_T_4 = np.concatenate((performanceB2_4[:,2], performanceB3_4[:,2], performanceB4_4[:,2]))
eta_4 = np.concatenate((performanceB2_4[:,7], performanceB3_4[:,7], performanceB4_4[:,7]))
#%% L8
performanceB2_8 = np.empty((0,9))
B = 2
for i in range(30):
    T_target_i = c_physical[i,1]
    case = 'validationBEMT_seed_42/case_'+str(i)+'/blades_'+str(B)
    data = np.load(case+'/compDataOper.npy')
    performanceB2_8 = np.vstack((performanceB2_8, data))
    
performanceB3_8 = np.empty((0,9))
B = 3
for i in range(30):
    T_target_i = c_physical[i,1]
    case = 'validationBEMT_seed_42/case_'+str(i)+'/blades_'+str(B)
    data = np.load(case+'/compDataOper.npy')
    performanceB3_8 = np.vstack((performanceB3_8, data))

performanceB4_8 = np.empty((0,9))
B = 4
for i in range(30):
    T_target_i = c_physical[i,1]
    case = 'validationBEMT_seed_42/case_'+str(i)+'/blades_'+str(B)
    data = np.load(case+'/compDataOper.npy')
    performanceB4_8 = np.vstack((performanceB4_8, data))

error_T_8 = np.concatenate((performanceB2_8[:,2], performanceB3_8[:,2], performanceB4_8[:,2]))
eta_8 = np.concatenate((performanceB2_8[:,7], performanceB3_8[:,7], performanceB4_8[:,7]))

#%% L12
performanceB2_12 = np.empty((0,9))
B = 2
for i in range(30):
    T_target_i = c_physical[i,1]
    case = 'validationBEMT_L12/case_'+str(i)+'/blades_'+str(B)
    data = np.load(case+'/compDataOper.npy')
    performanceB2_12 = np.vstack((performanceB2_12, data))
    
performanceB3_12 = np.empty((0,9))
B = 3
for i in range(30):
    T_target_i = c_physical[i,1]
    case = 'validationBEMT_L12/case_'+str(i)+'/blades_'+str(B)
    data = np.load(case+'/compDataOper.npy')
    performanceB3_12 = np.vstack((performanceB3_12, data))

performanceB4_12 = np.empty((0,9))
B = 4
for i in range(30):
    T_target_i = c_physical[i,1]
    case = 'validationBEMT_L12/case_'+str(i)+'/blades_'+str(B)
    data = np.load(case+'/compDataOper.npy')
    performanceB4_12 = np.vstack((performanceB4_12, data))

error_T_12 = np.concatenate((performanceB2_12[:,2], performanceB3_12[:,2], performanceB4_12[:,2]))
eta_12 = np.concatenate((performanceB2_12[:,7], performanceB3_12[:,7], performanceB4_12[:,7]))

#%%
print("\nL 4")
print(f"Total de hélices evaluadas: {len(error_T_4)}")

# ============================================================
# PORCENTAJES POR RANGO
# ============================================================
pct_lt5_4   = np.mean(error_T_4 < 5) * 100
pct_5_10_4  = np.mean((error_T_4 >= 5) & (error_T_4 < 10)) * 100
pct_gt10_4  = np.mean(error_T_4 >= 10) * 100
pct_lt10_4  = np.mean(error_T_4 < 10) * 100  # acumulado

print("\n--- Distribución por rango ---")
print(f"  Válido (Error_T < 5%):           {pct_lt5_4:.1f}%")
print(f"  Parcialmente válido (5-10%):     {pct_5_10_4:.1f}%")
print(f"  No válido (Error_T >= 10%):      {pct_gt10_4:.1f}%")
print(f"  Total con Error_T < 10%:         {pct_lt10_4:.1f}%")

# ============================================================
# ESTADÍSTICAS BÁSICAS
# ============================================================
print("\n--- Estadísticas ---")
print(f"  Media:    {np.mean(error_T_4):.2f}%")
print(f"  Mediana:  {np.median(error_T_4):.2f}%")
print(f"  Std:      {np.std(error_T_4):.2f}%")
print(f"  Mín:      {np.min(error_T_4):.2f}%")
print(f"  Máx:      {np.max(error_T_4):.2f}%")

#%%
print("\n L 8")
print(f"Total de hélices evaluadas: {len(error_T_8)}")

# ============================================================
# PORCENTAJES POR RANGO
# ============================================================
pct_lt5_8   = np.mean(error_T_8 < 5) * 100
pct_5_10_8  = np.mean((error_T_8 >= 5) & (error_T_8 < 10)) * 100
pct_gt10_8  = np.mean(error_T_8 >= 10) * 100
pct_lt10_8  = np.mean(error_T_8 < 10) * 100  # acumulado

print("\n--- Distribución por rango ---")
print(f"  Válido (Error_T < 5%):           {pct_lt5_8:.1f}%")
print(f"  Parcialmente válido (5-10%):     {pct_5_10_8:.1f}%")
print(f"  No válido (Error_T >= 10%):      {pct_gt10_8:.1f}%")
print(f"  Total con Error_T < 10%:         {pct_lt10_8:.1f}%")

# ============================================================
# ESTADÍSTICAS BÁSICAS
# ============================================================
print("\n--- Estadísticas ---")
print(f"  Media:    {np.mean(error_T_8):.2f}%")
print(f"  Mediana:  {np.median(error_T_8):.2f}%")
print(f"  Std:      {np.std(error_T_8):.2f}%")
print(f"  Mín:      {np.min(error_T_8):.2f}%")
print(f"  Máx:      {np.max(error_T_8):.2f}%")

#%%
print("\n L 12")
print(f"Total de hélices evaluadas: {len(error_T_12)}")

# ============================================================
# PORCENTAJES POR RANGO
# ============================================================
pct_lt5_12   = np.mean(error_T_12 < 5) * 100
pct_5_10_12  = np.mean((error_T_12 >= 5) & (error_T_12 < 10)) * 100
pct_gt10_12  = np.mean(error_T_12 >= 10) * 100
pct_lt10_12  = np.mean(error_T_12 < 10) * 100  # acumulado

print("\n--- Distribución por rango ---")
print(f"  Válido (Error_T < 5%):           {pct_lt5_12:.1f}%")
print(f"  Parcialmente válido (5-10%):     {pct_5_10_12:.1f}%")
print(f"  No válido (Error_T >= 10%):      {pct_gt10_12:.1f}%")
print(f"  Total con Error_T < 10%:         {pct_lt10_12:.1f}%")

# ============================================================
# ESTADÍSTICAS BÁSICAS
# ============================================================
print("\n--- Estadísticas ---")
print(f"  Media:    {np.mean(error_T_12):.2f}%")
print(f"  Mediana:  {np.median(error_T_12):.2f}%")
print(f"  Std:      {np.std(error_T_12):.2f}%")
print(f"  Mín:      {np.min(error_T_12):.2f}%")
print(f"  Máx:      {np.max(error_T_12):.2f}%")

#%%
print("\n--- Estadísticas eta_d (L 4)---")
print(f"  Media:    {np.mean(eta_4):.2f}%")
print(f"  Mediana:  {np.median(eta_4):.2f}%")
print(f"  Std:      {np.std(eta_4):.2f}%")
print(f"  Mín:      {np.min(eta_4):.2f}%")
print(f"  Máx:      {np.max(eta_4):.2f}%")

print("\n--- Estadísticas eta_d (L 8)---")
print(f"  Media:    {np.mean(eta_8):.2f}%")
print(f"  Mediana:  {np.median(eta_8):.2f}%")
print(f"  Std:      {np.std(eta_8):.2f}%")
print(f"  Mín:      {np.min(eta_8):.2f}%")
print(f"  Máx:      {np.max(eta_8):.2f}%")

print("\n--- Estadísticas eta_d (seed 12)---")
print(f"  Media:    {np.mean(eta_12):.2f}%")
print(f"  Mediana:  {np.median(eta_12):.2f}%")
print(f"  Std:      {np.std(eta_12):.2f}%")
print(f"  Mín:      {np.min(eta_12):.2f}%")
print(f"  Máx:      {np.max(eta_12):.2f}%")