import numpy as np
import json
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl

with open('scalers_cvae_hybrid.pkl', 'rb') as f:
    scalers = pickle.load(f)

#%% Seed 42
C42 = np.load('generated_designs_multi_seed/seed_21/c_conditions.npy')
c_physical42 = scalers['Y_cont'].inverse_transform(C42[0,:,:3])

X42 = np.load('generated_designs_multi_seed/seed_21/X_physical.npy')

performanceB2_42 = np.empty((0,9))
B = 2
for i in range(30):
    T_target_i = c_physical42[i,1]
    case = 'validationBEMT_seed_21/case_'+str(i)+'/blades_'+str(B)
    data = np.load(case+'/compDataOper.npy')
    performanceB2_42 = np.vstack((performanceB2_42, data))
    
performanceB3_42 = np.empty((0,9))
B = 3
for i in range(30):
    T_target_i = c_physical42[i,1]
    case = 'validationBEMT_seed_21/case_'+str(i)+'/blades_'+str(B)
    data = np.load(case+'/compDataOper.npy')
    performanceB3_42 = np.vstack((performanceB3_42, data))

performanceB4_42 = np.empty((0,9))
B = 4
for i in range(30):
    T_target_i = c_physical42[i,1]
    case = 'validationBEMT_seed_21/case_'+str(i)+'/blades_'+str(B)
    data = np.load(case+'/compDataOper.npy')
    performanceB4_42 = np.vstack((performanceB4_42, data))

error_T_42 = np.concatenate((performanceB2_42[:,2], performanceB3_42[:,2], performanceB4_42[:,2]))
eta_42 = np.concatenate((performanceB2_42[:,7], performanceB3_42[:,7], performanceB4_42[:,7]))
#%% Seed 123
C123 = np.load('generated_designs_multi_seed/seed_66/c_conditions.npy')
c_physical123 = scalers['Y_cont'].inverse_transform(C123[0,:,:3])

X123 = np.load('generated_designs_multi_seed/seed_66/X_physical.npy')

performanceB2_123 = np.empty((0,9))
B = 2
for i in range(30):
    T_target_i = c_physical123[i,1]
    case = 'validationBEMT_seed_66/case_'+str(i)+'/blades_'+str(B)
    data = np.load(case+'/compDataOper.npy')
    performanceB2_123 = np.vstack((performanceB2_123, data))
    
performanceB3_123 = np.empty((0,9))
B = 3
for i in range(30):
    T_target_i = c_physical123[i,1]
    case = 'validationBEMT_seed_66/case_'+str(i)+'/blades_'+str(B)
    data = np.load(case+'/compDataOper.npy')
    performanceB3_123 = np.vstack((performanceB3_123, data))

performanceB4_123 = np.empty((0,9))
B = 4
for i in range(30):
    T_target_i = c_physical123[i,1]
    case = 'validationBEMT_seed_66/case_'+str(i)+'/blades_'+str(B)
    data = np.load(case+'/compDataOper.npy')
    performanceB4_123 = np.vstack((performanceB4_123, data))
    
error_T_123 = np.concatenate((performanceB2_123[:,2], performanceB3_123[:,2], performanceB4_123[:,2]))
eta_123 = np.concatenate((performanceB2_123[:,7], performanceB3_123[:,7], performanceB4_123[:,7]))
#%% Seed 2024
C2024 = np.load('generated_designs_multi_seed/seed_512/c_conditions.npy')
c_physical2024 = scalers['Y_cont'].inverse_transform(C2024[0,:,:3])

X2024 = np.load('generated_designs_multi_seed/seed_512/X_physical.npy')

performanceB2_2024 = np.empty((0,9))
B = 2
for i in range(30):
    T_target_i = c_physical2024[i,1]
    case = 'validationBEMT_seed_512/case_'+str(i)+'/blades_'+str(B)
    data = np.load(case+'/compDataOper.npy')
    performanceB2_2024 = np.vstack((performanceB2_2024, data))
    
performanceB3_2024 = np.empty((0,9))
B = 3
for i in range(30):
    T_target_i = c_physical2024[i,1]
    case = 'validationBEMT_seed_512/case_'+str(i)+'/blades_'+str(B)
    data = np.load(case+'/compDataOper.npy')
    performanceB3_2024 = np.vstack((performanceB3_2024, data))

performanceB4_2024 = np.empty((0,9))
B = 4
for i in range(30):
    T_target_i = c_physical2024[i,1]
    case = 'validationBEMT_seed_512/case_'+str(i)+'/blades_'+str(B)
    data = np.load(case+'/compDataOper.npy')
    performanceB4_2024 = np.vstack((performanceB4_2024, data))

error_T_2024 = np.concatenate((performanceB2_2024[:,2], performanceB3_2024[:,2], performanceB4_2024[:,2]))
eta_2024 = np.concatenate((performanceB2_2024[:,7], performanceB3_2024[:,7], performanceB4_2024[:,7]))
#%%
print("Seed 21")
print(f"Total de hélices evaluadas: {len(error_T_42)}")

# ============================================================
# PORCENTAJES POR RANGO
# ============================================================
pct_lt5_42   = np.mean(error_T_42 < 5) * 100
pct_5_10_42  = np.mean((error_T_42 >= 5) & (error_T_42 < 10)) * 100
pct_gt10_42  = np.mean(error_T_42 >= 10) * 100
pct_lt10_42  = np.mean(error_T_42 < 10) * 100  # acumulado

print("\n--- Distribución por rango ---")
print(f"  Válido (Error_T < 5%):           {pct_lt5_42:.1f}%")
print(f"  Parcialmente válido (5-10%):     {pct_5_10_42:.1f}%")
print(f"  No válido (Error_T >= 10%):      {pct_gt10_42:.1f}%")
print(f"  Total con Error_T < 10%:         {pct_lt10_42:.1f}%")

# ============================================================
# ESTADÍSTICAS BÁSICAS
# ============================================================
print("\n--- Estadísticas ---")
print(f"  Media:    {np.mean(error_T_42):.2f}%")
print(f"  Mediana:  {np.median(error_T_42):.2f}%")
print(f"  Std:      {np.std(error_T_42):.2f}%")
print(f"  Mín:      {np.min(error_T_42):.2f}%")
print(f"  Máx:      {np.max(error_T_42):.2f}%")

#%%
print("Seed 66")
print(f"Total de hélices evaluadas: {len(error_T_123)}")

# ============================================================
# PORCENTAJES POR RANGO
# ============================================================
pct_lt5_123   = np.mean(error_T_123 < 5) * 100
pct_5_10_123  = np.mean((error_T_123 >= 5) & (error_T_123 < 10)) * 100
pct_gt10_123  = np.mean(error_T_123 >= 10) * 100
pct_lt10_123  = np.mean(error_T_123 < 10) * 100  # acumulado

print("\n--- Distribución por rango ---")
print(f"  Válido (Error_T < 5%):           {pct_lt5_123:.1f}%")
print(f"  Parcialmente válido (5-10%):     {pct_5_10_123:.1f}%")
print(f"  No válido (Error_T >= 10%):      {pct_gt10_123:.1f}%")
print(f"  Total con Error_T < 10%:         {pct_lt10_123:.1f}%")

# ============================================================
# ESTADÍSTICAS BÁSICAS
# ============================================================
print("\n--- Estadísticas ---")
print(f"  Media:    {np.mean(error_T_123):.2f}%")
print(f"  Mediana:  {np.median(error_T_123):.2f}%")
print(f"  Std:      {np.std(error_T_123):.2f}%")
print(f"  Mín:      {np.min(error_T_123):.2f}%")
print(f"  Máx:      {np.max(error_T_123):.2f}%")

#%%
print("Seed 512")
print(f"Total de hélices evaluadas: {len(error_T_2024)}")

# ============================================================
# PORCENTAJES POR RANGO
# ============================================================
pct_lt5_2024   = np.mean(error_T_2024 < 5) * 100
pct_5_10_2024  = np.mean((error_T_2024 >= 5) & (error_T_2024 < 10)) * 100
pct_gt10_2024  = np.mean(error_T_2024 >= 10) * 100
pct_lt10_2024  = np.mean(error_T_2024 < 10) * 100  # acumulado

print("\n--- Distribución por rango ---")
print(f"  Válido (Error_T < 5%):           {pct_lt5_2024:.1f}%")
print(f"  Parcialmente válido (5-10%):     {pct_5_10_2024:.1f}%")
print(f"  No válido (Error_T >= 10%):      {pct_gt10_2024:.1f}%")
print(f"  Total con Error_T < 10%:         {pct_lt10_2024:.1f}%")

# ============================================================
# ESTADÍSTICAS BÁSICAS
# ============================================================
print("\n--- Estadísticas ---")
print(f"  Media:    {np.mean(error_T_2024):.2f}%")
print(f"  Mediana:  {np.median(error_T_2024):.2f}%")
print(f"  Std:      {np.std(error_T_2024):.2f}%")
print(f"  Mín:      {np.min(error_T_2024):.2f}%")
print(f"  Máx:      {np.max(error_T_2024):.2f}%")

#%%
print("\n--- Estadísticas eta_d (seed 21)---")
print(f"  Media:    {np.mean(eta_42):.2f}%")
print(f"  Mediana:  {np.median(eta_42):.2f}%")
print(f"  Std:      {np.std(eta_42):.2f}%")
print(f"  Mín:      {np.min(eta_42):.2f}%")
print(f"  Máx:      {np.max(eta_42):.2f}%")

print("\n--- Estadísticas eta_d (seed 66)---")
print(f"  Media:    {np.mean(eta_123):.2f}%")
print(f"  Mediana:  {np.median(eta_123):.2f}%")
print(f"  Std:      {np.std(eta_123):.2f}%")
print(f"  Mín:      {np.min(eta_123):.2f}%")
print(f"  Máx:      {np.max(eta_123):.2f}%")

print("\n--- Estadísticas eta_d (seed 512)---")
print(f"  Media:    {np.mean(eta_2024):.2f}%")
print(f"  Mediana:  {np.median(eta_2024):.2f}%")
print(f"  Std:      {np.std(eta_2024):.2f}%")
print(f"  Mín:      {np.min(eta_2024):.2f}%")
print(f"  Máx:      {np.max(eta_2024):.2f}%")