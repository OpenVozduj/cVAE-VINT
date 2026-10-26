import numpy as np
from scipy.stats import qmc
import pickle

#%% 
N_tests = 30
# rangos efectivos
Omega_c = np.array([[0.4, 0.8], # d
                    [16, 118],    # T
                    [14, 58]])   # V

sampler = qmc.LatinHypercube(d=3) 
sample = sampler.random(n=N_tests)
c = qmc.scale(sample, Omega_c[:,0], Omega_c[:,1])

#%%
z_test = np.random.randn(50, 8).astype(np.float32)

#%%
with open('scalers_cvae_hybrid.pkl', 'rb') as f:
    scalers = pickle.load(f)
    
c_cont_scaled = scalers['Y_cont'].transform(c)

c_blades_1 = np.zeros_like(c)
c_blades_1[:,0] = 1

c_blades_2 = np.zeros_like(c)
c_blades_2[:,1] = 1

c_blades_3 = np.zeros_like(c)
c_blades_3[:,2] = 1

#%%
c_scaled_1 = np.hstack((c_cont_scaled, c_blades_1)).astype(np.float32)
c_scaled_2 = np.hstack((c_cont_scaled, c_blades_2)).astype(np.float32)
c_scaled_3 = np.hstack((c_cont_scaled, c_blades_3)).astype(np.float32)

np.save('c_scaled_1.npy', c_scaled_1)
np.save('c_scaled_2.npy', c_scaled_2)
np.save('c_scaled_3.npy', c_scaled_3)

np.save('z_test.npy', z_test)