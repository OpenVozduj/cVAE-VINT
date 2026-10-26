import numpy as np
import pickle
import os
import json
from sklearn.preprocessing import MinMaxScaler
from src import geoBlade as gb
from nn import openvint_vae as av
from nn import openvint_mlp as am
from nn import readerGraphics as rg

#%%
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):  
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

def loadNN():
    vae = av.OV_VAE()
    vae.build(input_shape=(256, 256, 2))
    vae.load_weights('nn/vae_5_2.weights.h5')
    
    mlp = am.OV_MLP()
    mlp.load_weights('nn/mlp2.weights.h5')
    return mlp, vae

def normParamCST():
    airfoils = np.load('src/A_arad_family.npy')
    scaler = MinMaxScaler(feature_range=(0, 1))
    normXT = scaler.fit_transform(airfoils)
    return normXT, scaler  

def coeffSections(scaler, mlp, vae, n_min, d, r_R, c, alpha, A,
                  nu, a):
    An = scaler.transform(A)
    zPred = mlp.predict(An)
    imgPred = vae.decoder.predict(zPred)
    n_s = n_min/60
    omega = 2*np.pi*n_s
    omegaR = omega*r_R*d/2
    Re = omegaR*c/nu
    M = omegaR/a
    
    cl = np.array([])
    cd = np.array([])
    for s in range(len(r_R)):
        cl_s, cd_s = rg.searchCoeffswithAlphaRe(alpha[s], Re[s], imgPred[s])
        if M[s] > 0.3:
            cl_sM = cl_s/np.sqrt(1-M[s]**2)
            cl = np.append(cl, cl_sM)
        else:    
            cl = np.append(cl, cl_s)
        cd = np.append(cd, cd_s)
        
    ca = np.column_stack((cl, cd))
    ca = np.column_stack((ca, Re))
    ca = np.column_stack((ca, M))
    return ca

def thrustPower(n_min, nB, d, r, r_R, c, alpha,
                cl, cd, V_inf, rho, nu, a):
    R = d/2
    n_s = n_min/60
    lambda_p = V_inf/(d*n_s)
    omega = 2*np.pi*n_s
    omegaR = omega*r_R*d/2
    
    u1_R = np.zeros_like(r_R)
    integral_u1_R = np.zeros_like(r_R)    
    v_rel = V_inf/omegaR[-1]
    for l in range(10):
        v1_R = -v_rel/2 + np.sqrt((v_rel/2)**2 + u1_R*(r_R - u1_R) + 2*integral_u1_R)
        U1_R = r_R - u1_R
        V1_R = v_rel + v1_R
        W1_R = np.sqrt(U1_R**2 + V1_R**2)
        beta_1 = np.arctan(V1_R/U1_R)*(180/np.pi)
        phi = alpha + beta_1
        K = cl/cd
        invK = 1/K
        sigma = nB*c/(R*np.pi)
        Gamma_R = sigma*cl*W1_R/8
        f_R = (2/np.pi)*np.arccos(np.exp(-0.5*nB*(1-r_R)/(r_R*np.sin(beta_1*(np.pi/180)))))
        u1_R = Gamma_R/(f_R*r_R)
        u1_s = u1_R**2/r_R
        for ns in range(len(r_R)):
            integral_u1_R[ns] = np.trapezoid(u1_s[ns:], r_R[ns:])
    
    dCt = 8*Gamma_R*(U1_R - invK*V1_R)
    dmk = 8*Gamma_R*(V1_R + invK*U1_R)*r_R
    Ct = np.trapezoid(dCt, r_R)
    mk = np.trapezoid(dmk, r_R)
    Thrust =  0.5*Ct*rho*omegaR[-1]**2*np.pi*R**2
    Power = 0.5*mk*rho*omegaR[-1]**3*np.pi*R**2
    alpha_p = Thrust/(rho*n_s**2*d**4)
    beta_p = Power/(rho*n_s**3*d**5)
    eta_d = alpha_p*lambda_p/beta_p
    eta_s = Ct**(3/2)/(2*mk)
    Torque = Power/(2*np.pi*n_s)
    
    TP = {}
    TP['T'] = Thrust
    TP['W'] = Power
    TP['eta_d'] = eta_d
    TP['eta_s'] = eta_s
    TP['H'] = Torque
    
    return TP, phi

#%%
mlp, vae = loadNN()
_, scaler = normParamCST()

#%%
rho = 1.225
nu = 1.4607e-5
a = 340.294
rR_min = 0.15
rR_max = 0.99

X = np.load('ablation_study_designs/L_12/X_physical.npy')

with open('scalers_cvae_hybrid.pkl', 'rb') as f:
    scalers = pickle.load(f)
    
C = np.load('generated_designs_multi_seed/seed_666/c_conditions.npy')
c_physical = scalers['Y_cont'].inverse_transform(C[0,:,:3])

#
dir_Val = 'validationBEMT_L12'
os.makedirs(dir_Val, exist_ok=True)


for i in range(30):
    dir_case =dir_Val + '/case_' + str(i)
    os.makedirs(dir_case, exist_ok=True)
    
    d = c_physical[i,0]
    T = c_physical[i,1]
    V_inf = c_physical[i,2]
    
    for j in range(3):
        dir_blade = dir_case+'/blades_' + str(j+2)
        os.makedirs(dir_blade, exist_ok=True)
        X_samples = X[i,j,:,:] 
        nB = j+2
        
        T_real = np.array([])
        W_real = np.array([])
        etad_real = np.array([])
        for k in range(50):
            dir_sample = dir_blade+'/sample_' + str(k) 
            os.makedirs(dir_sample, exist_ok=True)
            n_min = X_samples[k,14]
            W = X_samples[k,15]
            eta_d = X_samples[k,12]
            eta_s = X_samples[k,13]
            P_ijk = np.array([X_samples[k,0],
                              X_samples[k,1],
                              X_samples[k,2],
                              X_samples[k,3],
                              X_samples[k,6],
                              X_samples[k,7],
                              X_samples[k,8],
                              X_samples[k,4],
                              X_samples[k,9],
                              X_samples[k,10],
                              X_samples[k,11],
                              X_samples[k,5],
                              n_min,
                              nB,
                              d])
            
            dS = gb.paramBlade(P_ijk, dir_sample, r_pmin=rR_min, r_pmax=rR_max)
            A = dS[:,:12]
            r_R = dS[:,12]
            c = dS[:,13]
            alpha = dS[:,14]
            r = dS[:,15]
            
            ca = coeffSections(scaler, mlp, vae, n_min, d, r_R, c, alpha, A, nu, a)
            TP, phi = thrustPower(n_min, nB, d, r, r_R, c, alpha, ca[:,0], ca[:,1], V_inf, 
                                  rho, nu, a)
            
            T_real = np.append(T_real, TP['T'])
            W_real = np.append(W_real, TP['W'])
            etad_real = np.append(etad_real, TP['eta_d'])
            
            info_real_prop = {}
            info_real_prop['Thrust'] = TP['T']
            info_real_prop['Power'] = TP['W']
            info_real_prop['Torque'] = TP['H']
            info_real_prop['eta_d'] = TP['eta_d']
            info_real_prop['eta_s'] = TP['eta_s']
            info_real_prop['V_inf'] = V_inf
            design_param = {}
            design_param['c_d_r'] = P_ijk[0]
            design_param['c_d_m'] = P_ijk[1]
            design_param['c_d_t'] = P_ijk[2]
            design_param['r_cdm'] = P_ijk[3]
            design_param['alpha_r'] = P_ijk[4]
            design_param['alpha_m'] = P_ijk[5]
            design_param['alpha_t'] = P_ijk[6]
            design_param['r_alpham'] = P_ijk[7]
            design_param['yt_r'] = P_ijk[8]
            design_param['yt_m'] = P_ijk[9]
            design_param['yt_t'] = P_ijk[10]
            design_param['r_yt'] = P_ijk[11]
            design_param['n_min'] = P_ijk[12]
            design_param['B'] = P_ijk[13]
            design_param['d'] = P_ijk[14]
            info_real_prop['design_parameters'] = design_param
            sections_data = {}
            sections_data['r_R'] = r_R.tolist()
            sections_data['r'] = r.tolist()
            sections_data['c'] = c.tolist()
            sections_data['phi'] = phi.tolist()
            sections_data['alpha'] = alpha.tolist()
            info_real_prop['info_sections'] = sections_data
            
            json_object = json.dumps(info_real_prop, indent=11, cls=NumpyEncoder)      
            with open(dir_sample + '/info_real_prop.json', 'w') as outfile:
                outfile.write(json_object)
        
        T_tar = np.ones(50)*T
        W_pred = X_samples[:,15]
        etad_pred = X_samples[:,12]
        ErrorT = (abs(T_real - T_tar)/T_tar)*100
        ErrorW = (abs(W_pred - W_real)/W_real)*100
        ErrorEtad = (abs(etad_pred - etad_real)/etad_real)*100
        data_oper = np.column_stack((T_tar, T_real, ErrorT, W_pred, W_real, ErrorW,
                                     etad_pred, etad_real, ErrorEtad))
        X_geo = np.delete(X_samples, [12, 13, 15], axis=1)
        # geo_CV = np.var(X_geo, axis=0)**0.5/np.mean(X_geo, axis=0)
        # geo_std = np.std(X_geo, axis=0)
        # eta_std = np.std(etad_real)
        np.save(dir_blade+'/compDataOper.npy', data_oper)
        np.save(dir_blade+'/geoData.npy', X_geo)