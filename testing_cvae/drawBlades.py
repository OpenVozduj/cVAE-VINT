import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from src import geoAirfoil as ga

def blade3D(testDir):    
    os.mkdir(testDir+'/airfoils_blade')
    
    font = {'family' : 'Liberation Serif',
            'weight' : 'normal',
            'size'   : 10}
    mpl.rc('font', **font)
    mpl.rc('axes', linewidth=1)
    mpl.rc('lines', lw=1)
    
    f = open(testDir+'/info_real_prop.json')
    blade = json.load(f)
    
    A = np.load(testDir+'/A.npy')
    
    fig = plt.figure(66)
    ax = fig.add_subplot(projection='3d')
    R = blade['design_parameters']['d']/2
    ax.plot(np.zeros(2), np.zeros(2), np.array([0, R]), '--k')
    
    xtmin = 0.2
    
    for i in range(len(A)):
        X, YU, YL = ga.cstN5(A[i])
        XUI = np.delete(X, 0)
        YUI = np.delete(YU, 0)
        XUI = XUI[::-1]
        YUI = YUI[::-1]
        Xa = np.append(XUI, X, axis=0)
        Ya = np.append(YUI, YL, axis=0)
        # Xscale = Xa*blade['info_sections']['c'][i] - blade['info_sections']['xt'][i]*blade['info_sections']['c'][i]
        Xscale = Xa*blade['info_sections']['c'][i] - xtmin*blade['info_sections']['c'][i]
        Yscale = Ya*blade['info_sections']['c'][i]
        Xphi = Xscale*np.cos(-blade['info_sections']['phi'][i]*np.pi/180) - Yscale*np.sin(-blade['info_sections']['phi'][i]*np.pi/180)
        Yphi = Xscale*np.sin(-blade['info_sections']['phi'][i]*np.pi/180) + Yscale*np.cos(-blade['info_sections']['phi'][i]*np.pi/180)
        Z = np.ones_like(Xscale)*blade['info_sections']['r'][i]
        Points = np.column_stack((Xphi, Yphi))
        Points = np.column_stack((Points, Z))*1000
        ax.plot(Xphi, Yphi, Z, '-b')
        fileXFAirfoil = open(testDir+'/airfoils_blade/airfoil_'+str(i)+'.dat','w')
        for i in range(Xphi.size):
            Pi = ''
            for k in range(3):
                Pi = Pi + str(np.round(Points[i,k], 6))+' '
            Pi = Pi +'\n'
            fileXFAirfoil.write(Pi)
        fileXFAirfoil.close()
    
    ax.set_box_aspect(aspect=(1, 0.5, 4), zoom=1)
    plt.tight_layout()
    fig.savefig(testDir+'/blade3D.png')
    plt.show()
    # plt.close(fig)
    
#%%
B=4
cases = np.array([29, 10, 4, 9, 0])
bestThree = np.load('B4_blades_best.npy')
k = 0
for i in cases:
    case = 'validationBEMT/case_'+str(i)+'/blades_'+str(B)
    for j in range(3):
        sample = case+'/sample_'+str(int(bestThree[k,j]))
        blade3D(sample)
        print(sample+'\n')
    k += 1
