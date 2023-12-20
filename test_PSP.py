"""
Uncomment line 185 in ./include/neuron_population.h
Recompile the simulator
Run the code to estimate the EPSP
"""
# %%
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
path = Path('./test_PSP/')
path.mkdir(parents=True, exist_ok=True)
import time
import sys
sys.path.append('./utils/')
import configparser as cp
# import pnet
import struct as st
def import_bin_data(fname, use_col):
    f = open(fname, 'rb')
    shape = st.unpack('QQ', f.read(16))
    dat = np.zeros((shape[0], len(use_col)))
    for i in range(shape[0]):
        bf = np.array(st.unpack('d'*shape[1], f.read(8*shape[1])))
        dat[i] = bf[use_col] 
    f.close()
    return dat
#%% ---------------------
# generate networks, 
# including neuronal type vector, adjacent matrix, synaptic strength matrix, poisson seting matrix, spatial location matrix.
#network setting
N = 100
conn_mat_file = 'smat.npy'
poi_file = 'PoissonSetting.csv'
# interaction setting 
np.save(path/conn_mat_file, np.zeros((N,N)))

# poisson setting
pr = 0.1 * np.ones(N)     # unit kHz
ps = np.arange(1,N+1)*1e-3 # 
poi_setup = np.vstack((pr, np.zeros(N), ps, np.zeros(N))).T     # EPSP
# poi_setup = np.vstack((np.zeros(N),pr, np.zeros(N), ps)).T      # IPSP
np.savetxt(path/poi_file, poi_setup, delimiter=',')

# timing
T = 1e3
dt = 0.03125
stp = 0.5
# dictionary of config
config = cp.ConfigParser()
config['network'] = {
    'ne' : N,
    'simulator' : 'SSC',
    }
config['neuron'] = {
    'model' : 'LIF_GH',
    'tref'  : '2.0',
    }
config['synapse'] = {
    'file' : conn_mat_file,
    }
config['driving'] = {
    'file'  : poi_file,
    'seed'  : 1,
    }
config['time'] = {
    't'   : T,
    'dt'  : dt,
    'stp' : stp,
    }
config['output'] = {
    'v'   : 'true',
    }
with open(path/'config.ini', 'w') as configfile:
    config.write(configfile)

#%%
# print the estimated value of EPSPs and IPSPs
pse=0.02
psi=0.02
sii=0.02
def get_psp(s, dtype='e'):
    if dtype=='e':
        return s*6.51*10
    elif dtype=='i':
        return s*(-2.45)*10
print('pse : %f ( %3.3f mV)' % (pse, get_psp(pse)))
print('psi : %f ( %3.3f mV)' % (psi, get_psp(psi,'i')))


#%%
raster = np.genfromtxt(path/'raster.csv', delimiter=',')
plt.figure(figsize=(15,3))
plt.plot(raster[:,1], raster[:,0], '|', ms=3)
plt.xlim(0,1000)
plt.ylim(0,100)
#%%
v_data = import_bin_data(path/'V.bin', use_col=np.arange(N))
plt.figure(figsize=(15,3))
for i in range(0, 100):
    c = plt.cm.Greys(i/100)
    plt.plot(v_data[:,i], c=c, alpha=0.5)
plt.xlim(0,200)
# plt.ylim(0)
#%%
peak_id = np.mean(np.argmax(v_data[:80,:], axis=0)).astype(int)
plt.plot(np.arange(1,101)*1e-3,v_data[peak_id,:])
plt.plot([0,0.1], [0, v_data[peak_id,-1]])
plt.xlabel('E strength (dimensionless)', fontsize=16)
plt.ylabel('EPSP (mV)', fontsize=20)
# %%
peak_id = np.mean(np.argmax(-v_data[:160,:], axis=0)).astype(int)
plt.plot(np.arange(1,101)*1e-3,v_data[peak_id,:]*15)
plt.plot([0,0.1], [0, v_data[peak_id,-1]*15])
plt.xlabel('I strength (dimensionless)', fontsize=16)
plt.ylabel('IPSP (mV)', fontsize=20)