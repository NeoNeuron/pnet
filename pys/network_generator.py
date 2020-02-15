#!/usr/python3
import numpy as np
import struct as st
import argparse
import configparser as cp
import time
import sys

sys.path.append('./utils/')
import pnet as pnet
import spatialnet as sn

# generate networks, including neuronal type vector, adjacent matrix, synaptic strength matrix, poisson seting matrix, spatial location matrix.


parser = argparse.ArgumentParser(description = "generate required network architecture")
parser.add_argument('prefix', type=str, default='./', help = 'directory of source data and output data')
args = parser.parse_args()
#========================================
#network setting
model = 'LIF_GH'
Ne = 1200   # No. of exc neuron
Ni = 400   # No. of inh neuron
K  = 160    # connection degree

N = Ne + Ni

# interaction setting 
g = 3.0
Jee = 1.3e-1
Jie = 2.6e-1
Jei = 1.3e-1 * g
Jii = 2.6e-1 * g

see = Jee / np.sqrt(K)
sie = Jie / np.sqrt(K)
sei = Jei / np.sqrt(K)
sii = Jii / np.sqrt(K)

# poisson setting
pr_e = 3.0     # unit Hz
pr_i = 3.0     # unit Hz
ps_e = 2.50e-1     # 
ps_i = 0.50e-1    # 

# rescale poisson
pr_e *= K/1000
pr_i *= K/1000
ps_e /= np.sqrt(K)
ps_i /= np.sqrt(K)

# time
T = 1e3
dt = 0.03125
dt_sampling = 0.5
# spatial location of neurons
# currently using square grid

# print the estimated value of EPSPs and IPSPs

print('see : %f ( %3.3f mV)' % (see, see*100*(1/0.5-1/2)))
print('sie : %f ( %3.3f mV)' % (sie, sie*100*(1/0.5-1/2)))
print('sei : %f (-%3.3f mV)' % (sei, sei*100/7*(1/0.5-1/80)))
print('sii : %f (-%3.3f mV)' % (sii, sii*100/7*(1/0.5-1/80)))

#========================================
np.random.seed(0)

# connectivity matrix
# -------------------
start = time.time()
mat = np.zeros((N, N))
for i in range(N):
    if i < Ne:
        mat[np.random.choice(np.delete(np.arange(Ne), i), int(K/2), replace=False), i] = 1
        mat[np.random.choice(np.arange(Ne, N), int(K/2), replace=False), i] = 1
    elif i >= Ne:
        mat[np.random.choice(np.arange(Ne), int(K/2), replace=False), i] = 1
        mat[np.random.choice(np.delete(np.arange(Ne, N), i-Ne), int(K/2), replace=False), i] = 1
    #mat[np.random.choice(np.delete(np.arange(N), i), K, replace=False), i] = 1
finish = time.time()
print('>> adjacent matrix : %3.3f s' % (finish-start))

#start = time.time()
## regular network
#mat = np.zeros((N,N))
#for i in range(int(K/2)):
#    mat += np.eye(N, k= i+1)
#    mat += np.eye(N, k=-i-1)
#finish = time.time()
#print('>> adjacent matrix : %3.3f s' % (finish-start))

# delay matrix
# -----------------
start = time.time()
x,y = sn.gridmat(N)
gd = np.vstack( (x.flatten(), y.flatten()) ).T
dmat = sn.gen_delay_matrix(gd, 1.0)
finish = time.time()
print('>> delay matrix : %3.3f s' % (finish-start))

# create network object
net = pnet.network(Ne, Ni)
pm = {'model': model,
        'see' : see, 'sie' : sie, 'sei' : sei, 'sii' : sii, 
        'synapse_file': 'smat.npy',
        'con_mat' : mat,
        'space_file': 'dmat.npy',
        'delay_mat' : dmat,
        'pre_e' : pr_e, 'pse_e' : ps_e,
        'pre_i' : pr_i, 'pse_i' : ps_i,
        'poisson_file': 'PoissonSetting.csv', 'poisson_seed': 3,
        'T': T, 'dt': dt, 'stp': dt_sampling,
        'v_flag': True,
        'i_flag': False,
        'ge_flag': False,
        'gi_flag': False,
        }                                         
net.add(**pm)                                               
net.show()

# generate config file
net.saveconfig(args.prefix + 'config.ini')

# save
# ----
net.updatefiles(args.prefix)

#net.run(args.prefix, 'verbose')
