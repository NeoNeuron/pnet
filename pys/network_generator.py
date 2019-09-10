#!/usr/python3
import numpy as np
import struct as st
import argparse
import configparser as cp
import time

# generate networks, including neuronal type vector, adjacent matrix, synaptic strength matrix, poisson seting matrix, spatial location matrix.

# rewrite the configparser.ConfigParser
class MyConfigParser(cp.ConfigParser):
    def __init__(self,defaults=None):
        cp.ConfigParser.__init__(self,defaults=None)
    def optionxform(self, optionstr):
        return optionstr
#---------------------

parser = argparse.ArgumentParser(description = "generate required network architecture")
parser.add_argument('prefix', type=str, default='./', help = 'directory of source data and output data')
args = parser.parse_args()
#========================================
#network setting
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

# spatial location of neurons
# currently using square grid

# print the estimated value of EPSPs and IPSPs

print('see : %f ( %3.3f mV)' % (see, see*100*(1/0.5-1/2)))
print('sie : %f ( %3.3f mV)' % (sie, sie*100*(1/0.5-1/2)))
print('sei : %f (-%3.3f mV)' % (sei, sei*100/7*(1/0.5-1/80)))
print('sii : %f (-%3.3f mV)' % (sii, sii*100/7*(1/0.5-1/80)))

#========================================
# generate config file
config = MyConfigParser()
#---
config.add_section('network')
config['network']['Ne']   = str(Ne) 
config['network']['Ni']   = str(Ni) 
#---
config.add_section('neuron')
config['neuron']['model']   = 'LIF_GH'
config['neuron']['tref']    = '2.0'
#---
config.add_section('synapse')
config['synapse']['file']   = 'smat.npy'
#---
config.add_section('space')
config['space']['mode']     = '-1'
config['space']['delay']    = '3.0'
config['space']['speed']    = '0.3'
config['space']['file']     = 'coordinate.csv'
#---
config.add_section('driving')
config['driving']['file']   = 'PoissonSetting.csv'
config['driving']['seed']   = '3'
config['driving']['gmode']  = 'true'
#---
config.add_section('time')
config['time']['T']         = '1e3'
config['time']['dt']        = '0.03125'
config['time']['stp']       = '0.5'
#---
config.add_section('output')
config['output']['poi']     = 'false'
config['output']['V']       = 'true'
config['output']['I']       = 'true'
config['output']['GE']      = 'true'
config['output']['GI']      = 'true'
with open(args.prefix + '/config.ini', 'w') as configfile:
    config.write(configfile)
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

# matrix of synaptic strength;
# ----------------------------
start = time.time()
smat = np.zeros((N, N))
smat[0:Ne,0:Ne] = mat[0:Ne,0:Ne] * see
smat[Ne:,0:Ne] = mat[Ne:,0:Ne] * sie
smat[0:Ne,Ne:] = mat[0:Ne,Ne:] * sei
smat[Ne:,Ne:] = mat[Ne:,Ne:] * sii
finish = time.time()
print('>> strength matrix : %3.3f s' % (finish-start))

start = time.time()
# Poisson setting matrix
# ----------------------
pmat = np.zeros((N, 2))
pmat[0:Ne, 0] = pr_e
pmat[0:Ne, 1] = ps_e
pmat[Ne:-1, 0] = pr_i
pmat[Ne:-1, 1] = ps_i
#print(select_list)
#pmat[select_list, 0] *= 1
finish = time.time()
print('>> poisson setting : %3.3f s' % (finish-start))

start = time.time()
# coordinate matrix
# -----------------
grid_size = int(np.sqrt(N))
x,y = np.meshgrid(range(grid_size),range(grid_size))
x = (x+0.5)/grid_size
y = (y+0.5)/grid_size
gd = np.empty((N,2))
counter = 0
for i in range(grid_size):
    for j in range(grid_size):
        gd[counter][0]=x[i,j]
        gd[counter][1]=y[i,j]
        counter += 1
finish = time.time()
print('>> coordinate matrix : %3.3f s' % (finish-start))

# save
# ----
#start = time.time()
np.save(args.prefix + 'mat.npy', mat)
#np.savetxt(args.prefix + 'net.txt', mat, fmt = '%d')
#finish = time.time()
#print('>> output adjacent matrix : %3.3f s' % (finish-start))
#
#start = time.time()
np.savetxt(args.prefix + config['driving']['file'], pmat, delimiter = ',', fmt = '%.6f')
#finish = time.time()
#print('>> output poisson matrix : %3.3f s' % (finish-start))
#
#start = time.time()
np.save(args.prefix + config['synapse']['file'], smat)
#finish = time.time()
#print('>> output strength matrix : %3.3f s' % (finish-start))
#
#start = time.time()
np.savetxt(args.prefix + config['space']['file'], gd, delimiter = ',', fmt = '%.6f')
#finish = time.time()
#print('>> output coordinate matrix : %3.3f s' % (finish-start))
