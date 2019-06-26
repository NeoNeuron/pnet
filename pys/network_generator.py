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
Ne = 80   # No. of exc neuron
Ni = 20   # No. of inh neuron
K  = 10    # connection degree

N = Ne + Ni

# interaction setting 
Jee = 1.0e-1
Jie = 1.0e-1
Jei = 1.0e-1
Jii = 1.0e-1

see = Jee / np.sqrt(K)
sie = Jie / np.sqrt(K)
sei = Jei / np.sqrt(K)
sii = Jii / np.sqrt(K)

# poisson setting
pr_e = 20      # unit Hz
pr_i = 20      # unit Hz
ps_e = 1.0e-1     # 
ps_i = 0.8e-1     # 

# rescale poisson
pr_e *= K/1000
pr_i *= K/1000
ps_e /= np.sqrt(K)
ps_i /= np.sqrt(K)

# spatial location of neurons
# currently using square grid

#========================================
# generate config file
config = MyConfigParser()
config.read('doc/config_net.ini')
#---
config.add_section('network')
config['network']['size']   = str(N) 
#---
config.add_section('neuron')
config['neuron']['model']   = 'LIF_GH'
config['neuron']['tref']    = '2.0'
config['neuron']['file']    = 'ty_neu.csv'
#---
config.add_section('synapse')
config['synapse']['file']   = 'smat.npy'
#---
config.add_section('space')
config['space']['mode']     = '0'
config['space']['delay']    = '0.0'
config['space']['speed']    = '1.0'
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
config['output']['I']       = 'false'
config['output']['GE']      = 'false'
config['output']['GI']      = 'false'
with open(args.prefix + '/config.ini', 'w') as configfile:
    config.write(configfile)
#========================================

start = time.time()
# generate type list;
ty  = np.zeros(N)
ty[np.random.choice(N, Ne, replace=False)] = 1
finish = time.time()
print('>> type list : %3.3f s' % (finish-start))

start = time.time()
# generate connecting matrix;
mat = np.zeros((N, N))
for i in range(N):
    new_ty = np.delete(ty, i)
    mat[i, np.random.choice(np.delete(np.arange(N), i)[new_ty == 1], int(K/2), replace=False)] = 1
    mat[i, np.random.choice(np.delete(np.arange(N), i)[new_ty == 0], int(K/2), replace=False)] = 1
    #mat[i, np.random.choice(np.delete(np.arange(N), i), K, replace=False)] = 1
finish = time.time()
print('>> adjacent matrix : %3.3f s' % (finish-start))

start = time.time()
# generate matrix of synaptic strength;
smat = np.zeros((N, N))
for i in range(N):
    if ty[i]:
        con_list = mat[i]*ty
        smat[i, con_list==1] = see
        con_list = mat[i]*(1-ty)
        smat[i, con_list==1] = sei
    else:
        con_list = mat[i]*ty
        smat[i, con_list==1] = sie
        con_list = mat[i]*(1-ty)
        smat[i, con_list==1] = sii
finish = time.time()
print('>> strength matrix : %3.3f s' % (finish-start))

start = time.time()
# generate poisson setting matrix
pmat = np.zeros((N, 2))
pmat[ty==1, 0] = pr_e
pmat[ty==0, 0] = pr_i
pmat[ty==1, 1] = ps_e
pmat[ty==0, 1] = ps_i
finish = time.time()
print('>> poisson setting : %3.3f s' % (finish-start))

start = time.time()
# generate coordinate matrix
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

#start = time.time()
np.savetxt(args.prefix + config['neuron']['file'], ty, delimiter = ',', fmt = '%d')
#finish = time.time()
#print('>> output type list : %3.3f s' % (finish-start))
#
#start = time.time()
np.save(args.prefix + 'mat.npy', mat)
#finish = time.time()
#print('>> output adjacent matrix : %3.3f s' % (finish-start))
#
#start = time.time()
np.savetxt(args.prefix + config['driving']['file'], pmat, delimiter = ',', fmt = '%.3f')
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
