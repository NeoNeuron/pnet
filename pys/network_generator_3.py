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
Ne = 320   # No. of exc neuron
Ni1 = 76   # No. of inh neuron
Ni2 = 4   # No. of inh neuron
K  = 40    # connection degree

N = Ne + Ni1 + Ni2

# interaction setting 
#        E      I1      I2
#   E   Jee    Jei1    Jei2
#  I1   Ji1e   Ji1i1   Ji1i2
#  I2   Ji2e   Ji2i1   Ji2i2
#
Jee   = 1.0e-1
Ji1e  = 1.0e-1
Ji2e  = 1.0e-1

Jei1  = 1.8e-1
Ji1i1 = 1.8e-1
Ji2i1 = 1.8e-1

Jei2  = 2.4e-1
Ji1i2 = 2.4e-1
Ji2i2 = 2.2e-1

see   = Jee   / np.sqrt(K)
si1e  = Ji1e  / np.sqrt(K)
si2e  = Ji2e  / np.sqrt(K)
sei1  = Jei1  / np.sqrt(K)
si1i1 = Ji1i1 / np.sqrt(K)
si2i1 = Ji2i1 / np.sqrt(K)
sei2  = Jei2  / np.sqrt(K)
si1i2 = Ji1i2 / np.sqrt(K)
si2i2 = Ji2i2 / np.sqrt(K)

# poisson setting
pr_e  = 20      # unit Hz
pr_i1 = 20      # unit Hz
pr_i2 = 20      # unit Hz
ps_e  = 1.5e-1  # 
ps_i1 = 1.9e-1  # 
ps_i2 = 3.0e-1  # 

# rescale poisson
pr_e  *= K/1000
pr_i1 *= K/1000
pr_i2 *= K/1000
ps_e  /= np.sqrt(K)
ps_i1 /= np.sqrt(K)
ps_i2 /= np.sqrt(K)

# spatial location of neurons
# currently using square grid

#========================================
# generate config file
config = MyConfigParser()
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
config['time']['T']         = '1e5'
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

start = time.time()
# generate type list;
ty  = np.zeros(N)
ty[np.random.choice(N, Ne, replace=False)] = 1
tyi = np.zeros(N)
tyi[ty == 1] = 0
tyi[ty == 0] = 1
tyi[np.random.choice(np.flatnonzero(ty==0), Ni2, replace = False)] = 2

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
    if tyi[i] == 0:
        con_list = mat[i]*(tyi==0)
        smat[i, con_list==1] = see
        con_list = mat[i]*(tyi==1)
        smat[i, con_list==1] = sei1
        con_list = mat[i]*(tyi==2)
        smat[i, con_list==1] = sei2
    elif tyi[i] == 1:
        con_list = mat[i]*(tyi==0)
        smat[i, con_list==1] = si1e
        con_list = mat[i]*(tyi==1)
        smat[i, con_list==1] = si1i1
        con_list = mat[i]*(tyi==2)
        smat[i, con_list==1] = si1i2
    elif tyi[i] == 2:
        con_list = mat[i]*(tyi==0)
        smat[i, con_list==1] = si2e
        con_list = mat[i]*(tyi==1)
        smat[i, con_list==1] = si2i1
        con_list = mat[i]*(tyi==2)
        smat[i, con_list==1] = si2i2
finish = time.time()
print('>> strength matrix : %3.3f s' % (finish-start))

start = time.time()
# generate poisson setting matrix
pmat = np.zeros((N, 2))
pmat[tyi==0, 0] = pr_e
pmat[tyi==1, 0] = pr_i1
pmat[tyi==2, 0] = pr_i2
pmat[tyi==0, 1] = ps_e
pmat[tyi==1, 1] = ps_i1
pmat[tyi==2, 1] = ps_i2
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
np.savetxt(args.prefix + 'ty3.csv', tyi, delimiter = ',', fmt = '%d')
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
