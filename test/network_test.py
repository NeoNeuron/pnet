#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import struct as st
import argparse
import configparser as cp
import time

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
Ni = 20    # No. of inh neuron
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
pr_e = 10     # unit Hz
pr_i = 10     # unit Hz
ps_e = 3.0e-1     # 
ps_i = 3.0e-1     # 

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
config['space']['mode']     = '1'
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
config['time']['dt0']       = '0.5'
config['time']['reps']      = '10'
#---
config.add_section('output')
config['output']['poi']     = 'false'
with open(args.prefix + '/config.ini', 'w') as configfile:
    config.write(configfile)
#========================================
np.random.seed(0)

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
    mat[i, np.random.choice(np.delete(np.arange(N), i), K, replace=False)] = 1
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

# saving config files
np.savetxt(args.prefix + config['neuron']['file'], ty, delimiter = ',', fmt = '%d')
np.save(args.prefix + 'mat.npy', mat)
np.savetxt(args.prefix + config['driving']['file'], pmat, delimiter = ',', fmt = '%.6f')
np.save(args.prefix + config['synapse']['file'], smat)
np.savetxt(args.prefix + config['space']['file'], gd, delimiter = ',', fmt = '%.6f')



subprocess.call(['bin/net_sim_test', '--prefix', args.prefix])

# import test data and pre-process

# potential
f = open(args.prefix + '/data_network_test.bin', 'rb')
shape = st.unpack('QQ', f.read(8*2))
dat = np.empty(shape)
for i in range(shape[0]):
    dat[i] = np.array(st.unpack('d'*shape[1], f.read(8*shape[1])))
f.close()
# spikes
dat_spike = np.genfromtxt(args.prefix + '/data_network_raster.csv', delimiter = ',')
dat_spike = dat_spike[:,:-1]
dat = np.array([abs(x - dat[-1]) for x in dat[:-1]])
dat_spike = np.array([abs(x - dat_spike[-1]) for x in dat_spike[:-1]])
dat_mean = dat.mean(1)
dat_spike_mean = dat_spike.mean(1)

# plot figure

dt = np.logspace(1, dat.shape[0], num = dat.shape[0], base = 0.5)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12,5), dpi = 100)
# subplot 1
ax1.plot(dt, dat_mean, '-o', markerfacecolor = 'None', label = 'V')
c_est = dat_mean[-1]/dt[-1]**4
ax1.plot(dt, c_est*dt**4, label = '4th-order')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.legend()
ax1.grid()
ax1.set_title('Convergence of potential')
ax1.set_xlabel('Timing step (ms)', fontsize = 12)
ax1.set_ylabel('Relative deviation', fontsize = 12)
# subplot 2
ax2.plot(dt, dat_spike_mean, '-o', markerfacecolor = 'None', label = 'spike')
c_est = dat_spike_mean[-1]/dt[-1]**4
ax2.plot(dt, c_est*dt**4, label = '4th-order')
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_title('Convergence of Spike trains')
plt.legend()
plt.grid()
ax2.set_xlabel('Timing step (ms)', fontsize = 12)
ax2.set_ylabel('Relative deviation', fontsize = 12)
plt.tight_layout()
plt.savefig('network_test.png')
subprocess.call(['rm', '-f', args.prefix + '/data_network_test.bin', args.prefix + '/data_network_raster.csv'])
