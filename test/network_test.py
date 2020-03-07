#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import struct as st
import argparse
import configparser as cp
import time
import os
import sys
sys.path.append('./utils/')
import spatialnet as sn

parser = argparse.ArgumentParser(description = "generate required network architecture")
parser.add_argument('prefix', type=str, default='./', help = 'directory of source data and output data')
args = parser.parse_args()
#========================================

#network setting
model = 'LIF_GH'
simulator = 'SSC'
Ne = 80    # No. of exc neuron
Ni = 20    # No. of inh neuron
K  = 10    # connection degree

N = Ne + Ni

# interaction setting 
Jee = 2.0e-2
Jie = 2.0e-2
Jei = 1.0e-1
Jii = 1.0e-1

see = Jee / np.sqrt(K)
sie = Jie / np.sqrt(K)
sei = Jei / np.sqrt(K)
sii = Jii / np.sqrt(K)

# poisson setting
pr_e = 30     # unit Hz
pr_i = 30     # unit Hz
ps_e = 1.0e-1     # 
ps_i = 1.0e-1     # 

# rescale poisson
pr_e *= K/1000
pr_i *= K/1000
ps_e /= np.sqrt(K)
ps_i /= np.sqrt(K)

# timing
reps = 10
T = 1e3
t0 = 0.5
# spatial location of neurons
# currently using square grid

# print the estimated value of EPSPs and IPSPs
td_e=2
td_i=10
print('see : {:5.1e} ( {:5.2f} mV)'.format(see, see*(458.08*td_e**0.25-506.87)))
print('sie : {:5.1e} ( {:5.2f} mV)'.format(sie, sie*(458.08*td_e**0.25-506.87)))
print('sei : {:5.1e} (-{:5.2f} mV)'.format(sei, sei*(65.44*td_i**0.25-72.41)))
print('sii : {:5.1e} (-{:5.2f} mV)'.format(sii, sii*(65.44*td_i**0.25-72.41)))

#========================================
# generate config file
config = cp.ConfigParser()
#---
config.add_section('network')
config['network']['ne']         = str(Ne) 
config['network']['ni']         = str(Ni) 
config['network']['simulator']  = simulator 
#---
config.add_section('neuron')
config['neuron']['model']   = model
config['neuron']['tref']    = '2.0'
#---
config.add_section('synapse')
config['synapse']['file']   = 'smat.npy'
#---
config.add_section('space')
config['space']['file']     = 'dmat.npy'
#---
config.add_section('driving')
config['driving']['file']   = 'PoissonSetting.csv'
config['driving']['seed']   = '1'
#---
config.add_section('time')
config['time']['t']         = str(T)
config['time']['dt0']       = str(t0)
config['time']['reps']      = str(reps) 
#---
config.add_section('output')
config['output']['poi']     = 'false'
if (not os.path.isdir(args.prefix)):
    call_state = subprocess.call(['mkdir', '-p', args.prefix])
    if call_state!=0:
        raise RuntimeError('Fail to create directory '+args.prefix)
    
with open(args.prefix + '/config.ini', 'w') as configfile:
    config.write(configfile)
#========================================
np.random.seed(8)


# generate connecting matrix;
start = time.time()
mat = np.zeros((N, N))
if N > 1:
    for i in range(N):
        mat[i, np.random.choice(np.delete(np.arange(N), i), K, replace=False)] = 1
finish = time.time()
print('>> adjacent matrix : %3.3f s' % (finish-start))

# matrix of synaptic strength;
# ----------------------------
start = time.time()
smat = np.zeros((N, N))
smat[:Ne,:Ne] = mat[:Ne,:Ne] * see
smat[Ne:,:Ne] = mat[Ne:,:Ne] * sie
smat[:Ne,Ne:] = mat[:Ne,Ne:] * sei
smat[Ne:,Ne:] = mat[Ne:,Ne:] * sii
finish = time.time()
print('>> strength matrix : %3.3f s' % (finish-start))

# Poisson setting matrix
# ----------------------
start = time.time()
pmat = np.zeros((N, 4))
pmat[:Ne,0] = pr_e
pmat[:Ne,1] = 0.0
pmat[:Ne,2] = ps_e
pmat[:Ne,3] = 0.0
pmat[Ne:,0] = pr_i
pmat[Ne:,1] = 0.0
pmat[Ne:,2] = ps_i
pmat[Ne:,3] = 0.0
finish = time.time()
print('>> poisson setting : %3.3f s' % (finish-start))

# generate coordinate matrix
start = time.time()
grid_size = int(np.sqrt(N))
x,y = np.meshgrid(range(grid_size),range(grid_size))
x = (x+0.5)/grid_size
y = (y+0.5)/grid_size
gd = np.vstack( (x.flatten(), y.flatten()) ).T
dmat = sn.gen_delay_matrix(gd, 1.0)
finish = time.time()
print('>> coordinate matrix : %3.3f s' % (finish-start))

# saving config files
np.save(args.prefix + 'mat.npy', mat)
np.savetxt(args.prefix + config['driving']['file'], pmat, delimiter = ',', fmt = '%.6f')
np.save(args.prefix + config['synapse']['file'], smat)
np.save(args.prefix + config['space']['file'], dmat)


subprocess.call(['rm', '-f', args.prefix + '/ras_*.csv'])

call_state = subprocess.call(['bin/net_sim_test', '--prefix', args.prefix])
if call_state != 0:
    raise RuntimeError('Fail to execute test program.')

# import test data and pre-process

# potential
f = open(args.prefix + '/data_network_test.bin', 'rb')
shape = st.unpack('QQ', f.read(8*2))
dat = np.empty(shape)
for i in range(shape[0]):
    dat[i] = np.array(st.unpack('d'*shape[1], f.read(8*shape[1])))
f.close()
# spikes
dat_spike = np.zeros(reps)
for i in range(reps):
    dat_spike_raw = np.genfromtxt(args.prefix + '/ras_' + str(i) + '.csv', delimiter = ',')
    dat_spike[i] = dat_spike_raw[-1, 1]
    print('>> Mean firing rate : %3.3f, last spike from Neuron %d' % ((dat_spike_raw.shape[0]*1.0/(Ne+Ni)/T*1e3), dat_spike_raw[-1,0]))

dat = np.array([abs(x - dat[-1]) for x in dat[:-1]])
dat_spike = np.abs(dat_spike - dat_spike[-1])
dat_mean = dat.mean(1)
dat_spike_mean = dat_spike[:-1]

# plot figure
order = (1 if simulator = 'Simple' else 4)
dt = np.logspace(1, dat.shape[0], num = dat.shape[0], base = 0.5)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12,5), dpi = 100)
# subplot 1
ax1.plot(dt, dat_mean, '-o', markerfacecolor = 'None', label = 'V')
c_est = dat_mean[-1]/dt[-1]**order
ax1.plot(dt, c_est*dt**order, label = '{:d}-order'.format(order))
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.legend()
ax1.grid()
ax1.set_title('Convergence of potential')
ax1.set_xlabel('Timing step (ms)', fontsize = 12)
ax1.set_ylabel('Relative deviation', fontsize = 12)
# subplot 2
ax2.plot(dt, dat_spike_mean, '-o', markerfacecolor = 'None', label = 'spike')
c_est = dat_spike_mean[-1]/dt[-1]**order
ax2.plot(dt, c_est*dt**order, label = '{:d}-order'.format(order))
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_title('Convergence of Spike trains')
plt.legend()
plt.grid()
ax2.set_xlabel('Timing step (ms)', fontsize = 12)
ax2.set_ylabel('Relative deviation', fontsize = 12)
plt.tight_layout()
plt.savefig(args.prefix + 'network_test.png')
subprocess.call(['rm', '-f', args.prefix + '/data_network_test.bin', args.prefix + '/data_network_raster.csv'])
