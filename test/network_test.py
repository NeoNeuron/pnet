#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import struct as st

# excecute test program
subprocess.call(['bin/net_sim_test'])

# import test data and pre-process

# potential
f = open('test/data_network_test.bin', 'rb')
shape = st.unpack('QQ', f.read(8*2))
dat = np.empty(shape)
for i in range(shape[0]):
    dat[i] = np.array(st.unpack('d'*shape[1], f.read(8*shape[1])))
f.close()
# spikes
dat_spike = np.genfromtxt('test/data_network_raster.csv', delimiter = ',')
dat_spike = dat_spike[:,:-1]
dat = np.array([abs(x - dat[-1]) for x in dat[:-1]])
dat_spike = np.array([abs(x - dat_spike[-1]) for x in dat_spike[:-1]])
dat_mean = dat.mean(1)
dat_spike_mean = dat_spike.mean(1)

# plot figure

dt = np.logspace(1, dat.shape[0], num = dat.shape[0], base = 0.5)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12,5), dpi = 72)
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
plt.savefig('network_test.eps')
