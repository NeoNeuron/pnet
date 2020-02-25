#!/bin/python
# this script aims to draw rasterogram of neural network
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import sys
import argparse
import configparser as cp
import struct as st
import subprocess
import time
import os

def import_bin_data(fname, use_col):
    f = open(fname, 'rb')
    shape = st.unpack('QQ', f.read(16))
    dat = np.zeros((shape[0], len(use_col)))
    for i in range(shape[0]):
        bf = np.array(st.unpack('d'*shape[1], f.read(8*shape[1])))
        dat[i] = bf[use_col] 
    f.close()
    return dat

# config input parameter:
parser = argparse.ArgumentParser(description = "Integrated script to network state.")
parser.add_argument('dir', type = str, help = 'directory of source data and output data')
args = parser.parse_args()

# import config file

if (not os.path.isfile(args.dir+'/config.ini')):
    raise RuntimeError('Config file does not exist.')
config = cp.ConfigParser()
config.read(args.dir + '/config.ini')
Ne = int(config.get('network', 'ne'))
Ni = int(config.get('network', 'ni'))
neuron_num = Ne + Ni
tmax = float(config.get('time', 't'))

# create canvas

fig = plt.figure(figsize = (12,6), dpi = 80)

# config the plotting range (unit millisecond)
dt = 5
if tmax >= 1000:
    t_start = tmax - 1000
else:
    t_start = 0 
t_end = tmax 
t = np.arange(t_start, t_end, dt)

# draw raster plot
start = time.time()
exc_counter_max = 400
inh_counter_max = 400
if Ne < exc_counter_max:
    exc_counter_max = Ne
if Ni < inh_counter_max:
    inh_counter_max = Ni

ras = np.genfromtxt(args.dir + './raster.csv', delimiter = ',')
exc_mask = (ras[:,0]<exc_counter_max)
inh_mask = np.all([ras[:,0]>=Ne, ras[:,0]<inh_counter_max+Ne], axis=0)

ax1 = plt.subplot2grid((2,3), (0,0), colspan = 2, rowspan = 1)
ax1.scatter(ras[exc_mask,1],ras[exc_mask,0], s = 2,color='r')
ax1.scatter(ras[inh_mask,1],ras[inh_mask,0]-Ne+exc_counter_max,s = 2,color='b')
ax1.set_xlim(t_start, t_end)
ax1.set_ylim(0, exc_counter_max + inh_counter_max + 1)
ax1.set_ylabel('Indices')
ax1.set_xlabel('Time (ms)')
ax1.grid(linestyle='--')
finish = time.time()
print(">> raster plot time : %3.3f s" % (finish - start))


# plot the histogram of mean firing rate
start = time.time()
mrate = np.zeros(neuron_num)
isi_e = np.zeros(Ne)
isi_i = np.zeros(Ni)
for i in range(neuron_num):
    i_mask = ras[:,0]==i
    mrate[i] = (ras[:,0]==i).sum()
    if mrate[i] > 1:
        if i < Ne:
            isi_e[i] = np.mean(np.diff(np.sort(ras[i_mask,1])))
        else:
            isi_i[i-Ne] = np.mean(np.diff(np.sort(ras[i_mask,1])))
    else:
        if i < Ne:
            isi_e[i] = np.nan
        else:
            isi_i[i-Ne] = np.nan
ax2 = plt.subplot2grid((2,3), (0,2), colspan = 1, rowspan = 1)
n1, edge1 = np.histogram(mrate[:Ne]/mrate.mean(), 25)
n2, edge2 = np.histogram(mrate[Ne:]/mrate.mean(), 25)
n1 = n1*100/neuron_num
n2 = n2*100/neuron_num
ax2.bar(edge1[:-1], n1, color = 'r', width = edge1[1]-edge1[0], align='edge', label = 'Exc. neurons')
ax2.bar(edge2[:-1], n2, color = 'b', width = edge2[1]-edge2[0], align='edge', label = 'Inh. neurons')
#ax2.hist(mrate/(mrate.sum()/neuron_num), 50)
ax2.set_xlabel('Rate/mean rate')
ax2.set_ylabel('Pecentage of neurons (%)')
ax2.set_title('Mean firing rate {:5.2f} Hz'.format(mrate.mean()/tmax*1e3))
ax2.grid(linestyle='--')
ax2.legend()
finish = time.time()
print(">> firing rate hist time : %3.3f s" % (finish - start))

if mrate[:Ne].mean():
    print('-> Mean firing rate Exc : %5.3f Hz' % (mrate[:Ne].mean()/tmax*1e3))
else:
    print('-> Mean firing rate Exc : 0 Hz')
if mrate[Ne:].mean():
    print('-> Mean firing rate Inh : %5.3f Hz' % (mrate[Ne:].mean()/tmax*1e3))
else:
    print('-> Mean firing rate Inh : 0 Hz')

# draw distribution of ISI of network
start = time.time()
ax3 = plt.subplot2grid((2,3), (1,2), colspan = 1, rowspan = 1)
ax3.hist(isi_e[~np.isnan(isi_e)], 50, color = 'r', label = 'Exc. neurons')
ax3.hist(isi_i[~np.isnan(isi_i)], int(50/Ne*Ni), color = 'b', label = 'Inh. neurons')
ax3.set_xlabel('ISI (ms)')
ax3.set_ylabel('Number of Neurons')
ax3.legend()
ax3.grid(linestyle='--')
finish = time.time()
print(">> ISI hist time : %3.3f s" % (finish - start))

start = time.time()
# draw the excitatory current and inhibition current of the network
ax4 = plt.subplot2grid((2,3), (1,0), colspan = 2, rowspan = 1)
sample_id = [0, Ne]
V = import_bin_data(args.dir + '/V.bin', use_col = sample_id)
dt = float(config.get('time','stp'))
t = np.arange(t_start, t_end, dt)
t_start_id = int(t_start/dt)
ax4.plot(t, V[t_start_id:t_start_id+len(t),0], 'r', label = 'sample Exc. V')
ax4.plot(t, V[t_start_id:t_start_id+len(t),1], 'b', label = 'sample Inh. V')
ax4.set_xlabel('Time (ms)')
ax4.set_ylabel('Potential')
if os.path.isfile(args.dir + '/I.bin'):
    I = import_bin_data(args.dir + '/I.bin', use_col = sample_id)
    ax4t = ax4.twinx()
    ax4t.plot(t, I[t_start_id:t_start_id+len(t), 0], 'r-', label = 'sample Exc. current')
    ax4t.plot(t, I[t_start_id:t_start_id+len(t), 1], 'b-', label = 'sample Inh. current')
    ax4.set_ylabel('Current')
ax4.legend()
ax4.grid(linestyle='--')
finish = time.time()
print(">> voltage trace time : %3.3f s" % (finish - start))

start = time.time()
# tight up layout and save the figure
plt.tight_layout()
plt.savefig(args.dir + '/net_state.png')
plt.close()
finish = time.time()
print(">> savefig time : %3.3f s" % (finish - start))
