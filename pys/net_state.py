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

def import_bin_data(fname, use_col):
    f = open(fname, 'rb')
    shape = st.unpack('QQ', f.read(16))
    #dat = np.array(st.unpack('d'*shape[0]*shape[1], f.read(8*shape[0]*shape[1])))
    #dat = np.reshape(dat, shape)
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

# Run simulations
#subprocess.call(['cp', 'doc/config.ini', args.dir])
#p = subprocess.call(['./bin/net_sim', '-c', args.dir + '/config.ini', '--prefix', args.dir])

typepath = args.dir + '/ty_neu.csv'
types = np.genfromtxt(typepath, delimiter = ',', dtype = 'b')
Ne = types.sum()
Ni = len(types) - types.sum()

# import config file

config = cp.ConfigParser()
config.read(args.dir + '/config.ini')
Ne = int(config.get('network', 'Ne'))
Ni = int(config.get('network', 'Ni'))
neuron_num = Ne + Ni
tmax = float(config.get('time', 'T'))

# create canvas

fig = plt.figure(figsize = (12,6), dpi = 80)

# config the plotting range (unit millisecond)
dt = 5
t_start = tmax - 1000 
t_end = tmax 
t = np.arange(t_start, t_end, dt)
counts_exc = np.zeros(len(t))
counts_inh = np.zeros(len(t))

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
isi_e = np.array([])
isi_i = np.array([])
for i in range(neuron_num):
    i_mask = ras[:,0]==i
    mrate[i] = (ras[:,0]==i).sum()
    if mrate[i] > 0:
        if i < Ne:
            isi_e = np.append(isi_e, np.diff(np.sort(ras[i_mask,1])))
        else:
            isi_i = np.append(isi_i, np.diff(np.sort(ras[i_mask,1])))

ax2 = plt.subplot2grid((2,3), (0,2), colspan = 1, rowspan = 1)
n1, edge1 = np.histogram(mrate[0:Ne]/(mrate.sum()/neuron_num), 25)
n2, edge2 = np.histogram(mrate[Ne:-1]/(mrate.sum()/neuron_num), 25)
n1 = n1*100/neuron_num
n2 = n2*100/neuron_num
ax2.bar(edge1[:-1], n1, color = 'r', width = edge1[1]-edge1[0], align='edge', label = 'EXC Neurons')
ax2.bar(edge2[:-1], n2, color = 'b', width = edge2[1]-edge2[0], align='edge', label = 'INH Neurons')
#ax2.hist(mrate/(mrate.sum()/neuron_num), 50)
ax2.set_xlabel('Rate/mean rate')
ax2.set_ylabel('Pecentage of neurons (%)')
ax2.grid(linestyle='--')
ax2.legend()
finish = time.time()
print(">> firing rate hist time : %3.3f s" % (finish - start))

if mrate[0:Ne].mean():
    print('-> Mean firing rate Exc : %3.3f Hz' % (mrate[types==1].mean()/tmax*1e3))
else:
    print('-> Mean firing rate Exc : 0 Hz')
if mrate[Ne:-1].mean():
    print('-> Mean firing rate Inh : %3.3f Hz' % (mrate[types==0].mean()/tmax*1e3))
else:
    print('-> Mean firing rate Inh : 0 Hz')

#ty0 = np.genfromtxt(args.dir + 'ty3.csv', delimiter = ',')
#print('-> Mean firing rate Exc : %3.3f Hz' % (mrate[ty0==0].mean()/tmax*1e3))
#print('-> Mean firing rate Inh1: %3.3f Hz' % (mrate[ty0==1].mean()/tmax*1e3))
#print('-> Mean firing rate Inh2: %3.3f Hz' % (mrate[ty0==2].mean()/tmax*1e3))

# draw distribution of ISI of network
start = time.time()
ax3 = plt.subplot2grid((2,3), (1,2), colspan = 1, rowspan = 1)
ax3.hist(isi_e, 50, color = 'r', label = 'EXC Neurons')
ax3.hist(isi_i, 50, color = 'b', label = 'INH Neurons')
ax3.set_xlabel('ISI (ms)')
ax3.set_ylabel('Density')
ax3.legend()
ax3.grid(linestyle='--')
finish = time.time()
print(">> ISI hist time : %3.3f s" % (finish - start))

start = time.time()
# draw the excitatory current and inhibition current of the network
ax4 = plt.subplot2grid((2,3), (1,0), colspan = 2, rowspan = 1)
sample_id = [10]
V = import_bin_data(args.dir + '/V.bin', use_col = sample_id)
I = import_bin_data(args.dir + '/I.bin', use_col = sample_id)
GE = import_bin_data(args.dir + '/GE.bin', use_col = sample_id)
GI = import_bin_data(args.dir + '/GI.bin', use_col = sample_id)
ve = 14.0/3.0
vi = -2.0/3.0
Ie = GE*(ve-V)
Ii = GI*(vi-V)
#I = I.sum(1)
#Ie = Ie.sum(1)
#Ii = Ii.sum(1)
dt = 0.5
t = np.arange(t_start, t_end, dt)
t_start_id = int(t_start/dt)
ax4.plot(t, I[t_start_id:t_start_id+len(t)], 'k', label = 'Total current')
ax4.plot(t, np.ones(len(t))*I[t_start_id:t_start_id+len(t)].mean(), 'k')
ax4.plot(t, Ie[t_start_id:t_start_id+len(t)], 'r', label = 'Exc. current')
ax4.plot(t, np.ones(len(t))*Ie[t_start_id:t_start_id+len(t)].mean(), 'r')
ax4.plot(t, Ii[t_start_id:t_start_id+len(t)], 'b', label = 'Inh. current')
ax4.plot(t, np.ones(len(t))*Ii[t_start_id:t_start_id+len(t)].mean(), 'b')
ax4.set_xlabel('Time (ms)')
ax4.set_ylabel('Current')
ax4.legend()
ax4.grid(linestyle='--')
finish = time.time()
print(">> voltage trace time : %3.3f s" % (finish - start))

## draw the adjacent matrix of network
#start = time.time()
#X,Y = np.meshgrid(range(neuron_num + 1),range(neuron_num + 1))
#mat = np.load(args.dir + '/mat.npy')
#
#ax5 = plt.subplot2grid((2,5), (0,3), colspan = 2, rowspan = 2)
#cax5 = ax5.pcolormesh(X, Y, mat, cmap = 'Greys')
#ax5.set_xlabel('Pre-neurons')
#ax5.set_ylabel('Post-neurons')
#ax5.set_title('Connectivity Mat')
##ax5.grid(linestyle = '--')
##fig.colorbar(cax5, ax = ax5)
#finish = time.time()
#print(">> adjacent matrix time : %3.3f s" % (finish - start))

start = time.time()
# tight up layout and save the figure
plt.tight_layout()
plt.savefig(args.dir + '/net_state.png')
plt.close()
finish = time.time()
print(">> savefig time : %3.3f s" % (finish - start))
