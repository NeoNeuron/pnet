#!/usr/python3
import numpy as np
import struct as st
import argparse
import time

# generate networks, including neuronal type vector, adjacent matrix, synaptic strength matrix, poisson seting matrix, spatial location matrix.

parser = argparse.ArgumentParser(description = "generate required network architecture")
parser.add_argument('prefix', type=str, default='./', help = 'directory of source data and output data')
args = parser.parse_args()

#========================================

#network setting
Ne = 1200   # No. of exc neuron
Ni = 400    # No. of inh neuron
K  = 20     # connection degree

N = Ne + Ni

# interaction setting 
Jee = 1.0e-1
Jie = 1.0e-1
Jei = 2.0e-1
Jii = 2.6e-1

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

start = time.time()
# generate type list;
ty  = np.zeros(N)
ty[np.random.choice(N, Ne, replace=False)] = 1

# generate connecting matrix;
mat = np.zeros((N, N))
for i in range(N):
    new_ty = np.delete(ty, i)
    mat[i, np.random.choice(np.delete(np.arange(N), i)[new_ty == 1], int(K/2), replace=False)] = 1
    mat[i, np.random.choice(np.delete(np.arange(N), i)[new_ty == 0], int(K/2), replace=False)] = 1
    #mat[i, np.random.choice(np.delete(np.arange(N), i), K, replace=False)] = 1

# generate matrix of synaptic strength;
smat = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        if mat[i,j]:
            if ty[i]:
                if ty[j]:
                    smat[i,j] = see
                else:
                    smat[i,j] = sei
            else:
                if ty[j]:
                    smat[i,j] = sie
                else:
                    smat[i,j] = sii

# generate poisson setting matrix
pmat = np.zeros((N, 2))
pmat[ty==1, 0] = pr_e
pmat[ty==0, 0] = pr_i
pmat[ty==1, 1] = ps_e
pmat[ty==0, 1] = ps_i

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

np.savetxt(args.prefix + 'ty_neu.csv', ty, delimiter = ',', fmt = '%d')
np.savetxt(args.prefix + 'mat.csv', mat, delimiter = ',', fmt = '%d')
np.savetxt(args.prefix + 'PoissonSetting.csv', pmat, delimiter = ',', fmt = '%.3f')
np.savetxt(args.prefix + 'smat.csv', smat, delimiter = ',', fmt = '%.4e')
np.savetxt(args.prefix + 'coordinate.csv', gd, delimiter = ',', fmt = '%.6f')

finish = time.time()
print('>> cost time : %3.3f s' % (finish-start))
