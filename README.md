# Point-Neuronal-Network Simulation Frameworks
Point neuronal network simulator framework for academic use, initiating from my bachleor thesis.[^1][^2]

## Current supported neuronal models
- Conductance based leaky integrate-and-fire model (LIF_G)
- Conductance based leaky integrate-and-fire model with $1^{st}$ order smoothness in conductance (LIF_GH)
- Current based leaky integrate-and-fire model (LIF_I)

## Contents

```shell
pnet
├── doc       docs and *.ini like configuration files;
├── include   C++ header files;
├── src       C++ source files;
├── external  external dependence;
├── utils     python interface;
├── pys       python scripts;
└── test      test files;
```

## Get Started
### Installation

1. Build external `cnpy` library
	
	```bash
	mkdir build
	cd build
	cmake ../external/cnpy -DCMAKE_INSTALL_PREFIX=../
	make
	make install
	```

2. Build core pnet simulator

	```bash
	cd ..
	make -j
	```

### How to use

Runing core simulation program:

```shell
$ bin/neu_sim --prefix <path-of-config-file> -c <config filename>
```

### Getting more config info

```shell
$ bin/neu_sim -h
All Options:
  -h [ --help ]             produce help message
  -p [ --prefix ] arg (=./) prefix of output files
  -c [ --config ] arg       config file, relative to prefix
  -v [ --verbose ]          show output

Configs:
  --network.ne arg (=1)          number of Exc. neurons
  --network.ni arg (=0)          number of Inh. neurons
  --network.simulator arg (=SSC) One of Simple, SSC, SSC_Sparse.
  --neuron.model arg (=LIF_GH)   One of LIF_I, LIF_G, LIF_GH.
  --neuron.tref arg (=2)         (ms) refractory period
  --synapse.file arg             file of synaptic strength
  --space.file arg               file of spatial location of neurons
  --driving.file arg             file of Poisson settings
  --driving.seed arg             seed to generate Poisson point process
  --time.t arg (=1000)           (ms) total simulation time
  --time.dt arg (=0.03125)       (ms) simulation time step
  --time.stp arg (=0.5)          (ms) sampling time step
  --output.poi arg (=0)          output flag of Poisson Drive
  --output.v arg (=0)            output flag of V to V.bin
  --output.i arg (=0)            output flag of I to I.bin
  --output.ge arg (=0)           output flag of GE to GE.bin
  --output.gi arg (=0)           output flag of GI to GI.bin
  
```

### Python interface

Python interface for configuration of network simulation, including type of 
neuronal model, type of network simulator, network structure, network inputs,
spatial structure of 2-D network.

```python
>>> # simulate an all connected 100 LIF_GH neurons network
>>> from utils.pnet import *
>>> net = network(80, 20)
>>> mat = np.zeros((100,100), dtype=int)
>>> np.fill_diagonal(mat, 0)
>>> dmat = np.zeros((100,100))
>>> pm = {'model': 'LIF_GH', 'simulator' : 'SSC', 'tref': 2.0,
...:  'see' : 1e-3, 'sie' : 1e-3, 'sei' : 5e-3, 'sii' : 5e-3,
...:  'synapse_file': 'smat.npy',
...:  'con_mat' : mat,
...:  'space_file': 'dmat.npy',
...:  'delay_mat' : dmat,
...:  'pre_e' : 1.5, 'pse_e' : 5e-3,
...:  'pre_i' : 1.5, 'pse_i' : 5e-3,
...:  'poisson_file': 'PoissonSetting.csv', 'poisson_seed': 3,
...:  'T': 1e3, 'dt': 0.03125, 'stp': 0.5,
...:  'v_flag': True}
>>> # add new simulation configs
>>> net.add(**pm)
>>> # show network configs
>>> net.show()
========================================
Neuron Population:
        ne        ni        ke        ki
        80        20      99.0      99.0
----------------------------------------
Synapses:
       see       sie       sei       sii
  1.00e-03  1.00e-03  5.00e-03  5.00e-03
----------------------------------------
FFWD Poisson:
       pre       pse       pri       psi
  1.50e+00  5.00e-03  0.00e+00  0.00e+00
----------------------------------------
Afferent Connection:
   99   99   99   99   99   99   99   99
========================================
>>> # update configuration files
>>> net.updatefiles()
>>> # run simulation
>>> net.run('verbose')
(number of connections in sparse-mat 9900)
>> Initialization :     0.007 s
>> Done!
>> Simulation :         0.341 s
Total inter-neuronal interaction : 112563
Mean firing rate : 11.37 Hz

```

## Reference

[^1]: Shelley, M. J., & Tao, L. (2001). Efficient and accurate time-stepping schemes for integrate-and-fire neuronal networks. Journal of Computational Neuroscience, 11(2), 111-119.

[^2]: Rangan, A. V., & Cai, D. (2007). Fast numerical methods for simulating large-scale integrate-and-fire neuronal networks. Journal of computational neuroscience, 22(1), 81-100.
