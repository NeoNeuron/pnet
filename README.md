# Point-Neuronal-Network Simulation Frameworks
Point neuronal network simulator framework for academic use, initiating from my bachleor thesis.[^1][^2]


## Contents
```shell
pnet
├── doc: docs and *.ini like configuration files;
├── include: C++ header files;
└── src: C++ source files;
```
##Get Stated
Compile the core program:

```shell
$ make
```

Runing the core program of simulator:

```shell
$ ./bin/neu_sim <path-of-config-file> <dir-of-output-files>
```

## Reference

[^1]: Shelley, M. J., & Tao, L. (2001). Efficient and accurate time-stepping schemes for integrate-and-fire neuronal networks. Journal of Computational Neuroscience, 11(2), 111-119.

[^2]: Rangan, A. V., & Cai, D. (2007). Fast numerical methods for simulating large-scale integrate-and-fire neuronal networks. Journal of computational neuroscience, 22(1), 81-100.