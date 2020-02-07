#ifndef _COMMON_HEADER_H_
#define _COMMON_HEADER_H_
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <queue>
#include <cmath>
#include <string>
#include <cstring>
#include <algorithm>
#include <random>
#include <ctime>
#include <cstdlib>
#include <iomanip>
#include <cstdio>
#include <stdexcept>
#include <omp.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <boost/program_options.hpp>

extern std::mt19937 rand_gen;
extern std::uniform_real_distribution<> rand_distribution;
extern size_t NEURON_INTERACTION_TIME;
extern size_t SPIKE_NUMBER;

#endif //_COMMON_HEADER_H_
