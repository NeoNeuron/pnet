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
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>
#include <boost/program_options.hpp>

#ifndef DEBUG
  #define dbg_printf(...) ((void) 0);
#else 
  // Ref: https://en.wikipedia.org/wiki/Variadic_macro
  #define dbg_printf(format, ...) \
    printf ("DBG: %s(%u): " format "\n", __FILE__, __LINE__, ##__VA_ARGS__)
#endif

#define sNaN std::numeric_limits<size_t>::quiet_NaN()
#define dNaN std::numeric_limits<double>::quiet_NaN()
#define Inf  std::numeric_limits<double>::infinity();

extern size_t NEURON_INTERACTION_TIME;
extern size_t SPIKE_NUMBER;
extern size_t POISSON_CALL_TIME;

#endif //_COMMON_HEADER_H_
