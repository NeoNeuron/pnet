#ifndef _MATH_HELPER_
#define _MATH_HELPER_

double cubic_hermite_root(double x2, double fx1, double dfx1, double fx2, double dfx2, double rhs);

double root_search(double x2, double fx1, double fx2, double dfx1, double dfx2, double rhs, double err); 

#endif
