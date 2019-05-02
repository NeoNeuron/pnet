#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "math_helper.h"
/* poly/solve_cubic.c
 * 
 * Copyright (C) 1996, 1997, 1998, 1999, 2000, 2007, 2009 Brian Gough
 * 
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or (at
 * your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
 */

/* solve_cubic.c - finds the real roots of x^3 + a x^2 + b x + c = 0 */


#define SWAP(a,b) do { double tmp = b ; b = a ; a = tmp ; } while(0)

int 
gsl_poly_solve_cubic (double a, double b, double c, 
                      double *x0, double *x1, double *x2)
{
  double q = (a * a - 3 * b);
  double r = (2 * a * a * a - 9 * a * b + 27 * c);

  double Q = q / 9;
  double R = r / 54;

  double Q3 = Q * Q * Q;
  double R2 = R * R;

  double CR2 = 729 * r * r;
  double CQ3 = 2916 * q * q * q;

  if (R == 0 && Q == 0)
    {
      *x0 = - a / 3 ;
      *x1 = - a / 3 ;
      *x2 = - a / 3 ;
      return 3 ;
    }
  else if (CR2 == CQ3) 
    {
      /* this test is actually R2 == Q3, written in a form suitable
         for exact computation with integers */

      /* Due to finite precision some double roots may be missed, and
         considered to be a pair of complex roots z = x +/- epsilon i
         close to the real axis. */

      double sqrtQ = sqrt (Q);

      if (R > 0)
        {
          *x0 = -2 * sqrtQ  - a / 3;
          *x1 = sqrtQ - a / 3;
          *x2 = sqrtQ - a / 3;
        }
      else
        {
          *x0 = - sqrtQ  - a / 3;
          *x1 = - sqrtQ - a / 3;
          *x2 = 2 * sqrtQ - a / 3;
        }
      return 3 ;
    }
  else if (R2 < Q3)
    {
      double sgnR = (R >= 0 ? 1 : -1);
      double ratio = sgnR * sqrt (R2 / Q3);
      double theta = acos (ratio);
      double norm = -2 * sqrt (Q);
      *x0 = norm * cos (theta / 3) - a / 3;
      *x1 = norm * cos ((theta + 2.0 * M_PI) / 3) - a / 3;
      *x2 = norm * cos ((theta - 2.0 * M_PI) / 3) - a / 3;
      
      /* Sort *x0, *x1, *x2 into increasing order */

      if (*x0 > *x1)
        SWAP(*x0, *x1) ;
      
      if (*x1 > *x2)
        {
          SWAP(*x1, *x2) ;
          
          if (*x0 > *x1)
            SWAP(*x0, *x1) ;
        }
      
      return 3;
    }
  else
    {
      double sgnR = (R >= 0 ? 1 : -1);
      double A = -sgnR * pow (fabs (R) + sqrt (R2 - Q3), 1.0/3.0);
      double B = Q / A ;
      *x0 = A + B - a / 3;
      return 1;
    }
}

/* solve_quadratic.c - finds the real roots of a x^2 + b x + c = 0 */
int 
gsl_poly_solve_quadratic (double a, double b, double c, 
                          double *x0, double *x1)
{
  if (a == 0) /* Handle linear case */
    {
      if (b == 0)
        {
          return 0;
        }
      else
        {
          *x0 = -c / b;
          return 1;
        };
    }

  {
    double disc = b * b - 4 * a * c;
    
    if (disc > 0)
      {
        if (b == 0)
          {
            double r = sqrt (-c / a);
            *x0 = -r;
            *x1 =  r;
          }
        else
          {
            double sgnb = (b > 0 ? 1 : -1);
            double temp = -0.5 * (b + sgnb * sqrt (disc));
            double r1 = temp / a ;
            double r2 = c / temp ;
            
            if (r1 < r2) 
              {
                *x0 = r1 ;
                *x1 = r2 ;
              } 
            else 
              {
                *x0 = r2 ;
                  *x1 = r1 ;
              }
          }
        return 2;
      }
    else if (disc == 0) 
      {
        *x0 = -0.5 * b / a ;
        *x1 = -0.5 * b / a ;
        return 2 ;
      }
    else
      {
        return 0;
      }
  }
}

// Find the first real root of f(x) = m*x^3 + n*x^2 + p*x + q = rhs within [0, x2];
// where f(0) = fx1, f'(0) = dfx1, f(x2) = fx2, f'(x2) = dfx2;
// return the smallest root if solvable, otherwise, NaN;
// for fx1*fx2 <= 0, root must exist;
double cubic_hermite_root(double x2, double fx1, double fx2, double dfx1, double dfx2, double rhs) {
	// normalize fx1, fx2;
	fx1 -= rhs;
	fx2 -= rhs;
	// normalize to x = [0, 1];
	dfx1 *= x2;
	dfx2 *= x2;
	double c[4], r0 = NAN, r1 = NAN, r2 = NAN;
	// determine coefficents c[0], c[1], c[2], c[3];
	// f(x) = c[3]*x^3 + c[2] * x^2 + c[1] * x + c[0] = 0;
	c[0] = fx1;
	c[1] = dfx1;
	c[2] = -2 * dfx1 - dfx2 - 3 * (fx1 - fx2);
	c[3] = dfx1 + dfx2 + 2 * (fx1 - fx2);
	if (c[3] != 0) {
		if (c[0] != 0) {	
			// degenerate to linear function;
			// The cubic formula gives the larger solution a relatively small error;
			// x = 1 / t;
			// f(t) = c[3] + c[2] * t + c[1] * t^2 + c[0] * t^3 = 0
			// solve equations above instead;
			gsl_poly_solve_cubic(c[1] / c[0], c[2] / c[0], c[3] / c[0], &r0, &r1, &r2);
			r0 = 1 / r0;
			r1 = 1 / r1;
			r2 = 1 / r2;
			if (r0 > r1) SWAP(r0, r1);
			if (r1 > r2) {
				SWAP(r1, r2);
				if (r0 > r1) SWAP(r0, r1);
			}
		} else r0 = 0;
	} else { // degenerate to quadratic function;
		if (c[0] == 0 && c[1] == 0 && c[2] == 0) r0 = 0;
		gsl_poly_solve_quadratic(c[2], c[1], c[0], &r0, &r1);
	}
	if (0 <= r0 && r0 <= 1) return r0 * x2;
	if (0 <= r1 && r1 <= 1) return r1 * x2;
	if (0 <= r2 && r2 <= 1) return r2 * x2;
	if (fx1 * fx2 <= 0) return root_search(x2, fx1 + rhs, fx2 + rhs, dfx1 / x2, dfx2 / x2, rhs, 1e-15);
	return NAN;
}

// search root of hermite polynomial with Newton's Method;
// where f(0) = fx1, f'(0) = dfx1, f(x2) = fx2, f'(x2) = dfx2;
// rhs: right hand side of the equation;
// err: mannully chosen tolerant error;
// return the root;
double root_search(double x2, double fx1, double fx2, double dfx1, double dfx2, double rhs, double err) {
	double error = 1, root;
	root = x2*(rhs - fx1) / (fx2 - fx1);
	while (abs(error) > err) {
		error = ((2 * (fx1 - fx2) + x2*(dfx1 + dfx2))*pow(root, 3) / pow(x2, 3)
			+ (3 * (fx2 - fx1) - x2*(2 * dfx1 + dfx2))*pow(root, 2) / pow(x2, 2)
			+ dfx1*root + fx1 - rhs)
			/ ((2 * (fx1 - fx2) + x2*(dfx1 + dfx2)) * 3 * pow(root, 2) / pow(x2, 3)
				+ (3 * (fx2 - fx1) - x2*(2 * dfx1 + dfx2)) * 2 * root / pow(x2, 2)
				+ dfx1);
		root -= error;
	}
	return root;
}

