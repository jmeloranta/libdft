/*
 * Various initial guess functions for the order parameter.
 *
 * These can be mapped onto complex grid using cgrid_map().
 *
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <grid/grid.h>
#include <grid/au.h>
#include "dft.h"
#include "ot.h"

/* Cutoff for vortex core (to avoid NaN) */
#define R_M 0.05

/*
 * Initial guesses leading to creation of vortex line (along x, y, z).
 *
 * Multiply by sqrt(rho0) to get bulk asymptotic behavior.
 *
 */

EXPORT REAL complex dft_initial_vortex_x_n1(void *na, REAL x, REAL y, REAL z) {

  REAL d = SQRT(y * y + z * z);

  if(d < R_M) return 0.0;
  return (y + I * z) / d;
}

EXPORT REAL complex dft_initial_vortex_y_n1(void *na, REAL x, REAL y, REAL z) {

  REAL d = SQRT(x * x + z * z);

  if(d < R_M) return 0.0;
  return (x + I * z) / d;
}

EXPORT REAL complex dft_initial_vortex_z_n1(void *na, REAL x, REAL y, REAL z) {

  REAL d = SQRT(x * x + y * y);

  if(d < R_M) return 0.0;
  return (x + I * y) / d;
}

EXPORT REAL complex dft_initial_vortex_x_n2(void *na, REAL x, REAL y, REAL z) {

  REAL y2 = y * y, z2 = z * z;
  REAL d = SQRT(x * x + y * y);
  
  if(d < R_M) return 0.0;
  return ((y2 - z2) + I * 2 * y * z) / (y2 + z2);
}

EXPORT REAL complex dft_initial_vortex_y_n2(void *na, REAL x, REAL y, REAL z) {

  REAL x2 = x * x, z2 = z * z;
  REAL d = SQRT(x * x + y * y);
  
  if(d < R_M) return 0.0;  
  return ((x2 - z2) + I * 2 * x * z) / (x2 + z2);
}

EXPORT REAL complex dft_initial_vortex_z_n2(void *na, REAL x, REAL y, REAL z) {

  REAL x2 = x * x, y2 = y * y;
  REAL d = SQRT(x * x + y * y);
  
  if(d < R_M) return 0.0;  
  return ((x2 - y2) + I * 2 * x * y) / (x2 + y2);
}

/*
 * Initial guess for a bubble (can be used with cgrid_map() etc.).
 * Sharp bubble edge.
 *
 * prm    = Bubble radius (REAL *).
 * x      = x coordinate (REAL).
 * y      = y coordinate (REAL).
 * z      = z coordinate (REAL).
 *
 * Returns wave function (order parameter) value at (x, y, z).
 *
 * Note: This should be multiplied by sqrt(rho0) for bulk.
 *
 */

EXPORT REAL complex dft_initial_bubble(void *prm, REAL x, REAL y, REAL z) {

  double *rad = (REAL *) prm, r;

  r = SQRT(x * x + y * y + z * z);
  if(r < *rad) return 0.0;
  return 1.0;
}
