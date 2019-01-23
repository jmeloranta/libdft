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
 * Vortices using Feynman-Onsager ansatz along x, y, z.
 *
 */

EXPORT REAL dft_initial_vortex_x(void *param, REAL x, REAL y, REAL z) {

  REAL rp2 = y * y + z * z;
  wf *gwf = (wf *) param;

  if(rp2 < R_M * R_M) rp2 = R_M * R_M;
  return 1.0 / (2.0 * gwf->mass * rp2);
}

EXPORT REAL dft_initial_vortex_y(void *param, REAL x, REAL y, REAL z) {

  REAL rp2 = x * x + z * z;
  wf *gwf = (wf *) param;

  if(rp2 < R_M * R_M) rp2 = R_M * R_M;
  return 1.0 / (2.0 * gwf->mass * rp2);
}

EXPORT REAL dft_initial_vortex_z(void *param, REAL x, REAL y, REAL z) {

  REAL rp2 = x * x + y * y;
  wf *gwf = (wf *) param;

  if(rp2 < R_M * R_M) rp2 = R_M * R_M;
  return 1.0 / (2.0 * gwf->mass * rp2);
}


