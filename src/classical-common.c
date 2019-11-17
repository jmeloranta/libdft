/*
 * Common functions that are useful for classical DFT.
 *
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <grid/grid.h>
#include <grid/au.h>
#include "dft.h"
#include "ot.h"
#include "git-version.h"

/*
 * Function defining density dependent viscosity.
 *
 * rho = density (REAL; input).
 * prm = Parameters 0: rho0 = bulk density, 1: alpha = exponent, 2: bulk value for viscosity.
 *
 */

EXPORT REAL dft_classical_viscosity(REAL rho, void *prm) {

  REAL rho0 = ((REAL *) prm)[0];
  REAL alpha = ((REAL *) prm)[1];
  REAL visc = ((REAL *) prm)[2];

  return POW(rho / rho0, alpha) * visc;
}
