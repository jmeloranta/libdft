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
 * @FUNC{dft_initial_vortex_x_n1, "Initial guess for vortex line around x (single quantum)"}
 * @DESC{"Initial guesses leading to creation of vortex line along x. Use with cgrid_map().
         Multiply by the grid by $\sqrt(\rho_0)$ to get bulk asymptotic behavior"}
 * @ARG1{void *na, "User data (not used; can be NULL)"}
 * @ARG2{REAL x, "X coordinate"}
 * @ARG3{REAL y, "Y coordinate"}
 * @ARG4{REAL z, "Z coordinate"}
 * @RVAL{REAL complex, "Vortex function value at (x,y,z)"}
 *
 */

EXPORT REAL complex dft_initial_vortex_x_n1(void *na, REAL x, REAL y, REAL z) {

  REAL d = SQRT(y * y + z * z);

  if(d < R_M) return 0.0;
  return (y + I * z) / d;
}

/*
 * @FUNC{dft_initial_vortex_y_n1, "Initial guess for vortex line around y (single quantum)"}
 * @DESC{"Initial guesses leading to creation of vortex line along y. Use with cgrid_map().
         Multiply by the grid by $\sqrt(\rho_0)$ to get bulk asymptotic behavior"}
 * @ARG1{void *na, "User data (not used; can be NULL)"}
 * @ARG2{REAL x, "X coordinate"}
 * @ARG3{REAL y, "Y coordinate"}
 * @ARG4{REAL z, "Z coordinate"}
 * @RVAL{REAL complex, "Vortex function value at (x,y,z)"}
 *
 */

EXPORT REAL complex dft_initial_vortex_y_n1(void *na, REAL x, REAL y, REAL z) {

  REAL d = SQRT(x * x + z * z);

  if(d < R_M) return 0.0;
  return (x + I * z) / d;
}

/*
 * @FUNC{dft_initial_vortex_z_n1, "Initial guess for vortex line around z (single quantum)"}
 * @DESC{"Initial guesses leading to creation of vortex line along z. Use with cgrid_map().
         Multiply by the grid by $\sqrt(\rho_0)$ to get bulk asymptotic behavior"}
 * @ARG1{void *na, "User data (not used; can be NULL)"}
 * @ARG2{REAL x, "X coordinate"}
 * @ARG3{REAL y, "Y coordinate"}
 * @ARG4{REAL z, "Z coordinate"}
 * @RVAL{REAL complex, "Vortex function value at (x,y,z)"}
 *
 */

EXPORT REAL complex dft_initial_vortex_z_n1(void *na, REAL x, REAL y, REAL z) {

  REAL d = SQRT(x * x + y * y);

  if(d < R_M) return 0.0;
  return (x + I * y) / d;
}

/*
 * @FUNC{dft_initial_vortex_x_n2, "Initial guess for vortex line around x (two quanta)"}
 * @DESC{"Initial guesses leading to creation of vortex line along x. Use with cgrid_map().
         Multiply by the grid by $\sqrt(\rho_0)$ to get bulk asymptotic behavior"}
 * @ARG1{void *na, "User data (not used; can be NULL)"}
 * @ARG2{REAL x, "X coordinate"}
 * @ARG3{REAL y, "Y coordinate"}
 * @ARG4{REAL z, "Z coordinate"}
 * @RVAL{REAL complex, "Vortex function value at (x,y,z)"}
 *
 */

EXPORT REAL complex dft_initial_vortex_x_n2(void *na, REAL x, REAL y, REAL z) {

  REAL y2 = y * y, z2 = z * z;
  REAL d = SQRT(x * x + y * y);
  
  if(d < R_M) return 0.0;
  return ((y2 - z2) + I * 2 * y * z) / (y2 + z2);
}

/*
 * @FUNC{dft_initial_vortex_y_n2, "Initial guess for vortex line around y (two quanta)"}
 * @DESC{"Initial guesses leading to creation of vortex line along y. Use with cgrid_map().
         Multiply by the grid by $\sqrt(\rho_0)$ to get bulk asymptotic behavior"}
 * @ARG1{void *na, "User data (not used; can be NULL)"}
 * @ARG2{REAL x, "X coordinate"}
 * @ARG3{REAL y, "Y coordinate"}
 * @ARG4{REAL z, "Z coordinate"}
 * @RVAL{REAL complex, "Vortex function value at (x,y,z)"}
 *
 */

EXPORT REAL complex dft_initial_vortex_y_n2(void *na, REAL x, REAL y, REAL z) {

  REAL x2 = x * x, z2 = z * z;
  REAL d = SQRT(x * x + y * y);
  
  if(d < R_M) return 0.0;  
  return ((x2 - z2) + I * 2 * x * z) / (x2 + z2);
}

/*
 * @FUNC{dft_initial_vortex_z_n2, "Initial guess for vortex line around z (two quanta)"}
 * @DESC{"Initial guesses leading to creation of vortex line along z. Use with cgrid_map().
         Multiply by the grid by $\sqrt(\rho_0)$ to get bulk asymptotic behavior"}
 * @ARG1{void *na, "User data (not used; can be NULL)"}
 * @ARG2{REAL x, "X coordinate"}
 * @ARG3{REAL y, "Y coordinate"}
 * @ARG4{REAL z, "Z coordinate"}
 * @RVAL{REAL complex, "Vortex function value at (x,y,z)"}
 *
 */

EXPORT REAL complex dft_initial_vortex_z_n2(void *na, REAL x, REAL y, REAL z) {

  REAL x2 = x * x, y2 = y * y;
  REAL d = SQRT(x * x + y * y);
  
  if(d < R_M) return 0.0;  
  return ((x2 - y2) + I * 2 * x * y) / (x2 + y2);
}

/*
 * @FUNC{dft_initial_bubble, "Initial guess for a bubble"}
 * @DESC{"Initial guess for a bubble (can be used with cgrid_map() etc.). Produces a sharp bubble edge. 
          Note that this must be multiplied by $\sqrt{\rho_0}$ to get the bulk density"}
 * @ARG1{void *prm, "Pointer to bubble radius (REAL *)"}
 * @ARG2{REAL x, "X coordinate"}
 * @ARG2{REAL y, "Y coordinate"}
 * @ARG2{REAL z, "Z coordinate"}
 * @RVAL{REAL complex, "Returns wave function (order parameter) value at (x, y, z)"}
 *
 */

EXPORT REAL complex dft_initial_bubble(void *prm, REAL x, REAL y, REAL z) {

  REAL *rad = (REAL *) prm, r;

  r = SQRT(x * x + y * y + z * z);
  if(r < *rad) return 0.0;
  return 1.0;
}
