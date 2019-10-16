/*
 * Classical DFT routines. Madelung transformation is used to
 * convert the equation of state and viscosity into a non-linear
 * potential.
 * 
 * A typical use for these routines is:
 *
 * cgrid_zero(pot);
 * dft_classical_add_eos_potential(pot, ...);
 * dft_classical_add_viscous_potential(pot, ...);  // both functions leave pot in Fourier space
 * cgrid_poisson(pot);
 * cgrid_inverse_fft(pot);
 * 
 * Now pot holds the potential corresponding to the EOS and viscous response.
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
 * prm = Parameters (struct dft_visc_param *): rho0 = bulk density, alpha = exponent, viscosity = bulk value.
 *
 */

EXPORT REAL dft_classical_visc_func(REAL rho, void *prm) {

  struct dft_classical_visc_func_param *params = (struct dft_classical_visc_func_param *) prm;

  return POW(rho / params->rho0, params->viscosity_alpha) * params->viscosity;  // viscosity_alpha > 0
}

/*
 * Given the general equation of state (EOS): P = P(rho, T)
 * calculate the corresponding non-linear DFT potential.
 * 
 * This is obtained from: \Delta V = \nabla\cdot\frac{\nabla P}{\rho}.
 *
 * gwf    = Wave function (wf *; input).
 * pot    = Potential output (rgrid *; output).
 * P      = Current pressure grid (rgrid *; input). Overwritte with FFT of P on exit.
 * dens   = Density (rgrid *; input).
 * wrk    = Workspace 1 (rgrid *; input).
 *
 * No return value.
 *
 */

EXPORT void dft_classical_add_eos_potential(wf *gwf, rgrid *pot, rgrid *P, rgrid *density, rgrid *wrk) {

  rgrid_gradient_x(P, wrk);
  rgrid_division_eps(wrk, wrk, density, POISSON_EPS);
  rgrid_gradient_x(wrk, wrk);
  rgrid_sum(pot, pot, wrk);

  rgrid_gradient_y(P, wrk);
  rgrid_division_eps(wrk, wrk, density, POISSON_EPS);
  rgrid_gradient_y(wrk, wrk);
  rgrid_sum(pot, pot, wrk);

  rgrid_gradient_z(P, wrk);
  rgrid_division_eps(wrk, wrk, density, POISSON_EPS);
  rgrid_gradient_z(wrk, wrk);
  rgrid_sum(pot, pot, wrk);
}

/*
 * Viscous potential (from Navier-Stokes assuming irrotational liquid).
 *
 * gwf             = Wave function (wf *; input).
 * pot             = Output for potential (rgrid *; output).
 * shear_visc      = Grid containing the current shear (dynamic) viscosity in atomic units (input).
 * second_visc     = Grid containing the current 2nd viscosity in atomic units (input). Set to NULL if not needed.
 * vx              = X component of the velocity field (rgrid *; input). Overwritten on exit.
 * vy              = Y component of the velocity field (rgrid *; input). Overwritten on exit.
 * vz              = Z component of the velocity field (rgrid *; input). Overwritten on exit.
 * wrk1            = Workspace 1 (rgrid *; input).
 * wrk2            = Workspace 2 (rgrid *; input).
 * wrk3            = Workspace 3 (rgrid *; input).
 *
 * No return value.
 *
 */

EXPORT void dft_classical_add_viscous_potential(wf *gwf, rgrid *pot, rgrid *shear_visc, rgrid *second_visc, rgrid *vx, rgrid *vy, rgrid *vz, rgrid *wrk1, rgrid *wrk2, rgrid *wrk3) {

  /* X component */
  rgrid_gradient_x(vx, wrk1);
  /* d^2/dx^2 */
  rgrid_gradient_x(wrk1, wrk2);
  if(second_visc) rgrid_copy(wrk3, second_visc);
  else rgrid_zero(wrk3);
  rgrid_add_scaled(wrk3, 4.0 / 3.0, shear_visc);
  rgrid_product(vx, wrk2, wrk3);
  /* d^2/dxdy */
  rgrid_fft_gradient_y(wrk1, wrk2);
  if(second_visc) rgrid_copy(wrk3, second_visc);
  else rgrid_zero(wrk3);
  rgrid_add_scaled(wrk3, 1.0 / 3.0, shear_visc);
  rgrid_product(wrk2, wrk2, wrk3);
  rgrid_sum(vx, vx, wrk2);
  /* d^2/dxdz */
  rgrid_fft_gradient_z(wrk1, wrk2);
  if(second_visc) rgrid_copy(wrk3, second_visc);
  else rgrid_zero(wrk3);
  rgrid_add_scaled(wrk3, 1.0 / 3.0, shear_visc);
  rgrid_product(wrk2, wrk2, wrk3);
  rgrid_sum(vx, vx, wrk2);

  /* Y component */
  rgrid_gradient_y(vy, wrk1);
  /* d^2/dy^2 */
  rgrid_gradient_y(wrk1, wrk2);
  if(second_visc) rgrid_copy(wrk3, second_visc);
  else rgrid_zero(wrk3);
  rgrid_add_scaled(wrk3, 4.0 / 3.0, shear_visc);
  rgrid_product(vy, wrk2, wrk3);
  /* d^2/dydx */
  rgrid_fft_gradient_x(wrk1, wrk2);
  if(second_visc) rgrid_copy(wrk3, second_visc);
  else rgrid_zero(wrk3);
  rgrid_add_scaled(wrk3, 1.0 / 3.0, shear_visc);
  rgrid_product(wrk2, wrk2, wrk3);
  rgrid_sum(vy, vy, wrk2);
  /* d^2/dydz */
  rgrid_fft_gradient_z(wrk1, wrk2);
  if(second_visc) rgrid_copy(wrk3, second_visc);
  else rgrid_zero(wrk3);
  rgrid_add_scaled(wrk3, 1.0 / 3.0, shear_visc);
  rgrid_product(wrk2, wrk2, wrk3);
  rgrid_sum(vy, vy, wrk2);

  /* Z component */
  rgrid_gradient_z(vz, wrk1);
  /* d^2/dz^2 */
  rgrid_gradient_z(wrk1, wrk2);
  if(second_visc) rgrid_copy(wrk3, second_visc);
  else rgrid_zero(wrk3);
  rgrid_add_scaled(wrk3, 4.0 / 3.0, shear_visc);
  rgrid_product(vz, wrk2, wrk3);
  /* d^2/dzdy */
  rgrid_fft_gradient_y(wrk1, wrk2);
  if(second_visc) rgrid_copy(wrk3, second_visc);
  else rgrid_zero(wrk3);
  rgrid_add_scaled(wrk3, 1.0 / 3.0, shear_visc);
  rgrid_product(wrk2, wrk2, wrk3);
  rgrid_sum(vz, vz, wrk2);
  /* d^2/dzdx */
  rgrid_fft_gradient_x(wrk1, wrk2);
  if(second_visc) rgrid_copy(wrk3, second_visc);
  else rgrid_zero(wrk3);
  rgrid_add_scaled(wrk3, 1.0 / 3.0, shear_visc);
  rgrid_product(wrk2, wrk2, wrk3);
  rgrid_sum(vz, vz, wrk2);
  
  /* the final divergence */
  rgrid_div(wrk1, vx, vy, vz);
  rgrid_difference(pot, pot, wrk1); // add - sign
}
