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
 * rgrid_fft(pot);
 * rgrid_poisson(pot);
 * rgrid_inverse_fft(pot);
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
 * P      = Pressure from the EOS (rgrid *; input). One dimensional grid relating density to pressure.
 * dens   = Density (rgrid *; input).
 * wrk1   = Workspace 1 (rgrid *; input).
 * wrk2   = Workspace 2 (rgrid *; input).
 *
 * No return value.
 *
 */

EXPORT void dft_classical_add_eos_potential(wf *gwf, rgrid *pot, rgrid *P, rgrid *density, rgrid *wrk1, rgrid *wrk2) {

  if(P->nx != 1 || P->ny != 1) {
    fprintf(stderr, "libdft: Pressure must be one-dimensional grid.\n");
    abort();
  }

  rgrid_operate_one(wrk1, density, P, params);
  rgrid_fft(wrk1);

  rgrid_fft_gradient_x(wrk1, wrk2);
  rgrid_inverse_fft(wrk2);
  rgrid_division_eps(wrk2, wrk2, density, POISSON_EPS);
  rgrid_fft(wrk2);
  rgrid_fft_gradient_x(wrk2, wrk2);
  rgrid_inverse_fft(wrk2);
  rgrid_sum(pot, pot, wrk2);

  rgrid_fft_gradient_y(wrk1, wrk2);
  rgrid_inverse_fft(wrk2);
  rgrid_division_eps(wrk2, wrk2, density, POISSON_EPS);
  rgrid_fft(wrk2);
  rgrid_fft_gradient_y(wrk2, wrk2);
  rgrid_inverse_fft(wrk2);
  rgrid_sum(pot, pot, wrk2);

  rgrid_fft_gradient_z(wrk1, wrk2);
  rgrid_inverse_fft(wrk2);
  rgrid_division_eps(wrk2, wrk2, density, POISSON_EPS);
  rgrid_fft(wrk2);
  rgrid_fft_gradient_z(wrk2, wrk2);
  rgrid_inverse_fft(wrk2);
  rgrid_sum(pot, pot, wrk2);

  rgrid_fft_space(wrk1, 0);
}

/*
 * Viscous potential (from Navier-Stokes assuming irrotational liquid).
 *
 * gwf             = Wave function (wf *; input).
 * pot             = Output for potential (rgrid *; output).
 * shear_visc      = Function for calculating shear (dynamic) viscosity in atomic units (input).
 * shear_params    = Shear viscosity function parameters (void *; input).
 * wrk1            = Workspace 1 (rgrid *).
 * wrk2            = Workspace 2 (rgrid *).
 * wrk3            = Workspace 3 (rgrid *).
 * wrk4            = Workspace 4 (rgrid *).
 * wrk5            = Workspace 5 (rgrid *).
 * wrk6            = Workspace 6 (rgrid *).
 * wrk7            = Workspace 7 (rgrid *).
 * vx              = Workspace for vx (rgrid *). Overwritten.
 * vy              = Workspace for vy (rgrid *). Overwritten.
 * vz              = Workspace for vz (rgrid *). Overwritten.
 *
 * No return value.
 *
 * Due to the position dependency in the stress tensor, we evaluate the whole thing and do explicit div on that.
 *
 * TODO: Does not include second viscosity.
 *
 */

EXPORT void dft_classical_add_viscous_potential(wf *gwf, rgrid *pot, REAL (*shear_visc)(REAL, void *), void *shear_params, rgrid *wrk1, rgrid *wrk2, rgrid *wrk3, rgrid *wrk4, rgrid *wrk5, rgrid *wrk6, rgrid *wrk7, rgrid *vx, rgrid *vy, rgrid *vz) {

  grid_wf_velocity(gwf, vx, vy, vz);
  rgrid_fft(vx);
  rgrid_fft(vy);
  rgrid_fft(vz);
  
  /* Stress tensor elements (without viscosity) */
  /* 1 (diagonal; wrk2) */
  rgrid_fft_gradient_x(vx, wrk2);
  rgrid_fft_multiply(wrk2, 4.0/3.0);
  rgrid_fft_gradient_y(vy, wrk1);
  rgrid_fft_multiply(wrk1, -2.0/3.0);
  rgrid_fft_sum(wrk2, wrk2, wrk1);
  rgrid_fft_gradient_z(vz, wrk1);
  rgrid_fft_multiply(wrk1, -2.0/3.0);
  rgrid_fft_sum(wrk2, wrk2, wrk1);
  rgrid_inverse_fft(wrk2);

  /* 2 = 4 (symmetry; wrk3) */
  rgrid_fft_gradient_x(vy, wrk3);
  rgrid_fft_gradient_y(vx, wrk1);
  rgrid_fft_sum(wrk3, wrk3, wrk1);
  rgrid_inverse_fft(wrk3);

  /* 3 = 7 (symmetry; wrk4) */
  rgrid_fft_gradient_x(vz, wrk4);
  rgrid_fft_gradient_z(vx, wrk1);
  rgrid_fft_sum(wrk4, wrk4, wrk1);
  rgrid_inverse_fft(wrk4);

  /* 5 (diagonal; wrk5) */
  rgrid_fft_gradient_y(vy, wrk5);
  rgrid_fft_multiply(wrk5, 4.0/3.0);
  rgrid_fft_gradient_x(vx, wrk1);
  rgrid_fft_multiply(wrk1, -2.0/3.0);
  rgrid_fft_sum(wrk5, wrk5, wrk1);
  rgrid_fft_gradient_z(vz, wrk1);
  rgrid_fft_multiply(wrk1, -2.0/3.0);
  rgrid_fft_sum(wrk5, wrk5, wrk1);
  rgrid_inverse_fft(wrk5);

  /* 6 = 8 (symmetryl wrk6) */
  rgrid_fft_gradient_y(vz, wrk6);
  rgrid_fft_gradient_z(vy, wrk1);
  rgrid_fft_sum(wrk6, wrk6, wrk1);
  rgrid_inverse_fft(wrk6);

  /* 9 = (diagonal; wrk7) */
  rgrid_fft_gradient_z(vz, wrk7);
  rgrid_fft_multiply(wrk7, 4.0/3.0);
  rgrid_fft_gradient_x(vx, wrk1);
  rgrid_fft_multiply(wrk1, -2.0/3.0);
  rgrid_fft_sum(wrk7, wrk7, wrk1);
  rgrid_fft_gradient_y(vy, wrk1);
  rgrid_fft_multiply(wrk1, -2.0/3.0);
  rgrid_fft_sum(wrk7, wrk7, wrk1);
  rgrid_inverse_fft(wrk7);

  /* factor in viscosity (temp vx, vy = wrk8) */
  rgrid_fft_space(vx, 0);
  rgrid_fft_space(vy, 0);
  grid_wf_density(gwf, vx);
  rgrid_operate_one(vy, vx, shear_visc, &shear_params);
  rgrid_product(wrk2, wrk2, vy);
  rgrid_product(wrk3, wrk3, vy);
  rgrid_product(wrk4, wrk4, vy);
  rgrid_product(wrk5, wrk5, vy);
  rgrid_product(wrk6, wrk6, vy);
  rgrid_product(wrk7, wrk7, vy);

  /* x component of divergence (wrk1) */
  rgrid_div(wrk1, wrk2, wrk3, wrk4); // (d/dx) 1(wrk2) + (d/dy) 2(wrk3) + (d/dz) 3(wrk4)
  /* y component of divergence (wrk2) */
  rgrid_div(wrk2, wrk3, wrk5, wrk6); // (d/dx) 2(wrk3) + (d/dy) 5(wrk5) + (d/dz) 6(wrk6)
  /* x component of divergence (wrk3) */
  rgrid_div(wrk3, wrk4, wrk6, wrk7); // (d/dx) 3(wrk4) + (d/dy) 6(wrk6) + (d/dz) 9(wrk7)

  /* divide by rho (vx contains density) */
  rgrid_division_eps(wrk1, wrk1, vx, POISSON_EPS);
  rgrid_division_eps(wrk2, wrk2, vx, POISSON_EPS);
  rgrid_division_eps(wrk3, wrk3, vx, POISSON_EPS);
 
  /* the final divergence */
  rgrid_div(vx, wrk1, wrk2, wrk3);

  rgrid_difference(pot, pot, vx);  // Include the final - sign here
}
