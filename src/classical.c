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
 * Classical DFT potential for an ideal gas.
 *
 * pot     = Resulting potential (rgrid *; output). 
 * density = Gas density (rgrid *; input).
 * rho0    = Bulk density (REAL; input).
 * temp    = Temperature in K (REAL; input).
 * eps     = Low density epsilon (REAL; input).
 *
 * No return value.
 *
 */

EXPORT void dft_classical_ideal_gas(rgrid *pot, rgrid *density, REAL rho0, REAL temp, REAL eps) {

  rgrid_copy(pot, density);
  rgrid_multiply(pot, 1.0 / rho0);
  rgrid_log(pot, pot, eps);
  rgrid_multiply(pot, GRID_AUKB * temp);
}

/*
 * Classical DFT potential for Tait equation of state.
 * 
 * pot     = Resulting potential (rgrid *; output).
 * density = Liquid density (rgrid *; input).
 * rho0    = Bulk density (REAL; input).
 * k0      = Tait EOS parameter K0 (REAL; input).
 * n       = Tait EOS parameter n (REAL; input).
 * wrk     = Workspace (rgrid *; input).
 *
 * No return value.
 *
 * TODO: Not working - or not applicable at low densities?
 *
 */

EXPORT void dft_classical_tait(rgrid *pot, rgrid *density, REAL rho0, REAL k0, REAL n, rgrid *wrk) {

  rgrid_power(wrk, density, 2.0);
  rgrid_laplace(wrk, pot);
  rgrid_multiply(pot, (n - 2.0) / 2.0);

  rgrid_laplace(density, wrk);
  rgrid_product(wrk, wrk, density);
  rgrid_multiply(wrk, 3.0 - n);
  rgrid_sum(pot, pot, wrk);

  rgrid_copy(wrk, density);
  rgrid_multiply(wrk, 1.0 / rho0);
  rgrid_power(wrk, wrk, n - 3.0);
  rgrid_multiply(wrk, k0 * n / (rho0 * rho0 * rho0));

  rgrid_product(pot, pot, wrk);

  rgrid_fft(pot);
  rgrid_poisson(pot);
  rgrid_inverse_fft(pot);
}

/*
 * Viscous potential (from Navier-Stokes assuming irrotational liquid).
 *
 * gwf             = Wave function (wf *; input).
 * pot             = Output for potential (rgrid *; output).
 * shear_visc      = Function for calculating shear (dynamic) viscosity in atomic units (rfunction *; input).
 * wrk1            = Workspace 1 (rgrid *).
 * wrk2            = Workspace 2 (rgrid *).
 * wrk3            = Workspace 3 (rgrid *).
 * wrk4            = Workspace 4 (rgrid *).
 * wrk5            = Workspace 5 (rgrid *).
 * wrk6            = Workspace 6 (rgrid *).
 * wrk7            = Workspace 7 (rgrid *).
 * vx              = Workspace for vx (rgrid *). Overwritten (wrk8).
 * vy              = Workspace for vy (rgrid *). Overwritten (wrk9).
 * vz              = Workspace for vz (rgrid *). Overwritten (wrk10).
 *
 * No return value.
 *
 * Due to the position dependency in the stress tensor, we evaluate the whole thing and do explicit div on that.
 *
 * TODO: Does not include second viscosity.
 *
 */

EXPORT void dft_classical_add_viscous_potential(wf *gwf, rgrid *pot, rfunction *shear_visc, rgrid *wrk1, rgrid *wrk2, rgrid *wrk3, rgrid *wrk4, rgrid *wrk5, rgrid *wrk6, rgrid *wrk7, rgrid *vx, rgrid *vy, rgrid *vz) {

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
  rgrid_function_operate_one(vy, vx, shear_visc);
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

  rgrid_fft(vx);
  rgrid_poisson(vx);
  rgrid_inverse_fft(vx);

  rgrid_difference(pot, pot, vx);  // Include the final - sign here
}
