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
 * rgrid_fft_poisson(pot);
 * rgrid_inverse_fft_norm(pot);
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
 * @FUNC{dft_classical_ideal_gas, "Classical DFT potential for ideal gas"}
 * @DESC{"Calculate classical DFT potential for an ideal gas"}
 * @ARG1{rgrid *pot, "Potential (output)"}
 * @ARG2{rgrid *density, "Gas density (input)"}
 * @ARG3{REAL rho0, "Gas density"}
 * @ARG4{REAL temp, "Temperature (K)"}
 * @ARG5{REAL eps, "Low density epsilon"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT void dft_classical_ideal_gas(rgrid *pot, rgrid *density, REAL rho0, REAL temp, REAL eps) {

  rgrid_copy(pot, density);
  rgrid_multiply(pot, 1.0 / rho0);
  rgrid_log(pot, pot, eps);
  rgrid_multiply(pot, GRID_AUKB * temp);
}

/*
 * @FUNC{dft_classical_tait, "Classical DFT potential for Tait equation of state"}
 * @DESC{"Calculate classical DFT potential for Tait equation of state"}
 * @ARG1{rgrid *pot, "Potential (output)"}
 * @ARG2{rgrid *density, "Liquid density"}
 * @ARG3{REAL rho0, "Bulk density"}
 * @ARG4{REAL k0, "Tait EOS parameter K0"}
 * @ARG5{REAL n, "Tait EOS parameter n"}
 * @ARG6{rgrid *wrk, "Workspace"}
 * @RVAL{void, "No return value"}
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
  rgrid_fft_poisson(pot);
  rgrid_inverse_fft_norm(pot);
}

/*
 * @FUNC{dft_classical_add_viscous_potential, "Add viscous potential"}
 * @DESC{"Viscous potential (from Navier-Stokes assuming irrotational liquid)"}
 * @ARG1{wf *gwf, "Wavefunction"}
 * @ARG2{rgrid *pot, "Potential (output)"}
 * @ARG3{rfunction *shear_visc, "Function for calculating shear (dynamic) viscosity in atomic units"}
 * @ARG4{rgrid *wrk1, "Workspace 1"}
 * @ARG5{rgrid *wrk2, "Workspace 2"}
 * @ARG6{rgrid *wrk3, "Workspace 3"}
 * @ARG7{rgrid *wrk4, "Workspace 4"}
 * @ARG8{rgrid *wrk5, "Workspace 5"}
 * @ARG9{rgrid *wrk6, "Workspace 6"}
 * @ARG10{rgrid *wrk7, "Workspace 7"}
 * @ARG11{rgrid *vx, "Workspace for vx. Overwritten (wrk8)"}
 * @ARG12{rgrid *vy, "Workspace for vy. Overwritten (wrk9)"}
 * @ARG13{rgrid *vz, "Workspace for vz. Overwritten (wrk10)"}
 * @RVAL{void, "No return value"}x
 *
 * Due to the position dependency in the stress tensor, we evaluate the whole thing and do explicit div on that.
 *
 * TODO: Does not include second viscosity.
 *
 */

EXPORT void dft_classical_add_viscous_potential(wf *gwf, rgrid *pot, rfunction *shear_visc, rgrid *wrk1, rgrid *wrk2, rgrid *wrk3, rgrid *wrk4, rgrid *wrk5, rgrid *wrk6, rgrid *wrk7, rgrid *vx, rgrid *vy, rgrid *vz) {

  grid_wf_velocity(gwf, vx, vy, vz, DFT_EPS);
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
  rgrid_inverse_fft_norm(wrk2);

  /* 2 = 4 (symmetry; wrk3) */
  rgrid_fft_gradient_x(vy, wrk3);
  rgrid_fft_gradient_y(vx, wrk1);
  rgrid_fft_sum(wrk3, wrk3, wrk1);
  rgrid_inverse_fft_norm(wrk3);

  /* 3 = 7 (symmetry; wrk4) */
  rgrid_fft_gradient_x(vz, wrk4);
  rgrid_fft_gradient_z(vx, wrk1);
  rgrid_fft_sum(wrk4, wrk4, wrk1);
  rgrid_inverse_fft_norm(wrk4);

  /* 5 (diagonal; wrk5) */
  rgrid_fft_gradient_y(vy, wrk5);
  rgrid_fft_multiply(wrk5, 4.0/3.0);
  rgrid_fft_gradient_x(vx, wrk1);
  rgrid_fft_multiply(wrk1, -2.0/3.0);
  rgrid_fft_sum(wrk5, wrk5, wrk1);
  rgrid_fft_gradient_z(vz, wrk1);
  rgrid_fft_multiply(wrk1, -2.0/3.0);
  rgrid_fft_sum(wrk5, wrk5, wrk1);
  rgrid_inverse_fft_norm(wrk5);

  /* 6 = 8 (symmetryl wrk6) */
  rgrid_fft_gradient_y(vz, wrk6);
  rgrid_fft_gradient_z(vy, wrk1);
  rgrid_fft_sum(wrk6, wrk6, wrk1);
  rgrid_inverse_fft_norm(wrk6);

  /* 9 = (diagonal; wrk7) */
  rgrid_fft_gradient_z(vz, wrk7);
  rgrid_fft_multiply(wrk7, 4.0/3.0);
  rgrid_fft_gradient_x(vx, wrk1);
  rgrid_fft_multiply(wrk1, -2.0/3.0);
  rgrid_fft_sum(wrk7, wrk7, wrk1);
  rgrid_fft_gradient_y(vy, wrk1);
  rgrid_fft_multiply(wrk1, -2.0/3.0);
  rgrid_fft_sum(wrk7, wrk7, wrk1);
  rgrid_inverse_fft_norm(wrk7);

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
  rgrid_fft_poisson(vx);
  rgrid_inverse_fft_norm(vx);

  rgrid_difference(pot, pot, vx);  // Include the final - sign here
}

/*
 * @FUNC{dft_classical_viscosity, "Evaluate density dependent viscosity"}
 * @DESC{"Function defining the density dependent viscosity"}
 * @ARG1{REAL rho, "Density (input)"}
 * @ARG2{void *prm, "Parameters [0]: rho0 = bulk density, [1]: alpha = exponent, [2]: bulk value for viscosity"}
 * @RVAL{REAL, "Viscosity for given rho"}
 *
 */

EXPORT REAL dft_classical_viscosity(REAL rho, void *prm) {

  REAL rho0 = ((REAL *) prm)[0];
  REAL alpha = ((REAL *) prm)[1];
  REAL visc = ((REAL *) prm)[2];

  return POW(rho / rho0, alpha) * visc;
}
