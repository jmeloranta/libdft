/*
 * Classical DFT header.
 *
 */

/* Epsilon for dividing by rho */
#define POISSON_EPS 1E-8

struct dft_classical_visc_func_param {
  REAL rho0;
  REAL viscosity;
  REAL viscosity_alpha;
};
