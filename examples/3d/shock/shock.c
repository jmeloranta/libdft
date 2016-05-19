/*
 * Impurity atom in superfluid helium (no zero-point).
 * Optimize the liquid structure around a given initial
 * potential and then run dynamics on a given final potential.
 *
 * All input in a.u. except the time step, which is fs.
 *
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <grid/grid.h>
#include <grid/au.h>
#include <dft/dft.h>
#include <dft/ot.h>

#define MAXITER 160000
#define TS 1.0 /* fs */
#define OUTPUT 1000

#define HELIUM_MASS (4.002602 / GRID_AUTOAMU)

#define NX 64
#define NY 64
#define NZ (2*8192)
#define STEP 0.25

// #define FUNC (DFT_OT_PLAIN + DFT_OT_KC + DFT_OT_BACKFLOW)
#define FUNC (DFT_OT_PLAIN)
//#define FUNC DFT_GP

#define A0 (3.8003E5 / GRID_AUTOK)
#define A1 (4.6245 * GRID_AUTOANG)
#define OFFSET 0.1

#define IMAG_ITER 1000

#define GAUSSIAN

#define WIDTH 4.0
#define AMP ( 1.5 * dft_driver_otf->rho0)

double wall_pot(void *arg, double x, double y, double z) {

  double offset = *((double *) arg);
  
#ifdef GAUSSIAN
  return 0.0;
#else
  return A0 * exp(-A1 * (fabs(z) - offset)); 
#endif
}

double complex gauss(void *arg, double x, double y, double z) {

  double inv_width = *((double *) arg);
  double norm = 0.5 * M_2_SQRTPI * inv_width;

  norm = norm;
  return norm * cexp(-z * z * inv_width * inv_width);
}

int main(int argc, char **argv) {

  rgrid3d *ext_pot, *rworkspace;
  cgrid3d *potential_store;
  wf3d *gwf, *gwfp;
  long iter;
  double energy, natoms, offset, inv_width;
  char buf[512];

  /* Setup DFT driver parameters (256 x 256 x 256 grid) */
  dft_driver_setup_grid(NX, NY, NZ, STEP /* Bohr */, 0 /* threads */);
  /* Plain Orsay-Trento in imaginary time */
  dft_driver_setup_model(FUNC, DFT_DRIVER_IMAG_TIME, 0.0);
  /* No absorbing boundary */
  dft_driver_setup_boundaries(DFT_DRIVER_BOUNDARY_REGULAR, 2.0);
  /* Normalization condition */
  dft_driver_setup_normalization(DFT_DRIVER_NORMALIZE_BULK, 0, 3.0, 10);

  /* Initialize the DFT driver */
  dft_driver_initialize();

  /* Allocate space for external potential */
  ext_pot = dft_driver_alloc_rgrid();
  rworkspace = dft_driver_alloc_rgrid();
  potential_store = dft_driver_alloc_cgrid(); /* temporary storage */
  /* Read initial external potential from file */
  offset = 0.0;
  rgrid3d_map(ext_pot, wall_pot, &offset);

  /* Allocate space for wavefunctions (initialized to sqrt(rho0)) */
  gwf = dft_driver_alloc_wavefunction(HELIUM_MASS); /* helium wavefunction */
  gwfp = dft_driver_alloc_wavefunction(HELIUM_MASS);/* temp. wavefunction */
#ifdef GAUSSIAN
  inv_width = 1.0 / WIDTH;
  cgrid3d_map(gwf->grid, gauss, &inv_width);  
  cgrid3d_multiply(gwf->grid, AMP);
  cgrid3d_add(gwf->grid, dft_driver_otf->rho0);
  cgrid3d_power(gwf->grid, gwf->grid, 0.5);
#else
  cgrid3d_constant(gwf->grid, sqrt(dft_driver_otf->rho0));
#endif
  
#ifndef GAUSSIAN
  /* Step #1: Run 200 iterations using imaginary time for the initial state */
  for (iter = 0; iter < IMAG_ITER; iter++) {
    dft_driver_propagate_predict(DFT_DRIVER_PROPAGATE_HELIUM, ext_pot, gwf, gwfp, potential_store, TS, iter);
    dft_driver_propagate_correct(DFT_DRIVER_PROPAGATE_HELIUM, ext_pot, gwf, gwfp, potential_store, TS, iter);
  }
  /* At this point gwf contains the converged wavefunction */
  grid3d_wf_density(gwf, rworkspace);
  dft_driver_write_density(rworkspace, "initial");
#endif
  
  /* Step #2: Run real time simulation using the final state potential */
  dft_driver_setup_model(FUNC, DFT_DRIVER_REAL_TIME, 0.0);

#ifndef GAUSSIAN
  // Excited potential
  offset += OFFSET;
  rgrid3d_map(ext_pot, wall_pot, &offset);
#endif
  
  for (iter = 0; iter < MAXITER; iter++) {
    dft_driver_propagate_predict(DFT_DRIVER_PROPAGATE_HELIUM, ext_pot, gwf, gwfp, potential_store, TS, iter);
    dft_driver_propagate_correct(DFT_DRIVER_PROPAGATE_HELIUM, ext_pot, gwf, gwfp, potential_store, TS, iter);
    if(!(iter % OUTPUT)) {
      sprintf(buf, "final-%ld", iter);
      grid3d_wf_density(gwf, rworkspace);
      dft_driver_write_density(rworkspace, buf);
    }
  }
}
