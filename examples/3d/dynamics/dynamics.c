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

#define TS 50.0 /* fs */
#define NX 256
#define NY 256
#define NZ 256
#define STEP 0.5

#define INITIAL_POT_X "initial_pot.x"
#define INITIAL_POT_Y "initial_pot.y"
#define INITIAL_POT_Z "initial_pot.z"

#define FINAL_POT_X "final_pot.x"
#define FINAL_POT_Y "final_pot.y"
#define FINAL_POT_Z "final_pot.z"

#define HELIUM_MASS (4.002602 / GRID_AUTOAMU)

int main(int argc, char **argv) {

  rgrid3d *ext_pot, *rworkspace;
  cgrid3d *potential_store;
  wf3d *gwf, *gwfp;
  long iter;
  double energy, natoms;
  char buf[512];

  /* Setup DFT driver parameters (256 x 256 x 256 grid) */
  dft_driver_setup_grid(NX, NY, NZ, STEP, 0);
  /* Plain Orsay-Trento in imaginary time */
  dft_driver_setup_model(DFT_OT_PLAIN, DFT_DRIVER_IMAG_TIME, 0.0);
  /* No absorbing boundary */
  dft_driver_setup_boundary_type(DFT_DRIVER_BOUNDARY_REGULAR, 0.0, 0.0, 0.0);
  /* Normalization condition */
  dft_driver_setup_normalization(DFT_DRIVER_NORMALIZE_BULK, 0, 3.0, 10);

  /* Initialize the DFT driver */
  dft_driver_initialize();

  /* Allocate space for external potential */
  ext_pot = dft_driver_alloc_rgrid();
  rworkspace = dft_driver_alloc_rgrid();
  potential_store = dft_driver_alloc_cgrid(); /* temporary storage */
  /* Read initial external potential from file */
  dft_common_potential_map(DFT_DRIVER_AVERAGE_NONE, INITIAL_POT_X, INITIAL_POT_Y, INITIAL_POT_Z, ext_pot);

  /* Allocate space for wavefunctions (initialized to sqrt(rho0)) */
  gwf = dft_driver_alloc_wavefunction(HELIUM_MASS); /* helium wavefunction */
  gwfp = dft_driver_alloc_wavefunction(HELIUM_MASS);/* temp. wavefunction */
  cgrid3d_constant(gwf->grid, sqrt(dft_driver_otf->rho0));

  /* Step #1: Run 200 iterations using imaginary time for the initial state */
  for (iter = 0; iter < 200; iter++) {
    dft_driver_propagate_predict(DFT_DRIVER_PROPAGATE_HELIUM, ext_pot, gwf, gwfp, potential_store, TS, iter);
    dft_driver_propagate_correct(DFT_DRIVER_PROPAGATE_HELIUM, ext_pot, gwf, gwfp, potential_store, TS, iter);
  }
  /* At this point gwf contains the converged wavefunction */
  grid3d_wf_density(gwf, rworkspace);
  dft_driver_write_density(rworkspace, "initial");

  /* Step #2: Run real time simulation using the final state potential */
  dft_driver_setup_model(DFT_OT_PLAIN + DFT_OT_KC + DFT_OT_BACKFLOW, DFT_DRIVER_REAL_TIME, 0.0);
  dft_driver_setup_boundary_type(DFT_DRIVER_BOUNDARY_ITIME, STEP * (double) (NX/2 - 20), 0.1, 0.1);
  dft_common_potential_map(DFT_DRIVER_AVERAGE_NONE, FINAL_POT_X, FINAL_POT_Y, FINAL_POT_Z, ext_pot);

  for (iter = 0; iter < 8000; iter++) {
    dft_driver_propagate_predict(DFT_DRIVER_PROPAGATE_HELIUM, ext_pot, gwf, gwfp, potential_store, TS/10.0, iter);
    dft_driver_propagate_correct(DFT_DRIVER_PROPAGATE_HELIUM, ext_pot, gwf, gwfp, potential_store, TS/10.0, iter);
    if(!(iter % 10)) {
      sprintf(buf, "final-%ld", iter);
      grid3d_wf_density(gwf, rworkspace);
      dft_driver_write_density(rworkspace, buf);
    }
  }
}
