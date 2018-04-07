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
#define NX 128
#define NY 128
#define NZ 128
#define STEP 2.0
#define NTH 100

#define ABS_WIDTH 45.0      /* Absorbing boundary width */

#define PRESSURE (1.0 / GRID_AUTOBAR)

#define THREADS 0

#define HELIUM_MASS (4.002602 / GRID_AUTOAMU)

#define A0 (3.8003E5 / GRID_AUTOK)
#define A1 (1.6245 * GRID_AUTOANG)
#define A2 0.0
#define A3 0.0
#define A4 0.0
#define A5 0.0
#define RMIN 2.0
#define RADD (-6.0)

#define EXCITED_OFFSET 5.0

REAL pot_func(void *asd, REAL x, REAL y, REAL z) {

  REAL r, r2, r4, r6, r8, r10, tmp, offset;

  offset = *((REAL *) asd);
  r = SQRT(x * x + y * y + z * z);
  if(r < RMIN) r = RMIN;
  r += RADD + offset;

  r2 = r * r;
  r4 = r2 * r2;
  r6 = r4 * r2;
  r8 = r6 * r2;
  r10 = r8 * r2;
  tmp = A0 * EXP(-A1 * r) - A2 / r4 - A3 / r6 - A4 / r8 - A5 / r10;
  return tmp;
}

int main(int argc, char **argv) {

  rgrid3d *ext_pot, *rworkspace;
  cgrid3d *potential_store;
  wf3d *gwf, *gwfp;
  INT iter;
  REAL offset, mu0, rho0;
  char buf[512];

//  cuda_enable(1);

  /* Setup DFT driver parameters (grid) */
  dft_driver_setup_grid(NX, NY, NZ, STEP, THREADS);
  /* Plain Orsay-Trento in imaginary time */
  dft_driver_setup_model(DFT_OT_PLAIN, DFT_DRIVER_IMAG_TIME, 0.0);
  /* No absorbing boundary */
  dft_driver_setup_boundary_type(DFT_DRIVER_BOUNDARY_REGULAR, 0.0, 0.0, 0.0);
  /* Normalization condition */
  dft_driver_setup_normalization(DFT_DRIVER_DONT_NORMALIZE, 0, 3.0, 10);

  /* Initialize the DFT driver */
  dft_driver_initialize();

  /* density */
  rho0 = dft_ot_bulk_density_pressurized(dft_driver_otf, PRESSURE);
  dft_driver_otf->rho0 = rho0;
  /* chemical potential */
  mu0 = dft_ot_bulk_chempot_pressurized(dft_driver_otf, PRESSURE);

  /* Allocate space for external potential */
  ext_pot = dft_driver_alloc_rgrid("ext_pot");
  rworkspace = dft_driver_alloc_rgrid("rworkspace");
  potential_store = dft_driver_alloc_cgrid("potential_store"); /* temporary storage */
  /* Generate the initial potential */
  offset = 0.0;
  rgrid3d_map(ext_pot, pot_func, (void *) &offset);
  rgrid3d_add(ext_pot, -mu0); /* Add the chemical potential */

  /* Allocate space for wavefunctions (initialized to SQRT(rho0)) */
  gwf = dft_driver_alloc_wavefunction(HELIUM_MASS, "gwf"); /* helium wavefunction */
  gwfp = dft_driver_alloc_wavefunction(HELIUM_MASS, "gwfp");/* temp. wavefunction */
  cgrid3d_constant(gwf->grid, SQRT(dft_driver_otf->rho0));

  /* Step #1: Run 200 iterations using imaginary time for the initial state */
  for (iter = 0; iter < 200; iter++) {
    dft_driver_propagate_predict(DFT_DRIVER_PROPAGATE_HELIUM, ext_pot, gwf, gwfp, potential_store, TS, iter);
    dft_driver_propagate_correct(DFT_DRIVER_PROPAGATE_HELIUM, ext_pot, gwf, gwfp, potential_store, TS, iter);
  }
  /* At this point gwf contains the converged wavefunction */
  grid3d_wf_density(gwf, rworkspace);
  dft_driver_write_density(rworkspace, "initial");

  /* Step #2: Run real time simulation using the final state potential */
  dft_driver_setup_model(DFT_OT_PLAIN, DFT_DRIVER_REAL_TIME, 0.0);
  dft_driver_setup_boundary_type(DFT_DRIVER_BOUNDARY_ITIME, ABS_WIDTH, ABS_WIDTH, ABS_WIDTH);
  /* Generate the excited potential */
  offset = EXCITED_OFFSET;
  rgrid3d_map(ext_pot, pot_func, (void *) &offset);
  rgrid3d_add(ext_pot, -mu0); /* Add the chemical potential */

  for (iter = 0; iter < 80000; iter++) {
    dft_driver_propagate_predict(DFT_DRIVER_PROPAGATE_HELIUM, ext_pot, gwf, gwfp, potential_store, TS/10.0, iter);
    dft_driver_propagate_correct(DFT_DRIVER_PROPAGATE_HELIUM, ext_pot, gwf, gwfp, potential_store, TS/10.0, iter);
    if(!(iter % NTH)) {
      sprintf(buf, "final-" FMT_I, iter);
      grid3d_wf_density(gwf, rworkspace);
      dft_driver_write_density(rworkspace, buf);
    }
  }
  return 0;
}
