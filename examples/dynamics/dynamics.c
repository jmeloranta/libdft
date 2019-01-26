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

  dft_ot_functional *otf;
  rgrid *ext_pot, *rworkspace;
  cgrid *potential_store;
  wf *gwf, *gwfp;
  INT iter;
  REAL offset, mu0, rho0;
  char buf[512];
  grid_timer timer;

#ifdef USE_CUDA
  cuda_enable(1);
#endif

  /* Initialize threads & use wisdom */
  grid_set_fftw_flags(1);    // FFTW_MEASURE
  grid_threads_init(THREADS);
  grid_fft_read_wisdom(NULL);

  /* Allocate wave functions */
  if(!(gwf = grid_wf_alloc(NX, NY, NZ, STEP, DFT_HELIUM_MASS, WF_PERIODIC_BOUNDARY, WF_2ND_ORDER_FFT, "gwf"))) {
    fprintf(stderr, "Cannot allocate gwf.\n");
    exit(1);
  }
  gwfp = grid_wf_clone(gwf, "gwfp");

  /* Allocate OT functional */
  if(!(otf = dft_ot_alloc(DFT_OT_PLAIN | DFT_OT_BACKFLOW | DFT_OT_KC | DFT_OT_HD, gwf, DFT_MIN_SUBSTEPS, DFT_MAX_SUBSTEPS))) {
    fprintf(stderr, "Cannot allocate otf.\n");
    exit(1);
  }
  rho0 = dft_ot_bulk_density_pressurized(otf, PRESSURE);
  mu0 = dft_ot_bulk_chempot_pressurized(otf, PRESSURE);
  printf("mu0 = " FMT_R " K/atom, rho0 = " FMT_R " Angs^-3.\n", mu0 * GRID_AUTOK, rho0 / (GRID_AUTOANG * GRID_AUTOANG * GRID_AUTOANG));

  /* Initialize wave function */
  cgrid_constant(gwf->grid, SQRT(otf->rho0));

  /* Allocate space for external potential */
  ext_pot = rgrid_clone(otf->density, "ext_pot");
  rworkspace = rgrid_clone(otf->density, "rworkspace");
  potential_store = cgrid_clone(gwf->grid, "potential_store"); /* temporary storage */

  /* Generate the initial potential */
  offset = 0.0;
  rgrid_map(ext_pot, pot_func, (void *) &offset);

  /* Step #1: Run 200 iterations using imaginary time for the initial state */
  for (iter = 0; iter < 200; iter++) {

    if(iter == 5) grid_fft_write_wisdom(NULL);

    grid_timer_start(&timer);

    /* Predict-Correct */
    cgrid_zero(potential_store);
    dft_ot_potential(otf, potential_store, gwf);
    cgrid_add(potential_store, -mu0);
    grid_wf_propagate_predict(gwf, gwfp, potential_store, -I * TS / GRID_AUTOFS);
    dft_ot_potential(otf, potential_store, gwfp);
    cgrid_add(potential_store, -mu0);
    cgrid_multiply(potential_store, 0.5);  // Use (current + future) / 2
    grid_wf_propagate_correct(gwf, potential_store, -I * TS / GRID_AUTOFS);
    // Chemical potential included - no need to normalize

    fprintf(stderr, "One iteration = " FMT_R " wall clock seconds.\n", grid_timer_wall_clock_time(&timer));
  }
  /* At this point gwf contains the converged wavefunction */
  grid_wf_density(gwf, rworkspace);
  rgrid_write_grid("initial", rworkspace);

  /* Step #2: Run real time simulation using the final state potential */
  /* Generate the excited potential */
  offset = EXCITED_OFFSET;
  rgrid_map(ext_pot, pot_func, (void *) &offset);

  for (iter = 0; iter < 80000; iter++) {

    grid_timer_start(&timer);

    /* Predict-Correct */
    cgrid_zero(potential_store);
    dft_ot_potential(otf, potential_store, gwf);
    grid_add_real_to_complex_re(potential_store, ext_pot);
    cgrid_add(potential_store, -mu0);
    grid_wf_propagate_predict(gwf, gwfp, potential_store, (TS / 10.0) / GRID_AUTOFS);
    dft_ot_potential(otf, potential_store, gwfp);
    grid_add_real_to_complex_re(potential_store, ext_pot);
    cgrid_add(potential_store, -mu0);
    cgrid_multiply(potential_store, 0.5);  // Use (current + future) / 2
    grid_wf_propagate_correct(gwf, potential_store, (TS / 10.0) / GRID_AUTOFS);

    fprintf(stderr, "One iteration = " FMT_R " wall clock seconds.\n", grid_timer_wall_clock_time(&timer));

    if(!(iter % NTH)) {
      sprintf(buf, "final-" FMT_I, iter);
      grid_wf_density(gwf, rworkspace);
      rgrid_write_grid(buf, rworkspace);
    }
  }
  return 0;
}
