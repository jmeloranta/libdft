/*
 * Create a vortex ring in superfluid helium centered around Z = 0.
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

#define TS 30.0 /* fs */
#define NX 256
#define NY 256
#define NZ 256
#define STEP 1.0
#define NTH 100
#define THREADS 0

#define RING_RADIUS 25.0

#define PRESSURE (0.0 / GRID_AUTOBAR)

REAL rho0;

/* vortex ring initial guess */
REAL complex vring(void *asd, REAL x, REAL y, REAL z) {

  REAL xs = SQRT(x * x + y * y) - RING_RADIUS;
  REAL ys = z;
  REAL angle = ATAN2(ys,xs), r = SQRT(xs*xs + ys*ys);
 
  return (1.0 - EXP(-r)) * SQRT(rho0) * CEXP(I * angle);
}

int main(int argc, char **argv) {

  dft_ot_functional *otf;
  cgrid *potential_store;
  rgrid *rworkspace;
  wf *gwf, *gwfp;
  INT iter;
  REAL mu0, kin, pot, n;
  char buf[512];
  grid_timer timer;

#ifdef USE_CUDA
#define NGPUS 1
int gpus[] = {0};
  cuda_enable(1, NGPUS, gpus);  // enable CUDA ?
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
  if(!(otf = dft_ot_alloc(DFT_OT_PLAIN, gwf, DFT_MIN_SUBSTEPS, DFT_MAX_SUBSTEPS))) {
    fprintf(stderr, "Cannot allocate otf.\n");
    exit(1);
  }
  rho0 = dft_ot_bulk_density_pressurized(otf, PRESSURE);
  mu0 = dft_ot_bulk_chempot_pressurized(otf, PRESSURE);
  printf("mu0 = " FMT_R " K/atom, rho0 = " FMT_R " Angs^-3.\n", mu0 * GRID_AUTOK, rho0 / (GRID_AUTOANG * GRID_AUTOANG * GRID_AUTOANG));

  /* Allocate space for external potential */
  potential_store = cgrid_clone(gwf->grid, "potential_store"); /* temporary storage */
  rworkspace = rgrid_clone(otf->density, "rworkspace"); /* temporary storage */
 
  /* setup initial guess for vortex ring */
  grid_wf_map(gwf, vring, NULL);

  for (iter = 1; iter < 800000; iter++) {
    if(iter == 1 || !(iter % NTH)) {
      sprintf(buf, "vring-" FMT_I, iter);
      cgrid_write_grid(buf, gwf->grid);
      kin = grid_wf_energy(gwf, NULL);            /* Kinetic energy for gwf */
      dft_ot_energy_density(otf, rworkspace, gwf);
      pot = rgrid_integral(rworkspace);
      n = grid_wf_norm(gwf);
      printf("Iteration " FMT_I " helium natoms    = " FMT_R " particles.\n", iter, n);   /* Energy / particle in K */
      printf("Iteration " FMT_I " helium kinetic   = " FMT_R "\n", iter, kin * GRID_AUTOK);  /* Print result in K */
      printf("Iteration " FMT_I " helium potential = " FMT_R "\n", iter, pot * GRID_AUTOK);  /* Print result in K */
      printf("Iteration " FMT_I " helium energy    = " FMT_R "\n", iter, (kin + pot) * GRID_AUTOK);  /* Print result in K */
      fflush(stdout);
    }

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

    printf("Iteration " FMT_I " - Wall clock time = " FMT_R " seconds.\n", iter, grid_timer_wall_clock_time(&timer));
  }
  return 0;
}
