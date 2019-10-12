/*
 * Impurity atom in superfluid helium (no zero-point).
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

#define NX 128
#define NY 128
#define NZ 128
#define STEP 1.0
#define TS (1.0 / GRID_AUTOFS)

/* Bulk density: particles / unit volume */
#define RHO0 (0.0218360 * GRID_AUTOANG * GRID_AUTOANG * GRID_AUTOANG)

/* Temperature (K) */
#define TEMP 200.0

/* Mass */
#define MASS (4.002602 / GRID_AUTOAMU)

#define MAXITER 10000000
#define NTH 50

#define THREADS 0

/* Bubble parameters using exponential repulsion (approx. electron bubble) - RADD = 19.0 */
#define A0 (3.8003E5 / GRID_AUTOK)
#define A1 (1.6245 * GRID_AUTOANG)
#define A2 0.0
#define A3 0.0
#define A4 0.0
#define A5 0.0
#define RMIN 6.0
#define RADD 6.0  // was 6.0

REAL eos(REAL rho, void *params) {

//  REAL rho0 = ((REAL *) params)[0];
  REAL temp = ((REAL *) params)[1];

  return rho * GRID_AUKB * temp;
}

REAL pot_func(void *NA, REAL x, REAL y, REAL z) {

  REAL r, r2, r4, r6, r8, r10;

  r = SQRT(x * x + y * y + z * z);
  r -= RADD;
#ifdef RMIN
  if(r < RMIN) r = RMIN;
#endif

  r2 = r * r;
  r4 = r2 * r2;
  r6 = r4 * r2;
  r8 = r6 * r2;
  r10 = r8 * r2;
  return A0 * EXP(-A1 * r) 
#ifdef RMIN
   - A2 / r4 - A3 / r6 - A4 / r8 - A5 / r10
#endif
  ;
}

int main(int argc, char **argv) {

  cgrid *potential_store;
  rgrid *ext_pot, *cla_pot, *density, *wrk1, *wrk2, *wrk3;
  wf *gwf;
  INT iter;
  REAL natoms;
  grid_timer timer;

#undef USE_CUDA
#ifdef USE_CUDA
#define NGPUS 1
int gpus[] = {6};
  cuda_enable(1, NGPUS, gpus);
#endif

  /* Initialize threads & use wisdom */
  grid_set_fftw_flags(1);    // FFTW_MEASURE
  grid_threads_init(THREADS);
  grid_fft_read_wisdom(NULL);

  /* Allocate wave function */
  if(!(gwf = grid_wf_alloc(NX, NY, NZ, STEP, MASS, WF_PERIODIC_BOUNDARY, WF_2ND_ORDER_FFT, "gwf"))) {
    fprintf(stderr, "Cannot allocate gwf.\n");
    exit(1);
  }
  if(!(density = rgrid_alloc(NX, NY, NZ, STEP, RGRID_PERIODIC_BOUNDARY, NULL, "density"))) {
    fprintf(stderr, "Cannot allocate gwf.\n");
    exit(1);
  }

  /* Allocate space for external potential */
  ext_pot = rgrid_clone(density, "ext_pot");
  cla_pot = rgrid_clone(density, "cla_pot");
  wrk1 = rgrid_clone(density, "wrk1");
  wrk2 = rgrid_clone(density, "wrk2");
  wrk3 = rgrid_clone(density, "wrk3");
  potential_store = cgrid_clone(gwf->grid, "potential_store");

  /* Read external potential from file */
  rgrid_map(ext_pot, pot_func, NULL);

  grid_wf_constant(gwf, SQRT(RHO0));

  /* Run 200 iterations using imaginary time (10 fs time step) */

    for (iter = 0; iter < MAXITER; iter++) {

    if(iter == 5) grid_fft_write_wisdom(NULL);

    grid_timer_start(&timer);

    /* Predict-Correct */
    grid_real_to_complex_re(potential_store, ext_pot);
    grid_wf_density(gwf, density);
    dft_common_eos_pot(cla_pot, eos, density, RHO0, TEMP, wrk1, wrk2, wrk3);
    grid_add_real_to_complex_re(potential_store, cla_pot);

    grid_wf_propagate(gwf, potential_store, -I * TS);

    printf("Iteration " FMT_I " - Wall clock time = " FMT_R " seconds.\n", iter, grid_timer_wall_clock_time(&timer));

    if(!(iter % NTH)) {
      char buf[512];
      sprintf(buf, "output-" FMT_I, iter);
      grid_wf_density(gwf, density);
      rgrid_write_grid(buf, density);
      natoms = grid_wf_norm(gwf);
      printf("Number of He atoms is " FMT_R ".\n", natoms);
      fflush(stdout);
    }
  }

  return 0;
}
