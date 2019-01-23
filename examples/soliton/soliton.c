
/*
 * Create planar soliton in superfluid helium (propagating along X).
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

#define TS 5.0 /* fs */
#define NX 16384
#define NY 32
#define NZ 32
#define STEP 1.0
#define NTH 1000

#define THREADS 0

#define PRESSURE (0.0 / GRID_AUTOBAR)

/* by +-2 x LAMBDA_C */
/* #define SMOOTH */

#define SOLITON_AMP (0.2)   /* 10% of bulk */
#define SOLITON_N  300       /* width (in N * LAMBDA_C) */
#define LAMBDA_C (3.58 / GRID_AUTOANG)

#define THREADS 0

#define HELIUM_MASS (4.002602 / GRID_AUTOAMU)

/* Francesco's soliton initial guess - plane along z*/
REAL soliton(void *asd, REAL x, REAL y, REAL z) {

  REAL tmp;

  if(FABS(x) > SOLITON_N * LAMBDA_C) return 1.0;

  tmp = SIN(M_PI * x / LAMBDA_C);

  return 1.0 + SOLITON_AMP * tmp * tmp;
}

int main(int argc, char **argv) {

  dft_ot_functional *otf;
  rgrid *rworkspace;
  cgrid *potential_store;
  wf *gwf, *gwfp;
  INT iter;
  REAL mu0, rho0;
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
  if(!(gwf = grid_wf_alloc(NX, NY, NZ, STEP, DFT_HELIUM_MASS, WF_PERIODIC_BOUNDARY, WF_2ND_ORDER_PROPAGATOR, "gwf"))) {
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

  /* Allocate space for external potential */
  rworkspace = rgrid_clone(otf->density, "rworkspace");
  potential_store = cgrid_clone(gwf->grid, "cworkspace"); /* temporary storage */

  /* setup soliton */
  rgrid_map(rworkspace, soliton, NULL);
  rgrid_multiply(rworkspace, rho0);
  rgrid_power(rworkspace, rworkspace, 0.5);
  cgrid_zero(gwf->grid);   /* copy rho to wf real part */
  grid_real_to_complex_re(gwf->grid, rworkspace);

  for (iter = 0; iter < 800000; iter++) {

    if(iter == 5) grid_fft_write_wisdom(NULL);

    grid_timer_start(&timer);

    /* Predict-Correct */
    cgrid_copy(gwfp->grid, gwf->grid);
    cgrid_zero(potential_store);
    dft_ot_potential(otf, potential_store, gwf);
    cgrid_add(potential_store, -mu0);
    grid_wf_propagate_predict(gwfp, potential_store, -I * TS / GRID_AUTOFS);
    dft_ot_potential(otf, potential_store, gwfp);
    cgrid_add(potential_store, -mu0);
    cgrid_multiply(potential_store, 0.5);  // Use (current + future) / 2
    grid_wf_propagate_correct(gwf, potential_store, -I * TS / GRID_AUTOFS);
    // Chemical potential included - no need to normalize

    printf("Iteration " FMT_I " - Wall clock time = " FMT_R " seconds.\n", iter, grid_timer_wall_clock_time(&timer));

    if(!(iter % NTH)) {
      sprintf(buf, "soliton-" FMT_I, iter);
      grid_wf_density(gwf, rworkspace);
#ifdef SMOOTH
      rgrid_npoint_smooth(rworkspace2, rworkspace, (INT) (2.0 * LAMBDA_C / STEP));
      rgrid_write_grid(buf, rworkspace2);
#else
      rgrid_write_grid(buf, rworkspace);
#endif
    }
  }
  return 0;
}
