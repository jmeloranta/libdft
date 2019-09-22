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
#include <stdlib.h>
#include <dft/dft.h>
#include <dft/ot.h>

/* Time step in imag/real iterations (fs) */
#define TIME_STEP (40.0 / GRID_AUTOFS)

#define NX 256
#define NY 256
#define NZ 256
#define STEP 1.0

#define NITER 10000000
#define NTH 1000

#define THREADS 0

#define DFT_HELIUM_MASS (4.002602 / GRID_AUTOAMU) /* Helium mass in atomic units */
#define RHO0 (0.0218360 * GRID_AUTOANG * GRID_AUTOANG * GRID_AUTOANG)

#define NBINS 4096
#define BINSTEP 0.01

REAL complex wave(void *NA, REAL x, REAL y, REAL z) {

  REAL kx = 2.0 * M_PI / (NX * STEP);

//  return SQRT(RHO0) * (1.0 - EXP(-2.0 * (x*x + y * y + z * z)));
  return SQRT(RHO0) + 0.1 * CEXP(3.0*I * kx * (x + y + z));
}

int main(int argc, char **argv) {

  wf *gwf;
  cgrid *cworkspace;
  INT i, j;
  dft_ot_functional *otf;
  REAL bins[NBINS];
  rgrid *wrk1, *wrk2, *wrk3, *wrk4;

#ifdef USE_CUDA
#define NGPUS 1
int gpus[] = {0};
  cuda_enable(0, NGPUS, gpus);
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

  /* Allocate OT functional */
  if(!(otf = dft_ot_alloc(DFT_OT_PLAIN, gwf, DFT_MIN_SUBSTEPS, DFT_MAX_SUBSTEPS))) {
    fprintf(stderr, "Cannot allocate otf.\n");
    exit(1);
  }
  
  cworkspace = cgrid_clone(gwf->grid, "cworkspace");
  wrk1 = rgrid_clone(otf->density, "wrk1");
  wrk2 = rgrid_clone(otf->density, "wrk2");
  wrk3 = rgrid_clone(otf->density, "wrk3");
  wrk4 = rgrid_clone(otf->density, "wrk4");

  cgrid_constant(gwf->grid, SQRT(RHO0));
  cgrid_random(gwf->grid, 0.01 * SQRT(RHO0));

  for (i = 0; i < NITER; i++) {
    fprintf(stderr, "Iteration " FMT_I "\n", i);
    cgrid_zero(cworkspace);
    dft_ot_potential(otf, cworkspace, gwf);
    grid_wf_propagate(gwf, cworkspace, TIME_STEP);
    if(!(i % NTH)) {
      grid_wf_KE(gwf, bins, BINSTEP, NBINS, wrk1, wrk2, wrk3, wrk4);
      for(j = 0; j < NBINS; j++)
        printf(FMT_R " " FMT_R "\n", 0.1 * (REAL) j, bins[j]);
      printf("\n");
    }
  }

  return 0;
}
