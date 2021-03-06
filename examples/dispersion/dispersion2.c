/*
 * Calculate the dispersion relation up to specified n.
 *
 * TODO: This is sensitive to any numerical noise and may not be able to detect the turn over point correctly.
 *
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <complex.h>

#include <grid/grid.h>
#include <grid/au.h>
#include <dft/dft.h>
#include <dft/ot.h>

#define FUNCTIONAL (DFT_OT_PLAIN | DFT_OT_KC)
//#define FUNCTIONAL (DFT_OT_PLAIN | DFT_OT_KC | DFT_OT_BACKFLOW)
#define NX 256
#define NY 256
#define NZ 256
#define STEP 0.25 /* Bohr */
#define TS 5.0 /* fs */
#define AMP 1e-3 /* wave amplitude (of total rho0) */
#define PRED 0
#define DIRECTION 2     /* Plane wave direction: X = 0, Y = 1, Z = 2 */

#define PRESSURE (0.0 / GRID_AUTOPA)

#define THREADS 0

#define HELIUM_MASS (4.002602 / GRID_AUTOAMU)

//#undef USE_CUDA

int main(int argc, char **argv) {

  dft_ot_functional *otf;
  REAL k, kk, e, mu0;
  wf *gwf;
  
  /* parameters */
  if (argc != 4) {
    fprintf(stderr, "Usage: %s <k_min Ang^-1> <k_max Ang^-1> <k_step Ang^-1>\n", argv[0]);
    return 1;
  }

#ifdef USE_CUDA
#define NGPUS 1
int gpus[] = {0};
  cuda_enable(1, NGPUS, gpus);
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
  if(!(otf = dft_ot_alloc(FUNCTIONAL, gwf, DFT_MIN_SUBSTEPS, DFT_MAX_SUBSTEPS))) {
    fprintf(stderr, "Cannot allocate otf.\n");
    exit(1);
  }

  // Backflow limit
//  otf->max_bfpot = 1.4 / GRID_AUTOK;
  // Increase backflow to reduce roton gap
//  otf->c_bfpot = 1.0;
  if(otf->c_bfpot != 1.0) fprintf(stderr, "WARNING: Backflow coefficient not one!\n");
  otf->rho0 = dft_ot_bulk_density_pressurized(otf, PRESSURE);
  mu0 = dft_ot_bulk_chempot_pressurized(otf, PRESSURE);
  fprintf(stderr, "mu0 = " FMT_R " K/atom, rho0 = " FMT_R " Angs^-3.\n", mu0 * GRID_AUTOK, otf->rho0 / (GRID_AUTOANG * GRID_AUTOANG * GRID_AUTOANG));

  fprintf(stderr, "Applied P = " FMT_R " MPa.\n", PRESSURE * GRID_AUTOPA / 1E6);

  printf("# Dispersion relation for functional " FMT_I ".\n", otf->model);
  printf("0 0\n");
  for (k = (REAL) atof(argv[1]) * GRID_AUTOANG; k < (REAL) atof(argv[2]) * GRID_AUTOANG; k += (REAL) atof(argv[3]) * GRID_AUTOANG) {
    kk = k;
    e = dft_ot_dispersion(gwf, otf, TS / GRID_AUTOFS, &kk, AMP, PRED, DIRECTION);
    printf(FMT_R " " FMT_R "\n", kk / GRID_AUTOANG, e * GRID_AUTOK);
    fflush(stdout);
  }
  return 0;
}
