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

#define NX 128
#define NY 128
#define NZ 128
#define STEP 0.5 /* Bohr */
#define TS 10.0 /* fs */
#define AMP 1e-3 /* wave amplitude (of total rho0) */
#define PRED 0
#define DIRECTION 2     /* Plane wave direction: X = 0, Y = 1, Z = 2 */

#define PRESSURE 0.0

#define THREADS 0

#define HELIUM_MASS (4.002602 / GRID_AUTOAMU)

//#undef USE_CUDA

int main(int argc, char **argv) {

  dft_ot_functional *otf;
  REAL k, kk, e, rho0, mu0;
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
  if(!(otf = dft_ot_alloc(DFT_OT_PLAIN | DFT_OT_KC | DFT_OT_BACKFLOW | DFT_OT_HD, gwf, DFT_MIN_SUBSTEPS, DFT_MAX_SUBSTEPS))) {
    fprintf(stderr, "Cannot allocate otf.\n");
    exit(1);
  }

  // Backflow limits
//  otf->max_bfpot = 0.3 / GRID_AUTOK;

  rho0 = dft_ot_bulk_density_pressurized(otf, PRESSURE);
  mu0 = dft_ot_bulk_chempot_pressurized(otf, PRESSURE);
  fprintf(stderr, "mu0 = " FMT_R " K/atom, rho0 = " FMT_R " Angs^-3.\n", mu0 * GRID_AUTOK, rho0 / (GRID_AUTOANG * GRID_AUTOANG * GRID_AUTOANG));

  fprintf(stderr, "Applied P = " FMT_R " MPa.\n", dft_ot_bulk_pressure(otf, rho0) * GRID_AUTOPA / 1E6);

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
