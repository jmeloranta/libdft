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

#define NX 512
#define NY 256
#define NZ 256
#define STEP 1.0 /* Bohr */
#define TS 15.0 /* fs */
#define AMP 1e-2 /* wave amplitude (of total rho0) */
#define RHO0 (0.0218360 * GRID_AUTOANG * GRID_AUTOANG * GRID_AUTOANG)
#define PRED 0

#define THREADS 0

#define HELIUM_MASS (4.002602 / GRID_AUTOAMU)

int main(int argc, char **argv) {

  INT model;
  REAL k, kk, e;
  
  /* parameters */
  if (argc != 4) {
    fprintf(stderr, "Usage: %s <k_min Ang^-1> <k_max Ang^-1> <k_step Ang^-1>\n", argv[0]);
    return 1;
  }

  dft_driver_setup_grid(NX, NY, NZ, STEP, THREADS);

#ifdef USE_CUDA
  cuda_enable(1);
#endif

  model = DFT_OT_PLAIN;
  dft_driver_setup_model(model, DFT_DRIVER_REAL_TIME, RHO0);
  // Note: CN_NBC has really wrong BC for this but it will not hit the center of the box within the first cycle...
  dft_driver_kinetic = DFT_DRIVER_KINETIC_CN_NBC;
//  dft_driver_kinetic = DFT_DRIVER_KINETIC_FFT;

  dft_driver_setup_boundary_type(DFT_DRIVER_BOUNDARY_REGULAR, 0.0, 0.0, 0.0, 0.0);
  dft_driver_setup_normalization(DFT_DRIVER_DONT_NORMALIZE, 0, 0.0, 0);
  dft_driver_setup_boundary_condition(DFT_DRIVER_BC_NORMAL);
  dft_driver_initialize();
  fprintf(stderr, "Applied P = " FMT_R " MPa.\n", dft_ot_bulk_pressure(dft_driver_otf, RHO0) * GRID_AUTOPA / 1E6);

  printf("# Dispersion relation for functional " FMT_I ".\n", model);
  printf("0 0\n");
  for (k = (REAL) atof(argv[1]) * GRID_AUTOANG; k < (REAL) atof(argv[2]) * GRID_AUTOANG; k += (REAL) atof(argv[3]) * GRID_AUTOANG) {
    kk = k;
    e = dft_ot_dispersion(TS, &kk, AMP, PRED);
    printf(FMT_R " " FMT_R "\n", kk / GRID_AUTOANG, e * GRID_AUTOK);
    fflush(stdout);
  }
  return 0;
}
