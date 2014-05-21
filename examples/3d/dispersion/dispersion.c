/*
 * Try for example:
 * ./dispersion 1000 0 0 8 0.001 > test.dat
 *
 * Convert period (x) from fs to K: (1 / (x * 1E-15)) * 3.335E-11 * 1.439
 * (period -> Hz -> cm-1 -> K)
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

#define N 256
#define STEP 0.5 /* Bohr */
#define TS 40.0 /* fs */

#define RHO0 (0.0218360 * GRID_AUTOANG * GRID_AUTOANG * GRID_AUTOANG)

#define HELIUM_MASS (4.002602 / GRID_AUTOAMU)

typedef struct sWaveParams_struct {
  double kx, ky, kz;
  double a, rho;
} sWaveParams;

double complex wave(void *arg, double x, double y, double z);

int main(int argc, char **argv) {

  long l, iterations;
  double step, size;
  grid_timer timer;
  cgrid3d *potential_store;
  wf3d *gwf, *gwfp;
  rgrid3d *density;
  sWaveParams wave_params;
  char buf[512];
  
  /* parameters */
  if (argc != 6) {
    fprintf(stderr, "Usage: %s <iterations> <kx> <ky> <kz> <amplitude>\n", argv[0]);
    return 1;
  }
  
  iterations = atol(argv[1]);
  size = (N-1.0) * STEP;
  
  wave_params.kx = atof(argv[2]) * 2.0 * M_PI / size;
  wave_params.ky = atof(argv[3]) * 2.0 * M_PI / size;
  wave_params.kz = atof(argv[4]) * 2.0 * M_PI / size;
  wave_params.a = atof(argv[5]);
  wave_params.rho = RHO0;
  
  fprintf(stderr, "Momentum (%lf x %lf x %lf) Angs^-1\n", wave_params.kx / GRID_AUTOANG, wave_params.ky / GRID_AUTOANG, wave_params.kz / GRID_AUTOANG);

  dft_driver_setup_grid(N, N, N, STEP, 48); /* 6 threads */
  //dft_driver_setup_model(DFT_OT_PLAIN, DFT_DRIVER_REAL_TIME, 0.0);
  // dft_driver_setup_model(DFT_OT_PLAIN + DFT_OT_KC, DFT_DRIVER_REAL_TIME, 0.0);
  dft_driver_setup_model(DFT_OT_PLAIN + DFT_OT_KC + DFT_OT_BACKFLOW, DFT_DRIVER_REAL_TIME, 0.0);
  dft_driver_setup_boundaries(DFT_DRIVER_BOUNDARY_REGULAR, 2.0);
  dft_driver_setup_normalization(DFT_DRIVER_NORMALIZE_BULK, 0, 0.0, 0);
  dft_driver_initialize();
  density = dft_driver_alloc_rgrid();
  potential_store = dft_driver_alloc_cgrid();
  gwf = dft_driver_alloc_wavefunction(HELIUM_MASS);
  gwfp = dft_driver_alloc_wavefunction(HELIUM_MASS);
  
  grid3d_wf_map(gwf, wave, &wave_params);

  for(l = 0; l < iterations; l++) {
    grid_timer_start(&timer);
    dft_driver_propagate_predict(DFT_DRIVER_PROPAGATE_HELIUM, NULL, gwf, gwfp, potential_store, TS /* fs */, l);
    dft_driver_propagate_correct(DFT_DRIVER_PROPAGATE_HELIUM, NULL, gwf, gwfp, potential_store, TS /* fs */, l);
    grid3d_wf_density(gwf, density);
    printf("%lf %.10le\n", l * TS, rgrid3d_value_at_index(density, N/2, N/2, N/2));
    fflush(stdout);
    fprintf(stderr, "One iteration = %lf wall clock seconds.\n", grid_timer_wall_clock_time(&timer));
    //sprintf(buf, "output-%ld", l);
    //dft_driver_write_density(density, buf);
  }
  
  return 0;
}


double complex wave(void *arg, double x, double y, double z) {

  double kx = ((sWaveParams *) arg)->kx;
  double ky = ((sWaveParams *) arg)->ky;
  double kz = ((sWaveParams *) arg)->kz;
  double a = ((sWaveParams *) arg)->a;
  double psi = sqrt(((sWaveParams *) arg)->rho);
  
  return psi + 0.5 * a * psi * (cexp(I * (kx * x + ky * y + kz * z)) + cexp(-I*(kx * x + ky * y + kz * z)));
}
