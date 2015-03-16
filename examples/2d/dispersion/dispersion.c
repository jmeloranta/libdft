/*
 * Try for example:
 * ./dispersion 1000 8 0.001 > test.dat
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

#define N 512
#define STEP 0.4 /* Bohr */
#define TS 10.0 /* fs */

#define RHO0 (0.0218360 * GRID_AUTOANG * GRID_AUTOANG * GRID_AUTOANG)

#define HELIUM_MASS (4.002602 / GRID_AUTOAMU)

typedef struct sWaveParams_struct {
  double kz;
  double kr;
  double a, rho;
} sWaveParams;

double complex wave(void *arg, double z, double r);

int main(int argc, char **argv) {

  long l, iterations;
  double step, size;
  grid_timer timer;
  cgrid2d *potential_store;
  wf2d *gwf, *gwfp;
  rgrid2d *density;
  sWaveParams wave_params;
  char buf[512];
  
  /* parameters */
  if (argc != 5) {
    fprintf(stderr, "Usage: %s <iterations> <kz> <kr> <amplitude>\n", argv[0]);
    return 1;
  }
  
  iterations = atol(argv[1]);
  size = (N-1.0) * STEP;
  
  wave_params.kz = atof(argv[2]) * 2.0 * M_PI / size;
  wave_params.kr = atof(argv[3]) * 2.0 * M_PI / size;
  wave_params.a = atof(argv[4]);
  wave_params.rho = RHO0;
  
  fprintf(stderr, "Momentum (%lf, %lf) Angs^-1\n", wave_params.kz / GRID_AUTOANG, wave_params.kr / GRID_AUTOANG);

  dft_driver_setup_grid_2d(N, N, STEP, 32); /* 32 threads */
  //dft_driver_setup_model_2d(DFT_OT_PLAIN + DFT_OT_KC, DFT_DRIVER_REAL_TIME, 0.0);
  dft_driver_setup_model_2d(DFT_OT_PLAIN, DFT_DRIVER_REAL_TIME, 0.0);
  dft_driver_setup_boundaries_2d(DFT_DRIVER_BOUNDARY_REGULAR, 2.0);
  dft_driver_setup_normalization_2d(DFT_DRIVER_NORMALIZE_BULK, 0, 0.0, 0);
  dft_driver_initialize_2d();
  density = dft_driver_alloc_rgrid_2d();
  potential_store = dft_driver_alloc_cgrid_2d();
  gwf = dft_driver_alloc_wavefunction_2d(HELIUM_MASS);
  gwfp = dft_driver_alloc_wavefunction_2d(HELIUM_MASS);
  
  grid2d_wf_map_cyl(gwf, wave, &wave_params);

  for(l = 0; l < iterations; l++) {
    grid_timer_start(&timer);
    dft_driver_propagate_predict_2d(DFT_DRIVER_PROPAGATE_HELIUM, NULL, gwf, gwfp, potential_store, TS /* fs */, l);
    dft_driver_propagate_correct_2d(DFT_DRIVER_PROPAGATE_HELIUM, NULL, gwf, gwfp, potential_store, TS /* fs */, l);
    grid2d_wf_density(gwf, density);
    //    printf("%lf %.10le\n", l * TS, rgrid2d_value_at_index(density, N/2, 0));
    printf("%lf %.10le\n", l * TS, rgrid2d_value_at_index(density, 0, N/2));
    fflush(stdout);
    fprintf(stderr, "One iteration = %lf wall clock seconds.\n", grid_timer_wall_clock_time(&timer));
    //sprintf(buf, "output-%ld", l);
    //dft_driver_write_density(density, buf);
  }
  
  return 0;
}


double complex wave(void *arg, double z, double r) {

  double kz = ((sWaveParams *) arg)->kz;
  double kr = ((sWaveParams *) arg)->kr;
  double a = ((sWaveParams *) arg)->a;
  double psi = sqrt(((sWaveParams *) arg)->rho);
  
  return psi + 0.5 * a * psi * (cexp(I * (kz * z)) + cexp(-I*(kz * z))) + 0.5 * a * psi * (cexp(I * (kr * r)) + cexp(-I*(kr * r)));
}
