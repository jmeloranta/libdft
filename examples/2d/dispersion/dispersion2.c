/*
 * Calculate the dispersion relation up to specified n.
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

#define N 128
#define STEP 0.2 /* Bohr */
#define TS 10.0 /* fs */

#define RHO0 (0.0218360 * GRID_AUTOANG * GRID_AUTOANG * GRID_AUTOANG)

#define HELIUM_MASS (4.002602 / GRID_AUTOAMU)

typedef struct sWaveParams_struct {
  double kz;
  double a, rho;
} sWaveParams;

double complex wave(void *arg, double z, double r);

int main(int argc, char **argv) {

  long l, n, model;
  double mu0, prev_val;
  grid_timer timer;
  cgrid2d *potential_store;
  wf2d *gwf, *gwfp;
  rgrid2d *density, *pot;
  sWaveParams wave_params;
  
  /* parameters */
  if (argc != 3) {
    fprintf(stderr, "Usage: %s <n_min> <n_max>\n", argv[0]);
    return 1;
  }
  
  fprintf(stderr, "Minimum n corresponds to %le Angs^-1.\n", atof(argv[1]) * 2.0 * M_PI / (GRID_AUTOANG * N * STEP));
  fprintf(stderr, "Maxmimum n corresponds to %le Angs^-1.\n", atof(argv[2]) * 2.0 * M_PI / (GRID_AUTOANG * N * STEP));
  dft_driver_setup_grid_2d(N, N, STEP, 16); /* 6 threads */

  model = DFT_OT_PLAIN | DFT_OT_BACKFLOW | DFT_OT_KC;
  dft_driver_setup_model_2d(model, DFT_DRIVER_REAL_TIME, 0.0);

  dft_driver_setup_boundaries_2d(DFT_DRIVER_BOUNDARY_REGULAR, 2.0);
  dft_driver_setup_normalization_2d(DFT_DRIVER_DONT_NORMALIZE, 0, 0.0, 0);
  dft_driver_setup_boundary_condition_2d(DFT_DRIVER_BC_NORMAL);
  dft_driver_initialize_2d();
  density = dft_driver_alloc_rgrid_2d();
  potential_store = dft_driver_alloc_cgrid_2d();
  gwf = dft_driver_alloc_wavefunction_2d(HELIUM_MASS);
  gwfp = dft_driver_alloc_wavefunction_2d(HELIUM_MASS);
  pot = dft_driver_alloc_rgrid_2d();
  mu0 = dft_ot_bulk_chempot2_2d(dft_driver_otf_2d);
  rgrid2d_constant(pot, -mu0);

  printf("# Dispersion relation for functional %ld.\n", model);
  printf("0 0\n");
  for (n = atof(argv[1]); n <= atof(argv[2]); n++) {
    wave_params.kz = n * 2.0 * M_PI / (N * STEP);   /* TODO: should be able to choose x, y or z */
    wave_params.a = 1.0E-3;
    wave_params.rho = dft_ot_bulk_density_2d(dft_driver_otf_2d);
    grid2d_wf_map(gwf, wave, &wave_params);
    prev_val = 1E99;
    for(l = 0; ; l++) {
      grid_timer_start(&timer);
      dft_driver_propagate_predict_2d(DFT_DRIVER_PROPAGATE_HELIUM, pot, gwf, gwfp, potential_store, TS /* fs */, l);
      dft_driver_propagate_correct_2d(DFT_DRIVER_PROPAGATE_HELIUM, pot, gwf, gwfp, potential_store, TS /* fs */, l);
      grid2d_wf_density(gwf, density);
      if(rgrid2d_value_at_index(density, N/2, 0) > prev_val) {
	l--;
	break;
      }
      prev_val = rgrid2d_value_at_index(density, N/2, 0);
      fprintf(stderr, "One iteration = %lf wall clock seconds.\n", grid_timer_wall_clock_time(&timer));
    }
    printf("%le %le\n", wave_params.kz / GRID_AUTOANG, (1.0 / ((2.0 * l * TS) * 1E-15)) * 3.335E-11 * 1.439);
    fflush(stdout);
  }
  return 0;
}


double complex wave(void *arg, double z, double r) {

  double kz = ((sWaveParams *) arg)->kz;
  double a = ((sWaveParams *) arg)->a;
  double psi = sqrt(((sWaveParams *) arg)->rho);
  
  return psi + 0.5 * a * psi * (cexp(I * (kz * z)) + cexp(-I*(kz * z)));
}
