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

#define NX 512
#define NY 128
#define NZ 128
#define STEP 0.2 /* Bohr */
#define TS 10.0 /* fs */

#define THREADS 0

#define RHO0 (0.0218360 * GRID_AUTOANG * GRID_AUTOANG * GRID_AUTOANG)

#define HELIUM_MASS (4.002602 / GRID_AUTOAMU)

typedef struct sWaveParams_struct {
  REAL kx, ky, kz;
  REAL a, rho;
} sWaveParams;

REAL complex wave(void *arg, REAL x, REAL y, REAL z);

int main(int argc, char **argv) {

  INT l, n, model;
  REAL mu0, prev_val;
  grid_timer timer;
  cgrid3d *potential_store;
  wf3d *gwf, *gwfp;
  rgrid3d *density, *pot;
  sWaveParams wave_params;
  
  /* parameters */
  if (argc != 3) {
    fprintf(stderr, "Usage: %s <n_min> <n_max>\n", argv[0]);
    return 1;
  }
  
  fprintf(stderr, "Minimum n corresponds to " FMT_R " Angs^-1.\n", atof(argv[1]) * 2.0 * M_PI / (GRID_AUTOANG * NX * STEP));
  fprintf(stderr, "Maxmimum n corresponds to " FMT_R " Angs^-1.\n", atof(argv[2]) * 2.0 * M_PI / (GRID_AUTOANG * NX * STEP));
  dft_driver_setup_grid(NX, NY, NZ, STEP, THREADS);

  model = DFT_OT_PLAIN | DFT_OT_BACKFLOW | DFT_OT_KC;
  dft_driver_setup_model(model, DFT_DRIVER_REAL_TIME, RHO0);

  dft_driver_setup_boundary_type(DFT_DRIVER_BOUNDARY_REGULAR, 0.0, 0.0, 0.0, 0.0);
  dft_driver_setup_normalization(DFT_DRIVER_DONT_NORMALIZE, 0, 0.0, 0);
  dft_driver_setup_boundary_condition(DFT_DRIVER_BC_NORMAL);
  dft_driver_initialize();
  density = dft_driver_alloc_rgrid("density");
  potential_store = dft_driver_alloc_cgrid("potential_store");
  gwf = dft_driver_alloc_wavefunction(HELIUM_MASS, "gwf");
  gwfp = dft_driver_alloc_wavefunction(HELIUM_MASS, "gwfp");
  pot = dft_driver_alloc_rgrid("pot");
  mu0 = dft_ot_bulk_chempot2(dft_driver_otf);
  rgrid3d_constant(pot, -mu0);
  fprintf(stderr, "mu0 = " FMT_R "K.\n", mu0 * GRID_AUTOK);
  fprintf(stderr, "Applied P =" FMT_R " MPa.\n", dft_ot_bulk_pressure(dft_driver_otf, RHO0) * GRID_AUTOPA / 1E6);

  printf("# Dispersion relation for functional " FMT_I ".\n", model);
  printf("0 0\n");
  for (n = atof(argv[1]); n <= atof(argv[2]); n++) {
    wave_params.kx = n * 2.0 * M_PI / (NX * STEP);
    wave_params.ky = 0.0;
    wave_params.kz = 0.0;
    wave_params.a = 1.0E-3;
    wave_params.rho = RHO0;
    grid3d_wf_map(gwf, wave, &wave_params);
    prev_val = 1E99;
    for(l = 0; ; l++) {
      grid_timer_start(&timer);
      dft_driver_propagate_predict(DFT_DRIVER_PROPAGATE_HELIUM, pot, gwf, gwfp, potential_store, TS /* fs */, l);
      dft_driver_propagate_correct(DFT_DRIVER_PROPAGATE_HELIUM, pot, gwf, gwfp, potential_store, TS /* fs */, l);
      grid3d_wf_density(gwf, density);
      if(rgrid3d_value_at_index(density, NX/2, NY/2, NZ/2) > prev_val) {
	l--;
	break;
      }
      prev_val = rgrid3d_value_at_index(density, NX/2, NY/2, NZ/2);
      fprintf(stderr, "One iteration = " FMT_R " wall clock seconds.\n", grid_timer_wall_clock_time(&timer));
    }
    printf(FMT_R " " FMT_R "\n", wave_params.kx / GRID_AUTOANG, (1.0 / ((2.0 * l * TS) * 1E-15)) * 3.335E-11 * 1.439);
    fflush(stdout);
  }
  return 0;
}

REAL complex wave(void *arg, REAL x, REAL y, REAL z) {

  REAL kx = ((sWaveParams *) arg)->kx;
  REAL ky = ((sWaveParams *) arg)->ky;
  REAL kz = ((sWaveParams *) arg)->kz;
  REAL a = ((sWaveParams *) arg)->a;
  REAL psi = SQRT(((sWaveParams *) arg)->rho);
  
  return psi + 0.5 * a * psi * (CEXP(I * (kx * x + ky * y + kz * z)) + CEXP(-I*(kx * x + ky * y + kz * z)));
}
