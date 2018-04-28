/*
 * Try for example:
 * ./dispersion 1000 0 0 8 0.001 > test.dat
 *
 * Convert period (x) from fs to K: (1 / (2.0 * x * 1E-15)) * 3.335E-11 * 1.439
 * (period -> Hz -> cm-1 -> K)   TODO: Where did the 2.0 above come from?
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
#define STEP 2.0 /* Bohr */
#define TS 15.0 /* fs */

#define THREADS 0

#define RHO0 (0.0218360 * GRID_AUTOANG * GRID_AUTOANG * GRID_AUTOANG)

#define HELIUM_MASS (4.002602 / GRID_AUTOAMU)

typedef struct sWaveParams_struct {
  REAL kx, ky, kz;
  REAL a, rho;
} sWaveParams;

REAL complex wave(void *arg, REAL x, REAL y, REAL z);

int main(int argc, char **argv) {

  INT l, iterations;
  REAL sizex, sizey, sizez, mu0;
  grid_timer timer;
  cgrid *potential_store;
  wf *gwf, *gwfp;
  rgrid *density, *pot;
  sWaveParams wave_params;
  
  /* parameters */
  if (argc != 6) {
    fprintf(stderr, "Usage: %s <iterations> <kx> <ky> <kz> <amplitude>\n", argv[0]);
    return 1;
  }
  
  iterations = atol(argv[1]);
  sizex = NX * STEP;
  sizey = NY * STEP;
  sizez = NZ * STEP;

  wave_params.kx = (REAL) atof(argv[2]) * 2.0 * M_PI / sizex;
  wave_params.ky = (REAL) atof(argv[3]) * 2.0 * M_PI / sizey;
  wave_params.kz = (REAL) atof(argv[4]) * 2.0 * M_PI / sizez;
  wave_params.a = (REAL) atof(argv[5]);
  fprintf(stderr, "Momentum (" FMT_R " x " FMT_R " x " FMT_R ") Angs^-1\n", wave_params.kx / GRID_AUTOANG, wave_params.ky / GRID_AUTOANG, wave_params.kz / GRID_AUTOANG);
  
#ifdef USE_CUDA
  cuda_enable(1);
#endif

  dft_driver_setup_grid(NX, NY, NZ, STEP, THREADS);
  //dft_driver_setup_model(DFT_OT_PLAIN, DFT_DRIVER_REAL_TIME, 0.0);
  // dft_driver_setup_model(DFT_OT_PLAIN + DFT_OT_KC, DFT_DRIVER_REAL_TIME, 0.0);
  dft_driver_setup_model(DFT_OT_PLAIN, DFT_DRIVER_REAL_TIME, 0.0);
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
  rgrid_constant(pot, -mu0);
    
  wave_params.rho = dft_ot_bulk_density(dft_driver_otf);
  grid_wf_map(gwf, wave, &wave_params);

  for(l = 0; l < iterations; l++) {
    grid_timer_start(&timer);
    dft_driver_propagate_predict(DFT_DRIVER_PROPAGATE_HELIUM, pot, gwf, gwfp, potential_store, TS /* fs */, l);
    dft_driver_propagate_correct(DFT_DRIVER_PROPAGATE_HELIUM, pot, gwf, gwfp, potential_store, TS /* fs */, l);
    grid_wf_density(gwf, density);
    printf(FMT_R " " FMT_R "\n", ((REAL) l) * TS, rgrid_value_at_index(density, NX/2, NY/2, NZ/2));
    fflush(stdout);
    fprintf(stderr, "One iteration = " FMT_R " wall clock seconds.\n", grid_timer_wall_clock_time(&timer));
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
