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

#define NX 1
#define NY 1
#define NZ 512
#define STEP 1.0 /* Bohr */
#define TS 15.0 /* fs */

#define PRESSURE 0.0

#define THREADS 0

#define RHO0 (0.0218360 * GRID_AUTOANG * GRID_AUTOANG * GRID_AUTOANG)

#define HELIUM_MASS (4.002602 / GRID_AUTOAMU)

REAL complex wave(void *arg, REAL x, REAL y, REAL z);

int main(int argc, char **argv) {

  dft_ot_functional *otf;
  INT l, iterations;
  REAL sizex, sizey, sizez, mu0, rho0;
  grid_timer timer;
  cgrid *potential_store;
  wf *gwf, *gwfp;
  dft_plane_wave wave_params;
  
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

  /* Initialize threads & use wisdom */
  grid_set_fftw_flags(1);    // FFTW_MEASURE
  grid_threads_init(THREADS);
  grid_fft_read_wisdom(NULL);

  /* Allocate wave functions */
  if(!(gwf = grid_wf_alloc(NX, NY, NZ, STEP, DFT_HELIUM_MASS, WF_PERIODIC_BOUNDARY, WF_2ND_ORDER_FFT, "gwf"))) {
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

  potential_store = cgrid_clone(gwf->grid, "potential store");
    
  wave_params.rho = dft_ot_bulk_density(otf);
  grid_wf_map(gwf, dft_common_planewave, &wave_params);

  for(l = 0; l < iterations; l++) {

    if(l == 5) grid_fft_write_wisdom(NULL);

    grid_timer_start(&timer);

    /* Predict-Correct */
    cgrid_copy(gwfp->grid, gwf->grid);
    cgrid_zero(potential_store);
    dft_ot_potential(otf, potential_store, gwf);
    cgrid_add(potential_store, -mu0);
    grid_wf_propagate_predict(gwfp, potential_store, TS / GRID_AUTOFS);
    dft_ot_potential(otf, potential_store, gwfp);
    cgrid_add(potential_store, -mu0);
    cgrid_multiply(potential_store, 0.5);  // Use (current + future) / 2
    grid_wf_propagate_correct(gwf, potential_store, TS / GRID_AUTOFS);
    // Chemical potential included - no need to normalize

    printf(FMT_R " " FMT_R "\n", ((REAL) l) * TS, POW(CABS(cgrid_value_at_index(gwf->grid, NX/2, NY/2, NZ/2)), 2.0));
    fflush(stdout);

    fprintf(stderr, "One iteration = " FMT_R " wall clock seconds.\n", grid_timer_wall_clock_time(&timer));
  }
  
  return 0;
}

