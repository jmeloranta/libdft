/*
 * Try for example:
 * ./dispersion 1000 0 0 8 0.001 > test.dat
 *
 * Convert period (x) from fs to K: (1 / (2.0 * x * 1E-15)) * 3.335E-11 * 1.439
 * (period -> Hz -> cm-1 -> K)  Note: 2X since x is only half wave length 
 * 
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

#define N 8192
#define STEP 0.5 /* Bohr */
#define TS 5.0 /* fs */

#define THREADS 0

#define PREDICT_CORRECT /**/

#define HELIUM_MASS (4.002602 / GRID_AUTOAMU)

REAL complex wave(void *arg, REAL x, REAL y, REAL z);

extern void OT_INIT(rgrid *, rgrid *);
extern void OT_POT(rgrid *, rgrid *, rgrid *, rgrid *, rgrid *, rgrid *, rgrid *, rgrid *);

int main(int argc, char **argv) {

  INT l, iterations;
  REAL sizez, mu0, rho0;
  grid_timer timer;
  rgrid *rworkspace, *rworkspace2, *rworkspace3, *lj_tf, *rd_tf, *density_tf, *ot_pot, *spave_tf;
  wf *gwf;
#ifdef PREDICT_CORRECT 
  wf *gwfp;
  cgrid *potential_store;
#endif
  dft_plane_wave wave_params;
  
  /* parameters */
  if (argc != 4) {
    fprintf(stderr, "Usage: %s <iterations> <n> <amplitude>\n", argv[0]);
    return 1;
  }
  
  dft_driver_init_ot = 0;   /* We allocate the grids manually */

  iterations = atol(argv[1]);
  sizez = N * STEP;

  wave_params.kx = 0.0;
  wave_params.ky = 0.0;
  wave_params.kz = (REAL) atof(argv[2]) * 2.0 * M_PI / sizez;
  wave_params.a = (REAL) atof(argv[3]);
  fprintf(stderr, "Momentum (" FMT_R ") Angs^-1\n", wave_params.kz / GRID_AUTOANG);
  
#ifdef USE_CUDA
  cuda_enable(1);
#endif

  dft_driver_setup_grid(1, 1, N, STEP, THREADS);

// Does nothing
  dft_driver_setup_model(DFT_OT_PLAIN, DFT_DRIVER_REAL_TIME, 0.0);
//
  dft_driver_setup_boundary_type(DFT_DRIVER_BOUNDARY_REGULAR, 0.0, 0.0, 0.0, 0.0);
  dft_driver_setup_normalization(DFT_DRIVER_DONT_NORMALIZE, 0, 0.0, 0);
  dft_driver_setup_boundary_condition(DFT_DRIVER_BC_NORMAL);
  dft_driver_initialize();
  gwf = dft_driver_alloc_wavefunction(HELIUM_MASS, "gwf");
#ifdef PREDICT_CORRECT
  gwfp = dft_driver_alloc_wavefunction(HELIUM_MASS, "gwfp");
#endif
    
  /* Allocate space for external potential */
  rworkspace = dft_driver_alloc_rgrid("rworkspace");
  rworkspace2 = dft_driver_alloc_rgrid("rworkspace2");
  rworkspace3 = dft_driver_alloc_rgrid("rworkspace3");
#ifdef PREDICT_CORRECT
  potential_store = dft_driver_alloc_cgrid("cworkspace"); /* temporary storage */
#endif

  /* Propagator (default FFT) */
//  dft_driver_kinetic = DFT_DRIVER_KINETIC_CN_PBC;  

  rho0 = dft_ot_bulk_density(dft_driver_otf);
  wave_params.rho = rho0;
  grid_wf_map(gwf, dft_common_planewave, &wave_params);

  lj_tf = dft_driver_get_workspace(1, 1);
  rd_tf = dft_driver_get_workspace(2, 1);
  density_tf = dft_driver_get_workspace(3, 1);
  spave_tf = dft_driver_get_workspace(4, 1);
  ot_pot = dft_driver_get_workspace(5, 1);  
  OT_INIT(lj_tf, rd_tf);

  rgrid_constant(rworkspace, rho0);
  OT_POT(ot_pot, rworkspace, density_tf, rworkspace2, spave_tf, lj_tf, rd_tf, rworkspace3);
  mu0 = dft_ot_bulk_chempot2(dft_driver_otf);
  fprintf(stderr, "Calculated mu0 = " FMT_R " K; Analytical mu0 = " FMT_R " K.\n", mu0 * GRID_AUTOK, dft_ot_bulk_chempot2(dft_driver_otf) * GRID_AUTOK);

  for(l = 0; l < iterations; l++) {
    grid_timer_start(&timer);

    grid_wf_density(gwf, rworkspace);
    OT_POT(ot_pot, rworkspace, density_tf, rworkspace2, spave_tf, lj_tf, rd_tf, rworkspace3);

#ifndef PREDICT_CORRECT
    (void) dft_driver_propagate(DFT_DRIVER_PROPAGATE_OTHER, ot_pot, mu0, gwf, TS, l);
#else
    (void) dft_driver_propagate_predict(DFT_DRIVER_PROPAGATE_OTHER, ot_pot, mu0, gwf, gwfp, potential_store, TS /* fs */, l);

    grid_wf_density(gwfp, rworkspace);
    OT_POT(ot_pot, rworkspace, density_tf, rworkspace2, spave_tf, lj_tf, rd_tf, rworkspace3);

    dft_driver_propagate_correct(DFT_DRIVER_PROPAGATE_OTHER, ot_pot, mu0, gwf, gwfp, potential_store, TS /* fs */, l);
#endif

    printf(FMT_R " %.20le\n", ((REAL) l) * TS, POW(CABS(cgrid_value_at_index(gwf->grid, 0, 0, N/2)), 2.0));
    fflush(stdout);
    fprintf(stderr, "One iteration = " FMT_R " wall clock seconds.\n", grid_timer_wall_clock_time(&timer));
//    if(!(l % 10000)) {
//      char buf[512];
//      grid_wf_density(gwf, rworkspace);
//      sprintf(buf, "disp-" FMT_I, l);
//      dft_driver_write_density(rworkspace, buf);
//    }
  }
  
  return 0;
}
