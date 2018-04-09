/*
 * Dynamics of a bubble (formed by external potential) travelling at
 * constant velocity in liquid helium (moving background). 
 * 
 */

#include "bubble.h"

REAL round_veloc(REAL veloc) {   // Round to fit the simulation box

  INT n;
  REAL v;

  n = (INT) (0.5 + (NX * STEP * HELIUM_MASS * VX) / (HBAR * 2.0 * M_PI));
  v = ((REAL) n) * HBAR * 2.0 * M_PI / (NX * STEP * HELIUM_MASS);
  fprintf(stderr, "Requested velocity = %le m/s.\n", veloc * GRID_AUTOMPS);
  fprintf(stderr, "Nearest velocity compatible with PBC = %le m/s.\n", v * GRID_AUTOMPS);
  return v;
}

REAL momentum(REAL vx) {

  return HELIUM_MASS * vx / HBAR;
}

/* -I * cabs(tstep) = full imag time, cabs(tstep) = full real time */
REAL complex tstep(REAL complex tstep, INT iter) {
 
  REAL x = ((REAL) iter) / (REAL) STARTING_ITER;

  return (-I * (1.0 - x) + x) * CABS(tstep);  // not called with x > 1
}

int main(int argc, char *argv[]) {

  wf3d *gwf, *gwfp;
  cgrid3d *cworkspace;
  rgrid3d *ext_pot;
#ifdef OUTPUT_GRID
  char filename[2048];
#endif
  REAL vx, mu0, kx, rho0;
  INT iter;
  extern void analyze(wf3d *, INT, REAL);
  extern REAL pot_func(void *, REAL, REAL, REAL);
  
  /* Setup DFT driver parameters */
  dft_driver_setup_grid(NX, NY, NZ, STEP, THREADS);
#ifdef CUDA
  cuda_enable(1);
//  cuda_debug(1);
#endif

  /* FFTW planner flags */
  grid_set_fftw_flags(FFTW_PLANNER);

  /* Plain Orsay-Trento in real or imaginary time */
  dft_driver_setup_model(FUNCTIONAL, DFT_DRIVER_IMAG_TIME, 0.0);  // will be changed later
  
  /* Regular boundaries */
  dft_driver_setup_boundary_type(DFT_DRIVER_BOUNDARY_REGULAR, 0.0, 0.0, 0.0, 0.0);
  dft_driver_setup_boundary_condition(DFT_DRIVER_BC_NORMAL);
  
  /* Initialize */
  dft_driver_initialize();

  dft_driver_kinetic = KINETIC_PROPAGATOR;
  if(dft_driver_kinetic == DFT_DRIVER_KINETIC_CN_NBC) fprintf(stderr, "Kinetic propagator = Crank-Nicolson\n");
  if(dft_driver_kinetic == DFT_DRIVER_KINETIC_FFT) fprintf(stderr, "Kinetic propagator = FFT\n"); 

  /* bulk normalization (requires the correct chem. pot.) */
  /* TODO: when vx != 0, mu0 is affected. For now just use bulk renormalization - v contributes to mu0? */
//  dft_driver_setup_normalization(DFT_DRIVER_DONT_NORMALIZE, 4, 0.0, 0);
  dft_driver_setup_normalization(DFT_DRIVER_NORMALIZE_BULK, 4, 0.0, 0);
  
  /* get bulk density and chemical potential */
  rho0 = dft_ot_bulk_density_pressurized(dft_driver_otf, PRESSURE);
  dft_driver_otf->rho0 = rho0;
  mu0  = dft_ot_bulk_chempot_pressurized(dft_driver_otf, PRESSURE);
  fprintf(stderr, "rho0 = " FMT_R " Angs^-3, mu0 = " FMT_R " K.\n", rho0 / (0.529 * 0.529 * 0.529), mu0 * GRID_AUTOK);
  
  /* Allocate wavefunctions and grids */
  gwf = dft_driver_alloc_wavefunction(HELIUM_MASS, "gwf");  /* order parameter for current time (He liquid) */
  gwfp = dft_driver_alloc_wavefunction(HELIUM_MASS, "gwfp"); /* order parameter for future (predict) (He liquid) */

  cworkspace = dft_driver_alloc_cgrid("cworkspace");             /* allocate complex workspace */
  ext_pot = dft_driver_alloc_rgrid("ext_pot");                /* allocate real external potential grid */
  
  /* Setup frame of reference momentum (for both imaginary & real time) */
  vx = round_veloc(VX);     /* Round velocity to fit the spatial grid */
  kx = momentum(vx);
  dft_driver_setup_momentum(kx, 0.0, 0.0);
  cgrid3d_set_momentum(gwf->grid, kx, 0.0, 0.0);
  cgrid3d_set_momentum(gwfp->grid, kx, 0.0, 0.0);
  cgrid3d_set_momentum(cworkspace, kx, 0.0, 0.0);

  fprintf(stderr, "Time step in fs   = " FMT_R "\n", TIME_STEP);
  fprintf(stderr, "Time step in a.u. = " FMT_R "\n", TIME_STEP / GRID_AUTOFS);
  fprintf(stderr, "Relative velocity = (" FMT_R ", " FMT_R ", " FMT_R ") (au)\n", vx, 0.0, 0.0);
  fprintf(stderr, "Relative velocity = (" FMT_R ", " FMT_R ", " FMT_R ") (A/ps)\n", 
		  vx * 1000.0 * GRID_AUTOANG / GRID_AUTOFS, 0.0, 0.0);
  fprintf(stderr, "Relative velocity = (" FMT_R ", " FMT_R ", " FMT_R ") (m/s)\n", vx * GRID_AUTOMPS, 0.0, 0.0);

  rgrid3d_map(ext_pot, pot_func, NULL); /* External potential */
  rgrid3d_add(ext_pot, -mu0);

  dft_driver_setup_model(FUNCTIONAL, DFT_DRIVER_USER_TIME, rho0);  /* mixed real & imag time iterations for warm up */
  dft_driver_setup_boundary_type(DFT_DRIVER_BOUNDARY_REGULAR, 0.0, 0.0, 0.0, 0.0);

  if(argc == 1) {
    /* Mixed Imaginary & Real time iterations */
    fprintf(stderr, "Warm up iterations.\n");
    for(iter = 0; iter < STARTING_ITER; iter++) {
      (void) dft_driver_propagate_predict(DFT_DRIVER_PROPAGATE_HELIUM, ext_pot, gwf, gwfp, cworkspace, tstep(TIME_STEP, iter), iter); /* PREDICT */ 
      (void) dft_driver_propagate_correct(DFT_DRIVER_PROPAGATE_HELIUM, ext_pot, gwf, gwfp, cworkspace, tstep(TIME_STEP, iter), iter); /* CORRECT */ 
    }
  } else { /* restart from a file (.grd) */
    fprintf(stderr, "Continuing from checkpoint file %s.\n", argv[1]);
    dft_driver_read_grid(gwf->grid, argv[1]);
  }

  /* Real time iterations */
  dft_driver_setup_model(FUNCTIONAL, DFT_DRIVER_REAL_TIME, rho0);
#if KINETIC_PROPAGATOR == DFT_DRIVER_KINETIC_FFT
  fprintf(stderr, "FFT propagator, no absorbing boundaries.\n");
  dft_driver_setup_boundary_type(DFT_DRIVER_BOUNDARY_REGULAR, 0.0, 0.0, 0.0, 0.0);
#else
  dft_driver_setup_boundary_type(DFT_DRIVER_BOUNDARY_ITIME, 1.0, ABS_WIDTH_X, ABS_WIDTH_Y, ABS_WIDTH_Z);
  fprintf(stderr, "Absorption begins at (" FMT_R "," FMT_R "," FMT_R ") Bohr from the boundary\n",  ABS_WIDTH_X, ABS_WIDTH_Y, ABS_WIDTH_Z);
#endif

  fprintf(stderr, "Real time propagation.\n");
  for(iter = 0; iter < MAXITER; iter++) {
    if(!(iter % OUTPUT_ITER)) {   /* every OUTPUT_ITER iterations, write output */
#ifdef OUTPUT_GRID
      sprintf(filename, "liquid-" FMT_I, iter);
      dft_driver_write_grid(gwf->grid, filename);
#endif
      analyze(gwf, iter, vx);
    }
    (void) dft_driver_propagate_predict(DFT_DRIVER_PROPAGATE_HELIUM, ext_pot, gwf, gwfp, cworkspace, TIME_STEP, iter); /* PREDICT */ 
    (void) dft_driver_propagate_correct(DFT_DRIVER_PROPAGATE_HELIUM, ext_pot, gwf, gwfp, cworkspace, TIME_STEP, iter); /* CORRECT */ 
  }

  return 0;
}
