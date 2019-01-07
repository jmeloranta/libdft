/*
 * Dynamics of a bubble (formed by external potential) travelling at
 * constant velocity in liquid helium (moving background). 
 * 
 */

#include "bubble.h"

REAL round_veloc(REAL veloc) {   // Round to fit the simulation box

  INT n;
  REAL v;

  n = (INT) (0.5 + (NX * STEP * HELIUM_MASS * veloc) / (HBAR * 2.0 * M_PI));
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

/* -I * cabs(tstep) = full imag time, cabs(tstep) = full real time */
REAL complex tstep2(REAL complex tstep, INT iter) {
 
  return CABS(tstep);
//  return CABS(tstep) * (1.0 - 3E-2 * I);  // introduce small imag part
}

int main(int argc, char *argv[]) {

  wf *gwf;
#ifdef PC
  wf *gwfp;
  cgrid *cworkspace;
#endif
#ifdef SM
  rgrid *ext_pot;
#endif
#ifdef OUTPUT_GRID
  char filename[2048];
#endif
  REAL vx, mu0, kx, rho0;
  INT iter;
  extern void analyze(wf *, INT, REAL);
  extern REAL pot_func(void *, REAL, REAL, REAL);
  
  /* Setup DFT driver parameters */
  dft_driver_setup_grid(NX, NY, NZ, STEP, THREADS);
#ifdef USE_CUDA
  cuda_enable(1);
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
  dft_driver_setup_normalization(DFT_DRIVER_DONT_NORMALIZE, 4, 0.0, 0);
  
  /* get bulk density and chemical potential */
  rho0 = dft_ot_bulk_density_pressurized(dft_driver_otf, PRESSURE);
  dft_driver_otf->rho0 = rho0;
  mu0 = dft_ot_bulk_chempot_pressurized(dft_driver_otf, PRESSURE);
  fprintf(stderr, "Pressure = %le\n", PRESSURE * GRID_AUTOBAR);
  fprintf(stderr, "rho0 = " FMT_R " Angs^-3, mu0 = " FMT_R " K.\n", rho0 / (0.529 * 0.529 * 0.529), mu0 * GRID_AUTOK);
  
  /* Allocate wavefunctions and grids */
  gwf = dft_driver_alloc_wavefunction(HELIUM_MASS, "gwf");  /* order parameter for current time (He liquid) */
#ifdef PC
  gwfp = dft_driver_alloc_wavefunction(HELIUM_MASS, "gwfp"); /* order parameter for future (predict) (He liquid) */
  cworkspace = dft_driver_alloc_cgrid("cworkspace");             /* allocate complex workspace */
#endif
#ifdef SM
  ext_pot = dft_driver_alloc_rgrid("ext_pot");                /* allocate real external potential grid */
#else
  dft_driver_setup_potential(RMIN, RADD, A0, A1, A2, A3, A4, A5);
#endif
  fprintf(stderr, "Potential: RMIN = " FMT_R ", RADD = " FMT_R ", A0 = " FMT_R ", A1 = " FMT_R ", A2 = " FMT_R ", A3 = " FMT_R ", A4 = " FMT_R ", A5 = " FMT_R "\n", RMIN, RADD, A0, A1, A2, A3, A4, A5);
  
  /* Setup frame of reference momentum (for both imaginary & real time) */
  vx = round_veloc(VX);     /* Round velocity to fit the spatial grid */
  kx = momentum(vx);
  dft_driver_setup_momentum(kx, 0.0, 0.0);
  cgrid_set_momentum(gwf->grid, kx, 0.0, 0.0);
#ifdef PC
  cgrid_set_momentum(gwfp->grid, kx, 0.0, 0.0);
  cgrid_set_momentum(cworkspace, kx, 0.0, 0.0);
#endif

  fprintf(stderr, "Time step in fs   = " FMT_R "\n", TIME_STEP);
  fprintf(stderr, "Time step in a.u. = " FMT_R "\n", TIME_STEP / GRID_AUTOFS);
  fprintf(stderr, "Relative velocity = (" FMT_R ", " FMT_R ", " FMT_R ") (au)\n", vx, 0.0, 0.0);
  fprintf(stderr, "Relative velocity = (" FMT_R ", " FMT_R ", " FMT_R ") (A/ps)\n", 
		  vx * 1000.0 * GRID_AUTOANG / GRID_AUTOFS, 0.0, 0.0);
  fprintf(stderr, "Relative velocity = (" FMT_R ", " FMT_R ", " FMT_R ") (m/s)\n", vx * GRID_AUTOMPS, 0.0, 0.0);

#ifdef SM
#if SM == 0
  rgrid_map(ext_pot, pot_func, NULL); /* External potential */
#else
  fprintf(stderr, "Smooth mapping = " FMT_I ".\n", (INT) SM);
  rgrid_smooth_map(ext_pot, pot_func, NULL, SM); /* External potential */
#endif
#endif

  dft_driver_setup_model(FUNCTIONAL, DFT_DRIVER_USER_TIME, rho0);  /* mixed real & imag time iterations for warm up */
  dft_driver_setup_boundary_type(DFT_DRIVER_BOUNDARY_REGULAR, 0.0, 0.0, 0.0, 0.0);

  if(argc == 1) {
    /* Mixed Imaginary & Real time iterations */
    fprintf(stderr, "Warm up iterations.\n");
    for(iter = 0; iter < STARTING_ITER; iter++) {
#ifdef PC
#ifdef SM
      (void) dft_driver_propagate_predict(DFT_DRIVER_PROPAGATE_HELIUM, ext_pot, mu0, gwf, gwfp, cworkspace, tstep(TIME_STEP, iter), iter); /* PREDICT */ 
      (void) dft_driver_propagate_correct(DFT_DRIVER_PROPAGATE_HELIUM, ext_pot, mu0, gwf, gwfp, cworkspace, tstep(TIME_STEP, iter), iter); /* CORRECT */ 
#else
      (void) dft_driver_propagate_predict(DFT_DRIVER_PROPAGATE_HELIUM, NULL, mu0, gwf, gwfp, cworkspace, tstep(TIME_STEP, iter), iter); /* PREDICT */ 
      (void) dft_driver_propagate_correct(DFT_DRIVER_PROPAGATE_HELIUM, NULL, mu0, gwf, gwfp, cworkspace, tstep(TIME_STEP, iter), iter); /* CORRECT */ 
#endif
#else
#ifdef SM
      (void) dft_driver_propagate(DFT_DRIVER_PROPAGATE_HELIUM, ext_pot, mu0, gwf, tstep(TIME_STEP, iter), iter);
#else
      (void) dft_driver_propagate(DFT_DRIVER_PROPAGATE_HELIUM, NULL, mu0, gwf, tstep(TIME_STEP, iter), iter);
#endif
#endif
    }
    iter = 0;
  } else { /* restart from a file (.grd) */
    sscanf(argv[1], "bubble-" FMT_I ".grd", &iter);
    fprintf(stderr, "Continuing from checkpoint file %s at iteration " FMT_I ".\n", argv[1], iter);
    dft_driver_read_grid(gwf->grid, argv[1]);
  }

  /* Real time iterations */
  dft_driver_setup_model(FUNCTIONAL, DFT_DRIVER_USER_TIME, rho0);
#if KINETIC_PROPAGATOR == DFT_DRIVER_KINETIC_FFT
//  dft_driver_setup_boundary_type(DFT_DRIVER_BOUNDARY_ITIME, ABS_AMP, ABS_WIDTH_X, ABS_WIDTH_Y, ABS_WIDTH_Z);
//  fprintf(stderr, "Absorption begins at (" FMT_R "," FMT_R "," FMT_R ") Bohr from the boundary\n",  ABS_WIDTH_X, ABS_WIDTH_Y, ABS_WIDTH_Z);
  fprintf(stderr, "FFT propagator, no absorbing boundaries.\n");
  dft_driver_setup_boundary_type(DFT_DRIVER_BOUNDARY_REGULAR, 0.0, 0.0, 0.0, 0.0);
#else
  dft_driver_setup_boundary_type(DFT_DRIVER_BOUNDARY_ITIME, ABS_AMP, ABS_WIDTH_X, ABS_WIDTH_Y, ABS_WIDTH_Z);
  fprintf(stderr, "Absorption begins at (" FMT_R "," FMT_R "," FMT_R ") Bohr from the boundary\n",  ABS_WIDTH_X, ABS_WIDTH_Y, ABS_WIDTH_Z);
#endif

  fprintf(stderr, "Real time propagation.\n");
  for( ; iter < MAXITER; iter++) {
#ifdef OUTPUT_GRID
    if(!(iter % OUTPUT_GRID)) {
      sprintf(filename, "liquid-" FMT_R, ((REAL) iter) * TIME_STEP);
      dft_driver_write_grid(gwf->grid, filename);
    }
#endif
    if(!(iter % OUTPUT_ITER)) analyze(gwf, iter, vx);
#ifdef PC
#ifdef SM
    (void) dft_driver_propagate_predict(DFT_DRIVER_PROPAGATE_HELIUM, ext_pot, mu0, gwf, gwfp, cworkspace, tstep2(TIME_STEP, iter), iter); /* PREDICT */ 
    (void) dft_driver_propagate_correct(DFT_DRIVER_PROPAGATE_HELIUM, ext_pot, mu0, gwf, gwfp, cworkspace, tstep2(TIME_STEP, iter), iter); /* CORRECT */ 
#else
    (void) dft_driver_propagate_predict(DFT_DRIVER_PROPAGATE_HELIUM, NULL, mu0, gwf, gwfp, cworkspace, tstep2(TIME_STEP, iter), iter); /* PREDICT */ 
    (void) dft_driver_propagate_correct(DFT_DRIVER_PROPAGATE_HELIUM, NULL, mu0, gwf, gwfp, cworkspace, tstep2(TIME_STEP, iter), iter); /* CORRECT */ 
#endif
#else
#ifdef SM
    (void) dft_driver_propagate(DFT_DRIVER_PROPAGATE_HELIUM, ext_pot, mu0, gwf, tstep2(TIME_STEP, iter), iter);
#else
    (void) dft_driver_propagate(DFT_DRIVER_PROPAGATE_HELIUM, NULL, mu0, gwf, tstep2(TIME_STEP, iter), iter);
#endif
#endif
  }

  return 0;
}
