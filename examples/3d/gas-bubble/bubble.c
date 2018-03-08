/*
 * Dynamics of a bubble (formed by external potential) travelling at
 * constant velocity in liquid helium (moving background). 
 * 
 */

#include "bubble.h"

REAL global_time, rho0;
INT iter;

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
REAL complex tstep_func(void *asd, REAL complex tstep, INT i, INT j, INT k) {
 
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
  REAL vx, mu0, kx;
  extern REAL pot_func(void *, REAL, REAL, REAL);
  extern void analyze(wf3d *, INT, REAL);
  
  /* Setup DFT driver parameters */
  dft_driver_setup_grid(NX, NY, NZ, STEP, THREADS);

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
  gwf = dft_driver_alloc_wavefunction(HELIUM_MASS);  /* order parameter for current time (He liquid) */
  gwfp = dft_driver_alloc_wavefunction(HELIUM_MASS); /* order parameter for future (predict) (He liquid) */

  cworkspace = dft_driver_alloc_cgrid();             /* allocate complex workspace */
  ext_pot = dft_driver_alloc_rgrid();                /* allocate real external potential grid */
  
  /* Setup frame of reference momentum (for both imaginary & real time) */
  vx = round_veloc(VX);     /* Round velocity to fit the spatial grid */
  kx = momentum(vx);
  dft_driver_setup_momentum(kx, 0.0, 0.0);
  cgrid3d_set_momentum(gwf->grid, kx, 0.0, 0.0);
  cgrid3d_set_momentum(gwfp->grid, kx, 0.0, 0.0);
  cgrid3d_set_momentum(cworkspace, kx, 0.0, 0.0);

  fprintf(stderr, "Imaginary time step in a.u. = " FMT_R "\n", TIME_STEP_IMAG / GRID_AUTOFS);
  fprintf(stderr, "Real time step in a.u. = " FMT_R "\n", TIME_STEP_REAL / GRID_AUTOFS);
  fprintf(stderr, "Relative velocity = (" FMT_R ", " FMT_R ", " FMT_R ") (au)\n", vx, 0.0, 0.0);
  fprintf(stderr, "Relative velocity = (" FMT_R ", " FMT_R ", " FMT_R ") (A/ps)\n", 
		  vx * 1000.0 * GRID_AUTOANG / GRID_AUTOFS, 0.0, 0.0);
  fprintf(stderr, "Relative velocity = (" FMT_R ", " FMT_R ", " FMT_R ") (m/s)\n", vx * GRID_AUTOMPS, 0.0, 0.0);

  rgrid3d_map(ext_pot, pot_func, NULL); /* External potential */
  rgrid3d_add(ext_pot, -mu0);

  dft_driver_setup_model(FUNCTIONAL, DFT_DRIVER_REAL_TIME, rho0);  /* mixed real & imag time iterations for warm up */
  dft_driver_setup_boundary_type(DFT_DRIVER_BOUNDARY_ITIME, 0.2, ABS_WIDTH_X, ABS_WIDTH_Y, ABS_WIDTH_Z);

  if(argc == 1) {
    /* Mixed Imaginary & Real time iterations */
    fprintf(stderr, "Warm up iterations.\n");
    dft_driver_bc_function = &tstep_func;
    for(iter = 0; iter < STARTING_ITER; iter++) {
      (void) dft_driver_propagate_predict(DFT_DRIVER_PROPAGATE_HELIUM, ext_pot, gwf, gwfp, cworkspace, TIME_STEP_IMAG, iter); /* PREDICT */ 
      (void) dft_driver_propagate_correct(DFT_DRIVER_PROPAGATE_HELIUM, ext_pot, gwf, gwfp, cworkspace, TIME_STEP_IMAG, iter); /* CORRECT */ 
    }
  } else { /* restart from a file (.grd) */
    fprintf(stderr, "Continuing from checkpoint file %s.\n", argv[1]);
    dft_driver_read_grid(gwf->grid, argv[1]);
  }

  /* Real time iterations */
  fprintf(stderr, "Absorption begins at (" FMT_R "," FMT_R "," FMT_R ") Bohr from the boundary\n",  ABS_WIDTH_X, ABS_WIDTH_Y, ABS_WIDTH_Z);

  fprintf(stderr, "Real time propagation.\n");
  dft_driver_bc_function = NULL;
  for(iter = 0; iter < MAXITER; iter++) {
    if(!(iter % OUTPUT_ITER)) {   /* every OUTPUT_ITER iterations, write output */
#ifdef OUTPUT_GRID
      sprintf(filename, "liquid-" FMT_I, iter);
      dft_driver_write_grid(gwf->grid, filename);
#endif
      analyze(gwf, iter, vx);
    }
    (void) dft_driver_propagate_predict(DFT_DRIVER_PROPAGATE_HELIUM, ext_pot, gwf, gwfp, cworkspace, TIME_STEP_REAL, iter); /* PREDICT */ 
    (void) dft_driver_propagate_correct(DFT_DRIVER_PROPAGATE_HELIUM, ext_pot, gwf, gwfp, cworkspace, TIME_STEP_REAL, iter); /* CORRECT */ 
  }

  return 0;
}
