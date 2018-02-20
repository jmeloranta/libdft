/*
 * Dynamics of a bubble (formed by external potential) travelling at
 * constant velocity in liquid helium (moving background). 
 * 
 * All input in a.u. except the time step, which is in fs.
 *
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <complex.h>
#include <grid/grid.h>
#include <grid/au.h>
#include <dft/dft.h>
#include <dft/ot.h>

#define TIME_STEP_IMAG 30.0             /* Time step in imag iterations (fs) */
#define TIME_STEP_REAL 30.0             /* Time step for real time iterations (fs) */
#define FUNCTIONAL (DFT_OT_PLAIN)       /* Functional to be used (could add DFT_OT_KC and/or DFT_OT_BACKFLOW) */
#define STARTING_TIME 400000.0           /* Start real time simulation at this time (fs) - 10 ps (was 10,000) */
#define STARTING_ITER ((long) (STARTING_TIME / TIME_STEP_IMAG))
#define MAXITER 80000000                /* Maximum number of real time iterations */
#define OUTPUT_TIME 2500.0              /* Output interval time (fs) (2500) */
#define OUTPUT_ITER ((long) (OUTPUT_TIME / TIME_STEP_REAL))
#define VX (70.0 / GRID_AUTOMPS)        /* Flow velocity (m/s) */
#define PRESSURE (0.0 / GRID_AUTOBAR)   /* External pressure in bar (normal = 0) */

#define THREADS 0	/* # of parallel threads to use (0 = all) */
#define NX 512       	/* # of grid points along x */
#define NY 256          /* # of grid points along y */
#define NZ 256        	/* # of grid points along z */
#define STEP 2.0        /* spatial step length (Bohr) */
#define ABS_WIDTH_X 40.0  /* Width of the absorbing boundary */
#define ABS_WIDTH_Y 20.0  /* Width of the absorbing boundary */
#define ABS_WIDTH_Z 20.0  /* Width of the absorbing boundary */

/* #define KINETIC_PROPAGATOR DFT_DRIVER_KINETIC_FFT      /* FFT (unstable) */
#define KINETIC_PROPAGATOR DFT_DRIVER_KINETIC_CN_NBC /* Crank-Nicolson */

#define FFTW_PLANNER 2 /* 0: FFTW_ESTIMATE, 1: FFTW_MEASURE (default), 2: FFTW_PATIENT, 3: FFTW_EXHAUSTIVE */

/* Bubble parameters using exponential repulsion (approx. electron bubble) - RADD = 19.0 */
#define A0 (3.8003E5 / GRID_AUTOK)
#define A1 (1.6245 * GRID_AUTOANG)
#define A2 0.0
#define A3 0.0
#define A4 0.0
#define A5 0.0
#define RMIN 2.0
#define RADD 6.0

#define HELIUM_MASS (4.002602 / GRID_AUTOAMU) /* helium mass in AMU */

double global_time, rho0;
long iter;

double round_veloc(double veloc) {   // Round to fit the simulation box

  long n;
  double v;

  n = (long) (0.5 + (NX * STEP * HELIUM_MASS * VX) / (HBAR * 2.0 * M_PI));
  v = ((double) n) * HBAR * 2.0 * M_PI / (NX * STEP * HELIUM_MASS);
  fprintf(stderr, "Requested velocity = %le m/s.\n", veloc * GRID_AUTOMPS);
  fprintf(stderr, "Nearest velocity compatible with PBC = %le m/s.\n", v * GRID_AUTOMPS);
  return v;
}

double momentum(double vx) {

  return HELIUM_MASS * vx / HBAR;
}

double pot_func(void *NA, double x, double y, double z) {

  double r, r2, r4, r6, r8, r10;

  r = sqrt(x * x + y * y + z * z);
  r -= RADD;
  if(r < RMIN) r = RMIN;

  r2 = r * r;
  r4 = r2 * r2;
  r6 = r4 * r2;
  r8 = r6 * r2;
  r10 = r8 * r2;
  return A0 * exp(-A1 * r) - A2 / r4 - A3 / r6 - A4 / r8 - A5 / r10;
}

/* -I * cabs(tstep) = full imag time, cabs(tstep) = full real time */
double complex tstep_func(double complex tstep, long i, long j, long k) {
 
  double x = ((double) iter) / (double) STARTING_ITER;

  return (-I * (1.0 - x) + x) * cabs(tstep);  // not called with x > 1
}

int main(int argc, char *argv[]) {

  wf3d *gwf, *gwfp;
  cgrid3d *cworkspace;
  rgrid3d *ext_pot;
  char filename[2048];
  double vx, mu0, kx;
  FILE *fp;
  
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
  fprintf(stderr, "rho0 = %le Angs^-3, mu0 = %le K.\n", rho0 / (0.529 * 0.529 * 0.529), mu0 * GRID_AUTOK);
  
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

  fprintf(stderr, "Imaginary time step in a.u. = %le\n", TIME_STEP_IMAG / GRID_AUTOFS);
  fprintf(stderr, "Real time step in a.u. = %le\n", TIME_STEP_REAL / GRID_AUTOFS);
  fprintf(stderr, "Relative velocity = (%le, %le, %le) (au)\n", vx, 0.0, 0.0);
  fprintf(stderr, "Relative velocity = (%le, %le, %le) (A/ps)\n", 
		  vx * 1000.0 * GRID_AUTOANG / GRID_AUTOFS, 0.0, 0.0);
  fprintf(stderr, "Relative velocity = (%le, %le, %le) (m/s)\n", vx * GRID_AUTOMPS, 0.0, 0.0);

  rgrid3d_map(ext_pot, pot_func, NULL); /* External potential */
  rgrid3d_add(ext_pot, -mu0);

  if(!(fp = fopen("liquid.dat", "w"))) {
    fprintf(stderr, "Can't open parameter file for writing.\n");
    exit(1);
  }
  fprintf(fp, "%le\n", TIME_STEP_REAL);    /* real time simulation time step */
  fprintf(fp, "%le\n", vx * GRID_AUTOMPS); /* velocity */
  fprintf(fp, "%ld\n", OUTPUT_ITER);       /* output interval */
  fprintf(fp, "%le\n", A0);                /* potential params */
  fprintf(fp, "%le\n", A1);                
  fprintf(fp, "%le\n", A2);                
  fprintf(fp, "%le\n", A3);                
  fprintf(fp, "%le\n", A4);                
  fprintf(fp, "%le\n", A5);                
  fprintf(fp, "%le\n", RMIN);              
  fprintf(fp, "%le\n", RADD);
  fclose(fp);

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
  } else { /* restart from a fie (.grd) */
    fprintf(stderr, "Continuing from checkpoint file %s.\n", argv[1]);
    dft_driver_read_grid(gwf->grid, argv[1]);
  }

  /* Real time iterations */
  fprintf(stderr, "Absorption begins at (%le,%le,%le) Bohr from the boundary\n",  ABS_WIDTH_X, ABS_WIDTH_Y, ABS_WIDTH_Z);

  fprintf(stderr, "Real time propagation.\n");
  dft_driver_bc_function = NULL;
  for(iter = 0; iter < MAXITER; iter++) {
    if(!(iter % OUTPUT_ITER)) {   /* every OUTPUT_ITER iterations, write output */
      sprintf(filename, "liquid-%ld", iter);
      dft_driver_write_grid(gwf->grid, filename);
      fflush(stdout);
    }
    (void) dft_driver_propagate_predict(DFT_DRIVER_PROPAGATE_HELIUM, ext_pot, gwf, gwfp, cworkspace, TIME_STEP_REAL, iter); /* PREDICT */ 
    (void) dft_driver_propagate_correct(DFT_DRIVER_PROPAGATE_HELIUM, ext_pot, gwf, gwfp, cworkspace, TIME_STEP_REAL, iter); /* CORRECT */ 
  }

  return 0;
}
