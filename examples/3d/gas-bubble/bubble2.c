/*
 * Dynamics of a gas bubble (formed by external potential) travelling at
 * constant velocity in liquid helium. 
 * 
 * All input in a.u. except the time step, which is fs.
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

/* Only imaginary time */
#define TIME_STEP 50.0	/* Time step in fs (5 for real, 10 for imag) */
#define FUNCTIONAL (DFT_OT_PLAIN) /* Functional to be used (DFT_OT_PLAIN or DFT_GP) */
#define STARTING_ITER 200 /* Starting real time iterations (was 200) */
#define MAXITER (800000 + STARTING_ITER) /* Maximum number of iterations (was 300) */
#define OUTPUT     250	/* output every this iteration (was 250) */
#define ABS_WIDTH  20.0 /* Width of the absorbing boundary */

#define THREADS 0	/* # of parallel threads to use (0 = all) */
#define NX 512       	/* # of grid points along x */
#define NY 128          /* # of grid points along y */
#define NZ 128        	/* # of grid points along z */
#define STEP 2.0        /* spatial step length (Bohr) */

#define PRESSURE (1.0 / GRID_AUTOBAR)   /* External pressure in bar */
#define HELIUM_MASS (4.002602 / GRID_AUTOAMU) /* helium mass */

/* VISCOSITY * RHON */
//#define VISCOSITY (1.306E-6 * 0.162)
#define TEMP 1.6

/* Bubble parameters using exponential repulsion (approx. electron bubble) - RADD = 19.0 */
#define A0 (3.8003E5 / GRID_AUTOK)
#define A1 (1.6245 * GRID_AUTOANG)
#define A2 0.0
#define A3 0.0
#define A4 0.0
#define A5 0.0
#define RMIN 2.0
#define RADD 6.0

/* velocity components for the gas (m/s) */
#define VX	(80.0 / GRID_AUTOMPS)

double global_time, rho0;

double round_veloc(double veloc) {   // Round to fit the simulation box

  long n;
  double v;

  n = (long) (0.5 + (NX * STEP * HELIUM_MASS * VX) / (HBAR * 2.0 * M_PI));
  v = ((double) n) * HBAR * 2.0 * M_PI / (NX * STEP * HELIUM_MASS);
  printf("Requested velocity = %le m/s.\n", veloc * GRID_AUTOMPS);
  printf("Nearest velocity compatible with PBC = %le m/s.\n", v * GRID_AUTOMPS);
  return v;
}

double momentum(double vx) {

  return HELIUM_MASS * vx / HBAR;
}

/* Impurity must always be at the origin (dU/dx) */
double dpot_func(void *NA, double x, double y, double z) {

  double r, rp, r2, r3, r5, r7, r9, r11;

  rp = sqrt(x * x + y * y + z * z);
  r = rp - RADD;
  if(r < RMIN) return 0.0;

  r2 = r * r;
  r3 = r2 * r;
  r5 = r2 * r3;
  r7 = r5 * r2;
  r9 = r7 * r2;
  r11 = r9 * r2;
  
  return (x / rp) * (-A0 * A1 * exp(-A1 * r) + 4.0 * A2 / r5 + 6.0 * A3 / r7 + 8.0 * A4 / r9 + 10.0 * A5 / r11);
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

int main(int argc, char *argv[]) {

  wf3d *gwf, *gwfp;
  cgrid3d *cworkspace;
  rgrid3d *ext_pot;
  long iter;
  char filename[2048];
  double vx, mu0, kx;
  FILE *fp;
  
  /* Setup DFT driver parameters (256 x 256 x 256 grid) */
  dft_driver_setup_grid(NX, NY, NZ, STEP, THREADS);

  /* Setup frame of reference momentum */
  dft_driver_setup_momentum(0.0, 0.0, 0.0);

  /* Plain Orsay-Trento in real or imaginary time */
  dft_driver_setup_model(FUNCTIONAL, DFT_DRIVER_IMAG_TIME, 0.0);
  
  /* Regular boundaries */
  dft_driver_setup_boundary_type(DFT_DRIVER_BOUNDARY_REGULAR, 0.0, 0.0);
  dft_driver_setup_boundary_condition(DFT_DRIVER_BC_NORMAL);
  
  /* Initialize */
  dft_driver_initialize();

  /* bulk normalization (requires the correct chem. pot.) */
  dft_driver_setup_normalization(DFT_DRIVER_DONT_NORMALIZE, 4, 0.0, 0);
  
  /* get bulk density and chemical potential */
  rho0 = dft_ot_bulk_density_pressurized(dft_driver_otf, PRESSURE);
  dft_driver_otf->rho0 = rho0;
  mu0  = dft_ot_bulk_chempot_pressurized(dft_driver_otf, PRESSURE);
  printf("rho0 = %le Angs^-3, mu0 = %le K.\n", rho0 / (0.529 * 0.529 * 0.529), mu0 * GRID_AUTOK);

  /* Round velocity to fit the spatial grid */
  vx = round_veloc(VX);
  
  /* Allocate wavefunctions and grids */
  gwf = dft_driver_alloc_wavefunction(HELIUM_MASS);  /* order parameter for current time (He liquid) */
  gwfp = dft_driver_alloc_wavefunction(HELIUM_MASS); /* order parameter for future (predict) (He liquid) */

  cworkspace = dft_driver_alloc_cgrid();             /* allocate complex workspace */
  ext_pot = dft_driver_alloc_rgrid();                /* allocate real external potential grid */
  
  fprintf(stderr, "Time step in a.u. = %le\n", TIME_STEP / GRID_AUTOFS);
  fprintf(stderr, "Relative velocity = (%le, %le, %le) (au)\n", vx, 0.0, 0.0);
  fprintf(stderr, "Relative velocity = (%le, %le, %le) (A/ps)\n", 
		  vx * 1000.0 * GRID_AUTOANG / GRID_AUTOFS, 0.0, 0.0);
  fprintf(stderr, "Relative velocity = (%le, %le, %le) (m/s)\n", vx * GRID_AUTOMPS, 0.0, 0.0);

#ifdef VISCOSITY
  fprintf(stderr, "Reynolds # = %le\n", rho0 * vx * 20E-10 / (VISCOSITY / GRID_AUTOPAS));
#endif
  
  rgrid3d_map(ext_pot, pot_func, NULL); /* External potential */
  rgrid3d_add(ext_pot, -mu0);

  if(!(fp = fopen("liquid.dat", "w"))) {
    fprintf(stderr, "Can't open parameter file for writing.\n");
    exit(1);
  }
  fprintf(fp, "%le\n", TIME_STEP / 5.0);   /* real time simulation time step */
  fprintf(fp, "%le\n", vx * GRID_AUTOMPS); /* velocity */
  fprintf(fp, "%d\n", OUTPUT);            /* output interval */
  fprintf(fp, "%le\n", A0);                /* potential params */
  fprintf(fp, "%le\n", A1);                
  fprintf(fp, "%le\n", A2);                
  fprintf(fp, "%le\n", A3);                
  fprintf(fp, "%le\n", A4);                
  fprintf(fp, "%le\n", A5);                
  fprintf(fp, "%le\n", RMIN);              
  fprintf(fp, "%le\n", RADD);
  fclose(fp);

  /* Imaginary iterations */
  fprintf(stderr, "Imag time mode.\n");
  dft_driver_setup_model(FUNCTIONAL, DFT_DRIVER_IMAG_TIME, rho0);  /* imag time */
  for(iter = 0; iter < STARTING_ITER; iter++) {
    (void) dft_driver_propagate_predict(DFT_DRIVER_PROPAGATE_HELIUM, ext_pot, gwf, gwfp, cworkspace, TIME_STEP, iter); /* PREDICT */ 
    (void) dft_driver_propagate_correct(DFT_DRIVER_PROPAGATE_HELIUM, ext_pot, gwf, gwfp, cworkspace, TIME_STEP, iter); /* CORRECT */ 
  }

  /* Real time iterations */
  kx = momentum(vx);
  dft_driver_setup_momentum(kx, 0.0, 0.0);
  cgrid3d_set_momentum(gwf->grid, kx, 0.0, 0.0);
  cgrid3d_set_momentum(gwfp->grid, kx, 0.0, 0.0);
  cgrid3d_set_momentum(cworkspace, kx, 0.0, 0.0);
  dft_driver_setup_boundary_type(DFT_DRIVER_BOUNDARY_ITIME, 0.2, ABS_WIDTH);
  fprintf(stderr, "Absorption begins at %le Bohr from the boundary\n",  ABS_WIDTH);
  dft_driver_setup_model(FUNCTIONAL, DFT_DRIVER_REAL_TIME, rho0);  /* real time */
  fprintf(stderr, "Real time mode with time step %le fs.\n", TIME_STEP / 5.0);
  /* viscosity */
#ifdef VISCOSITY
  fprintf(stderr,"Viscosity using precomputed alpha. with T = %le\n", TEMP);
  dft_driver_setup_viscosity(VISCOSITY, 1.72 + 2.32E-10*exp(11.15*TEMP));
#endif
  for(iter = STARTING_ITER; iter < MAXITER; iter++) {
    
    (void) dft_driver_propagate_predict(DFT_DRIVER_PROPAGATE_HELIUM, ext_pot, gwf, gwfp, cworkspace, TIME_STEP/5.0, iter); /* PREDICT */ 
    (void) dft_driver_propagate_correct(DFT_DRIVER_PROPAGATE_HELIUM, ext_pot, gwf, gwfp, cworkspace, TIME_STEP/5.0, iter); /* CORRECT */ 
    
    if(!(iter % OUTPUT)) {   /* every OUTPUT iterations, write output */
      sprintf(filename, "liquid-%ld", iter);
      dft_driver_write_grid(gwf->grid, filename);
      fflush(stdout);
    }
  }
  return 0;
}
