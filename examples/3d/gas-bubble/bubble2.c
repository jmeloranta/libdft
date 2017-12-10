/*
 * Stationary state of a gas bubble (by external potential) travelling at
 * constant velocity in liquid helium. 
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
/* #define KEEP_IMAG        /* keep in imag time mode during normal time iter */
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
#define VX	(0.1 / GRID_AUTOMPS)
#define ACC     ((150.0 / GRID_AUTOMPS) / (1E-12 / GRID_AUTOS))

double global_time, rho0;

double velocity(double t) {

  return ACC * t + VX;
}

double momentum(double t) {

  return HELIUM_MASS * velocity(t) / HBAR;
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
  rgrid3d *ext_pot, *density, *current;
  long iter;
  char filename[2048];
  double kin, pot, prev_mom, cur_mom;
  double mu0, n, Fd1, Fd2, am;
  
  /* Setup DFT driver parameters (256 x 256 x 256 grid) */
  dft_driver_setup_grid(NX, NY, NZ, STEP /* Bohr */, THREADS /* threads */);
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
  
  /* Allocate wavefunctions and grids */
  gwf = dft_driver_alloc_wavefunction(HELIUM_MASS);  /* order parameter for current time (He liquid) */
  gwfp = dft_driver_alloc_wavefunction(HELIUM_MASS); /* order parameter for future (predict) (He liquid) */

  cworkspace = dft_driver_alloc_cgrid();             /* allocate complex workspace */
  ext_pot = dft_driver_alloc_rgrid();                /* allocate real external potential grid */
  density = dft_driver_alloc_rgrid();                /* allocate real density grid */
  current = dft_driver_alloc_rgrid();                /* allocate real density grid */
  
  fprintf(stderr, "Time step in a.u. = %le\n", TIME_STEP / GRID_AUTOFS);
  fprintf(stderr, "Relative velocity = (%le, %le, %le) (au)\n", VX, 0.0, 0.0);
  fprintf(stderr, "Relative velocity = (%le, %le, %le) (A/ps)\n", 
		  VX * 1000.0 * GRID_AUTOANG / GRID_AUTOFS, 0.0, 0.0);
  fprintf(stderr, "Relative velocity = (%le, %le, %le) (m/s)\n", VX * GRID_AUTOMPS, 0.0, 0.0);

#ifdef VISCOSITY
  fprintf(stderr, "Reynolds # = %le\n", rho0 * VX * 20E-10 / (VISCOSITY / GRID_AUTOPAS));
#endif
  
  rgrid3d_map(ext_pot, pot_func, NULL); /* External potential */
  rgrid3d_add(ext_pot, -mu0);

  for(iter = 0; iter < MAXITER; iter++) {
    double time_step;
    
    if(iter < STARTING_ITER) {
      dft_driver_setup_model(FUNCTIONAL, DFT_DRIVER_IMAG_TIME, rho0);  /* imag time */
      fprintf(stderr, "Imag time mode.\n");
      time_step = TIME_STEP;
    } else {
      double kx = momentum(time_step * (double) (iter - STARTING_ITER + 1));
      dft_driver_setup_momentum(kx, 0.0, 0.0);
      cgrid3d_set_momentum(gwf->grid, kx, 0.0, 0.0);
      cgrid3d_set_momentum(gwfp->grid, kx, 0.0, 0.0);
      cgrid3d_set_momentum(cworkspace, kx, 0.0, 0.0);
#ifndef KEEP_IMAG
      dft_driver_setup_boundary_type(DFT_DRIVER_BOUNDARY_ITIME, 0.2, ABS_WIDTH);
      fprintf(stderr, "Absorption begins at %le Bohr from the boundary\n",  ABS_WIDTH);
      dft_driver_setup_model(FUNCTIONAL, DFT_DRIVER_REAL_TIME, rho0);  /* real time */
      time_step = TIME_STEP / 5.0;
      fprintf(stderr, "Real time mode.\n");
#else
      fprintf(stderr, "Imag time mode.\n");      
#endif
      /* viscosity */
#ifdef VISCOSITY
      fprintf(stderr,"Viscosity using precomputed alpha. with T = %le\n", TEMP);
      dft_driver_setup_viscosity(VISCOSITY, 1.72 + 2.32E-10*exp(11.15*TEMP));
#endif
    }
    
    if(!(iter % OUTPUT)) prev_mom = dft_driver_Px(gwf);

    (void) dft_driver_propagate_predict(DFT_DRIVER_PROPAGATE_HELIUM, ext_pot, gwf, gwfp, cworkspace, time_step, iter); /* PREDICT */ 
    (void) dft_driver_propagate_correct(DFT_DRIVER_PROPAGATE_HELIUM, ext_pot, gwf, gwfp, cworkspace, time_step, iter); /* CORRECT */ 
    
    if(!(iter % OUTPUT)) cur_mom = dft_driver_Px(gwf);

    if(!(iter % OUTPUT)) {   /* every OUTPUT iterations, write output */
      double vx = velocity(time_step * (double) (iter - STARTING_ITER + 1));
      /* Helium energy */
      kin = dft_driver_kinetic_energy(gwf);            /* Kinetic energy for gwf */
      pot = dft_driver_potential_energy(gwf, ext_pot); /* Potential energy for gwf */
      //ene = kin + pot;           /* Total energy for gwf */
      n = dft_driver_natoms(gwf);
      printf("Iteration %ld current velocity = %le m/s\n", iter, vx * GRID_AUTOMPS);
      printf("Iteration %ld background kinetic = %.30lf\n", iter, n * (0.5 * HELIUM_MASS * vx * vx) * GRID_AUTOK);
      printf("Iteration %ld helium natoms    = %le particles.\n", iter, n);   /* Energy / particle in K */
      printf("Iteration %ld helium kinetic   = %.30lf\n", iter, kin * GRID_AUTOK);  /* Print result in K */
      printf("Iteration %ld helium potential = %.30lf\n", iter, pot * GRID_AUTOK);  /* Print result in K */
      printf("Iteration %ld helium energy    = %.30lf\n", iter, (kin + pot) * GRID_AUTOK);  /* Print result in K */
      fflush(stdout);

      /* Helium density */
      grid3d_wf_probability_flux_x(gwf, current);
      am = rgrid3d_integral(current) / vx;
      printf("Iteration %ld added mass = %.30lf\n", iter, am);
      sprintf(filename, "liquid-%ld", iter);              
      dft_driver_write_grid(gwf->grid, filename);
      grid3d_wf_density(gwf, density);                     /* Density from gwf */
      Fd1 = -rgrid3d_weighted_integral(density, dpot_func, NULL);
      Fd2 = (cur_mom - prev_mom) / (time_step / GRID_AUTOFS);
      printf("Drag force1 = %le (au).\n", Fd1);
      printf("Drag force2 = %le (au).\n", Fd2);
      printf("C_d1 * A = %le m^2\n", GRID_AUTOM * GRID_AUTOM * 2.0 * fabs(Fd1) / (DFT_HELIUM_MASS * rho0 * vx * vx));
      printf("C_d2 * A = %le m^2\n", GRID_AUTOM * GRID_AUTOM * 2.0 * fabs(Fd2) / (DFT_HELIUM_MASS * rho0 * vx * vx));
      printf("Reynolds1,1(test) = %le\n", ACC * HELIUM_MASS / Fd1);
      printf("Reynolds1,2(test) = %le\n", ACC * am * HELIUM_MASS / Fd1);
      printf("Reynolds2,1(test) = %le\n", ACC * HELIUM_MASS / Fd2);
      printf("Reynolds2,2(test) = %le\n", ACC * am * HELIUM_MASS / Fd2);
      fflush(stdout);
    }
  }
  return 0;
}
