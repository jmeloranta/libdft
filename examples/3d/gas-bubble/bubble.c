/*
 * Stationary state of a gas bubble travelling at
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
#define TIME_STEP 10.0	/* Time step in fs (5 for real, 10 for imag) */
#define STARTING_ITER 1 /* Starting iteration - be careful if set to zero */
#define MAXITER (20000 + STARTING_ITER) /* Maximum number of iterations (was 300) */
#define OUTPUT     100	/* output every this iteration */
#define THREADS 0	/* # of parallel threads to use */
#define NX 512       	/* # of grid points along x */
#define NY 512          /* # of grid points along y */
#define NZ 512        	/* # of grid points along z */
#define STEP 1.0        /* spatial step length (Bohr) */
// #define DENSITY (0.0220 * 0.529 * 0.529 * 0.529)     /* bulk liquid density (0.0 = default at SVP); was 0.0218360 */
#define PRESSURE (1.0 / GRID_AUTOBAR)   /* External pressure in bar */
#define HELIUM_MASS (4.002602 / GRID_AUTOAMU) /* helium mass */

#define BUBBLE_NHE 5  /* Number of He (gas) atoms in the bubble */
#define BUBBLE_TEMP 100.0 /* Gas temperature inside the bubble (K) */
#define BUBBLE_SIZE 20.0 /* initial bubble size */

#define TEMP 1.6
#define VISCOSITY (1.306E-6 * 0.162)

/* velocity components */
#define KX	(10.0 * 2.0 * M_PI / (NX * STEP))
#define KY	(0.0 * 2.0 * M_PI / (NX * STEP))
#define KZ	(0.0 * 2.0 * M_PI / (NX * STEP))
#define VX	(KX * HBAR / HELIUM_MASS)
#define VY	(KY * HBAR / HELIUM_MASS)
#define VZ	(KZ * HBAR / HELIUM_MASS)
#define EKIN	(0.5 * HELIUM_MASS * (VX * VX + VY * VY + VZ * VZ))

double global_time, rho0;

double complex bubble_func(void *na, double x, double y, double z) {

  if(sqrt(x*x + y*y + z*z) < BUBBLE_SIZE) return 0.0;
  else return sqrt(rho0);
}

int main(int argc, char *argv[]) {

  wf3d *gwf, *gwfp;
  wf3d *impwf, *impwfp; /* impurity wavefunction */
  cgrid3d *cworkspace;
  rgrid3d *pair_pot, *ext_pot, *density, *current;
  long iter;
  char filename[2048];
  double kin, pot;
  double mu0, n;
  dft_common_lj lj_params;
  
  /* Setup DFT driver parameters (256 x 256 x 256 grid) */
  dft_driver_setup_grid(NX, NY, NZ, STEP /* Bohr */, THREADS /* threads */);
  /* Setup frame of reference momentum */
  dft_driver_setup_momentum(KX, KY, KZ);

  /* Plain Orsay-Trento in real or imaginary time */
  dft_driver_setup_model(DFT_OT_PLAIN, 1, 0.0218360 * GRID_AUTOANG * GRID_AUTOANG * GRID_AUTOANG);   /* DFT_OT_HD = Orsay-Trento with high-densiy corr. , 1 = imag time */

  /* viscosity */
#ifdef VISOSITY
  printf("Using precomputed alpha. with T = %le\n", TEMP);
  dft_driver_setup_viscosity(VISCOSITY, 1.72 + 2.32E-10*exp(11.15*TEMP));
#endif
  
  /* Regular boundaries */
  dft_driver_setup_boundaries(DFT_DRIVER_BOUNDARY_REGULAR, 0.0);   /* regular periodic boundaries */
  dft_driver_setup_boundaries_damp(0.00);                          /* damping coeff., only needed for absorbing boundaries */
  dft_driver_setup_boundary_condition(DFT_DRIVER_BC_NORMAL);
  
  /* Initialize */
  dft_driver_initialize();

  /* bulk normalization */
  dft_driver_setup_normalization(DFT_DRIVER_DONT_NORMALIZE, 4, 0.0, 0);   /* Normalization: ZEROB = adjust grid point NX/4, NY/4, NZ/4 to bulk density after each imag. time iteration */
  
  /* get bulk density and chemical potential */
  rho0 = dft_ot_bulk_density_pressurized(dft_driver_otf, PRESSURE);
  dft_driver_otf->rho0 = rho0;
  mu0  = dft_ot_bulk_chempot_pressurized(dft_driver_otf, PRESSURE);
  printf("rho0 = %le Angs^-3, mu0 = %le K.\n", rho0 / (0.529 * 0.529 * 0.529), mu0 * GRID_AUTOK);
  
  /* Allocate wavefunctions and grids */
  gwf = dft_driver_alloc_wavefunction(HELIUM_MASS);  /* order parameter for current time (He liquid) */
  gwfp = dft_driver_alloc_wavefunction(HELIUM_MASS); /* order parameter for future (predict) (He liquid) */
  impwf = dft_driver_alloc_wavefunction(HELIUM_MASS);  /* impurity - order parameter for current time (He gas) */
  impwf->norm  = BUBBLE_NHE;
  impwfp = dft_driver_alloc_wavefunction(HELIUM_MASS);  /* impurity - order parameter for future (predict) */
  impwfp->norm = BUBBLE_NHE;
  printf("Number of He gas atoms in the bubble = %d at %le K.\n", BUBBLE_NHE, BUBBLE_TEMP);
  cworkspace = dft_driver_alloc_cgrid();             /* allocate complex workspace */
  pair_pot = dft_driver_alloc_rgrid();               /* allocate real external potential grid */
  ext_pot = dft_driver_alloc_rgrid();                /* allocate real external potential grid */
  density = dft_driver_alloc_rgrid();                /* allocate real density grid */
  current = dft_driver_alloc_rgrid();                /* allocate real density grid */

  fprintf(stderr, "Time step in a.u. = %le\n", TIME_STEP / GRID_AUTOFS);
  fprintf(stderr, "Relative velocity = (%le, %le, %le) (A/ps)\n", 
		  VX * 1000.0 * GRID_AUTOANG / GRID_AUTOFS,
		  VY * 1000.0 * GRID_AUTOANG / GRID_AUTOFS,
		  VZ * 1000.0 * GRID_AUTOANG / GRID_AUTOFS);
  fprintf(stderr, "Relative velocity = (%le, %le, %le) (m/s)\n", VX * GRID_AUTOMPS, VY * GRID_AUTOMPS, VZ * GRID_AUTOMPS);

  /* Initial wavefunctions. Read from file or set to initial guess */
#if 1
  /* Constant density (initial guess) */
  //cgrid3d_constant(gwf->grid, sqrt(rho0));
  cgrid3d_map(gwf->grid, bubble_func, NULL);
  /* Gaussian for impurity (initial guess) */
  double inv_width = 1.0 / BUBBLE_SIZE;
  cgrid3d_map(impwf->grid, dft_common_cgaussian, &inv_width);
#else
  /* Read liquid wavefunction from file */ 
  dft_driver_read_grid(gwf->grid, "liquid_input");
  /* Read impurity wavefunction from file */ 
  dft_driver_read_grid(impwf->grid, "impurity_input");
#endif

  /* Set the electron velocity to zero */
  cgrid3d_set_momentum(impwf->grid, 0.0, 0.0, 0.0);

  cgrid3d_copy(gwfp->grid, gwf->grid);                    /* make current and predicted wf's equal */
  cgrid3d_copy(impwfp->grid, impwf->grid);                /* make current and predicted wf's equal */
  
  /* Map Lennard-Jones He - He potential */
#define MINDIST (1.5 / GRID_AUTOANG)
  lj_params.epsilon = 10.22 / GRID_AUTOK;
  lj_params.sigma = 2.556 / GRID_AUTOANG;
  lj_params.h = MINDIST;
  lj_params.cval = dft_common_lj_func(MINDIST * MINDIST, lj_params.sigma, lj_params.epsilon);
  rgrid3d_adaptive_map(pair_pot, dft_common_lennard_jones, &lj_params, 4, 32, 0.01 / GRID_AUTOK);
  dft_driver_convolution_prepare(pair_pot, NULL);

  for(iter = STARTING_ITER; iter < MAXITER; iter++) { /* start from 1 to avoid automatic wf initialization to a constant value */

    /*** IMPURITY ***/
    /* 1. update potential */
    /* gas He - liquid He LJ */
    grid3d_wf_density(gwf, density);
    dft_driver_convolution_prepare(NULL, density);
    dft_driver_convolution_eval(ext_pot, density, pair_pot);
    /* add ideal bose gas contribution */
    grid3d_wf_density(impwf, density);
    dft_common_idealgas_params(BUBBLE_TEMP, HELIUM_MASS, 1.0);
#if 0
    rgrid3d_operate_one(current, density, dft_common_idealgas_op);
#else
    rgrid3d_operate_one(current, density, dft_common_classical_idealgas_op);
#endif
    rgrid3d_sum(ext_pot, ext_pot, current);
    /* 2. Predict + correct */
    (void) dft_driver_propagate_predict(DFT_DRIVER_PROPAGATE_OTHER, ext_pot, impwf, impwfp, cworkspace, TIME_STEP, iter); /* PREDICT */ 
    (void) dft_driver_propagate_correct(DFT_DRIVER_PROPAGATE_OTHER, ext_pot, impwf, impwfp, cworkspace, TIME_STEP, iter); /* CORRECT */
    /*3. if OUTPUT, compute energy*/
    if(!(iter % OUTPUT)){	
      double P, V;
      /* Impurity energy */
      grid3d_wf_density(impwf, density);
      kin = grid3d_wf_energy(impwf, NULL, cworkspace);     /*kinetic*/ 
      pot = rgrid3d_integral_of_product(ext_pot, density); /*potential*/
      printf("Iteration %ld impurity kinetic   = %.30lf\n", iter, kin * GRID_AUTOK);  /* Print result in K */
      printf("Iteration %ld impurity potential = %.30lf\n", iter, pot * GRID_AUTOK);  /* Print result in K */
      printf("Iteration %ld impurity energy    = %.30lf\n", iter, (kin + pot) * GRID_AUTOK);  /* Print result in K */
      fflush(stdout);
      /* Gas density */
      sprintf(filename, "gas-%ld", iter);
      //dft_driver_write_2d_density(density, filename);  /* Write 2D density slices to file */
      dft_driver_write_density(density, filename);      /* Write wavefunction to file */
      rgrid3d_multiply(density, 1.0 / rgrid3d_value(density, 0.0, 0.0, 0.0));
      V = rgrid3d_integral(density);
      P = BUBBLE_NHE * GRID_AUKB * BUBBLE_TEMP / V;
      printf("Bubble V = %le Bohr^3, P = %le torr\n", V, P * GRID_AUTOPA * 7.5006E-3);
    }
    
    /***  HELIUM  ***/
    /* 1. update potential */
    grid3d_wf_density(impwf, density);
    dft_driver_convolution_prepare(NULL, density);
    dft_driver_convolution_eval(ext_pot, density, pair_pot);
    rgrid3d_add(ext_pot, -mu0); /* Add the chemical potential */
    /*2. Predict + correct */
    (void) dft_driver_propagate_predict(DFT_DRIVER_PROPAGATE_NORMAL, ext_pot, gwf, gwfp, cworkspace, TIME_STEP, iter); /* PREDICT */ 
    (void) dft_driver_propagate_correct(DFT_DRIVER_PROPAGATE_NORMAL, ext_pot, gwf, gwfp, cworkspace, TIME_STEP, iter); /* CORRECT */ 
    
    if(!(iter % OUTPUT)) {   /* every OUTPUT iterations, write output */
      /* Helium energy */
      kin = dft_driver_kinetic_energy(gwf);            /* Kinetic energy for gwf */
      pot = dft_driver_potential_energy(gwf, ext_pot); /* Potential energy for gwf */
      //ene = kin + pot;           /* Total energy for gwf */
      n = dft_driver_natoms(gwf);
      printf("Iteration %ld background kinetic = %.30lf\n", iter, n * EKIN * GRID_AUTOK);
      printf("Iteration %ld helium natoms    = %le particles.\n", iter, n);   /* Energy / particle in K */
      printf("Iteration %ld helium kinetic   = %.30lf\n", iter, kin * GRID_AUTOK);  /* Print result in K */
      printf("Iteration %ld helium potential = %.30lf\n", iter, pot * GRID_AUTOK);  /* Print result in K */
      printf("Iteration %ld helium energy    = %.30lf\n", iter, (kin + pot) * GRID_AUTOK);  /* Print result in K */
      fflush(stdout);

      /* Helium density */
      if(VX != 0.0)
	grid3d_wf_probability_flux_x(gwf, current);
      else if(VY != 0.0)
	grid3d_wf_probability_flux_y(gwf, current);
      else
	grid3d_wf_probability_flux_z(gwf, current);

      if(VX != 0.0)
	printf("Iteration %ld added mass = %.30lf\n", iter, rgrid3d_integral(current) / VX); 
      else
	printf("VX = 0, no added mass.\n");
      fflush(stdout);
      grid3d_wf_density(gwf, density);                     /* Density from gwf */
      sprintf(filename, "liquid-%ld", iter);              
      dft_driver_write_density(density, filename);
    }
  }
  return 0;
}
