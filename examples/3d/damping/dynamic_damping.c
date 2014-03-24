/*
 * Real time propagation of a gaussian excitation 
 * with wavefunction damping.
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
#define TIME_STEP 0.5 /* Time step in fs (5 for real, 10 for imag) */
#define STARTING_ITER 1
#define MAXITER (10000 + STARTING_ITER) /* Maximum number of iterations (was 300) */
#define OUTPUT     100 /* output every this iteration */
#define THREADS 4     /* # of parallel threads to use */
#define NX 64         /* # of grid points along x */
#define NY 64         /* # of grid points along y */
#define NZ 64         /* # of grid points along z */
#define STEP 1.0       /* spatial step length (Bohr) */
#define HELIUM_MASS (4.002602 / GRID_AUTOAMU) /* helium mass */

/*
 * Parameters for the absorption.
 * 
 * Start damping at 16. bohr (the damping increases
 * very gently so it may not be noticable until 20. or so).
 *
 * Damping strenght = 0.01 * time step. It seems reasonable
 * to make it proportional to the time step, but there's
 * no real justification for it.
 *
 */
#define ABS	16.
#define DAMP	( 0.01 * TIME_STEP )

double global_time;

int main(int argc, char **argv) {

  wf3d *gwf, *gwfp;
  cgrid3d *cworkspace;
  rgrid3d *ext_pot, *density;
  char filename[2048];
  long iter;
  double kin, pot;
  double mu0 , rho0 ;

  /* Setup DFT driver parameters (256 x 256 x 256 grid) */
  dft_driver_setup_grid(NX, NY, NZ, STEP /* Bohr */, THREADS /* threads */);

  /* Plain Orsay-Trento in real or imaginary time 
   * Set density to 0.0 so it will be set to the
   * correct amount once the functional is defined
   */
  dft_driver_setup_model(DFT_OT_PLAIN , 0, 0.0 );   /* DFT_OT_PLAIN = Orsay-Trento without kinetic corr. or backflow, 1 = imag time */

  /* Regular boundaries */
  dft_driver_setup_boundaries( DFT_DRIVER_BOUNDARY_DAMPING , ABS );   /* regular periodic boundaries */
  dft_driver_setup_boundaries_damp( DAMP );      /* damping coeff., only needed for absorbing boundaries */
  dft_driver_setup_boundary_condition(DFT_DRIVER_BC_NORMAL );
  
  /* Initialize */
  dft_driver_initialize();

  /* bulk normalization */
  dft_driver_setup_normalization(DFT_DRIVER_DONT_NORMALIZE, 4, 0.0, 0);   /* Normalization: ZEROB = adjust grid point NX/4, NY/4, NZ/4 to bulk density after each imag. time iteration */
  
  /* get bulk density and chemical potential */
  rho0 = bulk_density(dft_driver_otf) ; /* this is the value un driver_rho0 */
  mu0  = bulk_chempot(dft_driver_otf) ;

  /* Allocate wavefunctions and grids */
  gwf = dft_driver_alloc_wavefunction(HELIUM_MASS);     /* order parameter for current time */
  gwfp = dft_driver_alloc_wavefunction(HELIUM_MASS);    /* order parameter for future (predict) */
  cworkspace = dft_driver_alloc_cgrid();                /* allocate complex workspace */
  ext_pot = dft_driver_alloc_rgrid();                   /* allocate real external potential grid */
  density = dft_driver_alloc_rgrid();                   /* allocate real density grid */

  fprintf(stderr, "Time step in a.u. = %le\n", TIME_STEP / GRID_AUTOFS);

  /* Initial wf:  a constant + gaussian perturbation */
  cgrid3d_constant(gwf->grid, sqrt(rho0)); /* Set order parameter initially to sqrt of density (constant) */
  /* The perturbation: */
  double inv_width = 0.5 ;
  cgrid3d_map(gwfp->grid, dft_common_cgaussian, &inv_width) ;
  cgrid3d_multiply(gwfp->grid, 0.5) ;
  cgrid3d_sum(gwf->grid, gwf->grid, gwfp->grid) ;

  cgrid3d_copy(gwfp->grid, gwf->grid);                    /* make current and predicted wf's equal */
  
  /*
   * Remember to include -mu0 when having absorption so there is no
   * oscillating global phase messing with the boundary region with fixed phase.
   */
  rgrid3d_constant(ext_pot, -mu0) ;


  for(iter = STARTING_ITER ; iter < MAXITER; iter++) { /* start from 1 to avoid automatic wf initialization to a constant value */
    /* PREDICT */
    (void) dft_driver_propagate_predict(DFT_DRIVER_PROPAGATE_HELIUM, ext_pot, gwf, gwfp, cworkspace, TIME_STEP, iter);
    /* CORRECT */
    (void) dft_driver_propagate_correct(DFT_DRIVER_PROPAGATE_HELIUM, ext_pot, gwf, gwfp, cworkspace, TIME_STEP, iter);

    if(! (iter % OUTPUT)){
      kin = dft_driver_kinetic_energy(gwf);            /* Kinetic energy for gwf */
      pot = dft_driver_potential_energy(gwf, ext_pot); /* Potential energy for gwf */
	    //ene = kin + pot;           /* Total energy for gwf */
      printf("Iteration %ld npart  = %le particles.\n", iter, dft_driver_natoms(gwf) );   /* Energy / particle in K */
      printf("Iteration %ld kinetic   = %.30lf\n", iter, kin * GRID_AUTOK);  /* Print result in K */
      printf("Iteration %ld potential = %.30lf\n", iter, pot * GRID_AUTOK);  /* Print result in K */
      printf("Iteration %ld energy    = %.30lf\n", iter, (kin + pot) * GRID_AUTOK);  /* Print result in K */

      grid3d_wf_density(gwf, density);                     /* Density from gwf */
      sprintf(filename, "wf-%ld", iter);              
      dft_driver_write_grid(gwf->grid, filename);         /* Write density to file */
      sprintf(filename, "density-%ld", iter);              
      dft_driver_write_2d_density(density, filename);
    }
  }
      return 0;
}
