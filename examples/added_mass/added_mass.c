/*
 * Stationary state of an electron bubble travelling at
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
#define TIME_STEP 20.0	/* Time step in fs (5 for real, 10 for imag) */
#define IMP_STEP 0.2	/* Time step in fs (5 for real, 10 for imag) */
#define MAXITER 20000  /* Maximum number of iterations (was 300) */
#define OUTPUT     200	/* output every this iteration */
#define THREADS 0	/* # of parallel threads to use */
#define NX 256       	/* # of grid points along x */
#define NY 256         /* # of grid points along y */
#define NZ 256        	/* # of grid points along z */
#define STEP 1.5        /* spatial step length (Bohr) */
#define PRESSURE 0.0    /* External pressure */
#define IMP_MASS 1.0 /* electron mass */

/* velocity components */
#define KX	(1.0 * 2.0 * M_PI / (NX * STEP))
#define KY	(0.0 * 2.0 * M_PI / (NY * STEP))
#define KZ	(0.0 * 2.0 * M_PI / (NZ * STEP))
#define VX	(KX * HBAR / DFT_HELIUM_MASS)
#define VY	(KY * HBAR / DFT_HELIUM_MASS)
#define VZ	(KZ * HBAR / DFT_HELIUM_MASS)
#define EKIN	(0.5 * DFT_HELIUM_MASS * (VX * VX + VY * VY + VZ * VZ))

REAL global_time;

int main(int argc, char *argv[]) {

  dft_ot_functional *otf;
  wf *gwf, *gwfp;
  wf *impwf;
  cgrid *potential;
  rgrid *pair_pot, *ext_pot, *rworkspace;
  INT iter;
  char filename[2048];
  REAL kin, pot;
  REAL rho0, mu0, n;
  grid_timer timer;

#ifdef USE_CUDA
#define NGPUS 1
  int gpus[NGPUS] = {0};
  cuda_enable(1, NGPUS, gpus);
#endif

  /* Initialize threads & use wisdom */
  grid_set_fftw_flags(1);    // FFTW_MEASURE
  grid_threads_init(THREADS);
  grid_fft_read_wisdom(NULL);

  grid_wf_analyze_method(1); // DEBUG

  /* Allocate wave functions */
  if(!(gwf = grid_wf_alloc(NX, NY, NZ, STEP, DFT_HELIUM_MASS, WF_PERIODIC_BOUNDARY, WF_2ND_ORDER_FFT, "gwf"))) {
    fprintf(stderr, "Cannot allocate gwf.\n");
    exit(1);
  }
  gwfp = grid_wf_clone(gwf, "gwfp");
  impwf = grid_wf_clone(gwf, "impwf");
  impwf->mass = IMP_MASS;

  /* Reference frame for gwf & gwfp */
  cgrid_set_momentum(gwf->grid, KX, KY, KZ);
  cgrid_set_momentum(gwfp->grid, KX, KY, KZ);

  /* Allocate OT functional */
  if(!(otf = dft_ot_alloc(DFT_OT_PLAIN, gwf, DFT_MIN_SUBSTEPS, DFT_MAX_SUBSTEPS))) {
    fprintf(stderr, "Cannot allocate otf.\n");
    exit(1);
  }
  rho0 = dft_ot_bulk_density_pressurized(otf, PRESSURE);
  mu0 = dft_ot_bulk_chempot_pressurized(otf, PRESSURE);
  printf("mu0 = " FMT_R " K/atom, rho0 = " FMT_R " Angs^-3.\n", mu0 * GRID_AUTOK, rho0 / (GRID_AUTOANG * GRID_AUTOANG * GRID_AUTOANG));
  
  /* Allocate grids */
  potential = cgrid_clone(gwf->grid, "potential"); /* allocate complex workspace (must be preserved during predict-correct) */
  pair_pot = rgrid_clone(otf->density, "pair_pot");/* allocate real external potential grid (seprate grid; cannot be overwritten) */
  ext_pot = rgrid_clone(otf->density, "ext pot");  /* allocate real external potential grid (used by predict-correct; separate grid) */
  rworkspace = rgrid_clone(otf->density, "rworkspace");

  fprintf(stderr, "Time step in a.u. = " FMT_R "\n", TIME_STEP / GRID_AUTOFS);
  fprintf(stderr, "Relative velocity = (" FMT_R "," FMT_R "," FMT_R ") (A/ps)\n", 
		  VX * 1000.0 * GRID_AUTOANG / GRID_AUTOFS,
		  VY * 1000.0 * GRID_AUTOANG / GRID_AUTOFS,
		  VZ * 1000.0 * GRID_AUTOANG / GRID_AUTOFS);

  /* Initial wavefunctions. Read from file or set to initial guess */
#if 1
  /* Constant density (initial guess) */
  cgrid_constant(gwf->grid, SQRT(rho0));
  /* Gaussian for impurity (initial guess) */
  REAL inv_width = 0.5;
  cgrid_map(impwf->grid, dft_common_cgaussian, &inv_width);
#else
  /* Read liquid wavefunction from file */ 
  cgrid_read_grid(gwf->grid, "liquid_input");
  /* Read impurity wavefunction from file */ 
  cgrod_read_grid(impwf->grid, "impurity_input");
#endif

  /* Read pair potential from file and do FFT */
  dft_common_potential_map(4, "../electron/jortner.dat", "../electron/jortner.dat", "../electron/jortner.dat", pair_pot);
  rgrid_fft(pair_pot);

  for(iter = 0; iter < MAXITER; iter++) { /* start from 1 to avoid automatic wf initialization to a constant value */
    if(iter == 5) grid_fft_write_wisdom(NULL);
    grid_timer_start(&timer);

    /*** IMPURITY ***/
    /* 1. update potential */

    grid_wf_density(gwf, otf->density);
    rgrid_fft(otf->density);
    rgrid_fft_convolute(ext_pot, otf->density, pair_pot);
    rgrid_inverse_fft(ext_pot);
    grid_real_to_complex_re(potential, ext_pot);

    /* 2. Propagate */
    grid_wf_propagate(impwf, potential, -I * IMP_STEP / GRID_AUTOFS);
    grid_wf_normalize(impwf);

    /* 3. if OUTPUT, compute energy */
    if(!(iter % OUTPUT)) {
      /* Impurity energy */
      kin = grid_wf_energy(impwf, NULL);     /* kinetic */ 
      pot = grid_wf_potential_energy(impwf, ext_pot);  /* potential */
      printf("Output at iteration " FMT_I ":\n", iter);
      printf("Impurity kinetic   = " FMT_R " K\n", kin * GRID_AUTOK);  /* Print result in K */
      printf("Impurity potential = " FMT_R " K\n", pot * GRID_AUTOK);  /* Print result in K */
      printf("Impurity energy    = " FMT_R " K\n", (kin + pot) * GRID_AUTOK);  /* Print result in K */
      fflush(stdout);
      /* Impurity density */
      grid_wf_density(impwf, otf->density);
      sprintf(filename, "ebubble_imp-" FMT_I, iter);
      rgrid_write_grid(filename, otf->density);      /* Write wavefunction to file */
    }
    
    /***  HELIUM  ***/
    /* 1. update potential */
    grid_wf_density(impwf, otf->density);
    rgrid_fft(otf->density);
    rgrid_fft_convolute(ext_pot, otf->density, pair_pot);
    rgrid_inverse_fft(ext_pot);
    rgrid_fft_space(otf->density, 0); // Reset back to real space    

    /* 2. Predict + correct */
    grid_real_to_complex_re(potential, ext_pot);
    dft_ot_potential(otf, potential, gwf);
    cgrid_add(potential, -mu0);
    grid_wf_propagate_predict(gwf, gwfp, potential, -I * TIME_STEP / GRID_AUTOFS);
    grid_add_real_to_complex_re(potential, ext_pot);
    dft_ot_potential(otf, potential, gwfp);
    cgrid_add(potential, -mu0);
    cgrid_multiply(potential, 0.5);  // Use (current + future) / 2
    grid_wf_propagate_correct(gwf, potential, -I * TIME_STEP / GRID_AUTOFS);
    // Chemical potential included - no need to normalize

    printf("Iteration " FMT_I " - Wall clock time = " FMT_R " seconds.\n", iter, grid_timer_wall_clock_time(&timer));

    if(!(iter % OUTPUT)) {   /* every OUTPUT iterations, write output */
      /* Helium energy */
      kin = grid_wf_energy(gwf, NULL); /* Kinetic energy for gwf */
      dft_ot_energy_density(otf, rworkspace, gwf);
      rgrid_add_scaled_product(rworkspace, 1.0, otf->density, ext_pot);
      pot = rgrid_integral(rworkspace);
      //ene = kin + pot;           /* Total energy for gwf */
      n = grid_wf_norm(gwf);
      printf("Background kinetic = " FMT_R "\n", n * EKIN * GRID_AUTOK);
      printf("Helium natoms    = " FMT_R " particles.\n", n);
      printf("Helium kinetic   = " FMT_R " K\n", kin * GRID_AUTOK);
      printf("Helium potential = " FMT_R " K\n", pot * GRID_AUTOK);
      printf("Helium energy    = " FMT_R " K\n", (kin + pot) * GRID_AUTOK);
      fflush(stdout);

      /* Helium density */
      if(VX != 0.0)
	grid_wf_probability_flux_x(gwf, rworkspace);
      else if(VY != 0.0)
	grid_wf_probability_flux_y(gwf, rworkspace);
      else
	grid_wf_probability_flux_z(gwf, rworkspace);

      if(VX != 0.0)
	printf("Added mass = " FMT_R "\n", rgrid_integral(rworkspace) / VX); 
      else
	printf("VX = 0, no added mass.\n");
      fflush(stdout);

      grid_wf_density(gwf, otf->density);                     /* Density from gwf */

      sprintf(filename, "ebubble_liquid-" FMT_I, iter);              
      rgrid_write_grid(filename, otf->density);
    }
  }
  return 0;
}

