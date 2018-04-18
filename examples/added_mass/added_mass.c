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
#define STARTING_ITER 1 /* Starting iteration - be careful if set to zero */
#define MAXITER (20000 + STARTING_ITER) /* Maximum number of iterations (was 300) */
#define OUTPUT     200	/* output every this iteration */
#define THREADS 0	/* # of parallel threads to use */
#define NX 128       	/* # of grid points along x */
#define NY 128          /* # of grid points along y */
#define NZ 128        	/* # of grid points along z */
#define STEP 1.5        /* spatial step length (Bohr) */
#define DENSITY (0.0218360 * 0.529 * 0.529 * 0.529)     /* bulk liquid density (0.0 = default at SVP) */
#define HELIUM_MASS (4.002602 / GRID_AUTOAMU) /* helium mass */
#define IMP_MASS 1.0 /* electron mass */

/* velocity components */
#define KX	(1.0 * 2.0 * M_PI / (NX * STEP))
#define KY	(0.0 * 2.0 * M_PI / (NX * STEP))
#define KZ	(0.0 * 2.0 * M_PI / (NX * STEP))
#define VX	(KX * HBAR / HELIUM_MASS)
#define VY	(KY * HBAR / HELIUM_MASS)
#define VZ	(KZ * HBAR / HELIUM_MASS)
#define EKIN	(0.5 * HELIUM_MASS * (VX * VX + VY * VY + VZ * VZ))

REAL global_time;

int main(int argc, char *argv[]) {

  wf *gwf, *gwfp;
  wf *impwf, *impwfp; /* impurity wavefunction */
  cgrid *cworkspace;
  rgrid *pair_pot, *ext_pot, *density, *current;
  INT iter;
  char filename[2048];
  REAL kin, pot;
  REAL rho0, mu0, n;

  /* Setup DFT driver parameters (256 x 256 x 256 grid) */
  dft_driver_setup_grid(NX, NY, NZ, STEP /* Bohr */, THREADS /* threads */);
#ifdef USE_CUDA
  cuda_enable(1);
#endif
  /* Setup frame of reference momentum */
  dft_driver_setup_momentum(KX, KY, KZ);

  /* Plain Orsay-Trento in real or imaginary time */
  dft_driver_setup_model(DFT_OT_PLAIN, 1, DENSITY);   /* DFT_OT_HD = Orsay-Trento with high-densiy corr. , 1 = imag time */

  /* Regular boundaries */
  dft_driver_setup_boundary_type(DFT_DRIVER_BOUNDARY_REGULAR, 0.0, 0.0, 0.0, 0.0);   /* regular periodic boundaries */
  dft_driver_setup_boundary_condition(DFT_DRIVER_BC_NORMAL);
  
  /* Initialize */
  dft_driver_initialize();
  dft_driver_kinetic = DFT_DRIVER_KINETIC_CN_NBC;
//  dft_driver_kinetic = DFT_DRIVER_KINETIC_FFT;

  /* bulk normalization */
  dft_driver_setup_normalization(DFT_DRIVER_DONT_NORMALIZE, 4, 0.0, 0);   /* Normalization: ZEROB = adjust grid point NX/4, NY/4, NZ/4 to bulk density after each imag. time iteration */
  
  /* get bulk density and chemical potential */
  rho0 = dft_ot_bulk_density(dft_driver_otf);
  mu0  = dft_ot_bulk_chempot(dft_driver_otf);
  printf("rho0 = " FMT_R " Angs^-3, mu0 = " FMT_R " K.\n", rho0 / (0.529 * 0.529 * 0.529), mu0 * GRID_AUTOK);

  /* Allocate wavefunctions and grids */
  gwf = dft_driver_alloc_wavefunction(HELIUM_MASS, "gwf");  /* order parameter for current time */
  gwfp = dft_driver_alloc_wavefunction(HELIUM_MASS, "gwfp"); /* order parameter for future (predict) */
  impwf = dft_driver_alloc_wavefunction(IMP_MASS, "impwf");   /* impurity - order parameter for current time */
  impwf->norm  = 1.0;
  impwfp = dft_driver_alloc_wavefunction(IMP_MASS, "impwfp");  /* impurity - order parameter for future (predict) */
  impwfp->norm = 1.0;
  cworkspace = dft_driver_alloc_cgrid("cworkspace");           /* allocate complex workspace (must be preserved during predict-correct) */
  pair_pot = dft_driver_alloc_rgrid("pair pot");               /* allocate real external potential grid (seprate grid; cannot be overwritten) */
  ext_pot = dft_driver_alloc_rgrid("ext pot");                 /* allocate real external potential grid (used by predict-correct; separate grid) */
  density = (rgrid *) dft_driver_get_workspace(10, 1);       /* used outside predict-correct */
  current = (rgrid *) dft_driver_get_workspace(1, 1);        /* used outside predict-correct */

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
  dft_driver_read_grid(gwf->grid, "liquid_input");
  /* Read impurity wavefunction from file */ 
  dft_driver_read_grid(impwf->grid, "impurity_input");
#endif

  /* Set the electron velocity to zero */
  cgrid_set_momentum(impwf->grid, 0.0, 0.0, 0.0);

  cgrid_copy(gwfp->grid, gwf->grid);                    /* make current and predicted wf's equal */
  cgrid_copy(impwfp->grid, impwf->grid);                /* make current and predicted wf's equal */
  
  /* Read pair potential from file and do FFT */
  dft_common_potential_map(DFT_DRIVER_AVERAGE_XYZ, "../electron/jortner.dat", "../electron/jortner.dat", "../electron/jortner.dat", pair_pot);
  dft_driver_convolution_prepare(pair_pot, NULL);

  for(iter = STARTING_ITER; iter < MAXITER; iter++) { /* start from 1 to avoid automatic wf initialization to a constant value */
    printf("Iteration = " FMT_I "\n", iter);
    /*** IMPURITY ***/
    /* 1. update potential */
    grid_wf_density(gwf, density);
    dft_driver_convolution_prepare(NULL, density);
    dft_driver_convolution_eval(ext_pot, density, pair_pot);
    /* no chemical potential for impurity */
    /* 2. Predict + correct */
    (void) dft_driver_propagate_predict(DFT_DRIVER_PROPAGATE_OTHER, ext_pot, impwf, impwfp, cworkspace, IMP_STEP, iter); /* PREDICT */ 
    (void) dft_driver_propagate_correct(DFT_DRIVER_PROPAGATE_OTHER, ext_pot, impwf, impwfp, cworkspace, IMP_STEP, iter); /* CORRECT */
    /* 3. if OUTPUT, compute energy*/
    if(!(iter % OUTPUT)){	
      /* Impurity energy */
      grid_wf_density(impwf, density);
      kin = grid_wf_energy(impwf, NULL, cworkspace) ;     /*kinetic*/ 
      pot = rgrid_integral_of_product(ext_pot, density) ; /*potential*/
      printf("Output at iteration " FMT_I ":\n", iter);
      printf("Impurity kinetic   = " FMT_R "\n", kin * GRID_AUTOK);  /* Print result in K */
      printf("Impurity potential = " FMT_R "\n", pot * GRID_AUTOK);  /* Print result in K */
      printf("Impurity energy    = " FMT_R "\n", (kin + pot) * GRID_AUTOK);  /* Print result in K */
      fflush(stdout);
      /* Impurity density */
      sprintf(filename, "ebubble_imp-" FMT_I, iter);
      //dft_driver_write_2d_density(density, filename);  /* Write 2D density slices to file */
      dft_driver_write_density(density, filename);      /* Write wavefunction to file */
    }
    
    /***  HELIUM  ***/
    /* 1. update potential */
    grid_wf_density(impwf, density);
    dft_driver_convolution_prepare(NULL, density);
    dft_driver_convolution_eval(ext_pot, density, pair_pot);
    rgrid_add(ext_pot, -mu0) ; /* Add the chemical potential */

    /* 2. Predict + correct */
    (void) dft_driver_propagate_predict(DFT_DRIVER_PROPAGATE_HELIUM, ext_pot, gwf, gwfp, cworkspace, TIME_STEP, iter); /* PREDICT */ 
    (void) dft_driver_propagate_correct(DFT_DRIVER_PROPAGATE_HELIUM, ext_pot, gwf, gwfp, cworkspace, TIME_STEP, iter); /* CORRECT */
    if(!(iter % OUTPUT)) {   /* every OUTPUT iterations, write output */
      /* Helium energy */
      kin = dft_driver_kinetic_energy(gwf);            /* Kinetic energy for gwf */
      pot = dft_driver_potential_energy(gwf, ext_pot); /* Potential energy for gwf */
      //ene = kin + pot;           /* Total energy for gwf */
      n = dft_driver_natoms(gwf) ;
      printf("Background kinetic = " FMT_R "\n", n * EKIN * GRID_AUTOK);
      printf("Helium natoms    = " FMT_R " particles.\n", n);
      printf("Helium kinetic   = " FMT_R " K\n", kin * GRID_AUTOK);
      printf("Helium potential = " FMT_R " K\n", pot * GRID_AUTOK);
      printf("Helium energy    = " FMT_R " K\n", (kin + pot) * GRID_AUTOK);
      fflush(stdout);

      /* Helium density */
      if(VX != 0.0)
	grid_wf_probability_flux_x(gwf, current);
      else if(VY != 0.0)
	grid_wf_probability_flux_y(gwf, current);
      else
	grid_wf_probability_flux_z(gwf, current);

      if(VX != 0.0)
	printf("Added mass = " FMT_R "\n", rgrid_integral(current) / VX); 
      else
	printf("VX = 0, no added mass.\n");
      fflush(stdout);

      grid_wf_density(gwf, density);                     /* Density from gwf */

      sprintf(filename, "ebubble_liquid-" FMT_I, iter);              
      dft_driver_write_density(density, filename);
    }
  }
  return 0;
}

