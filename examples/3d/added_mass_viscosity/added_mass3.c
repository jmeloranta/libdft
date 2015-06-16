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
#define TIME_STEP 100.0	/* Time step in fs (50-100) */
#define IMP_STEP 0.1	/* Time step in fs (0.01) */
#define MAXITER 500000 /* Maximum number of iterations (was 300) */
#define OUTPUT     500	/* output every this iteration */
#define THREADS 32	/* # of parallel threads to use */
#define NX 256       	/* # of grid points along x */
#define NY 256          /* # of grid points along y */
#define NZ 256      	/* # of grid points along z */
#define STEP 2.0        /* spatial step length (Bohr) */

#define HELIUM_MASS (4.002602 / GRID_AUTOAMU) /* helium mass */
#define IMP_MASS 1.0 /* electron mass */

/* velocity components (v_lix < 0) */
#define KX	(6.0 * 2.0 * M_PI / (NX * STEP))
#define KY	(0.0 * 2.0 * M_PI / (NX * STEP))
#define KZ	(0.0 * 2.0 * M_PI / (NX * STEP))
#define VX	(KX * HBAR / HELIUM_MASS)
#define VY	(KY * HBAR / HELIUM_MASS)
#define VZ	(KZ * HBAR / HELIUM_MASS)
#define EKIN	(0.5 * HELIUM_MASS * (VX * VX + VY * VY + VZ * VZ))

#define T2100MK

#ifdef T2100MK
/* Exp mobility = 0.0492 cm^2/Vs - gives 0.048 */
#define DENSITY (0.021954 * 0.529 * 0.529 * 0.529)     /* bulk liquid density */
#define VISCOSITY (1.71877E-6) /* In Pa s Donnelly: 1.73E-6 to 1.92E-6, recommended 1.803E-6 (2.0x) */
#define RHON    (0.752)      /* normal fraction Donnelly: 0.741 */
#define FUNCTIONAL DFT_OT_T2100MK
#endif

#ifdef T1800MK
/* Exp mobility = 0.097 cm^2/Vs - gives 1.08 */
#define DENSITY (0.021885 * 0.529 * 0.529 * 0.529)     /* bulk liquid density */
#define VISCOSITY (1.25E-6) /* In Pa s (x2.6) */
#define RHON    (0.35)       /* normal fraction */
#define FUNCTIONAL DFT_OT_T1800MK
#endif
  
#ifdef T1600MK
/* Exp mobility = 0.183 cm^2/Vs - gives 0.185 */
#define DENSITY (0.021845 * 0.529 * 0.529 * 0.529)     /* bulk liquid density */
#define VISCOSITY (1.30977E-6) /* In Pa s (was 2.5) */
#define RHON    0.171       /* normal fraction */
#define FUNCTIONAL DFT_OT_T1600MK
#endif

#ifdef T1200MK
/* Exp mobility = 1.0 cm^2/Vs - gives  */
#define DENSITY (0.021846 * 0.529 * 0.529 * 0.529)     /* bulk liquid density */
#define VISCOSITY (1.809E-6) /* In Pa s (1.5x) */
#define RHON    0.0289       /* normal fraction */
#define FUNCTIONAL DFT_OT_T1200MK
#endif

#ifdef T800MK
/* Exp mobility = 20.86 cm^2/Vs - gives  */
#define DENSITY (0.021876 * 0.529 * 0.529 * 0.529)     /* bulk liquid density */
#define VISCOSITY (15.823E-6) /* In Pa s (0.7) */
#define RHON    0.001       /* normal fraction (Donnelly 0.001, nist 0.0025) */
#define FUNCTIONAL DFT_OT_T800MK
#endif

#ifdef T400MK
/* Exp mobility = 438 cm^2/Vs - gives  */
#define DENSITY (0.021845 * 0.529 * 0.529 * 0.529)     /* bulk liquid density */
#define VISCOSITY (114.15E-6) /* In Pa s */
#define RHON    1.3E-6       /* normal fraction (2.89188E-6) */
#define FUNCTIONAL DFT_OT_T400MK
#endif

#ifdef T0MK
/* Exp mobility = infinity cm^2/Vs - gives 10^8 */
#define DENSITY (0.0218360 * 0.529 * 0.529 * 0.529)     /* bulk liquid density */
#define VISCOSITY (0.0) /* In Pa s */
#define RHON    0.0       /* normal fraction */
#define FUNCTIONAL DFT_OT_PLAIN
#endif

#define SBC     4.0         /* boundary condition: 4 = electron, 6 = + ion (for Stokes) */

double global_time;

double complex center_func(void *NA, double complex val, double x, double y, double z) {

  return x;   /* (x - x_0) but x_0 = 0 */
}

double center_func2(void *NA, double x, double y, double z) {

  return x;   /* (x - x_0) but x_0 = 0 */
}

double stddev_x(void *NA, double x, double y, double z) {

  return x * x;
}

double stddev_y(void *NA, double x, double y, double z) {

  return y * y;
}

double eval_force(wf3d *gwf, wf3d *impwf, rgrid3d *pair_pot, rgrid3d *dpair_pot, rgrid3d *workspace1, rgrid3d *workspace2) {

  double tmp;

  grid3d_wf_density(gwf, workspace1);
  dft_driver_convolution_prepare(workspace1, NULL);
  dft_driver_convolution_eval(workspace2, dpair_pot, workspace1);
  grid3d_wf_density(impwf, workspace1);
  rgrid3d_product(workspace1, workspace1, workspace2);
  tmp = rgrid3d_integral(workspace1);   /* double minus: from force calc and the change of reference frame (x' = -x) */
  return tmp;
}

int main(int argc, char *argv[]) {

  wf3d *gwf, *gwfp, *nwf, *total;
  wf3d *impwf, *impwfp; /* impurity wavefunction */
  cgrid3d *cworkspace;
  rgrid3d *pair_pot, *dpair_pot, *ext_pot, *density;
  rgrid3d *vx, *vy, *vz;
  long iter;
  char filename[2048];
  double kin, pot;
  double rho0, mu0, n;
  double force, mobility;

  /* Setup DFT driver parameters (256 x 256 x 256 grid) */
  dft_driver_setup_grid(NX, NY, NZ, STEP /* Bohr */, THREADS /* threads */);
  /* Setup frame of reference momentum */
  dft_driver_setup_momentum(KX, KY, KZ);

  /* Plain Orsay-Trento in real or imaginary time */
  dft_driver_setup_model(FUNCTIONAL, 1, DENSITY * (1.0 - RHON));   /* DFT_OT_HD = Orsay-Trento with high-densiy corr. , 1 = imag time */
  dft_driver_setup_normal_density(DENSITY * RHON);
  dft_driver_setup_viscosity(VISCOSITY);
  
  /* Regular boundaries */
  dft_driver_setup_boundaries(DFT_DRIVER_BOUNDARY_REGULAR, 0.0);   /* regular periodic boundaries */
  dft_driver_setup_boundaries_damp(0.00);                          /* damping coeff., only needed for absorbing boundaries */
  dft_driver_setup_boundary_condition(DFT_DRIVER_BC_NORMAL);
  
  /* Initialize */
  dft_driver_initialize();

  /* bulk normalization */
  dft_driver_setup_normalization(DFT_DRIVER_NORMALIZE_BULK, 4, 0.0, 0);   /* Normalization: ZEROB = adjust grid point NX/4, NY/4, NZ/4 to bulk density after each imag. time iteration */
  
  /* get bulk density and chemical potential */
  rho0 = dft_ot_bulk_density(dft_driver_otf);
  mu0  = dft_ot_bulk_chempot(dft_driver_otf);
  printf("rho0 = %le Angs^-3, mu0 = %le K.\n", rho0 / (0.529 * 0.529 * 0.529), mu0 * GRID_AUTOK);

  /* Allocate wavefunctions and grids */
  cworkspace = dft_driver_alloc_cgrid();             /* allocate complex workspace */
  pair_pot = dft_driver_alloc_rgrid();               /* allocate real external potential grid */
  dpair_pot = dft_driver_alloc_rgrid();               /* allocate real external potential grid */
  ext_pot = dft_driver_alloc_rgrid();                /* allocate real external potential grid */
  density = dft_driver_alloc_rgrid();                /* allocate real density grid */
  vx = dft_driver_alloc_rgrid();                /* allocate real density grid */
  vy = dft_driver_alloc_rgrid();                /* allocate real density grid */
  vz = dft_driver_alloc_rgrid();                /* allocate real density grid */
  impwf = dft_driver_alloc_wavefunction(IMP_MASS);   /* impurity - order parameter for current time */
  impwf->norm  = 1.0;
  impwfp = dft_driver_alloc_wavefunction(IMP_MASS);  /* impurity - order parameter for future (predict) */
  impwfp->norm = 1.0;
  cgrid3d_set_momentum(impwf->grid, 0.0, 0.0, 0.0); /* Electron at rest */
  cgrid3d_set_momentum(impwfp->grid, 0.0, 0.0, 0.0);
  gwf = dft_driver_alloc_wavefunction(HELIUM_MASS);  /* order parameter for current time */
  gwfp = dft_driver_alloc_wavefunction(HELIUM_MASS); /* order parameter for future (predict) */
  nwf = dft_driver_alloc_wavefunction(HELIUM_MASS); /* order parameter for normal fluid */
  total = dft_driver_alloc_wavefunction(HELIUM_MASS); /* Total wavefunction (normal & super) */
  
  fprintf(stderr, "Time step in a.u. = %le\n", TIME_STEP / GRID_AUTOFS);
  fprintf(stderr, "Relative velocity = ( %le , %le ,%le ) (A/ps)\n", 
		  VX * 1000.0 * GRID_AUTOANG / GRID_AUTOFS,
		  VY * 1000.0 * GRID_AUTOANG / GRID_AUTOFS,
		  VZ * 1000.0 * GRID_AUTOANG / GRID_AUTOFS);

  /* Initial wavefunctions. Read from file or set to initial guess */
  /* Constant density (initial guess) */
  cgrid3d_constant(gwf->grid, sqrt(rho0) * (1.0 - RHON));
  cgrid3d_constant(nwf->grid, sqrt(rho0) * RHON);
  /* Gaussian for impurity (initial guess) */
  double inv_width = 0.05;
  cgrid3d_map(impwf->grid, dft_common_cgaussian, &inv_width);
  cgrid3d_multiply(impwf->grid, 1.0 / sqrt(grid3d_wf_norm(impwf)));

  cgrid3d_copy(gwfp->grid, gwf->grid);                    /* make current and predicted wf's equal */
  cgrid3d_copy(impwfp->grid, impwf->grid);                /* make current and predicted wf's equal */
  
  /* Read pair potential from file and do FFT */
  dft_common_potential_map(DFT_DRIVER_AVERAGE_XYZ, "/home2/git/libdft/examples/3d/electron/jortner.dat", "/home2/git/libdft/examples/3d/electron/jortner.dat", "/home2/git/libdft/examples/3d/electron/jortner.dat", pair_pot);
  rgrid3d_fd_gradient_x(pair_pot, dpair_pot);
  dft_driver_convolution_prepare(pair_pot, dpair_pot);
  
  for(iter = 0; iter < MAXITER; iter++) { /* start from 1 to avoid automatic wf initialization to a constant value */
    /*** IMPURITY ***/
    /* 1. update potential */
    grid3d_wf_density(gwf, density);
    grid3d_wf_density(nwf, vx);
    rgrid3d_sum(density, density, vx); /* total density = normal + super */
    dft_driver_convolution_prepare(NULL, density);      
    dft_driver_convolution_eval(ext_pot, density, pair_pot);
    /*2. Predict + correct */
    (void) dft_driver_propagate_predict(DFT_DRIVER_PROPAGATE_OTHER, ext_pot, impwf, impwfp, cworkspace, IMP_STEP, iter); /* PREDICT */ 
    (void) dft_driver_propagate_correct(DFT_DRIVER_PROPAGATE_OTHER, ext_pot, impwf, impwfp, cworkspace, IMP_STEP, iter); /* CORRECT */
    force = creal(cgrid3d_grid_expectation_value_func(NULL, center_func, impwf->grid));
    printf("Expectation value of position (electron): %le\n", force * GRID_AUTOANG);
    /* keep electron at origin */
    cgrid3d_shift(cworkspace, impwf->grid, -force, 0.0, 0.0);
    cgrid3d_copy(impwf->grid, cworkspace);
    
    /*3. if OUTPUT, compute energy*/
    if(!(iter % OUTPUT)){	
      /* Impurity energy */
      grid3d_wf_density(impwf, density);
      kin = grid3d_wf_energy(impwf, NULL, cworkspace) ;     /*kinetic*/ 
      pot = rgrid3d_integral_of_product(ext_pot, density) ; /*potential*/
      printf("Iteration %ld impurity kinetic   = %.30lf\n", iter, kin * GRID_AUTOK);  /* Print result in K */
      printf("Iteration %ld impurity potential = %.30lf\n", iter, pot * GRID_AUTOK);  /* Print result in K */
      printf("Iteration %ld impurity energy    = %.30lf\n", iter, (kin + pot) * GRID_AUTOK);  /* Print result in K */
      /* Impurity density */
      sprintf(filename, "ebubble_imp-%ld", iter);
      //dft_driver_write_2d_density(density, filename);  /* Write 2D density slices to file */
      dft_driver_write_density(density, filename);      /* Write wavefunction to file */
    }
    
    /***  HELIUM  ***/
    /* 1. update potential */
    grid3d_wf_density(impwf, density);
    dft_driver_convolution_prepare(NULL, density);
    dft_driver_convolution_eval(ext_pot, density, pair_pot);
    //    rgrid3d_add(ext_pot, -mu0); /* Add the chemical potential */
    
    /*2a. Superfluid (missing predict-correct for gwf/nwf) */
    dft_driver_total_wf(total, gwf, nwf);
    (void) dft_driver_propagate_predict2(DFT_DRIVER_PROPAGATE_HELIUM, ext_pot, gwf, gwfp, cworkspace, total, TIME_STEP, iter);
    (void) dft_driver_propagate_correct2(DFT_DRIVER_PROPAGATE_HELIUM, ext_pot, gwf, gwfp, cworkspace, total, TIME_STEP, iter);
    
    /* 2b. Normal fluid (missing predict-correct for gwf/nwf) */
    (void) dft_driver_propagate_predict2(DFT_DRIVER_PROPAGATE_NORMAL, ext_pot, nwf, gwfp, cworkspace, total, TIME_STEP, iter);
    (void) dft_driver_propagate_correct2(DFT_DRIVER_PROPAGATE_NORMAL, ext_pot, nwf, gwfp, cworkspace, total, TIME_STEP, iter);
    
    if(!(iter % OUTPUT)) {   /* every OUTPUT iterations, write output */
      /* Helium energy */
      dft_driver_total_wf(total, gwf, nwf);
      kin = dft_driver_kinetic_energy(gwf);            /* Kinetic energy for gwf */
      kin += dft_driver_kinetic_energy(nwf);
      pot = dft_driver_potential_energy(total, ext_pot); /* Potential energy for gwf */
      //ene = kin + pot;           /* Total energy for gwf */
      n = dft_driver_natoms(total);
      printf("Iteration %ld background kinetic = %.30lf\n", iter, n * EKIN * GRID_AUTOK);
      printf("Iteration %ld helium natoms    = %le particles.\n", iter, n);   /* Energy / particle in K */
      printf("Iteration %ld helium kinetic   = %.30lf\n", iter, kin * GRID_AUTOK);  /* Print result in K */
      printf("Iteration %ld helium potential = %.30lf\n", iter, pot * GRID_AUTOK);  /* Print result in K */
      printf("Iteration %ld helium energy    = %.30lf\n", iter, (kin + pot) * GRID_AUTOK);  /* Print result in K */
      
      grid3d_wf_probability_flux_x(total, vx);
      printf("Iteration %ld added mass = %.30lf\n", iter, rgrid3d_integral(vx) / VX);

      force = eval_force(total, impwf, pair_pot, dpair_pot, ext_pot, density);  /* ext_pot & density are temps */
      printf("Drag force on ion = %le a.u.\n", force);
      printf("E-field = %le V/m\n", -force * GRID_AUTOVPM);
      printf("Target ion velocity = %le m/s\n", VX * GRID_AUTOMPS);
      mobility = -VX * GRID_AUTOMPS / (-force * GRID_AUTOVPM);
      printf("Mobility = %le [cm^2/(Vs)]\n", 1.0E4 * mobility); /* 1E4 = m^2 to cm^2 */
      printf("Hydrodynamic radius (Stokes) = %le Angs.\n", 1E10 * 1.602176565E-19 / (SBC * M_PI * mobility * RHON * VISCOSITY));
      grid3d_wf_density(total, density);                     /* Density from gwf */
      printf("Bubble asymmetry = %le\n", rgrid3d_weighted_integral(density, stddev_x, NULL) / rgrid3d_weighted_integral(density, stddev_y, NULL));

      /* write out normal fluid density */
      sprintf(filename, "ebubble_nliquid-%ld", iter);              
      grid3d_wf_density(nwf, density);
      dft_driver_write_density(density, filename);
      /* write out superfluid density */
      sprintf(filename, "ebubble_sliquid-%ld", iter);              
      grid3d_wf_density(gwf, density);
      dft_driver_write_density(density, filename);
      /* write out normal fluid flux field */
      grid3d_wf_probability_flux(nwf, vx, vy, vz);
      grid3d_wf_density(nwf, density);
      rgrid3d_multiply(density, VX); /* moving frame background */
      rgrid3d_difference(vx, density, vx);
      sprintf(filename, "ebubble_nliquid-vx-%ld", iter);              
      dft_driver_write_density(vx, filename);
      sprintf(filename, "ebubble_nliquid-vy-%ld", iter);              
      dft_driver_write_density(vy, filename);
      sprintf(filename, "ebubble_nliquid-vz-%ld", iter);              
      dft_driver_write_density(vz, filename);
      /* write out superfluid flux field */
      grid3d_wf_probability_flux(gwf, vx, vy, vz);
      grid3d_wf_density(gwf, density);
      rgrid3d_multiply(density, VX); /* moving frame background */
      rgrid3d_difference(vx, density, vx);
      sprintf(filename, "ebubble_sliquid-vx-%ld", iter);              
      dft_driver_write_density(vx, filename);
      sprintf(filename, "ebubble_sliquid-vy-%ld", iter);              
      dft_driver_write_density(vy, filename);
      sprintf(filename, "ebubble_sliquid-vz-%ld", iter);              
      dft_driver_write_density(vz, filename);
    }
  }
  return 0;
}
