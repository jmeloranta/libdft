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
#define THREADS 0	/* # of parallel threads to use (0 = all) */
#define PLANNING 1     /* 0 = estimate, 1 = measure, 2 = patient, 3 = exhaustive */
#define NX 512       	/* # of grid points along x */
#define NY 512         /* # of grid points along y */
#define NZ 512      	/* # of grid points along z */
#define STEP 1.0        /* spatial step length (Bohr) */

#define HELIUM_MASS (4.002602 / GRID_AUTOAMU) /* helium mass */
#define IMP_MASS 1.0 /* electron mass */

#define PSPOT "/home2/git/libdft/examples/3d/electron/jortner.dat"

/* velocity components */
#define KX	(1.0 * 2.0 * M_PI / (NX * STEP))
#define KY	(0.0 * 2.0 * M_PI / (NY * STEP))
#define KZ	(0.0 * 2.0 * M_PI / (NZ * STEP))
#define VX	(KX * HBAR / HELIUM_MASS)
#define VY	(KY * HBAR / HELIUM_MASS)
#define VZ	(KZ * HBAR / HELIUM_MASS)
#define EKIN	(0.5 * HELIUM_MASS * (VX * VX + VY * VY + VZ * VZ))

#define T1200MK
#define EPSILON 5E-5

#if 0
/* debug */
#define DENSITY (0.021983 * 0.529 * 0.529 * 0.529)     /* bulk liquid density */
#define VISCOSITY (2.4E-6)
#define RHON 1.0
//#define FUNCTIONAL DFT_OT_PLAIN
#define FUNCTIONAL (DFT_OT_T3000MK | DFT_OT_KC | DFT_OT_BACKFLOW)
#define IDENT "debug"
#endif

#ifdef T2100MK
/* Exp mobility = 0.0492 cm^2/Vs (Donnelly 0.05052) */
#define IDENT "T = 2.1 K"
#define DENSITY (0.021954 * 0.529 * 0.529 * 0.529)     /* bulk liquid density */
#define VISCOSITY (1.803E-6) /* In Pa s (Fred 1.719E-6) */
#define RHON    (0.741)      /* normal fraction (Fred 0.752)) */
#define FUNCTIONAL DFT_OT_T2100MK
#endif

#ifdef T2000MK
/* Exp mobility = 0.0685 cm^2/Vs (Donnelly) */
#define IDENT "T = 2.0 K"
#define DENSITY (0.021909 * 0.529 * 0.529 * 0.529)     /* bulk liquid density */
#define VISCOSITY (1.47E-6) /* In Pa s (Donnelly) */
#define RHON    (0.553)      /* normal fraction (Donnelly) */
#define FUNCTIONAL DFT_OT_T2000MK
#endif

#ifdef T1800MK
/* Exp mobility = 0.097 cm^2/Vs (Donnelly 0.1088) */
#define IDENT "T = 1.8 K"
#define DENSITY (0.021885 * 0.529 * 0.529 * 0.529)     /* bulk liquid density */
#define VISCOSITY (1.298E-6) /* In Pa s (Fred 1.25E-6, donnelly 1.298E-6) */
#define RHON    (0.313)       /* normal fraction (Fred 0.35)) */
#define FUNCTIONAL DFT_OT_T1800MK
#endif
  
#ifdef T1600MK
/* Exp mobility = 0.183 cm^2/Vs (Donnelly 0.1772) */
#define IDENT "T = 1.6 K"
#define DENSITY (0.021845 * 0.529 * 0.529 * 0.529)     /* bulk liquid density */
#define VISCOSITY (1.306E-6) /* In Pa s (Fred 1.310E-6)) */
#define RHON    0.162       /* normal fraction (Fred 0.171) */
#define FUNCTIONAL DFT_OT_T1600MK
#endif

#ifdef T1400MK
/* Exp mobility = 0.36 cm^2/Vs (Donnelly) */
#define IDENT "T = 1.4 K"
#define DENSITY (0.021837 * 0.529 * 0.529 * 0.529)     /* bulk liquid density */
#define VISCOSITY (1.40E-6) /* In Pa s (Donnelly) */
#define RHON    0.0745       /* normal fraction (Donnelly) */
#define FUNCTIONAL DFT_OT_T1400MK
#endif

#ifdef T1200MK
/* Exp mobility = 1.0 cm^2/Vs (Donnelly 0.9880) */
#define IDENT "T = 1.2 K"
#define DENSITY (0.021846 * 0.529 * 0.529 * 0.529)     /* bulk liquid density */
#define VISCOSITY (1.736E-6) /* In Pa s (Fred 1.809E-6) */
#define RHON    0.026       /* normal fraction (Fred 0.0289) */
#define FUNCTIONAL DFT_OT_T1200MK
#endif

#ifdef T800MK
/* Exp mobility = 20.86 cm^2/Vs (Donnelly 21.38) */
#define IDENT "T = 0.8 K"
#define DENSITY (0.021876 * 0.529 * 0.529 * 0.529)     /* bulk liquid density */
#define VISCOSITY (15.82E-6) /* In Pa s (Donnelly was 15.82E-6) */
#define RHON    9.27E-4       /* normal fraction */
#define FUNCTIONAL DFT_OT_T800MK
#endif

#ifdef T400MK
/* Exp mobility = 438 cm^2/Vs (Donnelly 454.6) */
#define IDENT "T = 0.4 K"
#define DENSITY (0.021845 * 0.529 * 0.529 * 0.529)     /* bulk liquid density */
#define VISCOSITY (114.15E-6) /* In Pa s (Fred ...) */
#define RHON    2.92E-6       /* normal fraction (Fred ...) */
#define FUNCTIONAL DFT_OT_T400MK
#endif

#ifdef T0MK
/* Exp mobility = infinity cm^2/Vs */
#define IDENT "T = 0 K"
#define DENSITY (0.0218360 * 0.529 * 0.529 * 0.529)     /* bulk liquid density */
#define VISCOSITY (0.0) /* In Pa s */
#define RHON    0.0       /* normal fraction */
#define FUNCTIONAL DFT_OT_PLAIN
#endif

#define SBC     4.0         /* boundary condition: 4 = electron, 6 = + ion (for Stokes) */

double global_time;

double dynamic_time_step(long iter) {

  return TIME_STEP;
  if(iter < 2000) return 200.0;
  if(iter < 5000) return 100.0;
  if(iter < 8000) return 50.0;
  else return 20.0;
}

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

/* Sign: - to + */
double eval_force(wf3d *gwf, wf3d *impwf, rgrid3d *pair_pot, rgrid3d *dpair_pot, rgrid3d *workspace1, rgrid3d *workspace2) {

  double tmp;

#if 1
  grid3d_wf_density(impwf, workspace1);
  dft_driver_convolution_prepare(workspace1, NULL);
  dft_driver_convolution_eval(workspace2, pair_pot, workspace1);
  rgrid3d_fd_gradient_x(workspace2, workspace1);
  grid3d_wf_density(gwf, workspace2);
  rgrid3d_product(workspace1, workspace1, workspace2);
  tmp = rgrid3d_integral(workspace1);   /* minus -> plus */
#else
  grid3d_wf_density(gwf, workspace1);
  dft_driver_convolution_prepare(workspace1, NULL);
  dft_driver_convolution_eval(workspace2, dpair_pot, workspace1);
  grid3d_wf_density(impwf, workspace1);
  rgrid3d_product(workspace1, workspace1, workspace2);
  tmp = -rgrid3d_integral(workspace1);
#endif

  return tmp;
}

int main(int argc, char *argv[]) {

  wf3d *gwf, *gwfp, *nwf, *nwfp, *total;
  wf3d *impwf, *impwfp; /* impurity wavefunction */
  cgrid3d *cpot_el, *cpot_super, *cpot_normal;
  rgrid3d *pair_pot, *dpair_pot, *ext_pot, *density;
  rgrid3d *vx, *vy, *vz;
  long iter;
  char filename[2048];
  double kin, pot;
  double rho0, mu0, n;
  double force, mobility, last_mobility = 0.0;
  double inv_width = 0.05;
  grid_timer timer;

  /* Set fftw planning */
  grid_set_fftw_flags(PLANNING);
  
  /* Setup grid driver parameters */
  dft_driver_setup_grid(NX, NY, NZ, STEP /* Bohr */, THREADS /* threads */);
  /* Setup frame of reference momentum */
  dft_driver_setup_momentum(KX, KY, KZ);

  printf("Run ID: %s\n", IDENT);
  
  /* Plain Orsay-Trento in real or imaginary time */
  dft_driver_setup_model(FUNCTIONAL, 1, DENSITY * (1.0 - RHON));   /* This holds the superfluid part */
  dft_driver_setup_normal_density(DENSITY * RHON);                 /* normal fluid */
  dft_driver_setup_viscosity(VISCOSITY);
  dft_driver_setup_viscosity_epsilon(EPSILON);

  /* Regular boundaries */
  dft_driver_setup_boundaries(DFT_DRIVER_BOUNDARY_REGULAR, 0.0);   /* regular periodic boundaries */
  dft_driver_setup_boundaries_damp(0.00);                          /* damping coeff., only needed for absorbing boundaries */
  dft_driver_setup_boundary_condition(DFT_DRIVER_BC_NORMAL);
  
  /* Initialize */
  dft_driver_initialize();

  /* normalization */
  dft_driver_setup_normalization(DFT_DRIVER_DONT_NORMALIZE, 4, 0.0, 0);
  
  /* get bulk density and chemical potential */
  rho0 = dft_ot_bulk_density(dft_driver_otf);
  mu0  = dft_ot_bulk_chempot(dft_driver_otf);
  printf("Bulk: rho0 = %le Angs^-3, mu0 = %le K.\n", rho0 / (0.529 * 0.529 * 0.529), mu0 * GRID_AUTOK);
  printf("Exp: rho0 = %le Angs^-3.\n", DENSITY / (0.529 * 0.529 * 0.529));
  
  /* Allocate wavefunctions and grids */
  cpot_el = dft_driver_alloc_cgrid();            
  cpot_super = dft_driver_alloc_cgrid();         
  cpot_normal = dft_driver_alloc_cgrid();        
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
  nwfp = dft_driver_alloc_wavefunction(HELIUM_MASS); /* order parameter for normal fluid (predict) */
  total = dft_driver_alloc_wavefunction(HELIUM_MASS); /* Total wavefunction (normal & super) */
  
  fprintf(stderr, "Time step in a.u. = %le\n", TIME_STEP / GRID_AUTOFS);
  fprintf(stderr, "Relative velocity = (%le, %le, %le) (a.u.)\n", VX, VY, VZ);
  fprintf(stderr, "Relative velocity = (%le, %le, %le) (m/s)\n", VX * GRID_AUTOMPS, VY * GRID_AUTOMPS, VZ * GRID_AUTOMPS);
  
  /* Initial wavefunctions. Read from file or set to initial guess */
  /* Constant density (initial guess) */
  cgrid3d_constant(gwf->grid, sqrt(rho0 * (1.0 - RHON)));
  cgrid3d_constant(nwf->grid, sqrt(rho0 * RHON));

  /* Gaussian for impurity (initial guess) */
  cgrid3d_map(impwf->grid, dft_common_cgaussian, &inv_width);
  cgrid3d_multiply(impwf->grid, 1.0 / sqrt(grid3d_wf_norm(impwf)));

  /* Read pair potential from file and do FFT */
  dft_common_potential_map(DFT_DRIVER_AVERAGE_XYZ, PSPOT, PSPOT, PSPOT, pair_pot);
  rgrid3d_fd_gradient_x(pair_pot, dpair_pot);
  dft_driver_convolution_prepare(pair_pot, dpair_pot);
  
  for(iter = 1; iter < MAXITER; iter++) {
    double ts;
    ts = dynamic_time_step(iter);
    printf("Time step = %le fs.\n", ts);
    
    grid_timer_start(&timer);
    printf("Iteration %ld took ", iter);
    /* FIRST HALF OF KINETIC ENERGY */
    dft_driver_propagate_kinetic_first(DFT_DRIVER_PROPAGATE_OTHER, impwf, IMP_STEP);
    dft_driver_propagate_kinetic_first(DFT_DRIVER_PROPAGATE_HELIUM, gwf, ts);
    dft_driver_propagate_kinetic_first(DFT_DRIVER_PROPAGATE_NORMAL, nwf, ts);

    /* PREDICT */

    /* electron external potential (TODO: potential pointing wrong way?) */
    dft_driver_total_wf(total, gwf, nwf);
    grid3d_wf_density(total, density);
    dft_driver_convolution_prepare(NULL, density);      
    dft_driver_convolution_eval(ext_pot, density, pair_pot);
    cgrid3d_copy(impwfp->grid, impwf->grid);
    grid3d_real_to_complex_re(cpot_el, ext_pot);
    dft_driver_propagate_potential(DFT_DRIVER_PROPAGATE_OTHER, impwfp, cpot_el, IMP_STEP);

    /* super external potential */
    cgrid3d_zero(cpot_super);
    dft_driver_ot_potential(total, cpot_super);
    grid3d_wf_density(impwf, density);
    dft_driver_convolution_prepare(NULL, density);
    dft_driver_convolution_eval(ext_pot, density, pair_pot);
    rgrid3d_add(ext_pot, -mu0); // chemical potential (same for super & normal)
    grid3d_add_real_to_complex_re(cpot_super, ext_pot);
    cgrid3d_copy(gwfp->grid, gwf->grid);
    dft_driver_propagate_potential(DFT_DRIVER_PROPAGATE_HELIUM, gwfp, cpot_super, ts);

    /* normal external potential */
    cgrid3d_copy(cpot_normal, cpot_super);
    dft_driver_viscous_potential(nwf, cpot_normal);
    cgrid3d_copy(nwfp->grid, nwf->grid);
    dft_driver_propagate_potential(DFT_DRIVER_PROPAGATE_NORMAL, nwfp, cpot_normal, ts);

    /* CORRECT */

    /* electron external potential (TODO: potential pointing wrong way?) */
    dft_driver_total_wf(total, gwfp, nwfp);
    grid3d_wf_density(total, density);
    dft_driver_convolution_prepare(NULL, density);      
    dft_driver_convolution_eval(ext_pot, density, pair_pot);
    grid3d_add_real_to_complex_re(cpot_el, ext_pot);
    cgrid3d_multiply(cpot_el, 0.5);
    dft_driver_propagate_potential(DFT_DRIVER_PROPAGATE_OTHER, impwf, cpot_el, IMP_STEP);

    /* super external potential */
    cgrid3d_zero(cpot_el);
    dft_driver_ot_potential(total, cpot_el); // cpot_el is temp
    grid3d_wf_density(impwfp, density);
    dft_driver_convolution_prepare(NULL, density);
    dft_driver_convolution_eval(ext_pot, density, pair_pot);
    rgrid3d_add(ext_pot, -mu0);   // Chemical potential (same for super & normal)
    grid3d_add_real_to_complex_re(cpot_el, ext_pot);
    cgrid3d_sum(cpot_super, cpot_super, cpot_el);
    cgrid3d_multiply(cpot_super, 0.5);
    dft_driver_propagate_potential(DFT_DRIVER_PROPAGATE_HELIUM, gwf, cpot_super, ts);

    /* normal external potential */
    cgrid3d_sum(cpot_normal, cpot_normal, cpot_el); // from above
    dft_driver_viscous_potential(nwf, cpot_normal);
    cgrid3d_multiply(cpot_normal, 0.5);
    dft_driver_propagate_potential(DFT_DRIVER_PROPAGATE_NORMAL, nwf, cpot_normal, ts);
    
    /* SECOND HALF OF KINETIC */
    dft_driver_propagate_kinetic_second(DFT_DRIVER_PROPAGATE_OTHER, impwf, IMP_STEP);
    dft_driver_propagate_kinetic_second(DFT_DRIVER_PROPAGATE_HELIUM, gwf, ts);
    dft_driver_propagate_kinetic_second(DFT_DRIVER_PROPAGATE_NORMAL, nwf, ts);
    
    printf("%lf wall clock seconds.\n", grid_timer_wall_clock_time(&timer));
    fflush(stdout);

    force = creal(cgrid3d_grid_expectation_value_func(NULL, center_func, impwf->grid)); // force is temp
    printf("Expectation value of position (electron): %le\n", force * GRID_AUTOANG);
    /* keep electron at origin */
    cgrid3d_shift(cpot_super, impwf->grid, -force, 0.0, 0.0);  // cpot_super is temp
    cgrid3d_copy(impwf->grid, cpot_super);
    
    if(iter && !(iter % OUTPUT)){	
      /* Impurity density */
      grid3d_wf_density(impwf, density);
      sprintf(filename, "ebubble_imp-%ld", iter);
      dft_driver_write_density(density, filename);

      /* Helium energy */
      dft_driver_convolution_prepare(NULL, density);
      dft_driver_convolution_eval(ext_pot, density, pair_pot);
      dft_driver_total_wf(total, gwf, nwf);
      kin = dft_driver_kinetic_energy(gwf);            /* Kinetic energy for gwf */
      kin += dft_driver_kinetic_energy(nwf);
      pot = dft_driver_potential_energy(total, ext_pot); /* Potential energy for gwf */
      n = dft_driver_natoms(total);
      printf("Iteration %ld background kinetic = %.30lf\n", iter, n * EKIN * GRID_AUTOK);
      printf("Iteration %ld helium natoms    = %le particles.\n", iter, n);   /* Energy / particle in K */
      printf("Iteration %ld helium kinetic   = %.30lf\n", iter, kin * GRID_AUTOK);  /* Print result in K */
      printf("Iteration %ld helium potential = %.30lf\n", iter, pot * GRID_AUTOK);  /* Print result in K */
      printf("Iteration %ld helium energy    = %.30lf\n", iter, (kin + pot) * GRID_AUTOK);  /* Print result in K */
      
      grid3d_wf_probability_flux_x(gwf, vx);
      grid3d_wf_probability_flux_x(nwf, vy);
      rgrid3d_sum(vx, vx, vy);
      printf("Iteration %ld added mass = %.30lf\n", iter, rgrid3d_integral(vx) / VX);

      force = eval_force(total, impwf, pair_pot, dpair_pot, ext_pot, density);  /* ext_pot & density are temps */
      force_normal = eval_force(nwf, impwf, pair_pot, dpair_pot, ext_pot, density);  /* ext_pot & density are temps */
      printf("Drag force on ion = %le a.u.\n", force);
      printf("E-field = %le V/m\n", -force * GRID_AUTOVPM);
      mobility = VX * GRID_AUTOMPS / (-force * GRID_AUTOVPM);
      printf("Mobility = %le [cm^2/(Vs)]\n", 1.0E4 * mobility); /* 1E4 = m^2 to cm^2 */
      printf("Hydrodynamic radius (Stokes) = %le Angs.\n", 1E10 * 1.602176565E-19 / (SBC * M_PI * mobility * RHON * VISCOSITY));
      printf("Mobility convergence = %le %%.\n", 100.0 * fabs(mobility - last_mobility) / mobility);
      last_mobility = mobility;

      grid3d_wf_density(impwf, density);
      printf("Electron asymmetry (stddev x/y) = %le\n", rgrid3d_weighted_integral(density, stddev_x, NULL) / rgrid3d_weighted_integral(density, stddev_y, NULL));

      /* write out normal fluid density */
      sprintf(filename, "ebubble_nliquid-%ld", iter);              
      grid3d_wf_density(nwf, density);
      dft_driver_write_density(density, filename);
      /* write out superfluid density */
      sprintf(filename, "ebubble_sliquid-%ld", iter);              
      grid3d_wf_density(gwf, density);
      dft_driver_write_density(density, filename);
      /* write out normal fluid velocity field */
      dft_driver_veloc_field(nwf, vx, vy, vz);
      rgrid3d_add(vx, -VX);
      sprintf(filename, "ebubble_nliquid-vx-%ld", iter);              
      dft_driver_write_density(vx, filename);
      sprintf(filename, "ebubble_nliquid-vy-%ld", iter);              
      dft_driver_write_density(vy, filename);
      sprintf(filename, "ebubble_nliquid-vz-%ld", iter);              
      dft_driver_write_density(vz, filename);
      /* write out superfluid velocity field */
      dft_driver_veloc_field(gwf, vx, vy, vz);
      rgrid3d_add(vx, -VX);
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
