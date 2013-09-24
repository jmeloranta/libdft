/*
 * One impurity atom dynamics in superfluid helium.
 * Impurity treated classically. Drag by electric field.
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

/********** USER SETTINGS *********/

/* #define EXP_P           /* pure exponential repulsion */
/* #define ZERO_P             /* zero potential */
/* #define CA_P            /* Ca+ ion */
/* #define K_P             /* K+ ion */
/* #define BE_P            /* Be+ ion */
/* #define SR_P            /* Sr+ ion */
/* #define BA_P            /* Ba+ ion */
#define CL_M            /* Cl- ion */
/* #define F_M             /* F- ion */
/* #define I_M             /* I- ion */
/* #define BR_M            /* Br- ion */

#define TIME_PROP 1    /* 0 = real time, 1 = imag time (only liquid) */
#define TIME_STEP 5.0 /* Time step in fs (5 for real, 10 for imag) */
#define MAXITER 60000  /* Maximum number of iterations */
#define OUTPUT 40      /* output every this iteration */
#define THREADS  16     /* # of parallel threads to use */

/* #define FORCE_SPHERICAL /* Force spherical symmetry */
/* #define UNCOUPLED   /* Uncouple ion from the liquid */

#define NZ 2048          /* # of grid points along x */
#define NR 1024           /* # of grid points along x */
#define STEP 0.2        /* spatial step length (Bohr) */
#define DENSITY 0.0     /* bulk liquid density (0.0 = default) */
#define DAMP 0.0    /* absorbing boundary damp coefficient (2.0E-2) */
#define DAMP_R 20.0      /* damp at this distance from the boundary */

//#define T0 (600000.0 / GRID_AUTOFS) /* warm up period (1ps) */
//#define EVAL 1.0E-7              /* final efield (3.98E-8) */
//#define EFIELDZ ((global_time < T0)?((EVAL/T0) * global_time):(EVAL))
//#define EFIELDZ (1E-7) /* E field along z (3.98E-8) */
#define EFIELDZ 1.0E-7     /* E field along z (imag time) */

#define HELIUM_MASS (4.002602 / GRID_AUTOAMU) /* helium mass */

#ifdef ZERO_P
#define IMASS (40.078 / GRID_AUTOAMU) /* ion atom mass (Ca) -- arbitrarily picked */
#endif

#ifdef EXP_P
#define IMASS (40.078 / GRID_AUTOAMU) /* ion atom mass (Ca) -- arbitrarily picked */
#endif

#ifdef CA_P
#define IMASS (40.078 / GRID_AUTOAMU) /* ion atom mass (Ca) */
#endif

#ifdef K_P
#define IMASS (39.0983 / GRID_AUTOAMU) /* ion atom mass (K) */
#endif

#ifdef BE_P
#define IMASS (9.0122 / GRID_AUTOAMU) /* ion atom mass (Be) */
#endif

#ifdef SR_P
#define IMASS (87.62 / GRID_AUTOAMU) /* ion atom mass (Sr) */
#endif

#ifdef BA_P
#define IMASS (137.327 / GRID_AUTOAMU) /* ion atom mass (Ba) */
#endif

#ifdef CL_M
#define IMASS (35.4527 / GRID_AUTOAMU) /* ion atom mass (Cl) */
#endif

#ifdef F_M
#define IMASS (18.998403 / GRID_AUTOAMU) /* ion atom mass (F) */
#endif

#ifdef I_M
#define IMASS (126.9044 / GRID_AUTOAMU) /* ion atom mass (I) */
#endif

#ifdef BR_M
#define IMASS (79.904 / GRID_AUTOAMU) /* ion atom mass (Br) */
#endif

#define IZ 0.0                     /* ion initial position (bohr) */
#define IVZ (0.0 / 2.187691E6)     /* ion initial velocity (m/s) */

double global_time;

/************ END USER SETTINGS ************/

#include "verlet.c"

int main(int argc, char *argv[]) {

  cgrid2d *cworkspace;
  rgrid2d *current_density,  *rworkspace;
  rgrid1d *radial;
  wf2d *gwf, *gwfp;
  long iter;
  char filename[2048];
  double iz = IZ, ivz = IVZ, tmp;
  double piz, pivz;
  double iaz = 0.0;
  double piaz;
  double cur_vel = IVZ;
  double fz;

  /* Setup DFT driver parameters (256 x 256 x 256 grid) */
  dft_driver_setup_grid_2d(NZ, NR, STEP /* Bohr */, THREADS /* threads */);

  /* Plain Orsay-Trento in real or imaginary time */
  dft_driver_setup_model_2d(DFT_OT_PLAIN + DFT_OT_HD, TIME_PROP, DENSITY);
  //dft_driver_setup_model_2d(DFT_GP, TIME_PROP, DENSITY);

  /* no absorbing boundaries */
  if(DAMP == 0.0) {
    fprintf(stderr, "Regular boundaries.\n");
    dft_driver_setup_boundaries_2d(DFT_DRIVER_BOUNDARY_REGULAR, 0.0);
    dft_driver_setup_boundaries_damp_2d(0.0);
  } else  {
    fprintf(stderr, "Absorbing boundaries (%le, %le)\n", DAMP_R, DAMP);
    dft_driver_setup_boundaries_2d(DFT_DRIVER_BOUNDARY_ABSORB, DAMP_R);
    dft_driver_setup_boundaries_damp_2d(DAMP);
  }

  /* bulk normalization */
  dft_driver_setup_normalization_2d(DFT_DRIVER_NORMALIZE_BULK, 0, 0.0, 0);

  /* Initialize */
  dft_driver_initialize_2d();

  /* Allocate wavefunctions and grids */
  gwf = dft_driver_alloc_wavefunction_2d(HELIUM_MASS);
  gwfp = dft_driver_alloc_wavefunction_2d(HELIUM_MASS);
  current_density = dft_driver_alloc_rgrid_2d();
  cworkspace = dft_driver_alloc_cgrid_2d();
  rworkspace = dft_driver_alloc_rgrid_2d();
  radial = rgrid1d_alloc(NZ, STEP, RGRID1D_PERIODIC_BOUNDARY, 0);

  iter = 0;
  if(!TIME_PROP) {  /* ext_pot is temp here */
    dft_driver_read_density_2d(current_density, "classical"); /* Initial density */
    rgrid2d_power(current_density, current_density, 0.5);
    grid2d_real_to_complex_re(gwf->grid, current_density);
    cgrid2d_copy(gwfp->grid, gwf->grid);
    grid2d_wf_density(gwf, current_density);
  } else if(argc > 1) {
    iter = atol(argv[1]);
    sprintf(filename, "iter-%ld-density", iter);
    fprintf(stderr, "Continuing imag. time iterations from %s\n", filename);
    iter++;
    dft_driver_read_density_2d(current_density, filename);
    rgrid2d_power(current_density, current_density, 0.5);
    grid2d_real_to_complex_re(gwf->grid, current_density);
    cgrid2d_copy(gwfp->grid, gwf->grid);
    grid2d_wf_density(gwf, current_density);
  }

  fprintf(stderr, "Time step in a.u. = %le\n", TIME_STEP / GRID_AUTOFS);

  /* propagate */
  for(; iter < MAXITER; iter++) {

    global_time = (TIME_STEP / GRID_AUTOFS) * (double) iter;

    fprintf(stderr, "Current EFIELDZ = %le\n", EFIELDZ);

#ifdef FORCE_SPHERICAL
    fprintf(stderr, "Forcing spherical symmetry.\n");
    dft_driver_force_spherical_2d(gwf, iz);
    cgrid2d_copy(gwfp->grid, gwf->grid);
#endif
    /* PREDICT */
    grid2d_wf_density(gwf, current_density);
    ZI = iz;
    rgrid2d_map_cyl(rworkspace, pot_func2, NULL);
    (void) dft_driver_propagate_predict_2d(DFT_DRIVER_PROPAGATE_HELIUM, rworkspace, gwf, gwfp, cworkspace, TIME_STEP, iter);
    if(!TIME_PROP) {
      ZI = iz;
      piz = iz;
      pivz = ivz;
      piaz = iaz;
      grid2d_wf_density(gwfp, current_density);
      (void) propagate_impurity(&piz, &pivz, &piaz, current_density);
      ZI = piz;
      rgrid2d_map_cyl(rworkspace, pot_func2, NULL);
    }
    /* CORRECT */
    (void) dft_driver_propagate_correct_2d(DFT_DRIVER_PROPAGATE_HELIUM, rworkspace, gwf, gwfp, cworkspace, TIME_STEP, iter);
    if(!TIME_PROP) {
      ZI = iz;
      grid2d_wf_density(gwf, current_density);
      fz = propagate_impurity(&iz, &ivz, &iaz, current_density);
      fprintf(stderr, "z = %le [a.u.].\n", iz);
      fprintf(stderr, "v = %le [m/s].\n", ivz * 2.187691E6);
      fprintf(stderr, "Added mass = %le (in units of He atoms).\n", IMASS * ((EFIELDZ / fz) - 1.0) / HELIUM_MASS);
      fprintf(stderr, "Added mass_2 = %le (in units of He atoms).\n", ((2.0 * EFIELDZ * (iz - IZ) / (ivz * ivz)) - IMASS) / HELIUM_MASS);
      ZI = iz;
      fprintf(stderr, "PE = %le K.\n", rgrid2d_weighted_integral_cyl(current_density, pot_func) * GRID_AUTOK);
      fprintf(stderr, "f_z = %le, f_z - EFIELDZ = %le\n", fz, fz - EFIELDZ);
      //      fprintf(stderr, "R_b = %le\n", dft_driver_spherical_rb_2d(current_density));
      fprintf(stderr, "Mobility = %le [cm^2/(Vs)]\n", 100.0 * ivz * 2.187691E6 / (EFIELDZ * 5.1422E9));
    }

    /* Write output to a file iter-XX.{x,y,z,grd} */
    if(!(iter % OUTPUT)) {
      long ii;
      FILE *fp;
      sprintf(filename, "iter-%ld-density", iter);
      grid2d_wf_density(gwf, rworkspace);
      dft_driver_write_density_2d(rworkspace, filename);
      dft_driver_radial_2d(radial, rworkspace, 0.01, 0.01, iz);
      sprintf(filename, "iter-%ld-spherical", iter);
      fp = fopen(filename, "w");
      for (ii = 0; ii < radial->nx; ii++)
	fprintf(fp, "%le %le\n", STEP * (double) ii, radial->value[ii]);
      fclose(fp);
      sprintf(filename, "iter-%ld-wf", iter);
      dft_driver_write_grid_2d(gwf->grid, filename);
      grid2d_wf_density(gwf, rworkspace);
      fprintf(stderr, "Max norm = %le.\n", dft_driver_norm_2d(rworkspace));
    }
  }
}
