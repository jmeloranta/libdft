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
#define CU_P             /* Cu atom */
/* #define AG_P             /* Ag atom */
/* #define CU2_P            /* Cu2 */
/* #define AG2_P            /* Ag2 ion */

#define TIME_PROP 0    /* 0 = real time, 1 = imag time (only liquid) */
#define TIME_STEP 10.0 /* Time step in fs (5 for real, 10 for imag) */
#define MAXITER 600000  /* Maximum number of iterations */
#define OUTPUT 5      /* output every this iteration */
#define THREADS 1     /* # of parallel threads to use */

#define NATOMS (1E5) /* Number of atoms in the column */

/* #define MAX_VELOC (0.0 / 2.187691E6)  /* maximum allowed velocity (gas blanket) */

#define NZ 1024          /* # of grid points along x */
#define NR 512           /* # of grid points along x */
#define STEP 0.4        /* spatial step length (Bohr) */
#define DENSITY 0.0     /* bulk liquid density (0.0 = default) */
#define DAMP 0.0    /* absorbing boundary damp coefficient (2.0E-2) */
#define DAMP_R 0.0      /* damp at this distance from the boundary */

#define HELIUM_MASS (4.002602 / GRID_AUTOAMU) /* helium mass */

#if TIME_PROP == 1
#define IZ  0.0     /* debug (was 9990.0) */
#define IVZ 0.0
#else
#define IZ (0.0)                     /* ion initial position (bohr) */
#ifdef MAX_VELOC
#define IVZ (-MAX_VELOC)
#else
#define IVZ (-1.0 / 2.187691E6)     /* ion initial velocity (m/s) */
#endif
#endif

double global_time;

/************ END USER SETTINGS ************/

#include "verlet.c"

int main(int argc, char *argv[]) {

  cgrid2d *cworkspace;
  rgrid2d *current_density,  *rworkspace;
  wf2d *gwf, *gwfp;
  long iter;
  char filename[2048];
  double iz = IZ, ivz = IVZ, iaz = 0.0;
  double piz, pivz, piaz;
  double fz, imp_pe, imp_ke, liq_e;

  /* Setup DFT driver parameters (256 x 256 x 256 grid) */
  dft_driver_setup_grid_2d(NZ, NR, STEP /* Bohr */, THREADS /* threads */);

  /* Plain Orsay-Trento in real or imaginary time */
  dft_driver_setup_model_2d(DFT_OT_PLAIN, TIME_PROP, DENSITY);

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

  /* Initialize */
  dft_driver_initialize_2d();

  /* surface normalization */
  dft_driver_setup_normalization_2d(DFT_DRIVER_NORMALIZE_SURFACE, NATOMS, 20.0, 1000);

  /* Allocate wavefunctions and grids */
  gwf = dft_driver_alloc_wavefunction_2d(HELIUM_MASS);
  gwfp = dft_driver_alloc_wavefunction_2d(HELIUM_MASS);
  current_density = dft_driver_alloc_rgrid_2d();
  cworkspace = dft_driver_alloc_cgrid_2d();
  rworkspace = dft_driver_alloc_rgrid_2d();

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

#if 0
  printf("Droplet surface area = %le a.u.\n", pow(6.0 * NATOMS * sqrt(M_PI) / otf->rho0, 2.0 / 3.0));
  printf("Column surface area  = %le a.u.\n", 2.0 * sqrt(M_PI * NATOMS * STEP * (NY-1) / otf->rho0));
#endif

  /* propagate */
  for(; iter < MAXITER; iter++) {

    printf("# of He = %le.\n", grid2d_wf_norm_cyl(gwf));
    if(!(iter % OUTPUT)) {
      fprintf(stderr, "f_z = %le\n", fz);
      fprintf(stderr, "z = %le [a.u.].\n", iz);
      fprintf(stderr, "v = %le [m/s].\n", ivz * 2.187691E6);
      fprintf(stderr, "Traj %le %le\n", iz, ivz * 2.187691E6);
      imp_pe = rgrid2d_weighted_integral_cyl(current_density, &pot_func, (void *) &iz) * GRID_AUTOK;
      fprintf(stderr, "PE = %le K.\n", imp_pe);
      imp_ke = 0.5 * IMASS * ivz * ivz * GRID_AUTOK;
      fprintf(stderr, "KE = %le K.\n", imp_ke);
      rgrid2d_zero(rworkspace); // don't double count PE in Total E.
      liq_e = dft_driver_energy_2d(gwf, rworkspace) * GRID_AUTOK;
      fprintf(stderr, "Liq E = %.20le K.\n", liq_e);
      fprintf(stderr, "Liq E / particle = %.20le K.\n", (liq_e + imp_pe) / grid2d_wf_norm_cyl(gwf));
      fprintf(stderr, "TE = %le K.\n", imp_ke + imp_pe + liq_e);
      sprintf(filename, "iter-%ld-density", iter);
      grid2d_wf_density(gwf, rworkspace);
      dft_driver_write_density_2d(rworkspace, filename);
      sprintf(filename, "iter-%ld-wf", iter);
      dft_driver_write_grid_2d(gwf->grid, filename);
      grid2d_wf_density(gwf, rworkspace);
    }

    global_time = (TIME_STEP / GRID_AUTOFS) * (double) iter;

    /* PREDICT */

    grid2d_wf_density(gwf, current_density);
    rgrid2d_map_cyl(rworkspace, pot_func, &iz);

    (void) dft_driver_propagate_predict_2d(DFT_DRIVER_PROPAGATE_HELIUM, rworkspace, gwf, gwfp, cworkspace, TIME_STEP, iter);

    if(!TIME_PROP) {
      piz = iz; pivz = ivz; piaz = iaz;
      (void) propagate_impurity(&piz, &pivz, &piaz, current_density); 
      rgrid2d_map_cyl(rworkspace, pot_func, &piz);
    }

    /* CORRECT */

    (void) dft_driver_propagate_correct_2d(DFT_DRIVER_PROPAGATE_HELIUM, rworkspace, gwf, gwfp, cworkspace, TIME_STEP, iter);

    if(!TIME_PROP) {
      grid2d_wf_density(gwf, current_density);
      fz = propagate_impurity(&iz, &ivz, &iaz, current_density);
    }
  }
}
