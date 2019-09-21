
/*
 * "One dimensional bubble" propagating in superfluid helium (propagating along Z).
 * 1-D version with X & Y coordinates integrated over in the non-linear
 * potential.
 *
 * All input in a.u. except the time step, which is fs.
 *
 */

/* Required system headers */
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

/* libgrid headers */
#include <grid/grid.h>
#include <grid/au.h>

/* libdft headers */
#include <dft/dft.h>
#include <dft/ot.h>

#define TS 10.0 /* Time step (fs) */
#define NZ (4096)  /* Length of the 1-D grid */
#define STEP 0.2    /* Step length for the grid */
#define IITER 10000 /* Number of warm-up imaginary time iterations */
#define SITER 50000 /* Stop liquid flow after this many iterations */
#define MAXITER 80000000  /* Maximum iterations */
#define NTH 1000          /* Output liquid density every NTH iterations */
#define VZ (2.0 / GRID_AUTOMPS)  /* Liquid velocity (m/s) */

#define ABS_LEN 60.0
#define ABS_AMP (10.0 * ABS_LEN / GRID_AUTOK)

#define PRESSURE (0.0 / GRID_AUTOBAR)  /* External pressure (bar) */
#define THREADS 16                     /* Use this many OpenMP threads */

/* Use Predict-correct for propagation? */
#define PC

/* Bubble parameters - exponential repulsion (approx. electron bubble) - RADD = 19.0 */
#define A0 (3.8003E5 / GRID_AUTOK)
#define A1 (1.6245 * GRID_AUTOANG)
#define A2 0.0
#define A3 0.0
#define A4 0.0
#define A5 0.0
#define RMIN 2.0
#define RADD 6.0

/* Initial guess for bubble radius (imag. time) */
#define BUBBLE_RADIUS (15.0 / GRID_AUTOANG)

/* Function generating the initial guess (1-d sphere) */
REAL complex bubble_init(void *prm, REAL x, REAL y, REAL z) {

  double *rho0 = (REAL *) prm;

  if(FABS(z) < BUBBLE_RADIUS) return 0.0;
  return SQRT(*rho0);
}

/* Round velocity to fit the simulation box */
REAL round_veloc(REAL veloc) {

  INT n;
  REAL v;

  n = (INT) (0.5 + (NZ * STEP * DFT_HELIUM_MASS * veloc) / (HBAR * 2.0 * M_PI));
  v = ((REAL) n) * HBAR * 2.0 * M_PI / (NZ * STEP * DFT_HELIUM_MASS);
  fprintf(stderr, "Requested velocity = %le m/s.\n", veloc * GRID_AUTOMPS);
  fprintf(stderr, "Nearest velocity compatible with PBC = %le m/s.\n", v * GRID_AUTOMPS);
  return v;
}

/* Given liquid velocity, calculate the momentum */
REAL momentum(REAL vz) {

  return DFT_HELIUM_MASS * vz / HBAR;
}

/* Potential producing the bubble (centered at origin, z = 0) */
REAL bubble(void *asd, REAL x, REAL y, REAL z) {

  REAL r, r2, r4, r6, r8, r10;

  r = FABS(z);
  r -= RADD;
  if(r < RMIN) r = RMIN;

  r2 = r * r;
  r4 = r2 * r2;
  r6 = r4 * r2;
  r8 = r6 * r2;
  r10 = r8 * r2;
  /* Exponential repulsion + dispersion series */
  return A0 * EXP(-A1 * r) - A2 / r4 - A3 / r6 - A4 / r8 - A5 / r10;
}

/* Main program - we start execution here */
int main(int argc, char **argv) {

  rgrid *density, *ext_pot; /* Real grids for density and external potential */
  cgrid *potential_store;   /* Complex grid holding potential */
  wf *gwf, *gwfp;           /* Wave function + predicted wave function */
  dft_ot_functional *otf;   /* Functional pointer to define DFT */
  INT iter;                 /* Iteration counter */
  REAL rho0, mu0, vz, kz;   /* liquid density, chemical potential, velocity, momentum */
  char buf[512];            /* Buffer for file name */
  REAL complex tstep;       /* Time step (complex) */
  grid_timer timer;         /* Timer structure to record execution time */

#ifdef USE_CUDA
#define NGPUS 1                /* Number of GPUs to use */
  int gpus[NGPUS] = {0};       /* Which GPUs to use? */
  cuda_enable(1, NGPUS, gpus); /* If cuda available, enable it */
#endif

  /* Initialize threads & use wisdom */
  grid_set_fftw_flags(1);    /* FFTW planning = FFTW_MEASURE */
  grid_threads_init(THREADS);/* Initialize OpenMP threads */
  grid_fft_read_wisdom(NULL);/* Use FFTW wisdom if available */

  /* Allocate wave functions: periodic boundaries and 2nd order FFT propagator */
  if(!(gwf = grid_wf_alloc(1, 1, NZ, STEP, DFT_HELIUM_MASS, WF_PERIODIC_BOUNDARY, WF_2ND_ORDER_FFT, "gwf"))) {
    fprintf(stderr, "Cannot allocate gwf.\n");
    exit(1);
  }
  gwfp = grid_wf_clone(gwf, "gwfp"); /* Clone gwf to gwfp (same but new grid) */

  /* Moving background at velocity VZ */
  vz = round_veloc(VZ);
  printf("VZ = " FMT_R " m/s\n", vz * GRID_AUTOMPS);
  kz = momentum(VZ);
  cgrid_set_momentum(gwf->grid, 0.0, 0.0, kz);  /* Set the moving background momentum to wave functions */
  cgrid_set_momentum(gwfp->grid, 0.0, 0.0, kz);

  /* Allocate OT functional (full Orsay-Trento) */
  if(!(otf = dft_ot_alloc(DFT_OT_PLAIN, gwf, DFT_MIN_SUBSTEPS, DFT_MAX_SUBSTEPS))) {
    fprintf(stderr, "Cannot allocate otf.\n");
    exit(1);
  }

  /* Bulk density at pressure PRESSURE */
  rho0 = dft_ot_bulk_density_pressurized(otf, PRESSURE);
  mu0 = dft_ot_bulk_chempot_pressurized(otf, PRESSURE);
  printf("mu0 = " FMT_R " K/atom, rho0 = " FMT_R " Angs^-3.\n", mu0 * GRID_AUTOK, rho0 / (GRID_AUTOANG * GRID_AUTOANG * GRID_AUTOANG));

  /* Use the real grid in otf structure (otf->density) rather than allocate new grid */
  density = otf->density;
  /* Allocate space for potential grid */
  potential_store = cgrid_clone(gwf->grid, "Potential store");
  /* Allocate space for external potential */
  ext_pot = rgrid_clone(density, "ext_pot");

  /* set up external potential (function bubble) */
  rgrid_map(ext_pot, bubble, NULL);

  /* set up initial density */
  if(argc == 2) {
    /* Read starting point from checkpoint file */
    FILE *fp;
    if(!(fp = fopen(argv[1], "r"))) {
      fprintf(stderr, "Can't open checkpoint .grd file.\n");
      exit(1);
    }
    sscanf(argv[1], "bubble-" FMT_I ".grd", &iter);
    cgrid_read(gwf->grid, fp);
    fclose(fp);
    fprintf(stderr, "Check point from %s with iteration = " FMT_I "\n", argv[1], iter);
  } else {
    /* Bubble initial guess */
    cgrid_map(gwf->grid, bubble_init, &rho0);
    iter = 0;
  }

  /* Main loop over iterations */
  for ( ; iter < MAXITER; iter++) {

    /* Output every NTH iteration */
    if(iter >= IITER && !(iter % NTH)) {
      sprintf(buf, "bubble-" FMT_I, iter); /* construct file name */
      grid_wf_density(gwf, density);  /* get density from gwf */
      rgrid_write_grid(buf, density); /* write density to disk */
    }

    /* determine time step */
    if(iter < IITER) tstep = -I * TS; /* Imaginary time */
    else {
      tstep = TS; /* Real time */
      grid_wf_boundary(gwf, gwfp, ABS_AMP, rho0, 0, 1, 0, 1, ABS_LEN / STEP, NZ - ABS_LEN / STEP); /* Absorbing BC */
    }
    /* After SITER's, stop the flow */
    if(iter > SITER) {
      cgrid_set_momentum(gwf->grid, 0.0, 0.0, 0.0);   /* Reset background velocity to zero */
      cgrid_set_momentum(gwfp->grid, 0.0, 0.0, 0.0);
      /* Reset the chemical potential (remove moving background contribution) */
      mu0 = dft_ot_bulk_chempot_pressurized(otf, PRESSURE);
    }

    grid_timer_start(&timer); /* start iteration timer */
    cgrid_zero(potential_store);  /* clear potential */
    /* If PC is defined, use predict-correct */
    /* If not, use single stepping */
#ifdef PC
    /* predict-correct */
    dft_ot_potential(otf, potential_store, gwf);  /* Add O-T potential at current time */
    grid_add_real_to_complex_re(potential_store, ext_pot); /* Add external potential */
    cgrid_add(potential_store, -mu0); /* Add -chemical potential */
    grid_wf_propagate_predict(gwf, gwfp, potential_store, tstep / GRID_AUTOFS); /* predict step */
    dft_ot_potential(otf, potential_store, gwfp);   /* Get O-T potential at prediction point */
    grid_add_real_to_complex_re(potential_store, ext_pot); /* Add external potential */
    cgrid_add(potential_store, -mu0);              /* add -chemical potential */
    cgrid_multiply(potential_store, 0.5);  /* For correct step, use potential (current + future) / 2 */
    grid_wf_propagate_correct(gwf, potential_store, tstep / GRID_AUTOFS); /* Take the correct step */
#else
    /* single stepping */
    dft_ot_potential(otf, potential_store, gwf);  /* Get O-T potential */
    grid_add_real_to_complex_re(potential_store, ext_pot); /* Add external potential */
    cgrid_add(potential_store, -mu0);             /* Add -chemical potential */
    if(iter >= IITER) grid_wf_absorb_potential(gwf, potential_store, ABS_AMP, rho0);
    grid_wf_propagate(gwf, potential_store, tstep / GRID_AUTOFS);  /* Propagate */
#endif

    /* Report Wall time used for the current iteration */
    printf("Iteration " FMT_I " - Wall clock time = " FMT_R " seconds.\n", iter, grid_timer_wall_clock_time(&timer));
    /* After five iterations, write out FFTW wisdom */
    if(iter == 5) grid_fft_write_wisdom(NULL);
  }
  return 0;  /* The End */
}
