
/*
 * "One dimensional bubble" propagating in superfluid helium (propagating along Z).
 * 1-D version with X & Y coordinates integrated over in the non-linear
 * potential.
 *
 * All input in a.u. except the time step, which is fs.
 *
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <grid/grid.h>
#include <grid/au.h>
#include <dft/dft.h>
#include <dft/ot.h>

#define TS 1.0 /* fs */
#define NZ (32768)
#define STEP 0.2
#define IITER 200000
#define SITER 250000
#define MAXITER 80000000
#define NTH 2000
#define VZ (2.0 / GRID_AUTOMPS)

#define PRESSURE (0.0 / GRID_AUTOBAR)
#define THREADS 0

#define HELIUM_MASS (4.002602 / GRID_AUTOAMU)

/* Bubble parameters using exponential repulsion (approx. electron bubble) - RADD = 19.0 */
#define A0 (3.8003E5 / GRID_AUTOK)
#define A1 (1.6245 * GRID_AUTOANG)
#define A2 0.0
#define A3 0.0
#define A4 0.0
#define A5 0.0
#define RMIN 2.0
#define RADD 6.0
#define BUBBLE_RADIUS (15.0 / GRID_AUTOANG)

REAL complex bubble_init(void *prm, REAL x, REAL y, REAL z) {

  double *rho0 = (REAL *) prm;

  if(FABS(z) < BUBBLE_RADIUS) return 0.0;
  return SQRT(*rho0);
}

REAL round_veloc(REAL veloc) {   // Round to fit the simulation box

  INT n;
  REAL v;

  n = (INT) (0.5 + (NZ * STEP * HELIUM_MASS * veloc) / (HBAR * 2.0 * M_PI));
  v = ((REAL) n) * HBAR * 2.0 * M_PI / (NZ * STEP * HELIUM_MASS);
  fprintf(stderr, "Requested velocity = %le m/s.\n", veloc * GRID_AUTOMPS);
  fprintf(stderr, "Nearest velocity compatible with PBC = %le m/s.\n", v * GRID_AUTOMPS);
  return v;
}

REAL momentum(REAL vz) {

  return HELIUM_MASS * vz / HBAR;
}

/* Bubble potential (centered at origin, z = 0) */
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
  return A0 * EXP(-A1 * r) - A2 / r4 - A3 / r6 - A4 / r8 - A5 / r10;
}

int main(int argc, char **argv) {

  rgrid *density, *ext_pot;
  cgrid *potential_store;
  wf *gwf, *gwfp;
  dft_ot_functional *otf;
  INT iter;
  REAL rho0, mu0, vz, kz;
  char buf[512];
  REAL complex tstep;
  grid_timer timer;

#ifdef USE_CUDA
  cuda_enable(1);
#endif

  /* Initialize threads & use wisdom */
  grid_set_fftw_flags(1);    // FFTW_MEASURE
  grid_threads_init(THREADS);
  grid_fft_read_wisdom(NULL);

  /* Allocate wave functions */
  if(!(gwf = grid_wf_alloc(1, 1, NZ, STEP, DFT_HELIUM_MASS, WF_PERIODIC_BOUNDARY, WF_2ND_ORDER_PROPAGATOR, "gwf"))) {
    fprintf(stderr, "Cannot allocate gwf.\n");
    exit(1);
  }
  gwfp = grid_wf_clone(gwf, "gwfp");

  /* Allocate OT functional */
  if(!(otf = dft_ot_alloc(DFT_OT_PLAIN | DFT_OT_BACKFLOW | DFT_OT_KC | DFT_OT_HD, gwf, DFT_MIN_SUBSTEPS, DFT_MAX_SUBSTEPS))) {
    fprintf(stderr, "Cannot allocate otf.\n");
    exit(1);
  }
  rho0 = dft_ot_bulk_density_pressurized(otf, PRESSURE);
  mu0 = dft_ot_bulk_chempot_pressurized(otf, PRESSURE);
  printf("mu0 = " FMT_R " K/atom, rho0 = " FMT_R " Angs^-3.\n", mu0 * GRID_AUTOK, rho0 / (GRID_AUTOANG * GRID_AUTOANG * GRID_AUTOANG));

  /* Moving background */
  vz = round_veloc(VZ);
  printf("VZ = " FMT_R " m/s\n", vz * GRID_AUTOMPS);
  kz = momentum(VZ);
  cgrid_set_momentum(gwf->grid, 0.0, 0.0, kz);
  cgrid_set_momentum(gwfp->grid, 0.0, 0.0, kz);

  /* Allocate space for external potential */
  density = otf->density;
  potential_store = cgrid_clone(gwf->grid, "Potential store");
  ext_pot = rgrid_clone(density, "ext_pot");

  /* set up external potential */
  rgrid_map(ext_pot, bubble, NULL);

  /* set up initial density */
  if(argc == 2) {
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
    cgrid_map(gwf->grid, bubble_init, &rho0);
    iter = 0;
  }

  for ( ; iter < MAXITER; iter++) {

    if(!(iter % NTH)) {
      sprintf(buf, "bubble-" FMT_I, iter);
      grid_wf_density(gwf, density);
      rgrid_write_grid(buf, density);
    }

    if(iter < IITER) tstep = -I * TS;
    else tstep = TS;

    if(iter > SITER) {
      cgrid_set_momentum(gwf->grid, 0.0, 0.0, 0.0);
      cgrid_set_momentum(gwfp->grid, 0.0, 0.0, 0.0);
    }

    grid_timer_start(&timer);
    cgrid_zero(potential_store);
    cgrid_copy(gwfp->grid, gwf->grid);
    dft_ot_potential(otf, potential_store, gwf);
    cgrid_add(potential_store, -mu0);
    grid_wf_propagate_predict(gwfp, potential_store, tstep / GRID_AUTOFS);
    dft_ot_potential(otf, potential_store, gwfp);
    cgrid_add(potential_store, -mu0);
    cgrid_multiply(potential_store, 0.5);  // Use (current + future) / 2
    grid_wf_propagate_correct(gwf, potential_store, tstep / GRID_AUTOFS);
    printf("Iteration " FMT_I " - Wall clock time = " FMT_R " seconds.\n", iter, grid_timer_wall_clock_time(&timer));
    if(iter == 5) grid_fft_write_wisdom(NULL);
  }
  return 0;
}
