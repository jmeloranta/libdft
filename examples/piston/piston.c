/*
 * Create sudden liquid compression by a moving piston.
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

#define TS 5.0 /* fs */
#define NX 1024       /* simulation box size */
#define NY 256
#define NZ 256
#define STEP 1.0     /* spatial grid step */
#define MAXITER 80000 /* maximum iterations */
#define INITIAL 400   /* Initial imaginary iterations */
#define NTH 100      /* output every NTH real time iterations */
#define THREADS 0    /* 0 = use all cores */

#define PISTON_VELOC (230.0 / GRID_AUTOMPS)   /* m/s */
#define PISTON_DIST  20.0     /* Bohr */

#define PRESSURE (1.0 / GRID_AUTOBAR)

#define HELIUM_MASS (4.002602 / GRID_AUTOAMU)

/* Bubble parameters using exponential repulsion (approx. electron bubble) - RADD = 19.0 */
#define A0 (3.8003E5 / GRID_AUTOK)
#define A1 (1.6245 * GRID_AUTOANG)
#define A2 0.0
#define A3 0.0
#define A4 0.0
#define A5 0.0
#define RMIN 2.0
#define RADD (6.0 + piston_pos)

REAL piston_pos = 0.0;

REAL piston(REAL time) {

  return PISTON_VELOC * time;
}

/* Impurity must always be at the origin (dU/dx) */
REAL dpot_func(void *NA, REAL x, REAL y, REAL z) {

  REAL r, rp, r2, r3, r5, r7, r9, r11;

  rp = SQRT(x * x + y * y + z * z);
  r = rp - RADD;
  if(r < RMIN) return 0.0;

  r2 = r * r;
  r3 = r2 * r;
  r5 = r2 * r3;
  r7 = r5 * r2;
  r9 = r7 * r2;
  r11 = r9 * r2;
  
  return (x / rp) * (-A0 * A1 * EXP(-A1 * r) + 4.0 * A2 / r5 + 6.0 * A3 / r7 + 8.0 * A4 / r9 + 10.0 * A5 / r11);
}

REAL pot_func(void *NA, REAL x, REAL y, REAL z) {

  REAL r, r2, r4, r6, r8, r10;

  r = SQRT(x * x + y * y + z * z);
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

  dft_ot_functional *otf;
  rgrid *ext_pot, *rworkspace;
  cgrid *potential_store;
  wf *gwf, *gwfp;
  INT iter;
  REAL mu0, rho0;
  char buf[512];
  grid_timer timer;

#ifdef USE_CUDA
  cuda_enable(1);
#endif

  /* Initialize threads & use wisdom */
  grid_set_fftw_flags(1);    // FFTW_MEASURE
  grid_threads_init(THREADS);
  grid_fft_read_wisdom(NULL);

  /* Allocate wave functions */
  if(!(gwf = grid_wf_alloc(NX, NY, NZ, STEP, DFT_HELIUM_MASS, WF_PERIODIC_BOUNDARY, WF_2ND_ORDER_PROPAGATOR, "gwf"))) {
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

  /* Allocate space for external potential */
  ext_pot = rgrid_clone(otf->density, "ext_pot");
  rworkspace = rgrid_clone(otf->density, "rworkspace");
  potential_store = cgrid_clone(gwf->grid, "potential_store"); /* temporary storage */

  /* map potential */
  rgrid_map(ext_pot, pot_func, NULL);

  grid_wf_constant(gwf, SQRT(rho0));

  /* Imag time iterations */
  for (iter = 0; iter < INITIAL; iter++) {

    if(iter == 5) grid_fft_write_wisdom(NULL);

    grid_timer_start(&timer);

    /* Predict-Correct */
    cgrid_copy(gwfp->grid, gwf->grid);
    grid_real_to_complex_re(potential_store, ext_pot);
    dft_ot_potential(otf, potential_store, gwf);
    cgrid_add(potential_store, -mu0);
    grid_wf_propagate_predict(gwfp, potential_store, -I * TS / GRID_AUTOFS);
    grid_add_real_to_complex_re(potential_store, ext_pot);
    dft_ot_potential(otf, potential_store, gwfp);
    cgrid_add(potential_store, -mu0);
    cgrid_multiply(potential_store, 0.5);  // Use (current + future) / 2
    grid_wf_propagate_correct(gwf, potential_store, -I * TS / GRID_AUTOFS);
    // Chemical potential included - no need to normalize

    printf("Iteration " FMT_I " - Wall clock time = " FMT_R " seconds.\n", iter, grid_timer_wall_clock_time(&timer));
  }

  /* Real time iterations */
  for (iter = 0; iter < MAXITER; iter++) {

    grid_timer_start(&timer);

    /* Predict-Correct */
    cgrid_copy(gwfp->grid, gwf->grid);
    grid_real_to_complex_re(potential_store, ext_pot);
    dft_ot_potential(otf, potential_store, gwf);
    cgrid_add(potential_store, -mu0);
    grid_wf_propagate_predict(gwfp, potential_store, TS / GRID_AUTOFS);
    grid_add_real_to_complex_re(potential_store, ext_pot);
    dft_ot_potential(otf, potential_store, gwfp);
    cgrid_add(potential_store, -mu0);
    cgrid_multiply(potential_store, 0.5);  // Use (current + future) / 2
    grid_wf_propagate_correct(gwf, potential_store, TS / GRID_AUTOFS);
    // Chemical potential included - no need to normalize

    printf("Iteration " FMT_I " - Wall clock time = " FMT_R " seconds.\n", iter, grid_timer_wall_clock_time(&timer));

    /* Move potential */
    if(piston_pos < PISTON_DIST) {
      piston_pos = piston(((REAL) iter) * TS / GRID_AUTOFS);
      rgrid_map(ext_pot, pot_func, NULL);
      /* end move */
    }
    if(!(iter % NTH)) {
      sprintf(buf, "piston-" FMT_I, iter);
      grid_wf_density(gwf, rworkspace);
      rgrid_write_grid(buf, rworkspace);
    }
  }
  return 0;
}
