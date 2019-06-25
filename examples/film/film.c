/*
 * Thin film of superfluid helium (0 K).
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
#define NX 32
#define NY 512
#define NZ 512
#define STEP 1.0
#define NTH 1000
#define THREADS 0

/* Start simulation after this many iterations */
#define START 1000

/* Number of He atoms (0 = no normalization) */
#define NHE 0

/* Imag. time component */
#define ITS (0.05 * TS)

#define PRESSURE (0.0 / GRID_AUTOBAR)

REAL rho0;

/* positions for vortex lines in yz-plane (vortex line along x) */

#define LINE1_Y   -20.0
#define LINE1_Z   0.0
#define DIR1 1.0
#define OFFSET1 0.0
// #define FIXLINE1

#define LINE2_Y  20.0
#define LINE2_Z  0.0
#define DIR2 1.0
#define OFFSET2 0.0
// #define FIXLINE2

/* "stick" holding vortex line in place */
REAL stick(void *prm, REAL x, REAL y, REAL z) {

  REAL dy, dz, val = 0.0;

#ifdef FIXLINE1
  dy = LINE1_Y - y;
  dz = LINE1_Z - z;

  if(SQRT(dy*dy + dz*dz) < STEP/2.0) val += 1E-2;
#endif

#ifdef FIXLINE2
  dy = LINE2_Y - y;
  dz = LINE2_Z - z;

  if(SQRT(dy*dy + dz*dz) < STEP/2.0) val += 1E-2;
#endif

  return val;
}

/* vortex ring initial guess (ring in yz-plane) */
REAL complex vline(void *prm, REAL x, REAL y, REAL z) {

  REAL r, angle, dir, offset;

  x = ((REAL *) prm)[0] - x;
  y = ((REAL *) prm)[1] - y;
  z = ((REAL *) prm)[2] - z;
  dir = ((REAL *) prm)[3];  
  offset = ((REAL *) prm)[4];
  r = SQRT(y * y + z * z);
  angle = ATAN2(y, z);

  return (1.0 - EXP(-r)) * SQRT(rho0) * CEXP(I * dir * (angle + offset));
}

int main(int argc, char **argv) {

  dft_ot_functional *otf;
  cgrid *potential_store;
  rgrid *rworkspace, *ext_pot;
  wf *gwf, *gwfp;
  INT iter;
  REAL mu0, kin, pot, n;
  char buf[512];
  grid_timer timer;
  REAL line1[] = {0.0, LINE1_Y, LINE1_Z, DIR1, OFFSET1}, line2[] = {0.0, LINE2_Y, LINE2_Z, DIR2, OFFSET2};
  REAL complex tstep;

#ifdef USE_CUDA
  cuda_enable(1);  // enable CUDA ?
#endif

  /* Initialize threads & use wisdom */
  grid_set_fftw_flags(1);    // FFTW_MEASURE
  grid_threads_init(THREADS);
  grid_fft_read_wisdom(NULL);

  /* Allocate wave functions */
//  if(!(gwf = grid_wf_alloc(NX, NY, NZ, STEP, DFT_HELIUM_MASS, WF_FFT_EOO_BOUNDARY, WF_2ND_ORDER_FFT, "gwf"))) {
  if(!(gwf = grid_wf_alloc(NX, NY, NZ, STEP, DFT_HELIUM_MASS, WF_PERIODIC_BOUNDARY, WF_2ND_ORDER_FFT, "gwf"))) {
    fprintf(stderr, "Cannot allocate gwf.\n");
    exit(1);
  }
  gwfp = grid_wf_clone(gwf, "gwfp");

  /* Allocate OT functional */
  if(!(otf = dft_ot_alloc(DFT_OT_PLAIN, gwf, DFT_MIN_SUBSTEPS, DFT_MAX_SUBSTEPS))) {
    fprintf(stderr, "Cannot allocate otf.\n");
    exit(1);
  }
  rho0 = dft_ot_bulk_density_pressurized(otf, PRESSURE);
  mu0 = dft_ot_bulk_chempot_pressurized(otf, PRESSURE);
  printf("mu0 = " FMT_R " K/atom, rho0 = " FMT_R " Angs^-3.\n", mu0 * GRID_AUTOK, rho0 / (GRID_AUTOANG * GRID_AUTOANG * GRID_AUTOANG));

  /* Allocate space for external potential */
  potential_store = cgrid_clone(gwf->grid, "potential_store"); /* temporary storage */
  rworkspace = rgrid_clone(otf->density, "rworkspace"); /* temporary storage */
  ext_pot = rgrid_clone(otf->density, "rworkspace"); /* potential storage */
 
  /* setup initial guess for two vortex lines */
  grid_wf_map(gwf, vline, line1);
  grid_wf_map(gwfp, vline, line2);
  cgrid_product(gwf->grid, gwf->grid, gwfp->grid);
  cgrid_multiply(gwf->grid, 1.0 / SQRT(rho0));

  /* external potential */
  rgrid_map(ext_pot, &stick, NULL);

#if NHE != 0
  gwf->norm = 5000;
#endif

  for (iter = 1; iter < 800000; iter++) {
    if(iter < START) tstep = -I * TS / GRID_AUTOFS;
    else tstep = (TS - I * ITS) / GRID_AUTOFS;
    if(iter == 1 || !(iter % NTH)) {
      sprintf(buf, "film-" FMT_I, iter);
      cgrid_write_grid(buf, gwf->grid);
      kin = grid_wf_energy(gwf, NULL);            /* Kinetic energy for gwf */
      dft_ot_energy_density(otf, rworkspace, gwf);
      pot = rgrid_integral(rworkspace);
      n = grid_wf_norm(gwf);
      printf("Iteration " FMT_I " helium natoms    = " FMT_R " particles.\n", iter, n);   /* Energy / particle in K */
      printf("Iteration " FMT_I " helium kinetic   = " FMT_R "\n", iter, kin * GRID_AUTOK);  /* Print result in K */
      printf("Iteration " FMT_I " helium potential = " FMT_R "\n", iter, pot * GRID_AUTOK);  /* Print result in K */
      printf("Iteration " FMT_I " helium energy    = " FMT_R "\n", iter, (kin + pot) * GRID_AUTOK);  /* Print result in K */
      fflush(stdout);
    }

    if(iter == 5) grid_fft_write_wisdom(NULL);

    grid_timer_start(&timer);

    /* Predict-Correct */
    grid_real_to_complex_re(potential_store, ext_pot);
    dft_ot_potential(otf, potential_store, gwf);
    cgrid_add(potential_store, -mu0);
    grid_wf_propagate_predict(gwf, gwfp, potential_store, tstep);
    dft_ot_potential(otf, potential_store, gwfp);
    cgrid_add(potential_store, -mu0);
    grid_add_real_to_complex_re(potential_store, ext_pot);
    cgrid_multiply(potential_store, 0.5);  // Use (current + future) / 2
    grid_wf_propagate_correct(gwf, potential_store, tstep);
    // Chemical potential included - no need to normalize
#if NHE != 0
    grid_wf_normalize(gwf);
#endif
    printf("Iteration " FMT_I " - Wall clock time = " FMT_R " seconds.\n", iter, grid_timer_wall_clock_time(&timer));
  }
  return 0;
}
