/*
 * Create a vortex ring in superfluid helium centered around Z = 0.
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

#define TS 0.0 /* fs */
#define ITS 1.0
#define NX 256
#define NY 256
#define NZ 256
#define STEP 0.25
#define ITERS 200
#define THREADS 0

#define LX (-15.0)
#define LY (-15.0)
#define LZ (-15.0)
#define UX 15.0
#define UY 15.0
#define UZ 15.0

//#define USE_FULL_VOLUME

//#undef USE_CUDA

#define RING_RADIUS 8.0

#define PRESSURE (0.0 / GRID_AUTOBAR)

REAL rho0;

/* vortex ring initial guess */
REAL complex vring(void *asd, REAL x, REAL y, REAL z) {

#ifdef RING_RADIUS
  REAL xs = SQRT(x * x + y * y) - RING_RADIUS;
  REAL ys = z;
  REAL angle = ATAN2(ys,xs), r = SQRT(xs*xs + ys*ys);
 
  return (1.0 - EXP(-r)) * SQRT(rho0) * CEXP(I * angle);
#else
  return SQRT(rho0) * dft_initial_vortex_z_n1(NULL, x, y, z);
#endif
}

REAL mod_grid_wf_kinetic_energy_cn(wf *gwf) {

  cgrid *grid = gwf->grid;

  if(!gwf->cworkspace) gwf->cworkspace = cgrid_alloc(grid->nx, grid->ny, grid->nz, grid->step, grid->value_outside, grid->outside_params_ptr, "WF cworkspace");

  /* (-2m/hbar^2) T psi */
  cgrid_fd_laplace(gwf->grid, gwf->cworkspace);
  cgrid_multiply(gwf->cworkspace, -HBAR * HBAR / (2.0 * gwf->mass));

  /* int psi^* (T + V) psi d^3r */
#ifdef USE_FULL_VOLUME
  return CREAL(cgrid_integral_of_conjugate_product(gwf->grid, gwf->cworkspace));
#else
  cgrid_conjugate_product(gwf->cworkspace, gwf->grid, gwf->cworkspace);
  return CREAL(cgrid_integral_region(gwf->cworkspace, LX, UX, LY, UY, LZ, UZ));
#endif
}

int main(int argc, char **argv) {

  dft_ot_functional *otf;
  cgrid *potential_store;
  rgrid *rworkspace;
  wf *gwf, *gwfp;
  INT iter;
  REAL mu0, kin, pot, n, e0;
  char buf[512];
  grid_timer timer;

#ifdef USE_CUDA
#define NGPUS 1
int gpus[] = {0};
  cuda_enable(1, NGPUS, gpus);  // enable CUDA ?
#endif

  /* Initialize threads & use wisdom */
  grid_set_fftw_flags(1);    // FFTW_MEASURE
  grid_threads_init(THREADS);
  grid_fft_read_wisdom(NULL);

  /* Allocate wave functions */
  if(!(gwf = grid_wf_alloc(NX, NY, NZ, STEP, DFT_HELIUM_MASS, WF_PERIODIC_BOUNDARY, WF_2ND_ORDER_FFT, "gwf"))) {
    fprintf(stderr, "Cannot allocate gwf.\n");
    exit(1);
  }
  gwfp = grid_wf_clone(gwf, "gwfp");

  /* Allocate OT functional */
  if(!(otf = dft_ot_alloc(DFT_OT_PLAIN | DFT_OT_KC | DFT_OT_BACKFLOW, gwf, DFT_MIN_SUBSTEPS, DFT_MAX_SUBSTEPS))) {
    fprintf(stderr, "Cannot allocate otf.\n");
    exit(1);
  }
  rho0 = dft_ot_bulk_density_pressurized(otf, PRESSURE);
  mu0 = dft_ot_bulk_chempot_pressurized(otf, PRESSURE);
  printf("mu0 = " FMT_R " K/atom, rho0 = " FMT_R " Angs^-3.\n", mu0 * GRID_AUTOK, rho0 / (GRID_AUTOANG * GRID_AUTOANG * GRID_AUTOANG));

  /* Allocate space for external potential */
  potential_store = cgrid_clone(gwf->grid, "potential_store"); /* temporary storage */
  rworkspace = rgrid_clone(otf->density, "rworkspace"); /* temporary storage */

  /* Get background energy */
  cgrid_constant(gwf->grid, SQRT(rho0));
  dft_ot_energy_density(otf, rworkspace, gwf);
  e0 = rgrid_integral_region(rworkspace, LX, UX, LY, UY, LZ, UZ);
  printf("e0 = " FMT_R " K\n", e0 * GRID_AUTOK);
 
  /* setup initial guess for vortex ring */
  grid_wf_map(gwf, vring, NULL);

  for (iter = 1; iter < ITERS; iter++) {

    if(iter == 5) grid_fft_write_wisdom(NULL);

    grid_timer_start(&timer);

    /* Predict-Correct */
    cgrid_zero(potential_store);
    dft_ot_potential(otf, potential_store, gwf);
    cgrid_add(potential_store, -mu0);
    grid_wf_propagate_predict(gwf, gwfp, potential_store, (TS - I * ITS) / GRID_AUTOFS);
    dft_ot_potential(otf, potential_store, gwfp);
    cgrid_add(potential_store, -mu0);
    cgrid_multiply(potential_store, 0.5);  // Use (current + future) / 2
    grid_wf_propagate_correct(gwf, potential_store, (TS  - I * ITS) / GRID_AUTOFS);
    // Chemical potential included - no need to normalize

    printf("Iteration " FMT_I " - Wall clock time = " FMT_R " seconds.\n", iter, grid_timer_wall_clock_time(&timer));
    fflush(stdout);
  }
  sprintf(buf, "vring-" FMT_I, iter);
  grid_wf_density(gwf, rworkspace);
  rgrid_write_grid(buf, rworkspace);
  cgrid_abs_power(gwf->grid, gwf->grid, 1.0); // Kill the phase to get just the core energy
  kin = mod_grid_wf_kinetic_energy_cn(gwf);
  grid_wf_density(gwf, rworkspace);
  dft_ot_energy_density(otf, rworkspace, gwf);
  pot = rgrid_integral_region(rworkspace, LX, UX, LY, UY, LZ, UZ);
  printf("pot = " FMT_R " K\n", pot * GRID_AUTOK);
  pot = pot - e0;  // remove uniform bulk energy
  n = grid_wf_norm(gwf);
  printf("Iteration " FMT_I " natoms          = " FMT_R " particles.\n", iter, n);   /* Energy / particle in K */
  printf("Iteration " FMT_I " kinetic         = " FMT_R " K\n", iter, kin * GRID_AUTOK);  /* Print result in K */
  printf("Iteration " FMT_I " potential       = " FMT_R " K\n", iter, pot * GRID_AUTOK);  /* Print result in K */
  printf("Iteration " FMT_I " total energy    = " FMT_R " K\n", iter, (kin + pot) * GRID_AUTOK);  /* Print result in K */
#ifdef RING_RADIUS
  printf("Iteration " FMT_I " core energy     = " FMT_R " K/Ang\n", iter, (kin + pot) * GRID_AUTOK / (GRID_AUTOANG * 2.0 * M_PI * RING_RADIUS));  /* Print result in K */
#else
  printf("Iteration " FMT_I " core energy     = " FMT_R " K/Ang\n", iter, (kin + pot) * GRID_AUTOK / (GRID_AUTOANG * (UZ - LZ)));  /* Print result in K */
#endif

  return 0;
}
