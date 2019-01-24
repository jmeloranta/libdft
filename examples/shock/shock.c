/*
 * Shock wave propagation in superfluid helium.
 *
 * All input in a.u. except the time step, which is fs.
 *
 * The initial condition for the shock is given by:
 *
 * \psi(z, 0) = \sqrt(\rho_0) if |z| > w
 * or
 * \psi(z, 0) = \sqrt(\rho_0 + \Delta)\exp(-(v_z/m_He)(z + w) / \hbar)
 *
 * where \Delta is the shock amplitude and v_z is the shock velocity
 * (discontinuity in both density and velocity)
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <grid/grid.h>
#include <grid/au.h>
#include <dft/dft.h>
#include <dft/ot.h>

#define MAXITER 160000
#define TS 10.0 /* fs */
#define OUTPUT 100

#define THREADS 0

#define DELTA (0.05 * rho0)
#define W 30.0
#define VZ (230.0 / GRID_AUTOMPS)
#define KZ (HELIUM_MASS * VZ / HBAR)

#define PRESSURE (0.0 / GRID_AUTOBAR)   /* External pressure in bar */
#define HELIUM_MASS (4.002602 / GRID_AUTOAMU)

#define FUNC (DFT_OT_PLAIN)
//#define FUNC (DFT_OT_PLAIN | DFT_OT_KC)
#define NX 64
#define NY 64
#define NZ 1024
#define STEP 1.0

struct params {
  REAL delta;
  REAL rho0;
  REAL w;
  REAL vz;
};

REAL complex gauss(void *arg, REAL x, REAL y, REAL z) {

  REAL delta = ((struct params *) arg)->delta;
  REAL rho0 = ((struct params *) arg)->rho0;
  REAL w = ((struct params *) arg)->w;
//  REAL vz = ((struct params *) arg)->vz;

//  if(FABS(z) < w) return SQRT(rho0 + delta) * CEXP(I * (vz / HELIUM_MASS) * (z + w) / HBAR);
  if(FABS(z) < w) return SQRT(rho0 + delta);
  else return SQRT(rho0);
}

int main(int argc, char **argv) {

  dft_ot_functional *otf;
  struct params sparams;
  rgrid *rworkspace;
  cgrid *potential_store;
  wf *gwf, *gwfp;
  INT iter;
  REAL rho0, mu0;
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
  if(!(gwf = grid_wf_alloc(NX, NY, NZ, STEP, DFT_HELIUM_MASS, WF_PERIODIC_BOUNDARY, WF_2ND_ORDER_FFT, "gwf"))) {
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
  rworkspace = rgrid_clone(otf->density, "rworkspace");
  potential_store = cgrid_clone(gwf->grid, "cworkspace"); /* temporary storage */

  sparams.delta = DELTA;
  sparams.rho0 = rho0;
  sparams.w = W;
  sparams.vz = VZ;
  cgrid_map(gwf->grid, gauss, (void *) &sparams);  
  
  for (iter = 0; iter < MAXITER; iter++) {

    if(iter == 5) grid_fft_write_wisdom(NULL);

    grid_timer_start(&timer);

    /* Predict-Correct */
    cgrid_copy(gwfp->grid, gwf->grid);
    cgrid_zero(potential_store);
    dft_ot_potential(otf, potential_store, gwf);
    cgrid_add(potential_store, -mu0);
    grid_wf_propagate_predict(gwfp, potential_store, TS / GRID_AUTOFS);
    dft_ot_potential(otf, potential_store, gwfp);
    cgrid_add(potential_store, -mu0);
    cgrid_multiply(potential_store, 0.5);  // Use (current + future) / 2
    grid_wf_propagate_correct(gwf, potential_store, TS / GRID_AUTOFS);
    // Chemical potential included - no need to normalize

    printf("Iteration " FMT_I " - Wall clock time = " FMT_R " seconds.\n", iter, grid_timer_wall_clock_time(&timer));

    if(!(iter % OUTPUT)) {
      sprintf(buf, "final-" FMT_I, iter);
      grid_wf_density(gwf, rworkspace);
      rgrid_write_grid(buf, rworkspace);
    }
  }
  return 0;
}
