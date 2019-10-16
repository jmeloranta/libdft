/*
 * Bubble in water (no viscosity). Tait's equation of state (from wikipedia).
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

#define NX 256
#define NY 256
#define NZ 256
#define STEP 20.0
#define TS (100.0 / GRID_AUTOFS)

/* Propagator */
#define PROPAGATOR WF_2ND_ORDER_FFT

/* Mass of water molecule */
#define MASS (18.02 / GRID_AUTOAMU)

/* Moving background (600.0E10 is about speed of sound) */
#define KX	(0.0 * 2.0 * M_PI / (NX * STEP))
#define KY	(0.0 * 2.0 * M_PI / (NY * STEP))
#define KZ	(1200.0 * 2.0 * M_PI / (NZ * STEP))
#define VX	(KX * HBAR / MASS)
#define VY	(KY * HBAR / MASS)
#define VZ	(KZ * HBAR / MASS)

/* Maximum local velocity that can be represented with the current grid (max phase change between adjacent grid points is 2pi) */
/* But this is based on FD - would FFT suffer from the same thing ? */
#define MAXVELOC (GRID_AUTOMPS * (HBAR / MASS) * 2.0 * M_PI / STEP)

/* Bulk density: 1000 kg/m3 -> per particle and in au */
#define RHO0 (((1000.0 / GRID_AUTOKG) / MASS) * GRID_AUTOM * GRID_AUTOM * GRID_AUTOM)

/* Shear viscosity */
#define VISC (0.89E-3 / GRID_AUTOPAS)

/* Temperature (K) -- does not affect anything here, Tait is fixed to 300 K */
#define TEMP 300.0

/* Atmospheric pressure */
#define P0 (101325.0 / GRID_AUTOPA)

/* Tait K0 */
#define TK0 2.15

/* Tait n */
#define Tn 7.15

/* Spherical cavity parameters using exponential repulsion (approx. electron bubble) - RADD = 19.0 */
#define A0 (3.8003E5 / GRID_AUTOK)
#define A1 (1.6245 * GRID_AUTOANG)
#define A2 0.0
#define A3 0.0
#define A4 0.0
#define A5 0.0
#define RMIN 6.0
#define RADD_INI 300.0
REAL RADD = RADD_INI;

/* Maximum number of iterations */
#define MAXITER 10000000

/* Output at every NTH iteration */
#define NTH 50

/* Number of CPU threads to use (0 = all) */
#define THREADS 0

REAL eos(REAL rho, void *params) {  // Tait works only at RT?

  REAL rho0 = ((REAL *) params)[0];
//  REAL temp = ((REAL *) params)[1];

  return (TK0 / Tn) * (POW(rho / rho0, Tn) - 1.0) / GRID_AUTOPA + P0;
}

REAL pot_func(void *NA, REAL x, REAL y, REAL z) {

  REAL r, r2, r4, r6, r8, r10;

  r = SQRT(x * x + y * y + z * z);
  r -= RADD;
#ifdef RMIN
  if(r < RMIN) r = RMIN;
#endif

  r2 = r * r;
  r4 = r2 * r2;
  r6 = r4 * r2;
  r8 = r6 * r2;
  r10 = r8 * r2;
  return A0 * EXP(-A1 * r) 
#ifdef RMIN
   - A2 / r4 - A3 / r6 - A4 / r8 - A5 / r10
#endif
  ;
}

int main(int argc, char **argv) {

  cgrid *potential_store;
  rgrid *ext_pot, *density, *wrk1, *wrk2, *wrk3, *wrk4, *wrk5, *wrk6, *wrk7;
  wf *gwf;
  INT iter;
  REAL nwater, eos_params[2] = {RHO0, TEMP};
  struct dft_classical_visc_func_param visc_params = {RHO0, VISC, 0.0};
  grid_timer timer;

#undef USE_CUDA
#ifdef USE_CUDA
#define NGPUS 1
int gpus[] = {6};
  cuda_enable(1, NGPUS, gpus);
#endif

  /* Initialize threads & use wisdom */
  grid_set_fftw_flags(1);    // FFTW_MEASURE
  grid_threads_init(THREADS);
  grid_fft_read_wisdom(NULL);

  /* Allocate wave function */
  if(!(gwf = grid_wf_alloc(NX, NY, NZ, STEP, MASS, WF_PERIODIC_BOUNDARY, PROPAGATOR, "gwf"))) {
    fprintf(stderr, "Cannot allocate gwf.\n");
    exit(1);
  }

  if(!(density = rgrid_alloc(NX, NY, NZ, STEP, RGRID_PERIODIC_BOUNDARY, NULL, "density"))) {
    fprintf(stderr, "Cannot allocate gwf.\n");
    exit(1);
  }

  /* Allocate space for external potential */
  ext_pot = rgrid_clone(density, "ext_pot");
  wrk1 = rgrid_clone(density, "wrk1");
  wrk2 = rgrid_clone(density, "wrk2");
  wrk3 = rgrid_clone(density, "wrk3");
  wrk4 = rgrid_clone(density, "wrk4");
  wrk5 = rgrid_clone(density, "wrk5");
  wrk6 = rgrid_clone(density, "wrk6");
  wrk7 = rgrid_clone(density, "wrk7");
  potential_store = cgrid_clone(gwf->grid, "potential_store");

  printf("Maximum velocity = " FMT_R " m/s.\n", MAXVELOC);

  /* Read external potential from file */
  rgrid_map(ext_pot, pot_func, NULL);

  grid_wf_constant(gwf, SQRT(RHO0));

  printf("Background velocity = " FMT_R " m/s\n", VZ * GRID_AUTOMPS);

  for (iter = 0; iter < MAXITER; iter++) {

    if(iter == 5) grid_fft_write_wisdom(NULL);

    grid_timer_start(&timer);

    /* Construct non-linear potential */
    cgrid_zero(potential_store);
    dft_classical_add_eos_potential(gwf, potential_store, eos, (void *) eos_params, wrk1, wrk2, wrk3, wrk4);
    dft_classical_add_viscous_potential(gwf, potential_store, dft_classical_visc_func, (void *) &visc_params, wrk1, wrk2, wrk3, wrk4, wrk5, wrk6, wrk7, density); // wrk8 = density
    cgrid_fft(potential_store);
    cgrid_poisson(potential_store);
    cgrid_inverse_fft(potential_store);
    grid_add_real_to_complex_re(potential_store, ext_pot);

    if(iter < 100)
      grid_wf_propagate(gwf, potential_store, -I * TS);
    else {
//      RADD = 1.5 * RADD_INI;
      cgrid_set_momentum(gwf->grid, KX, KY, KZ);      
      grid_wf_propagate(gwf, potential_store, TS);
    }
    printf("Iteration " FMT_I " - Wall clock time = " FMT_R " seconds.\n", iter, grid_timer_wall_clock_time(&timer));

    if(!(iter % NTH)) {
      char buf[512];
      sprintf(buf, "output-" FMT_I, iter);
      cgrid_write_grid(buf, gwf->grid);
      nwater = grid_wf_norm(gwf);
      printf("Number of water molecules is " FMT_R ".\n", nwater);
      fflush(stdout);
    }
  }

  return 0;
}
