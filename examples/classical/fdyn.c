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

//#undef USE_CUDA
#ifdef USE_CUDA
#define NGPUS 2
int gpus[] = {0, 1};
#endif

#define NX 256
#define NY 256
#define NZ 2048
#define STEP 200.0
#define TS (10.0 / GRID_AUTOFS)

/* Propagator */
#define PROPAGATOR WF_2ND_ORDER_FFT

/* Imag time for FFT stab */
#define FFT_STAB 0.005

/* CFFT constant */
#define CFFT_CONST 2.0

/* Initial imaginary iterations */
#define IITER 500

/* Mass of water molecule */
#define MASS (18.02 / GRID_AUTOAMU)

/* Moving background (no rounding) */
#define KX	(0.0 * 2.0 * M_PI / (NX * STEP))
#define KY	(0.0 * 2.0 * M_PI / (NY * STEP))
#define KZ	(6000.0 * 2.0 * M_PI / (NZ * STEP))
#define VX	(KX * HBAR / DFT_HELIUM_MASS)
#define VY	(KY * HBAR / DFT_HELIUM_MASS)
#define VZ	(KZ * HBAR / DFT_HELIUM_MASS)

/* Maximum local velocity that can be represented with the current grid (max phase change between adjacent grid points is 2pi) */
/* But this is based on FD - would FFT suffer from the same thing ? */
#define MAXVELOC (GRID_AUTOMPS * (HBAR / MASS) * 2.0 * M_PI / STEP)

/* Bulk density: 1000 kg/m3 -> per particle and in au */
#define RHO0 (((1000.0 / GRID_AUTOKG) / MASS) * GRID_AUTOM * GRID_AUTOM * GRID_AUTOM)

/* Shear viscosity */
#define VISC (1.0E-3 / GRID_AUTOPAS)

/* Temperature (K) */
#define TEMP 3.0

/* Tait K0 */
#define TK0 (3E8 / GRID_AUTOPA)

/* Tait n */
#define Tn 7.0

/* Spherical cavity parameters using exponential repulsion (approx. electron bubble) - RADD = 19.0 */
#define A0 (3.8003E5 / GRID_AUTOK)
#define A1 (1.6245 * GRID_AUTOANG)
#define A2 0.0
#define A3 0.0
#define A4 0.0
#define A5 0.0
#define RMIN 5.0
#define RADD_INI 1900.0
REAL RADD = RADD_INI;

/* Maximum number of iterations */
#define MAXITER 10000000

/* Output at every NTH iteration */
#define NTH 2000

/* Number of CPU threads to use (0 = all) */
#define THREADS 0

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
  rgrid *ext_pot, *density, *rpot, *wrk1, *wrk2, *wrk3, *wrk4, *wrk5, *wrk6, *wrk7, *wrk8, *wrk9;
  rfunction *rvisc;
  wf *gwf;
  INT iter;
  REAL nwater, visc_params[3] = {RHO0, VISC, 0.0};
  grid_timer timer;

#ifdef USE_CUDA
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
  gwf->cfft_width = CFFT_CONST;

  if(!(density = rgrid_alloc(NX, NY, NZ, STEP, RGRID_PERIODIC_BOUNDARY, NULL, "density"))) {
    fprintf(stderr, "Cannot allocate gwf.\n");
    exit(1);
  }

  if(!(rvisc = rgrid_function_alloc(dft_classical_viscosity, visc_params, 0.0, 1.5 * RHO0, 0.001 * RHO0, "Viscosity function"))) {
    fprintf(stderr, "Cannot allocate EOS function.\n");
    exit(1);
  }

  /* Allocate space for external potential */
  ext_pot = rgrid_clone(density, "ext_pot");
  rpot = rgrid_clone(density, "rpot");
  wrk1 = rgrid_clone(density, "wrk1");
  wrk2 = rgrid_clone(density, "wrk2");
  wrk3 = rgrid_clone(density, "wrk3");
  wrk4 = rgrid_clone(density, "wrk4");
  wrk5 = rgrid_clone(density, "wrk5");
  wrk6 = rgrid_clone(density, "wrk6");
  wrk7 = rgrid_clone(density, "wrk7");
  wrk8 = rgrid_clone(density, "wrk8");
  wrk9 = rgrid_clone(density, "wrk9");

  potential_store = cgrid_clone(gwf->grid, "potential_store");

  printf("Maximum velocity = " FMT_R " m/s.\n", MAXVELOC);

  /* Read external potential from file */
  rgrid_map(ext_pot, pot_func, NULL);

  grid_wf_constant(gwf, SQRT(RHO0));

  printf("Background velocity = " FMT_R " m/s\n", VZ * GRID_AUTOMPS);

  cgrid_set_momentum(gwf->grid, KX, KY, KZ);      

  for (iter = 0; iter < MAXITER; iter++) {

    if(iter == 5) grid_fft_write_wisdom(NULL);

    grid_timer_start(&timer);

    /* Construct non-linear potential */
    grid_wf_density(gwf, density);
    dft_classical_ideal_gas(rpot, density, RHO0, TEMP, 1E-3);
//    dft_classical_tait(rpot, density, RHO0, TK0, Tn, wrk1);
//    dft_classical_add_viscous_potential(gwf, rpot, rvisc, wrk1, wrk2, wrk3, wrk4, wrk5, wrk6, wrk7, wrk8, wrk9, density); // density = wrk8
    rgrid_sum(rpot, rpot, ext_pot);    
//    rgrid_threshold_clear(rpot, rpot, 500.0 / GRID_AUTOK, -500.0 / GRID_AUTOK, 500.0 / GRID_AUTOK, -500.0 / GRID_AUTOK);

    grid_real_to_complex_re(potential_store, rpot);

    if(iter < IITER)
      grid_wf_propagate(gwf, potential_store, -I * TS);
    else {
//      RADD = 1.5 * RADD_INI;
      grid_wf_propagate(gwf, potential_store, TS - I * TS * FFT_STAB);
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
