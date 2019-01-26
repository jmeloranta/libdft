/*
 * Impurity atom in superfluid helium (no zero-point).
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

#define NX 128
#define NY 128
#define NZ 128
#define STEP 0.5
#define TS 1.0E-3

#define PRESSURE 0.0

#define MAXITER 10000000
#define NTH 1000

#define THREADS 0

#define HELIUM_MASS (4.002602 / GRID_AUTOAMU)

/* Ion */
// #define EXP_P
#define K_P

#ifdef ZERO_P
#define A0 0.0
#define A1 0.0
#define A2 0.0
#define A3 0.0
#define A4 0.0
#define A5 0.0
#define RMIN 0.0
#define RADD 0.0
#endif

/* exponential repulsion (approx. electron bubble) - RADD = -19.0 */
#ifdef EXP_P
#define A0 (3.8003E5 / GRID_AUTOK)
#define A1 (1.6245 * GRID_AUTOANG)
#define A2 0.0
#define A3 0.0
#define A4 0.0
#define A5 0.0
#define RMIN 2.0
#define RADD (-6.0)
#endif

/* Ca+ */
#ifdef CA_P
#define A0 4.83692
#define A1 1.23684
#define A2 0.273202
#define A3 59.5463
#define A4 1134.51
#define A5 0.0
#define RMIN 5.0
#define RADD 0.0
#endif

/* K+ */
#ifdef K_P
#define A0 140.757
#define A1 2.26202
#define A2 0.722065
#define A3 0.00144039
#define A4 356.303
#define A5 1358.98
#define RMIN 4.0
#define RADD 0.0
#endif

/* Be+ */
#ifdef BE_P
#define A0 4.73292
#define A1 1.53925
#define A2 0.557845
#define A3 26.7013
#define A4 0.0
#define A5 0.0
#define RMIN 3.4
#define RADD 0.0
#endif

/* Sr+ */
#ifdef SR_P
#define A0 3.64975
#define A1 1.13451
#define A2 0.293483
#define A3 99.0206
#define A4 693.904
#define A5 0.0
#define RMIN 5.0
#define RADD 0.0
#endif

/* Cl- */
#ifdef CL_M
#define A0 11.1909
#define A1 1.50971
#define A2 0.72186
#define A3 17.2434
#define A4 0.0
#define A5 0.0
#define RMIN 4.2
#define RADD 0.0
#endif

/* F- */
#ifdef F_M
#define A0 5.16101
#define A1 1.62798
#define A2 0.773982
#define A3 1.09722
#define A4 0.0
#define A5 0.0
#define RMIN 4.1
#define RADD 0.0
#endif

/* I- */
#ifdef I_M
#define A0 13.6874
#define A1 1.38037
#define A2 0.696409
#define A3 37.3331 
#define A4 0.0
#define A5 0.0
#define RMIN 4.1
#define RADD 0.0
#endif

/* Br- */
#ifdef BR_M
#define A0 12.5686
#define A1 1.45686
#define A2 0.714525
#define A3 24.114
#define A4 0.0
#define A5 0.0
#define RMIN 5.0
#define RADD 0.0
#endif

REAL pot_func(void *asd, REAL x, REAL y, REAL z) {

  REAL r, r2, r4, r6, r8, r10, tmp, *asdf;

  if(asd) {
    asdf = asd;
    x -= *asdf;
  }
  r = SQRT(x * x + y * y + z * z);
  if(r < RMIN) r = RMIN;
  r += RADD;

  r2 = r * r;
  r4 = r2 * r2;
  r6 = r4 * r2;
  r8 = r6 * r2;
  r10 = r8 * r2;
  tmp = A0 * EXP(-A1 * r) - A2 / r4 - A3 / r6 - A4 / r8 - A5 / r10;
  return tmp;
}

int main(int argc, char **argv) {

  dft_ot_functional *otf;
  cgrid *potential_store;
  rgrid *ext_pot, *density;
  wf *gwf, *gwfp;
  INT iter;
  REAL energy, natoms, mu0, rho0;
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
  ext_pot = rgrid_clone(otf->density, "ext_pot");
  potential_store = cgrid_clone(gwf->grid, "potential_store");
  density = rgrid_clone(otf->density, "density");

  /* Read external potential from file */
  rgrid_map(ext_pot, pot_func, NULL);
  //dft_common_potential_map(0, "cl-pot.dat", "cl-pot.dat", "cl-pot.dat", ext_pot);

  grid_wf_constant(gwf, SQRT(rho0));

  /* Run 200 iterations using imaginary time (10 fs time step) */
  for (iter = 0; iter < MAXITER; iter++) {

    if(iter == 5) grid_fft_write_wisdom(NULL);

    grid_timer_start(&timer);

    /* Predict-Correct */
    grid_real_to_complex_re(potential_store, ext_pot);
    dft_ot_potential(otf, potential_store, gwf);
    cgrid_add(potential_store, -mu0);
    grid_wf_propagate_predict(gwf, gwfp, potential_store, -I * TS / GRID_AUTOFS);
    grid_add_real_to_complex_re(potential_store, ext_pot);
    dft_ot_potential(otf, potential_store, gwfp);
    cgrid_add(potential_store, -mu0);
    cgrid_multiply(potential_store, 0.5);  // Use (current + future) / 2
    grid_wf_propagate_correct(gwf, potential_store, -I * TS / GRID_AUTOFS);
    // Chemical potential included - no need to normalize

    printf("Iteration " FMT_I " - Wall clock time = " FMT_R " seconds.\n", iter, grid_timer_wall_clock_time(&timer));

    if(!(iter % NTH)) {
      char buf[512];
      sprintf(buf, "output-" FMT_I, iter);
      grid_wf_density(gwf, density);
      rgrid_write_grid(buf, density);
      sprintf(buf, "wf-output-" FMT_I, iter);
      cgrid_write_grid(buf, gwf->grid);
    }
    dft_ot_energy_density(otf, density, gwf);
    rgrid_add_scaled_product(density, 1.0, otf->density, ext_pot);
    energy = grid_wf_energy(gwf, NULL) + rgrid_integral(density);
    natoms = grid_wf_norm(gwf);
    printf("Total energy is " FMT_R " K\n", energy * GRID_AUTOK);
    printf("Number of He atoms is " FMT_R ".\n", natoms);
    printf("Energy / atom is " FMT_R " K\n", (energy/natoms) * GRID_AUTOK);
    fflush(stdout);
  }
  /* At this point gwf contains the converged wavefunction */
  grid_wf_density(gwf, density);
  rgrid_write_grid("output", density);
  return 0;
}
