/*
 * Impurity atom in superfluid helium (with zero-point).
 * Optimize the liquid structure around a given initial
 * potential and then run dynamics on a given final potential.
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

#define TS 80.0 /* fs */
#define NX 128
#define NY 128
#define NZ 128
#define STEP 2.0
#define THREADS 0

#define PRESSURE 0.0

/* He^* */
#define IMP_MASS (4.002602 / GRID_AUTOAMU)

#define INITIAL_POT_X "initial_pot.x"
#define INITIAL_POT_Y "initial_pot.y"
#define INITIAL_POT_Z "initial_pot.z"

#define FINAL_POT_X "final_pot.x"
#define FINAL_POT_Y "final_pot.y"
#define FINAL_POT_Z "final_pot.z"

int main(int argc, char **argv) {

  dft_ot_functional *otf;
  rgrid *ext_pot, *ext_pot2, *density;
  cgrid *potential_store;
  REAL width, rho0, mu0;
  wf *gwf, *gwfp;
  wf *imwf;
  INT iter;
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
  imwf = grid_wf_clone(gwf, "imwf");
  imwf->mass = IMP_MASS;

  /* Allocate OT functional */
  if(!(otf = dft_ot_alloc(DFT_OT_PLAIN | DFT_OT_BACKFLOW | DFT_OT_KC | DFT_OT_HD, gwf, DFT_MIN_SUBSTEPS, DFT_MAX_SUBSTEPS))) {
    fprintf(stderr, "Cannot allocate otf.\n");
    exit(1);
  }
  rho0 = dft_ot_bulk_density_pressurized(otf, PRESSURE);
  mu0 = dft_ot_bulk_chempot_pressurized(otf, PRESSURE);
  printf("mu0 = " FMT_R " K/atom, rho0 = " FMT_R " Angs^-3.\n", mu0 * GRID_AUTOK, rho0 / (GRID_AUTOANG * GRID_AUTOANG * GRID_AUTOANG));

  /* Initial impurity wf */
  width = 1.0 / 2.0; /* actually invese width */
  cgrid_map(imwf->grid, &dft_common_cgaussian, &width);

  /* Allocate space for external potential */
  ext_pot = rgrid_clone(otf->density, "ext_pot");
  ext_pot2 = rgrid_clone(otf->density, "ext_pot2");
  density = rgrid_clone(otf->density, "density");
  potential_store = cgrid_clone(gwf->grid, "potential_store"); /* temporary storage */

  /* Read external potential from file */
  dft_common_potential_map(0, INITIAL_POT_X, INITIAL_POT_Y, INITIAL_POT_Z, ext_pot);
  rgrid_fft(ext_pot);

  /* Step #1: Optimize structure */
  for (iter = 0; iter < 200; iter++) {

    if(iter == 5) grid_fft_write_wisdom(NULL);

    grid_timer_start(&timer);

    /* Helium: convolute impurity density with ext_pot -> ext_pot2 */
    grid_wf_density(imwf, density);
    rgrid_fft(density);
    rgrid_fft_convolute(ext_pot2, ext_pot, density);
    rgrid_inverse_fft(ext_pot2);
    /* Predict-Correct */
    cgrid_zero(potential_store);
    dft_ot_potential(otf, potential_store, gwf);
    grid_add_real_to_complex_re(potential_store, ext_pot2);
    cgrid_add(potential_store, -mu0);
    grid_wf_propagate_predict(gwf, gwfp, potential_store, -I * TS / GRID_AUTOFS);  // Imag time
    dft_ot_potential(otf, potential_store, gwfp);
    grid_add_real_to_complex_re(potential_store, ext_pot2);
    cgrid_add(potential_store, -mu0);
    cgrid_multiply(potential_store, 0.5);  // Use (current + future) / 2
    grid_wf_propagate_correct(gwf, potential_store, -I * TS / GRID_AUTOFS);   // Imag time
    // Chemical potential included - no need to normalize

    /* Impurity: convolute liquid density with ext_pot -> ext_pot2 */
    grid_wf_density(gwf, density);
    rgrid_fft(density);
    rgrid_fft_convolute(ext_pot2, ext_pot, density);
    rgrid_inverse_fft(ext_pot2);
    grid_real_to_complex_re(potential_store, ext_pot2);
    grid_wf_propagate(imwf, potential_store, -I * TS / GRID_AUTOFS);
    grid_wf_normalize(imwf);

    fprintf(stderr, "One iteration = " FMT_R " wall clock seconds.\n", grid_timer_wall_clock_time(&timer));
  }
  /* At this point gwf contains the converged wavefunction */
  cgrid_write_grid("initial1", gwf->grid);
  cgrid_write_grid("initial2", imwf->grid);

  /* Step #2: Propagate using the final state potential */
  dft_common_potential_map(0, FINAL_POT_X, FINAL_POT_Y, FINAL_POT_Z, ext_pot);
  rgrid_fft(ext_pot);

  for (iter = 0; iter < 200; iter++) {

    grid_timer_start(&timer);

    /* Helium: convolute impurity density with ext_pot -> ext_pot2 */
    grid_wf_density(imwf, density);
    rgrid_fft(density);
    rgrid_fft_convolute(ext_pot2, ext_pot, density);
    rgrid_inverse_fft(ext_pot2);
    /* Predict-Correct */
    cgrid_zero(potential_store);
    dft_ot_potential(otf, potential_store, gwf);
    grid_add_real_to_complex_re(potential_store, ext_pot2);
    cgrid_add(potential_store, -mu0);
    grid_wf_propagate_predict(gwf, gwfp, potential_store, -I * TS / GRID_AUTOFS);  // Imag time
    dft_ot_potential(otf, potential_store, gwfp);
    grid_add_real_to_complex_re(potential_store, ext_pot2);
    cgrid_add(potential_store, -mu0);
    cgrid_multiply(potential_store, 0.5);  // Use (current + future) / 2
    grid_wf_propagate_correct(gwf, potential_store, TS / GRID_AUTOFS);   // real time
    // Chemical potential included - no need to normalize

    /* Impurity: convolute liquid density with ext_pot -> ext_pot2 */
    grid_wf_density(gwf, density);
    rgrid_fft(density);
    rgrid_fft_convolute(ext_pot2, ext_pot, density);
    rgrid_inverse_fft(ext_pot2);
    grid_real_to_complex_re(potential_store, ext_pot2);
    grid_wf_propagate(imwf, potential_store, TS / GRID_AUTOFS);  // real time

    fprintf(stderr, "One iteration = " FMT_R " wall clock seconds.\n", grid_timer_wall_clock_time(&timer));

    if(!(iter % 10)) {
      sprintf(buf, "final1-" FMT_I, iter);
      cgrid_write_grid(buf, gwf->grid);
      sprintf(buf, "final2-" FMT_I, iter);
      cgrid_write_grid(buf, imwf->grid);
    }
  }
  return 0;
}
