/*
 * Compute absorption spectrum of solvated impurity (Cu2)
 * in bulk superfluid helium. Classical impurity (no zero-point).
 *
 * This uses the polarization expression, which requires dynamics
 * to be run also on the final state.
 *
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <grid/grid.h>
#include <grid/au.h>
#include <dft/dft.h>
#include <dft/ot.h>

/* DFT model */
#define MODEL (DFT_OT_PLAIN | DFT_OT_HD)

/* Time step length */
#define TS 5.0 /* fs */

/* Grid parameters */
#define NX 128
#define NY 128
#define NZ 128
#define STEP 1.0

/* External pressure */
#define PRESSURE 0.0

/* Number of OpenMP threads to use (0 = all available cores) */
#define THREADS 6

/* FFT zero-fill for spectrum calculation */
#define ZEROFILL 1024

/* Dephasing constant for spectrum calculation (exponential decay of polarization) */
#define TC 150.0

/* Initial imaginary time iterations */
#define IMITER 200

/* Real time dynamics iterations */
#define REITER 400

/* Ground state potential */
#define GND_X "potentials/2s-exp.dat"
#define GND_Y "potentials/2s-exp.dat"
#define GND_Z "potentials/2s-exp.dat"

/* Excited state potential */
#define EXC_X "potentials/2p-exp.dat"
#define EXC_Y "potentials/2p-exp.dat"
#define EXC_Z "potentials/2p-exp.dat"

int main(int argc, char **argv) {

  dft_ot_functional *otf;
  cgrid *potential_store;
  rgrid *ext_pot, *density, *spectrum;
  wf *gwf, *gwfp;
  INT iter;
  REAL energy, natoms, en, mu0, rho0;
  FILE *fp;

  grid_timer timer;

#ifdef USE_CUDA
  cuda_enable(1);
#endif

  /* Initialize threads & use wisdom */
  grid_threads_init(THREADS);
  grid_fft_read_wisdom(NULL);

  /* Allocate wave functions */
  if(!(gwf = grid_wf_alloc(NX, NY, NZ, STEP, DFT_HELIUM_MASS, WF_PERIODIC_BOUNDARY, WF_2ND_ORDER_FFT, "gwf"))) {
    fprintf(stderr, "Cannot allocate gwf.\n");
    exit(1);
  }
  gwfp = grid_wf_clone(gwf, "gwfp");

  /* Allocate OT functional */
  if(!(otf = dft_ot_alloc(MODEL, gwf, DFT_MIN_SUBSTEPS, DFT_MAX_SUBSTEPS))) {
    fprintf(stderr, "Cannot allocate otf.\n");
    exit(1);
  }
  rho0 = dft_ot_bulk_density_pressurized(otf, PRESSURE);
  mu0 = dft_ot_bulk_chempot_pressurized(otf, PRESSURE);
  printf("mu0 = " FMT_R " K/atom, rho0 = " FMT_R " Angs^-3.\n", mu0 * GRID_AUTOK, rho0 / (GRID_AUTOANG * GRID_AUTOANG * GRID_AUTOANG));

  /* Allocate space for external potential */
  ext_pot = rgrid_clone(otf->density, "ext_pot");
  density = rgrid_clone(otf->density, "density");
  potential_store = cgrid_clone(gwf->grid, "potential_store"); /* temporary storage */

  /* Read ground state potential from file */
  dft_common_potential_map(0, GND_X, GND_Y, GND_Z, ext_pot);

  if(argc == 1) {
    /* Run imaginary time */
    grid_wf_constant(gwf, SQRT(rho0));
    for (iter = 0; iter < IMITER; iter++) {

      if(iter == 5) grid_fft_write_wisdom(NULL);

      grid_timer_start(&timer);

      /* Predict-Correct */
      grid_real_to_complex_re(potential_store, ext_pot);
      dft_ot_potential(otf, potential_store, gwf);
      cgrid_add(potential_store, -mu0);
      grid_wf_propagate_predict(gwf, gwfp, potential_store, -4.0 * I * TS / GRID_AUTOFS);
      grid_add_real_to_complex_re(potential_store, ext_pot);
      dft_ot_potential(otf, potential_store, gwfp);
      cgrid_add(potential_store, -mu0);
      cgrid_multiply(potential_store, 0.5);  // Use (current + future) / 2
      grid_wf_propagate_correct(gwf, potential_store, -4.0 * I * TS / GRID_AUTOFS);
      // Chemical potential included - no need to normalize

      printf("Iteration " FMT_I " - Wall clock time = " FMT_R " seconds.\n", iter, grid_timer_wall_clock_time(&timer));
    }
    /* At this point gwf contains the converged wavefunction */
    cgrid_write_grid("output", gwf->grid);
  } else cgrid_read_grid(gwf->grid, argv[1]);
    
  dft_ot_energy_density(otf, density, gwf);
  rgrid_add_scaled_product(density, 1.0, otf->density, ext_pot);
  energy = grid_wf_energy(gwf, NULL) + rgrid_integral(density);
  natoms = grid_wf_norm(gwf);
  printf("Total energy is " FMT_R " K\n", energy * GRID_AUTOK);
  printf("Number of He atoms is " FMT_R ".\n", natoms);
  printf("Energy / atom is " FMT_R " K\n", (energy/natoms) * GRID_AUTOK);

  /* Read excited state potential from file */
  dft_common_potential_map(0, EXC_X, EXC_Y, EXC_Z, otf->density);

  /* Difference potential for dynamics and spectrum calculation */
  rgrid_difference(ext_pot, otf->density, ext_pot);

  /* Allocate 1-D complex grid for the spectrum */
  spectrum = rgrid_alloc(1, 1, 2048, TS / GRID_AUTOFS, RGRID_PERIODIC_BOUNDARY, 0, "Spectrum");
  spectrum->step = 0.5; /* This is in cm-1!!! */

  for (iter = 0; iter < REITER; iter++) {
    grid_timer_start(&timer);

    /* Predict-Correct */
    grid_real_to_complex_re(potential_store, ext_pot);
    dft_ot_potential(otf, potential_store, gwf);
    cgrid_add(potential_store, -mu0);
    grid_wf_propagate_predict(gwf, gwfp, potential_store, TS / GRID_AUTOFS);
    grid_add_real_to_complex_re(potential_store, ext_pot);
    dft_ot_potential(otf, potential_store, gwfp);
    cgrid_add(potential_store, -mu0);
    cgrid_multiply(potential_store, 0.5);  // Use (current + future) / 2
    grid_wf_propagate_correct(gwf, potential_store, TS / GRID_AUTOFS);

    printf("Iteration " FMT_I " - Wall clock time = " FMT_R " seconds.\n", iter, grid_timer_wall_clock_time(&timer));

    /* Collect energy difference values */ 
    dft_spectrum_bin_collect(gwf, ext_pot, spectrum, iter, TS / GRID_AUTOFS, TC / GRID_AUTOFS, otf->density);

    if(!(iter % 10)) {
      char buf[512];
      grid_wf_density(gwf, density);
      sprintf(buf, "realtime-" FMT_I, iter);
      rgrid_write_grid(buf, density);
    }
  }

  if(!(fp = fopen("spectrum.dat", "w"))) {
    fprintf(stderr, "Can't open spectrum.dat for writing.\n");
    exit(1);
  }
  for (iter = 0, en = -0.5 * spectrum->step * (REAL) spectrum->nz; iter < spectrum->nz; iter++, en += spectrum->step)
    fprintf(fp, FMT_R " " FMT_R "\n", en, rgrid_value_at_index(spectrum, 1, 1, iter));
  fclose(fp);
  printf("Spectrum written to spectrum.dat\n");
  exit(0);
}
