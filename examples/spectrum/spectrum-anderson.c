/*
 * Compute absorption spectrum of solvated impurity (Cu2)
 * in bulk superfluid helium. Classical impurity (no zero-point).
 *
 * This uses the Andersson expression, which does not require dynamics
 * to be run on the final state.
 *
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <grid/grid.h>
#include <grid/au.h>
#include <dft/dft.h>
#include <dft/ot.h>

/* Density functional to use */
#define MODEL (DFT_OT_PLAIN | DFT_OT_HD)

/* Time step */
#define TS 10.0 /* fs */

/* Grid parameters */
#define NX 128
#define NY 128
#define NZ 128
#define STEP 1.0

/* External pressure */
#define PRESSURE 0.0

/* Number of openmp threads to use (0 = all) */
#define THREADS 6

/* Number of imaginary time iterations (solvation) */
#define IMITER 200

/* Number of points for the time correlation function (uses TS as the time step) */
#define TCITER 2048

/* Ground state potentials (isotropic) */
#define GND_X "potentials/2s-exp.dat"
#define GND_Y "potentials/2s-exp.dat"
#define GND_Z "potentials/2s-exp.dat"

/* Excited state potentials (isotropic) */
#define EXC_X "potentials/2p-exp.dat"
#define EXC_Y "potentials/2p-exp.dat"
#define EXC_Z "potentials/2p-exp.dat"

int main(int argc, char **argv) {

  dft_ot_functional *otf;
  cgrid *potential_store, *spectrum;
  rgrid *ext_pot;
  wf *gwf, *gwfp;
  INT i;
  REAL mu0, rho0, en;
  FILE *fp;

  grid_timer timer;

#ifdef USE_CUDA
#define NGPUS 1
int gpus[] = {0};
  cuda_enable(1, NGPUS, gpus);
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
  potential_store = cgrid_clone(gwf->grid, "potential_store (temp)");

  /* Read the ground state potential from file */
  dft_common_potential_map(0, GND_X, GND_Y, GND_Z, ext_pot);

  /* Run in imaginary time to obtain the solvation structure */
  grid_wf_constant(gwf, SQRT(rho0)); /* Initial guess (uniform bulk) */
  for (i = 0; i < IMITER; i++) {

    if(i == 20) grid_fft_write_wisdom(NULL); /* Save FFTW wisdom after 20 iterations */

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

    printf("Iteration " FMT_I " - Wall clock time = " FMT_R " seconds.\n", i, grid_timer_wall_clock_time(&timer));
  }
  /* At this point gwf contains the converged wavefunction */
  cgrid_write_grid("output", gwf->grid);
    
  /* Read the excited state potential from file */
  dft_common_potential_map(0, EXC_X, EXC_Y, EXC_Z, otf->density);

  /* Compute the difference potential */
  rgrid_difference(ext_pot, otf->density, ext_pot); /* Final - Initial */
 
  grid_wf_density(gwf, otf->density);

  /* Allocate 1-D complex grid for the spectrum */
  spectrum = cgrid_alloc(1, 1, TCITER, TS / GRID_AUTOFS, CGRID_PERIODIC_BOUNDARY, 0, "Spectrum");

  dft_spectrum_anderson(otf->density, ext_pot, spectrum, potential_store);

  if(!(fp = fopen("spectrum.dat", "w"))) {
    fprintf(stderr, "Can't open spectrum.dat for writing.\n");
    exit(1);
  }
  for (i = 0, en = -0.5 * spectrum->step * (REAL) spectrum->nz; i < spectrum->nz; i++, en += spectrum->step)
    fprintf(fp, FMT_R " " FMT_R " " FMT_R "\n", en, CREAL(cgrid_value_at_index(spectrum, 1, 1, i)), CIMAG(cgrid_value_at_index(spectrum, 1, 1, i)));
  fclose(fp);
  printf("Spectrum written to spectrum.dat\n");
  exit(0);
}
