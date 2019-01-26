/*
 * Impurity atom in superfluid helium (with zero-point).
 * Includes zero-point for impurity.
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

#define NX 128
#define NY 128
#define NZ 128
#define STEP 1.0

#define THREADS 0

#define PRESSURE 0.0

#define ZEROFILL 1024

#define IMITER 200
#define REITER 400

/* use 2p-exp.dat for spherical average and 2p{x,y,z}-exp.dat for non-spherical */
#define UPPER_X "potentials/2px-exp.dat"
#define UPPER_Y "potentials/2py-exp.dat"
#define UPPER_Z "potentials/2pz-exp.dat"

#define LOWER_X "potentials/2s-exp.dat"
#define LOWER_Y "potentials/2s-exp.dat"
#define LOWER_Z "potentials/2s-exp.dat"

#define MODEL DFT_OT_T3000MK
#define RHO0 (0.0218360 * GRID_AUTOANG * GRID_AUTOANG * GRID_AUTOANG)
#define TC 150.0

#define IMP_MASS (4.002602 / GRID_AUTOAMU) /* He^* */

int main(int argc, char **argv) {

  dft_ot_functional *otf;
  cgrid *potential_store;
  cgrid *spectrum;
  rgrid *ext_pot, *ext_pot2, *density;
  wf *gwf, *gwfp;
  wf *imwf;
  INT iter;
  REAL energy, natoms, en, mu0, inv_width, rho0;
  FILE *fp;
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

  inv_width = 1.0 / 2.0;
  grid_wf_map(imwf, dft_common_cgaussian, &inv_width);

  /* Allocate space for external potential */
  ext_pot = rgrid_clone(otf->density, "ext_pot");
  ext_pot2 = rgrid_clone(otf->density, "ext_pot2");
  density = rgrid_clone(otf->density, "density");
  potential_store = cgrid_clone(gwf->grid, "potential_store"); /* temporary storage */

  /* Read external potential from file */
  dft_common_potential_map(0, LOWER_X, LOWER_Y, LOWER_Z, ext_pot);
  rgrid_fft(ext_pot);

  grid_wf_constant(gwf, SQRT(rho0));

  /* Run imaginary time iterations */
  for (iter = 0; iter < IMITER; iter++) {

    if(iter == 5) grid_fft_write_wisdom(NULL);

    grid_timer_start(&timer);

    /* convolute impurity density with ext_pot -> ext_pot2 */
    grid_wf_density(imwf, density);
    rgrid_fft(density);
    rgrid_fft_convolute(ext_pot2, ext_pot, density);
    rgrid_inverse_fft(ext_pot2);
    rgrid_add(ext_pot2, -mu0);

    /* Predict-Correct */
    grid_real_to_complex_re(potential_store, ext_pot2);
    dft_ot_potential(otf, potential_store, gwf);
    cgrid_add(potential_store, -mu0);
    grid_wf_propagate_predict(gwf, gwfp, potential_store, -I * TS / GRID_AUTOFS);
    grid_add_real_to_complex_re(potential_store, ext_pot);
    dft_ot_potential(otf, potential_store, gwfp);
    cgrid_add(potential_store, -mu0);
    cgrid_multiply(potential_store, 0.5);  // Use (current + future) / 2
    grid_wf_propagate_correct(gwf, potential_store, -I * TS / GRID_AUTOFS);
    // Chemical potential included - no need to normalize

    /* convolute liquid density with ext_pot -> ext_pot2 */
    grid_wf_density(gwf, density);
    rgrid_fft(density);
    rgrid_fft_convolute(ext_pot2, ext_pot, density);
    rgrid_inverse_fft(ext_pot2);

    /* Predict-Correct */
    grid_real_to_complex_re(potential_store, ext_pot2);
    grid_wf_propagate(imwf, potential_store, -I * TS / GRID_AUTOFS);

    printf("Iteration " FMT_I " - Wall clock time = " FMT_R " seconds.\n", iter, grid_timer_wall_clock_time(&timer));

  }

  /* At this point gwf contains the converged wavefunction */
  grid_wf_density(gwf, density);
  rgrid_write_grid("initial-helium", density);
  grid_wf_density(imwf, density);
  rgrid_write_grid("initial-imp", density);

  dft_ot_energy_density(otf, density, gwf);
  energy = grid_wf_energy(gwf, ext_pot) + rgrid_integral(density);
  natoms = grid_wf_norm(gwf);
  printf("Total energy of the liquid is " FMT_R " K\n", energy * GRID_AUTOK);
  printf("Number of He atoms is " FMT_R ".\n", natoms);
  printf("Energy / atom is " FMT_R " K\n", (energy/natoms) * GRID_AUTOK);

  /* Propagate only the liquid in real time */
  grid_wf_density(imwf, density);
  ext_pot = dft_spectrum_init(otf, density, REITER, ZEROFILL, 0, UPPER_X, UPPER_Y, UPPER_Z, 0, LOWER_X, LOWER_Y, LOWER_Z);
  rgrid_add(ext_pot, -mu0);
  for (iter = 0; iter < REITER; iter++) {

    /* Predict-Correct */
    grid_real_to_complex_re(potential_store, ext_pot2);
    dft_ot_potential(otf, potential_store, gwf);
    cgrid_add(potential_store, -mu0);
    grid_wf_propagate_predict(gwf, gwfp, potential_store, TS / GRID_AUTOFS);
    grid_add_real_to_complex_re(potential_store, ext_pot);
    dft_ot_potential(otf, potential_store, gwfp);
    cgrid_add(potential_store, -mu0);
    cgrid_multiply(potential_store, 0.5);  // Use (current + future) / 2
    grid_wf_propagate_correct(gwf, potential_store, TS / GRID_AUTOFS);
    // Chemical potential included - no need to normalize

    dft_spectrum_collect(otf, gwf);
    if(!(iter % 10)) {
      char buf[512];
      grid_wf_density(gwf, density);
      sprintf(buf, "realtime-" FMT_I, iter);
      rgrid_write_grid(buf, density);
    }
  }
  spectrum = dft_spectrum_evaluate(TS, TC);
  if(!(fp = fopen("spectrum.dat", "w"))) {
    fprintf(stderr, "Can't open spectrum.dat for writing.\n");
    exit(1);
  }
  for (iter = 0, en = -0.5 * spectrum->step * (REAL) (spectrum->nx - 1); iter < spectrum->nx; iter++, en += spectrum->step)
    // fprintf(fp, FMT_R " " FMT_R "\n", en, CREAL(cgrid_value_at_index(spectrum, 1, 1, iter)));
    fprintf(fp, FMT_R " " FMT_R "\n", en, POW(CREAL(cgrid_value_at_index(spectrum, 1, 1, iter)), 2.0) + POW(CIMAG(cgrid_value_at_index(spectrum, 1, 1, iter)), 2.0));
  fclose(fp);
  printf("Spectrum written to spectrum.dat\n");
  exit(0);
}
