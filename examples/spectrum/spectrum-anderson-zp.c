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
#define MODEL DFT_OT_PLAIN

/* Time step */
#define TS 10.0 /* fs */

/* Grid paramters */
#define NX 128
#define NY 128
#define NZ 128
#define STEP 1.0

/* Impurity mass */
#define IMP_MASS (2.0 / GRID_AUTOAMU)

/* External pressure */
#define PRESSURE 0.0

/* Number of openmp threads to use (0 = all) */
#define THREADS 6

/* Number of imaginary time iterations (solvation) */
#define IMITER 20

/* Number of points for the time correlation function (uses TS as the time step) */
#define TCITER 2048

/* Ground state potentials (isotropic) */
#define GND_X "potentials/2s-exp.dat"
#define GND_Y "potentials/2s-exp.dat"
#define GND_Z "potentials/2s-exp.dat"

/* Excited state potentials (isotropic) */
#define EXC_X "potentials/2px-exp.dat"
#define EXC_Y "potentials/2py-exp.dat"
#define EXC_Z "potentials/2pz-exp.dat"

int main(int argc, char **argv) {

  dft_ot_functional *otf;
  cgrid *potential_store, *wrk;
  cgrid *spectrum;
  rgrid *ext_pot, *ext_pot2;
  wf *gwf, *gwfp;
  wf *imwf;
  INT i;
  REAL en, mu0, inv_width, rho0;
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
  imwf = grid_wf_clone(gwf, "imwf");
  imwf->mass = IMP_MASS;
  imwf->norm = 1.0;

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
  ext_pot2 = rgrid_clone(otf->density, "ext_pot2");
  potential_store = cgrid_clone(gwf->grid, "potential_store");
  wrk = cgrid_clone(gwf->grid, "wrk");

  /* Read external potential (gnd) from file and FFT it */
  dft_common_potential_map(0, GND_X, GND_Y, GND_Z, ext_pot); /* 0 = spherical */
  rgrid_fft(ext_pot);

  /* Initial guess for impurity */
  inv_width = 1.0 / 2.0;
  grid_wf_map(imwf, dft_common_cgaussian, &inv_width);

  /* Run imaginary time iterations to obtain the solvation structure */
  grid_wf_constant(gwf, SQRT(rho0));
  for (i = 0; i < IMITER; i++) {

    if(i == 20) grid_fft_write_wisdom(NULL);

    grid_timer_start(&timer);

    /* convolute impurity density with ext_pot -> ext_pot2 (Helium external potential) */
    grid_wf_density(imwf, otf->density);
    rgrid_fft(otf->density);
    rgrid_fft_convolute(ext_pot2, ext_pot, otf->density);
    rgrid_inverse_fft_norm2(ext_pot2);

    /* Predict-Correct (Helium) */
    grid_real_to_complex_re(potential_store, ext_pot2);
    dft_ot_potential(otf, potential_store, gwf);
    cgrid_add(potential_store, -mu0);
    grid_wf_propagate_predict(gwf, gwfp, potential_store, -I * TS / GRID_AUTOFS);
    grid_add_real_to_complex_re(potential_store, ext_pot2);
    dft_ot_potential(otf, potential_store, gwfp);
    cgrid_add(potential_store, -mu0);
    cgrid_multiply(potential_store, 0.5);  // Use (current + future) / 2
    grid_wf_propagate_correct(gwf, potential_store, -I * TS / GRID_AUTOFS);

    /* convolute liquid density with ext_pot -> ext_pot2 (Impurity external potential) */
    grid_wf_density(gwf, otf->density);
    rgrid_fft(otf->density);
    rgrid_fft_convolute(ext_pot2, ext_pot, otf->density);
    rgrid_inverse_fft_norm2(ext_pot2);

    /* Propagate in single step (Impurity) */
    grid_real_to_complex_re(potential_store, ext_pot2);
    grid_wf_propagate(imwf, potential_store, -I * TS / GRID_AUTOFS);
    grid_wf_normalize(imwf);

    printf("Iteration " FMT_I " - Wall clock time = " FMT_R " seconds.\n", i, grid_timer_wall_clock_time(&timer));

  }

  /* At this point gwf contains the converged wavefunction */
  grid_wf_density(gwf, otf->density);
  rgrid_write_grid("initial-helium", otf->density);
  grid_wf_density(imwf, otf->density);
  rgrid_write_grid("initial-imp", otf->density);

  /* Allocate 1-D complex grid for the spectrum */
  spectrum = cgrid_alloc(1, 1, TCITER, TS / GRID_AUTOFS, CGRID_PERIODIC_BOUNDARY, 0, "Spectrum");

  /* Prepare the difference potential for calculating the spectrum */
  dft_common_potential_map(0, GND_X, GND_Y, GND_Z, ext_pot); /* 0 = spherical */
  dft_common_potential_map(0, EXC_X, EXC_Y, EXC_Z, ext_pot2); /* 0 = spherical */
  rgrid_difference(ext_pot, ext_pot2, ext_pot); /* Difference potential: final - initial */

  /* Liquid and impurity densities */
  grid_wf_density(gwf, otf->density);
  grid_wf_density(imwf, ext_pot2);

  /* Calculate the spectrum */
  dft_spectrum_anderson_zp(otf->density, ext_pot2, ext_pot, spectrum, potential_store, wrk);

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
