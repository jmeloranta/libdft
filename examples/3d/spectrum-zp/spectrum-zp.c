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

#define HELIUM_MASS (4.002602 / GRID_AUTOAMU)
#define IMP_MASS (4.002602 / GRID_AUTOAMU) /* He^* */

int main(int argc, char **argv) {

  cgrid3d *potential_store;
  cgrid1d *spectrum;
  rgrid3d *ext_pot, *ext_pot2, *density;
  wf3d *gwf, *gwfp;
  wf3d *imwf, *imwfp;
  long iter;
  double energy, natoms, en, mu0;
  FILE *fp;

  /* Setup DFT driver parameters */
  dft_driver_setup_grid(NX, NY, NZ, STEP, 0);
  /* Plain Orsay-Trento in imaginary time */
  dft_driver_setup_model(MODEL, DFT_DRIVER_IMAG_TIME, RHO0);
  /* No absorbing boundary */
  dft_driver_setup_boundaries(DFT_DRIVER_BOUNDARY_REGULAR, 0.0);
  /* Normalization condition */
  dft_driver_setup_normalization(DFT_DRIVER_DONT_NORMALIZE, 0, 0.0, 0);
  /* Neumann boundaries */
  dft_driver_setup_boundary_condition(DFT_DRIVER_BC_NEUMANN);

  /* Initialize the DFT driver */
  dft_driver_initialize();

  /* Allocate space for external potential */
  ext_pot = dft_driver_alloc_rgrid();
  ext_pot2 = dft_driver_alloc_rgrid();
  density = dft_driver_alloc_rgrid();
  potential_store = dft_driver_alloc_cgrid(); /* temporary storage */

  /* Read external potential from file */
  dft_common_potential_map(DFT_DRIVER_AVERAGE_NONE, LOWER_X, LOWER_Y, LOWER_Z, ext_pot);
  dft_driver_convolution_prepare(NULL, ext_pot);

  /* Allocate space for wavefunctions (initialized to sqrt(rho0)) */
  gwf = dft_driver_alloc_wavefunction(HELIUM_MASS); /* helium wavefunction */
  gwfp = dft_driver_alloc_wavefunction(HELIUM_MASS);/* temp. wavefunction */
  imwf = dft_driver_alloc_wavefunction(IMP_MASS); /*  imp. wavefunction */
  imwfp = dft_driver_alloc_wavefunction(IMP_MASS);/* temp. wavefunction */
  dft_driver_gaussian_wavefunction(imwf, 0.0, 0.0, 0.0, 2.0);
  dft_driver_gaussian_wavefunction(imwfp, 0.0, 0.0, 0.0, 2.0);

  mu0 = dft_ot_bulk_chempot2(dft_driver_otf);
  printf("mu0 = %le K.\n", mu0 * GRID_AUTOK);

  iter = 0;

  /* Run imaginary time iterations */
  for (; iter < IMITER; iter++) {
    /* convolute impurity density with ext_pot -> ext_pot2 */
    grid3d_wf_density(imwf, density);
    dft_driver_convolution_prepare(density, NULL);
    dft_driver_convolution_eval(ext_pot2, ext_pot, density);
    rgrid3d_add(ext_pot2, -mu0);
    dft_driver_propagate_predict(DFT_DRIVER_PROPAGATE_HELIUM, ext_pot2, gwf, gwfp, potential_store, TS, iter);
    dft_driver_propagate_correct(DFT_DRIVER_PROPAGATE_HELIUM, ext_pot2, gwf, gwfp, potential_store, TS, iter);

    /* convolute liquid density with ext_pot -> ext_pot2 */
    grid3d_wf_density(gwf, density);
    dft_driver_convolution_prepare(density, NULL);
    dft_driver_convolution_eval(ext_pot2, ext_pot, density);
    dft_driver_propagate_predict(DFT_DRIVER_PROPAGATE_OTHER, ext_pot2, imwf, imwfp, potential_store, TS, iter);
    dft_driver_propagate_correct(DFT_DRIVER_PROPAGATE_OTHER, ext_pot2, imwf, imwfp, potential_store, TS, iter);
  }

  /* At this point gwf contains the converged wavefunction */
  grid3d_wf_density(gwf, density);
  dft_driver_write_density(density, "initial-helium");
  grid3d_wf_density(imwf, density);
  dft_driver_write_density(density, "initial-imp");

  energy = dft_driver_energy(gwf, ext_pot);
  natoms = dft_driver_natoms(gwf);
  printf("Total energy of the liquid is %le K\n", energy * GRID_AUTOK);
  printf("Number of He atoms is %le.\n", natoms);
  printf("Energy / atom is %le K\n", (energy/natoms) * GRID_AUTOK);

  /* Propagate only the liquid in real time */
  dft_driver_setup_model(MODEL, DFT_DRIVER_REAL_TIME, RHO0);
  grid3d_wf_density(imwf, density);
  ext_pot = dft_driver_spectrum_init(density, REITER, ZEROFILL, DFT_DRIVER_AVERAGE_NONE, UPPER_X, UPPER_Y, UPPER_Z, DFT_DRIVER_AVERAGE_NONE, LOWER_X, LOWER_Y, LOWER_Z);
  rgrid3d_add(ext_pot, -mu0);
  for (iter = 0; iter < REITER; iter++) {
    dft_driver_propagate_predict(DFT_DRIVER_PROPAGATE_HELIUM, ext_pot2, gwf, gwfp, potential_store, TS, iter);
    dft_driver_propagate_correct(DFT_DRIVER_PROPAGATE_HELIUM, ext_pot2, gwf, gwfp, potential_store, TS, iter);
    dft_driver_spectrum_collect(gwf);
    if(!(iter % 10)) {
      char buf[512];
      grid3d_wf_density(gwf, density);
      sprintf(buf, "realtime-%ld", iter);
      dft_driver_write_density(density, buf);
    }
  }
  spectrum = dft_driver_spectrum_evaluate(TS, TC);
  if(!(fp = fopen("spectrum.dat", "w"))) {
    fprintf(stderr, "Can't open spectrum.dat for writing.\n");
    exit(1);
  }
  for (iter = 0, en = -0.5 * spectrum->step * (spectrum->nx - 1); iter < spectrum->nx; iter++, en += spectrum->step)
    // fprintf(fp, "%le %le\n", en, creal(cgrid1d_value_at_index(spectrum, iter)));
    fprintf(fp, "%le %le\n", en, pow(creal(cgrid1d_value_at_index(spectrum, iter)), 2.0) + pow(cimag(cgrid1d_value_at_index(spectrum, iter)), 2.0));
  fclose(fp);
  printf("Spectrum written to spectrum.dat\n");
  exit(0);
}
