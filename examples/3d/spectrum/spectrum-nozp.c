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

#define TMAX 4000.0 /* fs */
#define TSTEP 1.0 /* fs */

#define UPPER_X "potentials/2p-exp.dat"
#define UPPER_Y "potentials/2p-exp.dat"
#define UPPER_Z "potentials/2p-exp.dat"

#define LOWER_X "potentials/2s-exp.dat"
#define LOWER_Y "potentials/2s-exp.dat"
#define LOWER_Z "potentials/2s-exp.dat"

#define HELIUM_MASS (4.002602 / GRID_AUTOAMU)

int main(int argc, char **argv) {

  cgrid3d *potential_store;
  rgrid3d *ext_pot;
  cgrid1d *spectrum;
  wf3d *gwf, *gwfp;
  long iter;
  double energy, natoms, en;
  FILE *fp;

  /* Setup DFT driver parameters (256 x 256 x 256 grid) */
  dft_driver_setup_grid(180, 180, 180, 0.5 /* Bohr */, 16 /* threads */);
  /* Plain Orsay-Trento in imaginary time */
  dft_driver_setup_model(DFT_OT_PLAIN + DFT_OT_KC + DFT_OT_T1600MK, DFT_DRIVER_IMAG_TIME, 0.0260446 * (GRID_AUTOANG * GRID_AUTOANG * GRID_AUTOANG));
  /* No absorbing boundary */
  dft_driver_setup_boundaries(DFT_DRIVER_BOUNDARY_REGULAR, 0.0);
  /* Normalization condition */
  dft_driver_setup_normalization(DFT_DRIVER_NORMALIZE_BULK, 0, 3.0, 10);

  /* Initialize the DFT driver */
  dft_driver_initialize();

  /* Allocate space for external potential */
  ext_pot = dft_driver_alloc_rgrid();
  potential_store = dft_driver_alloc_cgrid(); /* temporary storage */
  /* Read external potential from file */
  dft_common_potential_map(DFT_DRIVER_AVERAGE_NONE, UPPER_X, UPPER_Y, UPPER_Z, ext_pot);

  /* Allocate space for wavefunctions (initialized to sqrt(rho0)) */
  gwf = dft_driver_alloc_wavefunction(HELIUM_MASS); /* helium wavefunction */
  gwfp = dft_driver_alloc_wavefunction(HELIUM_MASS);/* temp. wavefunction */

  /* Run 200 iterations using imaginary time (50 fs time step) */
  for (iter = 0; iter < 200; iter++) {
    dft_driver_propagate_predict(DFT_DRIVER_PROPAGATE_HELIUM, ext_pot, gwf, gwfp, potential_store, 80.0 /* fs */, iter);
    dft_driver_propagate_correct(DFT_DRIVER_PROPAGATE_HELIUM, ext_pot, gwf, gwfp, potential_store, 80.0 /* fs */, iter);
  }
  /* At this point gwf contains the converged wavefunction */
  dft_driver_write_grid(gwf->grid, "output");

  energy = dft_driver_energy(gwf, ext_pot);
  natoms = dft_driver_natoms(gwf);
  printf("Total energy is %le K\n", energy * GRID_AUTOK);
  printf("Number of He atoms is %le.\n", natoms);
  printf("Energy / atom is %le K\n", (energy/natoms) * GRID_AUTOK);

  grid3d_wf_density(gwf, ext_pot);
  spectrum = dft_driver_spectrum(ext_pot, TSTEP, TMAX, 0, UPPER_X, UPPER_Y, UPPER_Z, 0, LOWER_X, LOWER_Y, LOWER_Z);
  if(!(fp = fopen("spectrum.dat", "w"))) {
    fprintf(stderr, "Can't open spectrum.dat for writing.\n");
    exit(1);
  }
  for (iter = 0, en = -0.5 * spectrum->step * (spectrum->nx - 1); iter < spectrum->nx; iter++, en += spectrum->step)
    fprintf(fp, "%le %le\n", en, creal(cgrid1d_value_at_index(spectrum, iter)));
  fclose(fp);
  printf("Spectrum written to spectrum.dat\n");
  exit(0);
}
