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

#define HELIUM_MASS (4.002602 / GRID_AUTOAMU)

int main(int argc, char **argv) {

  cgrid3d *potential_store;
  rgrid3d *ext_pot, *density;
  wf3d *gwf, *gwfp;
  long iter;
  double energy, natoms;

  /* Setup DFT driver parameters (256 x 256 x 256 grid) */
  /* TODO: Right now, only powers of 2 grids work??!!! */
  dft_driver_setup_grid(256, 256, 256, 0.3 /* Bohr */, 16 /* threads */);
  /* Plain Orsay-Trento in imaginary time */
  //dft_driver_setup_model(DFT_OT_PLAIN + DFT_OT_KC + DFT_OT_HD, DFT_DRIVER_IMAG_TIME, 0.0);
  dft_driver_setup_model(DFT_OT_PLAIN, DFT_DRIVER_IMAG_TIME, 0.0);
  /* No absorbing boundary */
  dft_driver_setup_boundaries(DFT_DRIVER_BOUNDARY_REGULAR, 2.0);
  /* Normalization condition */
  dft_driver_setup_normalization(DFT_DRIVER_NORMALIZE_BULK, 0, 0.0, 0);

  /* Initialize the DFT driver */
  dft_driver_initialize();

  /* Allocate space for external potential */
  ext_pot = dft_driver_alloc_rgrid();
  potential_store = dft_driver_alloc_cgrid(); /* temporary storage */
  density = dft_driver_alloc_rgrid();

  /* Allocate space for wavefunctions (initialized to sqrt(rho0)) */
  gwf = dft_driver_alloc_wavefunction(HELIUM_MASS); /* helium wavefunction */
  gwfp = dft_driver_alloc_wavefunction(HELIUM_MASS);/* temp. wavefunction */

  // debug
#if 0
  dft_driver_read_2d_to_3d(ext_pot, "initial");
  grid3d_real_to_complex_re(gwf->grid, ext_pot);
  cgrid3d_power(gwf->grid, gwf->grid, 0.5);
  iter = 1;
#else
  iter = 0;
#endif

  /* Read external potential from file */
  dft_common_potential_map(DFT_DRIVER_AVERAGE_XYZ, "he2-He.dat-spline", "he2-He.dat-spline", "he2-He.dat-spline", ext_pot);

  /* Run 200 iterations using imaginary time (50 fs time step) */
  for (; iter < 20000; iter++) {
    dft_driver_propagate_predict(DFT_DRIVER_PROPAGATE_HELIUM, ext_pot, gwf, gwfp, potential_store, 10.0 /* fs */, iter);
    dft_driver_propagate_correct(DFT_DRIVER_PROPAGATE_HELIUM, ext_pot, gwf, gwfp, potential_store, 10.0 /* fs */, iter);
    if(!(iter % 100)) {
      char buf[512];
      sprintf(buf, "output-%ld", iter);
      grid3d_wf_density(gwf, density);
      dft_driver_write_density(density, buf);
      //energy = dft_driver_energy(gwf, ext_pot);
      //natoms = dft_driver_natoms(gwf);
      //printf("Total energy is %le K\n", energy * GRID_AUTOK);
      //printf("Number of He atoms is %le.\n", natoms);
      //printf("Energy / atom is %le K\n", (energy/natoms) * GRID_AUTOK);
    }
  }
  /* At this point gwf contains the converged wavefunction */
  //grid3d_wf_density(gwf, density);
  //dft_driver_write_density(density, "output");
}
