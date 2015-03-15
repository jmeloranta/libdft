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

  cgrid2d *potential_store;
  rgrid2d *ext_pot, *density;
  wf2d *gwf, *gwfp;
  long iter;
  double energy, natoms;

  /* Setup DFT driver parameters (256 x 256 x 256 grid) */
  dft_driver_setup_grid_2d(256, 256, 0.5 /* Bohr */, 16 /* threads */);
  /* Plain Orsay-Trento in imaginary time */
  dft_driver_setup_model_2d(DFT_OT_PLAIN | DFT_OT_KC, DFT_DRIVER_IMAG_TIME, 0.0);
  /* No absorbing boundary */
  dft_driver_setup_boundaries_2d(DFT_DRIVER_BOUNDARY_REGULAR, 2.0);
  /* Normalization condition */
  dft_driver_setup_normalization_2d(DFT_DRIVER_NORMALIZE_BULK, 0, 0.0, 0);

  /* Initialize the DFT driver */
  dft_driver_initialize_2d();

  /* Allocate space for external potential */
  ext_pot = dft_driver_alloc_rgrid_2d();
  potential_store = dft_driver_alloc_cgrid_2d(); /* temporary storage */
  density = dft_driver_alloc_rgrid_2d();

  /* Allocate space for wavefunctions (initialized to sqrt(rho0)) */
  gwf = dft_driver_alloc_wavefunction_2d(HELIUM_MASS); /* helium wavefunction */
  gwfp = dft_driver_alloc_wavefunction_2d(HELIUM_MASS);/* temp. wavefunction */
  
  /* Read external potential from file */
  dft_common_potential_map_2d(DFT_DRIVER_AVERAGE_NONE, "he2-He.dat-spline", "he2-He.dat-spline", ext_pot);

  /* Run 200 iterations using imaginary time (50 fs time step) */
  for (iter = 0; iter < 2000; iter++) {
    char buf[512];
    dft_driver_propagate_predict_2d(DFT_DRIVER_PROPAGATE_HELIUM, ext_pot, gwf, gwfp, potential_store, 10.0 /* fs */, iter);
    dft_driver_propagate_correct_2d(DFT_DRIVER_PROPAGATE_HELIUM, ext_pot, gwf, gwfp, potential_store, 10.0 /* fs */, iter);
    if(!(iter % 100)) {
      sprintf(buf, "output-%ld", iter);
      grid2d_wf_density(gwf, density);
      dft_driver_write_density_2d(density, buf);
      energy = dft_driver_energy_2d(gwf, ext_pot);
      natoms = dft_driver_natoms_2d(gwf);
      printf("Total energy is %le K\n", energy * GRID_AUTOK);
      printf("Number of He atoms is %le.\n", natoms);
      printf("Energy / atom is %le K\n", (energy/natoms) * GRID_AUTOK);
    }
  }
  /* At this point gwf contains the converged wavefunction */
  grid2d_wf_density(gwf, density);
  dft_driver_write_density_2d(density, "output");
}
