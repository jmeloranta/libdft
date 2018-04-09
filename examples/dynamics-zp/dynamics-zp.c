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

/* He^* */
#define IMP_MASS (4.002602 / GRID_AUTOAMU)

#define INITIAL_POT_X "initial_pot.x"
#define INITIAL_POT_Y "initial_pot.y"
#define INITIAL_POT_Z "initial_pot.z"

#define FINAL_POT_X "final_pot.x"
#define FINAL_POT_Y "final_pot.y"
#define FINAL_POT_Z "final_pot.z"

#define HELIUM_MASS (4.002602 / GRID_AUTOAMU)

int main(int argc, char **argv) {

  rgrid3d *ext_pot, *ext_pot2, *density;
  cgrid3d *potential_store;
  wf3d *gwf, *gwfp;
  wf3d *imwf, *imwfp;
  INT iter;
  char buf[512];

  /* Setup DFT driver parameters (256 x 256 x 256 grid) */
  dft_driver_setup_grid(180, 180, 180, 0.5 /* Bohr */, 16 /* threads */);
  /* Plain Orsay-Trento in imaginary time */
  dft_driver_setup_model(DFT_OT_PLAIN + DFT_OT_KC + DFT_OT_T1600MK, DFT_DRIVER_IMAG_TIME, 0.0218360 * (GRID_AUTOANG * GRID_AUTOANG * GRID_AUTOANG));
  /* No absorbing boundary */
  dft_driver_setup_boundary_type(DFT_DRIVER_BOUNDARY_REGULAR, 0.0, 0.0, 0.0, 0.0);
  /* Normalization condition */
  dft_driver_setup_normalization(DFT_DRIVER_NORMALIZE_BULK, 0, 3.0, 10);

  /* Initialize the DFT driver */
  dft_driver_initialize();

  /* Allocate space for external potential */
  ext_pot = dft_driver_alloc_rgrid("ext_pot");
  ext_pot2 = dft_driver_alloc_rgrid("ext_pot2");
  density = dft_driver_alloc_rgrid("density");
  potential_store = dft_driver_alloc_cgrid("potential_store"); /* temporary storage */

  /* Read external potential from file */
  dft_common_potential_map(DFT_DRIVER_AVERAGE_NONE, INITIAL_POT_X, INITIAL_POT_Y, INITIAL_POT_Z, ext_pot);
  dft_driver_convolution_prepare(NULL, ext_pot);

  /* Allocate space for wavefunctions (initialized to SQRT(rho0)) */
  gwf = dft_driver_alloc_wavefunction(HELIUM_MASS, "gwf"); /* helium wavefunction */
  gwfp = dft_driver_alloc_wavefunction(HELIUM_MASS, "gwfp");/* temp. wavefunction */
  imwf = dft_driver_alloc_wavefunction(IMP_MASS, "imwf"); /*  imp. wavefunction */
  imwfp = dft_driver_alloc_wavefunction(IMP_MASS, "imwfp");/* temp. wavefunction */
  dft_driver_gaussian_wavefunction(imwf, 0.0, 0.0, 0.0, 2.0);
  dft_driver_gaussian_wavefunction(imwfp, 0.0, 0.0, 0.0, 2.0);

  /* Step #1: Optimize structure */
  for (iter = 0; iter < 200; iter++) {
    /* convolute impurity density with ext_pot -> ext_pot2 */
    grid3d_wf_density(imwf, density);
    dft_driver_convolution_prepare(density, NULL);
    dft_driver_convolution_eval(ext_pot2, ext_pot, density);
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
  dft_driver_write_grid(gwf->grid, "initial1");
  dft_driver_write_grid(imwf->grid, "initial2");

  /* Step #2: Propagate using the final state potential */
  dft_driver_setup_model(DFT_OT_PLAIN + DFT_OT_KC + DFT_OT_T1600MK, DFT_DRIVER_REAL_TIME, 0.0218360 * (GRID_AUTOANG * GRID_AUTOANG * GRID_AUTOANG));
  dft_common_potential_map(DFT_DRIVER_AVERAGE_NONE, FINAL_POT_X, FINAL_POT_Y, FINAL_POT_Z, ext_pot);
  dft_driver_convolution_prepare(NULL, ext_pot);
  for (iter = 0; iter < 200; iter++) {
    /* convolute impurity density with ext_pot -> ext_pot2 */
    grid3d_wf_density(imwf, density);
    dft_driver_convolution_prepare(density, NULL);
    dft_driver_convolution_eval(ext_pot2, ext_pot, density);
    dft_driver_propagate_predict(DFT_DRIVER_PROPAGATE_HELIUM, ext_pot2, gwf, gwfp, potential_store, TS, iter);
    dft_driver_propagate_correct(DFT_DRIVER_PROPAGATE_HELIUM, ext_pot2, gwf, gwfp, potential_store, TS, iter);

    /* convolute liquid density with ext_pot -> ext_pot2 */
    grid3d_wf_density(gwf, density);
    dft_driver_convolution_prepare(density, NULL);
    dft_driver_convolution_eval(ext_pot2, ext_pot, density);
    dft_driver_propagate_predict(DFT_DRIVER_PROPAGATE_OTHER, ext_pot2, imwf, imwfp, potential_store, TS, iter);
    dft_driver_propagate_correct(DFT_DRIVER_PROPAGATE_OTHER, ext_pot2, imwf, imwfp, potential_store, TS, iter);
    if(!(iter % 10)) {
      sprintf(buf, "final1-" FMT_I, iter);
      dft_driver_write_grid(gwf->grid, buf);
      sprintf(buf, "final2-" FMT_I, iter);
      dft_driver_write_grid(imwf->grid, buf);
    }
  }
  return 0;
}
