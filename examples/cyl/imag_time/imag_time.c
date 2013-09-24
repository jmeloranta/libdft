/*
 * Impurity atom in superfluid helium (no zero-point). 3D cyl.
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

double complex vortex(void *NA, double r, double phi, double z) {

  //eturn (0.0218360 * (0.529 * 0.529 * 0.529)) * cexp(phi*I) * (1.0 - exp(-0.2*r));
  return (0.0218360 * (0.529 * 0.529 * 0.529)) * cexp(phi*I);
}

#define R_M 0.05

double vortex_z(void *na, double r, double phi, double z) {

  double rp2 = r * r;

  if(rp2 < R_M * R_M) rp2 = R_M * R_M;
  return 1.0 / (2.0 * HELIUM_MASS * rp2);
}

int main(int argc, char **argv) {

  cgrid3d *potential_store;
  cgrid3d *cart_wf;
  rgrid3d *ext_pot, *density;
  wf3d *gwf, *gwfp;
  long iter;
  double energy, natoms;
  rgrid3d *cart;
  
  /* Setup DFT driver parameters (256 x 256 x 256 grid) */
  dft_driver_setup_grid_cyl(128, 128, 256, 0.3 /* Bohr */, 8 /* threads */);
  /* Plain Orsay-Trento in imaginary time */
  //dft_driver_setup_model(DFT_OT_PLAIN + DFT_OT_KC + DFT_OT_HD, DFT_DRIVER_IMAG_TIME, 0.0);
  dft_driver_setup_model_cyl(DFT_ZERO, DFT_DRIVER_IMAG_TIME, 0.0);
  /* No absorbing boundary */
  dft_driver_setup_boundaries_cyl(DFT_DRIVER_BOUNDARY_REGULAR, 2.0);
  /* Normalization condition */
  dft_driver_setup_normalization_cyl(DFT_DRIVER_NORMALIZE_BULK, 0, 0.0, 0);

  /* Initialize the DFT driver */
  dft_driver_initialize_cyl();

  /* Allocate space for external potential */
  ext_pot = dft_driver_alloc_rgrid_cyl();
  potential_store = dft_driver_alloc_cgrid_cyl(); /* temporary storage */
  density = dft_driver_alloc_rgrid_cyl();
  cart_wf = dft_driver_alloc_cgrid_cyl_cart();

  /* Allocate space for wavefunctions (initialized to sqrt(rho0)) */
  gwf = dft_driver_alloc_wavefunction_cyl(HELIUM_MASS); /* helium wavefunction */
  gwfp = dft_driver_alloc_wavefunction_cyl(HELIUM_MASS);/* temp. wavefunction */

  /* Read external potential from file */
  // Onsager or zero?
  //rgrid3d_map_cyl(ext_pot, vortex_z, NULL);
  dft_common_potential_map_cyl(DFT_DRIVER_AVERAGE_XYZ, "zero.dat", "zero.dat", "zero.dat", ext_pot);
  cart = dft_driver_alloc_rgrid_cyl_cart();
  
#if 1
  cgrid3d_map_cyl(gwf->grid, vortex, NULL);
  cgrid3d_copy(gwfp->grid, gwf->grid);
  iter = 1;
#else
  iter = 0;
#endif

  /* Run 200 iterations using imaginary time (50 fs time step) */
  for (; iter < 20000; iter++) {
    dft_driver_propagate_predict_cyl(DFT_DRIVER_PROPAGATE_HELIUM, ext_pot, gwf, gwfp, potential_store, 50.0 /* fs */, iter);
    dft_driver_propagate_correct_cyl(DFT_DRIVER_PROPAGATE_HELIUM, ext_pot, gwf, gwfp, potential_store, 50.0 /* fs */, iter);
    if(!(iter % 10)) {
      char buf[512];
      sprintf(buf, "output-%ld", iter);
      grid3d_wf_density(gwf, density);
      dft_driver_write_density_cyl(density, buf);
      sprintf(buf, "coutput-%ld", iter);
      cgrid3d_phase(density, gwf->grid);
      dft_driver_write_density_cyl(density, buf);
      //      cgrid3d_map_cyl_on_cart(cart_wf, gwf->grid);
      //dft_driver_write_grid_cyl(cart_wf, buf);
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
