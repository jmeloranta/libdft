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

#define DYNAMIC_OMEGA /**/

#define HELIUM_MASS (4.002602 / GRID_AUTOAMU)
#define HBAR_SI 1.0545727E-34
#define C (2.99792458e8)
#define I_CONV (1.66054E-27 * 10E-10 * 10E-10)  
#define I_FREE (83.10 * I_CONV)   // these were in amu Angs^2

double switch_axis(void *xx, double x, double y, double z) {

  rgrid3d *grid = (rgrid3d *) xx;

  return rgrid3d_value(grid, z, y, x);  // swap x and z -> molecule along x axis
}

int main(int argc, char **argv) {

  cgrid3d *potential_store;
  rgrid3d *ext_pot, *density, *px, *py, *pz;
  wf3d *gwf, *gwfp;
  long iter, N;
  double energy, natoms, omega, rp, beff, ieff, lx, ly, lz;

  /* Setup DFT driver parameters (256 x 256 x 256 grid) */
  dft_driver_setup_grid(128, 128, 128, 0.5 /* Bohr */, 4 /* threads */);
  /* Plain Orsay-Trento in imaginary time */
  dft_driver_setup_model(DFT_OT_PLAIN, DFT_DRIVER_IMAG_TIME, 0.0);
  /* No absorbing boundary */
  dft_driver_setup_boundaries(DFT_DRIVER_BOUNDARY_REGULAR, 2.0);
  /* Normalization condition */
  //  dft_driver_setup_normalization(DFT_DRIVER_NORMALIZE_BULK, 0, 0.0, 0);
  // dft_driver_setup_normalization(DFT_DRIVER_DONT_NORMALIZE, 0, 0.0, 0);
  if(argc != 2) {
    fprintf(stderr, "Usage: imag_time N\n");
    exit(1);
  }
  N = atoi(argv[1]);
  if(N == 0) 
    dft_driver_setup_normalization(DFT_DRIVER_DONT_NORMALIZE, 0, 0.0, 1); // 1 = release center immediately
  else
    dft_driver_setup_normalization(DFT_DRIVER_NORMALIZE_DROPLET, N, 0.0, 1); // 1 = release center immediately

  /* Set up rotating liquid */
  dft_driver_kinetic = DFT_DRIVER_KINETIC_CN_NBC_ROT;

  /* Initialize the DFT driver */
  dft_driver_initialize();

  /* Allocate space for external potential */
  ext_pot = dft_driver_alloc_rgrid();
  potential_store = dft_driver_alloc_cgrid(); /* temporary storage */
  density = dft_driver_alloc_rgrid();
  px = dft_driver_alloc_rgrid();
  py = dft_driver_alloc_rgrid();
  pz = dft_driver_alloc_rgrid();

  /* Allocate space for wavefunctions (initialized to sqrt(rho0)) */
  gwf = dft_driver_alloc_wavefunction(HELIUM_MASS); /* helium wavefunction */
  gwfp = dft_driver_alloc_wavefunction(HELIUM_MASS);/* temp. wavefunction */

  /* Read external potential from file */
  density->value_outside = RGRID3D_DIRICHLET_BOUNDARY;  // for extrapolation to work
  //  dft_driver_read_density(px, "ocs_pairpot_256_0.25");      // molecule along z
  dft_driver_read_density(density, "ocs_pairpot_128_0.5");      // molecule along z
  density->value_outside = RGRID3D_PERIODIC_BOUNDARY;   // done, back to original
  rgrid3d_map(ext_pot, switch_axis, density);           // reorient from z to x
  rgrid3d_add(ext_pot, 7.2 / GRID_AUTOK);

  /* Initial omega */
  beff =  HBAR_SI / (4.0 * M_PI * C * I_FREE);
  rp = 1.0 / (2.0 * beff * 2.99793E10); /* rotational period in seconds */
  rp /= 2.4188843265E-17;  /* s -> au */
  omega = 2.0 * M_PI / rp;
  printf("Initial omega = %le\n", omega);

  /* Run 10000 iterations using imaginary time (10 fs time step) */
  for (iter = 0; iter < 10000; iter++) {
    
    // Effective mass
    dft_driver_L(gwf, &lx, &ly, &lz);
    ieff = lz / omega;  // Ieff
    ieff *= GRID_AUTOAMU * GRID_AUTOANG * GRID_AUTOANG; // To u Angs^2
    printf("I_eff = %le AMU Angs^2.\n", ieff);
    beff =  HBAR_SI / (4.0 * M_PI * C * (I_FREE + ieff * I_CONV));
    printf("B_eff = %le cm-1.\n", beff);
#ifndef DYNAMIC_OMEGA
    // Fixed B
    beff =  0.45 * HBAR_SI / (4.0 * M_PI * C * I_FREE);   // approx bulk value
    // end fixed B
#endif
    rp = 1.0 / (2.0 * beff * 2.99793E10); /* rotational period in seconds */
    rp /= 2.4188843265E-17;  /* s -> au */
    omega = 2.0 * M_PI / rp;
    printf("Omega = %le\n", omega);
    dft_driver_setup_rotation_omega(omega);

    dft_driver_propagate_predict(DFT_DRIVER_PROPAGATE_HELIUM, ext_pot, gwf, gwfp, potential_store, 10.0 /* fs */, iter);
    dft_driver_propagate_correct(DFT_DRIVER_PROPAGATE_HELIUM, ext_pot, gwf, gwfp, potential_store, 10.0 /* fs */, iter);

    if(!(iter % 1000)) {
      char buf[512];
      sprintf(buf, "output-%ld", iter);
      grid3d_wf_density(gwf, density);
      dft_driver_write_density(density, buf);
      energy = dft_driver_energy(gwf, ext_pot);
      natoms = dft_driver_natoms(gwf);
      printf("Total energy is %le K\n", energy * GRID_AUTOK);
      printf("Number of He atoms is %le.\n", natoms);
      printf("Energy / atom is %le K\n", (energy/natoms) * GRID_AUTOK);
#if 0
      grid3d_wf_probability_flux(gwf, px, py, pz);
      sprintf(buf, "flux_x-%ld", iter);
      dft_driver_write_density(px, buf);
      sprintf(buf, "flux_y-%ld", iter);
      dft_driver_write_density(py, buf);
      sprintf(buf, "flux_z-%ld", iter);
      dft_driver_write_density(pz, buf);
#endif
    }
  }
  /* At this point gwf contains the converged wavefunction */
  grid3d_wf_density(gwf, density);
  dft_driver_write_density(density, "output");
}
