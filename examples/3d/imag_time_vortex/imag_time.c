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

#define TIME_STEP 5.0  /* fs */
#define MAXITER 1000000
#define NX 128
#define NY 128
#define NZ 128
#define STEP 1.0

/* #define ONSAGER /**/

#define HELIUM_MASS (4.002602 / GRID_AUTOAMU)
#define HBAR 1.0        /* au */

int main(int argc, char **argv) {

  cgrid3d *potential_store;
  rgrid3d *ext_pot, *density, *px, *py, *pz;
  wf3d *gwf, *gwfp;
  long iter, N;
  double energy, natoms, mu0, rho0;

  /* Setup DFT driver parameters (256 x 256 x 256 grid) */
  dft_driver_setup_grid(NX, NY, NZ, STEP /* Bohr */, 32 /* threads */);
  /* Plain Orsay-Trento in imaginary time */
  dft_driver_setup_model(DFT_OT_PLAIN, DFT_DRIVER_IMAG_TIME, 0.0);
  /* No absorbing boundary */
  dft_driver_setup_boundaries(DFT_DRIVER_BOUNDARY_REGULAR, 2.0);
  /* Neumann boundaries */
  // DEBUG
  //dft_driver_setup_boundary_condition(DFT_DRIVER_BC_NORMAL);
  dft_driver_setup_boundary_condition(DFT_DRIVER_BC_NEUMANN);
  //dft_driver_setup_boundary_condition(DFT_DRIVER_BC_Z);

  /* Normalization condition */
  if(argc != 2) {
    fprintf(stderr, "Usage: imag_time N\n");
    exit(1);
  }
  N = atoi(argv[1]);
  if(N == 0) 
    dft_driver_setup_normalization(DFT_DRIVER_NORMALIZE_BULK, 0, 0.0, 1); // 1 = release center immediately
  else
    dft_driver_setup_normalization(DFT_DRIVER_NORMALIZE_DROPLET, N, 0.0, 1); // 1 = release center immediately

  printf("N = %ld\n", N);

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

  dft_common_potential_map(DFT_DRIVER_AVERAGE_NONE, "he2-He.dat-spline", "he2-He.dat-spline", "he2-He.dat-spline", density);
  rgrid3d_shift(ext_pot, density, 0.0, 0.0, 0.0);
  // debug
  rgrid3d_zero(ext_pot);
#ifdef ONSAGER
  dft_driver_vortex(ext_pot, DFT_DRIVER_VORTEX_X);
#endif

  mu0 = bulk_chempot(dft_driver_otf);
  rgrid3d_add(ext_pot, -mu0);
  rho0 = bulk_density(dft_driver_otf);

  cgrid3d_constant(gwf->grid, 1.0);
#ifndef ONSAGER
  dft_driver_vortex_initial(gwf, 1, DFT_DRIVER_VORTEX_Z);
#endif

  // DEBUG
  dft_driver_setup_rotation_omega(0.0);
  dft_driver_kinetic = DFT_DRIVER_KINETIC_CN_NBC_ROT;

  for (iter = 1; iter < MAXITER; iter++) {
    
    if(!(iter % 10)) {
      char buf[512];
      double complex c;
      grid3d_wf_density(gwf, density);
      sprintf(buf, "output-%ld", iter);
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
#ifndef ONSAGER
#if 0
      /* Orthogonalization */
      cgrid3d_constant(potential_store, sqrt(rho0));
      c = cgrid3d_integral_of_conjugate_product(potential_store, gwf->grid) / cgrid3d_integral_of_conjugate_product(potential_store,potential_store);
      printf("c = (%le,%le)\n", creal(c), cimag(c));
      cgrid3d_multiply(potential_store, c);
      cgrid3d_difference(gwf->grid, gwf->grid, potential_store);
#endif
#endif
    }

    dft_driver_propagate_predict(DFT_DRIVER_PROPAGATE_HELIUM, ext_pot, gwf, gwfp, potential_store, TIME_STEP, iter);
    dft_driver_propagate_correct(DFT_DRIVER_PROPAGATE_HELIUM, ext_pot, gwf, gwfp, potential_store, TIME_STEP, iter);

  }
  return 0;
}
