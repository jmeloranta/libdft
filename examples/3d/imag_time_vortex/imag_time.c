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
#define MAXITER 10000
#define NX 64
#define NY 64
#define NZ 64
#define STEP 1.0

#define HELIUM_MASS (4.002602 / GRID_AUTOAMU)
#define HBAR 1.0        /* au */

/* #define NPHASE (1.0) /**/

double complex vortex_phase(void *xx, double x, double y, double z) {

  double phi, d, dc;

  d = sqrt(x * x + y * y);
  dc = sqrt(x * x + y * y + z * z);
#ifdef NPHASE
  if(d < STEP) return 0.0;
  phi = M_PI - atan2(y, -x);
  return cexp(I * NPHASE * phi) / (dc * dc + 1E-6);
  //return cexp(I * NPHASE * phi);
#else
  return 1.0 / (dc * dc + 1E-6);
#endif
}

int main(int argc, char **argv) {

  cgrid3d *potential_store;
  rgrid3d *ext_pot, *density, *px, *py, *pz;
  wf3d *gwf, *gwfp;
  long iter, N;
  double energy, natoms;

  /* Setup DFT driver parameters (256 x 256 x 256 grid) */
  dft_driver_setup_grid(NX, NY, NZ, STEP /* Bohr */, 48 /* threads */);
  /* Plain Orsay-Trento in imaginary time */
  dft_driver_setup_model(DFT_OT_PLAIN + DFT_OT_HD, DFT_DRIVER_IMAG_TIME, 0.0);
  //dft_driver_setup_model(DFT_DR, DFT_DRIVER_IMAG_TIME, 0.0);
  /* No absorbing boundary */
  dft_driver_setup_boundaries(DFT_DRIVER_BOUNDARY_REGULAR, 2.0);
  /* Normalization condition */
  if(argc != 2) {
    fprintf(stderr, "Usage: imag_time N\n");
    exit(1);
  }
  N = atoi(argv[1]);
  if(N == 0) 
    dft_driver_setup_normalization(DFT_DRIVER_DONT_NORMALIZE, 0, 0.0, 1); // 1 = release center immediately
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

  rgrid3d_zero(ext_pot);
  //cgrid3d_constant(gwf->grid, 1.0);
  //dft_driver_vortex_initial(gwf, 1, DFT_DRIVER_VORTEX_Z);
  //cgrid3d_add(gwf->grid, 0.0);
  cgrid3d_map(gwf->grid, vortex_phase, NULL);

  for (iter = 1; iter < MAXITER; iter++) {
    
    dft_driver_propagate_predict(DFT_DRIVER_PROPAGATE_HELIUM, ext_pot, gwf, gwfp, potential_store, TIME_STEP, iter);
    dft_driver_propagate_correct(DFT_DRIVER_PROPAGATE_HELIUM, ext_pot, gwf, gwfp, potential_store, TIME_STEP, iter);

    if(!(iter % 100)) {
      char buf[512];
      grid3d_wf_density(gwf, density);
      sprintf(buf, "output-%ld", iter);
      dft_driver_write_density(density, buf);
      energy = dft_driver_energy(gwf, ext_pot);
      natoms = dft_driver_natoms(gwf);
      printf("Total energy is %le K\n", energy * GRID_AUTOK);
      printf("Number of He atoms is %le.\n", natoms);
      printf("Energy / atom is %le K\n", (energy/natoms) * GRID_AUTOK);
      grid3d_wf_probability_flux(gwf, px, py, pz);
      sprintf(buf, "flux_x-%ld", iter);
      dft_driver_write_density(px, buf);
      sprintf(buf, "flux_y-%ld", iter);
      dft_driver_write_density(py, buf);
      sprintf(buf, "flux_z-%ld", iter);
      dft_driver_write_density(pz, buf);
    }
  }
  return 0;
}
