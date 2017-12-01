/*
 * Impurity atom in superfluid helium (no zero-point).
 * Scan the impurity vortex distance.
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

#define TIME_STEP 20.0 /* fs */
#define MAXITER 5000   /* was 5000 */
#define NX 256
#define NY 128
#define NZ 128
#define STEP 1.0

#define IBEGIN 70.0
#define ISTEP 5.0
#define IEND 0.0

/* #define HE2STAR 1 /**/
#define HESTAR  1 /**/

/* #define ONSAGER /**/

#define HELIUM_MASS (4.002602 / GRID_AUTOAMU)
#define HBAR 1.0        /* au */

void zero_core(cgrid3d *grid) {

  long i, j, k;
  long nx = grid->nx, ny = grid->ny, nz = grid->nz;
  double x, y, step = grid->step;
  double complex *val = grid->value;
  
  for (i = 0; i < nx; i++)
    for (j = 0; j < ny; j++)
      for (k = 0; k < nz; k++) {
	x = (i - nx/2) * step;
	y = (j - ny/2) * step;
	if(sqrt(x * x + y * y) < step/2.0)
	  val[i * ny * nz + j * nz + k] = 0.0;
      }
}

int main(int argc, char **argv) {

  cgrid3d *potential_store;
  rgrid3d *ext_pot, *orig_pot, *density, *px, *py, *pz;
  wf3d *gwf, *gwfp;
  char buf[512];
  long iter, N;
  double energy, natoms, mu0, rho0, width, R;

  /* Setup DFT driver parameters (256 x 256 x 256 grid) */
  dft_driver_setup_grid(NX, NY, NZ, STEP /* Bohr */, 32 /* threads */);
  /* Plain Orsay-Trento in imaginary time */
  dft_driver_setup_model(DFT_OT_PLAIN, DFT_DRIVER_IMAG_TIME, 0.0);
  /* No absorbing boundary */
  dft_driver_setup_boundary_type(DFT_DRIVER_BOUNDARY_REGULAR, 0.0, 0.0);
  /* Neumann boundaries */
  dft_driver_setup_boundary_condition(DFT_DRIVER_BC_NEUMANN);

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
  orig_pot = dft_driver_alloc_rgrid();
  potential_store = dft_driver_alloc_cgrid(); /* temporary storage */
  density = dft_driver_alloc_rgrid();
  px = dft_driver_alloc_rgrid();
  py = dft_driver_alloc_rgrid();
  pz = dft_driver_alloc_rgrid();

  /* Allocate space for wavefunctions (initialized to sqrt(rho0)) */
  gwf = dft_driver_alloc_wavefunction(HELIUM_MASS); /* helium wavefunction */
  gwfp = dft_driver_alloc_wavefunction(HELIUM_MASS);/* temp. wavefunction */

#ifdef HE2STAR
  dft_common_potential_map(DFT_DRIVER_AVERAGE_NONE, "he2-He.dat-spline", "he2-He.dat-spline", "he2-He.dat-spline", orig_pot);
#endif
#ifdef HESTAR
  dft_common_potential_map(DFT_DRIVER_AVERAGE_NONE, "He-star-He.dat", "He-star-He.dat", "He-star-He.dat", orig_pot);
#endif
  mu0 = dft_ot_bulk_chempot(dft_driver_otf);
  rgrid3d_add(orig_pot, -mu0);
  rho0 = dft_ot_bulk_density(dft_driver_otf);

  if(N != 0) {
    width = 1.0 / 20.0;
    cgrid3d_map(gwf->grid, dft_common_cgaussian, (void *) &width);
  } else cgrid3d_constant(gwf->grid, sqrt(rho0));

#ifndef ONSAGER
  dft_driver_vortex_initial(gwf, 1, DFT_DRIVER_VORTEX_Z);
#endif

  for (R = IBEGIN; R >= IEND; R -= ISTEP) {
    rgrid3d_shift(ext_pot, orig_pot, R, 0.0, 0.0);
#ifdef ONSAGER
    dft_driver_vortex(ext_pot, DFT_DRIVER_VORTEX_X);
#endif
    // TODO: Do we need to enforce the vortex solution at each R?

    for (iter = 1; iter < ((R == IBEGIN)?10*MAXITER:MAXITER); iter++) {      
      dft_driver_propagate_predict(DFT_DRIVER_PROPAGATE_HELIUM, ext_pot, gwf, gwfp, potential_store, TIME_STEP, iter);
      dft_driver_propagate_correct(DFT_DRIVER_PROPAGATE_HELIUM, ext_pot, gwf, gwfp, potential_store, TIME_STEP, iter);
      zero_core(gwf->grid);
    }

    printf("Results for R = %le\n", R);
    grid3d_wf_density(gwf, density);
    sprintf(buf, "output-%lf", R);
    dft_driver_write_density(density, buf);
    energy = dft_driver_energy(gwf, ext_pot);
    natoms = dft_driver_natoms(gwf);
    printf("Total energy is %le K\n", energy * GRID_AUTOK);
    printf("Number of He atoms is %le.\n", natoms);
    printf("Energy / atom is %le K\n", (energy/natoms) * GRID_AUTOK);
    grid3d_wf_probability_flux(gwf, px, py, pz);
    sprintf(buf, "flux_x-%lf", R);
    dft_driver_write_density(px, buf);
    sprintf(buf, "flux_y-%lf", R);
    dft_driver_write_density(py, buf);
    sprintf(buf, "flux_z-%lf", R);
    dft_driver_write_density(pz, buf);
    printf("PES %le %le\n", R, energy * GRID_AUTOK);
  }
  return 0;
}
