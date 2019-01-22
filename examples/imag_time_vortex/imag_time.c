/*
 * Impurity atom in superfluid helium (no zero-point).
 * Interaction of impurity with vortex line.
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
#define MAXITER 10000000
#define NX 256
#define NY 256
#define NZ 256
#define STEP 0.5

/* Impurity */
#define HE2STAR 1
/* #define HESTAR  1 */
/* #define AG 1 */
/* #define CU 1 */
/* #define HE3PLUS 1 */

/* Onsager ansatz */
/* #define ONSAGER */

/* What to include? */
/* #define IMPURITY */
#define VORTEX
/* #define BOTH */

#define HELIUM_MASS (4.002602 / GRID_AUTOAMU)
#define HBAR 1.0        /* au */

void zero_core(cgrid *grid) {

  INT i, j, k;
  INT nx = grid->nx, ny = grid->ny, nz = grid->nz;
  REAL x, y, step = grid->step;
  REAL complex *val = grid->value;
  
  for (i = 0; i < nx; i++)
    for (j = 0; j < ny; j++)
      for (k = 0; k < nz; k++) {
	x = ((REAL) (i - nx/2)) * step;
	y = ((REAL) (j - ny/2)) * step;
	if(SQRT(x * x + y * y) < step/2.0)
	  val[i * ny * nz + j * nz + k] = 0.0;
      }
}

int main(int argc, char **argv) {

  cgrid *potential_store;
  rgrid *ext_pot, *density, *px, *py, *pz;
  wf *gwf, *gwfp;
  INT iter, N;
  REAL energy, natoms, mu0, rho0, width;

#ifdef USE_CUDA
  cuda_enable(1);  // enable CUDA ?
#endif

  /* Setup DFT driver parameters (256 x 256 x 256 grid) */
  dft_driver_setup_grid(NX, NY, NZ, STEP /* Bohr */, 0 /* threads */);
  /* Plain Orsay-Trento in imaginary time */
  dft_driver_setup_model(DFT_OT_PLAIN | DFT_OT_KC | DFT_OT_BACKFLOW, DFT_DRIVER_IMAG_TIME, 0.0);
  /* No absorbing boundary */
  dft_driver_setup_boundary_type(DFT_DRIVER_BOUNDARY_REGULAR, 0.0, 0.0, 0.0, 0.0);
//  dft_driver_setup_boundary_type(DFT_DRIVER_BC_Z, 0.0, 0.0, 0.0, 0.0);
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

  printf("N = " FMT_I "\n", (INT) N);

  /* Allocate space for wavefunctions (initialized to SQRT(rho0)) */
  gwf = dft_driver_alloc_wavefunction(HELIUM_MASS, "gwf"); /* helium wavefunction */
  gwfp = dft_driver_alloc_wavefunction(HELIUM_MASS, "gwfp");/* temp. wavefunction */

  /* Initialize the DFT driver */
  dft_driver_initialize(gwf);

  /* Allocate space for external potential */
  ext_pot = dft_driver_alloc_rgrid("ext_pot");
  potential_store = dft_driver_alloc_cgrid("potential_store"); /* temporary storage */
  density = dft_driver_alloc_rgrid("density");
  px = dft_driver_alloc_rgrid("px");
  py = dft_driver_alloc_rgrid("py");
  pz = dft_driver_alloc_rgrid("pz");

#ifdef HE2STAR
  dft_common_potential_map(DFT_DRIVER_AVERAGE_NONE, "he2-He.dat-spline", "he2-He.dat-spline", "he2-He.dat-spline", ext_pot);
#endif
#ifdef HESTAR
  dft_common_potential_map(DFT_DRIVER_AVERAGE_NONE, "He-star-He.dat", "He-star-He.dat", "He-star-He.dat", ext_pot);
#endif
#ifdef AG
  dft_common_potential_map(DFT_DRIVER_AVERAGE_NONE, "aghe-spline.dat", "aghe-spline.dat", "aghe-spline.dat", ext_pot);  
#endif
#ifdef CU
  dft_common_potential_map(DFT_DRIVER_AVERAGE_NONE, "cuhe-spline.dat", "cuhe-spline.dat", "cuhe-spline.dat", ext_pot);  
#endif
#ifdef HE3PLUS
  dft_common_potential_map(DFT_DRIVER_AVERAGE_NONE, "he3+-he-sph_ave.dat", "he3+-he-sph_ave.dat", "he3+-he-sph_ave.dat", ext_pot);
#endif
  //  rgrid_shift(ext_pot, density, 0.0, 0.0, 0.0);
#ifdef VORTEX
  rgrid_zero(ext_pot);
#endif

#ifdef ONSAGER
#if defined(VORTEX) || defined(BOTH)
  dft_driver_vortex(ext_pot, DFT_DRIVER_VORTEX_Z);
#endif
#endif

#if 1
  mu0 = dft_ot_bulk_chempot(dft_driver_otf);
  rho0 = dft_ot_bulk_density(dft_driver_otf);
#else
  rho0 = 0.00323;  // for GP
  mu0 = 0.0;
#endif

  if(N != 0) {
    width = 1.0 / 20.0;
    cgrid_map(gwf->grid, dft_common_cgaussian, (void *) &width);
  } else cgrid_constant(gwf->grid, SQRT(rho0));

#ifndef ONSAGER
#if defined(VORTEX) || defined(BOTH)
  dft_driver_vortex_initial(gwf, 1, DFT_DRIVER_VORTEX_Z);
#endif
#endif

  for (iter = 1; iter < MAXITER; iter++) {
    
    if(iter == 1 || !(iter % 200)) {
      char buf[512];
      grid_wf_density(gwf, density);
      sprintf(buf, "output-" FMT_I, iter);
      rgrid_write_grid(buf, density);
      sprintf(buf, "output-wf-" FMT_I, iter);
      cgrid_write_grid(buf, gwf->grid);

#if 0
      grid_wf_velocity(gwf, px, py, pz, DFT_VELOC_CUTOFF);
      rgrid_multiply(px, GRID_AUTOMPS);
      rgrid_multiply(py, GRID_AUTOMPS);
      rgrid_multiply(pz, GRID_AUTOMPS);
      sprintf(buf, "output-veloc-" FMT_I, iter);
      dft_driver_write_density(px, buf);
#endif

      dft_ot_energy_density(dft_driver_otf, density, gwf);
      rgrid_add_scaled_product(density, 1.0, dft_driver_otf->density, ext_pot);
      energy = grid_wf_energy(gwf, NULL) + rgrid_integral(density);
      natoms = grid_wf_norm(gwf);
      printf("Total energy is " FMT_R " K\n", energy * GRID_AUTOK);
      printf("Number of He atoms is " FMT_R ".\n", natoms);
      printf("Energy / atom is " FMT_R " K\n", (energy/natoms) * GRID_AUTOK);
      grid_wf_probability_flux(gwf, px, py, pz);
      sprintf(buf, "flux_x-" FMT_I, iter);
      rgrid_write_grid(buf, px);
      sprintf(buf, "flux_y-" FMT_I, iter);
      rgrid_write_grid(buf, py);
      sprintf(buf, "flux_z-" FMT_I, iter);
      rgrid_write_grid(buf, pz);
#if 0
      { INT k;
	dft_driver_veloc_field(gwf, px, py, pz);
	grid_wf_density(gwf, density);
	for (k = 0; k < px->nx * px->ny * px->nz; k++)
	  px->value[k] = px->value[k] * px->value[k] + py->value[k] * py->value[k] + pz->value[k] * pz->value[k];
	rgrid_product(px, px, density);
	rgrid_multiply(px, 0.5);
	sprintf(buf, "kinetic-" FMT_I, iter);
	dft_driver_write_density(px, buf);
      }
#endif
    }
    dft_driver_propagate_predict(DFT_DRIVER_PROPAGATE_HELIUM, ext_pot, mu0, gwf, gwfp, potential_store, TIME_STEP, iter);
    dft_driver_propagate_correct(DFT_DRIVER_PROPAGATE_HELIUM, ext_pot, mu0, gwf, gwfp, potential_store, TIME_STEP, iter);

#if defined(VORTEX) || defined(BOTH)
    zero_core(gwf->grid);
#endif
  }
  return 0;
}
