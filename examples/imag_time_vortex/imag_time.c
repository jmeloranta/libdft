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

#define PRESSURE 0.0
#define THREADS 0

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

  dft_ot_functional *otf;
  cgrid *potential_store;
  rgrid *ext_pot, *density, *px, *py, *pz;
  wf *gwf, *gwfp;
  INT iter, N;
  REAL energy, natoms, mu0, rho0, width;
  grid_timer timer;

  /* Normalization condition */
  if(argc != 2) {
    fprintf(stderr, "Usage: imag_time N\n");
    exit(1);
  }
  N = atoi(argv[1]);

#ifdef USE_CUDA
  cuda_enable(1);  // enable CUDA ?
#endif

  /* Initialize threads & use wisdom */
  grid_set_fftw_flags(1);    // FFTW_MEASURE
  grid_threads_init(THREADS);
  grid_fft_read_wisdom(NULL);

  /* Allocate wave functions */
  if(!(gwf = grid_wf_alloc(NX, NY, NZ, STEP, DFT_HELIUM_MASS, WF_PERIODIC_BOUNDARY, WF_2ND_ORDER_FFT, "gwf"))) {
    fprintf(stderr, "Cannot allocate gwf.\n");
    exit(1);
  }
  gwfp = grid_wf_clone(gwf, "gwfp");

  /* Allocate OT functional */
  if(!(otf = dft_ot_alloc(DFT_OT_PLAIN | DFT_OT_BACKFLOW | DFT_OT_KC | DFT_OT_HD, gwf, DFT_MIN_SUBSTEPS, DFT_MAX_SUBSTEPS))) {
    fprintf(stderr, "Cannot allocate otf.\n");
    exit(1);
  }
  rho0 = dft_ot_bulk_density_pressurized(otf, PRESSURE);
  mu0 = dft_ot_bulk_chempot_pressurized(otf, PRESSURE);
  printf("mu0 = " FMT_R " K/atom, rho0 = " FMT_R " Angs^-3.\n", mu0 * GRID_AUTOK, rho0 / (GRID_AUTOANG * GRID_AUTOANG * GRID_AUTOANG));

  printf("N = " FMT_I "\n", (INT) N);

  /* Allocate space for external potential */
  ext_pot = rgrid_clone(otf->density, "ext_pot");
  potential_store = cgrid_clone(gwf->grid, "potential_store");
  density = rgrid_clone(otf->density, "density");
  px = rgrid_clone(otf->density, "px");
  py = rgrid_clone(otf->density, "py");
  pz = rgrid_clone(otf->density, "pz");

#ifdef HE2STAR
  dft_common_potential_map(0, "he2-He.dat-spline", "he2-He.dat-spline", "he2-He.dat-spline", ext_pot);
#endif
#ifdef HESTAR
  dft_common_potential_map(0, "He-star-He.dat", "He-star-He.dat", "He-star-He.dat", ext_pot);
#endif
#ifdef AG
  dft_common_potential_map(0, "aghe-spline.dat", "aghe-spline.dat", "aghe-spline.dat", ext_pot);  
#endif
#ifdef CU
  dft_common_potential_map(0, "cuhe-spline.dat", "cuhe-spline.dat", "cuhe-spline.dat", ext_pot);  
#endif
#ifdef HE3PLUS
  dft_common_potential_map(0, "he3+-he-sph_ave.dat", "he3+-he-sph_ave.dat", "he3+-he-sph_ave.dat", ext_pot);
#endif
  //  rgrid_shift(ext_pot, density, 0.0, 0.0, 0.0);
#ifdef VORTEX
  rgrid_zero(ext_pot);
#endif

#ifdef ONSAGER
#if defined(VORTEX) || defined(BOTH)
  rgrid_map(otf->density, &dft_initial_vortex_z, (void *) gwf);
  rgrid_sum(ext_pot, ext_pot, otf->density);
#endif
#endif

  if(N != 0) {
    width = 1.0 / 20.0;
    cgrid_map(gwf->grid, dft_common_cgaussian, (void *) &width);
  } else cgrid_constant(gwf->grid, SQRT(rho0));

#ifndef ONSAGER
#if defined(VORTEX) || defined(BOTH)
  grid_wf_map(gwf, &dft_initial_vortex_z_n1, NULL);
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
      rgrid_write_grid(buf, px);
#endif

      dft_ot_energy_density(otf, density, gwf);
      rgrid_add_scaled_product(density, 1.0, otf->density, ext_pot);
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
	grid_wf_velocity(gwf, px, py, pz, 1E-8);
	grid_wf_density(gwf, density);
	for (k = 0; k < px->nx * px->ny * px->nz; k++)
	  px->value[k] = px->value[k] * px->value[k] + py->value[k] * py->value[k] + pz->value[k] * pz->value[k];
	rgrid_product(px, px, density);
	rgrid_multiply(px, 0.5);
	sprintf(buf, "kinetic-" FMT_I, iter);
	rgrid_write_grid(buf, px);
      }
#endif
    }

    if(iter == 5) grid_fft_write_wisdom(NULL);

    grid_timer_start(&timer);

    /* Predict-Correct */
    cgrid_copy(gwfp->grid, gwf->grid);
    grid_real_to_complex_re(potential_store, ext_pot);
    dft_ot_potential(otf, potential_store, gwf);
    cgrid_add(potential_store, -mu0);
    grid_wf_propagate_predict(gwfp, potential_store, -I * TIME_STEP / GRID_AUTOFS);
    grid_add_real_to_complex_re(potential_store, ext_pot);
    dft_ot_potential(otf, potential_store, gwfp);
    cgrid_add(potential_store, -mu0);
    cgrid_multiply(potential_store, 0.5);  // Use (current + future) / 2
    grid_wf_propagate_correct(gwf, potential_store, -I * TIME_STEP / GRID_AUTOFS);
    // Chemical potential included - no need to normalize

#if defined(VORTEX) || defined(BOTH)
    zero_core(gwf->grid);
#endif
  }
  return 0;
}
