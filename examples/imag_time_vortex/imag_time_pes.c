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
#define PRESSURE 0.0

#define THREADS 0

#define IBEGIN 70.0
#define ISTEP 5.0
#define IEND 0.0

/* #define HE2STAR 1 */
#define HESTAR  1

/* #define ONSAGER */

#define HELIUM_MASS (4.002602 / GRID_AUTOAMU)
#define HBAR 1.0        /* au */

int main(int argc, char **argv) {

  dft_ot_functional *otf;
  cgrid *potential_store;
  rgrid *ext_pot, *orig_pot, *density, *px, *py, *pz;
  wf *gwf, *gwfp;
  char buf[512];
  INT iter, N;
  REAL energy, natoms, mu0, rho0, width, R;
  grid_timer timer;

  /* Normalization condition */
  if(argc != 2) {
    fprintf(stderr, "Usage: imag_time N\n");
    exit(1);
  }
  N = atoi(argv[1]);

  printf("N = " FMT_I "\n", N);

#ifdef USE_CUDA
  cuda_enable(1);  // enable CUDA ?
#endif

  /* Initialize threads & use wisdom */
  grid_set_fftw_flags(1);    // FFTW_MEASURE
  grid_threads_init(THREADS);
  grid_fft_read_wisdom(NULL);

  /* Allocate wave functions */
  if(!(gwf = grid_wf_alloc(NX, NY, NZ, STEP, DFT_HELIUM_MASS, WF_PERIODIC_BOUNDARY, WF_2ND_ORDER_PROPAGATOR, "gwf"))) {
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

  /* Allocate space for external potential */
  ext_pot = rgrid_clone(otf->density, "ext_pot");
  orig_pot = rgrid_clone(otf->density, "orig_pot");
  potential_store = cgrid_clone(gwf->grid, "potential_store");
  density = rgrid_clone(otf->density, "density");
  px = rgrid_clone(otf->density, "px");
  py = rgrid_clone(otf->density, "py");
  pz = rgrid_clone(otf->density, "pz");

#ifdef HE2STAR
  dft_common_potential_map(0, "he2-He.dat-spline", "he2-He.dat-spline", "he2-He.dat-spline", orig_pot);
#endif
#ifdef HESTAR
  dft_common_potential_map(0, "He-star-He.dat", "He-star-He.dat", "He-star-He.dat", orig_pot);
#endif

  if(N != 0) {
    width = 1.0 / 20.0;
    cgrid_map(gwf->grid, dft_common_cgaussian, (void *) &width);
  } else cgrid_constant(gwf->grid, SQRT(rho0));

#ifndef ONSAGER
  grid_wf_map(gwf, dft_initial_vortex_z_n1, NULL);
#endif

  for (R = IBEGIN; R >= IEND; R -= ISTEP) {
    rgrid_shift(ext_pot, orig_pot, R, 0.0, 0.0);
#ifdef ONSAGER
    rgrid_map(otf->density, &dft_initial_vortex_x, (void *) gwf);
    rgrid_sum(ext_pot, ext_pot, otf->density);
#endif
    // TODO: Do we need to enforce the vortex solution at each R?

    for (iter = 1; iter < ((R == IBEGIN)?10*MAXITER:MAXITER); iter++) {      
      
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

    }

    printf("Results for R = " FMT_R "\n", R);
    grid_wf_density(gwf, density);
    sprintf(buf, "output-" FMT_R, R);
    rgrid_write_grid(buf, density);
    dft_ot_energy_density(otf, density, gwf);
    rgrid_add_scaled_product(density, 1.0, otf->density, ext_pot);
    energy = grid_wf_energy(gwf, NULL) + rgrid_integral(density);
    natoms = grid_wf_norm(gwf);
    printf("Total energy is " FMT_R " K\n", energy * GRID_AUTOK);
    printf("Number of He atoms is " FMT_R ".\n", natoms);
    printf("Energy / atom is " FMT_R " K\n", (energy/natoms) * GRID_AUTOK);
    grid_wf_probability_flux(gwf, px, py, pz);
    sprintf(buf, "flux_x-" FMT_R, R);
    rgrid_write_grid(buf, px);
    sprintf(buf, "flux_y-" FMT_R, R);
    rgrid_write_grid(buf, py);
    sprintf(buf, "flux_z-" FMT_R, R);
    rgrid_write_grid(buf, pz);
    printf("PES " FMT_R " " FMT_R "\n", R, energy * GRID_AUTOK);
  }
  return 0;
}
