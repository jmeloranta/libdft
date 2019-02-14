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

#define NX 128
#define NY 128
#define NZ 128
#define STEP 0.5
#define TS 1.0

#define PRESSURE 0.0

#define MAXITER 10000000
#define NTH 1000

#define THREADS 0

int main(int argc, char **argv) {

  dft_ot_functional *otf;
  cgrid *potential_store;
  rgrid *density;
  wf *gwf, *gwfp;
  INT iter;
  REAL energy, natoms, mu0, rho0;
  grid_timer timer;

#ifdef USE_CUDA
  cuda_enable(1);
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
  if(!(otf = dft_ot_alloc(DFT_OT_PLAIN | DFT_OT_KC | DFT_OT_BACKFLOW, gwf, DFT_MIN_SUBSTEPS, DFT_MAX_SUBSTEPS))) {
    fprintf(stderr, "Cannot allocate otf.\n");
    exit(1);
  }
  rho0 = dft_ot_bulk_density_pressurized(otf, PRESSURE);
  mu0 = dft_ot_bulk_chempot_pressurized(otf, PRESSURE);
  printf("mu0 = " FMT_R " K/atom, rho0 = " FMT_R " Angs^-3.\n", mu0 * GRID_AUTOK, rho0 / (GRID_AUTOANG * GRID_AUTOANG * GRID_AUTOANG));

  /* Allocate space for external potential */
  potential_store = cgrid_clone(gwf->grid, "potential_store");
  density = rgrid_clone(otf->density, "density");

  grid_wf_constant(gwf, SQRT(rho0));
  cgrid_random(gwf->grid, 9E-3);

  /* Run 200 iterations using imaginary time (10 fs time step) */
  REAL temp, itime = 0.0;

  gwf->norm = grid_wf_norm(gwf);
  for (iter = 0; iter < MAXITER; iter++) {

    temp = grid_wf_ideal_gas_temperature(gwf, otf->workspace1, otf->workspace2);
//    if(temp > TEMP && itime < TS/5.0) itime += 1E-2 * (temp - TEMP);
//    else itime = 0.0;
//    if(itime > 0.0) grid_wf_normalize(gwf);

    if(!(iter % NTH)) {
      dft_ot_energy_density(otf, density, gwf);
      printf("Total E      = " FMT_R " K.\n", grid_wf_energy_fft(gwf, density) * GRID_AUTOK);
      printf("Total E/CN   = " FMT_R " K.\n", grid_wf_energy_cn(gwf, density) * GRID_AUTOK);
      printf("Total K.E.   = " FMT_R " K.\n", grid_wf_kinetic_energy(gwf) * GRID_AUTOK);
      printf("QP energy    = " FMT_R " K.\n", grid_wf_kinetic_energy_qp(gwf, otf->workspace1, otf->workspace2) * GRID_AUTOK);
      printf("Flow energy  = " FMT_R " K.\n", grid_wf_kinetic_energy_flow(gwf, otf->workspace1, otf->workspace2) * GRID_AUTOK);
      printf("Itime        = " FMT_R " fs.\n", itime);
      printf("T            = " FMT_R " K.\n", temp);
      fflush(stdout);
    }

    if(iter == 5) grid_fft_write_wisdom(NULL);

//    grid_timer_start(&timer);

    cgrid_zero(potential_store);
    dft_ot_potential(otf, potential_store, gwf);
    cgrid_add(potential_store, -mu0);
    itime = 0.0;

    /* Predict-Correct */
    cgrid_zero(potential_store);
    dft_ot_potential(otf, potential_store, gwf);
    cgrid_add(potential_store, -mu0);
    grid_wf_propagate_predict(gwf, gwfp, potential_store, TS / GRID_AUTOFS);
    dft_ot_potential(otf, potential_store, gwfp);
    cgrid_add(potential_store, -mu0);
    cgrid_multiply(potential_store, 0.5);  // Use (current + future) / 2
    grid_wf_propagate_correct(gwf, potential_store, TS / GRID_AUTOFS);

//    grid_wf_propagate(gwf, potential_store, (-I * itime * TS + TS) / GRID_AUTOFS);

//    printf("Iteration " FMT_I " - Wall clock time = " FMT_R " seconds.\n", iter, grid_timer_wall_clock_time(&timer));

    if(!(iter % NTH)) {
      char buf[512];
      sprintf(buf, "output-" FMT_I, iter);
      grid_wf_density(gwf, density);
      rgrid_write_grid(buf, density);
      sprintf(buf, "wf-output-" FMT_I, iter);
      cgrid_write_grid(buf, gwf->grid);
      dft_ot_energy_density(otf, density, gwf);
      energy = grid_wf_energy(gwf, NULL) + rgrid_integral(density);
      natoms = grid_wf_norm(gwf);
//      printf("Total energy is " FMT_R " K\n", energy * GRID_AUTOK);
//      printf("Number of He atoms is " FMT_R ".\n", natoms);
//      printf("Energy / atom is " FMT_R " K\n", (energy/natoms) * GRID_AUTOK);
//      fflush(stdout);
    }
  }
  /* At this point gwf contains the converged wavefunction */
  grid_wf_density(gwf, density);
  rgrid_write_grid("output", density);
  return 0;
}
