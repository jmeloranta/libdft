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
#include <stdlib.h>

#define NX 256
#define NY 256
#define NZ 256
#define STEP 1.0
#define TS 0.01

#define PRESSURE 0.0

#define MAXITER 10000000
#define NTH 10

#define THREADS 0

#define TEMP 3.7

#define BOSE

REAL chem_pot = 0.0, mu0, natoms, volume, rho0;

#define DMU 1.E-3
#define END 1.0

#define BETA (1.0 / (GRID_AUKB * TEMP))

REAL distribution(REAL energy, REAL chem_pot) {

#ifdef BOSE
  return 1.0 / (EXP((energy - chem_pot) * BETA) - 1.0); 
#else
  return natoms * EXP(-(energy - chem_pot) * BETA);
#endif
}

REAL calc_chem_pot(cgrid *grid, REAL natoms) {  // Choose chempot such that sumer over all <n_i> = N.

  REAL chem_pot = -1e-2 / GRID_AUTOK, f, tmp;
  INT i, n = grid->nx * grid->ny * grid->nz;

  printf("Initial chem_pot " FMT_R "\n", chem_pot);
  do {
    f = 0.0;
    for (i = 0; i < n; i++)
      f += distribution(((REAL) grid->value[i]), chem_pot);
    printf("f = " FMT_R " natoms = " FMT_R "\n", f, natoms);
    f -= natoms;
    if(f > 0.0) tmp = fabs(chem_pot); else tmp = -fabs(chem_pot);
    f *= f;
    chem_pot -= tmp * DMU;
    printf("Current chemical potential = " FMT_R " K.\n", chem_pot * GRID_AUTOK);
  } while(fabs(f) > END);
  printf("Optimized chemical potential = " FMT_R " K.\n", chem_pot * GRID_AUTOK);
  return chem_pot;
}

int main(int argc, char **argv) {

  dft_ot_functional *otf;
  cgrid *potential_store;
  rgrid *density;
  wf *gwf, *gwfp;
  INT iter, i, j, k;
  REAL kx, ky, kz, chem_pot;
  REAL temp, qp, cl;
  FILE *fp;

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
  if(!(otf = dft_ot_alloc(DFT_OT_PLAIN | DFT_OT_HD | DFT_OT_KC | DFT_OT_BACKFLOW, gwf, DFT_MIN_SUBSTEPS, DFT_MAX_SUBSTEPS))) {
    fprintf(stderr, "Cannot allocate otf.\n");
    exit(1);
  }
  rho0 = dft_ot_bulk_density_pressurized(otf, PRESSURE);
  mu0 = dft_ot_bulk_chempot_pressurized(otf, PRESSURE);
  printf("mu0 = " FMT_R " K/atom, rho0 = " FMT_R " Angs^-3.\n", mu0 * GRID_AUTOK, rho0 / (GRID_AUTOANG * GRID_AUTOANG * GRID_AUTOANG));

  /* Allocate space for external potential */
  potential_store = cgrid_clone(gwf->grid, "potential_store");
  density = rgrid_clone(otf->density, "density");

  if(argc == 1) {
    volume = (((REAL) NX) * STEP) * (((REAL) NY) * STEP) * (((REAL) NZ) * STEP);
    natoms = rho0 * volume;
    cgrid_zero(gwf->grid);
    printf("Build energy distribution.\n");fflush(stdout);
    cuda_remove_block(gwf->grid->value, 1);
    srand48(time(0));
#pragma omp parallel for firstprivate(stdout,rho0,otf,gwf) private(i, j, k, kx, ky, kz) default(none)
    for(i = 0; i < NX; i++) {
      if(i < NX / 2) kx = ((REAL) i) * 2.0 * M_PI / (((REAL) NX) * STEP);
      else kx = ((REAL) (i - NX)) * 2.0 * M_PI / (((REAL) NX) * STEP);
//      printf("Starting block kx = " FMT_R "\n", kx);fflush(stdout);
      for (j = 0; j < NY; j++) {
        if(j < NY / 2) ky = ((REAL) j) * 2.0 * M_PI / (((REAL) NY) * STEP);
        else ky = ((REAL) (j - NY)) * 2.0 * M_PI / (((REAL) NY) * STEP);
        for(k = 0; k < NZ; k++) {
          REAL tot_k, ene;
          INT idx = (i * NY + j) * NZ + k;
          if(k < NZ / 2) kz = ((REAL) k) * 2.0 * M_PI / (((REAL) NZ) * STEP);
          else kz = ((REAL) (k - NZ)) * 2.0 * M_PI / (((REAL) NZ) * STEP);
          tot_k = SQRT(kx * kx + ky * ky + kz * kz);
          if(tot_k < 2.5 * GRID_AUTOANG) 
            ene = dft_ot_bulk_dispersion(otf, &tot_k, rho0);
          else
            ene = 20.0 / GRID_AUTOK;
//          printf("k = " FMT_R " Angs^-1, E(k) = " FMT_R " K\n", tot_k / GRID_AUTOANG, ene * GRID_AUTOK);
          gwf->grid->value[idx] = ene;
        }
      }
    }
    chem_pot = calc_chem_pot(gwf->grid, natoms);

    printf("Form thermal distribution.\n");fflush(stdout);
    cuda_remove_block(gwf->grid->value, 1);
    for(i = 0; i < NX; i++)
      for (j = 0; j < NY; j++)
        for(k = 0; k < NZ; k++) {
          INT idx = (i * NY + j) * NZ + k;
          REAL amp;
          amp = distribution((REAL) gwf->grid->value[idx], chem_pot);
          gwf->grid->value[idx] = SQRT(amp * rho0 / natoms) * CEXP(I * 2.0 * M_PI * drand48());
        }
    cgrid_inverse_fft(gwf->grid);
    if(!(fp = fopen("thermal.grd", "w"))) {
      fprintf(stderr, "Can't open thermal.grd for writing.\n");
      exit(1);
    }
    cgrid_write(gwf->grid, fp);
    fclose(fp);
  } else {
    if(!(fp = fopen(argv[1], "r"))) {
      fprintf(stderr, "Can't open thermal file for reading.\n");
      exit(1);
    }
    cgrid_read(gwf->grid, fp);
    fclose(fp);
  }
  printf("Box natoms    = " FMT_R "\n", natoms);
  printf("Actual natoms = " FMT_R "\n", grid_wf_norm(gwf));

  for (iter = 0; iter < MAXITER; iter++) {

    if(!(iter % NTH)) {
      REAL pot;

      qp = grid_wf_kinetic_energy_qp(gwf, otf->workspace1, otf->workspace2);
      if(otf->model & DFT_OT_KC) { // kinetic energy correlation is part of quantum pressure
        grid_wf_density(gwf, otf->density);
        dft_ot_energy_density_kc(otf, density, gwf, otf->density);
        qp += rgrid_integral(density);
      }
      if(otf->model & DFT_OT_BACKFLOW) { // backflow is part of classical kinetic energy
        grid_wf_density(gwf, otf->density);
        dft_ot_energy_density_bf(otf, density, gwf, otf->density);
        cl = rgrid_integral(density);
      } else cl = 0.0;

      temp = grid_wf_ideal_gas_temperature(gwf, cl, otf->workspace1, otf->workspace2);

      rgrid_zero(density);
      dft_ot_energy_density(otf, density, gwf);
      pot = rgrid_integral(density);
      printf("# of atoms   = " FMT_R ".\n", grid_wf_norm(gwf));
      printf("Total E/FFT  = " FMT_R " K.\n", (grid_wf_energy_fft(gwf, NULL) + pot) * GRID_AUTOK);
      printf("Total E/CN   = " FMT_R " K.\n", (grid_wf_energy_cn(gwf, NULL) + pot) * GRID_AUTOK);
      printf("QP energy    = " FMT_R " K.\n", qp * GRID_AUTOK);
      printf("Classical E. = " FMT_R " K.\n", (grid_wf_kinetic_energy_cn(gwf) - qp + cl) * GRID_AUTOK);
      printf("T            = " FMT_R " K.\n", temp);
      printf("Circulation  = " FMT_R ".\n", grid_wf_circulation(gwf, 1.0, otf->density, otf->workspace1, otf->workspace2, otf->workspace3));
      fflush(stdout);
    }

    if(iter == 5) grid_fft_write_wisdom(NULL);

    /* Predict-Correct */
    cgrid_zero(potential_store);
    dft_ot_potential(otf, potential_store, gwf);
    cgrid_add(potential_store, -mu0);
    grid_wf_propagate_predict(gwf, gwfp, potential_store, TS / GRID_AUTOFS);
    dft_ot_potential(otf, potential_store, gwfp);
    cgrid_add(potential_store, -mu0);
    cgrid_multiply(potential_store, 0.5);  // Use (current + future) / 2
    grid_wf_propagate_correct(gwf, potential_store, TS / GRID_AUTOFS);

    if(!(iter % (50*NTH))) {
      char buf[512];
      sprintf(buf, "output-" FMT_I, iter);
      grid_wf_density(gwf, density);
      rgrid_write_grid(buf, density);
      grid_wf_probability_flux(gwf, otf->workspace1, otf->workspace2, otf->workspace3);
      rgrid_abs_rot(density, otf->workspace1, otf->workspace2, otf->workspace3);
      rgrid_abs_power(density, density, 1.0);
      sprintf(buf, "circulation-" FMT_I, iter);
      rgrid_write_grid(buf, density);
    }
  }

  return 0;
}
