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
#include <stdlib.h>

#include "dispersion.h"

#define NX 256
#define NY 256
#define NZ 256
#define STEP 2.0

#define THREADS 0

#define DFT_HELIUM_MASS (4.002602 / GRID_AUTOAMU) /* Helium mass in atomic units */
#define RHO0 (0.0218360 * GRID_AUTOANG * GRID_AUTOANG * GRID_AUTOANG)

#define Q 1.0
#define DMU 1.E-1
#define END 0.1
#define BETA (1.0 / (GRID_AUKB * TEMP))
#define KUL 3.6  // In Angs^-1

REAL TEMP = 0.0;

REAL qexp(REAL q, REAL x) {

  if(q == 1.0) return EXP(x);
  return POW(1.0 + (1.0 - q) * x, 1.0 / (1.0 - q));
}

REAL distribution(REAL energy, REAL chem_pot) {

  return 1.0 / (POW(qexp(Q, BETA * chem_pot) * qexp(Q, -BETA * energy), -Q) - 1.0);
}

REAL calc_chem_pot(cgrid *grid, REAL natoms) {  // Choose chempot such that sumer over all <n_i> = N.

  REAL chem_pot = -1.0 / GRID_AUTOK, f, tmp = 0.0, scale = 1.0;
  INT i, n = grid->nx * grid->ny * grid->nz;

  printf("Initial chem_pot " FMT_R "\n", chem_pot);
  do {
    f = 0.0;
#pragma omp parallel for firstprivate(grid,chem_pot,n) private(i) default(none) reduction(+:f)
    for (i = 0; i < n; i++)
      if(CREAL(grid->value[i]) >= 0.0) f += distribution(CREAL(grid->value[i]), chem_pot);
    printf("f = " FMT_R " natoms = " FMT_R "\n", f, natoms);
    f -= natoms;
    if(f > 0.0) {
      if(tmp < 0.0) scale /= 2.0;
      tmp = fabs(chem_pot);
    } else {
      if(tmp > 0.0) scale /= 2.0;
      tmp = -fabs(chem_pot);
    }
    f *= f;
    chem_pot -= tmp * DMU * scale;
    printf("Current chemical potential = " FMT_R " K.\n", chem_pot * GRID_AUTOK);
  } while(fabs(f) > END);
  printf("Optimized chemical potential = " FMT_R " K.\n", chem_pot * GRID_AUTOK);
  return chem_pot;
}

REAL dispersion(REAL tot_k) {

  INT i;
  REAL val, x;

#if 1
#define HBAR 1.0
  return (238.8 / GRID_AUTOMPS * HBAR * tot_k); // phonon branch only
#endif
  tot_k /= GRID_AUTOANG;
  if(tot_k > KUL) return -1.0;
  i = (INT) (tot_k / DISP_K);
  if(i > DISP_PTS-2) return disp_e[DISP_PTS-1] / GRID_AUTOK;
  x = (tot_k - DISP_K * (REAL) i) / DISP_K;
  val = disp_e[i] * (1.0 - x) + x * disp_e[i+1];
  return val / GRID_AUTOK;
}

int main(int argc, char **argv) {

  cgrid *gwf;
  INT i, j, k;
  REAL kx, ky, kz, chem_pot, entropy, norm, energy;
  REAL tot_k, amp, natoms, volume, max_e;
  INT idx;

#ifdef USE_CUDA
  cuda_enable(0);
#endif

  TEMP = atof(argv[1]);
  printf("Temperature = " FMT_R " K.\n", TEMP);

  /* Initialize threads & use wisdom */
  grid_set_fftw_flags(1);    // FFTW_MEASURE
  grid_threads_init(THREADS);
  grid_fft_read_wisdom(NULL);

  /* Allocate wave functions */
  if(!(gwf = cgrid_alloc(NX, NY, NZ, STEP, CGRID_PERIODIC_BOUNDARY, NULL, "gwf"))) {
    fprintf(stderr, "Cannot allocate gwf.\n");
    exit(1);
  }

  volume = (((REAL) NX) * STEP) * (((REAL) NY) * STEP) * (((REAL) NZ) * STEP);
  natoms = RHO0 * volume;
  cgrid_zero(gwf);
  printf("Build energy distribution.\n");fflush(stdout);
  // corners have longer vectors
  printf("k vector extent (1/Angs^-1): +-" FMT_R "\n", (((REAL) (NX/2)) * 2.0 * M_PI / (((REAL) NX) * STEP)) / GRID_AUTOANG);
  printf("k vector step (1/Angs^-1): " FMT_R "\n", (2.0 * M_PI / (((REAL) NX) * STEP)) / GRID_AUTOANG);
#pragma omp parallel for firstprivate(gwf) private(i, j, k, kx, ky, kz, idx, tot_k) default(none)
  for(i = 0; i < NX; i++) {
    if(i < NX / 2) kx = ((REAL) i) * 2.0 * M_PI / (((REAL) NX) * STEP);
    else kx = ((REAL) (i - NX)) * 2.0 * M_PI / (((REAL) NX) * STEP);
    for (j = 0; j < NY; j++) {
      if(j < NY / 2) ky = ((REAL) j) * 2.0 * M_PI / (((REAL) NY) * STEP);
      else ky = ((REAL) (j - NY)) * 2.0 * M_PI / (((REAL) NY) * STEP);
      for(k = 0; k < NZ; k++) {
        if(k < NZ / 2) kz = ((REAL) k) * 2.0 * M_PI / (((REAL) NZ) * STEP);
        else kz = ((REAL) (k - NZ)) * 2.0 * M_PI / (((REAL) NZ) * STEP);
        idx = (i * NY + j) * NZ + k;
        tot_k = SQRT(kx * kx + ky * ky + kz * kz);
        gwf->value[idx] = dispersion(tot_k);
      }
    }
  }
  chem_pot = calc_chem_pot(gwf, natoms);

  printf("Form thermal distribution.\n");fflush(stdout);
  entropy = 0.0;
  norm = 0.0;
  energy = 0.0;
  max_e = 0.0;
//#pragma omp parallel for firstprivate(gwf,chem_pot,natoms) private(amp, idx, i, j, k) default(none) reduction(+:norm) reduction(+:entropy) reduction(+:energy)
  for(i = 0; i < NX; i++)
    for (j = 0; j < NY; j++)
      for(k = 0; k < NZ; k++) {
        idx = (i * NY + j) * NZ + k;
        if(CREAL(gwf->value[idx]) >= 0.0) {
          if(CREAL(gwf->value[idx]) > max_e) max_e = CREAL(gwf->value[idx]);
          amp = distribution(CREAL(gwf->value[idx]), chem_pot); // # of atoms on this state
          if(Q != 1.0) entropy += POW(amp / natoms, Q);
          energy += CREAL(gwf->value[idx]) * amp;
          gwf->value[idx] = amp / natoms;
          norm += CREAL(gwf->value[idx]);
        } else gwf->value[idx] = 0.0;
      }
  printf("Max E = " FMT_R " K.\n", max_e * GRID_AUTOK);
  printf("Norm = " FMT_R "\n", norm);
  printf("n0 / N = " FMT_R "\n", CREAL(gwf->value[0]));
  printf("n_ex / N = " FMT_R "\n", 1.0 - CREAL(gwf->value[0]));
  if(Q != 1.0) {
    entropy = GRID_AUTOJ * (GRID_AVOGADRO / natoms) * GRID_AUKB * (entropy - 1.0) / (1.0 - Q); // J / (mol K)
    printf("Entropy = " FMT_R " J / (g K).\n", entropy / 4.00); // 4He = 4 g/mol
  }
  printf("Total energy = " FMT_R " J / mol.\n", (energy / natoms) * GRID_AUTOJ * GRID_AVOGADRO);
  printf("Natoms    = " FMT_R "\n", natoms);
  return 0;
}
