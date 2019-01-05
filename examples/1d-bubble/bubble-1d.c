
/*
 * "One dimensional bubble" propagating in superfluid helium (propagating along Z).
 * 1-D version with X & Y coordinates integrated over in the non-linear
 * potential.
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

#define TS 1.0 /* fs */
#define NZ (32768)
#define STEP 0.2
#define IITER 20000
#define MAXITER 80000000
#define NTH 1000
#define VZ (45.0 / GRID_AUTOMPS)

#define PRESSURE (0.0 / GRID_AUTOBAR)
#define THREADS 0

#define HELIUM_MASS (4.002602 / GRID_AUTOAMU)

/* Bubble parameters using exponential repulsion (approx. electron bubble) - RADD = 19.0 */
#define A0 (3.8003E5 / GRID_AUTOK)
#define A1 (1.6245 * GRID_AUTOANG)
#define A2 0.0
#define A3 0.0
#define A4 0.0
#define A5 0.0
#define RMIN 2.0
#define RADD 6.0
#define BUBBLE_RADIUS (17.0 / GRID_AUTOANG)

REAL rho0;

REAL complex bubble_init(void *NA, REAL x, REAL y, REAL z) {

  if(FABS(z) < BUBBLE_RADIUS) return 0.0;
  return SQRT(rho0);
}

REAL round_veloc(REAL veloc) {   // Round to fit the simulation box

  INT n;
  REAL v;

  n = (INT) (0.5 + (NZ * STEP * HELIUM_MASS * veloc) / (HBAR * 2.0 * M_PI));
  v = ((REAL) n) * HBAR * 2.0 * M_PI / (NZ * STEP * HELIUM_MASS);
  fprintf(stderr, "Requested velocity = %le m/s.\n", veloc * GRID_AUTOMPS);
  fprintf(stderr, "Nearest velocity compatible with PBC = %le m/s.\n", v * GRID_AUTOMPS);
  return v;
}

REAL momentum(REAL vz) {

  return HELIUM_MASS * vz / HBAR;
}

/* Bubble potential (centered at origin, z = 0) */
REAL bubble(void *asd, REAL x, REAL y, REAL z) {

  REAL r, r2, r4, r6, r8, r10;

  r = FABS(z);
  r -= RADD;
  if(r < RMIN) r = RMIN;

  r2 = r * r;
  r4 = r2 * r2;
  r6 = r4 * r2;
  r8 = r6 * r2;
  r10 = r8 * r2;
  return A0 * EXP(-A1 * r) - A2 / r4 - A3 / r6 - A4 / r8 - A5 / r10;
}

int main(int argc, char **argv) {

  rgrid *density, *ext_pot;
  cgrid *potential_store;
  wf *gwf, *gwfp;
  INT iter;
  REAL mu0, vz, kz;
  char buf[512];
  REAL complex tstep;

#ifdef USE_CUDA
  cuda_enable(1);
#endif

  /* Setup DFT driver parameters (grid) */
  dft_driver_setup_grid(1, 1, NZ, STEP, THREADS);
  /* Plain Orsay-Trento in imaginary time */
  dft_driver_setup_model(DFT_OT_PLAIN | DFT_OT_BACKFLOW | DFT_OT_KC, DFT_DRIVER_USER_TIME, 0.0);
  /* No absorbing boundary */
  dft_driver_setup_boundary_type(DFT_DRIVER_BOUNDARY_REGULAR, 0.0, 0.0, 0.0, 0.0);
  /* Normalization condition */
  dft_driver_setup_normalization(DFT_DRIVER_DONT_NORMALIZE, 0, 3.0, 10);
  dft_driver_temp_disable_other_normalization = 1; // Do not normalize - we are using OTHER!!!

  /* Initialize the DFT driver */
  dft_driver_initialize();

  /* density */
  rho0 = dft_ot_bulk_density_pressurized(dft_driver_otf, PRESSURE);
  dft_driver_otf->rho0 = rho0;
  /* chemical potential */
  mu0 = dft_ot_bulk_chempot_pressurized(dft_driver_otf, PRESSURE);
  printf("mu0 = " FMT_R " K/atom, rho0 = " FMT_R " Angs^-3.\n", mu0 * GRID_AUTOK, rho0 / (GRID_AUTOANG * GRID_AUTOANG * GRID_AUTOANG));

  vz = round_veloc(VZ);
  printf("VZ = " FMT_R " m/s\n", vz * GRID_AUTOMPS);
  kz = momentum(VZ);
  dft_driver_setup_momentum(0.0, 0.0, kz);

  /* Allocate space for external potential */
  density = dft_driver_alloc_rgrid("rworkspace");
  potential_store = dft_driver_alloc_cgrid("cworkspace"); /* temporary storage */
  ext_pot = dft_driver_alloc_rgrid("ext_pot");

  /* Allocate space for wavefunctions (initialized to SQRT(rho0)) */
  gwf = dft_driver_alloc_wavefunction(HELIUM_MASS, "gwf"); /* helium wavefunction */
  gwfp = dft_driver_alloc_wavefunction(HELIUM_MASS, "gwfp");/* temp. wavefunction */

  /* set up external potential */
  rgrid_map(ext_pot, bubble, NULL);

  /* set up initial density */
  cgrid_map(gwf->grid, bubble_init, NULL);

  for (iter = 0; iter < MAXITER; iter++) {

    if(!(iter % NTH)) {
      sprintf(buf, "bubble-" FMT_I, iter);
      grid_wf_density(gwf, density);
      dft_driver_write_density(density, buf);
    }

    if(iter < IITER) tstep = -I * TS;
    else tstep = TS;

    dft_driver_propagate_predict(DFT_DRIVER_PROPAGATE_HELIUM, ext_pot, mu0, gwf, gwfp, potential_store, tstep, iter);
    dft_driver_propagate_correct(DFT_DRIVER_PROPAGATE_HELIUM, ext_pot, mu0, gwf, gwfp, potential_store, tstep, iter);
  }
  return 0;
}
