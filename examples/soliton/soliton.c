
/*
 * Create planar soliton in superfluid helium (propagating along X).
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

#define TS 5.0 /* fs */
#define NX 16384
#define NY 32
#define NZ 32
#define STEP 1.0
#define NTH 1000

#define PRESSURE (0.0 / GRID_AUTOBAR)

/* by +-2 x LAMBDA_C */
/* #define SMOOTH */

#define SOLITON_AMP (0.2)   /* 10% of bulk */
#define SOLITON_N  300       /* width (in N * LAMBDA_C) */
#define LAMBDA_C (3.58 / GRID_AUTOANG)

#define THREADS 0

#define HELIUM_MASS (4.002602 / GRID_AUTOAMU)

/* Francesco's soliton initial guess - plane along z*/
REAL soliton(void *asd, REAL x, REAL y, REAL z) {

  REAL tmp;

  if(FABS(x) > SOLITON_N * LAMBDA_C) return 1.0;

  tmp = SIN(M_PI * x / LAMBDA_C);

  return 1.0 + SOLITON_AMP * tmp * tmp;
}

int main(int argc, char **argv) {

  rgrid *rworkspace;
  cgrid *potential_store;
  wf *gwf, *gwfp;
  INT iter;
  REAL mu0, rho0;
  char buf[512];

  /* Setup DFT driver parameters (grid) */
  dft_driver_setup_grid(NX, NY, NZ, STEP, THREADS);
  /* Plain Orsay-Trento in imaginary time */
  dft_driver_setup_model(DFT_OT_PLAIN, DFT_DRIVER_REAL_TIME, 0.0);
  /* No absorbing boundary */
  dft_driver_setup_boundary_type(DFT_DRIVER_BOUNDARY_REGULAR, 0.0, 0.0, 0.0, 0.0);
  /* Normalization condition */
  dft_driver_setup_normalization(DFT_DRIVER_DONT_NORMALIZE, 0, 3.0, 10);

  /* Allocate space for wavefunctions (initialized to SQRT(rho0)) */
  gwf = dft_driver_alloc_wavefunction(HELIUM_MASS, "gwf"); /* helium wavefunction */
  gwfp = dft_driver_alloc_wavefunction(HELIUM_MASS, "gwfp");/* temp. wavefunction */

  /* Initialize the DFT driver */
  dft_driver_initialize(gwf);

  /* density */
  rho0 = dft_ot_bulk_density_pressurized(dft_driver_otf, PRESSURE);
  dft_driver_otf->rho0 = rho0;
  /* chemical potential */
  mu0 = dft_ot_bulk_chempot_pressurized(dft_driver_otf, PRESSURE);

  /* Allocate space for external potential */
  rworkspace = dft_driver_alloc_rgrid("rworkspace");
  potential_store = dft_driver_alloc_cgrid("cworkspace"); /* temporary storage */

  /* setup soliton (ext_pot is temp here) */
  rgrid_map(rworkspace, soliton, NULL);
  rgrid_multiply(rworkspace, rho0);
  rgrid_power(rworkspace, rworkspace, 0.5);
  cgrid_zero(gwf->grid);   /* copy rho to wf real part */
  grid_real_to_complex_re(gwf->grid, rworkspace);

  /* Generate the excited potential */

  for (iter = 0; iter < 800000; iter++) {
    dft_driver_propagate_predict(DFT_DRIVER_PROPAGATE_HELIUM, NULL, mu0, gwf, gwfp, potential_store, TS, iter);
    dft_driver_propagate_correct(DFT_DRIVER_PROPAGATE_HELIUM, NULL, mu0, gwf, gwfp, potential_store, TS, iter);
    if(!(iter % NTH)) {
      sprintf(buf, "soliton-" FMT_I, iter);
      grid_wf_density(gwf, rworkspace);
#ifdef SMOOTH
      rgrid_npoint_smooth(rworkspace2, rworkspace, (INT) (2.0 * LAMBDA_C / STEP));
      rgrid_write_grid(buf, rworkspace2);
#else
      rgrid_write_grid(buf, rworkspace);
#endif
    }
  }
  return 0;
}
