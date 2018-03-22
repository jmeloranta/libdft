
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
#define NX 256
#define NY 128
#define NZ 128
#define STEP 2.0
#define NTH 100

#define PRESSURE (1.0 / GRID_AUTOBAR)

/* #define SMOOTH    /* by +-2 x LAMBDA_C */

#define SOLITON_AMP (0.02)   /* 10% of bulk */
#define SOLITON_N  5       /* width (in N * LAMBDA_C) */
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

  rgrid3d *ext_pot, *rworkspace;
  cgrid3d *potential_store;
  wf3d *gwf, *gwfp;
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

  /* Initialize the DFT driver */
  dft_driver_initialize();

  /* density */
  rho0 = dft_ot_bulk_density_pressurized(dft_driver_otf, PRESSURE);
  dft_driver_otf->rho0 = rho0;
  /* chemical potential */
  mu0 = dft_ot_bulk_chempot_pressurized(dft_driver_otf, PRESSURE);

  /* Allocate space for external potential */
  ext_pot = dft_driver_alloc_rgrid();
  rworkspace = dft_driver_alloc_rgrid();
  potential_store = dft_driver_alloc_cgrid(); /* temporary storage */

  /* Allocate space for wavefunctions (initialized to SQRT(rho0)) */
  gwf = dft_driver_alloc_wavefunction(HELIUM_MASS); /* helium wavefunction */
  gwfp = dft_driver_alloc_wavefunction(HELIUM_MASS);/* temp. wavefunction */
  /* setup soliton (ext_pot is temp here) */
  rgrid3d_map(ext_pot, soliton, NULL);
  rgrid3d_multiply(ext_pot, rho0);
  rgrid3d_power(ext_pot, ext_pot, 0.5);
  cgrid3d_zero(gwf->grid);   /* copy rho to wf real part */
  grid3d_real_to_complex_re(gwf->grid, ext_pot);

  /* Generate the excited potential */
  rgrid3d_constant(ext_pot, -mu0); /* Add the chemical potential */

  for (iter = 0; iter < 80000; iter++) {
    dft_driver_propagate_predict(DFT_DRIVER_PROPAGATE_HELIUM, ext_pot, gwf, gwfp, potential_store, TS, iter);
    dft_driver_propagate_correct(DFT_DRIVER_PROPAGATE_HELIUM, ext_pot, gwf, gwfp, potential_store, TS, iter);
    if(!(iter % NTH)) {
      sprintf(buf, "soliton-" FMT_I, iter);
      grid3d_wf_density(gwf, rworkspace);
#ifdef SMOOTH
      dft_driver_npoint_smooth(rworkspace2, rworkspace, (INT) (2.0 * LAMBDA_C / STEP));
      dft_driver_write_density(rworkspace2, buf);
#else
      dft_driver_write_density(rworkspace, buf);
#endif
    }
  }
  return 0;
}
