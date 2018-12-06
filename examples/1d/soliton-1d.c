
/*
 * Create planar soliton in superfluid helium (propagating along X).
 * 1-D version with Y & Z coordinates integrated over in the non-linear
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

#define TS 0.1 /* fs */
#define NZ (5*8*32768)
#define STEP 0.2
#define MAXITER 80000000
#define NTH 500000

#define PRESSURE (0.0 / GRID_AUTOBAR)

/* #define SMOOTH    /* by +-2 x LAMBDA_C */

#define SOLITON_AMP (0.05)   /* 10% of bulk */
#define SOLITON_N  200       /* width (in N * LAMBDA_C) */
#define LAMBDA_C (3.58 / GRID_AUTOANG)

#define THREADS 0

#define HELIUM_MASS (4.002602 / GRID_AUTOAMU)

extern void OT_INIT(rgrid *, rgrid *);
extern void OT_POT(rgrid *, rgrid *, rgrid *, rgrid *, rgrid *, rgrid *, rgrid *, rgrid *);

/* Francesco's soliton initial guess - plane along z*/
REAL soliton(void *asd, REAL x, REAL y, REAL z) {

  REAL tmp;

  if(FABS(z) > SOLITON_N * LAMBDA_C) return 1.0;

  tmp = SIN(M_PI * z / LAMBDA_C);

  return 1.0 + SOLITON_AMP * tmp * tmp;
}

int main(int argc, char **argv) {

  rgrid *rworkspace, *rworkspace2, *rworkspace3, *lj_tf, *rd_tf, *density_tf, *spave_tf, *ot_pot;
  cgrid *potential_store;
  wf *gwf, *gwfp;
  INT iter;
  REAL mu0, rho0;
  char buf[512];

  dft_driver_init_ot = 0;   /* We allocate the grids manually */

#ifdef USE_CUDA
  cuda_enable(1);
#endif

  /* Setup DFT driver parameters (grid) */
  dft_driver_setup_grid(1, 1, NZ, STEP, THREADS);
// NOTE: This does nothing - external potential from ot-1d.c is used instead
  /* Plain Orsay-Trento in imaginary time */
  dft_driver_setup_model(DFT_OT_PLAIN, DFT_DRIVER_REAL_TIME, 0.0);
//
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
  rworkspace = dft_driver_alloc_rgrid("rworkspace");
  rworkspace2 = dft_driver_alloc_rgrid("rworkspace2");
  rworkspace3 = dft_driver_alloc_rgrid("rworkspace3");
  potential_store = dft_driver_alloc_cgrid("cworkspace"); /* temporary storage */

  /* Allocate space for wavefunctions (initialized to SQRT(rho0)) */
  gwf = dft_driver_alloc_wavefunction(HELIUM_MASS, "gwf"); /* helium wavefunction */
  gwfp = dft_driver_alloc_wavefunction(HELIUM_MASS, "gwfp");/* temp. wavefunction */
  /* setup soliton (ext_pot is temp here) */
  rgrid_map(rworkspace, soliton, NULL);
  rgrid_multiply(rworkspace, rho0);
  rgrid_power(rworkspace, rworkspace, 0.5);
  cgrid_zero(gwf->grid);   /* copy rho to wf real part */
  grid_real_to_complex_re(gwf->grid, rworkspace);

  lj_tf = dft_driver_get_workspace(1, 1);
  rd_tf = dft_driver_get_workspace(2, 1);
  density_tf = dft_driver_get_workspace(3, 1);
  spave_tf = dft_driver_get_workspace(4, 1);
  ot_pot = dft_driver_get_workspace(5, 1);  
  OT_INIT(lj_tf, rd_tf);

  for (iter = 0; iter < MAXITER; iter++) {

    grid_wf_density(gwf, rworkspace);
    OT_POT(ot_pot, rworkspace, density_tf, rworkspace2, spave_tf, lj_tf, rd_tf, rworkspace3);

    dft_driver_propagate_predict(DFT_DRIVER_PROPAGATE_OTHER, ot_pot, mu0, gwf, gwfp, potential_store, TS, iter);

    grid_wf_density(gwfp, rworkspace);
    OT_POT(ot_pot, rworkspace, density_tf, rworkspace2, spave_tf, lj_tf, rd_tf, rworkspace3);

    dft_driver_propagate_correct(DFT_DRIVER_PROPAGATE_OTHER, ot_pot, mu0, gwf, gwfp, potential_store, TS, iter);
    if(!(iter % NTH)) {
      sprintf(buf, "soliton-" FMT_I, iter);
      grid_wf_density(gwf, rworkspace);
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
