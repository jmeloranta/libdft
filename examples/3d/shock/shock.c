/*
 * Impurity atom in superfluid helium (no zero-point).
 * Optimize the liquid structure around a given initial
 * potential and then run dynamics on a given final potential.
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

#define MAXITER 160000
#define TS 10.0 /* fs */
#define OUTPUT 1000

#define PRESSURE (0.0 / GRID_AUTOBAR)   /* External pressure in bar */

#define HELIUM_MASS (4.002602 / GRID_AUTOAMU)

#define FUNC (DFT_OT_PLAIN)
//#define FUNC (DFT_OT_PLAIN | DFT_OT_KC)
#define NX 64
#define NY 64
#define NZ 4096
#define STEP 1.0

#define WIDTH (10.0 / 0.529)
#define AMP (0.0219 * 0.529 * 0.529 * 0.529)
#define SLAB 50.0

double complex gauss(void *arg, double x, double y, double z) {

  double inv_width = *((double *) arg), c = 0.0;
  double norm = 0.5 * M_2_SQRTPI * inv_width;

  // remove norm *  -- AMP * rho0 gives directly the amplitude
  //  if(z > SLAB) c = 10.0;
  //else if(z < -SLAB) c = -10.0;
  //else return 1.0;
  return norm * cexp(-(z - c) * (z - c) * inv_width * inv_width);
}

int main(int argc, char **argv) {

  rgrid3d *ext_pot, *rworkspace;
  cgrid3d *potential_store;
  wf3d *gwf, *gwfp;
  long iter;
  double inv_width, rho0, mu0;
  char buf[512];

  fprintf(stderr, "Time step = %le fs.\n", TS);
  fprintf(stderr, "Gaussian width = %le, amplitude = %le.\n", WIDTH, AMP);
  /* Setup DFT driver parameters (256 x 256 x 256 grid) */
  dft_driver_setup_grid(NX, NY, NZ, STEP /* Bohr */, 0 /* threads */);
  /* Plain Orsay-Trento in imaginary time */
  dft_driver_setup_model(FUNC, DFT_DRIVER_REAL_TIME, 0.0);
  /* No absorbing boundary */
  dft_driver_setup_boundary_type(DFT_DRIVER_BOUNDARY_REGULAR, 0.0, 0.0);
  /* Normalization condition */
  dft_driver_setup_normalization(DFT_DRIVER_DONT_NORMALIZE, 0, 0.0, 0);

  /* Initialize the DFT driver */
  dft_driver_initialize();

  /* Allocate space for external potential */
  ext_pot = dft_driver_alloc_rgrid();
  rworkspace = dft_driver_alloc_rgrid();
  potential_store = dft_driver_alloc_cgrid(); /* temporary storage */
  /* Read initial external potential from file */

  /* Allocate space for wavefunctions (initialized to sqrt(rho0)) */
  gwf = dft_driver_alloc_wavefunction(HELIUM_MASS); /* helium wavefunction */
  gwfp = dft_driver_alloc_wavefunction(HELIUM_MASS);/* temp. wavefunction */
  rho0 = dft_driver_otf->rho0 = dft_ot_bulk_density_pressurized(dft_driver_otf, PRESSURE);
  mu0  = dft_ot_bulk_chempot_pressurized(dft_driver_otf, PRESSURE);
  rgrid3d_zero(ext_pot);
  rgrid3d_add(ext_pot, -mu0); /* Add the chemical potential */

  inv_width = 1.0 / WIDTH;
  cgrid3d_map(gwf->grid, gauss, &inv_width);  
  cgrid3d_multiply(gwf->grid, AMP - rho0);
  cgrid3d_add(gwf->grid, rho0);
  printf("Gaussian max density = %le Angs^-3.\n", AMP / (0.520 * 0.529 * 0.529));
  cgrid3d_power(gwf->grid, gwf->grid, 0.5);
  
  //dft_driver_kinetic = DFT_DRIVER_KINETIC_CN_NBC;

  for (iter = 0; iter < MAXITER; iter++) {
    dft_driver_propagate_predict(DFT_DRIVER_PROPAGATE_HELIUM, ext_pot, gwf, gwfp, potential_store, TS, iter);
    dft_driver_propagate_correct(DFT_DRIVER_PROPAGATE_HELIUM, ext_pot, gwf, gwfp, potential_store, TS, iter);
    if(!(iter % OUTPUT)) {
      sprintf(buf, "final-%ld", iter);
      grid3d_wf_density(gwf, rworkspace);
      dft_driver_write_density(rworkspace, buf);
    }
  }
  return 0;
}
