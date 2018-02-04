/*
 * Create a vortex ring in superfluid helium centered around Z = 0.
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

#define TS 20.0 /* fs */
#define NX 128
#define NY 128
#define NZ 128
#define STEP 2.0
#define NTH 10
#define THREADS 4

#define RING_RADIUS 30.0

#define PRESSURE (1.0 / GRID_AUTOBAR)

#define HELIUM_MASS (4.002602 / GRID_AUTOAMU)

double rho0;

/* vortex ring initial guess */
double complex vring(void *asd, double x, double y, double z) {

  double xs = sqrt(x * x + y * y) - RING_RADIUS;
  double ys = z;
  double angle = atan2(ys,xs), r = sqrt(xs*xs + ys*ys);

  return (1.0 - exp(-r * r / 2.0)) * sqrt(rho0) * cexp(I * angle);
}

int main(int argc, char **argv) {

  rgrid3d *ext_pot, *rworkspace;
  cgrid3d *potential_store;
  wf3d *gwf, *gwfp;
  long iter;
  double energy, mu0;
  char buf[512];

  /* Setup DFT driver parameters (grid) */
  dft_driver_setup_grid(NX, NY, NZ, STEP, THREADS);
  /* Plain Orsay-Trento in imaginary time */
  dft_driver_setup_model(DFT_OT_PLAIN, DFT_DRIVER_IMAG_TIME, 0.0);
  /* No absorbing boundary */
  dft_driver_setup_boundary_type(DFT_DRIVER_BOUNDARY_REGULAR, 0.0, 0.0);
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

  /* Allocate space for wavefunctions (initialized to sqrt(rho0)) */
  gwf = dft_driver_alloc_wavefunction(HELIUM_MASS); /* helium wavefunction */
  gwfp = dft_driver_alloc_wavefunction(HELIUM_MASS);/* temp. wavefunction */
  /* setup initial guess for vortex ring */
  cgrid3d_map(gwf->grid, vring, NULL);

  /* Generate the excited potential */
  rgrid3d_constant(ext_pot, -mu0); /* Add the chemical potential */

  for (iter = 1; iter < 80000; iter++) {
    if(iter == 1 || !(iter % NTH)) {
      sprintf(buf, "vring-%ld", iter);
      dft_driver_write_grid(gwf->grid, buf);
    }
    dft_driver_propagate_predict(DFT_DRIVER_PROPAGATE_HELIUM, ext_pot, gwf, gwfp, potential_store, TS, iter);
    dft_driver_propagate_correct(DFT_DRIVER_PROPAGATE_HELIUM, ext_pot, gwf, gwfp, potential_store, TS, iter);
  }
}
