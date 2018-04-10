/*
 * Shock wave propagation in superfluid helium.
 *
 * All input in a.u. except the time step, which is fs.
 *
 * The initial condition for the shock is given by:
 *
 * \psi(z, 0) = \sqrt(\rho_0) if |z| > w
 * or
 * \psi(z, 0) = \sqrt(\rho_0 + \Delta)\exp(-(v_z/m_He)(z + w) / \hbar)
 *
 * where \Delta is the shock amplitude and v_z is the shock velocity
 * (discontinuity in both density and velocity)
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
#define OUTPUT 100

#define DELTA (0.05 * rho0)
#define W 30.0
#define VZ (230.0 / GRID_AUTOMPS)
#define KZ (HELIUM_MASS * VZ / HBAR)

#define PRESSURE (0.0 / GRID_AUTOBAR)   /* External pressure in bar */
#define HELIUM_MASS (4.002602 / GRID_AUTOAMU)

#define FUNC (DFT_OT_PLAIN)
//#define FUNC (DFT_OT_PLAIN | DFT_OT_KC)
#define NX 64
#define NY 64
#define NZ 1024
#define STEP 1.0

struct params {
  REAL delta;
  REAL rho0;
  REAL w;
  REAL vz;
};

REAL complex gauss(void *arg, REAL x, REAL y, REAL z) {

  REAL delta = ((struct params *) arg)->delta;
  REAL rho0 = ((struct params *) arg)->rho0;
  REAL w = ((struct params *) arg)->w;
  REAL vz = ((struct params *) arg)->vz;

//  if(FABS(z) < w) return SQRT(rho0 + delta) * CEXP(I * (vz / HELIUM_MASS) * (z + w) / HBAR);
  if(FABS(z) < w) return SQRT(rho0 + delta);
  else return SQRT(rho0);
}

int main(int argc, char **argv) {

  struct params sparams;
  rgrid3d *ext_pot, *rworkspace;
  cgrid3d *potential_store;
  wf3d *gwf, *gwfp;
  INT iter;
  REAL rho0, mu0;
  char buf[512];

  fprintf(stderr, "Time step = " FMT_R " fs.\n", TS);
  /* Setup DFT driver parameters (256 x 256 x 256 grid) */
  dft_driver_setup_grid(NX, NY, NZ, STEP /* Bohr */, 0 /* threads */);
  /* Plain Orsay-Trento in imaginary time */
  /* Setup frame of reference momentum */
  dft_driver_setup_momentum(0.0, 0.0, KZ);
  dft_driver_setup_model(FUNC, DFT_DRIVER_REAL_TIME, 0.0);
  /* No absorbing boundary */
  dft_driver_setup_boundary_type(DFT_DRIVER_BOUNDARY_REGULAR, 0.0, 0.0, 0.0, 0.0);
  /* Normalization condition */
  dft_driver_setup_normalization(DFT_DRIVER_DONT_NORMALIZE, 0, 0.0, 0);

  /* Initialize the DFT driver */
  dft_driver_initialize();

  /* Allocate space for external potential */
  ext_pot = dft_driver_alloc_rgrid("Ext pot");
  rworkspace = dft_driver_alloc_rgrid("rworkspace");
  potential_store = dft_driver_alloc_cgrid("cworkspace"); /* temporary storage */
  /* Read initial external potential from file */

  /* Allocate space for wavefunctions (initialized to SQRT(rho0)) */
  gwf = dft_driver_alloc_wavefunction(HELIUM_MASS, "gwf"); /* helium wavefunction */
  gwfp = dft_driver_alloc_wavefunction(HELIUM_MASS, "gwfp");/* temp. wavefunction */
  rho0 = dft_driver_otf->rho0 = dft_ot_bulk_density_pressurized(dft_driver_otf, PRESSURE);
  mu0  = dft_ot_bulk_chempot_pressurized(dft_driver_otf, PRESSURE);
  rgrid3d_zero(ext_pot);
  rgrid3d_add(ext_pot, -mu0); /* Add the chemical potential */

  sparams.delta = DELTA;
  sparams.rho0 = rho0;
  sparams.w = W;
  sparams.vz = VZ;
  cgrid3d_map(gwf->grid, gauss, (void *) &sparams);  
  
  //dft_driver_kinetic = DFT_DRIVER_KINETIC_CN_NBC;

  for (iter = 0; iter < MAXITER; iter++) {
    dft_driver_propagate_predict(DFT_DRIVER_PROPAGATE_HELIUM, ext_pot, gwf, gwfp, potential_store, TS, iter);
    dft_driver_propagate_correct(DFT_DRIVER_PROPAGATE_HELIUM, ext_pot, gwf, gwfp, potential_store, TS, iter);
    if(!(iter % OUTPUT)) {
      sprintf(buf, "final-" FMT_I, iter);
      grid3d_wf_density(gwf, rworkspace);
      dft_driver_write_density(rworkspace, buf);
    }
  }
  return 0;
}
