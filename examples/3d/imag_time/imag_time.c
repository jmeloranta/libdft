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

#define TS 1.0E-3

#define MAXITER 10000000
#define NTH 1000

#define THREADS 0

#define HELIUM_MASS (4.002602 / GRID_AUTOAMU)

/* Ion */
// #define EXP_P
#define K_P

#ifdef ZERO_P
#define A0 0.0
#define A1 0.0
#define A2 0.0
#define A3 0.0
#define A4 0.0
#define A5 0.0
#define RMIN 0.0
#define RADD 0.0
#endif

/* exponential repulsion (approx. electron bubble) - RADD = -19.0 */
#ifdef EXP_P
#define A0 (3.8003E5 / GRID_AUTOK)
#define A1 (1.6245 * GRID_AUTOANG)
#define A2 0.0
#define A3 0.0
#define A4 0.0
#define A5 0.0
#define RMIN 2.0
#define RADD (-6.0)
#endif

/* Ca+ */
#ifdef CA_P
#define A0 4.83692
#define A1 1.23684
#define A2 0.273202
#define A3 59.5463
#define A4 1134.51
#define A5 0.0
#define RMIN 5.0
#define RADD 0.0
#endif

/* K+ */
#ifdef K_P
#define A0 140.757
#define A1 2.26202
#define A2 0.722065
#define A3 0.00144039
#define A4 356.303
#define A5 1358.98
#define RMIN 4.0
#define RADD 0.0
#endif

/* Be+ */
#ifdef BE_P
#define A0 4.73292
#define A1 1.53925
#define A2 0.557845
#define A3 26.7013
#define A4 0.0
#define A5 0.0
#define RMIN 3.4
#define RADD 0.0
#endif

/* Sr+ */
#ifdef SR_P
#define A0 3.64975
#define A1 1.13451
#define A2 0.293483
#define A3 99.0206
#define A4 693.904
#define A5 0.0
#define RMIN 5.0
#define RADD 0.0
#endif

/* Cl- */
#ifdef CL_M
#define A0 11.1909
#define A1 1.50971
#define A2 0.72186
#define A3 17.2434
#define A4 0.0
#define A5 0.0
#define RMIN 4.2
#define RADD 0.0
#endif

/* F- */
#ifdef F_M
#define A0 5.16101
#define A1 1.62798
#define A2 0.773982
#define A3 1.09722
#define A4 0.0
#define A5 0.0
#define RMIN 4.1
#define RADD 0.0
#endif

/* I- */
#ifdef I_M
#define A0 13.6874
#define A1 1.38037
#define A2 0.696409
#define A3 37.3331 
#define A4 0.0
#define A5 0.0
#define RMIN 4.1
#define RADD 0.0
#endif

/* Br- */
#ifdef BR_M
#define A0 12.5686
#define A1 1.45686
#define A2 0.714525
#define A3 24.114
#define A4 0.0
#define A5 0.0
#define RMIN 5.0
#define RADD 0.0
#endif

double pot_func(void *asd, double x, double y, double z) {

  double r, r2, r4, r6, r8, r10, tmp, *asdf;

  if(asd) {
    asdf = asd;
    x -= *asdf;
  }
  r = sqrt(x * x + y * y + z * z);
  if(r < RMIN) r = RMIN;
  r += RADD;

  r2 = r * r;
  r4 = r2 * r2;
  r6 = r4 * r2;
  r8 = r6 * r2;
  r10 = r8 * r2;
  tmp = A0 * exp(-A1 * r) - A2 / r4 - A3 / r6 - A4 / r8 - A5 / r10;
  return tmp;
}

int main(int argc, char **argv) {

  cgrid3d *potential_store;
  rgrid3d *ext_pot, *density;
  wf3d *gwf, *gwfp;
  long iter;
  double energy, natoms, mu0, rho0;

  /* Setup DFT driver parameters */
  dft_driver_setup_grid(NX, NY, NZ, STEP, THREADS);
  /* Plain Orsay-Trento in imaginary time */
  dft_driver_setup_model(DFT_OT_PLAIN | DFT_OT_HD, DFT_DRIVER_IMAG_TIME, 0.0);
  /* No absorbing boundary */
  dft_driver_setup_boundaries(DFT_DRIVER_BOUNDARY_REGULAR, 2.0);
  /* Normalization condition */
  dft_driver_setup_normalization(DFT_DRIVER_DONT_NORMALIZE, 0, 0.0, 0);

  /* Initialize the DFT driver */
  dft_driver_initialize();

  /* Allocate space for external potential */
  ext_pot = dft_driver_alloc_rgrid();
  potential_store = dft_driver_alloc_cgrid(); /* temporary storage */
  density = dft_driver_alloc_rgrid();

  /* Allocate space for wavefunctions (initialized to sqrt(rho0)) */
  gwf = dft_driver_alloc_wavefunction(HELIUM_MASS); /* helium wavefunction */
  gwfp = dft_driver_alloc_wavefunction(HELIUM_MASS);/* temp. wavefunction */

  /* Read external potential from file */
  rgrid3d_map(ext_pot, pot_func, NULL);
  //dft_common_potential_map(DFT_DRIVER_AVERAGE_NONE, "cl-pot.dat", "cl-pot.dat", "cl-pot.dat", ext_pot);
  mu0 = dft_ot_bulk_chempot2(dft_driver_otf);
  rgrid3d_add(ext_pot, -mu0);
  rho0 = dft_driver_otf->rho0;
  printf("mu0 = %le K, rho0 = %le Angs^-3.\n", mu0 * GRID_AUTOK, rho0 / (GRID_AUTOANG * GRID_AUTOANG * GRID_AUTOANG));

  /* Run 200 iterations using imaginary time (10 fs time step) */
  for (iter = 0; iter < MAXITER; iter++) {
    dft_driver_propagate_predict(DFT_DRIVER_PROPAGATE_HELIUM, ext_pot, gwf, gwfp, potential_store, TS /* fs */, iter);
    dft_driver_propagate_correct(DFT_DRIVER_PROPAGATE_HELIUM, ext_pot, gwf, gwfp, potential_store, TS /* fs */, iter);
    if(!(iter % NTH)) {
      char buf[512];
      sprintf(buf, "output-%ld", iter);
      grid3d_wf_density(gwf, density);
      dft_driver_write_density(density, buf);
      sprintf(buf, "wf-output-%ld", iter);
      dft_driver_write_grid(gwf->grid, buf);
    }
    energy = dft_driver_energy(gwf, ext_pot);
    natoms = dft_driver_natoms(gwf);
    printf("Total energy is %le K\n", energy * GRID_AUTOK);
    printf("Number of He atoms is %le.\n", natoms);
    printf("Energy / atom is %le K\n", (energy/natoms) * GRID_AUTOK);
    fflush(stdout); fflush(stderr);
  }
  /* At this point gwf contains the converged wavefunction */
  grid3d_wf_density(gwf, density);
  dft_driver_write_density(density, "output");
}
