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

#define TS 5.0 /* fs */

#define NX 128
#define NY 128
#define NZ 128
#define STEP 1.0

#define ZEROFILL 1024

#define IMITER 200
#define REITER 400

#define UPPER_X "potentials/cu2-b-s.dat"
#define UPPER_Y "potentials/cu2-b-s.dat"
#define UPPER_Z "potentials/cu2-b-s.dat"

#define LOWER_X "potentials/cu2-x-s.dat"
#define LOWER_Y "potentials/cu2-x-s.dat"
#define LOWER_Z "potentials/cu2-x-s.dat"

/* #define MODEL (DFT_OT_PLAIN | DFT_OT_HD | DFT_OT_KC | DFT_OT_BACKFLOW) */
#define MODEL (DFT_OT_PLAIN | DFT_OT_HD)
/* #define RHO0 (0.0218360 * GRID_AUTOANG * GRID_AUTOANG * GRID_AUTOANG) */
#define RHO0 (0.0218360 * GRID_AUTOANG * GRID_AUTOANG * GRID_AUTOANG)
#define TC 150.0

#define HELIUM_MASS (4.002602 / GRID_AUTOAMU)

int main(int argc, char **argv) {

  cgrid3d *potential_store;
  rgrid3d *ext_pot, *density;
  cgrid1d *spectrum;
  wf3d *gwf, *gwfp;
  long iter;
  double energy, natoms, en, mu0;
  FILE *fp;

  /* Setup DFT driver parameters (256 x 256 x 256 grid) */
  dft_driver_setup_grid(NX, NY, NZ, STEP /* Bohr */, 32 /* threads */);
  /* Plain Orsay-Trento in imaginary time */
  dft_driver_setup_model(MODEL, DFT_DRIVER_IMAG_TIME, RHO0);
  /* No absorbing boundary */
  dft_driver_setup_boundaries(DFT_DRIVER_BOUNDARY_REGULAR, 0.0);
  /* Normalization condition */
  dft_driver_setup_normalization(DFT_DRIVER_NORMALIZE_BULK, 0, 3.0, 10);

  /* Initialize the DFT driver */
  dft_driver_initialize();

  /* Allocate space for external potential */
  ext_pot = dft_driver_alloc_rgrid();
  density = dft_driver_alloc_rgrid();
  potential_store = dft_driver_alloc_cgrid(); /* temporary storage */
  /* Read external potential from file */
  dft_common_potential_map(DFT_DRIVER_AVERAGE_NONE, LOWER_X, LOWER_Y, LOWER_Z, ext_pot);

  mu0 = dft_ot_bulk_chempot2(dft_driver_otf);
  printf("mu0 = %le K.\n", mu0 * GRID_AUTOK);
  
  /* Allocate space for wavefunctions (initialized to sqrt(rho0)) */
  gwf = dft_driver_alloc_wavefunction(HELIUM_MASS); /* helium wavefunction */
  gwfp = dft_driver_alloc_wavefunction(HELIUM_MASS);/* temp. wavefunction */

  if(argc == 1) {
    /* Run imaginary time */
    for (iter = 0; iter < IMITER; iter++) {
      dft_driver_propagate_predict(DFT_DRIVER_PROPAGATE_HELIUM, ext_pot, gwf, gwfp, potential_store, 4.0 * TS, iter);
      dft_driver_propagate_correct(DFT_DRIVER_PROPAGATE_HELIUM, ext_pot, gwf, gwfp, potential_store, 4.0 * TS, iter);
    }
    /* At this point gwf contains the converged wavefunction */
    dft_driver_write_grid(gwf->grid, "output");
  } else dft_driver_read_grid(gwf->grid, argv[1]);
    
  energy = dft_driver_energy(gwf, ext_pot);
  natoms = dft_driver_natoms(gwf);
  printf("Total energy is %le K\n", energy * GRID_AUTOK);
  printf("Number of He atoms is %le.\n", natoms);
  printf("Energy / atom is %le K\n", (energy/natoms) * GRID_AUTOK);

  dft_driver_setup_model(MODEL, DFT_DRIVER_REAL_TIME, RHO0);
  rgrid3d_free(ext_pot);
  ext_pot = dft_driver_spectrum_init(REITER, ZEROFILL, DFT_DRIVER_AVERAGE_NONE, UPPER_X, UPPER_Y, UPPER_Z, DFT_DRIVER_AVERAGE_NONE, LOWER_X, LOWER_Y, LOWER_Z);
  rgrid3d_add(ext_pot, -mu0);
  for (iter = 0; iter < REITER; iter++) {
    dft_driver_propagate_predict(DFT_DRIVER_PROPAGATE_HELIUM, ext_pot, gwf, gwfp, potential_store, TS, iter);
    dft_driver_propagate_correct(DFT_DRIVER_PROPAGATE_HELIUM, ext_pot, gwf, gwfp, potential_store, TS, iter);
    dft_driver_spectrum_collect(gwf);
    if(!(iter % 10)) {
      char buf[512];
      grid3d_wf_density(gwf, density);
      sprintf(buf, "realtime-%ld", iter);
      dft_driver_write_density(density, buf);
    }
  }
  spectrum = dft_driver_spectrum_evaluate(TS, TC);

  if(!(fp = fopen("spectrum.dat", "w"))) {
    fprintf(stderr, "Can't open spectrum.dat for writing.\n");
    exit(1);
  }
  for (iter = 0, en = -0.5 * spectrum->step * spectrum->nx; iter < spectrum->nx; iter++, en += spectrum->step)
    fprintf(fp, "%le %le\n", en, creal(cgrid1d_value_at_index(spectrum, iter)));
  fclose(fp);
  printf("Spectrum written to spectrum.dat\n");
  exit(0);
}
