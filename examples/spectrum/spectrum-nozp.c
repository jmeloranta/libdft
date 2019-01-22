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

  cgrid *potential_store;
  rgrid *ext_pot, *density;
  cgrid *spectrum;
  wf *gwf, *gwfp;
  INT iter;
  REAL energy, natoms, en, mu0;
  FILE *fp;

  /* Setup DFT driver parameters (256 x 256 x 256 grid) */
  dft_driver_setup_grid(NX, NY, NZ, STEP /* Bohr */, 32 /* threads */);
  /* Plain Orsay-Trento in imaginary time */
  dft_driver_setup_model(MODEL, DFT_DRIVER_IMAG_TIME, RHO0);
  /* No absorbing boundary */
  dft_driver_setup_boundary_type(DFT_DRIVER_BOUNDARY_REGULAR, 0.0, 0.0, 0.0, 0.0);
  /* Normalization condition */
  dft_driver_setup_normalization(DFT_DRIVER_NORMALIZE_BULK, 0, 3.0, 10);

  /* Allocate space for wavefunctions (initialized to SQRT(rho0)) */
  gwf = dft_driver_alloc_wavefunction(HELIUM_MASS, "gwf"); /* helium wavefunction */
  gwfp = dft_driver_alloc_wavefunction(HELIUM_MASS, "gwfp");/* temp. wavefunction */

  /* Initialize the DFT driver */
  dft_driver_initialize(gwf);

  /* Allocate space for external potential */
  ext_pot = dft_driver_alloc_rgrid("ext_pot");
  density = dft_driver_alloc_rgrid("density");
  potential_store = dft_driver_alloc_cgrid("potential_store"); /* temporary storage */
  /* Read external potential from file */
  dft_common_potential_map(DFT_DRIVER_AVERAGE_NONE, LOWER_X, LOWER_Y, LOWER_Z, ext_pot);

  mu0 = dft_ot_bulk_chempot2(dft_driver_otf);
  printf("mu0 = " FMT_R " K.\n", mu0 * GRID_AUTOK);  

  if(argc == 1) {
    /* Run imaginary time */
    for (iter = 0; iter < IMITER; iter++) {
      dft_driver_propagate_predict(DFT_DRIVER_PROPAGATE_HELIUM, ext_pot, mu0, gwf, gwfp, potential_store, 4.0 * TS, iter);
      dft_driver_propagate_correct(DFT_DRIVER_PROPAGATE_HELIUM, ext_pot, mu0, gwf, gwfp, potential_store, 4.0 * TS, iter);
    }
    /* At this point gwf contains the converged wavefunction */
    cgrid_write_grid("output", gwf->grid);
  } else cgrid_read_grid(gwf->grid, argv[1]);
    
  dft_ot_energy_density(dft_driver_otf, density, gwf);
  rgrid_add_scaled_product(density, 1.0, dft_driver_otf->density, ext_pot);
  energy = grid_wf_energy(gwf, NULL) + rgrid_integral(density);
  natoms = grid_wf_norm(gwf);
  printf("Total energy is " FMT_R " K\n", energy * GRID_AUTOK);
  printf("Number of He atoms is " FMT_R ".\n", natoms);
  printf("Energy / atom is " FMT_R " K\n", (energy/natoms) * GRID_AUTOK);

  dft_driver_setup_model(MODEL, DFT_DRIVER_REAL_TIME, RHO0);
  rgrid_free(ext_pot);
  ext_pot = dft_driver_spectrum_init(NULL, REITER, ZEROFILL, DFT_DRIVER_AVERAGE_NONE, UPPER_X, UPPER_Y, UPPER_Z, DFT_DRIVER_AVERAGE_NONE, LOWER_X, LOWER_Y, LOWER_Z);

  for (iter = 0; iter < REITER; iter++) {
    dft_driver_propagate_predict(DFT_DRIVER_PROPAGATE_HELIUM, ext_pot, mu0, gwf, gwfp, potential_store, TS, iter);
    dft_driver_propagate_correct(DFT_DRIVER_PROPAGATE_HELIUM, ext_pot, mu0, gwf, gwfp, potential_store, TS, iter);
    dft_driver_spectrum_collect(gwf);
    if(!(iter % 10)) {
      char buf[512];
      grid_wf_density(gwf, density);
      sprintf(buf, "realtime-" FMT_I, iter);
      rgrid_write_grid(buf, density);
    }
  }
  spectrum = dft_driver_spectrum_evaluate(TS, TC);

  if(!(fp = fopen("spectrum.dat", "w"))) {
    fprintf(stderr, "Can't open spectrum.dat for writing.\n");
    exit(1);
  }
  for (iter = 0, en = -0.5 * spectrum->step * (REAL) spectrum->nx; iter < spectrum->nx; iter++, en += spectrum->step)
    //    fprintf(fp, FMT_R " " FMT_R "\n", en, CREAL(cgrid_value_at_index(spectrum, 1, 1, iter)));
    fprintf(fp, FMT_R " " FMT_R "\n", en, POW(CREAL(cgrid_value_at_index(spectrum, 1, 1, iter)), 2.0) + POW(CIMAG(cgrid_value_at_index(spectrum, 1, 1, iter)), 2.0));
  fclose(fp);
  printf("Spectrum written to spectrum.dat\n");
  exit(0);
}
