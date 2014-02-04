/*
 * Solvated electron in superfluid helium.
 * The electron-helium pseudo potential requires
 * fairly good resolution to be evaluated correctly,
 * ca. 0.2 Bohr spatial step.
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

#define NST 100   /* NST steps of electron for every step of liquid */

double e_he_ps(void *, double, double, double);

/* Initial guess for bubble radius */
#define BUBBLE_RADIUS 25.0

#define MIN_SUBSTEPS 2
#define MAX_SUBSTEPS 16

double rho0;

double complex bubble(void *NA, double x, double y, double z) {

  if(sqrt(x*x + y*y + z*z) < BUBBLE_RADIUS) return 0.0;
  else return sqrt(rho0);
}

int main(int argc, char *argv[]) {

  FILE *fp;
  long k, l, nx, ny, nz, iterations, threads;
  long itp = 0, dump_nth, model;
  double step, time_step;
  char chk[256];
  long restart = 0;
  wf3d *gwf = 0;
  wf3d *gwfp = 0;
  wf3d *egwf = 0;
  wf3d *egwfp = 0;
  rgrid3d *density = 0;
  rgrid3d *pseudo = 0;
  cgrid3d *potential_store = 0;

  /* parameters */
  if (argc < 2) {
    fprintf(stderr, "Usage: %s <paramfile.dat>\n", argv[0]);
    return 1;
  }
  
  if(!(fp = fopen(argv[1], "r"))) {
    fprintf(stderr, "Unable to open %s\n", argv[1]);
    return 1;
  }

  if(fscanf(fp, " threads = %ld%*[^\n]", &threads) < 1) {
    fprintf(stderr, "Invalid number of threads.\n");
    exit(1);
  }

  if(fscanf(fp, " grid = %ld %ld %ld%*[^\n]", &nx, &ny, &nz) < 3) {
    fprintf(stderr, "Invalid grid dimensions.\n");
    exit(1);
  }
  fprintf(stderr, "Grid (%ldx%ldx%ld)\n", nx, ny, nz);

  if(fscanf(fp, " gstep = %le%*[^\n]", &step) < 1) {
    fprintf(stderr, "Invalid grid step.\n");
    exit(1);
  }
  
  if(fscanf(fp, " timestep = %le%*[^\n]", &time_step) < 1) {
    fprintf(stderr, "Invalid time step.\n");
    exit(1);
  }

  if(fscanf(fp, " iter = %ld%*[^\n]", &iterations) < 1) {
    fprintf(stderr, "Invalid number of iterations.\n");
    exit(1);
  }

  if(fscanf(fp, " itermode = %ld%*[^\n]", &itp) < 1) {
    fprintf(stderr, "Invalid iteration mode (0 = real time, 1 = imaginary time).\n");
    exit(1);
  }

  if(fscanf(fp, " dump = %ld%*[^\n]", &dump_nth) < 1) {
    fprintf(stderr, "Invalid dump iteration specification.\n");
    exit(1);
  }
  if(fscanf(fp, " model = %ld%*[^\n]", &model) < 1) {
    fprintf(stderr, "Invalid model.\n");
    exit(1);
  }
  if(fscanf(fp, " rho0 = %le%*[^\n]", &rho0) < 1) {
    fprintf(stderr, "Invalid density.\n");
    exit(1);
  }
  rho0 *= GRID_AUTOANG * GRID_AUTOANG * GRID_AUTOANG;
  printf("rho0 = %le a.u.\n", rho0);
  if(fscanf(fp, " restart = %ld%*[^\n]", &restart) < 1) {
    fprintf(stderr, "Invalid restart data.\n");
    exit(1);
  }
  printf("restart = %ld.\n", restart);
  fclose(fp);

  /* allocate memory (3 x grid dimension, */
  printf("Model = %ld.\n", model);
  dft_driver_setup_grid(nx, ny, nz, step, threads);
  dft_driver_setup_model(model, itp, rho0);
  dft_driver_setup_boundaries(DFT_DRIVER_BOUNDARY_REGULAR, 2.0);
  dft_driver_setup_normalization(DFT_DRIVER_NORMALIZE_BULK, 0, 0.0, 0);
  dft_driver_initialize();

  density = dft_driver_alloc_rgrid();
  pseudo = dft_driver_alloc_rgrid();
  potential_store = dft_driver_alloc_cgrid();
  gwf = dft_driver_alloc_wavefunction(DFT_HELIUM_MASS);
  gwfp = dft_driver_alloc_wavefunction(DFT_HELIUM_MASS);
  egwf = dft_driver_alloc_wavefunction(1.0); /* electron mass */
  egwf->norm = 1.0; /* one electron */
  egwfp = dft_driver_alloc_wavefunction(1.0); /* electron mass */
  egwfp->norm = 1.0; /* one electron */

  /* initialize wavefunctions */
  dft_driver_gaussian_wavefunction(egwf, 0.0, 0.0, 0.0, 14.5);
  grid3d_wf_normalize(egwf);
  cgrid3d_map(gwf->grid, bubble, (void *) NULL);

  if(restart) {
    fprintf(stderr, "Restart calculation\n");
    dft_driver_read_density(density, "restart.chk");
    rgrid3d_power(density, density, 0.5);
    grid3d_real_to_complex_re(gwf->grid, density);
    cgrid3d_copy(gwfp->grid, gwf->grid);
    dft_driver_read_density(density, "el-restart.chk");
    rgrid3d_power(density, density, 0.5);
    grid3d_real_to_complex_re(egwf->grid, density);
    cgrid3d_copy(egwfp->grid, egwf->grid);
  }
  
  dft_common_potential_map(DFT_DRIVER_AVERAGE_NONE, "jortner.dat", "jortner.dat", "jortner.dat", pseudo);
  dft_driver_convolution_prepare(pseudo, NULL);

  /* solve */

  for(l = 1; l < iterations; l++) {

    if(!(l % dump_nth) || l == iterations-1 || l == 1) {
      /* Dump helium density */
      grid3d_wf_density(gwf, density);
      sprintf(chk, "helium-%ld", l);
      dft_driver_write_density(density, chk);
      /* Dump electron density */
      sprintf(chk, "el-%ld", l);
      grid3d_wf_density(egwf, density);
      dft_driver_write_density(density, chk);
      /* Dump electron wavefunction */
      sprintf(chk, "el-wf-%ld", l);
      dft_driver_write_grid(egwf->grid, chk);
      /* Dump helium wavefunction */
      sprintf(chk, "helium-wf-%ld", l);
      dft_driver_write_grid(gwf->grid, chk);
    }

    /***** Electron *****/
    grid3d_wf_density(gwf, density);
    dft_driver_convolution_prepare(density, NULL);
    dft_driver_convolution_eval(density, density, pseudo);
    for(k = 0; k < NST; k++) {
      dft_driver_propagate_predict(DFT_DRIVER_PROPAGATE_OTHER, density /* ..potential.. */, egwf, egwfp, potential_store, time_step/NST, k);
      dft_driver_propagate_correct(DFT_DRIVER_PROPAGATE_OTHER, density /* ..potential.. */, egwf, egwfp, potential_store, time_step/NST, k);
    }

    /***** Helium *****/
    grid3d_wf_density(egwf, density);
    dft_driver_convolution_prepare(density, NULL);
    dft_driver_convolution_eval(density, density, pseudo);
    dft_driver_propagate_predict(DFT_DRIVER_PROPAGATE_HELIUM, density /* ..potential.. */, gwf, gwfp, potential_store, time_step, l);
    dft_driver_propagate_correct(DFT_DRIVER_PROPAGATE_HELIUM, density /* ..potential.. */, gwf, gwfp, potential_store, time_step, l);
  }
}
