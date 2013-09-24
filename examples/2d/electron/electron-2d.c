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

double e_he_ps_2d(void *, double, double);

/* Initial guess for bubble radius */
#define BUBBLE_RADIUS 35.0

#define MIN_SUBSTEPS 2
#define MAX_SUBSTEPS 16

double complex electron(void *NA, double z, double r) {

  if(sqrt(z*z + r*r) < BUBBLE_RADIUS) return 1.0;
  else return 0.0;
}

double complex liquid(void *NA, double z, double r) {

  if(sqrt(z*z + r*r) > BUBBLE_RADIUS) return sqrt(0.0218360 * 0.529 * 0.529 * 0.529);
  else return 0.0;
}

int main(int argc, char *argv[]) {

  FILE *fp;
  long k, l, nz, nr, iterations, threads;
  long itp = 0, dump_nth, model;
  double step, time_step;
  double rho0;
  char chk[256];
  long restart = 0;
  wf2d *gwf = 0;
  wf2d *gwfp = 0;
  wf2d *egwf = 0;
  wf2d *egwfp = 0;
  rgrid2d *density = 0;
  rgrid2d *pseudo = 0;
  cgrid2d *potential_store = 0;

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

  if(fscanf(fp, " grid = %ld %ld%*[^\n]", &nz, &nr) < 2) {
    fprintf(stderr, "Invalid grid dimensions.\n");
    exit(1);
  }
  fprintf(stderr, "Grid (%ldx%ld)\n", nz, nr);

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
  dft_driver_setup_grid_2d(nz, nr, step, threads);
  dft_driver_setup_model_2d(model, itp, rho0);
  dft_driver_setup_boundaries_2d(DFT_DRIVER_BOUNDARY_REGULAR, 2.0);
  dft_driver_setup_normalization_2d(DFT_DRIVER_NORMALIZE_BULK, 0, 0.0, 0);
  dft_driver_initialize_2d();

  density = dft_driver_alloc_rgrid_2d();
  pseudo = dft_driver_alloc_rgrid_2d();
  potential_store = dft_driver_alloc_cgrid_2d();
  gwf = dft_driver_alloc_wavefunction_2d(DFT_HELIUM_MASS);
  gwfp = dft_driver_alloc_wavefunction_2d(DFT_HELIUM_MASS);
  egwf = dft_driver_alloc_wavefunction_2d(1.0); /* electron mass */
  egwf->norm = 1.0; /* one electron */
  egwfp = dft_driver_alloc_wavefunction_2d(1.0); /* electron mass */
  egwfp->norm = 1.0; /* one electron */

  /* initialize wavefunctions */
  cgrid2d_map_cyl(egwf->grid, electron, (void *) NULL);
  grid2d_wf_normalize_cyl(egwf);
  cgrid2d_map_cyl(gwf->grid, liquid, (void *) NULL);

  if(restart) {
    fprintf(stderr, "Restart calculation\n");
    dft_driver_read_density_2d(density, "restart.chk");
    rgrid2d_power(density, density, 0.5);
    grid2d_real_to_complex_re(gwf->grid, density);
    cgrid2d_copy(gwfp->grid, gwf->grid);
    dft_driver_read_density_2d(density, "el-restart.chk");
    rgrid2d_power(density, density, 0.5);
    grid2d_real_to_complex_re(egwf->grid, density);
    cgrid2d_copy(egwfp->grid, egwf->grid);
  }
  
  rgrid2d_adaptive_map_cyl(pseudo, e_he_ps_2d, (void *) NULL, MIN_SUBSTEPS, MAX_SUBSTEPS, 0.01 / GRID_AUTOK);
  //rgrid2d_map_cyl(pseudo, e_he_ps_2d, (void *) NULL);

  dft_driver_convolution_prepare_2d(pseudo, NULL);

  /* solve */

  for(l = 1; l < iterations; l++) {   /* start from one to prevent wf init */

    if(!(l % dump_nth) || l == iterations-1) {
      /* Dump helium density */
      grid2d_wf_density(gwf, density);
      sprintf(chk, "helium-%ld", l);
      dft_driver_write_density_2d(density, chk);
      sprintf(chk, "el-%ld", l);
      grid2d_wf_density(egwf, density);
      dft_driver_write_density_2d(density, chk);
    }

    /***** Electron *****/
    grid2d_wf_density(gwf, density);
    dft_driver_convolution_prepare_2d(density, NULL);
    dft_driver_convolution_eval_2d(density, density, pseudo);
#define NST 100
    for(k = 0; k < NST; k++) {
      dft_driver_propagate_predict_2d(DFT_DRIVER_PROPAGATE_OTHER, density /* ..potential.. */, egwf, egwfp, potential_store, time_step/NST, l);
      dft_driver_propagate_correct_2d(DFT_DRIVER_PROPAGATE_OTHER, density /* ..potential.. */, egwf, egwfp, potential_store, time_step/NST, l);
    }

    /***** Helium *****/
    grid2d_wf_density(egwf, density);
    dft_driver_convolution_prepare_2d(density, NULL);
    dft_driver_convolution_eval_2d(density, density, pseudo);
    dft_driver_propagate_predict_2d(DFT_DRIVER_PROPAGATE_HELIUM, density /* ..potential.. */, gwf, gwfp, potential_store, time_step, l);
    dft_driver_propagate_correct_2d(DFT_DRIVER_PROPAGATE_HELIUM, density /* ..potential.. */, gwf, gwfp, potential_store, time_step, l);
  }
}
