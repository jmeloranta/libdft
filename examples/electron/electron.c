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

/* Initial guess for bubble radius */
#define BUBBLE_RADIUS 1.0

#define INCLUDE_VORTEX 1 /**/
/* Include electron? */
/* #define INCLUDE_ELECTRON 1 */

REAL rho0;

REAL complex bubble(void *NA, REAL x, REAL y, REAL z) {

  if(SQRT(x*x + y*y + z*z) < BUBBLE_RADIUS) return 0.0;
  return SQRT(rho0);
}

int main(int argc, char *argv[]) {

  FILE *fp;
  INT l, nx, ny, nz, iterations, threads;
  INT itp = 0, dump_nth, model;
  REAL step, time_step, mu0, time_step_el, width;
  char chk[256];
  INT restart = 0;
  wf *gwf = 0;
  wf *gwfp = 0;
  wf *egwf = 0;
  wf *egwfp = 0;
  rgrid *rworkspace = NULL, *pseudo = NULL;
  cgrid *potential_store = 0;

  /* parameters */
  if (argc < 2) {
    fprintf(stderr, "Usage: %s <paramfile.dat>\n", argv[0]);
    return 1;
  }
  
  if(!(fp = fopen(argv[1], "r"))) {
    fprintf(stderr, "Unable to open %s\n", argv[1]);
    return 1;
  }

  if(fscanf(fp, " threads = " FMT_I "%*[^\n]", &threads) < 1) {
    fprintf(stderr, "Invalid number of threads.\n");
    exit(1);
  }

  if(fscanf(fp, " grid = " FMT_I " " FMT_I " " FMT_I "%*[^\n]", &nx, &ny, &nz) < 3) {
    fprintf(stderr, "Invalid grid dimensions.\n");
    exit(1);
  }
  fprintf(stderr, "Grid (" FMT_I "x" FMT_I "x" FMT_I ")\n", nx, ny, nz);

  if(fscanf(fp, " gstep = " FMT_R "%*[^\n]", &step) < 1) {
    fprintf(stderr, "Invalid grid step.\n");
    exit(1);
  }
  
  if(fscanf(fp, " timestep = " FMT_R "%*[^\n]", &time_step) < 1) {
    fprintf(stderr, "Invalid time step.\n");
    exit(1);
  }

  if(fscanf(fp, " timestep_el = " FMT_R "%*[^\n]", &time_step_el) < 1) {
    fprintf(stderr, "Invalid time step.\n");
    exit(1);
  }

  fprintf(stderr, "Liquid time step = " FMT_R " fs, electron time step = " FMT_R " fs.\n", time_step, time_step_el);
  
  if(fscanf(fp, " iter = " FMT_I "%*[^\n]", &iterations) < 1) {
    fprintf(stderr, "Invalid number of iterations.\n");
    exit(1);
  }

  if(fscanf(fp, " itermode = " FMT_I "%*[^\n]", &itp) < 1) {
    fprintf(stderr, "Invalid iteration mode (0 = real time, 1 = imaginary time).\n");
    exit(1);
  }

  if(fscanf(fp, " dump = " FMT_I "%*[^\n]", &dump_nth) < 1) {
    fprintf(stderr, "Invalid dump iteration specification.\n");
    exit(1);
  }
  if(fscanf(fp, " model = " FMT_I "%*[^\n]", &model) < 1) {
    fprintf(stderr, "Invalid model.\n");
    exit(1);
  }
  if(fscanf(fp, " rho0 = " FMT_R "%*[^\n]", &rho0) < 1) {
    fprintf(stderr, "Invalid density.\n");
    exit(1);
  }
  rho0 *= GRID_AUTOANG * GRID_AUTOANG * GRID_AUTOANG;
  fprintf(stderr,"rho0 = " FMT_R " a.u.\n", rho0);
  if(fscanf(fp, " restart = " FMT_I "%*[^\n]", &restart) < 1) {
    fprintf(stderr, "Invalid restart data.\n");
    exit(1);
  }
  printf("restart = " FMT_I ".\n", restart);
  fclose(fp);

#ifdef USE_CUDA
  cuda_enable(1);  // enable CUDA ?
#endif

  /* allocate memory (3 x grid dimension, */
  fprintf(stderr,"Model = " FMT_I ".\n", model);
  dft_driver_setup_grid(nx, ny, nz, step, threads);
  dft_driver_setup_model(model, itp, rho0);
  dft_driver_setup_boundary_type(DFT_DRIVER_BOUNDARY_REGULAR, 0.0, 0.0, 0.0, 0.0);
  dft_driver_setup_normalization(DFT_DRIVER_DONT_NORMALIZE, 0, 0.0, 0);
  /* Neumann boundaries */
  dft_driver_setup_boundary_condition(DFT_DRIVER_BC_NEUMANN);

  gwf = dft_driver_alloc_wavefunction(DFT_HELIUM_MASS, "gwf");
  gwfp = dft_driver_alloc_wavefunction(DFT_HELIUM_MASS, "gwfp");
  egwf = dft_driver_alloc_wavefunction(1.0, "egwf"); /* electron mass */
  egwf->norm = 1.0; /* one electron */
  egwfp = dft_driver_alloc_wavefunction(1.0, "egwfp"); /* electron mass */
  egwfp->norm = 1.0; /* one electron */

  dft_driver_initialize(gwf);

  rworkspace = dft_driver_alloc_rgrid("rworkspace");
  pseudo = dft_driver_alloc_rgrid("pseudo");
  potential_store = dft_driver_alloc_cgrid("potential_store");

  /* initialize wavefunctions */
  width = 1.0 / 14.5; /* actually inverse width */
  cgrid_map(egwf->grid, &dft_common_cgaussian, &width);
  grid_wf_normalize(egwf);
  cgrid_map(gwf->grid, bubble, (void *) NULL);

  if(restart) {
    fprintf(stderr, "Restart calculation\n");
    rgrid_read_grid(rworkspace, "restart.chk");
    rgrid_power(rworkspace, rworkspace, 0.5);
    grid_real_to_complex_re(gwf->grid, rworkspace);
    cgrid_copy(gwfp->grid, gwf->grid);
    rgrid_read_grid(rworkspace, "el-restart.chk");
    rgrid_power(rworkspace, rworkspace, 0.5);
    grid_real_to_complex_re(egwf->grid, rworkspace);
    cgrid_copy(egwfp->grid, egwf->grid);
    l = 1;
  } else l = 0;

#ifdef INCLUDE_ELECTRON  
  fprintf(stderr,"Electron included.\n");
  dft_common_potential_map(DFT_DRIVER_AVERAGE_NONE, "jortner.dat", "jortner.dat", "jortner.dat", pseudo);
  rgrid_fft(pseudo);
#else
  rgrid_zero(pseudo);
#endif

  fprintf(stderr,"Specified rho0 = " FMT_R " Angs^-3\n", rho0);
  mu0 = dft_ot_bulk_chempot2(dft_driver_otf);
  fprintf(stderr,"mu0 = " FMT_R " K.\n", mu0 * GRID_AUTOK);
  fprintf(stderr,"Applied P = " FMT_R " MPa.\n", dft_ot_bulk_pressure(dft_driver_otf, rho0) * GRID_AUTOPA / 1E6);
  
  /* Include vortex line initial guess along Z */
#ifdef INCLUDE_VORTEX
  fprintf(stderr,"Vortex included.\n");
  dft_driver_vortex_initial(gwf, 1, DFT_DRIVER_VORTEX_Z);
#endif

  /* solve */
  for(; l < iterations; l++) {

    if(!(l % dump_nth) || l == iterations-1 || l == 1) {
      REAL energy, natoms;
      energy = grid_wf_energy(gwf, NULL);
#ifdef INCLUDE_ELECTRON      
      energy += grid_wf_energy(egwf, NULL);
      grid_wf_density(gwf, rworkspace);
      rgrid_fft(rworkspace);
      rgrid_fft_convolute(rworkspace, rworkspace, pseudo);
      rgrid_inverse_fft(rworkspace);
      
      grid_wf_density(egwf, dft_driver_otf->density);
      rgrid_product(dft_driver_otf->density, dft_driver_otf->density, rworkspace);
      energy += rgrid_integral(dft_driver_otf->density);      /* Liquid - impurity interaction energy */
#endif      
      natoms = grid_wf_norm(gwf);
      fprintf(stderr,"Energy with respect to bulk = " FMT_R " K.\n", (energy - dft_ot_bulk_energy(dft_driver_otf, rho0) * natoms / rho0) * GRID_AUTOK);
      fprintf(stderr,"Number of He atoms = " FMT_R ".\n", natoms);
      fprintf(stderr,"mu0 = %le K, energy/natoms = " FMT_R " K\n", mu0 * GRID_AUTOK,  GRID_AUTOK * energy / natoms);

      /* Dump helium density */
      grid_wf_density(gwf, rworkspace);
      sprintf(chk, "helium-" FMT_I, l);
      rgrid_write_grid(chk, rworkspace);
#ifdef INCLUDE_ELECTRON
      /* Dump electron density */
      sprintf(chk, "el-" FMT_I, l);
      grid_wf_density(egwf, rworkspace);
      rgrid_write_grid(chk, rworkspace);
      /* Dump electron wavefunction */
      sprintf(chk, "el-wf-" FMT_I, l);
      cgrid_write_grid(chk, egwf->grid);
#endif
      /* Dump helium wavefunction */
      sprintf(chk, "helium-wf-" FMT_I, l);
      cgrid_write_grid(chk, gwf->grid);
    }

#ifdef INCLUDE_ELECTRON
    /***** Electron *****/
    grid_wf_density(gwf, rworkspace);
    rgrid_fft(rworkspace);
    rgrid_fft_convolute(rworkspace, rworkspace, pseudo);
    rgrid_inverse_fft(rworkspace);
    /* It is OK to run just one step - in imaginary time but not in real time. */
    dft_driver_propagate_predict(DFT_DRIVER_PROPAGATE_OTHER, rworkspace /* ..potential.. */, 0.0, egwf, egwfp, potential_store, time_step_el, l);
    dft_driver_propagate_correct(DFT_DRIVER_PROPAGATE_OTHER, rworkspace /* ..potential.. */, 0.0, egwf, egwfp, potential_store, time_step_el, l);
#else
    cgrid_zero(egwf->grid);
#endif

    /***** Helium *****/
#ifdef INCLUDE_ELECTRON
    grid_wf_density(egwf, rworkspace);
    rgrid_fft(rworkspace);
    rgrid_fft_convolute(rworkspace, rworkspace, pseudo);
    rgrid_inverse_fft(rworkspace);
#else
    rgrid_zero(rworkspace);
#endif
    dft_driver_propagate_predict(DFT_DRIVER_PROPAGATE_HELIUM, rworkspace /* ..potential.. */, mu0, gwf, gwfp, potential_store, time_step, l);
    dft_driver_propagate_correct(DFT_DRIVER_PROPAGATE_HELIUM, rworkspace /* ..potential.. */, mu0, gwf, gwfp, potential_store, time_step, l);
  }
  return 0;
}
