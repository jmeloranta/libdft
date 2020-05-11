/*
 * Bulk helium at a given temperature (explicit thermal excitations).
 *
 * 1. Start from random initial order parameter.
 * 2. Cooling phase with a mixture of real and imaginary time (the ratio defines the cooling rate).
 * 3. Real time iterations to achieve the proper thermal equilibrium.
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

/* Time integration and spatial grid parameters */
#define TS 0.25 /* fs (was 5) */
#define ITS 4.0 /* fs */

#define NX 256
#define NY 256
#define NZ 256
#define STEP 1.0

/* E(k) */
#define KSPECTRUM /**/
#define NBINS 512
#define BINSTEP 0.1
#define DENS_EPS 1E-3

/* Predict-correct? */
#define PC

/* Use dealiasing during real time propagation? (2/3-rule) - does not seem to help? */
//#define DEALIAS

/* Functional to use (was DFT_OT_PLAIN; GP2 is test)  -- TODO: There is a problem with backflow, energy keeps increasing? HD does not help. Predict-correct or shoter time step? or shorter grid step? */
#define FUNCTIONAL (DFT_OT_PLAIN | DFT_OT_KC | DFT_OT_BACKFLOW | DFT_OT_HD)
//#define FUNCTIONAL (DFT_OT_PLAIN)

/* Pressure */
#define PRESSURE (0.0 / GRID_AUTOBAR)

/* Normalization - Since we have the thermal excitations, the chemical potential method is not good... Hence explicit normalization */
#define HE_NORM (rho0 * ((REAL) NX) * STEP * ((REAL) NY) * STEP * ((REAL) NZ) * STEP)

/* Lambda temperature */
#define TLAMBDA 2.17

/* The number of cooling iterations (mixture of real and imaginary) */
/* 2000 = 2.16 K (1fs real + 4fs imag) */
/* 3000 = 2.07 K (based on enthalpy) */
/* 5000 = 1.96 K */
/* 20000 = 1.78 K */
#define COOL 2000

/* The number of thermalization iterations (real time) */
#define THERMAL 20000

/* Output every NTH iteration (was 5000) */
#define NTH 500

/* Use all threads available on the computer */
#define THREADS 0

/* GPU allocation */
#ifdef USE_CUDA
int gpus[MAX_GPU];
int ngpus;
#endif

REAL rho0, mu0;
FILE *fpm = NULL, *fpp = NULL, *fp1 = NULL, *fp2 = NULL;

REAL complex random_start(void *NA, REAL x, REAL y, REAL z) {

  return SQRT(rho0 * drand48()) * CEXP(I * 2.0 * (drand48() - 0.5) * M_PI);
}

void print_stats(INT iter, wf *gwf, dft_ot_functional *otf, cgrid *potential_store, rgrid *rworkspace) {

  char buf[512];
  FILE *fp;
#ifdef KSPECTRUM
  static REAL *bins = NULL;
#endif
  REAL n, kin, pot;
  INT i;

  if(!bins) {
    if(!(bins = (REAL *) malloc(sizeof(REAL) * NBINS))) {
      fprintf(stderr, "Can't allocate memory for bins.\n");
      exit(1);
    }
  }

  printf("***** Statistics for iteration " FMT_I "\n", iter);
  grid_wf_average_occupation(gwf, bins, BINSTEP, NBINS, potential_store, 2);
  sprintf(buf, "occ-" FMT_I ".dat", iter);
  if(!(fp = fopen(buf, "w"))) {
    fprintf(stderr, "Can't open %s.\n", buf);
    exit(1);
  }
  for(i = 0; i < NBINS; i++)
    if(bins[i] != 0.0) fprintf(fp, FMT_R " " FMT_R "\n", (BINSTEP * (REAL) i) / GRID_AUTOANG, bins[i]);
  fclose(fp);
  n = grid_wf_norm(gwf);
#if 0
  kin = grid_wf_kinetic_energy_classical(gwf, otf->workspace1, otf->workspace2, DFT_EPS);
  if(otf->model & DFT_OT_BACKFLOW) {
    rgrid_zero(rworkspace);
    grid_wf_density(gwf, otf->density);
    dft_ot_energy_density_bf(otf, rworkspace, gwf, otf->density);
    kin += rgrid_integral(rworkspace);
  }
#else
  kin = grid_wf_energy(gwf, NULL);            /* Kinetic energy for gwf */
  dft_ot_energy_density(otf, rworkspace, gwf);
  pot = rgrid_integral(rworkspace) - mu0 * n;
  kin += pot;
#endif
  pot = (kin / n) * GRID_AUTOJ * GRID_AVOGADRO;
  printf("Thermal energy = " FMT_R " J/mol, T_BEC = " FMT_R " K, ", pot, grid_wf_temperature(gwf, TLAMBDA));
  // Search for the matching enthalpy
  kin = 0.0; // actually temperature here
  while(1) {
    if(dft_bulk_exp_enthalpy(kin, NULL, NULL) >= pot) break;
    kin += 0.01; // search with 0.01 K accuracy
  }
  printf("T_enth = " FMT_R " K\n", kin);

  pot = grid_wf_superfluid(gwf);
  printf("FRACTION: " FMT_R " " FMT_R " " FMT_R "\n", kin, pot, 1.0 - pot); // Temperature, superfluid fraction, normal fraction

  sprintf(buf, "thermal-" FMT_I, iter);
  cgrid_write_grid(buf, gwf->grid);

  /* The whole thing */
  grid_wf_KE(gwf, bins, BINSTEP, NBINS, otf->workspace1, otf->workspace2, otf->workspace3, otf->workspace4, DENS_EPS);
  sprintf(buf, "ke-" FMT_I ".dat", iter);
  if(!(fp = fopen(buf, "w"))) {
    fprintf(stderr, "Can't open %s.\n", buf);
    exit(1);
  }
  for(i = 1; i < NBINS; i++)  /* Leave out the DC component (= zero) */
    fprintf(fp, FMT_R " " FMT_R "\n", (BINSTEP * (REAL) i) / GRID_AUTOANG, bins[i] * GRID_AUTOK * GRID_AUTOANG); /* Angs^{-1} K*Angs */
  fclose(fp);

  /* Incompressible part */
  grid_wf_incomp_KE(gwf, bins, BINSTEP, NBINS, otf->workspace1, otf->workspace2, otf->workspace3, otf->workspace4, otf->workspace5, DENS_EPS);
  sprintf(buf, "ke-incomp-" FMT_I ".dat", iter);
  if(!(fp = fopen(buf, "w"))) {
    fprintf(stderr, "Can't open %s.\n", buf);
    exit(1);
  }
  for(i = 1; i < NBINS; i++)  /* Leave out the DC component (= zero) */
    fprintf(fp, FMT_R " " FMT_R "\n", (BINSTEP * (REAL) i) / GRID_AUTOANG, bins[i] * GRID_AUTOK * GRID_AUTOANG); /* Angs^{-1} K*Angs */
  fclose(fp);

  /* Compressible part */
  grid_wf_comp_KE(gwf, bins, BINSTEP, NBINS, otf->workspace1, otf->workspace2, otf->workspace3, otf->workspace4, DENS_EPS);
  sprintf(buf, "ke-comp-" FMT_I ".dat", iter);
  if(!(fp = fopen(buf, "w"))) {
    fprintf(stderr, "Can't open %s.\n", buf);
    exit(1);
  }
  for(i = 1; i < NBINS; i++)  /* Leave out the DC component (= zero) */
    fprintf(fp, FMT_R " " FMT_R "\n", (BINSTEP * (REAL) i) / GRID_AUTOANG, bins[i] * GRID_AUTOK * GRID_AUTOANG); /* Angs^{-1} K*Angs */
  fclose(fp);

  /* Bin |curl rho v|: small values for large rings and no strain, large values for small rings or high strain */
  grid_wf_probability_flux(gwf, otf->workspace1, otf->workspace2, otf->workspace3);
  rgrid_abs_rot(otf->workspace4, otf->workspace1, otf->workspace2, otf->workspace3);
  rgrid_abs_power(otf->workspace4, otf->workspace4, 1.0); // increase exponent for contrast
  sprintf(buf, "momentum-" FMT_I ".dat", iter);
  kin = rgrid_max(otf->workspace4); // max value for bin
  pot = kin / (REAL) NBINS;         // bin step
  rgrid_histogram(otf->workspace4, bins, NBINS, kin / (REAL) NBINS);
  if(!(fp = fopen(buf, "w"))) {
    fprintf(stderr, "Can't open %s.\n", buf);
    exit(1);
  }
  for(i = 0; i < NBINS; i++)
    fprintf(fp, FMT_R " " FMT_R "\n", pot * (REAL) i, bins[i]);   // write histogram
  fclose(fp);

  kin = grid_wf_energy(gwf, NULL);            /* Kinetic energy for gwf */
  dft_ot_energy_density(otf, rworkspace, gwf);
  pot = rgrid_integral(rworkspace) - mu0 * n;
  printf("Helium natoms       = " FMT_R " particles.\n", n);   /* Energy / particle in K */
  printf("Helium kinetic E    = " FMT_R " K\n", kin * GRID_AUTOK);  /* Print result in K */
  printf("Helium potential E  = " FMT_R " K\n", pot * GRID_AUTOK);  /* Print result in K */
  printf("Helium energy       = " FMT_R " K\n", (kin + pot) * GRID_AUTOK);  /* Print result in K */
  fflush(stdout);
}

int main(int argc, char **argv) {

  dft_ot_functional *otf;
  cgrid *potential_store;
  rgrid *rworkspace;
  wf *gwf;
#ifdef PC
  wf *gwfp;
#endif
  INT iter, i;
  grid_timer timer;
  REAL complex tstep;

  if(argc < 2) {
    fprintf(stderr, "Usage: thermal <gpu1> <gpu2> ...\n");
    exit(1);
  }

#ifdef USE_CUDA
  ngpus = argc-1;
  for(i = 0; i < ngpus; i++) 
    gpus[i] = atoi(argv[i+1]);

  cuda_enable(1, ngpus, gpus);
#endif

  /* Initialize threads & use wisdom */
  grid_set_fftw_flags(1);    // FFTW_MEASURE
  grid_threads_init(THREADS);
  grid_fft_read_wisdom(NULL);

  grid_wf_analyze_method(1);  // FD = 0, FFT = 1

  /* Allocate wave functions */
  if(!(gwf = grid_wf_alloc(NX, NY, NZ, STEP, DFT_HELIUM_MASS, WF_PERIODIC_BOUNDARY, WF_2ND_ORDER_FFT, "gwf"))) {
    fprintf(stderr, "Cannot allocate gwf.\n");
    exit(1);
  }
#ifdef PC
  if(!(gwfp = grid_wf_alloc(NX, NY, NZ, STEP, DFT_HELIUM_MASS, WF_PERIODIC_BOUNDARY, WF_2ND_ORDER_FFT, "gwfp"))) {
    fprintf(stderr, "Cannot allocate gwfp.\n");
    exit(1);
  }
#endif  

  /* Allocate OT functional */
  if(!(otf = dft_ot_alloc(FUNCTIONAL, gwf, DFT_MIN_SUBSTEPS, DFT_MAX_SUBSTEPS))) {
    fprintf(stderr, "Cannot allocate otf.\n");
    exit(1);
  }
  rho0 = dft_ot_bulk_density_pressurized(otf, PRESSURE);
  mu0 = dft_ot_bulk_chempot_pressurized(otf, PRESSURE);
  printf("Bulk mu0 = " FMT_R " K/atom, Bulk rho0 = " FMT_R " Angs^-3.\n", mu0 * GRID_AUTOK, rho0 / (GRID_AUTOANG * GRID_AUTOANG * GRID_AUTOANG));

  /* Allocate space for external potential */
  potential_store = cgrid_clone(gwf->grid, "potential_store"); /* temporary storage */
  rworkspace = rgrid_clone(otf->density, "external potential");

  /* 1. Random initial guess */
  cgrid_map(gwf->grid, random_start, gwf->grid);

  /* Make sure that we have enough workspaces reserved */
  if(!(otf->workspace2)) otf->workspace2 = rgrid_clone(otf->density, "OT workspace 2");
  if(!(otf->workspace3)) otf->workspace3 = rgrid_clone(otf->density, "OT workspace 3");
  if(!(otf->workspace4)) otf->workspace4 = rgrid_clone(otf->density, "OT Workspace 4");
  if(!(otf->workspace5)) otf->workspace5 = rgrid_clone(otf->density, "OT Workspace 5");

  /* 2. Cooling period */
  gwf->norm = gwfp->norm = HE_NORM;
  printf("Cooling...\n");
  tstep = (TS - I * ITS) / GRID_AUTOFS;
  for (iter = 0; iter < COOL; iter++) {
    grid_timer_start(&timer);
    if(iter == 10) grid_fft_write_wisdom(NULL);
#ifdef PC
    /* Predict-Correct */
    cgrid_constant(potential_store, -mu0);
    dft_ot_potential(otf, potential_store, gwf);
    grid_wf_propagate_predict(gwf, gwfp, potential_store, tstep);

    cgrid_add(potential_store, -mu0);
    dft_ot_potential(otf, potential_store, gwfp);
    cgrid_multiply(potential_store, 0.5);  // Use (current + future) / 2
    grid_wf_propagate_correct(gwf, potential_store, tstep);
#else /* PC */
    /* Propagate */
    cgrid_constant(potential_store, -mu0);
    dft_ot_potential(otf, potential_store, gwf);
    grid_wf_propagate(gwf, potential_store, tstep);
#endif /* PC */
    grid_wf_normalize(gwf);    
    printf("Cooling Iteration " FMT_I " - Wall clock time = " FMT_R " seconds.\n", iter, grid_timer_wall_clock_time(&timer)); fflush(stdout);
    if(iter && !(iter % NTH)) print_stats(iter, gwf, otf, potential_store, rworkspace);
  }

  printf("Equilibriating...\n");
  tstep = TS / GRID_AUTOFS;
  for (; iter < COOL + THERMAL; iter++) {
    grid_timer_start(&timer);
#ifdef PC
    /* Predict-Correct */
    cgrid_constant(potential_store, -mu0);
    dft_ot_potential(otf, potential_store, gwf);
#ifdef DEALIAS
    cgrid_fft(potential_store);
    cgrid_dealias(potential_store, 1);
    cgrid_inverse_fft_norm(potential_store);
#endif
    grid_wf_propagate_predict(gwf, gwfp, potential_store, tstep);

    cgrid_add(potential_store, -mu0);
    dft_ot_potential(otf, potential_store, gwfp);
    cgrid_multiply(potential_store, 0.5);  // Use (current + future) / 2
#ifdef DEALIAS
    cgrid_fft(potential_store);
    cgrid_dealias(potential_store, 1);
    cgrid_inverse_fft_norm(potential_store);
#endif
    grid_wf_propagate_correct(gwf, potential_store, tstep);
#else /* PC */
    /* Propagate */
    cgrid_constant(potential_store, -mu0);
    dft_ot_potential(otf, potential_store, gwf);
    grid_wf_propagate(gwf, potential_store, tstep);
#endif /* PC */

#ifdef DEALIAS
    cgrid_fft(gwf->grid);
    cgrid_dealias(gwf->grid, 1);
    cgrid_inverse_fft_norm(gwf->grid);
#endif

    if(iter == 0 || !(iter % NTH)) print_stats(iter, gwf, otf, potential_store, rworkspace);
  }
  return 0;
}
