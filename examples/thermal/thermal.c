/*
 * Bulk helium at a given temperature (explicit thermal excitations).
 *
 * 1. Start from random initial order parameter.
 * 2. Cooling phase with a mixture of real and imaginary time (the ratio defines the cooling rate).
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

/* Time integration method */
#define TIMEINT WF_2ND_ORDER_FFT
//#define TIMEINT WF_2ND_ORDER_CN

/* FD(0) or FFT(1) properties */
#define PROPERTIES 0

/* Time step for real and imaginary time */
#define TS 0.1 /* fs */
#define ITS1 10.0 /* ifs */
#define ITS2 (0.0 * TS * 0.1) /* ifs (10% of TS works but still a bit too fast) */

/* Grid */
#define NX 128
#define NY 128
#define NZ 128
#define STEP 2.0

/* Boundary handling (continuous bulk or spherical droplet) */
//#define RADIUS (0.9 * STEP * ((REAL) NX) / 2.0)

/* E(k) */
#define KSPECTRUM /**/
#define NBINS 512
#define BINSTEP 0.1
#define DENS_EPS 1E-3

/* Predict-correct? (not available for 4th order splitting) */
#define PC

/* Use dealiasing during real time propagation? */
#define DEALIAS
#define DEALIAS_VAL (2.8 * GRID_AUTOANG)

/* Functional to use */

/* Coarse functional to get to TEMP_SWITCH - numerically more stable */
//#define FUNCTIONAL (DFT_OT_PLAIN | DFT_OT_KC | DFT_OT_BACKFLOW | DFT_OT_HD)
#define FUNCTIONAL (DFT_OT_PLAIN)
//#define FUNCTIONAL (DFT_GP2)

/* Fine functional to use below 3.0 K - less stable */
//#define FUNCTIONAL_FINE (DFT_OT_PLAIN | DFT_OT_KC | DFT_OT_BACKFLOW | DFT_OT_HD)
#define FUNCTIONAL_FINE (DFT_OT_PLAIN)
//#define FUNCTIONAL_FINE (DFT_GP2)

/* Switch over temperature from FUNCTIONAL to FUNCTIONAL_FINE */
#define TEMP_SWITCH 99.0

/* B-E temperature */
#define BE_TEMP 1.0

/* Pressure */
#define PRESSURE (0.0 / GRID_AUTOBAR)

/* Number of cooling iterations */
#define IITER 500L

/* Number of real time iterations */
#define RITER 200000000L

/* Output every NTH iteration (was 1000) */
#define NTH 6000L

/* How many CPU cores to use (0 = all available) */
#define THREADS 0

/* Random seed (drand48) */
#define RANDOM_SEED 1234L

/* Write grid files? */
#define WRITE_GRD 6000L

/* Enable / disable GPU */
//#undef USE_CUDA

/* GPU allocation */
#ifdef USE_CUDA
int gpus[MAX_GPU];
int ngpus;
#endif

dft_ot_functional *otf;
REAL rho0, mu0;
REAL complex tstep;
FILE *fpm = NULL, *fpp = NULL, *fp1 = NULL, *fp2 = NULL;
rgrid *epot = NULL;

/* Given |k|,  T and total number of atoms, returns the B-E population */
REAL population(REAL k, REAL T, REAL N) {

  REAL mu, eps;

#ifdef DEALIAS_VAL
  if(k > DEALIAS_VAL) return 0.0;
#endif
  mu = -GRID_AUKB * T * LOG(1.0 + 1.0 / (N * dft_exp_bulk_superfluid_fraction(T)));
  eps = dft_ot_bulk_dispersion(otf, &k, rho0);
//  eps = dft_exp_bulk_dispersion(k / GRID_AUTOANG) / GRID_AUTOK;
  return 1.0 / (EXP((eps - mu) / (GRID_AUKB * T)) - 1.0);
}

REAL complex be_pop(void *xx, REAL kx, REAL ky, REAL kz) {

  REAL *N = (REAL *) xx;

// population x random phase
  return SQRT(population(SQRT(kx * kx + ky * ky + kz * kz), BE_TEMP, *N)) * CEXP(2.0 * M_PI * (2.0 * (drand48() - 0.5)) * I);
}

/* External potential */
#define A1 (100.0 / GRID_AUTOK)
#define A2 (1.0)
REAL ext_pot(void *NA, REAL x, REAL y, REAL z) {

#ifdef RADIUS
  REAL r = SQRT(x*x + y*y + z*z);

  if(r > RADIUS) return A1;
  return A1 * EXP(-A2 * (RADIUS - r));
#else
  return 0.0;
#endif
}

REAL get_energy(wf *gwf, wf *gnd, dft_ot_functional *otf, rgrid *rworkspace) {

  REAL tot, tot_gnd, n;

  n = grid_wf_norm(gwf);
  dft_ot_energy_density(otf, rworkspace, gwf);
  tot = (grid_wf_energy(gwf, epot) + rgrid_integral(rworkspace)) / n;

  n = grid_wf_norm(gnd);
  dft_ot_energy_density(otf, rworkspace, gnd);
  tot_gnd = (grid_wf_energy(gnd, epot) + rgrid_integral(rworkspace)) / n;

  return (tot - tot_gnd) * GRID_AUTOJ * GRID_AVOGADRO; // J / mol
}

void dealias(cgrid *grid) {

#ifdef DEALIAS
    cgrid_fft(grid);
#ifdef DEALIAS_VAL
    cgrid_dealias2(grid, DEALIAS_VAL);
#else
    cgrid_dealias(grid, 2);
#endif /* DEALIAS_VAL */
    cgrid_inverse_fft_norm(grid);
#endif /* DEALIAS */
}

void print_stats(INT iter, wf *gwf, wf *gnd, dft_ot_functional *otf, cgrid *potential_store, rgrid *rworkspace) {

  char buf[512];
  FILE *fp;
#ifdef KSPECTRUM
  static REAL *bins = NULL;
#endif
  REAL energy, tmp, tmp2;
  INT i, upd = 0;

  if(!bins) {
    if(!(bins = (REAL *) malloc(sizeof(REAL) * NBINS))) {
      fprintf(stderr, "Can't allocate memory for bins.\n");
      exit(1);
    }
  }

  printf("***** Statistics for iteration " FMT_I "\n", iter);
  grid_wf_average_occupation(gwf, bins, BINSTEP, NBINS, potential_store);
  sprintf(buf, "occ-" FMT_I ".dat", iter);
  if(!(fp = fopen(buf, "w"))) {
    fprintf(stderr, "Can't open %s.\n", buf);
    exit(1);
  }
  for(i = 0; i < NBINS; i++)
    if(bins[i] != 0.0) fprintf(fp, FMT_R " " FMT_R "\n", (BINSTEP * (REAL) i) / GRID_AUTOANG, bins[i]);
  fclose(fp);

  energy = get_energy(gwf, gnd, otf, rworkspace);
  tmp = grid_wf_superfluid2(gwf, gnd);
  tmp2 = dft_exp_bulk_enthalpy_inverse(energy);
  printf("Thermal energy = " FMT_R " J/mol, T_BEC = " FMT_R " K, ", energy, dft_exp_bulk_superfluid_fraction_inverse(tmp));
  printf("T_enth = " FMT_R " K\n", tmp2);

  printf("FRACTION: " FMT_R " " FMT_R " " FMT_R "\n", tmp2, tmp, 1.0 - tmp); // Temperature, superfluid fraction, normal fraction

  if(tmp2 < TEMP_SWITCH) {
    printf("Below switching temperature.\n");
    upd = 1;
  }

  /* Kinetic energy */
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
  tmp = rgrid_max(otf->workspace4); // max value for bin
  tmp2 = tmp / (REAL) NBINS;         // bin step
  rgrid_histogram(otf->workspace4, bins, NBINS, tmp / (REAL) NBINS);
  if(!(fp = fopen(buf, "w"))) {
    fprintf(stderr, "Can't open %s.\n", buf);
    exit(1);
  }
  for(i = 0; i < NBINS; i++)
    fprintf(fp, FMT_R " " FMT_R "\n", tmp2 * (REAL) i, bins[i]);   // write histogram
  fclose(fp);

  tmp = grid_wf_energy(gwf, epot);            /* Kinetic energy for gwf */
  dft_ot_energy_density(otf, rworkspace, gwf);
// TODO: not quite right for droplets, use get_energy()
  tmp2 = rgrid_integral(rworkspace) - dft_ot_bulk_energy(otf, rho0) * (STEP * STEP * STEP * (REAL) (NX * NY * NZ));
//  tmp2 = rgrid_integral(rworkspace) - mu0 * grid_wf_norm(gwf);
  printf("Helium natoms       = " FMT_R " particles.\n", grid_wf_norm(gwf));   /* Energy / particle in K */
  printf("Helium kinetic E    = " FMT_R " K\n", tmp * GRID_AUTOK);  /* Print result in K */
  printf("Helium classical KE = " FMT_R " K\n", grid_wf_kinetic_energy_classical(gwf, otf->workspace1, otf->workspace2, DENS_EPS) * GRID_AUTOK);
  printf("Helium quantum KE   = " FMT_R " K\n", grid_wf_kinetic_energy_qp(gwf, otf->workspace1, otf->workspace2, otf->workspace3) * GRID_AUTOK);
  printf("Helium potential E  = " FMT_R " K\n", tmp2 * GRID_AUTOK);  /* Print result in K */
  printf("Helium energy       = " FMT_R " K\n", (tmp + tmp2) * GRID_AUTOK);  /* Print result in K */
  fflush(stdout);

  if(upd) {
    otf->model = FUNCTIONAL_FINE;  // Time to switch to FINE functional
    printf("Switched to fine functional.\n");
    tstep = CREAL(tstep);
  }
}

int main(int argc, char **argv) {

  cgrid *potential_store;
  rgrid *rworkspace;
  wf *gwf, *gnd;
#ifdef PC
  wf *gwfp;
#endif
  INT iter, i;
  grid_timer timer;

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

  grid_wf_analyze_method(PROPERTIES);

  srand48(RANDOM_SEED);

  /* Allocate wave functions */
  if(!(gwf = grid_wf_alloc(NX, NY, NZ, STEP, DFT_HELIUM_MASS, WF_PERIODIC_BOUNDARY, TIMEINT, "gwf"))) {
    fprintf(stderr, "Cannot allocate gwf.\n");
    exit(1);
  }
  if(!(gnd = grid_wf_alloc(NX, NY, NZ, STEP, DFT_HELIUM_MASS, WF_PERIODIC_BOUNDARY, TIMEINT, "gnd"))) {
    fprintf(stderr, "Cannot allocate gnd.\n");
    exit(1);
  }
#ifdef PC
  if(!(gwfp = grid_wf_alloc(NX, NY, NZ, STEP, DFT_HELIUM_MASS, WF_PERIODIC_BOUNDARY, TIMEINT, "gwfp"))) {
    fprintf(stderr, "Cannot allocate gwfp.\n");
    exit(1);
  }
#endif  

  /* Allocate OT functional */
  if(!(otf = dft_ot_alloc(FUNCTIONAL_FINE, gwf, DFT_MIN_SUBSTEPS, DFT_MAX_SUBSTEPS))) {
    fprintf(stderr, "Cannot allocate otf.\n");
    exit(1);
  }
  otf->model = FUNCTIONAL;

  rho0 = dft_ot_bulk_density_pressurized(otf, PRESSURE);
  mu0 = dft_ot_bulk_chempot_pressurized(otf, PRESSURE);
  printf("Bulk mu0 = " FMT_R " K/atom, Bulk rho0 = " FMT_R " Angs^-3.\n", mu0 * GRID_AUTOK, rho0 / (GRID_AUTOANG * GRID_AUTOANG * GRID_AUTOANG));

  /* Allocate space for external potential */
  potential_store = cgrid_clone(gwf->grid, "potential_store"); /* temporary storage */
  rworkspace = rgrid_clone(otf->density, "rworkspae");
  epot = rgrid_clone(otf->density, "external potential");

  /* Make sure that we have enough workspaces reserved */
  if(!(otf->workspace2)) otf->workspace2 = rgrid_clone(otf->density, "OT workspace 2");
  if(!(otf->workspace3)) otf->workspace3 = rgrid_clone(otf->density, "OT workspace 3");
  if(!(otf->workspace4)) otf->workspace4 = rgrid_clone(otf->density, "OT Workspace 4");
  if(!(otf->workspace5)) otf->workspace5 = rgrid_clone(otf->density, "OT Workspace 5");

  printf("Fourier space range +-: " FMT_R " Angs^-1 with step " FMT_R " Angs^-1.\n", 2.0 * M_PI / (STEP * GRID_AUTOANG), 2.0 * M_PI / (STEP * GRID_AUTOANG * (REAL) NX));

  /* 1. Ground state optimization */
  rgrid_map(epot, ext_pot, NULL);
  cgrid_constant(gnd->grid, SQRT(rho0));

  printf("Ground state calculation...\n");
  for(iter = 0; iter < IITER; iter++) {
    /* Propagate gnd */
    cgrid_constant(potential_store, -mu0);
    grid_add_real_to_complex_re(potential_store, epot);
    dft_ot_potential(otf, potential_store, gnd);
    grid_wf_propagate(gnd, potential_store, -I * ITS1 / GRID_AUTOFS);
    cgrid_multiply(gnd->grid, SQRT(rho0) / CABS(cgrid_value(gnd->grid, 0.0, 0.0, 0.0)));
  }
#ifdef WRITE_GRD
  cgrid_write_grid("gnd", gnd->grid);
#endif
  gnd->norm = gwf->norm = cgrid_integral_of_square(gnd->grid);

  /* 2. Start with ground state + random perturbation */
  cgrid_mapk(gwf->grid, be_pop, &(gwf->norm));
  cgrid_inverse_fft(gwf->grid);
  cgrid_multiply(gwf->grid, 1.0 / SQRT(STEP * STEP * STEP * NX * NY * NZ));
  grid_wf_normalize(gwf);
  dft_ot_energy_density(otf, rworkspace, gwf);

  /* 3. Real time simulation */
  printf("Dynamics...\n");
  tstep = (TS - ITS2 * I) / GRID_AUTOFS;
  grid_timer_start(&timer);
#ifdef PC
  gwfp->norm = gwf->norm;
#endif
  for (iter = 0; iter < RITER; iter++) {

    if(iter == 100) grid_fft_write_wisdom(NULL);

    dealias(gwf->grid);

    if(!(iter % NTH)) {
      printf("Wall clock time = " FMT_R " seconds.\n", grid_timer_wall_clock_time(&timer)); fflush(stdout);
      print_stats(iter, gwf, gnd, otf, potential_store, rworkspace);
      grid_timer_start(&timer);
    }

    if(CIMAG(tstep) != 0.0) grid_wf_normalize(gwf);

#ifdef PC
    /* Predict-Correct */
    cgrid_constant(potential_store, -mu0);
    grid_add_real_to_complex_re(potential_store, epot);
    dft_ot_potential(otf, potential_store, gwf);

    dealias(potential_store);

    grid_wf_propagate_predict(gwf, gwfp, potential_store, tstep);

    dealias(gwfp->grid);

    if(CIMAG(tstep) != 0.0) grid_wf_normalize(gwfp);

    cgrid_add(potential_store, -mu0); // not exact chem. pot. hence normalization above needed
    grid_add_real_to_complex_re(potential_store, epot);
    dft_ot_potential(otf, potential_store, gwfp);
    cgrid_multiply(potential_store, 0.5);  // Use (current + future) / 2

    dealias(potential_store);

    grid_wf_propagate_correct(gwf, potential_store, tstep);

#else /* PC */

    /* Propagate */
    cgrid_constant(potential_store, -mu0);
    grid_add_real_to_complex_re(potential_store, epot);
    dft_ot_potential(otf, potential_store, gwf);

    dealias(potential_store);

    grid_wf_propagate(gwf, potential_store, tstep);

#endif /* PC */

#ifdef WRITE_GRD
    if(!(iter % WRITE_GRD)) {
      char buf[512];
      sprintf(buf, "thermal-" FMT_I, iter);
      cgrid_write_grid(buf, gwf->grid);
    }
#endif

  }

  return 0;
}
