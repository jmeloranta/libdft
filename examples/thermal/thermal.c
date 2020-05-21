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

/* Time integration method */
#define TIMEINT WF_2ND_ORDER_FFT

/* Time step for real and imaginary time */
#define TS 1.0 /* fs */
#define ITS (1.0 * TS) /* fs (10% of TS works but still a bit too fast) */

/* Grid */
#define NX 256
#define NY 256
#define NZ 256
#define STEP 0.5

/* Boundary handling (continuous bulk or spherical droplet) */
#define RADIUS (0.8 * STEP * ((REAL) NX) / 2.0)

/* E(k) */
#define KSPECTRUM /**/
#define NBINS 512
#define BINSTEP 0.1
#define DENS_EPS 1E-3

/* Predict-correct? (not available for 4th order splitting) */
#define PC

/* Use dealiasing during real time propagation? */
//#define DEALIAS
#define DEALIAS_VAL (2.8 * GRID_AUTOANG)

/* Functional to use */

/* Coarse functional to get to 3.0 K - numerically stable */
#define FUNCTIONAL (DFT_OT_PLAIN)

/* Fine functional to use below 3.0 K - less stable */
//#define FUNCTIONAL_FINE (DFT_OT_PLAIN | DFT_OT_KC | DFT_OT_BACKFLOW | DFT_OT_HD)
#define FUNCTIONAL_FINE (DFT_OT_PLAIN)

/* Switch over temperature from FUNCTIONAL to FUNCTIONAL_FINE */
#define TEMP_SWITCH 1.5

/* Pressure */
#define PRESSURE (0.0 / GRID_AUTOBAR)

/* Normalization - Since we have the thermal excitations, the chemical potential method is not good... Hence explicit normalization */
#ifdef RADIUS
#define HE_NORM (rho0 * (4.0 / 3.0) * M_PI * RADIUS * RADIUS * RADIUS)
#else
#define HE_NORM (rho0 * ((REAL) NX) * STEP * ((REAL) NY) * STEP * ((REAL) NZ) * STEP)
#endif

/* The number of cooling iterations (mixture of real and imaginary) */
#define COOL 2000000

/* Output every NTH iteration (was 5000) */
#define NTH 200

/* Use all threads available on the computer */
#define THREADS 0

/* Random seed (drand48) */
#define RANDOM_SEED 123467L

/* Write grid files? */
// #define WRITE_GRD

/* Disable cuda ? (TODO: Strange issue of not getting the correct kinetic energy with FFTW; CUFFT works OK) */
// #undef USE_CUDA

/* GPU allocation */
#ifdef USE_CUDA
int gpus[MAX_GPU];
int ngpus;
#endif

REAL rho0, mu0;
REAL complex tstep;
FILE *fpm = NULL, *fpp = NULL, *fp1 = NULL, *fp2 = NULL;

/* Random phases and equal amplitude for each k-point the reciprocal space */
REAL complex random_start(void *NA, REAL x, REAL y, REAL z) {

#ifdef RADIUS
  if(x*x + y*y + z*z > RADIUS*RADIUS) return 0.0;
#endif
  return SQRT(rho0) * CEXP(I * 2.0 * (drand48() - 0.5) * M_PI);
}

REAL get_energy(wf *gwf, dft_ot_functional *otf, rgrid *rworkspace) {

  REAL kin, pot, n;

  kin = grid_wf_energy(gwf, NULL);            /* Kinetic energy for gwf */
  dft_ot_energy_density(otf, rworkspace, gwf);
  pot = rgrid_integral(rworkspace) - mu0 * grid_wf_norm(gwf);
  kin += pot;
  n = grid_wf_norm(gwf);
  return (kin / n) * GRID_AUTOJ * GRID_AVOGADRO; // J / mol
}

void print_stats(INT iter, wf *gwf, dft_ot_functional *otf, cgrid *potential_store, rgrid *rworkspace) {

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
  grid_wf_average_occupation(gwf, bins, BINSTEP, NBINS, potential_store, 2);
  sprintf(buf, "occ-" FMT_I ".dat", iter);
  if(!(fp = fopen(buf, "w"))) {
    fprintf(stderr, "Can't open %s.\n", buf);
    exit(1);
  }
  for(i = 0; i < NBINS; i++)
    if(bins[i] != 0.0) fprintf(fp, FMT_R " " FMT_R "\n", (BINSTEP * (REAL) i) / GRID_AUTOANG, bins[i]);
  fclose(fp);

  energy = get_energy(gwf, otf, rworkspace);
  tmp = grid_wf_superfluid(gwf);
  tmp2 = dft_exp_bulk_enthalpy_inverse(energy);
  printf("Thermal energy = " FMT_R " J/mol, T_BEC = " FMT_R " K, ", energy, dft_exp_bulk_superfluid_fraction_inverse(tmp));
  printf("T_enth = " FMT_R " K\n", tmp2);

  printf("FRACTION: " FMT_R " " FMT_R " " FMT_R "\n", tmp2, tmp, 1.0 - tmp); // Temperature, superfluid fraction, normal fraction

  if(tmp2 < TEMP_SWITCH) {
    printf("Below swtiching temperature.\n");
    upd = 1;
  }

#ifdef WRITE_GRD
  sprintf(buf, "thermal-" FMT_I, iter);
  cgrid_write_grid(buf, gwf->grid);
#endif

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

  tmp = grid_wf_energy(gwf, NULL);            /* Kinetic energy for gwf */
  dft_ot_energy_density(otf, rworkspace, gwf);
  tmp2 = rgrid_integral(rworkspace) - mu0 * grid_wf_norm(gwf);
  printf("Helium natoms       = " FMT_R " particles.\n", grid_wf_norm(gwf));   /* Energy / particle in K */
  printf("Helium kinetic E    = " FMT_R " K\n", tmp * GRID_AUTOK);  /* Print result in K */
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

  dft_ot_functional *otf;
  cgrid *potential_store;
  rgrid *rworkspace;
  wf *gwf;
#ifdef PC
  wf *gwfp;
#endif
  INT iter, i;

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

  srand48(RANDOM_SEED);

  /* Allocate wave functions */
  if(!(gwf = grid_wf_alloc(NX, NY, NZ, STEP, DFT_HELIUM_MASS, WF_PERIODIC_BOUNDARY, TIMEINT, "gwf"))) {
    fprintf(stderr, "Cannot allocate gwf.\n");
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
  rworkspace = rgrid_clone(otf->density, "external potential");

  /* 1. Random initial guess -- here everything on CPU all the way to cgrid_multiply */
  cgrid_map(gwf->grid, random_start, gwf->grid);
#ifdef DEALIAS_VAL
  cgrid_dealias2(gwf->grid, DEALIAS_VAL); // Remove high wavenumber components from the initial guess
  cgrid_fftw_inv(gwf->grid);
  cgrid_multiply(gwf->grid, gwf->grid->fft_norm);
#endif

  /* Make sure that we have enough workspaces reserved */
  if(!(otf->workspace2)) otf->workspace2 = rgrid_clone(otf->density, "OT workspace 2");
  if(!(otf->workspace3)) otf->workspace3 = rgrid_clone(otf->density, "OT workspace 3");
  if(!(otf->workspace4)) otf->workspace4 = rgrid_clone(otf->density, "OT Workspace 4");
  if(!(otf->workspace5)) otf->workspace5 = rgrid_clone(otf->density, "OT Workspace 5");

  /* 2. Cooling period */
  gwf->norm 
#ifdef PC
    = gwfp->norm 
#endif
    = HE_NORM;
  printf("Cooling...\n");
  tstep = (TS - I * ITS) / GRID_AUTOFS;
  for (iter = 0; iter < COOL; iter++) {

    if(iter == 100) grid_fft_write_wisdom(NULL);

#ifdef DEALIAS
    cgrid_fft(gwf->grid);
    cgrid_dealias2(gwf->grid, DEALIAS_VAL);
    cgrid_inverse_fft_norm(gwf->grid);
#endif

#ifdef PC
    /* Predict-Correct */
    cgrid_constant(potential_store, -mu0);
    dft_ot_potential(otf, potential_store, gwf);
#ifdef DEALIAS
    cgrid_fft(potential_store);
    cgrid_dealias2(potential_store, DEALIAS_VAL);
    cgrid_inverse_fft_norm(potential_store);
#endif
    grid_wf_propagate_predict(gwf, gwfp, potential_store, tstep);
    grid_wf_normalize(gwfp);

    cgrid_add(potential_store, -mu0); // not exact chem. pot. hence normalization above needed
    dft_ot_potential(otf, potential_store, gwfp);
    cgrid_multiply(potential_store, 0.5);  // Use (current + future) / 2
#ifdef DEALIAS
    cgrid_fft(potential_store);
    cgrid_dealias2(potential_store, DEALIAS_VAL);
    cgrid_inverse_fft_norm(potential_store);
#endif
    grid_wf_propagate_correct(gwf, potential_store, tstep);
#else /* PC */
    /* Propagate */
    cgrid_constant(potential_store, -mu0);
    dft_ot_potential(otf, potential_store, gwf);
#ifdef DEALIAS
    cgrid_fft(potential_store);
    cgrid_dealias2(potential_store, DEALIAS_VAL);
    cgrid_inverse_fft_norm(potential_store);
#endif
    grid_wf_propagate(gwf, potential_store, tstep);
#endif /* PC */
    grid_wf_normalize(gwf);

    if(iter == 0 || !(iter % NTH))
      print_stats(iter, gwf, otf, potential_store, rworkspace);
  }

  return 0;
}
