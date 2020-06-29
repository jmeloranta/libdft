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
#define TS 1.0 /* fs */
#define ITS (0.001 * TS) /* ifs (10% of TS works but still a bit too fast); 0.0001 * TS */
#define TS_SWITCH 0.05 /* fs */
#define ITS_SWITCH (0.001 * TS_SWITCH) /* fs */

/* Grid */
#define NX 128
#define NY 128
#define NZ 128
#define STEP 0.5

/* E(k) */
#define KSPECTRUM /**/
#define NBINS 512
#define BINSTEP (2.0 * M_PI / (NX * STEP))   // Assumes NX = NY = NZ
#define DENS_EPS 1E-3

/* Predict-correct? (not available for 4th order splitting) */
#define PC

/* Use dealiasing during real time propagation? */
//#define DEALIAS
//#define DEALIAS_VAL (2.6 * GRID_AUTOANG)

/* Functional to use */
/* Coarse functional to get to TEMP_SWITCH - numerically more stable */
//#define FUNCTIONAL (DFT_OT_PLAIN | DFT_OT_KC | DFT_OT_BACKFLOW | DFT_OT_HD)
#define FUNCTIONAL (DFT_OT_PLAIN)
//#define FUNCTIONAL (DFT_GP2)

/* Fine functional to use below 3.0 K - less stable */
#define FUNCTIONAL_FINE (DFT_OT_PLAIN | DFT_OT_KC | DFT_OT_BACKFLOW | DFT_OT_HD)
//#define FUNCTIONAL_FINE (DFT_OT_PLAIN)
//#define FUNCTIONAL_FINE (DFT_GP2)

/* Switch over temperature from FUNCTIONAL to FUNCTIONAL_FINE */
#define TEMP_SWITCH 2.3

/* Pressure */
#define PRESSURE (0.0 / GRID_AUTOBAR)

/* Number of real time iterations */
#define RITER 200000000L

/* Output every NTH iteration (was 1000) */
#define NTH 2000L

/* How many CPU cores to use (0 = all available) */
#define THREADS 0

/* Random seed (drand48) */
#define RANDOM_SEED 1234L

/* Write grid files? */
//#define WRITE_GRD 1000L

/* Enable / disable GPU */
// #undef USE_CUDA

/* Temperature search accuracy (K) */
#define SEARCH_ACC 1E-4

/* Random noise scale */
//#define SCALE (5000.0 / GRID_AUTOK)

/* Roton energy */
#define ROTON_E (12.0 / GRID_AUTOK)
#define ROTON_K (1.9 * GRID_AUTOANG)

/* GPU allocation */
#ifdef USE_CUDA
int gpus[MAX_GPU];
int ngpus;
#endif

dft_ot_functional *otf;
REAL rho0, mu0;
REAL complex tstep;
FILE *fpm = NULL, *fpp = NULL, *fp1 = NULL, *fp2 = NULL;

REAL temperature(REAL occ) {

  return -ROTON_E / (LOG(occ) * GRID_AUKB);
}

/* Random initial guess with <P> = 0 */
void initial_guess(cgrid *grid) {

  INT i, j, k, nx = grid->nx, ny = grid->ny, nz = grid->nz, nx2 = nx / 2, ny2 = ny / 2, nz2 = nz / 2;
  REAL complex tval;
  
#ifdef USE_CUDA
  if(cuda_status()) cuda_remove_block(grid->value, 0);
#endif
  cgrid_zero(grid);
  for(i = 0; i <= nx2; i++)
    for(j = 0; j <= ny2; j++)
      for(k = 0; k <= nz2; k++) {
        if(i == 0 && j == 0 && k == 0) cgrid_value_to_index(grid, i, j, k, SQRT(rho0));
        else {

          tval = SQRT(rho0) * CEXP(I * 2.0 * (drand48() - 0.5) * M_PI);
          cgrid_value_to_index(grid, i, j, k, tval);
// comment for same phases for +- k
          tval = SQRT(rho0) * CEXP(I * 2.0 * (drand48() - 0.5) * M_PI);
          cgrid_value_to_index(grid, nx - i - 1, ny - j - 1, nz - k - 1, tval);
        }
      }
}

REAL get_energy(wf *gwf, dft_ot_functional *otf, rgrid *rworkspace) {

  REAL tot, tot_gnd, n;

  n = grid_wf_norm(gwf);
  dft_ot_energy_density(otf, rworkspace, gwf);
  tot = (grid_wf_energy(gwf, NULL) + rgrid_integral(rworkspace)) / n;

  tot_gnd = (dft_ot_bulk_energy(otf, rho0) * (STEP * STEP * STEP * (REAL) (NX * NY * NZ))) / n;

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

INT upd = 0;

void print_stats(INT iter, wf *gwf, wf *gwfp, dft_ot_functional *otf, cgrid *potential_store, rgrid *rworkspace) {

  char buf[512];
  FILE *fp;
#ifdef KSPECTRUM
  static REAL *bins = NULL;
#endif
  REAL energy, ke_tot, pe_tot, ke_qp, ke_cl, natoms, tmp, tmp2, temp;
  INT i;

  if(!bins) {
    if(!(bins = (REAL *) malloc(sizeof(REAL) * NBINS))) {
      fprintf(stderr, "Can't allocate memory for bins.\n");
      exit(1);
    }
  }

  natoms = grid_wf_norm(gwf);

  printf("***** Statistics for iteration " FMT_I "\n", iter);

  energy = get_energy(gwf, otf, rworkspace);
  temp = dft_exp_bulk_enthalpy_inverse(energy, SEARCH_ACC);
  printf("Thermal energy = " FMT_R " J/mol Tenth = " FMT_R " K ", energy, temp);

  if(temp <= TEMP_SWITCH) upd = 1;

  grid_wf_average_occupation(gwf, bins, BINSTEP, NBINS, potential_store);
  sprintf(buf, "occ-" FMT_I ".dat", iter);
  if(!(fp = fopen(buf, "w"))) {
    fprintf(stderr, "Can't open %s.\n", buf);
    exit(1);
  }
  for(i = 0; i < NBINS; i++)
    if(bins[i] != 0.0) fprintf(fp, FMT_R " " FMT_R "\n", (BINSTEP * (REAL) i) / GRID_AUTOANG, bins[i]);
  fclose(fp);
  printf("Occ = " FMT_R "\n", bins[(INT) (0.5 + ROTON_K / BINSTEP)]);

  printf("Temperature = " FMT_R " K\n", temperature(bins[(INT) (0.5 + ROTON_K / BINSTEP)]));

  grid_wf_total_occupation(gwf, bins, BINSTEP, NBINS, potential_store);
  sprintf(buf, "tocc-" FMT_I ".dat", iter);
  if(!(fp = fopen(buf, "w"))) {
    fprintf(stderr, "Can't open %s.\n", buf);
    exit(1);
  }
  for(i = 0; i < NBINS; i++)
    if(bins[i] != 0.0) fprintf(fp, FMT_R " " FMT_R "\n", (BINSTEP * (REAL) i) / GRID_AUTOANG, bins[i]);
  fclose(fp);

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
  if (tmp > 1E-12) {
    tmp2 = tmp / (REAL) NBINS;         // bin step
    rgrid_histogram(otf->workspace4, bins, NBINS, tmp / (REAL) NBINS);
    if(!(fp = fopen(buf, "w"))) {
      fprintf(stderr, "Can't open %s.\n", buf);
      exit(1);
    }
    for(i = 0; i < NBINS; i++)
      fprintf(fp, FMT_R " " FMT_R "\n", tmp2 * (REAL) i, bins[i]);   // write histogram
    fclose(fp);
  }

  ke_tot = grid_wf_energy(gwf, NULL);
  dft_ot_energy_density(otf, rworkspace, gwf);
  pe_tot = rgrid_integral(rworkspace) - dft_ot_bulk_energy(otf, rho0) * (STEP * STEP * STEP * (REAL) (NX * NY * NZ));
  ke_qp = grid_wf_kinetic_energy_qp(gwf, otf->workspace1, otf->workspace2, otf->workspace3);
  ke_cl = ke_tot - ke_qp;

  if(otf->model & DFT_OT_BACKFLOW) {
    REAL tmp;
    rgrid_zero(rworkspace);
    grid_wf_density(gwf, otf->density);    
    dft_ot_energy_density_bf(otf, rworkspace, gwf, otf->density);
    tmp = rgrid_integral(rworkspace);
    ke_cl += tmp;
    ke_tot += tmp;
  }
  if(otf->model & DFT_OT_KC) {
    REAL tmp;
    rgrid_zero(rworkspace);
    grid_wf_density(gwf, otf->density);
    dft_ot_energy_density_kc(otf, rworkspace, gwf, otf->density); 
    tmp = rgrid_integral(rworkspace);
    ke_qp += tmp;
    ke_tot += tmp;
  }
  
  printf("Helium natoms       = " FMT_R " particles.\n", natoms);
  printf("Helium kinetic E    = " FMT_R " K\n", ke_tot * GRID_AUTOK);
  printf("Helium classical KE = " FMT_R " K\n", ke_cl * GRID_AUTOK);
  printf("Helium quantum KE   = " FMT_R " K\n", ke_qp * GRID_AUTOK);
  printf("Helium potential E  = " FMT_R " K\n", pe_tot * GRID_AUTOK);
  printf("Helium energy       = " FMT_R " K\n", (ke_tot + pe_tot) * GRID_AUTOK);
  fflush(stdout);

  if(upd) {
    otf->model = FUNCTIONAL_FINE;  // Time to switch to FINE functional
    printf("Switched to fine functional.\n");
    tstep = (TS_SWITCH - I * ITS_SWITCH) / GRID_AUTOFS;
  }
}

int main(int argc, char **argv) {

  cgrid *potential_store;
  rgrid *rworkspace;
  wf *gwf;
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
  grid_set_fftw_flags(0);    // FFTW_ESTIMATE
  grid_threads_init(THREADS);
  grid_fft_read_wisdom(NULL);

  grid_wf_analyze_method(PROPERTIES);

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
  rworkspace = rgrid_clone(otf->density, "rworkspae");

  /* Make sure that we have enough workspaces reserved */
  if(!(otf->workspace2)) otf->workspace2 = rgrid_clone(otf->density, "OT workspace 2");
  if(!(otf->workspace3)) otf->workspace3 = rgrid_clone(otf->density, "OT workspace 3");
  if(!(otf->workspace4)) otf->workspace4 = rgrid_clone(otf->density, "OT Workspace 4");
  if(!(otf->workspace5)) otf->workspace5 = rgrid_clone(otf->density, "OT Workspace 5");

  printf("Fourier space range +-: " FMT_R " Angs^-1 with step " FMT_R " Angs^-1.\n", 2.0 * M_PI / (STEP * GRID_AUTOANG), 2.0 * M_PI / (STEP * GRID_AUTOANG * (REAL) NX));

  gwf->norm = rho0 * (STEP * STEP * STEP * (REAL) (NX * NY * NZ));
#ifdef PC
  gwfp->norm = gwf->norm;
#endif

  // DEBUG
//#define TEMP 2.2
//#define GAP 17.6
//  printf("Occ = " FMT_R "\n", (gwf->norm / boltzmann(TEMP)) * EXP(-GAP / TEMP)); exit(0);

  /* 2. Start with ground state + random perturbation */
  initial_guess(gwf->grid);
  cgrid_inverse_fft(gwf->grid);
  grid_wf_normalize(gwf);

  /* 3. Real time simulation */
  printf("Dynamics...\n");
  tstep = (TS - ITS * I) / GRID_AUTOFS;
  grid_timer_start(&timer);
  for (iter = 0; iter < RITER; iter++) {

    if(iter == 100) grid_fft_write_wisdom(NULL);

    dealias(gwf->grid);
    if(CIMAG(tstep) != 0.0) grid_wf_normalize(gwf);

    if(!(iter % NTH)) {
      printf("Wall clock time = " FMT_R " seconds.\n", grid_timer_wall_clock_time(&timer)); fflush(stdout);
      print_stats(iter, gwf, gwfp, otf, potential_store, rworkspace);
      grid_timer_start(&timer);
    }

#ifdef PC
    /* Predict-Correct */
    cgrid_constant(potential_store, -mu0);
#ifdef SCALE
    if(CIMAG(tstep) != 0.0) {
      rgrid_zero(rworkspace);
      rgrid_random_uniform(rworkspace, SCALE);
      grid_add_real_to_complex_re(potential_store, rworkspace);
    }
#endif
    dft_ot_potential(otf, potential_store, gwf);

    dealias(potential_store);

    grid_wf_propagate_predict(gwf, gwfp, potential_store, tstep);

    dealias(gwfp->grid);
    if(CIMAG(tstep) != 0.0) grid_wf_normalize(gwfp);

    cgrid_add(potential_store, -mu0); // not exact chem. pot. hence normalization above needed
#ifdef SCALE
    if(CIMAG(tstep) != 0.0) {
      rgrid_zero(rworkspace);
      rgrid_random_uniform(rworkspace, SCALE);
      grid_add_real_to_complex_re(potential_store, rworkspace);
    }
#endif
    dft_ot_potential(otf, potential_store, gwfp);
    cgrid_multiply(potential_store, 0.5);  // Use (current + future) / 2

    dealias(potential_store);
    if(CIMAG(tstep) != 0.0) grid_wf_normalize(gwfp);

    grid_wf_propagate_correct(gwf, potential_store, tstep);

#else /* PC */

    /* Propagate */
    cgrid_constant(potential_store, -mu0);
#ifdef SCALE
    if(CIMAG(tstep) != 0.0) {
      rgrid_zero(rworkspace);
      rgrid_random_uniform(rworkspace, SCALE);
      grid_add_real_to_complex_re(potential_store, rworkspace);
    }
#endif
    dft_ot_potential(otf, potential_store, gwf);

    dealias(potential_store);
    if(CIMAG(tstep) != 0.0) grid_wf_normalize(gwf);

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
