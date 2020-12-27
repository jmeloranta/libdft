/*
 * Bulk helium at a given temperature (explicit thermal excitations using the stochastic approach).
 *
 * 1. Start from constant initial order parameter.
 * 2. Propagate in imaginary time using the stochastic approach.
 *
 * All input in a.u. except the time step, which is fs.
 *
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <unistd.h>
#include <grid/grid.h>
#include <grid/au.h>
#include <dft/dft.h>
#include <dft/ot.h>

/* Time integration method */
#define TIMEINT WF_2ND_ORDER_FFT

/* FD(0) or FFT(1) properties */
#define PROPERTIES 0

/* Time step for real and imaginary time */
#define TS (10.0 / GRID_AUTOFS)

/* Real time step after reaching thermal equilibrium */
#define RTS (TS / 100.0)

/* Iteration when to switch to real time propagation */
#define SWITCH 20000000L

/* Grid */
#define NX 128
#define NY 128
#define NZ 128
#define STEP 0.25

/* UV cutoff for random noise (in Fourier space) */
#define UV_CUTOFF (2.25 * GRID_AUTOANG)

/* Use dealiasing during real time propagation? (must use WF_XND_ORDER_CFFT propagator) */
#define DEALIAS_VAL (2.4 * GRID_AUTOANG)

/* Functional to use */
#define FUNCTIONAL (DFT_OT_PLAIN | DFT_OT_KC | DFT_OT_BACKFLOW | DFT_OT_HD)
//#define FUNCTIONAL (DFT_OT_PLAIN | DFT_OT_KC | DFT_OT_BACKFLOW)
//#define FUNCTIONAL (DFT_OT_PLAIN)
//#define FUNCTIONAL DFT_GP2

/* Fix mu0 or adjust dynamically? (if yes then MU0STEP = adjustment penalty amplitude) */
#define FIXMU0 /**/
#define MU0STEP 1.0E-8

/* Pressure */
#define PRESSURE (0.0 / GRID_AUTOBAR)

/* Bulk density at T (Angs^-3) */
#define RHO0 (0.0218360 * (145.5 / 145.2))

/* Random noise scale */
#define TXI 1.5
// TXI = T * XI
#define SCALE (SQRT(2.0 * TXI * GRID_AUKB * TS / (STEP * STEP * STEP)))

/* Constant (0 K) or random (infinite T) initial guess */
// #define RANDOM

/* Average roton energy with the bin corresponding to ROTON_K */
//#define ROTON_E (10.0 / GRID_AUTOK)
//#define ROTON_K (1.9 * GRID_AUTOANG)
#define ROTON_E (9.6 / GRID_AUTOK)
#define ROTON_K (1.855234 * GRID_AUTOANG)
//#define ROTON_E (9.2 / GRID_AUTOK)
//#define ROTON_K (1.115 * GRID_AUTOANG)

/* E(k) */
#define NBINS 512
#define BINSTEP (2.0 * M_PI / (NX * STEP))   // Assumes NX = NY = NZ
#define DENS_EPS 1E-3

/* Number of real time iterations */
#define RITER 200000000L

/* Output every NTH iteration (was 1000) */
#define NTH 100L

/* Write grid files? */
#define WRITE_GRD 2000L

/* Rolling energy iteration interval (in units of NTH) */
#define ROLLING 400

/* How many CPU cores to use (0 = all available) */
#define THREADS 0

/* Random seed (drand48) */
#define RANDOM_SEED 1234L

/* Enable / disable GPU */
// #undef USE_CUDA

/* Temperature search accuracy (K) */
#define SEARCH_ACC 1E-4

/* Molar mass for 4He */
#define MMASS 4.0026

/* GPU allocation */
#ifdef USE_CUDA
int gpus[MAX_GPU];
int ngpus;
#endif

dft_ot_functional *otf;
REAL rho0, mu0, P0, S0;
REAL complex tstep, half_tstep;
FILE *fpm = NULL, *fpp = NULL, *fp1 = NULL, *fp2 = NULL;

double rolling_e[ROLLING], rolling_tent[ROLLING], rolling_trot[ROLLING], rolling_entropy[ROLLING];
INT rolling_ct = 0;

REAL *bins = NULL;

REAL temperature(REAL occ) {

  return -ROTON_E / (LOG(occ + 1E-32) * GRID_AUKB);
}

/* Random initial guess with <P> = 0 */
void initial_guess(cgrid *grid) {

  INT i, j, k, nx = grid->nx, ny = grid->ny, nz = grid->nz;
  REAL complex tval;
  
#ifdef USE_CUDA
  if(cuda_status()) cuda_remove_block(grid->value, 0);
#endif
  cgrid_zero(grid);
  for(i = 0; i < nx; i++)
    for(j = 0; j < ny; j++)
      for(k = 0; k < nz; k++) {
          tval = SQRT(rho0) * CEXP(I * 2.0 * (drand48() - 0.5) * M_PI);
          cgrid_value_to_index(grid, i, j, k, tval);
      }
}

REAL get_energy(wf *gwf, dft_ot_functional *otf, rgrid *rworkspace) {

  REAL tot, tot_gnd, n, rho0K = 0.0218360 * GRID_AUTOANG * GRID_AUTOANG * GRID_AUTOANG;

  n = grid_wf_norm(gwf);
  dft_ot_energy_density(otf, rworkspace, gwf);
#if PROPERTIES == 0
  tot = (grid_wf_energy_cn(gwf, NULL) + rgrid_integral(rworkspace)) / n;
#else
  tot = (grid_wf_energy_fft(gwf, NULL) + rgrid_integral(rworkspace)) / n;
#endif
  tot_gnd = dft_ot_bulk_energy(otf, rho0K) * (STEP * STEP * STEP * (REAL) (NX * NY * NZ)) / n;

  return (tot - tot_gnd) * GRID_AUTOJ * GRID_AVOGADRO; // J / mol
}

void print_stats(INT iter, wf *gwf, dft_ot_functional *otf, cgrid *potential_store, rgrid *rworkspace) {

  char buf[512];
  FILE *fp;
  static REAL *bins = NULL;
  REAL energy, ke_tot, pe_tot, ke_qp, ke_cl, natoms, tmp, tmp2, tmp3, temp, temp2;
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
  printf("Internal energy = " FMT_R " J/g Tenth = " FMT_R " K ", energy / MMASS, temp);

  grid_wf_average_occupation(gwf, bins, BINSTEP, NBINS, potential_store);
  sprintf(buf, "occ-" FMT_I ".dat", iter);
  if(!(fp = fopen(buf, "w"))) {
    fprintf(stderr, "Can't open %s.\n", buf);
    exit(1);
  }
  for(i = 0; i < NBINS; i++)
    if(bins[i] != 0.0) fprintf(fp, FMT_R " " FMT_R "\n", (BINSTEP * (REAL) i) / GRID_AUTOANG, bins[i]);
  fclose(fp);
  printf("Roton  occ = " FMT_R " ", bins[(INT) (0.5 + ROTON_K / BINSTEP)]);
  printf("Ground occ = " FMT_R "\n", bins[0] / natoms);

  printf("Temperature = " FMT_R " K, Internal energy = " FMT_R " J/g\n", (temp2 = temperature(bins[(INT) (0.5 + ROTON_K / BINSTEP)])), energy / MMASS);

  printf("Entropy = " FMT_R " J / (g K)\n", (tmp3 = (grid_wf_entropy(gwf, potential_store) - S0) * GRID_AUTOJ / (natoms * DFT_HELIUM_MASS * GRID_AUTOKG * 1000.0)));

  /* Rolling averages and std dev */
  rolling_e[rolling_ct] = energy;
  rolling_tent[rolling_ct] = temp;
  rolling_trot[rolling_ct] = temp2;
  rolling_entropy[rolling_ct] = tmp3;
  rolling_ct++;
  if(rolling_ct == ROLLING) {
    REAL re = 0.0, rtent = 0.0, rtrot = 0.0, rentropy = 0.0;
    REAL re_std = 0.0, rtent_std = 0.0, rtrot_std = 0.0, rentropy_std = 0.0;
    for (i = 0; i < rolling_ct; i++) {
      re += rolling_e[i];
      rtent += rolling_tent[i];
      rtrot += rolling_trot[i];
      rentropy += rolling_entropy[i];
    }
    re /= (REAL) rolling_ct;
    rtent /= (REAL) rolling_ct;
    rtrot /= (REAL) rolling_ct;
    rentropy /= (REAL) rolling_ct;
    for (i = 0; i < rolling_ct; i++) {
      re_std += (rolling_e[i] - re) * (rolling_e[i] - re);
      rtent_std += (rolling_tent[i] - rtent) * (rolling_tent[i] - rtent);
      rtrot_std += (rolling_trot[i] - rtrot) * (rolling_trot[i] - rtrot);
      rentropy_std += (rolling_entropy[i] - rentropy) * (rolling_entropy[i] - rentropy);
    }
    re_std = SQRT(re_std / (REAL) (rolling_ct - 1));
    rtent_std = SQRT(rtent_std / (REAL) (rolling_ct - 1));
    rtrot_std = SQRT(rtrot_std / (REAL) (rolling_ct - 1));
    rentropy_std = SQRT(rentropy_std / (REAL) (rolling_ct - 1));
    re /= MMASS;  // per gram
    re_std /= MMASS; 
    printf("*** Rolling values (U, S, Tent, Trot): " FMT_R " +- " FMT_R ", "
                                                     FMT_R " +- " FMT_R ", "
                                                     FMT_R " +- " FMT_R ", "
                                                     FMT_R " +- " FMT_R "\n",
                  re, re_std, rentropy, rentropy_std, rtent, rtent_std, rtrot, rtrot_std);
    printf("*** Rolling Xi = " FMT_R "\n", TXI / rtrot);
    rolling_ct = 0;
  }

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
  printf("Total vorticity at " FMT_R " = " FMT_R "\n", energy, rgrid_integral(otf->workspace4));
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
  
  printf("Helium natoms       = " FMT_R " particles.\n", natoms);
  printf("Helium kinetic E    = " FMT_R " K\n", ke_tot * GRID_AUTOK);
  printf("Helium classical KE = " FMT_R " K\n", ke_cl * GRID_AUTOK);
  printf("Helium quantum KE   = " FMT_R " K\n", ke_qp * GRID_AUTOK);
  printf("Helium potential E  = " FMT_R " K\n", pe_tot * GRID_AUTOK);
  printf("Helium energy       = " FMT_R " K\n", (ke_tot + pe_tot) * GRID_AUTOK);
  fflush(stdout);
}

int main(int argc, char **argv) {

  cgrid *potential_store;
  rgrid *rworkspace;
  wf *gwf;
  INT iter, i, kala = 0;
  grid_timer timer;

  printf("Gamma equivalent = " FMT_R "\n", SQRT(2.0 * TXI * GRID_AUKB));

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

  /* Allocate space for external potential */
  potential_store = cgrid_clone(gwf->grid, "potential_store"); /* temporary storage */

  /* Allocate OT functional */
  if(!(otf = dft_ot_alloc(FUNCTIONAL, gwf, DFT_MIN_SUBSTEPS, DFT_MAX_SUBSTEPS))) {
    fprintf(stderr, "Cannot allocate otf.\n");
    exit(1);
  }

  // Backflow limits
//  otf->max_bfpot = 10.0 / GRID_AUTOK; // was 0.5
  rho0 = RHO0 * GRID_AUTOANG * GRID_AUTOANG * GRID_AUTOANG;
  mu0 = dft_ot_bulk_chempot_pressurized(otf, PRESSURE);
  gwf->norm = rho0 * (STEP * STEP * STEP * (REAL) (NX * NY * NZ));

  printf("Bulk mu0 = " FMT_R " K/atom, Bulk rho0 = " FMT_R " Angs^-3.\n", mu0 * GRID_AUTOK, rho0 / (GRID_AUTOANG * GRID_AUTOANG * GRID_AUTOANG));

  /* Make sure that we have enough workspaces reserved */
  rworkspace = rgrid_clone(otf->density, "rworkspace");
  if(!(otf->workspace2)) otf->workspace2 = rgrid_clone(otf->density, "OT workspace 2");
  if(!(otf->workspace3)) otf->workspace3 = rgrid_clone(otf->density, "OT workspace 3");
  if(!(otf->workspace4)) otf->workspace4 = rgrid_clone(otf->density, "OT Workspace 4");
  if(!(otf->workspace5)) otf->workspace5 = rgrid_clone(otf->density, "OT Workspace 5");

  printf("Fourier space range +-: " FMT_R " Angs^-1 with step " FMT_R " Angs^-1.\n", 2.0 * M_PI / (STEP * GRID_AUTOANG), 2.0 * M_PI / (STEP * GRID_AUTOANG * (REAL) NX));

  /* 2. Start with ground state or random */
  cgrid_constant(gwf->grid, SQRT(rho0));
  P0 = GRID_AUTOPA * dft_ot_pressure(otf, rworkspace, gwf, 0.0);
  S0 = grid_wf_entropy(gwf, potential_store); // residual (or finite grid) entropy
#ifdef RANDOM
  initial_guess(gwf->grid);
  cgrid_inverse_fft(gwf->grid);
  grid_wf_normalize(gwf);
#endif

  printf("Imaginary time propagation...\n");
  gwf->kmax = DEALIAS_VAL;
  tstep = -I * TS;
  half_tstep = 0.5 * tstep;
  grid_timer_start(&timer);

  for (iter = 0; iter < RITER; iter++) {

    if(iter > SWITCH && !kala) {
      printf("Switched to real time propagation.\n");
      tstep = RTS;
      half_tstep = 0.5 * tstep;
      kala = 1;
    }

    if(iter == 100) grid_fft_write_wisdom(NULL);

    if(!(iter % NTH)) {
      printf("Normalization: mu0 = " FMT_R " K/atom, rho0 = " FMT_R " Angs^-3\n", mu0 * GRID_AUTOK, rho0 / (GRID_AUTOANG * GRID_AUTOANG * GRID_AUTOANG));
      printf("Wall clock time = " FMT_R " seconds.\n", grid_timer_wall_clock_time(&timer)); fflush(stdout);
      print_stats(iter, gwf, otf, potential_store, rworkspace);
      grid_timer_start(&timer);
    }

    /* Kinetic delta t/2 */
    cgrid_fft(gwf->grid);
    grid_wf_propagate_kinetic_fft(gwf, half_tstep);
    cgrid_inverse_fft_norm(gwf->grid);

    /* Potential delta t */
    cgrid_constant(potential_store, -mu0);
    dft_ot_potential(otf, potential_store, gwf);
    grid_wf_propagate_potential(gwf, tstep, potential_store, 0.0); // no moving bkg (0.0)

    if(!kala) {
      /* Random term x delta t (add here to improve the accuracy) */
      cgrid_zero(potential_store);
      cgrid_random_normal(potential_store, SCALE * (1.0 + I) / 2.0);  // both components have one -> / 2 
      cgrid_fft(potential_store); // Filter high wavenumber components out
      cgrid_dealias2(potential_store, UV_CUTOFF);
      cgrid_inverse_fft_norm(potential_store);
      cgrid_sum(gwf->grid, gwf->grid, potential_store);
    }

    /* Kinetic delta t/2 */
    cgrid_fft(gwf->grid);
    grid_wf_propagate_kinetic_fft(gwf, half_tstep);
    cgrid_inverse_fft_norm(gwf->grid);

    /* Normalize since imaginary time present (and mu is not exact) */
#ifdef FIXMU0
    if(!kala) grid_wf_normalize(gwf);
#else
    if(!kala) {
      REAL nor = grid_wf_norm(gwf);
      mu0 += -(MU0STEP / GRID_AUTOK) * (nor - gwf->norm) * TS;
      rho0 = nor / (STEP * STEP * STEP * (REAL) (NX * NY * NZ));
    }
#endif 

    /* Dealias if real time propagation */
    if(kala) {
      cgrid_fft(gwf->grid); // Filter high wavenumber components out
      cgrid_dealias2(gwf->grid, DEALIAS_VAL);
      cgrid_inverse_fft_norm(gwf->grid);
    }

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
