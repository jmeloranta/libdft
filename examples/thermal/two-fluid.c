/*
 * Two-fluid test. Just for demo purposes, did not work too well in practice...
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

/* FD(0) or FFT(1) properties */
#define PROPERTIES 0

/* Time step for imaginary time */
#define TS (10.0 / GRID_AUTOFS)

/* Real time step after reaching thermal equilibrium */
#define RTS (0.1 / GRID_AUTOFS)

/* Iteration when to switch to real time propagation */
#define SWITCH 50000000L

/* Grid */
#define NX 64
#define NY 64
#define NZ 64
#define STEP 0.5

/* Bulk density at 0 K and zero pressure (Angs^-3) */
#define RHO0 ((145.9 / 145.2) * 0.0218360 * GRID_AUTOANG * GRID_AUTOANG * GRID_AUTOANG)

/* Super and normal fractions */
#define SF 0.5
#define NF (1.0 - SF)

/* Random noise scale */
// TXI = T * XI
#define TXI 0.3

/* UV cutoff for random noise (in Fourier space) */
#define UV_CUTOFF (2.25 * GRID_AUTOANG)  // 2.25

/* Number ofiterations */
#define RITER 200000000L

/* Functional to use -- REMOVE HD */
#define FUNCTIONAL (DFT_OT_PLAIN | DFT_OT_KC | DFT_OT_BACKFLOW)
//#define FUNCTIONAL (DFT_OT_PLAIN | DFT_OT_KC | DFT_OT_BACKFLOW)
//#define FUNCTIONAL (DFT_OT_PLAIN | DFT_OT_BACKFLOW)
//#define FUNCTIONAL DFT_GP2

/* How many CPU cores to use (0 = all available) */
#define THREADS 0

/* Random seed (drand48) */
#define RANDOM_SEED 1234L

/* Write grid files? */
#define WRITE_GRD 1000L

/* GPU allocation */
#ifdef USE_CUDA
int gpus[MAX_GPU];
int ngpus;
#endif

dft_ot_functional *otf;

int main(int argc, char **argv) {

  cgrid *potential_store;
  rgrid *workspace1, *workspace2, *workspace3, *workspace4, *workspace5, *workspace6, *workspace7, *workspace8, *workspace9;
  wf *gwf_s, *gwf_n;
  INT iter;
  REAL scale;
  grid_timer timer;
  REAL complex tstep, half_tstep;

  if(argc < 2) {
    fprintf(stderr, "Usage: thermal <gpu1> <gpu2> ...\n");
    exit(1);
  }

#ifdef USE_CUDA
  INT i;
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
  if(!(gwf_s = grid_wf_alloc(NX, NY, NZ, STEP, DFT_HELIUM_MASS, WF_PERIODIC_BOUNDARY, WF_2ND_ORDER_FFT, "gwf_s"))) {
    fprintf(stderr, "Cannot allocate gwf_s.\n");
    exit(1);
  }
  gwf_n = grid_wf_clone(gwf_s, "gwf_n");

  /* Allocate space for external potential */
  potential_store = cgrid_clone(gwf_s->grid, "potential_store"); /* temporary storage */

  /* Allocate OT functional */
  if(!(otf = dft_ot_alloc(FUNCTIONAL, gwf_s, DFT_MIN_SUBSTEPS, DFT_MAX_SUBSTEPS))) {
    fprintf(stderr, "Cannot allocate otf.\n");
    exit(1);
  }

  gwf_s->norm = SF * RHO0 * (STEP * STEP * STEP * (REAL) (NX * NY * NZ));
  gwf_n->norm = NF * RHO0 * (STEP * STEP * STEP * (REAL) (NX * NY * NZ));

  scale = SQRT(2.0 * TXI * GRID_AUKB * TS / (STEP * STEP * STEP));  // TS / SQRT(TS) = SQRT(TS); TS from propagation + 1/SQRT(TS)

  /* Make sure that we have enough workspaces reserved */
  if(!(otf->workspace1)) otf->workspace1 = rgrid_clone(otf->density, "OT workspace 1");
  if(!(otf->workspace2)) otf->workspace2 = rgrid_clone(otf->density, "OT workspace 2");
  if(!(otf->workspace3)) otf->workspace3 = rgrid_clone(otf->density, "OT workspace 3");
  if(!(otf->workspace4)) otf->workspace4 = rgrid_clone(otf->density, "OT Workspace 4");
  if(!(otf->workspace5)) otf->workspace5 = rgrid_clone(otf->density, "OT Workspace 5");
  if(!(otf->workspace6)) otf->workspace6 = rgrid_clone(otf->density, "OT Workspace 6");
  if(!(otf->workspace7)) otf->workspace7 = rgrid_clone(otf->density, "OT Workspace 7");
  if(!(otf->workspace8)) otf->workspace8 = rgrid_clone(otf->density, "OT Workspace 8");
  if(!(otf->workspace9)) otf->workspace9 = rgrid_clone(otf->density, "OT Workspace 9");
  workspace1 = otf->workspace1;
  workspace2 = otf->workspace2;
  workspace3 = otf->workspace3;
  workspace4 = otf->workspace4;
  workspace5 = otf->workspace5;
  workspace6 = otf->workspace6;
  workspace7 = otf->workspace7;
  workspace8 = otf->workspace8;
  workspace9 = otf->workspace9;

  cgrid_constant(gwf_s->grid, SF * SQRT(RHO0));
  cgrid_constant(gwf_n->grid, NF * SQRT(RHO0));

  printf("Imaginary time propagation...\n");
  tstep = -I * TS;
  half_tstep = 0.5 * tstep;

  grid_timer_start(&timer);

  for (iter = 0; iter < RITER; iter++) {

    if(iter == 100) grid_fft_write_wisdom(NULL);

    /* Kinetic delta t/2: super */
    cgrid_fft(gwf_s->grid);
    grid_wf_propagate_kinetic_fft(gwf_s, half_tstep);
    cgrid_inverse_fft_norm(gwf_s->grid);
    /* Kinetic delta t/2: normal */
    cgrid_fft(gwf_n->grid);
    grid_wf_propagate_kinetic_fft(gwf_n, half_tstep);
    cgrid_inverse_fft_norm(gwf_n->grid);

    /* Potential delta t */
    grid_wf_density(gwf_s, workspace1);
    grid_wf_density(gwf_n, workspace2);
    rgrid_sum(otf->density, workspace1, workspace2); // rho = rho_s + rho_n
    grid_wf_probability_flux(gwf_s, workspace2, workspace3, workspace4);  
    grid_wf_probability_flux(gwf_n, workspace5, workspace6, workspace7);  
    rgrid_sum(workspace2, workspace2, workspace5); // j_x = rho_s v_s,x + rho_n v_n,x
    rgrid_sum(workspace3, workspace3, workspace6); // j_y = rho_s v_s,y + rho_n v_n,y
    rgrid_sum(workspace4, workspace4, workspace7); // j_z = rho_s v_s,z + rho_n v_n,z
    rgrid_division_eps(workspace7, workspace2, otf->density, otf->div_epsilon); // v_x = (rho_s v_s,x + rho_n v_n,x) / rho
    rgrid_division_eps(workspace8, workspace3, otf->density, otf->div_epsilon); // v_y = (rho_s v_s,y + rho_n v_n,y) / rho
    rgrid_division_eps(workspace9, workspace4, otf->density, otf->div_epsilon); // v_z = (rho_s v_s,z + rho_n v_n,z) / rho
    cgrid_zero(potential_store);
    dft_ot_potential2(otf, potential_store, otf->density, workspace7, workspace8, workspace9);
    grid_wf_propagate_potential(gwf_s, tstep, potential_store, 0.0); // no moving bkg (0.0)
    grid_wf_propagate_potential(gwf_n, tstep, potential_store, 0.0); // no moving bkg (0.0)

    /* Random term x delta t (add here to improve the accuracy): only for normal fraction */
    cgrid_zero(potential_store);
    cgrid_random_normal(potential_store, scale * (1.0 + I) / 2.0);  // both components have one -> / 2 
    cgrid_fft(potential_store); // Filter high wavenumber components out
    cgrid_dealias2(potential_store, 0.0, UV_CUTOFF);
    cgrid_inverse_fft_norm(potential_store);
    cgrid_sum(gwf_n->grid, gwf_n->grid, potential_store);

    /* Kinetic delta t/2: super */
    cgrid_fft(gwf_s->grid);
    grid_wf_propagate_kinetic_fft(gwf_s, half_tstep);
    cgrid_inverse_fft_norm(gwf_s->grid);
    /* Kinetic delta t/2: normal */
    cgrid_fft(gwf_n->grid);
    grid_wf_propagate_kinetic_fft(gwf_n, half_tstep);
    cgrid_inverse_fft_norm(gwf_n->grid);

    grid_wf_normalize(gwf_s);
    grid_wf_normalize(gwf_n);

    if(!(iter % WRITE_GRD)) {
      char buf[512];
      sprintf(buf, "super-" FMT_I, iter);
      cgrid_write_grid(buf, gwf_s->grid);
      sprintf(buf, "normal-" FMT_I, iter);
      cgrid_write_grid(buf, gwf_n->grid);
    }
  }

  return 0;
}
