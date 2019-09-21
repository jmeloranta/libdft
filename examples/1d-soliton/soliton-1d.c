
/*
 * Create planar soliton in superfluid helium (propagating along Z).
 * 1-D version with X & Y coordinates integrated over in the non-linear
 * potential.
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

#define TS 0.1 /* fs */
#define NZ (5*8*32768)
#define STEP 0.2
#define MAXITER 80000000
#define NTH 500000

#define PRESSURE (0.0 / GRID_AUTOBAR)

#define SOLITON_AMP 0.2   /* 10% of bulk */
#define SOLITON_N  200       /* width (in N * LAMBDA_C) */
#define LAMBDA_C (3.58 / GRID_AUTOANG)

#define THREADS 0

/* Francesco's soliton initial guess - plane along z*/
REAL soliton(void *asd, REAL x, REAL y, REAL z) {

  REAL tmp;

  if(FABS(z) > SOLITON_N * LAMBDA_C) return 1.0;

  tmp = SIN(M_PI * z / LAMBDA_C);

  return 1.0 + SOLITON_AMP * tmp * tmp;
}

int main(int argc, char **argv) {

  dft_ot_functional *otf;
  rgrid *density;
  cgrid *potential_store;
  wf *gwf, *gwfp;
  INT iter;
  REAL mu0, rho0;
  char buf[512];
  grid_timer timer;

#ifdef USE_CUDA
#define NGPUS 1
  int gpus[NGPUS] = {0};
  cuda_enable(1, NGPUS, gpus);
#endif

  /* Initialize threads & use wisdom */
  grid_set_fftw_flags(1);    // FFTW_MEASURE
  grid_threads_init(THREADS);
  grid_fft_read_wisdom(NULL);

  /* Allocate wave functions */
  if(!(gwf = grid_wf_alloc(1, 1, NZ, STEP, DFT_HELIUM_MASS, WF_PERIODIC_BOUNDARY, WF_2ND_ORDER_FFT, "gwf"))) {
    fprintf(stderr, "Cannot allocate gwf.\n");
    exit(1);
  }
  gwfp = grid_wf_clone(gwf, "gwfp");

  /* Allocate OT functional */
  if(!(otf = dft_ot_alloc(DFT_OT_PLAIN | DFT_OT_BACKFLOW | DFT_OT_KC | DFT_OT_HD, gwf, DFT_MIN_SUBSTEPS, DFT_MAX_SUBSTEPS))) {
    fprintf(stderr, "Cannot allocate otf.\n");
    exit(1);
  }
  rho0 = dft_ot_bulk_density_pressurized(otf, PRESSURE);
  mu0 = dft_ot_bulk_chempot_pressurized(otf, PRESSURE);
  printf("mu0 = " FMT_R " K/atom, rho0 = " FMT_R " Angs^-3.\n", mu0 * GRID_AUTOK, rho0 / (GRID_AUTOANG * GRID_AUTOANG * GRID_AUTOANG));

  /* bulk density and chemical potential */
  rho0 = dft_ot_bulk_density_pressurized(otf, PRESSURE);
  mu0 = dft_ot_bulk_chempot_pressurized(otf, PRESSURE);

  /* Allocate space for external potential */
  potential_store = cgrid_clone(gwf->grid, "potential store");
  density = rgrid_clone(otf->density, "density");

  if(argc == 2) {
    FILE *fp;
    if(!(fp = fopen(argv[1], "r"))) {
      fprintf(stderr, "Can't open checkpoint .grd file.\n");
      exit(1);
    }
    sscanf(argv[1], "soliton-" FMT_I ".grd", &iter);
    cgrid_read(gwf->grid, fp);
    fclose(fp);
    fprintf(stderr, "Check point from %s with iteration = " FMT_I "\n", argv[1], iter);
  } else {
    /* setup soliton (density is temp here) */
    rgrid_map(density, soliton, NULL);
    rgrid_multiply(density, rho0);
    rgrid_power(density, density, 0.5);
    cgrid_zero(gwf->grid);   /* copy rho to wf real part */
    grid_real_to_complex_re(gwf->grid, density);
    iter = 0;
  }

  for ( ; iter < MAXITER; iter++) {

    /* Predict-Correct */
    grid_timer_start(&timer);
    cgrid_zero(potential_store);
    cgrid_copy(gwfp->grid, gwf->grid);
    dft_ot_potential(otf, potential_store, gwf);
    cgrid_add(potential_store, -mu0);
    grid_wf_propagate_predict(gwf, gwfp, potential_store, TS / GRID_AUTOFS);
    dft_ot_potential(otf, potential_store, gwfp);
    cgrid_add(potential_store, -mu0);
    cgrid_multiply(potential_store, 0.5);  // Use (current + future) / 2
    grid_wf_propagate_correct(gwf, potential_store, TS / GRID_AUTOFS);
    printf("Iteration " FMT_I " - Wall clock time = " FMT_R " seconds.\n", iter, grid_timer_wall_clock_time(&timer));

    if(iter == 5) grid_fft_write_wisdom(NULL);

    if(!(iter % NTH)) {
      sprintf(buf, "soliton-" FMT_I, iter);
      grid_wf_density(gwf, density);
      rgrid_write_grid(buf, density);
    }
  }
  return 0;
}
