/*
 * Thin film of superfluid helium (0 K).
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

#define TS 5.0 /* fs */
#define NX 32
#define NY 1024
#define NZ 1024
#define STEP 1.0
#define NTH 1000
#define THREADS 0

/* safe zone without boundary interference */
#define SNY (NY / 2)
#define SNZ (NZ / 2)

#define XVAL(i) (((REAL) (i - NX/2)) * STEP)
#define YVAL(j) (((REAL) (j - SNY/2)) * STEP)
#define ZVAL(k) (((REAL) (k - SNZ/2)) * STEP)

/* Start simulation after this many iterations */
#define START 1000

/* Number of He atoms (0 = no normalization) */
#define NHE 0

/* Imag. time component (dissipation) */
#define ITS (0.05 * TS)

#define PRESSURE (0.0 / GRID_AUTOBAR)

#define RANDOM_SEED 1234567L

REAL rho0;

/* Number of vortex line pais (with opposite circulation) in YZ-plane */
#define NPAIRS 8
#define PAIR_DIST (SNY * STEP / 5.0)

REAL y[2*NPAIRS], z[2*NPAIRS];
INT npts = 0;

INT check_proximity(REAL yy, REAL zz) {

  INT i;
  REAL y2, z2;

  for (i = 0; i < npts; i++) {
    y2 = yy - y[i]; y2 *= y2;
    z2 = zz - z[i]; z2 *= z2;
    if(SQRT(y2 + z2) < PAIR_DIST) return 1;  // something too close
  }
  return 0; // OK
}

/* vortex ring initial guess (ring in yz-plane) */
REAL complex vline(void *prm, REAL x, REAL y, REAL z) {

  REAL r, angle, dir, offset;

  x = ((REAL *) prm)[0] - x;
  y = ((REAL *) prm)[1] - y;
  z = ((REAL *) prm)[2] - z;
  dir = ((REAL *) prm)[3];  
  offset = ((REAL *) prm)[4];
  r = SQRT(y * y + z * z);
  angle = ATAN2(y, z);

  return (1.0 - EXP(-r)) * SQRT(rho0) * CEXP(I * dir * (angle + offset));
}

int main(int argc, char **argv) {

  dft_ot_functional *otf;
  cgrid *potential_store;
  rgrid *rworkspace;
  wf *gwf;
  INT iter;
  REAL mu0, kin, pot, n, line[5];
  char buf[512];
  grid_timer timer;
  REAL complex tstep;

#ifdef USE_CUDA
  cuda_enable(1);  // enable CUDA ?
#endif

  /* Initialize threads & use wisdom */
  grid_set_fftw_flags(1);    // FFTW_MEASURE
  grid_threads_init(THREADS);
  grid_fft_read_wisdom(NULL);

  /* Allocate wave functions */
//  if(!(gwf = grid_wf_alloc(NX, NY, NZ, STEP, DFT_HELIUM_MASS, WF_FFT_EOO_BOUNDARY, WF_2ND_ORDER_FFT, "gwf"))) {
  if(!(gwf = grid_wf_alloc(NX, NY, NZ, STEP, DFT_HELIUM_MASS, WF_PERIODIC_BOUNDARY, WF_2ND_ORDER_FFT, "gwf"))) {
    fprintf(stderr, "Cannot allocate gwf.\n");
    exit(1);
  }

  /* Allocate OT functional */
  if(!(otf = dft_ot_alloc(DFT_OT_PLAIN, gwf, DFT_MIN_SUBSTEPS, DFT_MAX_SUBSTEPS))) {
    fprintf(stderr, "Cannot allocate otf.\n");
    exit(1);
  }
  rho0 = dft_ot_bulk_density_pressurized(otf, PRESSURE);
  mu0 = dft_ot_bulk_chempot_pressurized(otf, PRESSURE);
  printf("mu0 = " FMT_R " K/atom, rho0 = " FMT_R " Angs^-3.\n", mu0 * GRID_AUTOK, rho0 / (GRID_AUTOANG * GRID_AUTOANG * GRID_AUTOANG));

  /* Allocate space for external potential */
  potential_store = cgrid_clone(gwf->grid, "potential_store"); /* temporary storage */
  rworkspace = rgrid_clone(otf->density, "rworkspace"); /* temporary storage */
 
  /* setup initial guess for vortex lines */
  srand48(RANDOM_SEED); // or time(0)
  printf("Random seed = %ld\n", RANDOM_SEED);
  cgrid_constant(gwf->grid, SQRT(rho0));
  for (iter = 0; iter < NPAIRS; iter++) {
    REAL rv;
    // First in pair
    do {
      line[0] = 0.0; 
      line[1] = YVAL(0) + drand48() * STEP * (REAL) SNY;
      line[2] = ZVAL(0) + drand48() * STEP * (REAL) SNZ;
    } while(check_proximity(line[1], line[2]));
    line[3] = 1.0; line[4] = 0.0;
    y[npts] = line[1];
    z[npts] = line[2];
    npts++;
    cgrid_map(potential_store, vline, line);
    cgrid_product(gwf->grid, gwf->grid, potential_store);
    cgrid_multiply(gwf->grid, 1.0 / SQRT(rho0));
    printf("Pair+ " FMT_I ": " FMT_R "," FMT_R "\n", iter, line[1], line[2]); fflush(stdout);
    // Second in pair
    line[0] = 0.0; 
    do {
      rv = drand48();
      line[1] += SIN(2.0 * M_PI * rv) * PAIR_DIST;
      line[2] += COS(2.0 * M_PI * rv) * PAIR_DIST;
    } while (FABS(line[1]) >= YVAL(SNY) || FABS(line[2]) >= ZVAL(SNZ) || check_proximity(line[1], line[2]));
    line[3] = -1.0; line[4] = 0.0;
    y[npts] = line[1];
    z[npts] = line[2];
    npts++;
    cgrid_map(potential_store, vline, line);
    cgrid_product(gwf->grid, gwf->grid, potential_store);
    cgrid_multiply(gwf->grid, 1.0 / SQRT(rho0));
    printf("Pair- " FMT_I ": " FMT_R "," FMT_R "\n", iter, line[1], line[2]); fflush(stdout);
  }

  /* external potential */

#if NHE != 0
  gwf->norm = 5000;
#endif

  for (iter = 1; iter < 800000; iter++) {
    if(iter < START) tstep = -I * TS / GRID_AUTOFS;
    else tstep = (TS - I * ITS) / GRID_AUTOFS;
    if(iter == 1 || !(iter % NTH)) {
      printf("Iteration " FMT_I " - Wall clock time = " FMT_R " seconds.\n", iter, grid_timer_wall_clock_time(&timer));
      sprintf(buf, "film-" FMT_I, iter);
      cgrid_write_grid(buf, gwf->grid);
      kin = grid_wf_energy(gwf, NULL);            /* Kinetic energy for gwf */
      dft_ot_energy_density(otf, rworkspace, gwf);
      pot = rgrid_integral(rworkspace);
      n = grid_wf_norm(gwf);
      printf("Iteration " FMT_I " helium natoms    = " FMT_R " particles.\n", iter, n);   /* Energy / particle in K */
      printf("Iteration " FMT_I " helium kinetic   = " FMT_R "\n", iter, kin * GRID_AUTOK);  /* Print result in K */
      printf("Iteration " FMT_I " helium potential = " FMT_R "\n", iter, pot * GRID_AUTOK);  /* Print result in K */
      printf("Iteration " FMT_I " helium energy    = " FMT_R "\n", iter, (kin + pot) * GRID_AUTOK);  /* Print result in K */
      fflush(stdout);
    }

    if(iter == 5) grid_fft_write_wisdom(NULL);

    grid_timer_start(&timer);

    /* Predict-Correct */
    cgrid_zero(potential_store);
    dft_ot_potential(otf, potential_store, gwf);
    cgrid_add(potential_store, -mu0);
    grid_wf_propagate(gwf, potential_store, tstep);
#if NHE != 0
    grid_wf_normalize(gwf);
#endif
  }
  return 0;
}
