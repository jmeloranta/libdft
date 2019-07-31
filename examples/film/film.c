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

#define TS 30.0 /* fs */
/* Was 32x1024x1024 step = 1.0 */
#define NX 32
#define NY 4096
#define NZ 4096
#define STEP 1.0
#define NTH 1000
#define THREADS 0

/* Print vortex line locations only? (otherwise write full grids) */
#define LINE_LOCATIONS_ONLY

/* safe zone without boundary interference */
#define SNY (NY / 2)
#define SNZ (NZ / 2)

/* Start simulation after this many iterations */
#define START 200

/* Imag. time component (dissipation) */
#define ITS (0.02 * TS)

#define PRESSURE (0.0 / GRID_AUTOBAR)

#define RANDOM_SEED 1234567L

REAL rho0;

/* Number of vortex line pais (with opposite circulation) in YZ-plane */
#define NPAIRS 16
/* #define PAIR_DIST (SNY * STEP / 5.0) */
#define PAIR_DIST 100.0

/* These hold the initial guess vortex line positions */
/* and later on the vortex line positions determined */
/* during the simulation. */
REAL y[2*NPAIRS], z[2*NPAIRS];
INT npts = 0;

INT check_proximity(REAL yy, REAL zz, REAL dist) {

  INT i;
  REAL y2, z2;

  for (i = 0; i < npts; i++) {
    y2 = yy - y[i]; y2 *= y2;
    z2 = zz - z[i]; z2 *= z2;
    if(SQRT(y2 + z2) < dist) return 1;  // something too close
  }
  return 0; // OK
}

#define CORE_DENS (rho0 / 4.0)  // for step = 2.0, 2.0 is ok
#define MIN_DIST_CORE 3.0

void locate_lines(rgrid *density) {

  INT i, k, j;
  REAL yy, zz;

  npts = 0;
  i = NX / 2;
  for (j = 0; j < NY; j++) {
    yy = ((REAL) (j - NY/2)) * STEP;
    for (k = 0; k < NZ; k++) {
      zz = ((REAL) (k - NZ/2)) * STEP;
      if(FABS(rgrid_value_at_index(density, i, j, k)) < CORE_DENS && !check_proximity(yy, zz, MIN_DIST_CORE)) {
        if(npts >= 2*NPAIRS) {
          fprintf(stderr, "Error: More lines than generated initially!\n");
          exit(1);
        }
        printf(FMT_R " " FMT_R "\n", yy, zz);
        y[npts] = yy;
        z[npts] = zz;
        npts++;
      }
    }
  }
}

void print_lines() {

  INT i;

//  printf("XXX " FMT_I "\n", npts);
  printf("XXX\n");
  for (i = 0; i < npts; i++)
    printf("XXX " FMT_R " " FMT_R "\n", y[i], z[i]);
}  

void print_pair_dist(char *file) {

  INT i, j;
  FILE *fp;

  if(!(fp = fopen(file, "w"))) {
    fprintf(stderr, "Can't open pair dist file.\n");
    exit(1);
  }
  for (i = 0; i < npts; i++)
    for (j = i+1; j < npts; j++)
      fprintf(fp, FMT_R "\n", SQRT((y[i] - y[j]) * (y[i] - y[j]) + (z[i] - z[j]) * (z[i] - z[j])));
  fclose(fp);
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
    // First line in pair
    do {
      line[0] = 0.0;
      line[1] = -(STEP/2.0) * ((REAL) SNY) + drand48() * STEP * (REAL) SNY; // origin +- displacement
      line[2] = -(STEP/2.0) * ((REAL) SNZ) + drand48() * STEP * (REAL) SNZ;
    } while(FABS(line[1]) > STEP * SNY / 2.0 || FABS(line[2]) > STEP * SNZ / 2.0 || check_proximity(line[1], line[2], PAIR_DIST));
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
    } while (check_proximity(line[1], line[2], PAIR_DIST));
    line[3] = -1.0; 
    line[4] = 0.0;
    y[npts] = line[1];
    z[npts] = line[2];
    npts++;
    cgrid_map(potential_store, vline, line);
    cgrid_product(gwf->grid, gwf->grid, potential_store);
    cgrid_multiply(gwf->grid, 1.0 / SQRT(rho0));
    printf("Pair- " FMT_I ": " FMT_R "," FMT_R "\n", iter, line[1], line[2]); fflush(stdout);
  }

  for (iter = 1; iter < 800000; iter++) {
    if(iter < START) tstep = -I * TS / GRID_AUTOFS;
    else tstep = (TS - I * ITS) / GRID_AUTOFS;
    if(iter == 1 || !(iter % NTH)) {
      printf("Iteration " FMT_I " - Wall clock time = " FMT_R " seconds.\n", iter, grid_timer_wall_clock_time(&timer));
#ifdef LINE_LOCATIONS_ONLY
      grid_wf_density(gwf, rworkspace);
      locate_lines(rworkspace);
      print_lines();
      sprintf(buf, "film-" FMT_I ".pair", iter);
      print_pair_dist(buf);
#else
      sprintf(buf, "film-" FMT_I, iter);
      cgrid_write_grid(buf, gwf->grid);
#endif
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
