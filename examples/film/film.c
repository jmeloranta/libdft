/*
 * Thin film of superfluid helium (0 K).
 *
 * All input in a.u. except the time step, which is fs.
 *
 * (X, Y) ordering of data for multi-GPU.
 *
 * 10 fs OK with 0.5 Bohr step.
 * 10 fs OK with 2.0 Bohr step.
 * 20 fs NOT with 2.0 Bohr step. Blows up around iter 20,000.
 *
 * 10 fs with 2.0 Bohr diverges after 58,000 iterations.
 *
 * slab thickness 32 Bohr.
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
#define TS 2.0 /* fs (was 5) */
#define ITS 40.0 /* fs */
//#define TEMP 0.5  /* Temperature (leave undefined if zero Kelvin) */

#ifdef TEMP
#define LAMBDA 0.0    /* Empirical dissipation parameter */
#define RANDOM_WIDTH ((TS / GRID_AUTOFS) * M_SQRT1_2 * (1.0 + I) * SQRT(2.0 * GRID_AUKB * TEMP * LAMBDA / DFT_HELIUM_MASS))
#else
#define LAMBDA 0.00   /* Tsubota's gamma corresponds to LAMBDA = 0.02 */
#endif

#define NX 256
#define NY 256
#define NZ 32     // This must be 32 Bohr
#define STEP 1.0
#define MAXITER 8000000

/* E(k) */
#define KSPECTRUM /**/
#define NBINS 60
#define BINSTEP 0.05
#define DENS_EPS 1E-3

/* Predict-correct? */
//#define PC

/* Functional to use (was DFT_OT_PLAIN; GP2 is test)  -- TODO: There is a problem with backflow, energy keeps increasing? HD does not help. Predict-correct or shoter time step? or shorter grid step? */
//#define FUNCTIONAL (DFT_OT_PLAIN | DFT_OT_KC | DFT_OT_BACKFLOW)
#define FUNCTIONAL (DFT_OT_PLAIN)

/* Pressure */
#define PRESSURE (0.0 / GRID_AUTOBAR)

/* Vortex line params (define only one!) */
#define RANDOM_INITIAL       /* Random initial guess */
//#define RANDOM_LINES         /* Random line positions */
//#define MANUAL_LINES         /* Enter vortex lines manually */
#define RANDOM_SEED 12345678L /* Random seed for generating initial vortex line coordinates */
#define NPAIRS 100000           /* Number of + and - vortex pairs (was 1000) */
#define PAIR_DIST 20.0       /* Min. distance between + and - vortex pairs */
#define UNRESTRICTED_PAIRS   /* If defined, PAIR_DIST for the + and - pairs is not enforced */
#define MAX_DIST (HE_RADIUS-200.0)       /* Maximum distance for vortex lines from the origin */
#define NRETRY   10000       /* # of retries for locating the pair. If not successful, start over */

/* Normalization - now use this many % of the width for the radius (need some empty space due to periodic bc) */
#define HE_RADIUS ((NX * STEP / 2.0) - 50.0)
//#define HE_NORM (rho0 * M_PI * HE_RADIUS * HE_RADIUS * STEP * (REAL) (NZ-1))
#define HE_NORM (rho0 * ((REAL) NX) * STEP * ((REAL) NY) * STEP * ((REAL) NZ) * STEP)

/* Print vortex line locations only? (otherwise write full grids) */
//#define LINE_LOCATIONS_ONLY

/* Vortex line search specific parameters */
#define MIN_DIST_CORE 3.0  // min distance between cores (annihilate below this)
#define ADJUST 0.4  // |rot| adjust or CURRENTLY density threshold adjust
//#define DIST_CUTOFF (HE_RADIUS - 150.0) // allow lines to be inside this radius
#define DIST_CUTOFF 1E20 // allow lines to be inside this radius

/* Start simulation after this many iterations (1: columng, 2: column+vortices) */
#define TT 0.7   // Temperature in Kelvin
#define NN 2.2   // 2.2 for each dimension
#define DIM 2.0  // 2.0 = film, 3.0 = bulk liquid
#define START1 (0)  // before adding vortex lines (was 1000)
#define START2 ((INT) (POW(1.10083823E8, DIM / 3.0) * (GRID_AUTOFS / ITS) * POW(TT, -NN*DIM)))

/* Output every NTH iteration (was 5000) */
#define NTH 2000

/* Use all threads available on the computer */
#define THREADS 0

/* GPU allocation */
#ifdef USE_CUDA
int gpus[MAX_GPU];
int ngpus;
#endif

REAL rho0, mu0;
REAL xp[2*NPAIRS], yp[2*NPAIRS], xm[2*NPAIRS], ym[2*NPAIRS];
INT nptsp = 0, nptsm = 0;

INT check_proximity(REAL xx, REAL yy, REAL dist) {

  INT i;
  REAL x2, y2, dist2 = dist*dist;

  for (i = 0; i < nptsm; i++) {
    x2 = xx - xm[i]; x2 *= x2;
    y2 = yy - ym[i]; y2 *= y2;
    if(x2 + y2 < dist2) return 1;  // too close
  }

  for (i = 0; i < nptsp; i++) {
    x2 = xx - xp[i]; x2 *= x2;
    y2 = yy - yp[i]; y2 *= y2;
    if(x2 + y2 < dist2) return 1;  // too close
  }

  return 0; // all clear
}

int check_boundary(REAL x, REAL y, REAL cutoff) {

  if(SQRT(x*x + y*y) > cutoff) return 1;
  return 0; // inside
}

int check_line_center(rgrid *density, rgrid *rot, REAL m, INT i, INT j, INT k) {

#if 0
  if(FABS(rgrid_value_at_index(rot, i, j, k)) > m) return 1;
  else return 0;
#endif

#if 0
  REAL tmp;
  tmp = rgrid_value_at_index(density, i, j, k);
  if(rgrid_value_at_index(density, i-2, j, k) > rgrid_value_at_index(density, i-1, j, k) &&
     rgrid_value_at_index(density, i-1, j, k) > tmp &&
     tmp < rgrid_value_at_index(density, i+1, j, k) &&
     rgrid_value_at_index(density, i+1, j, k) < rgrid_value_at_index(density, i+2, j, k) &&
     rgrid_value_at_index(density, i, j-2, k) > rgrid_value_at_index(density, i, j-1, k) &&
     rgrid_value_at_index(density, i, j-1, k) > tmp &&
     tmp < rgrid_value_at_index(density, i, j+1, k) &&
     rgrid_value_at_index(density, i, j+1, k) < rgrid_value_at_index(density, i, j+2, k)
     && tmp < rho0/10.0) return 1;
//  if(tmp < rgrid_value_at_index(density, i-1, j, k) && tmp < rgrid_value_at_index(density, i+1, j, k) && tmp < rgrid_value_at_index(density, i, j-1, k) && tmp < rgrid_value_at_index(density, i, j+1, k) && tmp < rho0/10.0) return 1;
  else return 0;
#endif

#if 1
  REAL tmp;
  tmp = rgrid_value_at_index(density, i, j, k);
  if(tmp <= rgrid_value_at_index(density, i-1, j, k) &&
     tmp <= rgrid_value_at_index(density, i+1, j, k) &&
     tmp <= rgrid_value_at_index(density, i, j-1, k) &&  
     tmp <= rgrid_value_at_index(density, i, j+1, k) &&  
     tmp <= rgrid_value_at_index(density, i+1, j+1, k) &&  
     tmp <= rgrid_value_at_index(density, i-1, j-1, k) &&  
     tmp <= rgrid_value_at_index(density, i+1, j-1, k) &&  
     tmp <= rgrid_value_at_index(density, i-1, j+1, k) && tmp < rho0*ADJUST) return 1;
  else return 0;
#endif
}

void locate_lines(rgrid *density, rgrid *rot) {

  INT i, k, j;
  REAL xx, yy, tmp;
  static REAL m = -1.0;

#ifdef USE_CUDA
  cuda_remove_block(density->value, 1);
  cuda_remove_block(rot->value, 1);
#endif
  nptsm = nptsp = 0;
  k = NZ / 2;
  if(m < 0.0) {
    m = rgrid_max(rot) * ADJUST;
    tmp = FABS(rgrid_min(rot)) * ADJUST;
    if(tmp > m) m = tmp;
    printf("m = " FMT_R "\n", m);
  }
  for (i = 0; i < NX; i++) {
    xx = ((REAL) (i - NX/2)) * STEP;
    for (j = 0; j < NY; j++) {
      yy = ((REAL) (j - NY/2)) * STEP;
      if(!check_boundary(xx, yy, DIST_CUTOFF) && check_line_center(density, rot, m, i, j, k) && !check_proximity(xx, yy, MIN_DIST_CORE)) {
        tmp = (rgrid_value_at_index(rot, i, j, k) + rgrid_value_at_index(rot, i-1, j, k) + rgrid_value_at_index(rot, i+1, j, k) + rgrid_value_at_index(rot, i, j-1, k) + rgrid_value_at_index(rot, i, j+1, k)) / 5.0;
        if(tmp > 0.0) { // switched < to > to get the +/- correctly
          if(nptsm >= 2*NPAIRS) {
            fprintf(stderr, "Error(-): More lines than generated initially!\n");
            exit(1);
          }
          xm[nptsm] = xx;
          ym[nptsm] = yy;
          nptsm++;
        } else {
          if(nptsp >= 2*NPAIRS) {
            fprintf(stderr, "Error(+): More lines than generated initially!\n");
            exit(1);
          }
          xp[nptsp] = xx;
          yp[nptsp] = yy;
          nptsp++;
        }
      }
    }
  }
}

FILE *fpm = NULL, *fpp = NULL, *fp1 = NULL, *fp2 = NULL;

void print_lines() {

  INT i;

  printf("Line density: " FMT_R " at " FMT_R ".\n", (((REAL) (nptsm + nptsp)) / 2.0) / (NX * STEP * GRID_AUTOANG * NY * STEP * GRID_AUTOANG), TT / 0.89);
  if(!fpm || !fpp) {
    if(!(fpm = fopen("minus.dat", "w"))) exit(1);
    if(!(fpp = fopen("plus.dat", "w"))) exit(1);
    if(!(fp1 = fopen("nminus.dat", "w"))) exit(1);
    if(!(fp2 = fopen("nplus.dat", "w"))) exit(1);
  }
  fprintf(fp1, FMT_I "\n", nptsm);
  fprintf(fp2, FMT_I "\n", nptsp);
  fflush(fp1);
  fflush(fp2);
  fprintf(fpm, FMT_R " " FMT_R "\n", -DIST_CUTOFF, -DIST_CUTOFF);
  fprintf(fpm, FMT_R " " FMT_R "\n", DIST_CUTOFF, -DIST_CUTOFF);
  fprintf(fpm, FMT_R " " FMT_R "\n", -DIST_CUTOFF, DIST_CUTOFF);
  fprintf(fpm, FMT_R " " FMT_R "\n", DIST_CUTOFF, DIST_CUTOFF);
  for (i = 0; i < nptsm; i++)
    fprintf(fpm, FMT_R " " FMT_R "\n", xm[i], ym[i]);
  fprintf(fpp, FMT_R " " FMT_R "\n", -DIST_CUTOFF, -DIST_CUTOFF);
  fprintf(fpp, FMT_R " " FMT_R "\n", DIST_CUTOFF, -DIST_CUTOFF);
  fprintf(fpp, FMT_R " " FMT_R "\n", -DIST_CUTOFF, DIST_CUTOFF);
  fprintf(fpp, FMT_R " " FMT_R "\n", DIST_CUTOFF, DIST_CUTOFF);
  for (i = 0; i < nptsp; i++)
    fprintf(fpp, FMT_R " " FMT_R "\n", xp[i], yp[i]);
  fprintf(fpm, "\n");
  fprintf(fpp, "\n");
  fflush(fpm);
  fflush(fpp);
}  

#define BOXXL (NX * STEP)
#define BOXYL (NY * STEP)

/* Consider all +/- pairs */
void print_pair_dist(char *file) {

  INT i, j;
  REAL dx, dy;
  FILE *fp;

  if(!(fp = fopen(file, "w"))) {
    fprintf(stderr, "Can't open pair dist file.\n");
    exit(1);
  }
  for (i = 0; i < nptsp; i++)
    for (j = 0; j < nptsm; j++) {
      dx = xp[i] - xm[j];
      dy = yp[i] - ym[j];
// We don't have periodic boundary at the moment - cylinder!
#if 0
      /* periodic boundary */
      dx -= BOXXL * (REAL) ((INT) (0.5 + dx / BOXXL));
      dy -= BOXYL * (REAL) ((INT) (0.5 + dy / BOXYL));
#endif
      fprintf(fp, FMT_R "\n", SQRT(dx * dx + dy * dy));
  }
  fclose(fp);
}

/* Consider only nearest +/- pairs */
void print_pair_dist2(char *file) {

  INT i, j;
  REAL dx, dy, m, tmp;
  FILE *fp;

  if(!(fp = fopen(file, "w"))) {
    fprintf(stderr, "Can't open pair dist file.\n");
    exit(1);
  }
  for (i = 0; i < nptsp; i++) {
    m = 1E18;
    for (j = 0; j < nptsm; j++) {
      dx = xp[i] - xm[j];
      dy = yp[i] - ym[j];
// We don't have periodic boundary at the moment - cylinder!
#if 0
      /* periodic boundary */
      dx -= BOXXL * (REAL) ((INT) (0.5 + dx / BOXXL));
      dy -= BOXYL * (REAL) ((INT) (0.5 + dy / BOXYL));
#endif
      if((tmp = SQRT(dx * dx + dy * dy)) < m) m = tmp;
    }
    fprintf(fp, FMT_R "\n", m);
  }
  fclose(fp);
}

/* Consider only nearest +/+ or -/- pairs */
void print_pair_dist3(char *file) {

  INT i, j;
  REAL dx, dy, m, tmp;
  FILE *fp;

  if(!(fp = fopen(file, "w"))) {
    fprintf(stderr, "Can't open pair dist file.\n");
    exit(1);
  }
  for (i = 0; i < nptsp; i++) { // +/+
    m = 1E18;
    for (j = 0; j < nptsp; j++) {
      dx = xp[i] - xp[j];
      dy = yp[i] - yp[j];
// We don't have periodic boundary at the moment - cylinder!
#if 0
      /* periodic boundary */
      dx -= BOXXL * (REAL) ((INT) (0.5 + dx / BOXXL));
      dy -= BOXYL * (REAL) ((INT) (0.5 + dy / BOXYL));
#endif
      if((tmp = SQRT(dx * dx + dy * dy)) < m && i != j) m = tmp;
    }
    fprintf(fp, FMT_R "\n", m);
  }
  for (i = 0; i < nptsm; i++) {  // -/-
    m = 1E18;
    for (j = 0; j < nptsm; j++) {
      dx = xm[i] - xm[j];
      dy = ym[i] - ym[j];
// We don't have periodic boundary at the moment - cylinder!
#if 0
      /* periodic boundary */
      dx -= BOXXL * (REAL) ((INT) (0.5 + dx / BOXXL));
      dy -= BOXYL * (REAL) ((INT) (0.5 + dy / BOXYL));
#endif
      if((tmp = SQRT(dx * dx + dy * dy)) < m && i != j) m = tmp;
    }
    fprintf(fp, FMT_R "\n", m);
  }


  fclose(fp);
}

/* Calculate the line distance distribution from the origin (for diffusive spread) */
void print_distance_dist(char *file) {

  INT i;
  FILE *fp;

  if(!(fp = fopen(file, "w"))) {
    fprintf(stderr, "Can't open pair dist file.\n");
    exit(1);
  }
  for (i = 0; i < nptsp; i++)
    fprintf(fp, FMT_R "\n", SQRT(xp[i] * xp[i] + yp[i] * yp[i]));
  for (i = 0; i < nptsm; i++)
    fprintf(fp, FMT_R "\n", SQRT(xm[i] * xm[i] + ym[i] * ym[i]));
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
  r = SQRT(x * x + y * y);
  angle = ATAN2(x, y);

  return (1.0 - EXP(-r)) * SQRT(rho0) * CEXP(I * dir * (angle + offset));
}

REAL complex random_start(void *prm, REAL x, REAL y, REAL z) {

  cgrid *grid = (cgrid *) prm;
  INT i = ((INT) ((x + grid->x0) / grid->step) + grid->nx / 2),
      j = ((INT) ((y + grid->y0) / grid->step) + grid->ny / 2),
      k = ((INT) ((z + grid->z0) / grid->step) + grid->nz / 2),
      ny = grid->ny, nz = grid->nz;

// DEBUG: 3D!!!
#if 1
  grid->value[i * ny * nz + j * nz + k] = SQRT(rho0) * CEXP(I * 2.0 * (drand48() - 0.5) * M_PI);
  if(k) return grid->value[i * ny * nz + j * nz]; // k = 0
  else return grid->value[i * ny * nz + j * nz + k];
#else
  return SQRT(rho0) * CEXP(I * 2.0 * (drand48() - 0.5) * M_PI);
#endif
}

REAL complex init_guess(void *NA, REAL x, REAL y, REAL z) {

  if(x * x + y * y < HE_RADIUS * HE_RADIUS) return SQRT(rho0);
  else return 0.0;
}

int main(int argc, char **argv) {

  dft_ot_functional *otf;
  cgrid *potential_store;
  rgrid *rworkspace;
  wf *gwf;
#ifdef PC
  wf *gwfp;
#endif
  FILE *fp;
  INT iter, try, i, j, k;
  REAL kin, pot, n, linep[5], linem[5], ebulk, ethr;
  char buf[512], file[512];
  grid_timer timer;
  REAL complex tstep;
#ifdef KSPECTRUM
  REAL *bins = NULL;
#endif

  if(argc < 2) {
    fprintf(stderr, "Usage: film <gpu1> <gpu2> ...\n");
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

#ifdef HE_NORM
  cgrid_map(gwf->grid, init_guess, NULL);
#else
  cgrid_constant(gwf->grid, SQRT(rho0));
#endif

  /* setup initial guess for vortex lines */
#ifdef RANDOM_INITIAL
  printf("Random initial guess.\n");
  cgrid_map(gwf->grid, random_start, gwf->grid);
#endif

  tstep = -I * ITS / GRID_AUTOFS;
  /* Relax vortices for a bit (1: the column) */
  printf("Cylinder equilibriation...");
#ifdef HE_NORM
  gwf->norm = HE_NORM;
#endif
  for (iter = 0; iter < START1; iter++) {
    grid_timer_start(&timer);
    cgrid_constant(potential_store, -mu0);
    dft_ot_potential(otf, potential_store, gwf);
    grid_wf_propagate(gwf, potential_store, tstep);
    printf("Iteration " FMT_I " - Wall clock time = " FMT_R " seconds.\n", iter, grid_timer_wall_clock_time(&timer)); fflush(stdout);
#ifdef HE_NORM
    grid_wf_normalize(gwf);
#endif
  }
  printf("done.\n");

#ifdef MANUAL_LINES
  linep[0] = 0.0;
  linep[1] = 50.0;
  linep[2] = 0.0;
  linep[3] = 1.0;
  linep[4] = 0.0;
  cgrid_map(potential_store, vline, linep);
  cgrid_product(gwf->grid, gwf->grid, potential_store);
  cgrid_multiply(gwf->grid, 1.0 / SQRT(rho0));
  printf("Line (%c): " FMT_R "," FMT_R "\n", linep[3]==1.0?'+':'-', linep[0], linep[1]); fflush(stdout);

  linep[0] = 0.0;
  linep[1] = -50.0;
  linep[2] = 0.0;
  linep[3] = -1.0;
  linep[4] = 0.0;
  cgrid_map(potential_store, vline, linep);
  cgrid_product(gwf->grid, gwf->grid, potential_store);
  cgrid_multiply(gwf->grid, 1.0 / SQRT(rho0));
  printf("Line (%c): " FMT_R "," FMT_R "\n", linep[3]==1.0?'+':'-', linep[0], linep[1]); fflush(stdout);
#endif

#ifdef RANDOM_LINES
  printf("Random line positions initial guess.\n");
  srand48(RANDOM_SEED); // or time(0)
  printf("Random seed = %ld\n", RANDOM_SEED);
  printf("Initial pair coverage = " FMT_R " pairs / Bohr^2\n", NPAIRS / (M_PI * MAX_DIST * MAX_DIST));
  printf("Initial pair coverage = " FMT_R " pairs / Angs^2\n", NPAIRS / (GRID_AUTOANG * GRID_AUTOANG * M_PI * MAX_DIST * MAX_DIST));
  printf("Initial line coverage = " FMT_R " pairs / Bohr^2\n", 2.0 * NPAIRS / (M_PI * MAX_DIST * MAX_DIST));
  printf("Initial line coverage = " FMT_R " pairs / Angs^2\n", 2.0 * NPAIRS / (GRID_AUTOANG * GRID_AUTOANG * M_PI * MAX_DIST * MAX_DIST));
  nptsm = nptsp = 0;
  for (iter = 0; iter < NPAIRS; iter++) {
    REAL rv;
    printf("Attempting pair #" FMT_I "\n", iter); fflush(stdout);
    // First line in pair
    do {
#ifdef MAX_DIST
      xp[nptsp] = linep[0] = 2.0 * (drand48() - 0.5) * MAX_DIST;
      yp[nptsp] = linep[1] = 2.0 * (drand48() - 0.5) * MAX_DIST;
#else
      xp[nptsp] = linep[0] = -(STEP/2.0) * NX + drand48() * STEP * NX; // origin +- displacement
      yp[nptsp] = linep[1] = -(STEP/2.0) * NY + drand48() * STEP * NY;
#endif
      linep[2] = 0.0;
    } while(check_proximity(linep[0], linep[1], PAIR_DIST) || check_boundary(linep[0], linep[1], MAX_DIST));
    nptsp++;
    linep[3] = 1.0; 
    linep[4] = 0.0;
    // Second in pair
#ifdef UNRESTRICTED_PAIRS
    do {
#ifdef MAX_DIST
      xm[nptsm] = linem[0] = 2.0 * (drand48() - 0.5) * MAX_DIST;
      ym[nptsm] = linem[1] = 2.0 * (drand48() - 0.5) * MAX_DIST;
#else
      xm[nptsm] = linem[0] = -(STEP/2.0) * NX + drand48() * STEP * NX; // origin +- displacement
      ym[nptsm] = linem[1] = -(STEP/2.0) * NY + drand48() * STEP * NY;
#endif
    } while(check_proximity(linem[0], linem[1], PAIR_DIST) || check_boundary(linem[0], linem[1], MAX_DIST));
    nptsm++;
    linem[2] = 0.0;
    linem[3] = -1.0; 
    linem[4] = 0.0;
#else
    linem[0] = 0.0; 
    try = 0;
    do {
      rv = drand48();
      linem[0] = linep[0] + SIN(2.0 * M_PI * rv) * PAIR_DIST;
      linem[1] = linep[1] + COS(2.0 * M_PI * rv) * PAIR_DIST;
      xm[nptsm] = linem[0];
      ym[nptsm] = linem[1];
      if(try > NRETRY) {
        nptsp--;
        iter--;
        printf("Too many retries - start over.\n"); fflush(stdout);
        break;
      }
      try++;
    } while (check_proximity(linem[0], linem[1], PAIR_DIST) || check_boundary(linem[0], linem[1], MAX_DIST));
    if(try > NRETRY) continue;    
    nptsm++;
    linem[3] = -1.0; 
    linem[4] = 0.0;
#endif

    cgrid_map(potential_store, vline, linep);
    cgrid_product(gwf->grid, gwf->grid, potential_store);
    cgrid_multiply(gwf->grid, 1.0 / SQRT(rho0));
    printf("Pair+ " FMT_I ": " FMT_R "," FMT_R "\n", iter, linep[0], linep[1]); fflush(stdout);

    cgrid_map(potential_store, vline, linem);
    cgrid_product(gwf->grid, gwf->grid, potential_store);
    cgrid_multiply(gwf->grid, 1.0 / SQRT(rho0));
    printf("Pair- " FMT_I ": " FMT_R "," FMT_R "\n", iter, linem[0], linem[1]); fflush(stdout);
  }
#endif

  /* Relax vortices for a bit (2: after putting in vortices) */
  printf("Vortex equilibriation...");
  tstep = -I * ITS / GRID_AUTOFS;
  printf("START2 = " FMT_I "\n", START2);
  for (iter = 0; iter < START2; iter++) {
    grid_timer_start(&timer);
    cgrid_constant(potential_store, -mu0);
    dft_ot_potential(otf, potential_store, gwf);
    grid_wf_propagate(gwf, potential_store, tstep);
    printf("Iteration " FMT_I " - Wall clock time = " FMT_R " seconds.\n", iter, grid_timer_wall_clock_time(&timer)); fflush(stdout);
#ifdef HE_NORM
    grid_wf_normalize(gwf);    
#endif
  }
  printf("done.\n");

  printf("Starting dynamics.\n");
  grid_timer_start(&timer);
  for (iter = 0; iter < MAXITER; iter++) {
    tstep = TS * (1.0 - I * LAMBDA) / GRID_AUTOFS;     
    if(iter == 0 || !(iter % NTH)) {
      printf("Iteration " FMT_I " - Wall clock time = " FMT_R " seconds.\n", iter, grid_timer_wall_clock_time(&timer)); fflush(stdout);
#ifdef LINE_LOCATIONS_ONLY
      grid_wf_probability_flux_x(gwf, otf->workspace1);
      if(!otf->workspace2) otf->workspace2 = rgrid_clone(otf->density, "OT workspace 2");
      if(!otf->workspace3) otf->workspace3 = rgrid_clone(otf->density, "OT workspace 3");
      grid_wf_probability_flux_y(gwf, otf->workspace2);
      rgrid_rot(NULL, NULL, otf->workspace3, otf->workspace1, otf->workspace2, NULL);
      grid_wf_density(gwf, otf->density);
      locate_lines(otf->density, otf->workspace3);
      print_lines();
      sprintf(buf, "film-" FMT_I ".pair", iter);
      print_pair_dist(buf);
      sprintf(buf, "film-" FMT_I ".nopair", iter);  // Nearest +/-
      print_pair_dist2(buf);
      sprintf(buf, "film-" FMT_I ".nspair", iter);  // Nearest +/+ or -/-
      print_pair_dist3(buf);
      sprintf(buf, "film-" FMT_I ".dst", iter);  // distance distrib with respect to origin
      print_distance_dist(buf);
#else
      sprintf(buf, "film-" FMT_I, iter);
      cgrid_write_grid(buf, gwf->grid);
#endif
#ifdef KSPECTRUM
      if(!(otf->workspace1)) otf->workspace1 = rgrid_clone(otf->density, "OT Workspace 1");
      if(!(otf->workspace2)) otf->workspace2 = rgrid_clone(otf->density, "OT Workspace 2");
      if(!(otf->workspace3)) otf->workspace3 = rgrid_clone(otf->density, "OT Workspace 3");
      if(!(otf->workspace4)) otf->workspace4 = rgrid_clone(otf->density, "OT Workspace 4");
      if(!(otf->workspace5)) otf->workspace5 = rgrid_clone(otf->density, "OT Workspace 5");

      if(!bins) {
        if(!(bins = (REAL *) malloc(sizeof(REAL) * NBINS))) {
          fprintf(stderr, "Can't allocate memory for bins.\n");
          exit(1);
        }
      }

      /* The whole thing */
      grid_wf_KE(gwf, bins, BINSTEP, NBINS, otf->workspace1, otf->workspace2, otf->workspace3, DENS_EPS);
      sprintf(file, "ke-" FMT_I ".dat", iter);
      if(!(fp = fopen(file, "w"))) {
        fprintf(stderr, "Can't open %s.\n", file);
        exit(1);
      }
      for(i = 1; i < NBINS; i++)  /* Leave out the DC component (= zero) */
        fprintf(fp, FMT_R " " FMT_R "\n", (BINSTEP * (REAL) i) / GRID_AUTOANG, bins[i] * GRID_AUTOK * GRID_AUTOANG); /* Angs^{-1} K*Angs */
      fclose(fp);

      /* Incompressible part */
      grid_wf_incomp_KE(gwf, bins, BINSTEP, NBINS, otf->workspace1, otf->workspace2, otf->workspace3, otf->workspace4, otf->workspace5, DENS_EPS);
      sprintf(file, "ke-incomp-" FMT_I ".dat", iter);
      if(!(fp = fopen(file, "w"))) {
        fprintf(stderr, "Can't open %s.\n", file);
        exit(1);
      }
      for(i = 1; i < NBINS; i++)  /* Leave out the DC component (= zero) */
        fprintf(fp, FMT_R " " FMT_R "\n", (BINSTEP * (REAL) i) / GRID_AUTOANG, bins[i] * GRID_AUTOK * GRID_AUTOANG); /* Angs^{-1} K*Angs */
      fclose(fp);

      /* Compressible part */
      grid_wf_comp_KE(gwf, bins, BINSTEP, NBINS, otf->workspace1, otf->workspace2, otf->workspace3, otf->workspace4, DENS_EPS);
      sprintf(file, "ke-comp-" FMT_I ".dat", iter);
      if(!(fp = fopen(file, "w"))) {
        fprintf(stderr, "Can't open %s.\n", file);
        exit(1);
      }
      for(i = 1; i < NBINS; i++)  /* Leave out the DC component (= zero) */
        fprintf(fp, FMT_R " " FMT_R "\n", (BINSTEP * (REAL) i) / GRID_AUTOANG, bins[i] * GRID_AUTOK * GRID_AUTOANG); /* Angs^{-1} K*Angs */
      fclose(fp);
#endif
      kin = grid_wf_energy(gwf, NULL);            /* Kinetic energy for gwf */
      dft_ot_energy_density(otf, rworkspace, gwf);
      n = grid_wf_norm(gwf);
      pot = rgrid_integral(rworkspace) - mu0 * n;
      printf("Iteration " FMT_I " helium natoms       = " FMT_R " particles.\n", iter, n);   /* Energy / particle in K */
      printf("Iteration " FMT_I " helium kinetic      = " FMT_R " K\n", iter, kin * GRID_AUTOK);  /* Print result in K */
      printf("Iteration " FMT_I " helium potential    = " FMT_R " K\n", iter, pot * GRID_AUTOK);  /* Print result in K */
      printf("Iteration " FMT_I " helium energy       = " FMT_R " K\n", iter, (kin + pot) * GRID_AUTOK);  /* Print result in K */
      printf("Iteration " FMT_I " heat                = " FMT_R " J / g\n", iter, (kin + pot) * GRID_AUTOJ * GRID_AVOGADRO / (4.0 * n));  /* 4.0 g/mol = 4He */
      fflush(stdout);
      grid_timer_start(&timer);
    }

    if(iter == 5) grid_fft_write_wisdom(NULL);

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
#ifdef RANDOM_WIDTH
    cgrid_random_uniform(gwf->grid, RANDOM_WIDTH);
#endif  
  }
  return 0;
}
