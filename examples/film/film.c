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
#define TS 10.0 /* fs */
#define NX 2048
#define NY 2048
#define NZ 16
#define STEP 2.0
#define MAXITER 8000000

/* kmax setting (0.0 to disable) */
#define KMAX (0.0 * GRID_AUTOANG)

/* GPU allocation */
#ifdef USE_CUDA
#define NGPUS 8
int gpus[] = {0, 1, 2, 3, 4, 5, 6, 7};
#endif

/* Functional to use (was DFT_OT_PLAIN; GP2 is test) */
#define FUNCTIONAL DFT_OT_PLAIN

/* Predict-correct? (with abs BC only works with CN) */
//#define PC

/* Propagator: WF_2ND_ORDER_CN or WF_2ND_ORDER_FFT */
#define PROPAGATOR WF_2ND_ORDER_CFFT

#if PROPAGATOR == WF_2ND_ORDER_CN
#define BOUNDARY WF_NEUMANN_BOUNDARY
#else
#define BOUNDARY WF_PERIODIC_BOUNDARY
#endif

/* Vortex line params */
#define RANDOM_SEED 1234567L /* Random seed for generating initial vortex line coordinates */
#define NPAIRS 100           /* Number of + and - vortex pairs */
#define PAIR_DIST 40.0       /* Min. distance between + and - vortex pairs */
#define MANUAL_LINES         /* Enter vortex lines manually */
#define UNRESTRICTED_PAIRS   /* If defined, PAIR_DIST for the + and - pairs is not enforced */
#define ITS (0.02 * TS)      /* Imag. time component (dissipation; 0 = none or 1 = full) */
                             /* Tsubota gamma = 0.02 */
#define NRETRY   10000       /* # of retries for locating the pair. If not successful, start over */

/* Pressure */
#define PRESSURE (0.0 / GRID_AUTOBAR)

/* Start simulation after this many iterations */
#define START (400)  // vortex lines (was 4000)

/* Output every NTH iteration (was 5000) */
#define NTH 500

/* Absorbing boundary region */
#define ABS_WIDTH_X 100.0
#define ABS_WIDTH_Y 100.0

/* Use all threads available on the computer */
#define THREADS 0

/* Print vortex line locations only? (otherwise write full grids) */
//#define LINE_LOCATIONS_ONLY

/* Vortex line search parameters */
#define MIN_DIST_CORE 4.0
#define ADJUST 0.8

REAL rho0, mu0;

REAL xp[2*NPAIRS], yp[2*NPAIRS], xm[2*NPAIRS], ym[2*NPAIRS];
INT nptsp = 0, nptsm = 0;

INT check_proximity(REAL xx, REAL yy, REAL dist) {

  INT i;
  REAL x2, y2;

  for (i = 0; i < nptsm; i++) {
    x2 = xx - xm[i]; x2 *= x2;
    y2 = yy - ym[i]; y2 *= y2;
    if(SQRT(x2 + y2) < dist) return 1;  // too close
  }

  for (i = 0; i < nptsp; i++) {
    x2 = xx - xp[i]; x2 *= x2;
    y2 = yy - yp[i]; y2 *= y2;
    if(SQRT(x2 + y2) < dist) return 1;  // too close
  }

  return 0; // all clear
}

int check_boundary(REAL x, REAL y) {

  if(SQRT(x*x + y*y) > 0.6 * (STEP * NX / 2.0)) return 1;  // in the boundary region

  return 0; // inside
}

void locate_lines(rgrid *rot) {

  INT i, k, j;
  REAL xx, yy, tmp;
  static REAL m = -1.0;

  nptsm = nptsp = 0;
  k = NZ / 2;
  if(m < 0.0) {
    REAL tmp;
    m = rgrid_max(rot) * ADJUST;
    tmp = FABS(rgrid_min(rot)) * ADJUST;
    if(tmp > m) m = tmp;
    printf("m = " FMT_R "\n", m);
  }
  for (i = 0; i < NX; i++) {
    xx = ((REAL) (i - NX/2)) * STEP;
    for (j = 0; j < NY; j++) {
      yy = ((REAL) (j - NY/2)) * STEP;
      if(!check_boundary(xx, yy) && FABS(tmp = rgrid_value_at_index(rot, i, j, k)) > m && !check_proximity(xx, yy, MIN_DIST_CORE)) {
        printf(FMT_R " " FMT_R, xx, yy);
        if(tmp < 0.0) {
          if(nptsm >= 2*NPAIRS) {
            fprintf(stderr, "Error(-): More lines than generated initially!\n");
            exit(1);
          }
          xm[nptsm] = xx;
          ym[nptsm] = yy;
          nptsm++;
          printf(" -\n");
        } else {
          if(nptsp >= 2*NPAIRS) {
            fprintf(stderr, "Error(+): More lines than generated initially!\n");
            exit(1);
          }
          xp[nptsp] = xx;
          yp[nptsp] = yy;
          nptsp++;
          printf(" +\n");
        }
      }
    }
  }
}

void print_lines() {

  INT i;

  printf("YYY1 " FMT_I "\n", nptsp);
  printf("YYY2 " FMT_I "\n", nptsm);
  printf("XXX\n");
  for (i = 0; i < nptsm; i++)
    printf("XXX " FMT_R " " FMT_R "\n", xm[i], ym[i]);
  for (i = 0; i < nptsp; i++)
    printf("XXX " FMT_R " " FMT_R "\n", xp[i], yp[i]);
}  

#define BOXXL (NX * STEP)
#define BOXYL (NY * STEP)
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
      /* periodic boundary */
      dx -= BOXXL * (REAL) ((INT) (0.5 + dx / BOXXL));
      dy -= BOXYL * (REAL) ((INT) (0.5 + dy / BOXYL));
      fprintf(fp, FMT_R "\n", SQRT(dx * dx + dy * dy));
  }
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

int main(int argc, char **argv) {

  dft_ot_functional *otf;
  cgrid *potential_store;
  rgrid *rworkspace;
  wf *gwf;
#ifdef PC
  wf *gwfp;
#endif
  INT iter, try, i, j, k;
  REAL kin, pot, n, linep[5], linem[5];
  char buf[512];
  grid_timer timer;
  REAL complex tstep;

#ifdef USE_CUDA
  cuda_enable(1, NGPUS, gpus);
#endif

  /* Initialize threads & use wisdom */
  grid_set_fftw_flags(1);    // FFTW_MEASURE
  grid_threads_init(THREADS);
  grid_fft_read_wisdom(NULL);

  /* Allocate wave functions */
  if(!(gwf = grid_wf_alloc(NX, NY, NZ, STEP, DFT_HELIUM_MASS, BOUNDARY, PROPAGATOR, "gwf"))) {
    fprintf(stderr, "Cannot allocate gwf.\n");
    exit(1);
  }
#ifdef PC
  if(!(gwfp = grid_wf_alloc(NX, NY, NZ, STEP, DFT_HELIUM_MASS, BOUNDARY, PROPAGATOR, "gwfp"))) {
    fprintf(stderr, "Cannot allocate gwfp.\n");
    exit(1);
  }
#endif  

  grid_wf_set_kmax(gwf, KMAX);

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

  cgrid_constant(gwf->grid, SQRT(rho0));

  /* setup initial guess for vortex lines */
#ifdef MANUAL_LINES

#if 0
  linep[0] = -200.0;
  linep[1] = 0.0;
  linep[2] = 0.0;
  linep[3] = -1.0;
  linep[4] = 0.0;
  cgrid_map(potential_store, vline, linep);
  cgrid_product(gwf->grid, gwf->grid, potential_store);
  cgrid_multiply(gwf->grid, 1.0 / SQRT(rho0));
  printf("Line (%c): " FMT_R "," FMT_R "\n", linep[3]==1.0?'+':'-', linep[0], linep[1]); fflush(stdout);
#endif

  linep[0] = 0.0;
  linep[1] = 0.0;
  linep[2] = 0.0;
  linep[3] = -1.0;
  linep[4] = 0.0;
  cgrid_map(potential_store, vline, linep);
  cgrid_product(gwf->grid, gwf->grid, potential_store);
  cgrid_multiply(gwf->grid, 1.0 / SQRT(rho0));
  printf("Line (%c): " FMT_R "," FMT_R "\n", linep[3]==1.0?'+':'-', linep[0], linep[1]); fflush(stdout);

#if 0
  linep[0] = 200.0;
  linep[1] = 0.0;
  linep[2] = 0.0;
  linep[3] = 1.0;
  linep[4] = 0.0;
  cgrid_map(potential_store, vline, linep);
  cgrid_product(gwf->grid, gwf->grid, potential_store);
  cgrid_multiply(gwf->grid, 1.0 / SQRT(rho0));
  printf("Line (%c): " FMT_R "," FMT_R "\n", linep[3]==1.0?'+':'-', linep[0], linep[1]); fflush(stdout);
#endif

#else
  srand48(RANDOM_SEED); // or time(0)
  printf("Random seed = %ld\n", RANDOM_SEED);
  nptsm = nptsp = 0;
  for (iter = 0; iter < NPAIRS; iter++) {
    REAL rv;
    printf("Attempting pair #" FMT_I "\n", iter); fflush(stdout);
    // First line in pair
    do {
      xp[nptsp] = linep[0] = -(STEP/2.0) * NX + drand48() * STEP * NX; // origin +- displacement
      yp[nptsp] = linep[1] = -(STEP/2.0) * NY + drand48() * STEP * NY;
      linep[2] = 0.0;
    } while(check_proximity(linep[0], linep[1], PAIR_DIST) || check_boundary(linep[0], linep[1]));
    nptsp++;
    linep[3] = 1.0; 
    linep[4] = 0.0;
    // Second in pair
#ifdef UNRESTRICTED_PAIRS
    do {
      xm[nptsm] = linem[0] = -(STEP/2.0) * NX + drand48() * STEP * NX; // origin +- displacement
      ym[nptsm] = linem[1] = -(STEP/2.0) * NY + drand48() * STEP * NY;
      linem[0] = 0.0;
    } while(check_proximity(linem[0], linem[1], PAIR_DIST) || check_boundary(linem[0], linem[1]));
    nptsm++;
    linem[3] = -1.0; 
    linem[4] = 0.0;
#else
    linem[0] = 0.0; 
    try = 0;
    do {
      rv = drand48();
      linem[0] += SIN(2.0 * M_PI * rv) * PAIR_DIST;
      linem[1] += COS(2.0 * M_PI * rv) * PAIR_DIST;
      xm[nptsm] = linem[0];
      ym[nptsm] = linem[1];
      if(try > NRETRY) {
        nptsp--;
        iter--;
        printf("Too many retries - start over.\n"); fflush(stdout);
        break;
      }
      try++;
    } while (check_proximity(linem[0], linem[1], PAIR_DIST) || check_boundary(linem[0], linem[1]));
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

  /* Relax vortices for a bit */
  printf("Vortex equilibriation...");
  tstep = -I * TS / GRID_AUTOFS;
  for (iter = 0; iter < START; iter++) {
    grid_timer_start(&timer);
    cgrid_constant(potential_store, -mu0);
    dft_ot_potential(otf, potential_store, gwf);
    grid_wf_propagate(gwf, potential_store, tstep);
    printf("Iteration " FMT_I " - Wall clock time = " FMT_R " seconds.\n", iter, grid_timer_wall_clock_time(&timer)); fflush(stdout);
  }
  printf("done.\n");

#ifdef PC
  grid_wf_boundary(gwf, gwfp, (INT) (ABS_WIDTH_X / STEP), NX - (INT) (ABS_WIDTH_X / STEP), (INT) (ABS_WIDTH_Y / STEP), NY - (INT) (ABS_WIDTH_Y / STEP), 0, NZ);
#else
  grid_wf_boundary(gwf, NULL, (INT) (ABS_WIDTH_X / STEP), NX - (INT) (ABS_WIDTH_X / STEP), (INT) (ABS_WIDTH_Y / STEP), NY - (INT) (ABS_WIDTH_Y / STEP), 0, NZ);
#endif

  printf("Starting dynamics.\n");
  tstep = (TS - I * ITS) / GRID_AUTOFS;     
  grid_timer_start(&timer);
  for (iter = 0; iter < MAXITER; iter++) {
    if(iter == 0 || !(iter % NTH)) {
      printf("Iteration " FMT_I " - Wall clock time = " FMT_R " seconds.\n", iter, grid_timer_wall_clock_time(&timer)); fflush(stdout);
#ifdef LINE_LOCATIONS_ONLY
      grid_wf_probability_flux_x(gwf, otf->workspace1);
      if(!otf->workspace2) otf->workspace2 = rgrid_clone(otf->density, "OT workspace 2");
      grid_wf_probability_flux_y(gwf, otf->workspace2);
      rgrid_rot(NULL, NULL, otf->density, otf->workspace1, otf->workspace2, NULL);
      locate_lines(otf->density);
      print_lines();
      sprintf(buf, "film-" FMT_I ".pair", iter);
      print_pair_dist(buf);
#else
      sprintf(buf, "film-" FMT_I, iter);
      cgrid_write_grid(buf, gwf->grid);
#endif
      kin = grid_wf_energy(gwf, NULL);            /* Kinetic energy for gwf */
      dft_ot_energy_density(otf, rworkspace, gwf);
      n = grid_wf_norm(gwf);
      pot = rgrid_integral(rworkspace) - mu0 * n;
      printf("Iteration " FMT_I " helium natoms    = " FMT_R " particles.\n", iter, n);   /* Energy / particle in K */
      printf("Iteration " FMT_I " helium kinetic   = " FMT_R " K\n", iter, kin * GRID_AUTOK);  /* Print result in K */
      printf("Iteration " FMT_I " helium potential = " FMT_R " K\n", iter, pot * GRID_AUTOK);  /* Print result in K */
      printf("Iteration " FMT_I " helium energy    = " FMT_R " K\n", iter, (kin + pot) * GRID_AUTOK);  /* Print result in K */
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
  }
  return 0;
}
