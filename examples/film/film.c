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

/* Time integration and spatial grid parameters */
#define TS 5.0 /* fs */
#define NX 16
#define NY 512
#define NZ 512
#define STEP 2.0

/* Predict-correct? */
//#define PC

/* Vortex line params */
#define RANDOM_SEED 1234567L /* Random seed for generating initial vortex line coordinates */
#define NPAIRS 100           /* Number of + and - vortex pairs */
#define PAIR_DIST 40.0       /* Min. distance between + and - vortex pairs */
#define UNRESTRICTED_PAIRS   /* If defined, PAIR_DIST for the + and - pairs is not enforced */
#define ITS (0.00 * TS)      /* Imag. time component (dissipation; 0 = none or 1 = full) */
                             /* Tsubota gamma = 0.02 */
#define NRETRY   10000       /* # of retries for locating the pair. If not successful, start over */

/* Pressure */
#define PRESSURE (0.0 / GRID_AUTOBAR)

/* Start simulation after this many iterations */
#define START 50  // vortex lines
#define START2 4000  // cylinder

/* Output every NTH iteration was 10000 */
#define NTH 10000

/* Use all threads available on the computer */
#define THREADS 0

/* Print vortex line locations only? (otherwise write full grids) */
//#define LINE_LOCATIONS_ONLY

/* safe zone without boundary interference */
#define SNY (0.9*NY)
#define SNZ (0.9*NZ)

/* Vortex line search parameters */
#define MIN_DIST_CORE 4.0
#define ADJUST 0.7

/* Initial guess parameters (undefine MU0 to impose particle # normalization) */
// #define MU0 (-6.91448 / GRID_AUTOK)
#define RADIUS (NY * STEP / 2.0 - 50.0)

REAL rho0, mu0;

REAL yp[2*NPAIRS], zp[2*NPAIRS], ym[2*NPAIRS], zm[2*NPAIRS];
INT nptsp = 0, nptsm = 0;

/* Cylinder initial guess */
REAL complex cylinder(void *NA, REAL x, REAL y, REAL z) {

  if(SQRT(y*y + z*z) < RADIUS) return SQRT(rho0);
  else return 0.0;
}

INT check_proximity(REAL yy, REAL zz, REAL dist) {

  INT i;
  REAL y2, z2;

  for (i = 0; i < nptsm; i++) {
    y2 = yy - ym[i]; y2 *= y2;
    z2 = zz - zm[i]; z2 *= z2;
    if(SQRT(y2 + z2) < dist) return 1;  // too close
  }

  for (i = 0; i < nptsp; i++) {
    y2 = yy - yp[i]; y2 *= y2;
    z2 = zz - zp[i]; z2 *= z2;
    if(SQRT(y2 + z2) < dist) return 1;  // too close
  }

  return 0; // all clear
}

int check_boundary(REAL y, REAL z) {

  if(FABS(y) >= STEP * SNY / 2.0) return 1;
  if(FABS(z) >= STEP * SNZ / 2.0) return 1;
  return 0;
}

void locate_lines(rgrid *rot) {

  INT i, k, j;
  REAL yy, zz, tmp;
  static REAL m = -1.0;

  nptsm = nptsp = 0;
  i = NX / 2;
  if(m < 0.0) {
    m = rgrid_max(rot) * ADJUST;
    printf("m = " FMT_R "\n", m);
  }
  for (j = 0; j < NY; j++) {
    yy = ((REAL) (j - NY/2)) * STEP;
    for (k = 0; k < NZ; k++) {
      zz = ((REAL) (k - NZ/2)) * STEP;
      if(FABS(tmp = rgrid_value_at_index(rot, i, j, k)) > m && !check_proximity(yy, zz, MIN_DIST_CORE)) {
        printf(FMT_R " " FMT_R, yy, zz);
        if(tmp < 0.0) {
          if(nptsm >= 2*NPAIRS) {
            fprintf(stderr, "Error(-): More lines than generated initially!\n");
            exit(1);
          }
          ym[nptsm] = yy;
          zm[nptsm] = zz;
          nptsm++;
          printf(" -\n");
        } else {
          if(nptsp >= 2*NPAIRS) {
            fprintf(stderr, "Error(+): More lines than generated initially!\n");
            exit(1);
          }
          yp[nptsp] = yy;
          zp[nptsp] = zz;
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
    printf("XXX " FMT_R " " FMT_R "\n", ym[i], zm[i]);
  for (i = 0; i < nptsp; i++)
    printf("XXX " FMT_R " " FMT_R "\n", yp[i], zp[i]);
}  

#define BOXYL (NY * STEP)
#define BOXZL (NZ * STEP)
void print_pair_dist(char *file) {

  INT i, j;
  REAL dy, dz;
  FILE *fp;

  if(!(fp = fopen(file, "w"))) {
    fprintf(stderr, "Can't open pair dist file.\n");
    exit(1);
  }
  for (i = 0; i < nptsp; i++)
    for (j = 0; j < nptsm; j++) {
      dy = yp[i] - ym[j];
      dz = zp[i] - zm[j];
      /* periodic boundary */
      dy -= BOXYL * (REAL) ((INT) (0.5 + dy / BOXYL));
      dz -= BOXZL * (REAL) ((INT) (0.5 + dz / BOXZL));
      fprintf(fp, FMT_R "\n", SQRT(dy * dy + dz * dz));
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
  r = SQRT(y * y + z * z);
  angle = ATAN2(y, z);

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
  cuda_enable(1);  // enable CUDA ?
#endif

  /* Initialize threads & use wisdom */
  grid_set_fftw_flags(1);    // FFTW_MEASURE
  grid_threads_init(THREADS);
  grid_fft_read_wisdom(NULL);

  /* Allocate wave functions */
//  if(!(gwf = grid_wf_alloc(NX, NY, NZ, STEP, DFT_HELIUM_MASS, WF_FFT_EOO_BOUNDARY, WF_2ND_ORDER_FFT, "gwf"))) {
  if(!(gwf = grid_wf_alloc(NX, NY, NZ, STEP, DFT_HELIUM_MASS, WF_PERIODIC_BOUNDARY, WF_2ND_ORDER_CFFT, "gwf"))) {
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
  if(!(otf = dft_ot_alloc(DFT_OT_PLAIN, gwf, DFT_MIN_SUBSTEPS, DFT_MAX_SUBSTEPS))) {
    fprintf(stderr, "Cannot allocate otf.\n");
    exit(1);
  }
  rho0 = dft_ot_bulk_density_pressurized(otf, PRESSURE);
  mu0 = dft_ot_bulk_chempot_pressurized(otf, PRESSURE);
  printf("Bulk mu0 = " FMT_R " K/atom, Bulk rho0 = " FMT_R " Angs^-3.\n", mu0 * GRID_AUTOK, rho0 / (GRID_AUTOANG * GRID_AUTOANG * GRID_AUTOANG));

#ifdef MU0
  mu0 = MU0;
  printf("Using mu0 = " FMT_R " K\n", MU0 * GRID_AUTOK);
#else
  mu0 = 0.0;
#endif

  /* Allocate space for external potential */
  potential_store = cgrid_clone(gwf->grid, "potential_store"); /* temporary storage */
  rworkspace = rgrid_clone(otf->density, "external potential");

  /* Set up cylinder */
  cgrid_map(gwf->grid, &cylinder, NULL);
  gwf->norm = grid_wf_norm(gwf);
  /* Relax for a bit */
  printf("Cylinder equilibriation...");
  tstep = - 5.0 * I * TS / GRID_AUTOFS;
  for (iter = 0; iter < START2; iter++) {
    grid_timer_start(&timer);
    cgrid_constant(potential_store, -mu0);
    dft_ot_potential(otf, potential_store, gwf);
    grid_wf_propagate(gwf, potential_store, tstep);
#ifndef MU0
    grid_wf_normalize(gwf);
#endif
    printf("Iteration " FMT_I " - Wall clock time = " FMT_R " seconds.\n", iter, grid_timer_wall_clock_time(&timer));
  }
  printf("done.\n");

  /* setup initial guess for vortex lines */
  srand48(RANDOM_SEED); // or time(0)
  printf("Random seed = %ld\n", RANDOM_SEED);
  nptsm = nptsp = 0;
  for (iter = 0; iter < NPAIRS; iter++) {
    REAL rv;
    printf("Attempting pair #" FMT_I "\n", iter); fflush(stdout);
    // First line in pair
    do {
      linep[0] = 0.0;
      yp[nptsp] = linep[1] = -(STEP/2.0) * SNY + drand48() * STEP * SNY; // origin +- displacement
      zp[nptsp] = linep[2] = -(STEP/2.0) * SNZ + drand48() * STEP * SNZ;
    } while(check_proximity(linep[1], linep[2], PAIR_DIST));
    nptsp++;
    linep[3] = 1.0; 
    linep[4] = 0.0;

    // Second in pair
#ifdef UNRESTRICTED_PAIRS
    do {
      linem[0] = 0.0;
      ym[nptsm] = linem[1] = -(STEP/2.0) * SNY + drand48() * STEP * SNY; // origin +- displacement
      zm[nptsm] = linem[2] = -(STEP/2.0) * SNZ + drand48() * STEP * SNZ;
    } while(check_proximity(linem[1], linem[2], PAIR_DIST));
    nptsm++;
    linem[3] = -1.0; 
    linem[4] = 0.0;
#else
    linem[0] = 0.0; 
    try = 0;
    do {
      rv = drand48();
      linem[1] += SIN(2.0 * M_PI * rv) * PAIR_DIST;
      linem[2] += COS(2.0 * M_PI * rv) * PAIR_DIST;
      ym[nptsm] = linem[1];
      zm[nptsm] = linem[2];
      if(try > NRETRY) {
        nptsp--;
        iter--;
        printf("Too many retries - start over.\n"); fflush(stdout);
        break;
      }
      try++;
    } while (check_proximity(linem[1], linem[2], PAIR_DIST) || check_boundary(linem[1], linem[2]));
    if(try > NRETRY) continue;    
    nptsm++;
    linem[3] = -1.0; 
    linem[4] = 0.0;
#endif

    cgrid_map(potential_store, vline, linep);
    cgrid_product(gwf->grid, gwf->grid, potential_store);
    cgrid_multiply(gwf->grid, 1.0 / SQRT(rho0));
    printf("Pair+ " FMT_I ": " FMT_R "," FMT_R "\n", iter, linep[1], linep[2]); fflush(stdout);

    cgrid_map(potential_store, vline, linem);
    cgrid_product(gwf->grid, gwf->grid, potential_store);
    cgrid_multiply(gwf->grid, 1.0 / SQRT(rho0));
    printf("Pair- " FMT_I ": " FMT_R "," FMT_R "\n", iter, linem[1], linem[2]); fflush(stdout);
  }

  /* Relax vortices for a bit */
  printf("Vortex equilibriation...");
  tstep = -I * TS / GRID_AUTOFS;
  for (iter = 0; iter < START; iter++) {
    grid_timer_start(&timer);
    cgrid_constant(potential_store, -mu0);
    dft_ot_potential(otf, potential_store, gwf);
    grid_wf_propagate(gwf, potential_store, tstep);
#ifndef MU0
    grid_wf_normalize(gwf);
#endif
    printf("Iteration " FMT_I " - Wall clock time = " FMT_R " seconds.\n", iter, grid_timer_wall_clock_time(&timer));
  }
  printf("done.\n");

  printf("Starting dynamics.\n");
  mu0 = dft_ot_bulk_chempot_pressurized(otf, PRESSURE);  // Use bulk chemical potential (approximate)
  tstep = (TS - I * ITS) / GRID_AUTOFS;     
  grid_timer_start(&timer);
  for (iter = 0; iter < 800000; iter++) {
    if(iter == 0 || !(iter % NTH)) {
      printf("Iteration " FMT_I " - Wall clock time = " FMT_R " seconds.\n", iter, grid_timer_wall_clock_time(&timer));
#ifdef LINE_LOCATIONS_ONLY
      grid_wf_probability_flux(gwf, NULL, otf->workspace1, otf->workspace2);
      rgrid_rot(otf->density, NULL, NULL, NULL, otf->workspace1, otf->workspace2);
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
      pot = rgrid_integral(rworkspace) - mu0;
      n = grid_wf_norm(gwf);
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
// If chem pot is not known, this could help with imag time?
//     grid_wf_normalize(gwf);
  }
  return 0;
}
