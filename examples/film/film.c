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
#define TS 40.0 /* fs */
#define NX 16
#define NY 2048
#define NZ 2048
#define STEP 2.0

/* Vortex line params */
#define RANDOM_SEED 1234567L /* Random seed for generating initial vortex line coordinates */
#define NPAIRS 128        /* Number of + and - vortex pairs */
#define PAIR_DIST 100.0  /* Minimum distance between vortices allowed */
#define ITS (0.02 * TS)  /* Imag. time component (dissipation; 0 = none or 1 = full) */

/* Pressure */
#define PRESSURE (0.0 / GRID_AUTOBAR)

/* Start simulation after this many iterations */
#define START 200

/* Output every NTH iteration */
#define NTH 10000

/* Use all threads available on the computer */
#define THREADS 0

/* Print vortex line locations only? (otherwise write full grids) */
#define LINE_LOCATIONS_ONLY

/* safe zone without boundary interference (was / 2) */
#define SNY (0.9*NY)
#define SNZ (0.9*NZ)

/* Vortex line search parameters */
#define MIN_DIST_CORE 3.0
#define ADJUST 0.75

REAL rho0;

REAL yp[NPAIRS], zp[NPAIRS], ym[NPAIRS], zm[NPAIRS];
INT nptsp = 0, nptsm = 0;

INT check_proximity(REAL yy, REAL zz, REAL dist) {

  INT i;
  REAL y2, z2;

  for (i = 0; i < nptsm; i++) {
    y2 = yy - ym[i]; y2 *= y2;
    z2 = zz - zm[i]; z2 *= z2;
    if(SQRT(y2 + z2) < dist) return 1;  // something too close
  }

  for (i = 0; i < nptsp; i++) {
    y2 = yy - yp[i]; y2 *= y2;
    z2 = zz - zp[i]; z2 *= z2;
    if(SQRT(y2 + z2) < dist) return 1;  // something too close
  }

  return 0; // OK
}

void locate_lines(rgrid *rot) {

  INT i, k, j;
  REAL yy, zz, m, tmp;

  nptsm = nptsp = 0;
  i = NX / 2;
  m = rgrid_max(rot) * ADJUST;
  printf("m = " FMT_R "\n", m);
  for (j = 0; j < NY; j++) {
    yy = ((REAL) (j - NY/2)) * STEP;
    for (k = 0; k < NZ; k++) {
      zz = ((REAL) (k - NZ/2)) * STEP;
      if(FABS(tmp = rgrid_value_at_index(rot, i, j, k)) > m && !check_proximity(yy, zz, MIN_DIST_CORE)) {
        printf(FMT_R " " FMT_R, yy, zz);
        if(tmp < 0.0) {
          if(nptsm >= NPAIRS) {
            fprintf(stderr, "Error: More lines than generated initially!\n");
            exit(1);
          }
          ym[nptsm] = yy;
          zm[nptsm] = zz;
          nptsm++;
          printf(" -\n");
        } else {
          if(nptsp >= NPAIRS) {
            fprintf(stderr, "Error: More lines than generated initially!\n");
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

//  printf("XXX " FMT_I "\n", npts);
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
  rgrid *rworkspace, *fy, *fz;
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
  fy = rgrid_clone(rworkspace, "fy");
  fz = rgrid_clone(rworkspace, "fz");
 
  /* setup initial guess for vortex lines */
  srand48(RANDOM_SEED); // or time(0)
  printf("Random seed = %ld\n", RANDOM_SEED);
  cgrid_constant(gwf->grid, SQRT(rho0));
  nptsm = nptsp = 0;
  for (iter = 0; iter < NPAIRS; iter++) {
    REAL rv;
    // First line in pair
    do {
      line[0] = 0.0;
      yp[nptsp] = line[1] = -(STEP/2.0) * SNY + drand48() * STEP * SNY; // origin +- displacement
      zp[nptsp] = line[2] = -(STEP/2.0) * SNZ + drand48() * STEP * SNZ;
//    } while(FABS(line[1]) > STEP * SNY / 2.0 || FABS(line[2]) > STEP * SNZ / 2.0 || check_proximity(line[1], line[2], PAIR_DIST));
    } while(check_proximity(line[1], line[2], PAIR_DIST));
    nptsp++;
    line[3] = 1.0; line[4] = 0.0;
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
      ym[nptsm] = line[1];
      zm[nptsm] = line[2];
    } while (check_proximity(line[1], line[2], PAIR_DIST));
    nptsm++;
    line[3] = -1.0; 
    line[4] = 0.0;
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
      grid_wf_probability_flux(gwf, NULL, fy, fz);
      rgrid_rot(rworkspace, NULL, NULL, NULL, fy, fz);
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
