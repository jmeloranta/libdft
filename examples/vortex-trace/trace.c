/*
 * Trace vortex loops: produce vortex ring count and their lengths.
 *
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <grid/grid.h>
#include <grid/au.h>
#include <dft/dft.h>
#include <dft/ot.h>

#define THREADS 0

#define NLOOPS 512
#define NPOINTS 2048

/* Threshold for starting to follow a line */
#define THRESH2 0.9
/* Threshold for considering a point while folloing a line */
#define THRESH 0.02
/* Usually WIDTH = WIDTH2 = DROP */
/* Point closer than this to the line may be dropped (in units of grid step). ~ vortex diameter */
#define WIDTH 2
/* Point closer than this is considered as part of one line (in units of grid step). */
#define WIDTH2 2 
/* Move back parameter */
#define DROP 2
/* Power for |curl(rho v)/rho|^n */
#define POWER 1.0
/* Epsilon for division */
#define EPS 1.0E-4

INT nloops = 0, npoints[NLOOPS], nx, ny, nz;
INT xp[NLOOPS][NPOINTS], yp[NLOOPS][NPOINTS], zp[NLOOPS][NPOINTS];
REAL threshold;
rgrid *circ, *cur_x, *cur_y, *cur_z, *density;
cgrid *tmp;

INT dist(INT i, INT j, INT k, INT ii, INT jj, INT kk) {

  INT nx = circ->nx, ny = circ->ny, nz = circ->nz, nx2 = nx / 2, ny2 = ny / 2, nz2 = nz / 2;
  INT iii, jjj, kkk;

  i %= nx;
  if(i < 0) i += nx;
  j %= ny;
  if(j < 0) j += ny;
  k %= nz;
  if(k < 0) k += nz;

  ii %= nx;
  if(ii < 0) ii += nx;
  jj %= ny;
  if(jj < 0) jj += ny;
  kk %= nz;
  if(kk < 0) kk += nz;

  iii = ABS(i - ii);
  if(iii > nx2) iii = iii - nx2 + 1;
  jjj = ABS(j - jj);
  if(jjj > ny2) jjj = jjj - ny2 + 1;
  kkk = ABS(k - kk);
  if(kkk > nz2) kkk = kkk - nz2 + 1;

  return iii * iii + jjj * jjj + kkk * kkk;
}

INT is_part(INT loop, INT i, INT j, INT k) { // is given point part of existing loop?

  INT m;

  for (m = 0; m < npoints[loop]; m++)
    if(dist(xp[loop][m], yp[loop][m], zp[loop][m], i, j, k) <= WIDTH*WIDTH) return 1;
  return 0;
}

INT is_part2(INT loop, INT i, INT j, INT k) { // is given point part of existing loop, consider up to 0 npoints-DROP

  INT m;

  if(npoints[loop] < DROP) return 0;
  for (m = 0; m < npoints[loop]-DROP; m++)
    if(dist(xp[loop][m], yp[loop][m], zp[loop][m], i, j, k) <= WIDTH*WIDTH) return 1;
  return 0;
}

void trace_line(rgrid *circ, INT i, INT j, INT k) {

  INT l, m, n, max_l = 0, max_m = 0, max_n = 0, nx = circ->nx, ny = circ->ny, nz = circ->nz;
  REAL mval, tmp;

  /* Check if point already part of a loop */
  for (l = 0; l < nloops; l++)
    if(is_part(l, i, j, k)) return;
  
  /* Trace loop and add the points. End if closes or no new point found */
  while (1) {
    /* Search for new i, j, k */
    mval = 0.0;
    for(l = i-WIDTH2; l <= i+WIDTH2; l++) 
      for(m = j-WIDTH2; m <= j+WIDTH2; m++) 
        for(n = k-WIDTH2; n <= k+WIDTH2; n++) {
           if(l == i && m == j && n == k) continue;
           if(!is_part2(nloops, l, m, n) && (tmp = rgrid_value_at_index(circ, l, m, n)) > threshold && tmp > mval) {
             mval = tmp;
             max_l = l;
             max_m = m;
             max_n = n;
           }
        }
    if(mval == 0.0) break; // Dead end
    i = max_l;
    j = max_m;
    k = max_n;
    max_l = i % nx;
    if(max_l < 0) max_l += nx;
    max_m = j % ny;
    if(max_m < 0) max_m += ny;
    max_n = k % nz;
    if(max_n < 0) max_n += nz;
    xp[nloops][npoints[nloops]] = max_l;
    yp[nloops][npoints[nloops]] = max_m;
    zp[nloops][npoints[nloops]] = max_n;
    npoints[nloops]++;
    if(npoints[nloops] == NPOINTS) {
      fprintf(stderr, "Too many points.\n");
      exit(1);
    }
  }
  nloops++;
  if(nloops == NLOOPS) {
    fprintf(stderr, "Too many loops.\n");
    exit(1);
  }
}

int main(int argc, char **argv) {

  wf *wf;
  INT i, j, k;
  FILE *fp;
  char buf[512];

  /* Initialize threads & use wisdom */
  grid_set_fftw_flags(1);    // FFTW_MEASURE
  grid_threads_init(THREADS);
  grid_fft_read_wisdom(NULL);

  if(!(fp = fopen(argv[1], "r"))) {
    fprintf(stderr, "Can't open file %s.\n", argv[1]);
    exit(1);
  }
  tmp = cgrid_read(NULL, fp);
  fclose(fp);
  fprintf(stderr, "grid: nx = " FMT_I ", ny = " FMT_I ", nz = " FMT_I ", step = " FMT_R ".\n", tmp->nx, tmp->ny, tmp->nz, tmp->step);

  /* Allocate grid functions */
  if(!(wf = grid_wf_alloc(tmp->nx, tmp->ny, tmp->nz, tmp->step, DFT_HELIUM_MASS, WF_PERIODIC_BOUNDARY, WF_2ND_ORDER_FFT, "wf"))) {
    fprintf(stderr, "Cannot allocate grid.\n");
    exit(1);
  }
  cgrid_free(wf->grid);
  wf->grid = tmp;

  /* Allocate grid functions */
  if(!(circ = rgrid_alloc(tmp->nx, tmp->ny, tmp->nz, tmp->step, RGRID_PERIODIC_BOUNDARY, NULL, "circ"))) {
    fprintf(stderr, "Cannot allocate grid.\n");
    exit(1);
  }
  cur_x = rgrid_clone(circ, "cur_x");
  cur_y = rgrid_clone(circ, "cur_y");
  cur_z = rgrid_clone(circ, "cur_z");
  density = rgrid_clone(circ, "density");
  
  cgrid_read_grid(wf->grid, argv[1]);

  grid_wf_probability_flux(wf, cur_x, cur_y, cur_z);
  grid_wf_density(wf, density);

  rgrid_abs_rot(circ, cur_x, cur_y, cur_z);
  rgrid_division_eps(circ, circ, density, EPS);
  rgrid_abs_power(circ, circ, POWER);
  threshold = rgrid_max(circ) * THRESH;
  fprintf(stderr, "Threshold = " FMT_R "\n", threshold);
//  rgrid_write_grid("test", circ); exit(0);

  for(i = 0; i < NLOOPS; i++)
    npoints[i] = 0;

  for(i = 0; i < tmp->nx; i++)
    for(j = 0; j < tmp->ny; j++)
      for(k = 0; k < tmp->nz; k++)
//        if(rgrid_value_at_index(circ, i, j, k) > threshold) printf(FMT_I " " FMT_I " " FMT_I "\n", i, j, k);
        if(rgrid_value_at_index(circ, i, j, k) > (THRESH2/THRESH) * threshold) trace_line(circ, i, j, k);

  printf("nloops = " FMT_I " found.\n", nloops);
  for(i = 0; i < nloops; i++) {
    if(npoints[i] == 0) continue;
    sprintf(buf, "lines-" FMT_I ".dat", i);
    if(!(fp = fopen(buf, "w"))) {
      fprintf(stderr, "Can't open lines.dat\n");
      exit(1);
    }
    fprintf(fp, "# Loop " FMT_I "\n", i);
    for(j = 0; j < npoints[i]; j++)
      fprintf(fp, FMT_I " " FMT_I " " FMT_I "\n", xp[i][j], yp[i][j], zp[i][j]);
    fprintf(fp, "\n");
    fclose(fp);    
  }

  return 0;
}
