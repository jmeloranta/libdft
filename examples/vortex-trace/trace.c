/*
 * Trace vortex lines: produce vortex ring count and their lengths.
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

/* Vortex line recognition tuning parameters */
#define CUT 0.5         // start at 0.5 * max on grid
#define AVEPTS 4
#define DIST_CUT 1.8    // drop points within DIST_CUT * STEP (0.0 = disabled)
#define RINGCUT 20.0    // include points on ring with distance < RINGCUT * STEP
#define RINGCLOSE 20.0  // max distance between first and last points in ring, RINGCLOSE * STEP (0.0 = disabled)
#define PERIODIC 0      // 1 = use periodic BC for distance, 0 = use direct distance

#define MAXPTS 65535
#define MAXRINGS 25
#define MAXRING_PTS 256
REAL x[MAXPTS], y[MAXPTS], z[MAXPTS];
INT npts = 0;

REAL ring_x[MAXRINGS][MAXRING_PTS], ring_y[MAXRINGS][MAXRING_PTS], ring_z[MAXRINGS][MAXRING_PTS];
INT max_ring = 0, max_rings[MAXRINGS];

REAL STEP;
REAL LX, LY, LZ;
REAL dist(REAL x1, REAL y1, REAL z1, REAL x2, REAL y2, REAL z2) {

  REAL dx, dy, dz;

#if PERIODIC == 1
  REAL tmp;
  // Periodic boundary along X
  dx = (x1 - x2) * (x1 - x2);
  tmp = (x1 - (x2 + LX)) * (x1 - (x2 + LX));
  if(tmp < dx) dx = tmp;
  tmp = (x1 - (x2 - LX)) * (x1 - (x2 - LX));
  if(tmp < dx) dx = tmp;

  // Periodic boundary along Y
  dy = (y1 - y2) * (y1 - y2);
  tmp = (y1 - (y2 + LY)) * (y1 - (y2 + LY));
  if(tmp < dy) dy = tmp;
  tmp = (y1 - (y2 - LY)) * (y1 - (y2 - LY));
  if(tmp < dy) dy = tmp;

  // Periodic boundary along Z
  dz = (z1 - z2) * (z1 - z2);
  tmp = (z1 - (z2 + LZ)) * (z1 - (z2 + LZ));
  if(tmp < dz) dz = tmp;
  tmp = (z1 - (z2 - LZ)) * (z1 - (z2 - LZ));
  if(tmp < dz) dz = tmp;
#else
  dx = (x1 - x2) * (x1 - x2);
  dy = (y1 - y2) * (y1 - y2);
  dz = (z1 - z2) * (z1 - z2);
#endif

  return SQRT(dx + dy + dz);
}

void ring_close_check() {

  INT r;

  if(RINGCLOSE == 0.0) return;
  for (r = 0; r < max_ring; r++) 
    if(dist(ring_x[r][0], ring_y[r][0], ring_z[r][0], ring_x[r][max_rings[r]-1], ring_y[r][max_rings[r]-1], ring_z[r][max_rings[r]-1]) > STEP * RINGCLOSE)
      max_rings[r] = 0; // discard non-closed ring
}

void segment() {

  INT i, j, r, bj, rp = 0;
  REAL d, tmp;

  for(r = 0; r < MAXRINGS; r++) { // for all rings
    rp = 0;
    for (i = 0; i < npts; i++)  // find starting point
      if(x[i] != FP_NAN) break;
    if(i == npts) break;    // no starting points available? -> stop
    ring_x[r][rp] = x[i];   // add the point found as the first
    ring_y[r][rp] = y[i];   // point in the current ring
    ring_z[r][rp] = z[i];
    rp++;
    x[i] = y[i] = z[i] = FP_NAN;  // mark point as used
    while(1) {  // search for closest point to add
      d = 1E99;
      bj = -1;
      for (j = 0; j < npts; j++) {
        if(i == j || x[j] == FP_NAN) continue;
        tmp = dist(ring_x[r][rp-1], ring_y[r][rp-1], ring_z[r][rp-1], x[j], y[j], z[j]);
        if(tmp < d) {
          d = tmp;
          bj = j;
        }
      }
      if(d > RINGCUT * STEP) break; // ring complete? (no neighbors found)
      ring_x[r][rp] = x[bj];  // add new point to the ring
      ring_y[r][rp] = y[bj];
      ring_z[r][rp] = z[bj];
      x[bj] = y[bj] = z[bj] = FP_NAN;  // mark point as used
      rp++;
    }
    max_rings[r] = rp;
    max_ring++;
  }
}

void drop_points() {

  INT i, j;

  if(DIST_CUT == 0.0) return;
  for (i = 0; i < npts; i++)
    for (j = 0; j < npts; j++) {
      if(i == j) continue;
      if(x[i] == FP_NAN || x[j] == FP_NAN) continue;
      if(dist(x[i], y[i], z[i], x[j], y[j], z[j]) < DIST_CUT * STEP)
        x[j] = y[j] = z[j] = FP_NAN;
    }
}

void local_max(rgrid *circ, INT i, INT j, INT k) {

  INT ii, jj, kk;
  INT iim = 0, jjm = 0, kkm = 0;
  REAL max_val = -1.0, tmp;

  for (ii = i - AVEPTS; ii < i + AVEPTS; ii++) {
    for(jj = j - AVEPTS; jj < j + AVEPTS; jj++) {
      for(kk = k - AVEPTS; kk < k + AVEPTS; kk++) {
        tmp = rgrid_value_at_index(circ, ii, jj, kk);
        if(tmp > max_val) {
          max_val = tmp;
          iim = ii;
          jjm = jj;
          kkm = kk;
        }
      }
    }
  }
  x[npts] = ((REAL) (iim - circ->nx/2)) * circ->step;
  y[npts] = ((REAL) (jjm - circ->ny/2)) * circ->step;
  z[npts] = ((REAL) (kkm - circ->nz/2)) * circ->step;
  npts++;
  if(npts == MAXPTS) {
    printf("Too many points.\n");
    exit(1);
  }
}

int main(int argc, char **argv) {

  rgrid *circ, *cur_x, *cur_y, *cur_z;
  cgrid *tmp;
  wf *wf;
  REAL max_val, min_val, length;
  INT i, j, k;
  FILE *fp;

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

  LX = ((REAL) tmp->nx) * tmp->step;
  LY = ((REAL) tmp->ny) * tmp->step;
  LZ = ((REAL) tmp->nz) * tmp->step;
  STEP = tmp->step;

  /* Allocate grid functions */
  if(!(circ = rgrid_alloc(tmp->nx, tmp->ny, tmp->nz, tmp->step, RGRID_PERIODIC_BOUNDARY, NULL, "circ"))) {
    fprintf(stderr, "Cannot allocate grid.\n");
    exit(1);
  }
  cur_x = rgrid_clone(circ, "cur_x");
  cur_y = rgrid_clone(circ, "cur_y");
  cur_z = rgrid_clone(circ, "cur_z");

  cgrid_read_grid(wf->grid, argv[1]);

  grid_wf_probability_flux(wf, cur_x, cur_y, cur_z);

  rgrid_abs_rot(circ, cur_x, cur_y, cur_z);
  rgrid_abs_power(circ, circ, 1.0);

  min_val = rgrid_min(circ);
  max_val = rgrid_max(circ);
  fprintf(stderr, "Maximum = " FMT_R "\n", max_val);
  fprintf(stderr, "Minimum = " FMT_R "\n", min_val);
  if(max_val < 1E-7) max_val = 1E-7;

  for (i = 0; i < circ->nx; i++) {
    for (j = 0; j < circ->ny; j++) {
      for (k = 0; k < circ->nz; k++) {
        if(rgrid_value_at_index(circ, i, j, k) > CUT * max_val) local_max(circ, i, j, k);
      }
    }
  }

  drop_points();

  segment();
  
  ring_close_check();

  for(i = 0; i < max_ring; i++) { // loop over rings
    if(max_rings[i] == 0) continue;
//    printf("Ring #" FMT_I "\n", i+1);
    length = 0.0;
    for(j = 0; j < max_rings[i]; j++) {
      printf(FMT_R " " FMT_R " " FMT_R "\n", ring_x[i][j], ring_y[i][j], ring_z[i][j]);
      if(j) length += dist(ring_x[i][j-1], ring_y[i][j-1], ring_z[i][j-1], ring_x[i][j], ring_y[i][j], ring_z[i][j]);
    }
    printf("\n");
    fprintf(stderr, "Ring #" FMT_I " length = " FMT_R "\n", i+1, length);    
  }

  return 0;
}
