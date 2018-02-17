/*
 * Analyze wavefunction files from grid files on disk.
 *
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <complex.h>
#include <grid/grid.h>
#include <grid/au.h>

#define HELIUM_MASS (4.002602 / GRID_AUTOAMU) /* helium mass */

#define NBINS 512
#define BIN_MAX 4.0
#define BIN_STEP (BIN_MAX / NBINS)
#define OUT_INTERVAL 1000

double bins[NBINS];
long nvals[NBINS];

#define NN 2.0

double A0, A1, A2, A3, A4, A5, RMIN, RADD;

/* Impurity must always be at the origin (dU/dx) */
double dpot_func(void *NA, double x, double y, double z) {

  double r, rp, r2, r3, r5, r7, r9, r11;

  rp = sqrt(x * x + y * y + z * z);
  r = rp - RADD;
  if(r < RMIN) return 0.0;

  r2 = r * r;
  r3 = r2 * r;
  r5 = r2 * r3;
  r7 = r5 * r2;
  r9 = r7 * r2;
  r11 = r9 * r2;
  
  return (x / rp) * (-A0 * A1 * exp(-A1 * r) + 4.0 * A2 / r5 + 6.0 * A3 / r7 + 8.0 * A4 / r9 + 10.0 * A5 / r11);
}

double pot_func(void *NA, double x, double y, double z) {

  double r, r2, r4, r6, r8, r10;

  r = sqrt(x * x + y * y + z * z);
  r -= RADD;
  if(r < RMIN) r = RMIN;

  r2 = r * r;
  r4 = r2 * r2;
  r6 = r4 * r2;
  r8 = r6 * r2;
  r10 = r8 * r2;
  return A0 * exp(-A1 * r) - A2 / r4 - A3 / r6 - A4 / r8 - A5 / r10;
}

void output_k(cgrid3d *grid) {

  double re, im, step = grid->step, kx, ky, kz, kk;
  long i, j, k, nx = grid->nx, ny = grid->ny, nz = grid->nz, ind, nynz = ny * nz;

  for (i = 0; i < nx; i++) {
    if (i < nx / 2) kx = 2.0 * M_PI * i / (nx * step);
    else kx = 2.0 * M_PI * (i - nx) / (nx * step);
    for (j = 0; j < ny; j++) {
      if (j < ny / 2) ky = 2.0 * M_PI * j / (ny * step);
      else ky = 2.0 * M_PI * (j - ny) / (ny * step);
      for (k = 0; k < nz; k++) {
        kz = 2.0 * M_PI * k / (nz * step);
        re = creal(grid->value[i * nynz + j * nz + k]);
        im = creal(grid->value[i * nynz + j * nz + k]);
        kk = sqrt(kx*kx + ky*ky + kz*kz);
        ind = (long) (kk / BIN_STEP);
        if(ind < NBINS) {
          bins[ind] += (re*re + im*im);
          nvals[ind] += 1;
        } else fprintf(stderr, "Warning: access outside bins.\n");
      }
    }
  }
}

int main(int argc, char *argv[]) {

  long nx, ny, nz, iter, iter_step;
  double step, time_step, vx, cur_mom = 0.0, prev_mom = 0.0;
  wf3d *wf;
  rgrid3d *cur_x, *cur_y, *cur_z, *circ;
  FILE *fp;
  char filename[1024];

  if(argc != 1) {
    fprintf(stderr, "Usage: analyze\n");
    exit(1);
  }
  if(!(fp = fopen("liquid.dat", "r"))) {
    fprintf(stderr, "Can't open file liquid.dat\n");
    exit(1);
  }
  fscanf(fp, " %le", &time_step); 
  printf("Time step = %le fs.\n", time_step);
  time_step /= GRID_AUTOFS;
  fscanf(fp, " %le", &vx); 
  printf("Velocity = %le m/s.\n", vx);
  vx /= GRID_AUTOMPS;
  fscanf(fp, " %ld", &iter_step);
  fscanf(fp, " %le", &A0);
  fscanf(fp, " %le", &A1);
  fscanf(fp, " %le", &A2);
  fscanf(fp, " %le", &A3);
  fscanf(fp, " %le", &A4);
  fscanf(fp, " %le", &A5);
  fscanf(fp, " %le", &RMIN);
  fscanf(fp, " %le", &RADD);
  fclose(fp);

  grid_threads_init(0);
  for(iter = 0; iter < NBINS; iter++) {
    bins[iter] = 0.0;
    nvals[iter] = 0;
  }

  for(iter = 0; ; iter += iter_step) {
    printf("Current time = %le fs.\n", ((double) iter) * time_step * GRID_AUTOFS);

    sprintf(filename, "liquid-%ld.grd", iter);
    if(!(fp = fopen(filename, "r"))) {
      fprintf(stderr, "Can't open file %s -- stopping.\n", filename);
      exit(0);
    }
    if(!iter) {
      grid3d_read_peek(fp, &nx, &ny, &nz, &step);
      if(!(wf = grid3d_wf_alloc(nx, ny, nz, step, HELIUM_MASS, WF3D_PERIODIC_BOUNDARY, WF3D_2ND_ORDER_PROPAGATOR))) {
        fprintf(stderr, "Not enough memory to allocate grid.\n");
        exit(1);
      }

      if(!(cur_x = rgrid3d_alloc(nx, ny, nz, step, RGRID3D_PERIODIC_BOUNDARY, NULL))) {
        fprintf(stderr, "Not enough memory to allocate grid.\n");
        exit(1);
      }
      if(!(cur_y = rgrid3d_alloc(nx, ny, nz, step, RGRID3D_PERIODIC_BOUNDARY, NULL))) {
        fprintf(stderr, "Not enough memory to allocate grid.\n");
        exit(1);
      }
      if(!(cur_z = rgrid3d_alloc(nx, ny, nz, step, RGRID3D_PERIODIC_BOUNDARY, NULL))) {
        fprintf(stderr, "Not enough memory to allocate grid.\n");
        exit(1);
      }
      if(!(circ = rgrid3d_alloc(nx, ny, nz, step, RGRID3D_PERIODIC_BOUNDARY, NULL))) {
        fprintf(stderr, "Not enough memory to allocate grid.\n");
        exit(1);
      }
    }             
    cgrid3d_read(wf->grid, fp);
    fclose(fp);

    grid3d_wf_probability_flux(wf, cur_x, cur_y, cur_z);
    cur_mom = rgrid3d_integral(cur_x) * wf->mass;
    
    if(vx > 0.0) printf("Added mass = %le He atoms.\n", cur_mom / (wf->mass * vx));
    grid3d_wf_density(wf, circ);
    printf("Number of He atoms = %le\n", rgrid3d_integral(circ));
    printf("Drag force1 = %le (au).\n", -rgrid3d_weighted_integral(circ, dpot_func, NULL)); // circ = density here
    printf("Drag force2 = %le (au).\n",  (cur_mom - prev_mom) / (time_step * ((double) iter_step)));
    prev_mom = cur_mom;

    rgrid3d_abs_rot(circ, cur_x, cur_y, cur_z);
    rgrid3d_power(circ, circ, NN);
    printf("Total circulation = %le (au; NN = %le).\n", rgrid3d_integral(circ), NN);
    fflush(stdout);    
#if 0
    if(!(iter % OUT_INTERVAL)) {
      long j;
      rgrid3d_fft(circ);
      sprintf(filename, "tmp/momentum-%ld", iter);
      if(!(fp = fopen(filename, "w"))) {
        fprintf(stderr, "Can't open momentum.dat.\n");
        exit(1);
      }
      output_k(circ->cint);
      for(j = 0; j < NBINS; j++)
        if(nvals[j]) fprintf(fp, "%le %le\n", BIN_STEP * (double) j, GRID_AUTOK * bins[j] / (double) nvals[j]); /* Y-axis scale?*/
        else fprintf(fp, "%le 0.0\n", BIN_STEP * (double) j);
      fclose(fp);
    }
#endif
  }
  exit(0); /* not reached */
}  

