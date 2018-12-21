/*
 * 1-D OT potential (no kc or bf).
 *
 */

#include <math.h>
#include <grid/grid.h>
#include <grid/au.h>
#include <dft/dft.h>
#include <dft/ot.h>

/* Adaptive mapping of the aux functions */
// #define ADAPTIVE 

#define OT_H (2.1903 / GRID_AUTOANG)
#define OT_SIGMA (2.556 / GRID_AUTOANG)
#define OT_EPS (10.22 / GRID_AUTOK)
#define OT_C2 (-2.411857E4 / (GRID_AUTOK * GRID_AUTOANG * GRID_AUTOANG * GRID_AUTOANG * GRID_AUTOANG * GRID_AUTOANG * GRID_AUTOANG))
#define OT_C3 (1.858496E6 / (GRID_AUTOK * GRID_AUTOANG * GRID_AUTOANG * GRID_AUTOANG * GRID_AUTOANG * GRID_AUTOANG * GRID_AUTOANG * GRID_AUTOANG * GRID_AUTOANG * GRID_AUTOANG))

#define POWH6 (OT_H * OT_H * OT_H * OT_H * OT_H * OT_H)
#define POWH10 (OT_H * OT_H * OT_H * OT_H * OT_H * OT_H * OT_H * OT_H * OT_H * OT_H)
#define OT_SIGMA6 (OT_SIGMA * OT_SIGMA * OT_SIGMA * OT_SIGMA * OT_SIGMA * OT_SIGMA)

REAL T_LJ(void *na, REAL x, REAL y, REAL z) { /* (z, zp) */

  REAL zzp = FABS(z); /* z - zp */

  if(zzp < OT_H) return 2.0 * M_PI * OT_EPS * OT_SIGMA6 * (2.0 * OT_SIGMA6 - 5.0 * POWH6) / (5.0 * POWH10);
  return 2.0 * M_PI * OT_EPS * OT_SIGMA6 * (2.0 * OT_SIGMA6 - 5.0 * POW(zzp, 6.0)) / (5.0 * POW(zzp, 10.0));
}

REAL T_SP(void *na, REAL x, REAL y, REAL z) {

  REAL tmp = OT_H * OT_H - z * z;

  if(tmp < 0.0) return 0.0;
  else return (3.0 / (4.0 * OT_H * OT_H * OT_H)) * tmp;
}

void T_1(rgrid *output, rgrid *density_tf, rgrid *lj_tf) {

  dft_driver_convolution_eval(output, lj_tf, density_tf);
}

void rho_bar(rgrid *spave, rgrid *density_tf, rgrid *rd_tf) {

  dft_driver_convolution_eval(spave, density_tf, rd_tf);
}

void T_2(rgrid *output, rgrid *spave) {

  rgrid_ipower(output, spave, 2);
  rgrid_multiply(output, OT_C2 / 2.0);
}

void T_3(rgrid *output, rgrid *density, rgrid *spave, rgrid *rd_tf) {

  rgrid_product(output, density, spave);
  rgrid_fft(output);  

  dft_driver_convolution_eval(output, output, rd_tf);
  rgrid_multiply(output, OT_C2);
}

void T_4(rgrid *output, rgrid *spave) {

  rgrid_ipower(output, spave, 3);
  rgrid_multiply(output, OT_C3 / 3.0);
}

void T_5(rgrid *output, rgrid *density, rgrid *spave, rgrid *rd_tf) {

  rgrid_product(output, density, spave);
  rgrid_product(output, output, spave);
  rgrid_fft(output);

  dft_driver_convolution_eval(output, output, rd_tf);
  rgrid_multiply(output, OT_C3);
}

/**** User callable functions */

/* Grids that must be allocated:
 *
 * lj_tf
 * rd_tf
 * density, density_tf
 * spave, spave_tf
 * workspace
 *
 */
 
void OT_INIT(rgrid *lj_tf, rgrid *rd_tf) {

#ifndef ADAPTIVE
  rgrid_map(lj_tf, &T_LJ, NULL);
#else
  rgrid_adaptive_map(lj_tf, &T_LJ, NULL, 4, 32, 0.01 / GRID_AUTOK);
#endif
  rgrid_fft(lj_tf);

#ifndef ADAPTIVE
  rgrid_map(rd_tf, &T_SP, NULL);
#else
  rgrid_adaptive_map(rd_tf, &T_SP, NULL, 4, 32, 0.01 / GRID_AUTOK);
#endif
  rgrid_fft(rd_tf);
}

void OT_POT(rgrid *potential, rgrid *density, rgrid *density_tf, rgrid *spave, rgrid *spave_tf, rgrid *lj_tf, rgrid *rd_tf, rgrid *workspace) {

  rgrid_zero(potential);

  rgrid_copy(density_tf, density);
  rgrid_fft(density_tf);

  rho_bar(spave, density_tf, rd_tf);
  rgrid_copy(spave_tf, spave);
  rgrid_fft(spave_tf);

  T_1(workspace, density_tf, lj_tf);
  rgrid_sum(potential, potential, workspace);

  T_2(workspace, spave);
  rgrid_sum(potential, potential, workspace);

  T_3(workspace, density, spave, rd_tf);
  rgrid_sum(potential, potential, workspace);

  T_4(workspace, spave);
  rgrid_sum(potential, potential, workspace);

  T_5(workspace, density, spave, rd_tf);
  rgrid_sum(potential, potential, workspace);
}
