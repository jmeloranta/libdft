/*
 * Analytical (OT) static response function (-1/X).
 *
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <complex.h>

#include <grid/grid.h>
#include <grid/au.h>
#include <dft/dft.h>
#include <dft/ot.h>

#define RHO0 (0.0218360 * GRID_AUTOANG * GRID_AUTOANG * GRID_AUTOANG)

int main(int argc, char **argv) {

  dft_ot_functional otf;
  double k, w;
  
  dft_ot_temperature(&otf, DFT_OT_PLAIN | DFT_OT_KC);

  for (k = 0.0; k < 1.5; k += 0.02) {
    double kk;
    kk = k;
    w = dft_ot_bulk_istatic(&otf, &kk, RHO0);
    printf("%le %le\n", kk / GRID_AUTOANG,  1.0 / (GRID_AUTOK * w));
    fflush(stdout);
  }
  return 0;
}
