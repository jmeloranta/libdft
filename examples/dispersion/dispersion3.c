/*
 * Analytical (OT) dispersion relation.
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
  REAL k, w;
  
  dft_ot_temperature(&otf, DFT_OT_PLAIN | DFT_OT_KC | DFT_OT_BACKFLOW);
//  dft_ot_temperature(&otf, DFT_OT_PLAIN | DFT_OT_BACKFLOW);

  for (k = 0.0; k < 1.5; k += 0.02) {
    REAL kk;
    kk = k;
    w = dft_ot_bulk_dispersion(&otf, &kk, RHO0); // kk overwritten with the point of evaluation
    if(k == 0.0) {
      printf("# Dispersion relation for functional " FMT_I " (Angs^-1 and K).\n", otf.model); // avoid overlapping print with init txts
      printf("# Applied P = " FMT_R " MPa.\n", dft_ot_bulk_pressure(&otf, RHO0) * GRID_AUTOPA / 1E6);
    }
    printf(FMT_R " " FMT_R "\n", kk / GRID_AUTOANG, w * GRID_AUTOK);
    fflush(stdout);
  }
  return 0;
}
