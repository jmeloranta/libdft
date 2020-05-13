/*
 * Experimental dispersion relation at SVP (Angs^-1 and Kelvin).
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

int main(int argc, char **argv) {

  REAL k;
  
  for (k = 0.0; k < 1.5; k += 0.02)
    printf(FMT_R " " FMT_R "\n", k / GRID_AUTOANG, dft_exp_bulk_dispersion(k / GRID_AUTOANG));

  return 0;
}
