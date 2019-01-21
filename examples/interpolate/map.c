/*
 * Given angular cuts of potentials, assemble the full cartesian grid using
 * interpolation.
 *
 * Takes files for angles 0, ..., Pi on the command line.
 *
 */

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <grid/grid.h>
#include <dft/dft.h>

void dft_common_pot_interpolate(INT n, char **files, rgrid *out) ;

#define NX 128
#define NY 128
#define NZ 128
#define STEP 0.5

int main(int argc, char **argv) {
  
  rgrid *cart;

  cart = rgrid_alloc(NX, NY, NZ, STEP, RGRID_PERIODIC_BOUNDARY, NULL, "cart");
  dft_common_pot_interpolate(argc-1, &argv[1], cart);
  rgrid_write_grid("out", cart);
  return 0;
}
