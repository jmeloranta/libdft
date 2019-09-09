/*
 * Spectroscopy related routines (Part 3): Simple energy difference binning between the
 * initial and final states during dynamics.
 *
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <grid/grid.h>
#include <grid/au.h>
#include "dft.h"
#include "ot.h"

/*
 * Call dft_spectrum_bin_init() first to set things up.
 * Then collect data during real time dynamics using
 * dft_spectrum_bin_collect(). Finally print the spectrum
 * using the grid pointer returned by dft_spectrum_bin_collect().
 * Note that the energy axis is in wavenumbers!
 *
 */

/*
 * Initialize binning-based evaluation of spectrum.
 *
 * idensity   = NULL: no averaging of pair potentials or impurity density for convoluting with pair potential.
 *              Note that the impurity density is assumed to be time independent! (rgrid *; input)
 *              This is overwritten on exit.
 * diffpot    = Difference potential grid (rgird *; input). If imdensity != NULL, this will get convoluted with imdensity on exit.
 * bin        = 1-D bin grid (spectrum) (rgrid *; input). This must be 1-D grid.
 *
 * No return value.
 *
 */

EXPORT void dft_spectrum_bin_init(rgrid *idensity, rgrid *diffpot, rgrid *bin) {

  if(idensity) {
    rgrid_fft(idensity);
    rgrid_fft(diffpot);
    rgrid_fft_convolute(diffpot, diffpot, idensity);
    rgrid_inverse_fft(diffpot);
  }

  if(bin->nx != 1 || bin->ny != 1) {
    fprintf(stderr, "libdft: Bin grid must be one dimensional.\n");
    exit(1);
  }
  rgrid_host_lock(bin); // Must stay in host memory
  rgrid_zero(bin);
}

/*
 * Add point at a given time to the bin. The potential difference grid is taken from xxdiff above.
 *
 * gwf     = Current wave function (wf *; input).
 * density = Liquid density (rgrid *; input). Will be overwritten on exit!
 * diffpot = Difference potential (rgrid *; input).
 * bin     = Bin grid (spectrum) (rgrid *; input). This must be 1-D grid.
 * time    = Current time in au (REAL; input).
 * tc      = Exponential decay constant for energy bin contributions (REAL; input).
 *
 * No return value.
 *
 */

EXPORT void dft_spectrum_bin_collect(wf *gwf, rgrid *density, rgrid *diffpot, rgrid *bin, REAL time, REAL tc) {

  REAL energy;
  INT idx;

  rgrid_product(density, density, diffpot);
  energy = rgrid_integral(density) * GRID_AUTOCM1;
  fprintf(stderr, "libdft: bin collect with energy = " FMT_R " cm-1.\n", energy);
  idx = ((INT) (energy / bin->step)) + bin->nz / 2; 
  if(idx < 0 || idx >= bin->nz) return;
  bin->value[idx] += EXP(-time / tc);

  return;
}
