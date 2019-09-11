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
 * Collect data during real time dynamics using
 * dft_spectrum_bin_collect(). Finally print the spectrum
 * using the grid pointer returned by dft_spectrum_bin_collect().
 * Note that the energy axis is in wavenumbers!
 *
 */

/*
 * Add point at a given time to the bin. The potential difference grid is taken from xxdiff above.
 *
 * gwf     = Current wave function (wf *; input).
 * diffpot = Difference potential (rgrid *; input).
 * bin     = Bin grid (spectrum) (cgrid *; input). This must be 1-D grid.
 * iter    = Current time iteration (INT; input).
 * tstep   = Time step length (REAL; input).
 * tc      = Exponential decay constant for energy bin contributions (REAL; input).
 * wrk     = Workspace (rgrid *; input).
 *
 * No return value.
 *
 */

EXPORT void dft_spectrum_bin_collect(wf *gwf, rgrid *diffpot, rgrid *bin, INT iter, REAL tstep, REAL tc, rgrid *wrk) {

  REAL energy;
  INT idx;

  if(bin->nx != 1 || bin->ny != 1) {
    fprintf(stderr, "libdft: Bin grid must be one dimensional (dft_spectrum_bin_collect).\n");
    exit(1);
  }

  grid_wf_density(gwf, wrk);
  rgrid_product(wrk, wrk, diffpot);
  energy = rgrid_integral(wrk) * GRID_AUTOCM1;
  fprintf(stderr, "libdft: bin collect with energy = " FMT_R " cm-1.\n", energy);
  idx = ((INT) (energy / bin->step)) + bin->nz / 2; 
  if(idx < 0 || idx >= bin->nz) return;
  bin->value[idx] += EXP(-((REAL) iter) * tstep / tc);

  return;
}
