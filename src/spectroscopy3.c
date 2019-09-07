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

static rgrid *xxdiff = NULL, *xxbin = NULL;

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
 * otf        = Orsay-Trento functional pointer (dft_ot_functional *; input).
 * idensity   = NULL: no averaging of pair potentials or impurity density for convoluting with pair potential.
 *              Note that the impurity density is assumed to be time independent! (rgrid *; input)
 *              This is overwritten on exit.
 * nt         = Maximum number of time steps to be collected (INT; input).
 * binstep    = Bin step length (in cm$^{-1}$) (REAL; input).
 * zerofill   = How many zeros to fill in before FFT (INT; input).
 * finalave   = Averaging on the final state (see dft_common_potential_map()) (INT; input).
 * finalx     = Final potential file name along-x (char *, input).
 * finaly     = Final potential file name along-y (char *, input).
 * finalz     = Final potential file name along-z (char *, input).
 * initialave = Averaging on the initial state (see dft_common_potential_map()) (INT; input).
 * initialx   = Initial potential file name along-x (char *, input).
 * initialy   = Initial potential file name along-y (char *, input).
 * initialz   = Initial potential file name along-z (char *, input).
 *
 * Returns the difference potential (rgrid *).
 *
 */

EXPORT rgrid *dft_spectrumn_bin_init(dft_ot_functional *otf, rgrid *idensity, INT nt, REAL binstep, char finalave, char *finalx, char *finaly, char *finalz, char initialave, char *initialx, char *initialy, char *initialz) {

  rgrid *workspace1, *workspace2, *workspace3, *workspace4;

  if(!otf->workspace1) otf->workspace1 = rgrid_clone(otf->density, "OTF workspace 1");
  if(!otf->workspace2) otf->workspace2 = rgrid_clone(otf->density, "OTF workspace 2");
  if(!otf->workspace3) otf->workspace3 = rgrid_clone(otf->density, "OTF workspace 3");
  if(!otf->workspace4) otf->workspace4 = rgrid_clone(otf->density, "OTF workspace 4");

  workspace1 = otf->workspace1;
  workspace2 = otf->workspace2;
  workspace3 = otf->workspace3;
  workspace4 = otf->workspace4;

  dft_common_potential_map(finalave, finalx, finaly, finalz, workspace1);
  dft_common_potential_map(initialave, initialx, initialy, initialz, workspace2);

  if(idensity) {
    rgrid_fft(idensity);
    rgrid_fft(workspace1);
    rgrid_fft_convolute(workspace3, workspace1, idensity);
    rgrid_inverse_fft(workspace3);
    rgrid_fft(workspace2);
    rgrid_fft_convolute(workspace4, workspace2, idensity);
    rgrid_inverse_fft(workspace4);
  } else {
    rgrid_copy(workspace3, workspace1);
    rgrid_copy(workspace4, workspace2);
  }
  /* wrk3 = final state potential, wrk4 = initial state potential */

  if(!xxdiff)
    xxdiff = rgrid_alloc(workspace1->nx, workspace1->ny, workspace1->nz, workspace1->step, RGRID_PERIODIC_BOUNDARY, 0, "xxdiff");

  if(!xxbin) {
    xxbin = rgrid_alloc(1, 1, nt, binstep, RGRID_PERIODIC_BOUNDARY, 0, "xxbin");
    rgrid_host_lock(xxbin); // Must stay in host memory
  }
  rgrid_zero(xxbin);

  rgrid_difference(xxdiff, workspace3, workspace4);

  return xxdiff;
}

/*
 * Add point at a given time to the bin. The potential difference grid is taken from xxdiff above.
 *
 * otf  = Orsay-Trento functional pointer (dft_ot_functional *; input).
 * gwf  = Current wave function (wf *; input).
 * time = Current time in au (REAL; input).
 * tc   = Exponential decay constant for energy bin contributions (REAL; input).
 *
 * Returns pointer to the 1-D bin grid (rgrid *). This contains the spectrum.
 *
 */

EXPORT rgrid *dft_spectrum_bin_collect(dft_ot_functional *otf, wf *gwf, REAL time, REAL tc) {

  rgrid *workspace1;
  REAL energy;
  INT idx;

  if(!otf->workspace1) otf->workspace1 = rgrid_clone(otf->density, "OTF workspace 1");
  workspace1 = otf->workspace1;
  grid_wf_density(gwf, workspace1);
  rgrid_product(workspace1, workspace1, xxdiff);
  energy = rgrid_integral(workspace1) * GRID_AUTOCM1;
  fprintf(stderr, "libdft: bin collect with energy = " FMT_R " cm-1.\n", energy);
  idx = ((INT) (energy / xxbin->step)) + xxbin->nz / 2; 
  if(idx < 0 || idx >= xxbin->nz) return xxbin;
  xxbin->value[idx] += EXP(-time / tc);
  return xxbin;
}
