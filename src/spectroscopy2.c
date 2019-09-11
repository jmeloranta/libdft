/*
 * Spectroscopy related routines (Part 2): direct calculation of 1st order polarizability.
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
 * Routines for evaluating the dynamic lineshape (similar to CPL 396, 155 (2004) but
 * see intro of JCP 141, 014107 (2014) + references there in). The dynamics should be run on average
 * potential of gnd and excited states (returned by the init routine).
 *
 * 1) During the trajectory, call function:
 *     dft_spectrum_pol_collect() to record the time dependent difference
 *     energy (difference potential convoluted with the time dependent
 *     liquid density).
 *
 * 2) At the end, call the following function to evaluate the spectrum:
 *     dft_spectrum_pol_evaluate() to evaluate the lineshape.
 *
 */

/*
 * Collect the difference energy data. 
 *
 * gwf      = Current wavefunction (used for calculating the liquid density) (wf *, input).
 * diffpot  = Difference potential: Final - Initial state (rgrid *; input).
 * spectrum = Spectrum where the energy values are initially stored (cgrid *; input/output).
 * iter     = Current time step iteration (INT; input).
 * wrk      = Workspace (rgrid *; input).
 * 
 * No return value.
 *
 */

EXPORT void dft_spectrum_pol_collect(wf *gwf, rgrid *diffpot, cgrid *spectrum, INT iter, rgrid *wrk) {

  if(spectrum->nx != 1 || spectrum->nz != 1) {
    fprintf(stderr, "libdft: spectrum must be one dimensional grid (dft_spectrum_pol_collect).\n");
    exit(1);
  }
  if(iter >= spectrum->nz) {
    fprintf(stderr, "libdft: Spectrum allocated with too few points (dft_spectrum_pol_collect).\n");
    exit(1);
  }

  grid_wf_density(gwf, wrk);
  rgrid_product(wrk, wrk, diffpot);
  spectrum->value[iter] = rgrid_integral(wrk);

  fprintf(stderr, "libdft: spectrum collect complete (point = " FMT_I ", value = " FMT_R " K).\n", iter, CREAL(spectrum->value[iter]) * GRID_AUTOK);
}

/*
 * Evaluate the spectrum.
 *
 * spectrum = On entry: 1-D grid of potential difference values. On exit: spectrum (cgrid *; input/output).
 * tstep    = Time step length at which the energy difference data was collected
 *          (time stepin atomic units) (REAL, input).
 * tc       = Exponential decay time constant (atomic units; REAL, input).
 * wrk      = Workspace (cgrid *; input).
 *
 * Returns a pointer to the calculated spectrum (grid *). X-axis in cm$^{-1}$.
 *
 */

EXPORT cgrid *dft_spectrum_pol_evaluate(cgrid *spectrum, REAL tstep, REAL tc, cgrid *wrk) {

  INT t, tp;

  cgrid_copy(wrk, spectrum);

  cgrid_zero(spectrum);

  /* P(t) - full expression - see the Eloranta/Apkarian CPL paper on lineshapes */
  /* NOTE: Instead of propagating the liquid on the excited state, it is run on the average (V_e + V_g)/2 potential */
  /* The experssion is slightly different than in the papers but does the same */
  spectrum->value[0] = 1.0;
  fprintf(stderr, "libdft: Polarization at time 0 fs = 0.\n");
  for (t = 1; t < spectrum->nz; t++) {
    spectrum->value[t] = 0.0;
    for (tp = 0; tp < t; tp++)
      spectrum->value[t] += wrk->value[tp] * tstep;
    spectrum->value[t] = CEXP(-((REAL) t) * tstep / tc + I * spectrum->value[t]);
  }

  /* flip zero frequency to the middle */
  for (t = 0; t < spectrum->nz; t++)
    spectrum->value[t] *= POW(-1.0, (REAL) t);
  
  cgrid_fft(spectrum);

  spectrum->step = GRID_HZTOCM1 / (spectrum->step * GRID_AUTOFS * 1E-15 * (REAL) spectrum->nz);

  for(t = 0; t < spectrum->nz; t++)
    spectrum->value[t] = CABS(spectrum->value[t]) * CABS(spectrum->value[t]);
  
  return spectrum;
}
