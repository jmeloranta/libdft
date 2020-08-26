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
 * @FUNC{dft_spectrum_pol_collect, "Dynamic lineshape: Collect the data"}
 * @DESC{"Collect the difference energy data. Routines for evaluating the dynamic lineshape (similar to CPL 396, 155 (2004) but
          see intro of JCP 141, 014107 (2014) + references there in). The dynamics should be run on average
          potential of gnd and excited states (returned by the init routine).\\
          1) During the trajectory, call function:\\
          dft_spectrum_pol_collect() to record the time dependent difference
          energy (difference potential convoluted with the time dependent
          liquid density).\\
          2) At the end, call the following function to evaluate the spectrum:\\
          dft_spectrum_pol_evaluate() to evaluate the lineshape"}
 * @ARG1{wf *gwf, "Current wavefunction (used for calculating the liquid density) (input)"}
 * @ARG2{rgrid *diffpot, "Difference potential: Final - Initial state (input)"}
 * @ARG3{cgrid *spectrum, "Spectrum where the energy values are initially stored"}
 * @ARG4{INT iter, "Current time step iteration (input)"}
 * @ARG5{rgrid *wrk, "Workspace"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT void dft_spectrum_pol_collect(wf *gwf, rgrid *diffpot, cgrid *spectrum, INT iter, rgrid *wrk) {

#ifdef GRID_MGPU
  cgrid_host_lock(spectrum);
#endif
  if(spectrum->nx != 1 || spectrum->ny != 1) {
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
 * @FUNC{dft_spectrum_pol_evaluate, "Dynamic lineshape: Evaluate spectrum"}
 * @DESC{Evaluate the spectrum. See dft_spectrum_pol_collect()"}
 * @ARG1{cgrid *spectrum, "On entry: 1-D grid of potential difference values. On exit: evaluated spectrum"}
 * @ARG2{REAL tstep, "Time step length at which the energy difference data was collected (time stepin atomic units)"}
 * @ARG3{REAL tc, "Exponential decay time constant (atomic units; input)"}
 * @ARG4{cgrid *wrk, "Workspace"}
 * @RVAL{cgrid *, "Returns a pointer to the calculated spectrum. X-axis in cm$^{-1}$"}
 *
 */

EXPORT cgrid *dft_spectrum_pol_evaluate(cgrid *spectrum, REAL tstep, REAL tc, cgrid *wrk) {

  INT t, tp;

#ifdef GRID_MGPU
  cgrid_host_lock(spectrum);
#endif

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
  
#ifdef GRID_MGPU
  cgrid_host_unlock(spectrum);
#endif

  return spectrum;
}
