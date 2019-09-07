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
 * 1) Initialize the difference potential:
 *     dft_spectrum_pol_init().
 * 
 * 2) During the trajectory, call function:
 *     dft_spectrum_pol_collect() to record the time dependent difference
 *     energy (difference potential convoluted with the time dependent
 *     liquid density).
 *
 * 3) At the end, call the following function to evaluate the spectrum:
 *     dft_spectrum_pol_evaluate() to evaluate the lineshape.
 *
 */

/*
 * Collect the time dependent difference energy data.
 * 
 * otf        = Orsay-Trento functional pointer (dft_ot_functional *; input).
 * idensity   = NULL: no averaging of pair potentials or impurity density for convoluting with pair potential.
 *              Note that the impurity density is assumed to be time independent! (rgrid *; input)
 *              This is overwritten on exit.
 * nt         = Maximum number of time steps to be collected (INT; input).
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
 * Returns average potential for dynamics.
 *
 */

static rgrid *xxdiff = NULL, *xxave = NULL;
static cgrid *tdpot = NULL;
static INT ntime, cur_time, zerofill;

EXPORT rgrid *dft_spectrum_pol_init(dft_ot_functional *otf, rgrid *idensity, INT nt, INT zf, char finalave, char *finalx, char *finaly, char *finalz, char initialave, char *initialx, char *initialy, char *initialz) {
 
  rgrid *workspace1, *workspace2, *workspace3, *workspace4;
  INT nx, ny, nz;
  REAL step;

  if(!otf->workspace1) otf->workspace1 = rgrid_clone(otf->density, "OTF workspace 1");
  if(!otf->workspace2) otf->workspace2 = rgrid_clone(otf->density, "OTF workspace 2");
  if(!otf->workspace3) otf->workspace3 = rgrid_clone(otf->density, "OTF workspace 3");
  if(!otf->workspace4) otf->workspace4 = rgrid_clone(otf->density, "OTF workspace 4");

  workspace1 = otf->workspace1;
  workspace2 = otf->workspace2;
  workspace3 = otf->workspace3;
  workspace4 = otf->workspace4;
  nx = workspace1->nx;
  ny = workspace1->ny;
  nz = workspace1->nz;
  step = workspace1->step;

  cur_time = 0;  /* watch out, these are static global variables */
  ntime = nt;
  zerofill = zf;

  if(!tdpot) {
    tdpot = cgrid_alloc(1, 1, ntime + zf, 0.1, CGRID_PERIODIC_BOUNDARY, 0, "tdpot");
    cgrid_host_lock(tdpot); // Must stay in host memory
  }

  if(!xxdiff)
    xxdiff = rgrid_alloc(nx, ny, nz, step, RGRID_PERIODIC_BOUNDARY, 0, "xxdiff");
  if(!xxave)
    xxave = rgrid_alloc(nx, ny, nz, step, RGRID_PERIODIC_BOUNDARY, 0, "xxave");

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

  rgrid_difference(xxdiff, workspace3, workspace4);
  rgrid_sum(xxave, workspace3, workspace4);
  rgrid_multiply(xxave, 0.5);

  return xxave;
}

/*
 * Collect the time dependent difference energy data. Same as above but with direct
 * grid input for potentials.
 * 
 * otf      = Orsay-Trento functional pointer (dft_ot_functional *; input).
 * nt       = Maximum number of time steps to be collected (INT, input).
 * zerofill = How many zeros to fill in before FFT (INT, input).
 * final    = Final state potential grid (rgrid *, input).
 * initial    = Initial state potential grid (rgrid *, input).
 *
 * Returns average potential for dynamics.
 *
 */

EXPORT rgrid *dft_spectrum_pol_init2(dft_ot_functional *otf, INT nt, INT zf, rgrid *upper, rgrid *lower) {

  INT nx, ny, nz;
  REAL step;

  nx = otf->density->nx;
  ny = otf->density->ny;
  nz = otf->density->nz;
  step = otf->density->step;

  cur_time = 0;
  ntime = nt;
  zerofill = zf;

  if(!tdpot) {
    tdpot = cgrid_alloc(1, 1, ntime + zf, 0.1, CGRID_PERIODIC_BOUNDARY, 0, "tdpot");
    cgrid_host_lock(tdpot); // Must stay in host memory
  }

  if(!xxdiff)
    xxdiff = rgrid_alloc(nx, ny, nz, step, RGRID_PERIODIC_BOUNDARY, 0, "xxdiff");
  if(!xxave)
    xxave = rgrid_alloc(nx, ny, nz, step, RGRID_PERIODIC_BOUNDARY, 0, "xxave");

  rgrid_difference(xxdiff, upper, lower);
  rgrid_sum(xxave, upper, lower);
  rgrid_multiply(xxave, 0.5);

  return xxave;
}

/*
 * Collect the difference energy data (user specified).
 *
 * val = difference energy value to be inserted (input, REAL).
 *
 */

EXPORT void dft_spectrum_pol_collect_user(REAL val) {

  if(cur_time > ntime) {
    fprintf(stderr, "libdft: initialized with too few points (spectrum collect).\n");
    exit(1);
  }
  tdpot->value[cur_time] = val;

  fprintf(stderr, "libdft: spectrum collect complete (point = " FMT_I ", value = " FMT_R " K).\n", cur_time, CREAL(tdpot->value[cur_time]) * GRID_AUTOK);
  cur_time++;
}

/*
 * Collect the difference energy data. 
 *
 * otf     = Functional pointer (dft_ot_functional *).
 * gwf     = the current wavefunction (used for calculating the liquid density) (wf *, input).
 *
 */

EXPORT void dft_spectrum_pol_collect(dft_ot_functional *otf, wf *gwf) {

  rgrid *workspace1;

  if(!otf->workspace1) otf->workspace1 = rgrid_clone(otf->density, "OTF workspace 1");
  workspace1 = otf->workspace1;

  if(cur_time > ntime) {
    fprintf(stderr, "libdft: initialized with too few points (spectrum collect).\n");
    exit(1);
  }
  grid_wf_density(gwf, workspace1);
  rgrid_product(workspace1, workspace1, xxdiff);
  tdpot->value[cur_time] = rgrid_integral(workspace1);

  fprintf(stderr, "libdft: spectrum collect complete (point = " FMT_I ", value = " FMT_R " K).\n", cur_time, CREAL(tdpot->value[cur_time]) * GRID_AUTOK);
  cur_time++;
}

/*
 * Evaluate the spectrum (full expression).
 *
 * tstep  = Time step length at which the energy difference data was collected
 *          (time stepin atomic units) (REAL, input).
 * tc     = Exponential decay time constant (atomic units; REAL, input).
 *
 * Returns a pointer to the calculated spectrum (grid *). X-axis in cm$^{-1}$.
 *
 */

EXPORT cgrid *dft_spectrum_pol_evaluate(REAL tstep, REAL tc) {

  INT t, npts;
  static cgrid *spectrum = NULL;

  if(cur_time > ntime) {
    printf(FMT_I " " FMT_I "\n", cur_time, ntime);
    fprintf(stderr, "libdft: cur_time >= ntime. Increase ntime.\n");
    exit(1);
  }

  npts = cur_time + zerofill;

  if(!spectrum)
    spectrum = cgrid_alloc(1, 1, npts, GRID_HZTOCM1 / (tstep * GRID_AUTOFS * 1E-15 * ((REAL) npts)), CGRID_PERIODIC_BOUNDARY, 0, "spectrum");

  cgrid_zero(spectrum);

  /* P(t) - full expression - see the Eloranta/Apkarian CPL paper on lineshapes */
  /* NOTE: Instead of propagating the liquid on the excited state, it is run on the average (V_e + V_g)/2 potential */
  spectrum->value[0] = 0.0;
  fprintf(stderr, "libdft: Polarization at time 0 fs = 0.\n");
  for (t = 1; t < cur_time; t++) {
    REAL tmp;
    REAL complex tmp2;
    INT tp, tpp;
    tmp2 = 0.0;
    for(tp = 0; tp < t; tp++) {
      tmp = 0.0;
      for(tpp = tp; tpp < t; tpp++)
	tmp += CREAL(tdpot->value[tpp]) * tstep;
      tmp2 += CEXP(-I * tmp) * tstep;
    }
    spectrum->value[t] = -2.0 * CIMAG(tmp2) * EXP(-((REAL) t) * tstep / tc);
    fprintf(stderr, "libdft: Polarization at time " FMT_R " fs = %le.\n", ((REAL) t) * tstep * GRID_AUTOFS, CREAL(spectrum->value[t]));
  }
  
  /* zero fill */
  for (t = cur_time; t < npts; t++)
    spectrum->value[t] = 0.0;

  /* flip zero frequency to the middle */
  for (t = 0; t < npts; t++)
    spectrum->value[t] *= POW(-1.0, (REAL) t);
  
  cgrid_inverse_fft(spectrum);

  for(t = 0; t < npts; t++)
    spectrum->value[t] = CIMAG(spectrum->value[t]);
  
  return spectrum;
}
