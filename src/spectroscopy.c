/*
 * Spectroscopy related routines.
 *
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <grid/grid.h>
#include <grid/au.h>
#include "dft.h"
#include "ot.h"

extern REAL dft_driver_step;
extern INT dft_driver_nx, dft_driver_ny, dft_driver_nz;

/*
 * Evaluate absorption/emission spectrum using the Andersson
 * expression. No zero-point correction for the impurity.
 *
 * density  = Current liquid density (rgrid *; input).
 * tstep    = Time step for constructing the time correlation function
 *            (REAL; input in fs). Typically around 1 fs.
 * endtime  = End time in constructing the time correlation function
 *            (REAL; input in fs). Typically less than 10,000 fs.
 * finalave = Averaging of the final state potential.
 *            0: no averaging, 1 = average XY, 2 = average YZ, 3 = average XZ,
 *            4 = average XYZ.
 * finalx   = Final state potential along the X axis (char *; input).
 * finaly   = Final state potential along the Y axis (char *; input).
 * finalz   = Final state potential along the Z axis (char *; input).
 * initialave = Averaging of the initial state potential.
 *            0: no averaging, 1 = average XY, 2 = average YZ, 3 = average XZ,
 *            4 = average XYZ.
 * initialx   = Initial state potential along the X axis (char *; input).
 * initialy   = Initial state potential along the Y axis (char *; input).
 * initialz   = Initial state potential along the Z axis (char *; input).
 *
 * NOTE: This needs complex grid input!! 
 *
 * Returns the spectrum (cgrid *; output). Note that this is statically
 * allocated and overwritten by a subsequent call to this routine.
 * The spectrum will start from smaller to larger frequencies.
 * The spacing between the points is included in cm-1.
 *
 */

static REAL complex dft_eval_exp(REAL complex a, void *NA) { /* a contains t */

  return (1.0 - CEXP(-I * a));
}

static REAL complex dft_do_int(rgrid *dens, rgrid *dpot, REAL t, cgrid *wrk) {

  grid_real_to_complex_re(wrk, dpot);
  cgrid_multiply(wrk, t);
  cgrid_operate_one(wrk, wrk, dft_eval_exp, NULL);
  grid_product_complex_with_real(wrk, dens);
  return cgrid_integral(wrk);            // debug: This should have minus in front?! Sign error elsewhere? (does not appear in ZP?!)
}

EXPORT cgrid *dft_driver_spectrum(rgrid *density, REAL tstep, REAL endtime, char finalave, char *finalx, char *finaly, char *finalz, char initialave, char *initialx, char *initialy, char *initialz) {

  rgrid *dpot, *workspace1 = dft_driver_otf->workspace1, *workspace2 = dft_driver_otf->workspace2;
  cgrid *wrk[256];
  static cgrid *corr = NULL;
  REAL t;
  INT i, ntime;
  static INT prev_ntime = -1;

  endtime /= GRID_AUTOFS;
  tstep /= GRID_AUTOFS;
  ntime = (1 + (int) (endtime / tstep));
  dpot = rgrid_alloc(density->nx, density->ny, density->nz, density->step, RGRID_PERIODIC_BOUNDARY, 0, "DR spectrum dpot");
  // TODO: FIXME - this may allocate lot of memory!
  for (i = 0; i < omp_get_max_threads(); i++)
    wrk[i] = cgrid_alloc(density->nx, density->ny, density->nz, density->step, CGRID_PERIODIC_BOUNDARY, 0, "DR spectrum wrk");
  if(ntime != prev_ntime) {
    if(corr) cgrid_free(corr);
    corr = cgrid_alloc(1, 1, ntime, 0.1, CGRID_PERIODIC_BOUNDARY, 0, "correlation function");
    prev_ntime = ntime;
  }

  rgrid_claim(workspace1);
  rgrid_claim(workspace2);
  dft_common_potential_map(finalave, finalx, finaly, finalz, workspace1);
  dft_common_potential_map(initialave, initialx, initialy, initialz, workspace2);
  rgrid_difference(dpot, workspace1, workspace2); /* final - initial */
  
  rgrid_product(workspace1, dpot, density);
  fprintf(stderr, "libdft: Average shift = " FMT_R " cm-1.\n", rgrid_integral(workspace1) * GRID_AUTOCM1);
  rgrid_release(workspace1);
  rgrid_release(workspace2);

#pragma omp parallel for firstprivate(tstep,ntime,density,dpot,corr,wrk) private(i,t) default(none) schedule(runtime)
  for(i = 0; i < ntime; i++) {
    t = tstep * (REAL) i;
    corr->value[i] = CEXP(dft_do_int(density, dpot, t, wrk[omp_get_thread_num()])) * POW(-1.0, (REAL) i);
  }
  cgrid_fft(corr);
  for (i = 0; i < corr->nx; i++)
    corr->value[i] = CABS(corr->value[i]);
  
  corr->step = GRID_HZTOCM1 / (tstep * GRID_AUTOFS * 1E-15 * (REAL) ntime);

  rgrid_free(dpot);
  for(i = 0; i < omp_get_max_threads(); i++)
    cgrid_free(wrk[i]);

  return corr;
}

/*
 *
 * TODO: This still needs to be modified so that it takes rgrid
 * for density and imdensity.
 *
 * Evaluate absorption/emission spectrum using the Andersson
 * expression. Zero-point correction for the impurity included.
 *
 * density  = Current liquid density (rgrid *; input).
 * imdensity= Current impurity zero-point density (cgrid *; input).
 * tstep    = Time step for constructing the time correlation function
 *            (REAL; input in fs). Typically around 1 fs.
 * endtime  = End time in constructing the time correlation function
 *            (REAL; input in fs). Typically less than 10,000 fs.
 * upperave = Averaging of the upperial state potential.
 *            0: no averaging, 1 = average XY, 2 = average YZ, 3 = average XZ,
 *            4 = average XYZ.
 * upperx   = Upper state potential along the X axis (char *; input).
 * uppery   = Upper state potential along the Y axis (char *; input).
 * upperz   = Upper state potential along the Z axis (char *; input).
 * lowerave = Averaging of the lower state potential.
 *            0: no averaging, 1 = average XY, 2 = average YZ, 3 = average XZ,
 *            4 = average XYZ.
 * lowerx   = Lower state potential along the X axis (char *; input).
 * lowery   = Lower state potential along the Y axis (char *; input).
 * lowerz   = Lower state potential along the Z axis (char *; input).
 *
 * NOTE: This needs complex grid input!! 
 *
 * Returns the spectrum (cgrid *; output). Note that this is statically
 * allocated and overwritten by a subsequent call to this routine.
 * The spectrum will start from smaller to larger frequencies.
 * The spacing between the points is included in cm-1.
 *
 */

static void do_gexp(cgrid *gexp, rgrid *dpot, REAL t) {

  grid_real_to_complex_re(gexp, dpot);
  cgrid_multiply(gexp, t);
  cgrid_operate_one(gexp, gexp, dft_eval_exp, NULL);
  cgrid_fft(gexp);  
#if 0
  cgrid_zero(gexp);
  cgrid_add_scaled(gexp, t, dpot);
  cgrid_operate_one(gexp, gexp, dft_eval_exp, NULL);
  cgrid_fft(gexp);
#endif
}

static REAL complex dft_do_int2(cgrid *gexp, rgrid *imdens, cgrid *fft_dens, REAL t, cgrid *wrk) {

  cgrid_fft_convolute(wrk, fft_dens, gexp);
  cgrid_inverse_fft(wrk);
  grid_product_complex_with_real(wrk, imdens);

  return -cgrid_integral(wrk);
#if 0
  cgrid_zero(wrk);
  cgrid_fft_convolute(wrk, dens, gexp);
  cgrid_inverse_fft(wrk);
  cgrid_product(wrk, wrk, imdens);

  return -cgrid_integral(wrk);
#endif
}

EXPORT cgrid *dft_driver_spectrum_zp(rgrid *density, rgrid *imdensity, REAL tstep, REAL endtime, char upperave, char *upperx, char *uppery, char *upperz, char lowerave, char *lowerx, char *lowery, char *lowerz) {

  cgrid *wrk, *fft_density, *gexp;
  rgrid *dpot, *workspace1 = dft_driver_otf->workspace1, *workspace2 = dft_driver_otf->workspace2;
  static cgrid *corr = NULL;
  REAL t;
  INT i, ntime;
  static INT prev_ntime = -1;

  endtime /= GRID_AUTOFS;
  tstep /= GRID_AUTOFS;
  ntime = (1 + (int) (endtime / tstep));
  dpot = rgrid_alloc(density->nx, density->ny, density->nz, density->step, RGRID_PERIODIC_BOUNDARY, 0, "DR spectrum_zp dpot");
  fft_density = cgrid_alloc(density->nx, density->ny, density->nz, density->step, CGRID_PERIODIC_BOUNDARY, 0, "DR spectrum_zp fftd");
  wrk = cgrid_alloc(density->nx, density->ny, density->nz, density->step, CGRID_PERIODIC_BOUNDARY, 0, "DR spectrum_zp wrk");
  gexp = cgrid_alloc(density->nx, density->ny, density->nz, density->step, CGRID_PERIODIC_BOUNDARY, 0, "DR spectrum_zp gexp");
  if(ntime != prev_ntime) {
    if(corr) cgrid_free(corr);
    corr = cgrid_alloc(1, 1, ntime, 0.1, CGRID_PERIODIC_BOUNDARY, 0, "correlation function");
    prev_ntime = ntime;
  }
  
  rgrid_claim(workspace1);
  rgrid_claim(workspace2);
  fprintf(stderr, "libdft: Upper level potential.\n");
  dft_common_potential_map(upperave, upperx, uppery, upperz, workspace1);
  fprintf(stderr, "libdft: Lower level potential.\n");
  dft_common_potential_map(lowerave, lowerx, lowery, lowerz, workspace2);
  rgrid_difference(dpot, workspace2, workspace1);
  rgrid_release(workspace1);
  rgrid_release(workspace2);
  
  grid_real_to_complex_re(fft_density, density);
  cgrid_fft(fft_density);
  
  // can't run in parallel - actually no much sense since the most time intensive
  // part is the fft (which runs in parallel)
  for(i = 0; i < ntime; i++) {
    t = tstep * (REAL) i;
    do_gexp(gexp, dpot, t); /* gexp grid + FFT */
    corr->value[i] = CEXP(dft_do_int2(gexp, imdensity, fft_density, t, wrk)) * POW(-1.0, (REAL) i);
    fprintf(stderr,"libdft: Corr(" FMT_R " fs) = " FMT_R " " FMT_R "\n", t * GRID_AUTOFS, CREAL(corr->value[i]), CIMAG(corr->value[i]));
  }
  cgrid_fft(corr);
  for (i = 0; i < corr->nx; i++)
    corr->value[i] = CABS(corr->value[i]);
  
  corr->step = GRID_HZTOCM1 / (tstep * GRID_AUTOFS * 1E-15 * (REAL) ntime);
  
  rgrid_free(dpot);
  cgrid_free(fft_density);
  cgrid_free(wrk);
  cgrid_free(gexp);
  
  return corr;
}

/*
 * Routines for evaluating the dynamic lineshape (similar to CPL 396, 155 (2004) but
 * see intro of JCP 141, 014107 (2014) + references there in). The dynamics should be run on average
 * potential of gnd and excited states (returned by the init routine).
 *
 * 1) Initialize the difference potential:
 *     dft_driver_spectrum_init().
 * 
 * 2) During the trajectory, call function:
 *     dft_driver_spectrum_collect() to record the time dependent difference
 *     energy (difference potential convoluted with the time dependent
 *     liquid density).
 *
 * 3) At the end, call the following function to evaluate the spectrum:
 *     dft_driver_spectrum_evaluate() to evaluate the lineshape.
 *
 */

/*
 * Collect the time dependent difference energy data.
 * 
 * idensity = NULL: no averaging of pair potentials, rgrid *: impurity density for convoluting with pair potential. (input)
 * nt       = Maximum number of time steps to be collected (INT, input).
 * zerofill = How many zeros to fill in before FFT (int, input).
 * upperave = Averaging on the upper state (see dft_driver_potential_map()) (int, input).
 * upperx   = Upper potential file name along-x (char *, input).
 * uppery   = Upper potential file name along-y (char *, input).
 * upperz   = Upper potential file name along-z (char *, input).
 * lowerave = Averaging on the lower state (see dft_driver_potential_map()) (int, input).
 * lowerx   = Lower potential file name along-x (char *, input).
 * lowery   = Lower potential file name along-y (char *, input).
 * lowerz   = Lower potential file name along-z (char *, input).
 *
 * Returns difference potential for dynamics.
 *
 */

static rgrid *xxdiff = NULL, *xxave = NULL;
static cgrid *tdpot = NULL;
static INT ntime, cur_time, zerofill;

EXPORT rgrid *dft_driver_spectrum_init(rgrid *idensity, INT nt, INT zf, char upperave, char *upperx, char *uppery, char *upperz, char lowerave, char *lowerx, char *lowery, char *lowerz) {
 
  rgrid *workspace1 = dft_driver_otf->workspace1, *workspace2 = dft_driver_otf->workspace2, *workspace3 = dft_driver_otf->workspace3;
  rgrid *workspace4 = dft_driver_otf->workspace4;

  cur_time = 0;
  ntime = nt;
  zerofill = zf;
  if(!tdpot)
    tdpot = cgrid_alloc(1, 1, ntime + zf, 0.1, CGRID_PERIODIC_BOUNDARY, 0, "tdpot");
  if(upperx == NULL) return NULL;   /* potentials not given */
  if(!xxdiff)
    xxdiff = rgrid_alloc(dft_driver_nx, dft_driver_ny, dft_driver_nz, dft_driver_step, RGRID_PERIODIC_BOUNDARY, 0, "xxdiff");
  if(!xxave)
    xxave = rgrid_alloc(dft_driver_nx, dft_driver_ny, dft_driver_nz, dft_driver_step, RGRID_PERIODIC_BOUNDARY, 0, "xxave");
  fprintf(stderr, "libdft: Upper level potential.\n");
  rgrid_claim(workspace1); rgrid_claim(workspace2);
  rgrid_claim(workspace3); rgrid_claim(workspace4);
  dft_common_potential_map(upperave, upperx, uppery, upperz, workspace1);
  fprintf(stderr, "libdft: Lower level potential.\n");
  dft_common_potential_map(lowerave, lowerx, lowery, lowerz, workspace2);
  fprintf(stderr, "libdft: spectrum init complete.\n");
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

  rgrid_difference(xxdiff, workspace3, workspace4);
  rgrid_sum(xxave, workspace3, workspace4);
  rgrid_multiply(xxave, 0.5);
  rgrid_release(workspace1); rgrid_release(workspace2);
  rgrid_release(workspace3); rgrid_release(workspace4);
  return xxave;
}

/*
 * Collect the time dependent difference energy data. Same as above but with direct
 * grid input for potentials.
 * 
 * nt       = Maximum number of time steps to be collected (INT, input).
 * zerofill = How many zeros to fill in before FFT (int, input).
 * upper    = upper state potential grid (rgrid *, input).
 * lower    = lower state potential grid (rgrid *, input).
 *
 * Returns difference potential for dynamics.
 */

EXPORT rgrid *dft_driver_spectrum_init2(INT nt, INT zf, rgrid *upper, rgrid *lower) {

  cur_time = 0;
  ntime = nt;
  zerofill = zf;
  if(!tdpot)
    tdpot = cgrid_alloc(1, 1, ntime + zf, 0.1, CGRID_PERIODIC_BOUNDARY, 0, "tdpot");
  if(upper == NULL) return NULL; /* not given */
  if(!xxdiff)
    xxdiff = rgrid_alloc(dft_driver_nx, dft_driver_ny, dft_driver_nz, dft_driver_step, RGRID_PERIODIC_BOUNDARY, 0, "xxdiff");
  if(!xxave)
    xxave = rgrid_alloc(dft_driver_nx, dft_driver_ny, dft_driver_nz, dft_driver_step, RGRID_PERIODIC_BOUNDARY, 0, "xxave");
  rgrid_difference(xxdiff, upper, lower);
  rgrid_sum(xxave, upper, lower);
  rgrid_multiply(xxave, 0.5);
  return xxave;
}

/*
 * Collect the difference energy data (user calculated).
 *
 * val = difference energy value to be inserted (input, REAL).
 *
 */

EXPORT void dft_driver_spectrum_collect_user(REAL val) {

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
 * gwf     = the current wavefunction (used for calculating the liquid density) (wf *, input).
 *
 */

EXPORT void dft_driver_spectrum_collect(wf *gwf) {

  rgrid *workspace1 = dft_driver_otf->workspace1;

  if(cur_time > ntime) {
    fprintf(stderr, "libdft: initialized with too few points (spectrum collect).\n");
    exit(1);
  }
  rgrid_claim(workspace1);
  grid_wf_density(gwf, workspace1);
  rgrid_product(workspace1, workspace1, xxdiff);
  tdpot->value[cur_time] = rgrid_integral(workspace1);
  rgrid_release(workspace1);

  fprintf(stderr, "libdft: spectrum collect complete (point = " FMT_I ", value = " FMT_R " K).\n", cur_time, CREAL(tdpot->value[cur_time]) * GRID_AUTOK);
  cur_time++;
}

/*
 * Evaluate the spectrum.
 *
 * tstep       = Time step length at which the energy difference data was collected
 *               (fs; usually the simulation time step) (REAL, input).
 * tc          = Exponential decay time constant (fs; REAL, input).
 *
 * Returns a pointer to the calculated spectrum (grid *). X-axis in cm$^{-1}$.
 *
 */

EXPORT cgrid *dft_driver_spectrum_evaluate(REAL tstep, REAL tc) {

  INT t, npts;
  static cgrid *spectrum = NULL;

  if(cur_time > ntime) {
    printf(FMT_I " " FMT_I "\n", cur_time, ntime);
    fprintf(stderr, "libdft: cur_time >= ntime. Increase ntime.\n");
    exit(1);
  }

  tstep /= GRID_AUTOFS;
  tc /= GRID_AUTOFS;
  npts = 2 * (cur_time + zerofill - 1);
  if(!spectrum)
    spectrum = cgrid_alloc(1, 1, npts, GRID_HZTOCM1 / (tstep * GRID_AUTOFS * 1E-15 * ((REAL) npts)), CGRID_PERIODIC_BOUNDARY, 0, "spectrum");

#define SEMICLASSICAL /* */
  
#ifndef SEMICLASSICAL
  /* P(t) - full expression - see the Eloranta/Apkarian CPL paper on lineshapes */
  /* Instead of propagating the liquid on the excited state, it is run on the average (V_e + V_g)/2 potential */
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
	tmp += tdpot->value[tpp] * tstep;
      tmp2 += CEXP(-(I / HBAR) * tmp) * tstep;
    }
    spectrum->value[t] = -2.0 * CIMAG(tmp2) * EXP(-t * tstep / tc);
    fprintf(stderr, "libdft: Polarization at time " FMT_R " fs = %le.\n", t * tstep * GRID_AUTOFS, CREAL(spectrum->value[t]));
    spectrum->value[npts - t] = -spectrum->value[t];
  }
#else
  { REAL complex last, tmp;
    REAL ct;
    /* This seems to perform poorly - not in use */
    /* Construct semiclassical dipole autocorrelation function */
    last = (REAL) tdpot->value[0];
    tdpot->value[0] = 0.0;
    for(t = 1; t < cur_time; t++) {
      tmp = tdpot->value[t];
      tdpot->value[t] = tdpot->value[t-1] + tstep * (last + tmp)/2.0;
      last = tmp;
    }
    
    if(tc < 0.0) tc = -tc;
    spectrum->value[0] = 1.0;
    for (t = 1; t < cur_time; t++) {
      ct = tstep * (REAL) t;
      spectrum->value[t] = CEXP(-I * tdpot->value[t] / HBAR - ct / tc); /* transition dipole moment = 1 */
      spectrum->value[npts - t] = CEXP(I * tdpot->value[t] / HBAR - ct / tc); /* last point rolled over */
    }
  }
#endif
  
  /* zero fill */
  for (t = cur_time; t < cur_time + zerofill; t++)
    spectrum->value[t] = spectrum->value[npts - t] = 0.0;

  /* flip zero frequency to the middle */
  for (t = 0; t < 2 * (cur_time + zerofill - 1); t++)
    spectrum->value[t] *= POW(-1.0, (REAL) t);
  
  cgrid_inverse_fft(spectrum);

  /* Make the spectrum appear in the real part rather than imaginary */
#ifndef SEMI
  for(t = 0; t < npts; t++)
    spectrum->value[t] *= -I;
#endif
  
  return spectrum;
}
