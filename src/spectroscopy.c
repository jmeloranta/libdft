/*
 * Spectroscopy related routines (Part 1).
 *
 * TODO: Add simple binning of energies as a function of time.
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
 * Evaluate absorption/emission spectrum using the Andersson
 * expression (no dynamics). No zero-point correction for the impurity.
 *
 * otf        = Orsay-Trento functional pointer (dft_ot_functional *; input).
 * density    = Current liquid density (rgrid *; input).
 * tstep      = Time step for constructing the time correlation function
 *              (REAL; input in fs). Typically around 1 fs.
 * endtime    = End time in constructing the time correlation function
 *              (REAL; input in fs). Typically less than 10,000 fs.
 * finalave   = Averaging of the final state potential.
 *              0: no averaging, 1 = average XY, 2 = average YZ, 3 = average XZ,
 *              4 = average XYZ.
 * finalx     = Final state potential along the X axis (char *; input).
 * finaly     = Final state potential along the Y axis (char *; input).
 * finalz     = Final state potential along the Z axis (char *; input).
 * initialave = Averaging of the initial state potential.
 *              0: no averaging, 1 = average XY, 2 = average YZ, 3 = average XZ,
 *              4 = average XYZ.
 * initialx   = Initial state potential along the X axis (char *; input).
 * initialy   = Initial state potential along the Y axis (char *; input).
 * initialz   = Initial state potential along the Z axis (char *; input).
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
  return cgrid_integral(wrk);
}

EXPORT cgrid *dft_spectrum(dft_ot_functional *otf, rgrid *density, REAL tstep, REAL endtime, char finalave, char *finalx, char *finaly, char *finalz, char initialave, char *initialx, char *initialy, char *initialz) {

  rgrid *workspace1, *workspace2;
  cgrid *cworkspace;
  static cgrid *corr = NULL;
  REAL t;
  INT i, ntime;
  static INT prev_ntime = -1;

  if(!otf->workspace1) otf->workspace1 = rgrid_clone(otf->density, "OTF workspace 1");
  if(!otf->workspace2) otf->workspace2 = rgrid_clone(otf->density, "OTF workspace 2");

  workspace1 = otf->workspace1;
  workspace2 = otf->workspace2;

  cworkspace = cgrid_alloc(density->nx, density->ny, density->nz, density->step, CGRID_PERIODIC_BOUNDARY, 0, "cworkspace (dft_spectrum)");

  endtime /= GRID_AUTOFS;
  tstep /= GRID_AUTOFS;
  ntime = (1 + (int) (endtime / tstep));
  if(ntime != prev_ntime) {
    if(corr) cgrid_free(corr);
    corr = cgrid_alloc(1, 1, ntime, 0.1, CGRID_PERIODIC_BOUNDARY, 0, "correlation function");
    cgrid_host_lock(corr); // Since we operate on this grid directly, make sure that it can never be on GPU
    prev_ntime = ntime;
  }

  rgrid_claim(workspace1);
  rgrid_claim(workspace2);

  dft_common_potential_map(finalave, finalx, finaly, finalz, workspace1);
  dft_common_potential_map(initialave, initialx, initialy, initialz, workspace2);
  rgrid_difference(workspace1, workspace1, workspace2); /* final - initial */
  
  rgrid_product(workspace2, workspace1, density);  // dpot * density = average shift
  fprintf(stderr, "libdft: Average shift = " FMT_R " cm-1.\n", rgrid_integral(workspace2) * GRID_AUTOCM1);

   for(i = 0; i < ntime; i++) {
    t = tstep * (REAL) i;
    corr->value[i] = CEXP(dft_do_int(density, workspace1, t, cworkspace)) * POW(-1.0, (REAL) i);  // Omit minus sign from exponent since we are doing forward FFT...
  }
  rgrid_release(workspace1);
  rgrid_release(workspace2);

  cgrid_fft(corr);
  for (i = 0; i < corr->nx; i++)
    corr->value[i] = CABS(corr->value[i]);

  corr->step = GRID_HZTOCM1 / (tstep * GRID_AUTOFS * 1E-15 * (REAL) ntime);

  cgrid_free(cworkspace);

  return corr;
}

/*
 *
 * Evaluate absorption/emission spectrum using the Andersson
 * expression. Zero-point correction for the impurity included.
 *
 * otf      = Orsay-Trento functional pointer (dft_ot_functional *; input).
 * density  = Current liquid density (rgrid *; input).
 * imdensity= Current impurity zero-point density (rgrid *; input).
 * tstep    = Time step for constructing the time correlation function
 *            (REAL; input in fs). Typically around 1 fs.
 * endtime  = End time in constructing the time correlation function
 *            (REAL; input in fs). Typically less than 10,000 fs.
 * upperave = Averaging of the final state potential.
 *            0: no averaging, 1 = average XY, 2 = average YZ, 3 = average XZ,
 *            4 = average XYZ.
 * finalx   = Final state potential along the X axis (char *; input).
 * finaly   = Final state potential along the Y axis (char *; input).
 * finalz   = Final state potential along the Z axis (char *; input).
 * lowerave = Averaging of the inital state potential.
 *            0: no averaging, 1 = average XY, 2 = average YZ, 3 = average XZ,
 *            4 = average XYZ.
 * initial  = Lower state potential along the X axis (char *; input).
 * initial  = Lower state potential along the Y axis (char *; input).
 * initial  = Lower state potential along the Z axis (char *; input).
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
  cgrid_fft(gexp);    // prepare for convolution with density
}

static REAL complex dft_do_int2(cgrid *gexp, rgrid *imdens, cgrid *fft_dens, REAL t, cgrid *wrk) {

  cgrid_fft_convolute(wrk, fft_dens, gexp);
  cgrid_inverse_fft(wrk);
  grid_product_complex_with_real(wrk, imdens);

  return cgrid_integral(wrk);
}

EXPORT cgrid *dft_spectrum_zp(dft_ot_functional *otf, rgrid *density, rgrid *imdensity, REAL tstep, REAL endtime, char finalave, char *finalx, char *finaly, char *finalz, char initialave, char *initialx, char *initialy, char *initialz) {

  cgrid *wrk, *fft_density, *gexp;
  rgrid *workspace1 = otf->workspace1, *workspace2 = otf->workspace2;
  static cgrid *corr = NULL;
  REAL t;
  INT i, ntime;
  static INT prev_ntime = -1;

  endtime /= GRID_AUTOFS;
  tstep /= GRID_AUTOFS;
  ntime = (1 + (int) (endtime / tstep));

  if(!otf->workspace1) otf->workspace1 = rgrid_clone(otf->density, "OTF workspace 1");
  if(!otf->workspace2) otf->workspace2 = rgrid_clone(otf->density, "OTF workspace 2");

  workspace1 = otf->workspace1;
  workspace2 = otf->workspace2;

  fft_density = cgrid_alloc(density->nx, density->ny, density->nz, density->step, CGRID_PERIODIC_BOUNDARY, 0, "DR spectrum_zp fftd");
  wrk = cgrid_alloc(density->nx, density->ny, density->nz, density->step, CGRID_PERIODIC_BOUNDARY, 0, "DR spectrum_zp wrk");
  gexp = cgrid_alloc(density->nx, density->ny, density->nz, density->step, CGRID_PERIODIC_BOUNDARY, 0, "DR spectrum_zp gexp");

  if(ntime != prev_ntime) {
    if(corr) cgrid_free(corr);
    corr = cgrid_alloc(1, 1, ntime, 0.1, CGRID_PERIODIC_BOUNDARY, 0, "correlation function");
    cgrid_host_lock(corr); // This may not move to GPU as we operate on it directly
    prev_ntime = ntime;
  }
  
  rgrid_claim(workspace1);
  rgrid_claim(workspace2);

  dft_common_potential_map(finalave, finalx, finaly, finalz, workspace1);
  dft_common_potential_map(initialave, initialx, initialy, initialz, workspace2);
  rgrid_difference(workspace1, workspace1, workspace2);

  rgrid_release(workspace2);
  
  grid_real_to_complex_re(fft_density, density);
  cgrid_fft(fft_density);
  
  for(i = 0; i < ntime; i++) {
    t = tstep * (REAL) i;
    do_gexp(gexp, workspace1, t); /* gexp grid + FFT  (workspace1 = dpot) */
    corr->value[i] = CEXP(dft_do_int2(gexp, imdensity, fft_density, t, wrk)) * POW(-1.0, (REAL) i);
    fprintf(stderr,"libdft: Corr(" FMT_R " fs) = " FMT_R " " FMT_R "\n", t * GRID_AUTOFS, CREAL(corr->value[i]), CIMAG(corr->value[i]));
  }
  rgrid_release(workspace1);

  cgrid_fft(corr);   // forward, so omit minus sign in the exponent above
  for (i = 0; i < corr->nx; i++)
    corr->value[i] = CABS(corr->value[i]);
  
  corr->step = GRID_HZTOCM1 / (tstep * GRID_AUTOFS * 1E-15 * (REAL) ntime);
  
  cgrid_free(fft_density);
  cgrid_free(wrk);
  cgrid_free(gexp);
  
  return corr;
}
