/*
 * Spectroscopy related routines (Part 1b): Andersson lineshape with zero-point for impurity.
 *
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <grid/grid.h>
#include <grid/au.h>
#include "dft.h"
#include "ot.h"

/* Local auxiliary functions */

static REAL complex dft_eval_exp(REAL complex a, void *NA) { /* a contains t */

  return (1.0 - CEXP(-I * a));
}

static void do_gexp(cgrid *gexp, rgrid *dpot, REAL t) {

  grid_real_to_complex_re(gexp, dpot);
  cgrid_multiply(gexp, t);
  cgrid_operate_one(gexp, gexp, dft_eval_exp, NULL);
  cgrid_fft(gexp);    // prepare for convolution with density
}

static REAL complex dft_do_int2(cgrid *gexp, rgrid *imdens, cgrid *fft_dens, REAL t) {

  cgrid_fft_convolute(gexp, fft_dens, gexp);
  cgrid_inverse_fft(gexp);
  grid_product_complex_with_real(gexp, imdens);

  return cgrid_integral(gexp);
}

/* End aux */

/*
 *
 * Evaluate absorption/emission spectrum using the Andersson
 * expression. Zero-point correction for the impurity included.
 *
 * density   = Current liquid density (rgrid *; input).
 * imdensity = Current impurity zero-point density (rgrid *; input).
 * diffpot   = Difference potential (rgrid *; input).
 * spectrum  = Complex spectrum grid (cgrid *; input/output).
 * wrk1      = Workspace 1 (cgrid *).
 * wrk2      = Workspace 2 (cgrid *).
 *
 * The spectrum will start from smaller to larger frequencies.
 * The spacing between the points is included in cm-1.
 *
 * No return value.
 *
 */

EXPORT void dft_spectrum_andersson_zp(rgrid *density, rgrid *imdensity, rgrid *diffpot, cgrid *spectrum, cgrid *wrk1, cgrid *wrk2) {

  INT i;
  REAL t;

  if(spectrum->nx != 1 || spectrum->ny != 1) {
    fprintf(stderr, "libgrid: spectrum must be 1-D grid.\n");
    exit(1);
  }
  cgrid_host_lock(spectrum);

  grid_real_to_complex_re(wrk2, density);
  cgrid_fft(wrk2);
  
  for(i = 0; i < spectrum->nz; i++) {
    t = spectrum->step * (REAL) i;
    do_gexp(wrk1, diffpot, t); /* gexp grid + FFT */
    spectrum->value[i] = CEXP(dft_do_int2(wrk1, imdensity, wrk2, t)) * POW(-1.0, (REAL) i);
    fprintf(stderr,"libdft: Corr(" FMT_R " fs) = " FMT_R " " FMT_R "\n", t * GRID_AUTOFS, CREAL(spectrum->value[i]), CIMAG(spectrum->value[i]));
  }

  cgrid_fft(spectrum);   // forward, so omit minus sign in the exponent above
  for (i = 0; i < spectrum->nz; i++)
    spectrum->value[i] = CABS(spectrum->value[i]);  // TODO: real, imag, or power?
  spectrum->step = GRID_HZTOCM1 / (spectrum->step * GRID_AUTOFS * 1E-15 * (REAL) spectrum->nz);

  cgrid_host_unlock(spectrum);
}
