/*
 * Spectroscopy related routines (Part 1a): Andersson lineshape for classical impurity.
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

static REAL complex dft_do_int(rgrid *dens, rgrid *dpot, REAL t, cgrid *wrk) {

  grid_real_to_complex_re(wrk, dpot);
  cgrid_multiply(wrk, t);
  cgrid_operate_one(wrk, wrk, dft_eval_exp, NULL);
  grid_product_complex_with_real(wrk, dens);
  return cgrid_integral(wrk);
}

/* End aux */

/*
 * Evaluate absorption/emission spectrum using the Andersson
 * expression (no dynamics). No zero-point correction for the impurity.
 *
 * density    = Current liquid density (rgrid *; input/output). Overwritten on exit!
 * diffpot    = Difference potential (rgrid *; input).
 * spectrum   = Complex spectrum grid (cgrid *; input/output). This must be 1-D grid.
 *              On input: nz and step are the number of points used and time step.
 *              On output: step is the spectrum step length in cm-1.
 * wrk        = Workspace (cgrid *).
 *
 * The spectrum will start from smaller to larger frequencies.
 * The spacing between the points is included in cm-1.
 *
 * No return value.
 *
 */

EXPORT void dft_spectrum_andersson(rgrid *density, rgrid *diffpot, cgrid *spectrum, cgrid *wrk) {

  INT i;

  if(spectrum->nx != 1 || spectrum->ny != 1) {
    fprintf(stderr, "libgrid: spectrum must be 1-D grid.\n");
    exit(1);
  }
  cgrid_host_lock(spectrum);

  for(i = 0; i < spectrum->nz; i++)
    spectrum->value[i] = CEXP(dft_do_int(density, diffpot, spectrum->step * (REAL) i, wrk)) * POW(-1.0, (REAL) i);  // Omit minus sign from exponent since we are doing forward FFT...
  cgrid_fft(spectrum);
  for(i = 0; i < spectrum->nz; i++)
    spectrum->value[i] = CREAL(spectrum->value[i]); // TODO: real, imag, or power?
  spectrum->step = GRID_HZTOCM1 / (spectrum->step * GRID_AUTOFS * 1E-15 * (REAL) spectrum->nz);

  rgrid_product(density, density, diffpot);
  fprintf(stderr, "libdft: Average shift = " FMT_R " cm-1.\n", rgrid_integral(density) * GRID_AUTOCM1);

  cgrid_host_unlock(spectrum);
}
