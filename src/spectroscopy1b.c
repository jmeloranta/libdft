/*
 * Spectroscopy related routines (Part 1b): Anderson lineshape with zero-point for impurity.
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

static REAL complex dft_do_int(cgrid *fft_dens, rgrid *imdens, rgrid *dpot, REAL t, cgrid *wrk) {

  grid_real_to_complex_re(wrk, dpot);
  cgrid_multiply(wrk, t);
  cgrid_operate_one(wrk, wrk, dft_eval_exp, NULL);

  cgrid_fft(wrk);
  cgrid_fft_convolute(wrk, fft_dens, wrk);
  cgrid_inverse_fft_norm2(wrk);
  grid_product_complex_with_real(wrk, imdens);
  return cgrid_integral(wrk);
}

/* End aux */

/*
 *
 * Evaluate absorption/emission spectrum using the Anderson
 * expression. Zero-point correction for the impurity included.
 *
 * density   = Current liquid density (rgrid *; input).
 * imdensity = Current impurity zero-point density (rgrid *; input).
 * diffpot   = Difference potential: Final state - Initial state (rgrid *; input).
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

EXPORT void dft_spectrum_anderson_zp(rgrid *density, rgrid *imdensity, rgrid *diffpot, cgrid *spectrum, cgrid *wrk1, cgrid *wrk2) {

  INT i;

  if(spectrum->nx != 1 || spectrum->ny != 1) {
    fprintf(stderr, "libgrid: spectrum must be 1-D grid.\n");
    exit(1);
  }
#ifdef GRID_MGPU
  cgrid_host_lock(spectrum);
#endif

  grid_real_to_complex_re(wrk2, density);
  cgrid_fft(wrk2);
  
  for(i = 0; i < spectrum->nz; i++) {
    if(i <= spectrum->nz/2)
      spectrum->value[i] = CEXP(-dft_do_int(wrk2, imdensity, diffpot, ((REAL) i) * spectrum->step, wrk1));
    else
      spectrum->value[i] = CEXP(-dft_do_int(wrk2, imdensity, diffpot, ((REAL) (i - spectrum->nz)) * spectrum->step, wrk1));
    if(i & 1) spectrum->value[i] *= -1.0;
  }

  cgrid_inverse_fft(spectrum);

  spectrum->step = GRID_HZTOCM1 / (spectrum->step * GRID_AUTOFS * 1E-15 * (REAL) spectrum->nz);

  rgrid_fft(diffpot);
  rgrid_fft(imdensity);
  rgrid_fft_convolute(diffpot, diffpot, imdensity);
  rgrid_inverse_fft_norm2(diffpot);
  rgrid_product(density, density, diffpot);
  fprintf(stderr, "libdft: Average shift = " FMT_R " cm-1.\n", rgrid_integral(density) * GRID_AUTOCM1);

#ifdef GRID_MGPU
  cgrid_host_unlock(spectrum);
#endif
}
