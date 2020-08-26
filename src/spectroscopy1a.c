/*
 * Spectroscopy related routines (Part 1a): Anderson lineshape for classical impurity.
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
 * @FUNC{dft_spectrum_anderson, "Evaluate absorption/emission spectrum using the Anderson formula"}
 * @DESC{"Evaluate absorption/emission spectrum using the Anderson expression (no dynamics). 
          No zero-point correction for the impurity. The spectrum will start from smaller to larger frequencies.
          The spacing between the points is included in cm$^{-1}$"}
 * @ARG1{rgrid *density, "Current liquid density (input). Overwritten on exit"}
 * @ARG2{rgrid *diffpot, "Difference potential: Final state - Initial state (input)"}
 * @ARG3{cgrid *spectrum, "1-D complex spectrum grid (input/output). On input: nz and step in spectrum are the number of points used and step is thetime step. On output: step is the spectrum step length in cm$^{-1}$"}
 * @ARG4{cgrid *wrk, "Complex workspace"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT void dft_spectrum_anderson(rgrid *density, rgrid *diffpot, cgrid *spectrum, cgrid *wrk) {

  INT i;

  if(spectrum->nx != 1 || spectrum->ny != 1) {
    fprintf(stderr, "libgrid: spectrum must be 1-D grid.\n");
    exit(1);
  }
#ifdef GRID_MGPU
  cgrid_host_lock(spectrum);
#endif

  for(i = 0; i < spectrum->nz; i++) {
    if(i <= spectrum->nz/2)
      spectrum->value[i] = CEXP(-dft_do_int(density, diffpot, ((REAL) i) * spectrum->step, wrk));
    else
      spectrum->value[i] = CEXP(-dft_do_int(density, diffpot, ((REAL) (i - spectrum->nz)) * spectrum->step, wrk));
    if(i & 1) spectrum->value[i] *= -1.0;
  }

  cgrid_inverse_fft(spectrum);

  spectrum->step = GRID_HZTOCM1 / (spectrum->step * GRID_AUTOFS * 1E-15 * (REAL) spectrum->nz);

  rgrid_product(density, density, diffpot);
  fprintf(stderr, "libdft: Average shift = " FMT_R " cm-1.\n", rgrid_integral(density) * GRID_AUTOCM1);

#ifdef GRID_MGPU
  cgrid_host_unlock(spectrum);
#endif
}
