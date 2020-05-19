/*
 * Reference experimental data for bulk superfluid helium (Donnelly and Barenghi).
 *
 * Eveything in SI units - NOT ATOMIC UNITS.
 *
 */

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "helium-exp-bulk.h"
#define EXPORT

/*
 * Spline routine to extract spline data (J. Phys. Chem. Ref. Data 27, 1217 (1998)).
 *
 * ncap7 = number of knots, internal and external
 * k = pointer to array of knots
 * c = pointer to array of coefficients
 * x = value of independent variable
 * first = address of variable to hold first derivative
 * second = address of variable to hold second derivative
 * This function returns the value of the spline at point X
 *
 * NOTE: Returns 1E99 if error occurred.
 *
 */

static REAL dft_exp_bulk_spline_eval(INT ncap7, REAL *k, REAL *c, REAL x, REAL *first, REAL *second) {

  INT j, j1r, L;
  REAL s, k1r, k2, k3, k4, k5, k6, e2, e3, e4, e5, c11, cd11, c21, cd21, cd31, c12, cd12;
  REAL cdd12, c22, cd22, cdd22, c31;

  if(x >= k[3] && x <= k[ncap7-3]) {
    j1r = -1;
    j = ncap7 - 7;
    L = (j1r + j) / 2;
    while(j - j1r > 1) {
      if(x >= k[L+4]) j1r = L;
      else j = L;
      L = (j1r + j) / 2;
    }

    k1r = k[j + 1];
    k2 = k[j + 2];
    k3 = k[j + 3];
    k4 = k[j + 4];
    k5 = k[j + 5];
    k6 = k[j + 6];
    e2 = x - k2;
    e3 = x - k3;
    e4 = k4 - x;
    e5 = k5 - x;
    c11 = ((x - k1r) * c[j+1] + e4 * c[j]) / (k4-k1r);
    cd11 = (c[j+1] - c[j]) / (k4 - k1r);
    c21 = (e2 * c[j+2] + e5 * c[j+1]) / (k5 - k2);
    cd21 =(c[j+2] - c[j+1]) / (k5 - k2);
    c31 = (e3 * c[j + 3] + (k6 - x) * c[j + 2]) / (k6 - k3);
    cd31 = (c[j + 3] - c[j + 2]) / (k6 - k3);
    c12 = (e2 * c21 + e4 * c11) / (k4 - k2);
    cd12 = (c21 + e2 * cd21 - c11 + e4 * cd11) / (k4 - k2);
    cdd12 = 2 * (cd21 - cd11) / (k4 - k2);
    c22 = (e3 * c31 + e5 * c21) / (k5 - k3);
    cd22 = (c31 + e3 * cd31 - c21 + e5 * cd21) / (k5 - k3);
    cdd22 = 2 * (cd31 - cd21) / (k4 - k3);
    s = (e3 * c22 + e4 * c12) / (k4 - k3);
    *first = (e3 * cd22 + c22 + e4 * cd12 - c12) / (k4 - k3);
    *second = (e3 * cdd22 + 2 * cd22 + e4 * cdd12 - 2 * cd12) / (k4 - k3);
  } else {
    printf("libdft: Requested value outside spline region in dft_bulk_spline_eval().\n");
    *first = *second = 0.0;
    return 1E99; // Error status return
  }
  return s;
}

/*
 * Enthalpy at saturated vapor pressure and a given temperature.
 *
 * temperature = Temperature at which the enthalpy is requested (REAL; input).
 * first       = First derivative of enthalpy at the temperature (REAL *; output). If NULL, not computed.
 * second      = Second derivative of enthalpy at the temperature (REAL *; output). If NULL, not computed.
 *
 * Returns enthalpy (J / mol) at the temperature (REAL).
 *
 */

EXPORT REAL dft_exp_bulk_enthalpy(REAL temperature, REAL *first, REAL *second) {

  REAL f, s, e;

  e = dft_exp_bulk_spline_eval(DFT_BULK_ENTHALPY_KNOTS, dft_bulk_enthalpy_k, dft_bulk_enthalpy_c, temperature, &f, &s);
  if(first) *first = f;
  if(second) *second = s;  
  return e;
}

/*
 * Return temperature for given enthalpy (inverse of the above). The inversion is unique.
 *
 * enthalpy = Enthalpy at which the temperature is requested (REAL; input).
 *
 * Returns the temperature (REAL).
 *
 */

EXPORT REAL dft_exp_bulk_enthalpy_inverse(REAL energy) {

  REAL temp = 0.0;

  // Search for the matching enthalpy
  while(1) {
    if(dft_exp_bulk_enthalpy(temp, NULL, NULL) >= energy) break;
    temp += 0.01; // search with 0.01 K accuracy
  }
  return temp;
}

/*
 * Dispersion relation at saturated vapor pressure.
 *
 * wavenumber  = Wavenumber for which the energy is computed (REAL; input).
 *
 * Returns Energy (Kelvin) corresponding to the wavenumber (REAL).
 *
 */

EXPORT REAL dft_exp_bulk_dispersion(REAL k) {

  REAL f, s, e;

  /* The spline data does not extend down to zero... use linear interpolation there */
  if(k < dft_bulk_dispersion_k[3])
    return (k / dft_bulk_dispersion_k[3]) * dft_exp_bulk_spline_eval(DFT_BULK_DISPERSION_KNOTS, dft_bulk_dispersion_k, dft_bulk_dispersion_c, dft_bulk_dispersion_k[3], &f, &s);
  e = dft_exp_bulk_spline_eval(DFT_BULK_DISPERSION_KNOTS, dft_bulk_dispersion_k, dft_bulk_dispersion_c, k, &f, &s);
  return e;
}

/*
 * Supefluid fraction at saturated vapor pressure and a given temperature.
 *
 * temperature = Temperature at which the enthalpy is requested (REAL; input).
 *
 * Returns superfluid fraction between 0.0 and 1.0 (REAL).
 *
 */

EXPORT REAL dft_exp_bulk_superfluid_fraction(REAL temperature) {

  REAL f, s, e;

  e = dft_exp_bulk_spline_eval(DFT_BULK_SUPERFRACTION_KNOTS, dft_bulk_superfraction_k, dft_bulk_superfraction_c, temperature, &f, &s);
  return e;
}

/*
 * Return temperature for given superfluid fraction (inverse of the above). The inversion is unique.
 *
 * sfrac = Superfluid fraction at which the temperature is requested (REAL; input).
 *
 * Returns the temperature (REAL).
 *
 */

EXPORT REAL dft_exp_bulk_superfluid_fraction_inverse(REAL sfrac) {

  REAL temp = 0.0;

  // Search for the matching enthalpy
  while(1) {
    if(dft_exp_bulk_superfluid_fraction(temp) <= sfrac || temp >= 2.1768) break;
    temp += 0.01; // search with 0.01 accuracy
  }
  return temp;
}
