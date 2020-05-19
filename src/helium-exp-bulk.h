/*
 * Spline curve data from J. Phys. Chem. Ref. Data 27, 1217 (1998).
 *
 */

/* Enthalpy as a function of temperature (SVP) */
#define DFT_BULK_ENTHALPY_KNOTS 26
REAL dft_bulk_enthalpy_k[] = {0.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.37, 0.5, 0.61, 0.74395, 0.87, 1.02755, 1.25, 1.5, 1.75, 2.0, 2.13, 2.190120, 2.232940, 2.854200, 4.0, 4.9, 4.9, 4.9, 4.9};
REAL dft_bulk_enthalpy_c[] = {0.0, 0.0, -1.65E-6, 1.801E-5, 1.185E-4, 4.108550E-4, 1.0332E-3, 2.71E-3, 6.09E-3, 1.784E-2, 6.03274E-2, 2.4005E-1, 8.64372E-1, 2.4454, 5.55436, 8.8, 12.2013, 15.581, 20.04, 32.231, 47.0464, 58.4893};

/* Dispersion relation (SVP) */
#define DFT_BULK_DISPERSION_KNOTS 14
REAL dft_bulk_dispersion_k[] = {0.0894, 0.0894, 0.0894, 0.0894, 0.15, 0.510, 1.60, 2.023, 2.42, 2.665, 3.60, 3.60, 3.60, 3.60};
REAL dft_bulk_dispersion_c[] = {1.53895, 1.932, 4.8, 14.85, 14.88, 5.9384, 16.5014, 17.72455, 18.43656, 18.43545};

/* Superfluid fraction as a function (SVP) */
#define DFT_BULK_SUPERFRACTION_KNOTS 25
REAL dft_bulk_superfraction_k[] = {0.0, 0.0, 0.0, 0.0, 0.443, 0.9012, 1.5419, 1.7540, 1.918, 2.111, 2.156991, 2.173218, 2.175647, 2.176358, 2.176568, 2.176692, 2.176766, 2.176791, 2.176798, 2.176799, 2.17679999, 2.1768, 2.1768, 2.1768, 2.1768};
REAL dft_bulk_superfraction_c[] = {1.451275432822459E-1, 1.451334563362309E-1, 1.449759191497576E-1, 1.455008000684433E-1, 1.4075E-1, 1.095E-1, 8.15E-2, 5.30E-2, 2.1E-2, 8.904576E-3, 3.053214E-3, 1.494043E-3, 8.342826E-4, 5.10686E-4, 2.8379E-4, 1.287426E-4, 5.202569E-5, 2.153580E-5, 8.564206E-6, 3.567958E-6, 0.0};

