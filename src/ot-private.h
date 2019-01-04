/*
 * Private functions to ot.c
 *
 */

/*
 * Backflow potential function.
 *
 */

EXPORT REAL dft_ot_backflow_pot(void *arg, REAL x, REAL y, REAL z) {

  REAL g11 = ((dft_ot_bf *) arg)->g11;
  REAL g12 = ((dft_ot_bf *) arg)->g12;
  REAL g21 = ((dft_ot_bf *) arg)->g21;
  REAL g22 = ((dft_ot_bf *) arg)->g22;
  REAL a1 = ((dft_ot_bf *) arg)->a1;
  REAL a2 = ((dft_ot_bf *) arg)->a2;
  REAL r2 = x * x + y * y + z * z;
  
  return (g11 + g12 * r2) * EXP(-a1 * r2) + (g21 + g22 * r2) * EXP(-a2 * r2);
}

/*
 * Backflow potential function (1-D).
 *
 */

EXPORT REAL dft_ot_backflow_pot_1d(void *arg, REAL x, REAL y, REAL z) {

  REAL g11 = ((dft_ot_bf *) arg)->g11;
  REAL g12 = ((dft_ot_bf *) arg)->g12;
  REAL g21 = ((dft_ot_bf *) arg)->g21;
  REAL g22 = ((dft_ot_bf *) arg)->g22;
  REAL a1 = ((dft_ot_bf *) arg)->a1;
  REAL a2 = ((dft_ot_bf *) arg)->a2;
  REAL z2 = z * z;
  
  return M_PI * ((g11 + g12 * (1.0 + a1 * z2) / a1) * EXP(-a1 * z2) / a1
               + (g21 + g22 * (1.0 + a2 * z2) / a2) * EXP(-a2 * z2) / a2);
}
