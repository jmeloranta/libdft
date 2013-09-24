/*
 * Runge-Kutta for propagating the classical particle.
 *
 * (incuded from classical.c)
 *
 */

#ifdef ZERO_P
#define IMASS (63.546 / GRID_AUTOAMU) /* arbitrarily picked */
#define A0 0.0
#define A1 0.0
#define A2 0.0
#define A3 0.0
#define A4 0.0
#define A5 0.0
#define RMIN 0.0
#define RADD 0.0
#endif

/* exponential repulsion */
#ifdef EXP_P
#define IMASS (63.546 / GRID_AUTOAMU) /* arbitrarily picked */
#define A0 (3.8003E5 / GRID_AUTOK)
#define A1 (1.6245 * GRID_AUTOANG)
#define A2 0.0
#define A3 0.0
#define A4 0.0
#define A5 0.0
#define RMIN 3.0
#define RADD (-8.0)
#endif

/* Cu */
#ifdef CU_P
#define IMASS (63.546 / GRID_AUTOAMU) /* ion atom mass (Cu) */
#define A0 5.00496
#define A1 1.32322
#define A2 0.00915782
#define A3 7.55343
#define A4 1885.19
#define A5 (-16100.1)
#define RMIN 3.6    /* was 3.6 */
#define RADD 0.0
#endif

double dpot_func(void *arg, double z, double rr) {

  double ZI = *((double *) arg);
  double r = sqrt(rr * rr + (z - ZI) * (z - ZI)) + RADD;
  double r2 = r * r;
  double r3 = r2 * r;
  double r5 = r2 * r3;
  double r7 = r5 * r2;
  double r9 = r7 * r2;
  double r11 = r9 * r2;

  if(r < RMIN) return 0.0;   /* hopefully no liquid density in the core region */
  return ((ZI - z) / r) * (-A0 * A1 * exp(-A1 * r) + 4.0 * A2 / r5 + 6.0 * A3 / r7 + 8.0 * A4 / r9 + 10.0 * A5 / r11);
}

double pot_func(void *arg, double z, double rr) {

  double ZI = *((double *) arg);
  double r = sqrt(rr * rr + (z - ZI) * (z - ZI)) + RADD;
  double r2, r3, r4, r6, r8, r10, tmp;

  if(r < RMIN) return r = RMIN;
  r2 = r * r;
  r3 = r2 * r;
  r4 = r2 * r2;
  r6 = r4 * r2;
  r8 = r6 * r2;
  r10 = r8 * r2;
  return A0 * exp(-A1 * r) - A2 / r4 - A3 / r6 - A4 / r8 - A5 / r10;
}

/* Force evaluation for the classical atom/molecule */
/* Checked with numerical gradient */
double force(rgrid2d *density, double ZI) {

  return -rgrid2d_weighted_integral_cyl(density, dpot_func, (void *) &ZI);
}

double propagate_impurity(double *z, double *vz, double *az, rgrid2d *density) {

  double time_step = TIME_STEP / GRID_AUTOFS;
  double fz, vhalf;

  vhalf = *vz + 0.5 * (*az) * time_step;
  *z += vhalf * time_step;
  fz = force(density, *z);
  *az = fz / IMASS;
  *vz = vhalf + 0.5 * (*az) * time_step;

#ifdef MAX_VELOC
  if(fabs(*vz) > MAX_VELOC) {
    if(*vz > 0.0) *vz = MAX_VELOC;
    else *vz = -MAX_VELOC;
    *az = 0.0;
  }
#endif

  return fz;
}
