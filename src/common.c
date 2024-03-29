/*
 * Common routines.
 *
 */

#include <stdio.h>
#include <math.h>
#include <grid/grid.h>
#include <grid/au.h>
#include "dft.h"
#include "ot.h"

/*
 * Tunable numerical parameters.
 *
 */

#define EPS 1.0E-8
#define CUTOFF (50.0 / GRID_AUTOK)

/*
 * @FUNC{dft_common_lj_func, "Function for Lennard-Jones potential"}
 * @DESC{"Lennard-Jones function"}
 * @ARG1{REAL r2, "Distance (r2 = r$^2$)"}
 * @ARG2{REAL sig, "Sigma in LJ potential"}
 * @ARG3{REAL eps, "Epsilon in LJ potential"}
 * @RVAL{REAL, "Returns Lennard-Jones potential at r$^2$"}
 *
 */

EXPORT inline REAL dft_common_lj_func(REAL r2, REAL sig, REAL eps) {

  /* s = (sigma/r)^6 */
  r2 = sig * sig / r2; /* r already squared elsewhere */
  r2 = r2 * r2 * r2;
  
  /* Vlj = 4 * eps ( (sigma/r)^12 - (sigma/r)^6 ) */ 
  return 4.0 * eps * r2 * (r2 - 1.0);
}

/*
 * @FUNC{dft_common_lennard_jones, "Lennard-Jones potential for rgrid_map()"}
 * @DESC{"Lennard-Jones potential to be used with grid map() routines
          Note that the LJ potential has zero core when $r < h$"}
 * @ARG1{void *arg, "Pointer to dft_common_lj structure for specifying the potential parameters"}
 * @ARG2{REAL x, "X-coordinate"}
 * @ARG3{REAL y, "Y-coordinate"}
 * @ARG4{REAL z, "Z-coordinate"}
 * @RVAL{REAL, "Returns Lennard-Jones potential at (x,y,z)"}
 *
 */

EXPORT inline REAL dft_common_lennard_jones(void *arg, REAL x, REAL y, REAL z) {

  REAL h = ((dft_common_lj *) arg)->h;
  REAL sig = ((dft_common_lj *) arg)->sigma;
  REAL eps = ((dft_common_lj *) arg)->epsilon;
  REAL cval = ((dft_common_lj *) arg)->cval;
  REAL r2;

  r2 = x * x + y * y + z * z;
  
  if (r2 <= h * h) return cval;
  
  return dft_common_lj_func(r2, sig, eps);
}

/*
 * @FUNC{dft_common_lennard_jones_1d, "Effective 1-D Lennard-Jones potential for rgrid_map()"}
 * @DESC{"Effective 1D Lennard-Jones potential to be used with grid map() routines
          Note that the LJ potential has zero core when $r < h$"}
 * @ARG1{void *arg, "Pointer to dft_common_lj structure specifying the LJ parameters"}
 * @ARG2{REAL x, "X-coordinate (here = 0.0)"}
 * @ARG3{REAL y, "Y-coordinate (here = 0.0)"}
 * @ARG4{REAL z, "Z-coordinate"}
 * @RVAL{REAL, "Returns Lennard-Jones potential at (0,0,z)"}
 *
 */

EXPORT inline REAL dft_common_lennard_jones_1d(void *arg, REAL x, REAL y, REAL z) {

  REAL h = ((dft_common_lj *) arg)->h;
  REAL sig = ((dft_common_lj *) arg)->sigma;
  REAL eps = ((dft_common_lj *) arg)->epsilon;
  REAL s6 = POW(sig, 6.0);

  z = FABS(z);
  if(z < h) z = h;
  return 2.0 * M_PI * eps * s6 * (2.0 * s6 - 5.0 * POW(z, 6.0)) / (5.0 * POW(z, 10.0));
}

/*
 * @FUNC{dft_common_lennard_jones_smooth, "Lennard-Jones potential with smoothed core for rgrid_map()"}
 * @DESC{"Lennard-Jones potential with smoothed core to be used with grid map()
          routines. Parameters passed in arg (see the regular LJ above)"}
 * @ARG1{void *arg, "Pointer to dft_common_lj structure (LJ parameters)"}
 * @ARG2{REAL x, "X-coordinate"}
 * @ARG3{REAL y, "Y-coordinate"}
 * @ARG4{REAL z, "Z-coordinate"}
 * @RVAL{REAL, "Returns Lennard-Jones potential at (x,y,z)"}
 *
 */

EXPORT inline REAL dft_common_lennard_jones_smooth(void *arg, REAL x, REAL y, REAL z) {

  REAL h = ((dft_common_lj *) arg)->h;
  REAL sig = ((dft_common_lj *) arg)->sigma;
  REAL eps = ((dft_common_lj *) arg)->epsilon;
  REAL r2, h2 = h * h;

  r2 = x * x + y * y + z * z;
  
  /* Ul(h) * (r/h)^4 */
  if (r2 < h2) return dft_common_lj_func(h2, sig, eps) * r2 * r2 / (h2 * h2);
  
  return dft_common_lj_func(r2, sig, eps);
}

/*
 * @FUNC{dft_common_spherical_avg, "Spherical averaging function to be used with rgrid_map()"}
 * @DESC{"Spherical average function to be used with grid map() routines.
          The sphere radius is passed in arg"}
 * @ARG1{void *arg, "Pointer to the radius of the sphere (REAL *)"}
 * @ARG2{REAL x, "X-coordinate"}
 * @ARG3{REAL y, "Y-coordinate"}
 * @ARG4{REAL z, "Z-coordinate"}
 * @RVAL{REAL, "Returns the value of the spherical average function at (x, y, z)"}
 *
 */

EXPORT inline REAL dft_common_spherical_avg(void *arg, REAL x, REAL y, REAL z) {

  REAL h = *((REAL *) arg), h2 = h * h;

  if (x*x + y*y + z*z <= h2)
    return 3.0 / (4.0 * M_PI * h * h2);
  return 0.0;
}

/*
 * @FUNC{dft_common_spherical_avg_1d, "Spherical averaging 1-D function to be used with rgrid_map()"}
 * @DESC{"1-D Spherical average function to be used with grid map() routines.
          The sphere radius is passed in arg"}
 * @ARG1{void *arg, "Pointer to the radius of the sphere (REAL *)"}
 * @ARG2{REAL x, "X-coordinate (here = 0.0)"}
 * @ARG3{REAL y, "Y-coordinate (here = 0.0)"}
 * @ARG4{REAL z, "Z-coordinate"}
 * @RVAL{REAL, "Returns the value of the 1-D spherical average function at (0, 0, z)"}
 *
 */

EXPORT inline REAL dft_common_spherical_avg_1d(void *arg, REAL x, REAL y, REAL z) {

  REAL h = *((REAL *) arg), h2 = h * h;

  z = FABS(z);
  if(z > h) return 0.0;
  else return 3.0 * (h2 - z * z) / (4.0 * h2 *h);
}

/*
 * @FUNC{dft_common_spherical_avg_k, "Spherical averaging function in reciprocal space"}
 * @DESC{"Spherical average function in reciprocal space to be used with grid map() routines.
          The sphere radius is passed in arg"}
 * @ARG1{void *arg, "Pointer to the radius of the sphere, hk (REAL *)"}
 * @ARG2{REAL kx, "kx-coordinate"}
 * @ARG3{REAL ky, "ky-coordinate"}
 * @ARG4{REAL kz, "kz-coordinate"}
 * @RVAL{REAL, "Returns the value of the spherical average function at (kx, ky, kz)"}
 *
 */

EXPORT inline REAL dft_common_spherical_avg_k(void *arg, REAL kx, REAL ky, REAL kz) {

  REAL hk = *((REAL *) arg) * SQRT(kx * kx + ky * ky + kz * kz);

  if(hk < 1.e-5) return 1.0 - 0.1 * hk * hk; /* second order Taylor expansion */
  return 3.0 * (SIN(hk) - hk * COS(hk)) / (hk * hk * hk);
}

/*
 * @FUNC{dft_common_gaussian, "Gaussian function for rgrid_map()"}
 * @DESC{"Gaussian function to be used with grid map() functions.
          The gaussian is centered at (0,0,0) and width is given in arg"}
 * @ARG1{void *arg, "Inverse width of the gaussian function (REAL *)"}
 * @ARG2{REAL x, "X-coordinate"}
 * @ARG3{REAL y, "Y-coordinate"}
 * @ARG4{REAL z, "Z-coordinate"}
 * @RVAL{REAL, "Returns the value of the value of the gaussian function at (x, y, z)"}
 * 
 */ 

EXPORT inline REAL dft_common_gaussian(void *arg, REAL x, REAL y, REAL z) {

  REAL inv_width = *((REAL *) arg);
  REAL norm = 0.5 * M_2_SQRTPI * inv_width;

  norm = norm * norm * norm;
  return norm * EXP(-(x * x + y * y + z * z) * inv_width * inv_width);
}

/*
 * @FUNC{dft_common_gaussian_1d, "Gaussian 1-D function for rgrid_map()"}
 * @DESC{"Effective 1D Gaussian function to be used with grid map() functions.
          The gaussian is centered at (0,0,0) and width is given in arg"}
 * @ARG1{void *arg, "Inverse width of the gaussian function (REAL *)"}
 * @ARG2{REAL x, "X-coordinate (here = 0.0)"}
 * @ARG3{REAL y, "Y-coordinate (here = 0.0)"}
 * @ARG4{REAL z, "Z-coordinate"}
 * @RVAL{REAL, "Returns the value of the value of the gaussian function at (0, 0, z)"}
 * 
 */ 

EXPORT inline REAL dft_common_gaussian_1d(void *arg, REAL x, REAL y, REAL z) {

  REAL inv_width = *((REAL *) arg);
  REAL norm = 0.5 * M_2_SQRTPI * inv_width;

  return norm * EXP(-z * z * inv_width * inv_width);
}

/*
 * 
 * @FUNC{dft_common_cgaussian, "Gaussian function for cgrid_map()"}
 * @DESC{"Complex Gaussian function to be used with cgrid map() functions.
          The gaussian is centered at (0,0,0) and width is given in arg"}
 * @ARG1{void *arg, "Inverse width of the gaussian function (REAL *)"}
 * @ARG2{REAL x, "X-coordinate"}
 * @ARG3{REAL y, "Y-coordinate"}
 * @ARG4{REAL z, "Z-coordinate"}
 * @RVAL{REAL complex, "Returns the value of the value of the gaussian function at (x, y, z)"}
 * 
 */ 

EXPORT inline REAL complex dft_common_cgaussian(void *arg, REAL x, REAL y, REAL z) {

  REAL inv_width = *((REAL *) arg);
  REAL norm = 0.5 * M_2_SQRTPI * inv_width;

  norm = norm * norm * norm;
  return (REAL complex) norm * EXP(-(x * x + y * y + z * z) * inv_width * inv_width);
}

/*
 * @FUNC{dft_common_lwl3, "Thermal de Broglie wavelength"}
 * @DESC{"Return thermal wavelength for a particle with mass 'mass' at temperature 'temp'"}
 * @ARG1{REAL mass, "Mass of the particle"}
 * @ARG2{REAL temp, "Temperature"}
 * @RVAL{REAL, "Returns the thermal wavelength"}
 *
 */

EXPORT inline REAL dft_common_lwl3(REAL mass, REAL temp) {

  REAL lwl;

  /* hbar = 1 */
  lwl = SQRT(2.0 * M_PI * HBAR * HBAR / (mass * GRID_AUKB * temp));
  return lwl * lwl * lwl;
}

/*
 * Functions for polylogarithms (direct evaluation or polynomial fits).
 *
 */

/* Use brute force evaluation of the series or polynomial fits? */
#define BRUTE_FORCE

#ifndef BRUTE_FORCE
/* Li_{1/2}(z) fit parameters */
#define LENA 9
static REAL A[LENA] = {-0.110646, 5.75395, 37.2433, 123.411, 228.972, 241.484, 143.839, 45.0107, 5.74587};

/* Li_{3/2}(z) fit parameters */
#define LENB 10
static REAL B[LENB] = {0.0025713, 1.0185, 0.2307, -0.15674, 1.0025, 1.8146, -1.8981, -3.0619, 1.438, 1.9332};

/* Li_{5/2}(z) fit parameters */
#define LENC 7
static REAL C[LENC] = {-0.00022908, 1.0032, 0.18414, 0.03958, -0.0024563, 0.057067, 0.052367};

/* parameters for g_{3/2}(z) = \rho\lambda^3 */
#define LEND 5
static REAL D[LEND] = {5.913E-5, 0.99913, -0.35069, 0.053981, -0.0038613};
#endif

/* 
 * @FUNC{dft_common_g, "Evaluate polylog ($g_s(z)$)"}
 * @DESC{"Evaluate $g_s(z)$ (polylog)"}
 * @ARG1{REAL z, "z argument of polylog"}
 * @ARG2{REAL s, "s argument of polylog"}
 * @RVAL{REAL, "Returns the polylog value"}
 *
 */

#ifdef BRUTE_FORCE
#define NTERMS 256
static inline REAL dft_common_g(REAL z, REAL s) { /* Brute force approach - the polynomial fits are not very accurate... */

  REAL val = 0.0, zk = 1.0;
  INT k;

  for (k = 1; k <= NTERMS; k++) {
    zk *= z;
    val += zk / POW((REAL) k, s);
  }
  return val;
}
#endif

/*
 * @FUNC{dft_common_fit_g12, "Evaluate polylog $g_{12}$"}
 * @DESC{"Evaluate polylog $g_{1/2}(z$)"}
 * @ARG1{REAL z, "Argument of polylog"}
 * @RVAL{REAL, "Returns the polylog value"}
 *
 */

EXPORT inline REAL dft_common_fit_g12(REAL z) {

#ifdef BRUTE_FORCE
  return dft_common_g(z, 1.0/2.0);
#else
  INT i;
  REAL rv = 0.0, e = 1.0;

  if(FABS(z) > 1.0) fprintf(stderr, "polylog: warning |z| > 1.\n");
  if(FABS(z) < 0.1) return z;   /* linear region -- causes a small discontinuous step (does this work for negative z?; well we don't need those amyway) */
  z -= 1.0;
  rv = A[0] / (z + EPS);
  for (i = 1; i < LENA; i++) {   /* reduced to LENA - 1 otherwise would overflow the array access */
    rv += A[i] * e;
    e *= z;
  }
  if(z + 1.0 > 0.0 && rv < 0.0) rv = 0.0; /* small values may give wrong sign (is this still neded?) */
  return rv;
#endif
}

/*
 * @FUNC{dft_common_fit_g32, "Evaluate polylog $g_{3/2}$"}
 * @DESC{"Evaluate polylog $g_{3/2}(z)$"}
 * @ARG1{REAL z, "Argument to polylog"}
 * @RVAL{REAL, "Returns the polylog"}
 *
 */

EXPORT inline REAL dft_common_fit_g32(REAL z) {

#ifdef BRUTE_FORCE
  return dft_common_g(z, 3.0/2.0);
#else
  INT i;
  REAL rv = 0.0, e = 1.0;

  if(FABS(z) > 1.0) fprintf(stderr, "polylog: warning |z| > 1.\n");
  for (i = 0; i < LENB; i++) {
    rv += B[i] * e;
    e *= z;
  }
  return rv;
#endif
}

/*
 * @FUNC{dft_common_fit_g52, "Evaluate polylog $g_{5/2}$"}
 * @DESC{"Evaluate polylog $g_{5/2}(z)$"}
 * @ARG1{REAL z, "Argument of polylog"}
 * @RVAL{REAL, "Returns the polylog"}
 *
 */

EXPORT inline REAL dft_common_fit_g52(REAL z) {

#ifdef BRUTE_FORCE
  return dft_common_g(z, 5.0/2.0);
#else
  INT i;
  REAL rv = 0.0, e = 1.0;

  if(FABS(z) > 1.0) fprintf(stderr, "polylog: warning |z| > 1.\n");
  for (i = 0; i < LENC; i++) {
    rv += C[i] * e;
    e *= z;
  }
  if(z < 1E-3) return 0.0;
  return rv;
#endif
}

/*
 * @FUNC{dft_common_fit_z, "Invert polylog $g_{3/2}$"}
 * @DESC{"Evaluate $z(\rho, T)$ (polylog) by inverting $g_{3/2}$"}
 * @ARG1{REAL val, "Value where $g_{3/2}$ is to be inverted"}
 * @RVAL{REAL, "Returns the inverted value"}
 *
 */

#define STOP 1E-6
#define GOLDEN ((SQRT(5.0) + 1.0) / 2.0)

EXPORT inline REAL dft_common_fit_z(REAL val) {

#ifdef BRUTE_FORCE
  /* Golden sectioning */
  REAL a, b, c, d, fc, fd, tmp;

  if(val >= dft_common_fit_g32(1.0)) return 1.0; /* g_{3/2}(1) */

  a = 0.0;
  b = 1.0;

  c = b - (b - a) / GOLDEN;
  d = a + (b - a) / GOLDEN;

  while (FABS(c - d) > STOP) {

    tmp = val - dft_common_fit_g32(c);
    fc = tmp * tmp;
    tmp = val - dft_common_fit_g32(d);
    fd = tmp * tmp;

    if(fc < fd) b = d; else a = c;

    c = b - (b - a) / GOLDEN;
    d = a + (b - a) / GOLDEN;
  }
  return (b + a) / 2.0;       
#else
  INT i;
  REAL rv = 0.0, e = 1.0;

  if(val >= dft_common_fit_g32(1.0)) return 1.0; /* g_{3/2}(1) */
  for (i = 0; i < LEND; i++) {
    rv += D[i] * e;
    e *= val;
  }
  if(rv <= 0.0) rv = 1E-6;
  return rv;
#endif
}

/*
 * @FUNC{dft_common_classical_idealgas_dEdRho, "Classical ideal gas: $dE/\rho$"}
 * @DESC{"Classical ideal gas. Free energy / volume derivative with respect to rho: $d(A/V) / d\rho$"}
 * @ARG1{REAL rhop, "Gas density"}
 * @RVAL{REAL, "Returns the free energy / volume derivative"}
 *
 */

EXPORT REAL dft_common_classical_idealgas_dEdRho(REAL rhop, void *params) {

  dft_ot_functional *otf = (dft_ot_functional *) params;
  REAL l3, val;

  l3 = dft_common_lwl3(otf->mass, otf->temp);
  val = GRID_AUKB * otf->temp * LOG(rhop * l3 + 1E-6*EPS);    // -log(1/x) = log(x)
  //  if (val > CUTOFF) val = CUTOFF;
  return val;
}

/*
 * @FUNC{dft_common_classical_idealgas_energy, "Classical ideal gas: energy per volume"}
 * @DESC{"Classical ideal gas. NVT free energy / volume (i.e., $A/V$, $A = U - TS$)"}
 * @ARG1{REAL rhop, "Gas density"}
 * @ARG2{dft_ot_functional *params, "Pointer to dft_ot_functional structure to get temperature and mass"}
 * @RVAL{REAL, "Returns free energy / volume"}
 *
 */

EXPORT REAL dft_common_classical_idealgas_energy(REAL rhop, void *params) {

  REAL l3;
  dft_ot_functional *otf = (dft_ot_functional *) params;

  l3 = dft_common_lwl3(otf->mass, otf->temp);
  return -rhop * GRID_AUKB * otf->temp * (1.0 - LOG(rhop*l3 + 1E-6*EPS));
}

/*
 * @FUNC{dft_common_bose_idealgas_energy, "Ideal bose gas: energy per volume"}
 * @DESC{"Ideal bose gas. $NVT$ free energy / volume (i.e., $A/V$, $A = U - TS$)"}
 * @ARG1{REAL rhop, "Gas density"}
 * @RVAL{REAL, "Returns free energy / volume"}
 *
 */

EXPORT REAL dft_common_bose_idealgas_energy(REAL rhop, void *params) {

  REAL z, l3;
  dft_ot_functional *otf = (dft_ot_functional *) params;

  l3 = dft_common_lwl3(otf->mass, otf->temp);
  z = dft_common_fit_z(rhop * l3);
  return (otf->c4 * GRID_AUKB * otf->temp * (rhop * LOG(z) - dft_common_fit_g52(z) / l3));
}

/*
 * @FUNC{dft_common_bose_idealgas_dEdRho, "Ideal bose gas: $dE/d\rho$"}
 * @DESC{"Ideal bose gas. Derivative of energy / volume with respect to $\rho$"}
 * @ARG1{REAL rhop, "Gas density"}
 *
 * Returns free energy / volume derivative.
 */

/* Matches the difference of dft_common_idealgas_energy_op() */
// #define USE_DIFFERENCE

EXPORT REAL dft_common_bose_idealgas_dEdRho(REAL rhop, void *params) {

#ifdef USE_DIFFERENCE
#define DIFF_EPS 1E-12
  return (dft_common_bose_idealgas_energy(rhop + DIFF_EPS, params) - dft_common_bose_idealgas_energy(rhop - DIFF_EPS, params)) / (2.0 * DIFF_EPS);
#else
  REAL l3, z0, rl3, g12, g32;
  REAL tmp;
  dft_ot_functional *otf = (dft_ot_functional *) params;

  l3 = dft_common_lwl3(otf->mass, otf->temp);
  rl3 = rhop * l3;
  tmp = dft_common_fit_g32(1.0);
  if(rl3 >= tmp) return -otf->c4 * GRID_AUKB * (otf->temp / l3) * tmp;
  z0 = dft_common_fit_z(rl3);    /* check these fits */
  g12 = dft_common_fit_g12(z0);
  g32 = dft_common_fit_g32(z0);

  return otf->c4 * GRID_AUKB * otf->temp * (LOG(z0) + rl3 / g12 - g32 / g12);
#endif
}

/*
 * @FUNC{dft_common_read_pot, "Read potential data from ASCII file"}
 * @DESC{Read 1-D potential from file (ASCII). Equidistant steps for potential required."}
 * @ARG1{char *file, "Filename"}
 * @ARG2{dft_extpot *pot, "Place the potential in this structure"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT void dft_common_read_pot(char *file, dft_extpot *pot) {

  FILE *fp;
  INT i;
  REAL b = 0.0, s = 0.0, x = 0.0, px;

  if(!(fp = fopen(file, "r"))) {
    fprintf(stderr, "libdft: Can't open %s.\n", file);
    exit(1);
  }

  for (i = 0; i < DFT_MAX_POTENTIAL_POINTS; i++) {
    px = x;
    if(fscanf(fp, " " FMT_R " " FMT_R, &x, &(pot->points[i])) != 2) break;
    if(i == 0) {
      b = x;
      continue;
    }
    if(i == 1) {
      s = x - b;
      if(s < 0.0) {
	fprintf(stderr, "libdft: Potential distance in wrong order.\n");
	exit(1);
      }
      continue;
    }
    if(FABS(x - px - s) > s/10.0) {
      fprintf(stderr, "libdft: Potential step not constant.\n");
      exit(1);
    }
  }
  fclose(fp);
  pot->begin = b;
  pot->length = i;
  pot->step = s;
}

/*
 * @FUNC{dft_common_extpot, "Map 1-D potential to 3-D grid"}
 * @DESC{"External potential suitable for grid map() routines"}
 * @ARG1{dft_extpot_set *arg, "Potential"}
 * @ARG2{REAL x, "X-coordinate"}
 * @ARG3{REAL y, "Y-coordinate"}
 * @ARG4{REAL z, "Z-coordinate"}
 * @RVAL{REAL, "Returns the potential value"}
 */

REAL dft_common_extpot(void *arg, REAL x, REAL y, REAL z) {
   
  REAL r, px, py, pz, tmp;
  INT i;
  dft_extpot_set *set = (dft_extpot_set *) arg;
  REAL *pot_x = set->x->points, *pot_y = set->y->points, *pot_z = set->z->points;
  REAL bx = set->x->begin, by = set->y->begin, bz = set->z->begin;
  REAL sx = set->x->step, sy = set->y->step, sz = set->z->step;
  INT lx = set->x->length, ly = set->y->length, lz = set->z->length;
  char aver = set->average;
  REAL theta0 = set->theta0, theta, cos_theta, sin_theta;
  REAL phi0 = set->phi0, phi, cos_phi, sin_phi;
  REAL x0 = set->x0, y0 = set->y0, z0 = set->z0;

  /* shift origin */
  x -= x0;
  y -= y0;
  z -= z0;
  r = SQRT(x * x + y * y + z * z);
  
  /* x */
  i = (INT) ((r - bx) / sx);
  if(i < 0) {
    /*    px = pot_x[0] + 1.0E-3 * pot_x[0] * (REAL) labs(i); */
    px = pot_x[0];
  } else if (i > lx-1) {
    px =  0.0;    /* was pot_x[lx-1] */
  } else px = pot_x[i];

  /* y */
  i = (INT) ((r - by) / sy);
  if(i < 0) {
    /*    py = pot_y[0] + 1.0E-3 * pot_y[0] * (REAL) labs(i); */
    py = pot_y[0];
  } else if (i > ly-1) {
    py =  0.0; /* was pot_y[ly-1] */
  } else py = pot_y[i];

  /* z */
  i = (INT) ((r - bz) / sz);
  if(i < 0) {
    /*    pz = pot_z[0] + 1.0E-3 * pot_z[0] * (REAL) labs(i); */
    pz = pot_z[0];
  } else if (i > lz-1) {
    pz = 0.0;  /* was pot_z[lz-1] */
  } else pz = pot_z[i];

  switch(aver) {
  case 0: /* no averaging */
    break;
  case 1: /* XY average */
    tmp = (px + py) / 2.0;
    px = py = tmp;
    break;
  case 2: /* YZ average */
    tmp = (py + pz) / 2.0;
    py = pz = tmp;
    break;
  case 3: /* XZ average */
    tmp = (px + pz) / 2.0;
    px = pz = tmp;
    break;
  case 4: /* XYZ average */
    return (px + py + pz) / 3.0;
  }

  theta = ACOS(z / (r + 1E-3)) - theta0;
  phi = ATAN(y / (x + 1E-3)) - phi0;
  sin_theta = SIN(theta);
  sin_theta *= sin_theta;
  cos_theta = COS(theta);
  cos_theta *= cos_theta;
  sin_phi = SIN(phi);
  sin_phi *= sin_phi;
  cos_phi = COS(phi);
  cos_phi *= cos_phi;
  return px * sin_theta * cos_phi + py * sin_theta * sin_phi + pz * cos_theta;
  
  /* simpler way if no titling of the potential is needed */
  //if(r==0.)
  //	  return (px + py + pz) / 3.0 ;
  //return ( px * x * x + py * y * y + pz * z * z ) / ( r * r ) ;
}

/*
 * @FUNC{dft_common_potential_map, "Map potential files onto real grid"}
 * @DESC{"Map a potential given by an ascii file onto a grid"}
 * @ARG1{char average, "0 = no averaging, 1 = average XY, 2 = average YZ, 3 = average XZ, 4 = average XYZ"}
 * @ARG2{char *file_x, "File name for potential along x axis"}
 * @ARG3{char *file_y, "File name for potential along y axis"}
 * @ARG4{char *file_z, "File name for potential along z axis"}
 * @ARG5{rgrid *potential, "Output potential grid"}
 * @ARG6{REAL theta0, "Rotation angle theta for the potential"}
 * @ARG7{REAL phi0, "Rotation angle phi for the potential"}
 * @ARG8{REAL x0, "New origin x"}
 * @ARG9{REAL y0, "New origin y"}
 * @ARG10{REAL z0, "New origin z"}
 * @RVAL{void, "No return value"}
 *
 */
	
EXPORT void dft_common_potential_map(char average, char *filex, char *filey, char *filez, rgrid *potential, REAL theta0, REAL phi0, REAL x0, REAL y0, REAL z0) {

  dft_extpot x, y, z;
  dft_extpot_set set;
  
  set.x = &x;
  set.y = &y;
  set.z = &z;
  set.x0 = x0;
  set.y0 = y0;
  set.z0 = z0;
  set.average = average;
  set.theta0 = theta0;
  set.phi0 = phi0;

  fprintf(stderr, "libdft: Mapping potential file with x = %s, y = %s, z = %s. Average = %d - ", filex, filey, filez, average);
  dft_common_read_pot(filex, &x);
  dft_common_read_pot(filey, &y);
  dft_common_read_pot(filez, &z);
  rgrid_map(potential, dft_common_extpot, (void *) &set);
  fprintf(stderr, "done.\n");
}

/*
 * @FUNC{dft_common_potential_smooth_map_rotate_shift, "Map potential files onto real grid (with smoothing)"}
 * @DESC{"Map a potential given by an ascii file onto a grid (with smoothing)"}
 * @ARG1{char average, "0 = no averaging, 1 = average XY, 2 = average YZ, 3 = average XZ, 4 = average XYZ"}
 * @ARG2{char *file_x, "File name for potential along x axis"}
 * @ARG3{char *file_y, "File name for potential along y axis"}
 * @ARG4{char *file_z, "File name for potential along z axis"}
 * @ARG5{rgrid *potential, "Output potential grid"}
 * @ARG6{REAL theta0, "Rotation angle theta for the potential"}
 * @ARG7{REAL phi0, "Rotation angle phi for the potential"}
 * @ARG8{REAL x0, "New origin x"}
 * @ARG9{REAL y0, "New origin y"}
 * @ARG10{REAL z0, "New origin z"}
 * @RVAL{void, "No return value"}
 *
 */
	
EXPORT void dft_common_potential_smooth_map(char average, char *filex, char *filey, char *filez, rgrid *potential, REAL theta0, REAL phi0, REAL x0, REAL y0, REAL z0) {

  dft_extpot x, y, z;
  dft_extpot_set set;
  
  set.x = &x;
  set.y = &y;
  set.z = &z;
  set.x0 = x0;
  set.y0 = y0;
  set.z0 = z0;
  set.average = average;
  set.theta0 = theta0;
  set.phi0 = phi0;

  fprintf(stderr, "libdft: Mapping potential file with x = %s, y = %s, z = %s. Average = %d - ", filex, filey, filez, average);
  dft_common_read_pot(filex, &x);
  dft_common_read_pot(filey, &y);
  dft_common_read_pot(filez, &z);
  rgrid_smooth_map(potential, dft_common_extpot, (void *) &set, 10); /* TODO allow changing this */
  fprintf(stderr, "done.\n");
}

/*
 *
 * This is an auxiliary routine and should not be called by users (hence not exported).
 *
 */

static rgrid *dft_common_pot_interpolate_read(INT n, char **files) {

  REAL r, pot_begin, pot_step;
  INT i, j, k;
  INT nr, nphi, pot_length;
  rgrid *cyl;
  dft_extpot pot;

  if (n < 0) {
    fprintf(stderr, "libdft: ni or n negative (dft_common_pot_interpolate()).\n");
    exit(1);
  }

  /* snoop for the potential parameters */
  dft_common_read_pot(files[0], &pot);
  pot_begin = pot.begin;
  pot_step = pot.step;
  pot_length = pot.length;
  nr = pot.length + (INT) (pot_begin / pot_step);   /* enough space for the potential + the empty core, which is set to constant */
  nphi = n; /* Only [0,Pi] stored - ]Pi,2Pi[ by symmetry */

  cyl = rgrid_alloc(nr, nphi, 1, pot.step, RGRID_PERIODIC_BOUNDARY, NULL, "cyl");
  
  /* For each direction */
  for (j = 0; j < n; j++) {
    dft_common_read_pot(files[j], &pot);
    if (pot.begin != pot_begin || pot.step != pot_step || pot.length != pot_length) {
      fprintf(stderr, "libdft: Inconsistent potentials in dft_common_pot_interpolate().\n");
      exit(1);
    }
    /* map the current direction on the grid */
    for (i = k = 0; i < nr; i++) {
      r = pot_step * (REAL) i;
      if (r < pot_begin)
        rgrid_value_to_index(cyl, i, j, 0, pot.points[0]);
      else
        rgrid_value_to_index(cyl, i, j, 0, pot.points[k++]);
    }
  }

  return cyl;
}

/*
 *
 * This is an auxiliary routine and should not be called by users (hence not exported).
 *
 */

static inline REAL eval_value_at_index_cyl(rgrid *grid, INT i, INT j, INT k) {

  INT nr = grid->nx, nphi = grid->ny, nz = grid->nz;

  if (i < 0 || j < 0 || k < 0 || i >= nr || j >= nphi || k >= nz) {
    if(i < 0) {
      i = ABS(i);
      j += nphi/2;
    }
    if(i >= nr) i = nr - 1;
    j %= nphi;
    if (j < 0) j = nphi + j;
    k %= nz;
    if (k < 0) k = nz + k;
  }
  return rgrid_value_at_index(grid, i, j, k);
//  return grid->value[(i*nphi + j)*nz + k];
}

/*
 *
 * This is an auxiliary routine and should not be called by users (hence not exported).
 *
 */

static inline REAL eval_value_cyl(rgrid *grid, REAL r, REAL phi, REAL z) {

  REAL f000, f100, f010, f001, f110, f101, f011, f111;
  INT i, j, k, nphi = grid->ny;
  REAL omz, omr, omphi, step = grid->step, step_phi = 2.0 * M_PI / (REAL) nphi;
  
  /* i to index and 0 <= r < 1 */
  r = r / step;
  i = (INT) r;
  r = r - (REAL) i;
  
  /* j to index and 0 <= phi < 1 */
  phi = phi / step_phi;
  j = (INT) phi;
  phi = phi - (REAL) j;

  /* k to index and 0 <= z < 1 */
  k = (INT) (z /= step);
  if (z < 0) k--;
  z -= (REAL) k;
  k += grid->nz / 2;

  /*
   * Linear interpolation 
   *
   * f(r,phi,z) = (1-r) (1-phi) (1-z) f(0,0,0) + r (1-phi) (1-z) f(1,0,0) + (1-r) phi (1-z) f(0,1,0) + (1-r) (1-phi) z f(0,0,1) 
   *            + r       phi   (1-z) f(1,1,0) + r (1-phi)   z   f(1,0,1) + (1-r) phi   z   f(0,1,1) +   r     phi   z f(1,1,1)
   */ 
  f000 = eval_value_at_index_cyl(grid, i, j, k);
  f100 = eval_value_at_index_cyl(grid, i+1, j, k);
  f010 = eval_value_at_index_cyl(grid, i, j+1, k);
  f001 = eval_value_at_index_cyl(grid, i, j, k+1);
  f110 = eval_value_at_index_cyl(grid, i+1, j+1, k);
  f101 = eval_value_at_index_cyl(grid, i+1, j, k+1);
  f011 = eval_value_at_index_cyl(grid, i, j+1, k+1);
  f111 = eval_value_at_index_cyl(grid, i+1, j+1, k+1);
  
  omr = 1.0 - r;
  omphi = 1.0 - phi;
  omz = 1.0 - z;

  return omr * omphi * omz * f000 + r * omphi * omz * f100 + omr * phi * omz * f010 + omr * omphi * z * f001
    + r * phi * omz * f110 + r * omphi * z * f101 + omr * phi * z * f011 + r * phi * z * f111;
}

/*
 *
 * This is an auxiliary routine and should not be called by users (hence not exported).
 *
 */

static inline REAL dft_common_interpolate_value(rgrid *grid, REAL r, REAL phi, REAL *x, REAL *y) {

  REAL f0, f1;
  INT i, j, nphi = grid->ny;
  REAL step = grid->step, step_phi = M_PI / (REAL) (nphi-1), err_estim;
  
  /* i to index and 0 <= r < 1 */
  r = r / step;
  i = (INT) r;
  r = r - (REAL) i;

  /*
   * Polynomial along phi
   *
   */

  if(phi > M_PI) phi = 2.0 * M_PI - phi;

  /* Evaluate f(0, phi) */
  for (j = 0; j < nphi; j++) {
    x[j] = step_phi * (REAL) j;
    y[j] = eval_value_at_index_cyl(grid, i, j, 0);
  }
  f0 = grid_polynomial_interpolate(x, y, nphi, phi, &err_estim);

  /* Evaluate f(1, phi) */
  for (j = 0; j < nphi; j++) {
    x[j] = step_phi * (REAL) j;
    y[j] = eval_value_at_index_cyl(grid, i+1, j, 0);
  }
  f1 = grid_polynomial_interpolate(x, y, nphi, phi, &err_estim);

  /*
   * Linear interpolation for r
   *
   * f(r,phi) = (1-r) f(0, phi) + r f(1,phi)
   */ 

  return (1.0 - r) * f0 + r * f1;
}

/*
 *
 * This is an auxiliary routine and should not be called by users (hence not exported).
 *
 * Similar to eval_value_cyl but includes spline interpolation for phi and z = 0
 *
 */

static inline REAL dft_common_spline_value(rgrid *grid, REAL r, REAL phi, REAL *x, REAL *y, REAL *y2) {

  REAL f0, f1;
  INT i, j, nphi = grid->ny;
  REAL step = grid->step, step_phi = M_PI / (REAL) (nphi-1);
  
  /* i to index and 0 <= r < 1 */
  r = r / step;
  i = (INT) r;
  r = r - (REAL) i;
  
  /*
   * Polynomial along phi
   *
   */

  if(phi > M_PI) phi = 2.0 * M_PI - phi;

  /* Evaluate f(0, phi) */
  for (j = 0; j < nphi; j++) {
    x[j] = step_phi * (REAL) j;
    y[j] = eval_value_at_index_cyl(grid, i, j, 0);
  }
  grid_spline_ypp(x, y, nphi, 0.0 , 0.0 , y2 ) ;
  f0 = grid_spline_interpolate(x, y, y2, nphi, phi);

  /* Evaluate f(1, phi) */
  for (j = 0; j < nphi; j++) {
    x[j] = step_phi * (REAL) j;
    y[j] = eval_value_at_index_cyl(grid, i+1, j, 0);
  }
  grid_spline_ypp(x, y, nphi, 0.0 , 0.0 , y2 ) ;
  f1 = grid_spline_interpolate(x, y, y2, nphi, phi);

  /*
   * Linear interpolation for r
   *
   * f(r,phi) = (1-r) f(0, phi) + r f(1,phi)
   */ 

  return (1.0 - r) * f0 + r * f1;
}

/*
 * @FUNC{dft_common_pot_interpolate, "Interpolate 3-D potential from 1-D cuts"}
 * @DESC{"Produce interpolated 3-D potential energy surface from n 1-D cuts along 
          phi = 0, pi/n, ..., pi directions (symmetric for ]Pi,2Pi[).
          Requires cylindrical symmetry for the overall potential (i.e., linear molecule).
          All potentials must have the same range, steps, number of points"}
 * @ARG1{INT n, "Number of potentials along n different angles"}
 * @ARG2{char **file, "Array of strings for file names containing the potentials"}
 * @ARG3{rgrid *out, "3-D cartesian grid containing the angular interpolated potential data."}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT void dft_common_pot_interpolate(INT n, char **files, rgrid *out) {

  REAL x, y, z, r, phi, step = out->step, x0 = out->x0, y0 = out->y0, z0 = out->z0;
  INT nx = out->nx, ny = out->ny, nz = out->nz, i, j, k;
  REAL *tmp1, *tmp2;
  rgrid *cyl;

  if(!(tmp1 = (REAL *) malloc(sizeof(REAL) * (size_t) n)) || !(tmp2 = (REAL *) malloc(sizeof(REAL) * (size_t) n))) {
    fprintf(stderr, "libgrid: Out of memory in dft_common_interpolate().\n");
    exit(1);
  }

  cyl = dft_common_pot_interpolate_read(n, files);    /* allocates cyl */
  /* map cyl_large to cart */
  for (i = 0; i < nx; i++) {
    REAL x2;
    x = ((REAL) (i - nx/2)) * step - x0;
    x2 = x * x;
    for (j = 0; j < ny; j++) {
      REAL y2;
      y = ((REAL) (j - ny/2)) * step - y0;
      y2 = y * y;
      for (k = 0; k < nz; k++) {
	z = ((REAL) (k - nz/2)) * step - z0;
	r = SQRT(x2 + y2 + z * z);
	phi = M_PI - ATAN2(SQRT(x2 + y2), -z);
	rgrid_value_to_index(out, i, j, k, dft_common_interpolate_value(cyl, r, phi, tmp1, tmp2));
      }
    }
  }
  rgrid_free(cyl);
  free(tmp1);
  free(tmp2);
}

/*
 * @FUNC{dft_common_pot_spline, "Interpolate 3-D potential from 1-D cuts (spline)"}
 * @DESC{"Produce interpolated spline 3-D potential energy surface from n 1-D cuts along 
          phi = 0, pi/n, ..., pi directions (symmetric for ]Pi,2Pi[).
          Requires cylindrical symmetry for the overall potential (i.e., linear molecule).
          All potentials must have the same range, steps, number of points"}
 * @ARG1{INT n, "Number of potentials along n different angles"}
 * @ARG2{char **files, "Array of strings for file names containing the potentials"}
 * @ARG3{rgird *out, "3-D cartesian grid containing the angular interpolated potential data"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT void dft_common_pot_spline(INT n, char **files, rgrid *out) {

  REAL x, y, z, r, phi, step = out->step, x0 = out->x0, y0 = out->y0, z0 = out->z0;
  INT nx = out->nx, ny = out->ny, nz = out->nz, i, j, k;
  REAL *tmp1, *tmp2 , *tmp3;
  rgrid *cyl;

  if(!(tmp1 = (REAL *) malloc(sizeof(REAL) * (size_t) n)) || !(tmp2 = (REAL *) malloc(sizeof(REAL) * (size_t) n)) || !(tmp3 = (REAL *) malloc(sizeof(REAL) * (size_t) n)) ) {
    fprintf(stderr, "libgrid: Out of memory in dft_common_interpolate().\n");
    exit(1);
  }

  cyl = dft_common_pot_interpolate_read(n, files);    /* allocates cyl */
  /* map cyl_large to cart */
  for (i = 0; i < nx; i++) {
    REAL x2;
    x = ((REAL) (i - nx/2)) * step - x0;
    x2 = x * x;
    for (j = 0; j < ny; j++) {
      REAL y2;
      y = ((REAL) (j - ny/2)) * step - y0;
      y2 = y * y;
      for (k = 0; k < nz; k++) {
	z = ((REAL) (k - nz/2)) * step - z0;
	r = SQRT(x2 + y2 + z * z);
	phi = M_PI - ATAN2(SQRT(x2 + y2), -z);
	rgrid_value_to_index(out, i, j, k, dft_common_spline_value(cyl, r, phi, tmp1, tmp2, tmp3));
      }
    }
  }
  rgrid_free(cyl);
  free(tmp1);
  free(tmp2);
}

/*
 * @FUNC{dft_common_pot_angularderiv, "Second derivative with respect to angle"}
 * @DESC{"Compute the numerical second derivative with respect to the angle of the potential.
          Computed via second-order finite difference formula:\\
          f''(i) = h**2 * (f(i-1) - 2*f(i) + f(i+1))\\
          n this case h = 2*pi / nx"}
 * @ARG1{INT n, "Number of potential files"}
 * @ARG2{char **files, "Array of file names"}
 * @ARG3{rgrid *out, "Output grid"}
 * @RVAL{void, "No return value"}
 *
 * TODO: There's a better way to do this now that spline is implemented.
 * spline generates the second derivative as by-product.
 *
 */

EXPORT void dft_common_pot_angularderiv(INT n, char **files, rgrid *out) {

  REAL x, y, z, r, phi, step_cyl, step = out->step, x0 = out->x0, y0 = out->y0, z0 = out->z0;
  INT nx = out->nx, ny = out->ny, nz = out->nz, i, j, k, nphi , nr;
  rgrid *cyl_pot, *cyl_k;

  cyl_pot = dft_common_pot_interpolate_read(n, files);    /* cyl_pot allocated */
  nr = cyl_pot->nx;
  nphi = cyl_pot->ny;
  step_cyl = cyl_pot->step;

  cyl_k = rgrid_alloc(nr, nphi, 1, step_cyl, RGRID_PERIODIC_BOUNDARY, NULL, "cyl_k"); 

  /* second derivative respect to theta */
  REAL inv_step2 = ((REAL) (nphi * nphi)) / (2.0 * 2.0 * M_PI * M_PI);
  for (i = 0; i < nr; i++) {
    for (j = 0; j < nphi ; j++) {
              rgrid_value_to_index(cyl_k, i, j, 0, 
                          inv_step2 * (
			         rgrid_value_at_index(cyl_pot, i, j-1, 0)
			  -2.0 * rgrid_value_at_index(cyl_pot, i, j  , 0)
			  +      rgrid_value_at_index(cyl_pot, i, j+1, 0)));
    }
  }

  /* map cyl_k to cart */
  for (i = 0; i < nx; i++) {
    REAL x2;
    x = ((REAL) (i - nx/2)) * step - x0;
    x2 = x * x;
    for (j = 0; j < ny; j++) {
      REAL y2;
      y = ((REAL) (j - ny/2)) * step - y0;
      y2 = y * y;
      for (k = 0; k < nz; k++) {
	z = ((REAL) (k - nz/2)) * step - z0;
	r = SQRT(x2 + y2 + z * z);
	phi = M_PI - ATAN2(SQRT(x2 + y2), -z);
	rgrid_value_to_index(out, i, j, k, eval_value_cyl(cyl_k, r, phi, 0.0));
      }
    }
  }

  rgrid_free(cyl_pot);
  rgrid_free(cyl_k);
}

/*
 * @FUNC{dft_common_pot_average, "Spherically averaged 1-D potential from multiple 1-D cuts"}
 * @DESC{"Spherically averaged 1-D potential energy surface from n 1-D cuts along phi = 0, pi/n, ..., pi directions.
         Note that this is different than in dft_common_pot_interpolate where you must give potentials for the whole 2pi.
         Requires cylindrical symmetry for the overall potential (i.e., linear molecule).
         All potentials must have the same range, steps, number of points"}
 * @ARG1{INT n, "Number of potentials along n different angles"}
 * @ARG2{char **files, "Array of strings for file names containing the potentials"}
 * @ARG3{rgrid *out, "3-D grid containing the angular interpolated potential grid"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT void dft_common_pot_average(INT n, char **files, rgrid *out) {
  
  REAL x, y, z, r, step = out->step, pot_begin, pot_step, x0 = out->x0, y0 = out->y0, z0 = out->z0;
  INT nx = out->nx, ny = out->ny, nz = out->nz, i, j, k, nr, pot_length;
  dft_extpot pot;
  dft_extpot pot_ave;
  REAL angular_weight;
  
  /* snoop for the potential parameters */
  dft_common_read_pot(files[0], &pot_ave);
  pot_begin  = pot_ave.begin;
  pot_step   = pot_ave.step;
  pot_length = pot_ave.length;
  /* Erase the values in pot_ave */
  for(k = 0; k < pot_length; k++)
    pot_ave.points[k] = 0.0;
  nr = pot_length + (INT) (pot_begin / pot_step);   /* enough space for the potential + the empty core, which is set to constant */

  /* Construct the 1D potential averaging all directions */
  for (j = 0; j < n; j++) {
    dft_common_read_pot(files[j], &pot);
    if (pot.begin != pot_begin || pot.step != pot_step || pot.length != pot_length) {
      fprintf(stderr, "libdft: Inconsistent potentials in dft_common_pot_interpolate().\n");
      exit(1);
    }
    if(j == 0 || j == n-1)
      angular_weight = 0.25 - (1.0 + ((REAL) (n-1)) * ((REAL) (n-1)) * COS(M_PI/((REAL) (n-1)))) / (REAL) (4*n*(n-2));
    else
      angular_weight = (((REAL) (n-1)) * ((REAL) (n-1)) * SIN(M_PI/(REAL) (n-1)) * SIN(M_PI*((REAL) j)/(REAL) (n-1))) / (REAL) (2 * n * (n-2));

    for(k = 0; k < pot.length; k++)
      pot_ave.points[k] += angular_weight * pot.points[k];
  }

  /* Map the 1D pot to cartesian grid */
  for (i = 0; i < nx; i++) {
    x = ((REAL) (i - nx/2)) * step - x0;
    for (j = 0; j < ny; j++) {
      y = ((REAL) (j - ny/2)) * step - y0;
      for (k = 0; k < nz; k++) {
	z = ((REAL) (k - nz/2)) * step - z0;
	r = SQRT(x * x + y * y + z * z);
        if (r < pot_begin)
	  rgrid_value_to_index(out, i, j, k, pot_ave.points[0]);
	else {
	  nr = (INT) ((r - pot_begin) / pot_step);
	  if(nr < pot_ave.length)
	    rgrid_value_to_index(out, i, j, k, pot_ave.points[nr]);
	  else
	    rgrid_value_to_index(out, i, j, k, 0.0);
	}
      }
    }
  }
}

/*
 * @FUNC{dft_common_planewave, "Plane wave function suitable for cgrid_map()"}
 * @DESC{"Plane wave suitable for cgrid_map() function. arg specifies the plane wave params according to dft_plane_wave structure"}
 * @ARG1{void *arg, "Plane wave parameters (dft_plane_wave *)"}
 * @ARG2{REAL x, "X coordinate"}
 * @ARG3{REAL y, "Y coordinate"}
 * @ARG4{REAL z, "Z coordinate"}
 * @RVAL{REAL, "Returns the plane wave function value at (x, y, z)"}
 *
 */

EXPORT REAL complex dft_common_planewave(void *arg, REAL x, REAL y, REAL z) {

  REAL kx = ((dft_plane_wave *) arg)->kx;
  REAL ky = ((dft_plane_wave *) arg)->ky;
  REAL kz = ((dft_plane_wave *) arg)->kz;
  REAL a = ((dft_plane_wave *) arg)->a;
  REAL psi = SQRT(((dft_plane_wave *) arg)->rho);
  
//  return psi + 0.5 * a * psi * (CEXP(I * (kx * x + ky * y + kz * z)) + CEXP(-I * (kx * x + ky * y + kz * z)));
  return psi + a * psi * COS(kx * x + ky * y + kz * z);
}

/*
 * @FUNC{dft_common_vortex_x, "Feynman-Onsager vortex line potential along x-axis"}
 * @DESC{"Vortex line potential using Feynman-Onsager ansatz along x-axis. Note that this does not produce phase circulation!"}
 * @ARG1{void *param, "Mass (REAL *)"}
 * @ARG2{REAL x, "X-coordinate"}
 * @ARG3{REAL y, "y-coordinate"}
 * @ARG4{REAL z, "z-coordinate"}
 * @RVAL{REAL, "Potential value at (x,y,z)"}
 *
 */

/* Cutoff for vortex core (to avoid NaN) */
#define R_M 0.05

EXPORT REAL dft_common_vortex_x(void *param, REAL x, REAL y, REAL z) {

  REAL rp2 = y * y + z * z;
  REAL *mass = (REAL *) param;

  if(rp2 < R_M * R_M) rp2 = R_M * R_M;
  return 1.0 / (2.0 * *mass * rp2);
}

/*
 * @FUNC{dft_common_vortex_y, "Feynman-Onsager vortex line potential along y-axis"}
 * @DESC{"Vortex line potential using Feynman-Onsager ansatz along y-axis. Note that this does not produce phase circulation!"}
 * @ARG1{void *param, "Mass (REAL *)"}
 * @ARG2{REAL x, "X-coordinate"}
 * @ARG3{REAL y, "y-coordinate"}
 * @ARG4{REAL z, "z-coordinate"}
 * @RVAL{REAL, "Potential value at (x,y,z)"}
 *
 */

EXPORT REAL dft_common_vortex_y(void *param, REAL x, REAL y, REAL z) {

  REAL rp2 = x * x + z * z;
  REAL *mass = (REAL *) param;

  if(rp2 < R_M * R_M) rp2 = R_M * R_M;
  return 1.0 / (2.0 * *mass * rp2);
}

/*
 * @FUNC{dft_common_vortex_z, "Feynman-Onsager vortex line potential along z-axis"}
 * @DESC{"Vortex line potential using Feynman-Onsager ansatz along z-axis. Note that this does not produce phase circulation!"}
 * @ARG1{void *param, "Mass (REAL *)"}
 * @ARG2{REAL x, "X-coordinate"}
 * @ARG3{REAL y, "y-coordinate"}
 * @ARG4{REAL z, "z-coordinate"}
 * @RVAL{REAL, "Potential value at (x,y,z)"}
 *
 */

EXPORT REAL dft_common_vortex_z(void *param, REAL x, REAL y, REAL z) {

  REAL rp2 = x * x + y * y;
  REAL *mass = (REAL *) param;

  if(rp2 < R_M * R_M) rp2 = R_M * R_M;
  return 1.0 / (2.0 * *mass * rp2);
}
