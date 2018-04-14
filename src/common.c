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
 * pow() function that uses integer exponents (fast): x^y
 *
 * x = variable x above (REAL).
 * y = exponent (INT).
 *
 * TODO: Change things back everywhere so that ipow is mostly used (except for DR).
 *
 * Returns x^y.
 *
 */

EXPORT inline REAL dft_common_ipow(REAL x, INT y) {
  
  INT i;
  REAL v = 1.0;

  for (i = 0; i < y; i++)
    v *= x;

  return v;
}

/*
 * Lennard-Jones function (r2 = r^2).
 *
 * r2  = r^2 (REAL).
 * sig = sigma in LJ potential (REAL).
 * eps = epsilon in LJ potential (REAL).
 *
 * Returns Lennard-Jones potential at r^2.
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
 * Lennard-Jones potential to be used with grid map() routines (3D)
 * Note that the LJ potential has zero core when r < h.
 *
 * The potential paramegers are passed in arg (ot_common_lh data type).
 *
 * arg = pointer to dft_common_lj structure (void *).
 * x   = x-coordinate (REAL).
 * y   = y-coordinate (REAL).
 * z   = z-coordinate (REAL).
 *
 * Returns Lennard-Jones potential at (x,y,z).
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
 * Lennard-Jones potential with smoothed core to be used with grid map()
 * routines. Parameters passed in arg (see the regular LJ above).
 *
 * The potential paramegers are passed in arg (ot_common_lh data type).
 *
 * arg = pointer to dft_common_lj structure (void *).
 * x   = x-coordinate (REAL).
 * y   = y-coordinate (REAL).
 * z   = z-coordinate (REAL).
 *
 * Returns Lennard-Jones potential at (x,y,z).
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
 * Spherical average function to be used with grid map() routines (3D).
 * The sphere radius is passed in arg.
 *
 * arg = pointer to the radius of the sphere (REAL *).
 * x   = x-coordinate (REAL).
 * y   = y-coordinate (REAL).
 * z   = z-coordinate (REAL).
 *
 * Returns the value of the spherical average function at (x, y, z).
 *
 */

EXPORT inline REAL dft_common_spherical_avg(void *arg, REAL x, REAL y, REAL z) {

  REAL h = *((REAL *) arg), h2 = h * h;

  if (x*x + y*y + z*z <= h2)
    return 3.0 / (4.0 * M_PI * h * h2);
  return 0.0;
}


/*
 * Spherical average function IN MOMENTUM SPACE to be used with grid map() routines (3D).
 * The sphere radius is passed in arg.
 *
 * arg = pointer to the radius of the sphere, hk (REAL *).
 * kx   = kx-coordinate (REAL).
 * ky   = ky-coordinate (REAL).
 * kz   = kz-coordinate (REAL).
 *
 * Returns the value of the spherical average function at (kx, ky, kz).
 *
 */

EXPORT inline REAL dft_common_spherical_avg_k(void *arg, REAL kx, REAL ky, REAL kz) {

  REAL hk = *((REAL *) arg) * SQRT(kx * kx + ky * ky + kz * kz);

  if(hk < 1.e-5) return 1.0 - 0.1 * hk * hk; /* second order Taylor expansion */
  return 3.0 * (SIN(hk) - hk * COS(hk)) / (hk * hk * hk);
}

/*
 * Gaussian function to be used with grid map() functions (3D).
 * The gaussian is centered at (0,0,0) and width is given in arg.
 *
 * arg = Inverse width of the gaussian function (REAL *).
 * x   = x-coordinate (REAL).
 * y   = y-coordinate (REAL).
 * z   = z-coordinate (REAL).
 *
 * Returns the value of the value of the gaussian function at (x, y, z).
 * 
 */ 

EXPORT inline REAL dft_common_gaussian(void *arg, REAL x, REAL y, REAL z) {

  REAL inv_width = *((REAL *) arg);
  REAL norm = 0.5 * M_2_SQRTPI * inv_width;

  norm = norm * norm * norm;
  return norm * EXP(-(x * x + y * y + z * z) * inv_width * inv_width);
}

/*
 * Complex Gaussian function to be used with cgrid map() functions (3D).
 * The gaussian is centered at (0,0,0) and width is given in arg.
 *
 * arg = Inverse width of the gaussian function (REAL *).
 * x   = x-coordinate (REAL).
 * y   = y-coordinate (REAL).
 * z   = z-coordinate (REAL).
 *
 * Returns the value of the value of the gaussian function at (x, y, z).
 * 
 */ 

EXPORT inline REAL complex dft_common_cgaussian(void *arg, REAL x, REAL y, REAL z) {

  REAL inv_width = *((REAL *) arg);
  REAL norm = 0.5 * M_2_SQRTPI * inv_width;

  norm = norm * norm * norm;
  return (REAL complex) norm * EXP(-(x * x + y * y + z * z) * inv_width * inv_width);
}

/*
 * Return thermal wavelength for a particle with mass "mass" at temperature
 * "temp".
 *
 * mass = Mass of the particle (REAL).
 * temp = Temperature (REAL).
 *
 * Returns thermal wavelength.
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
 * Evaluate g_s(z) (polylog).
 *
 * z = argument of polylog (REAL).
 * s = argument of polylog (REAL).
 *
 * Returns the polylog.
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
 * Evaluate g_{1/2}(z) using polynomial fit.
 *
 * z = argument of polylog (REAL).
 *
 * Returns the polylog value.
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
 * Evaluate g_{3/2}(z) (polylog)
 *
 * z = argument to polylog (REAL).
 *
 * Returns the polylog. 
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
 * Evaluate g_{5/2}(z) (polylog)
 *
 * z = Argument of polylog (REAL).
 *
 * Returns the polylog.
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
 * Evaluate z(rho, T) (polylog). Invert g_{3/2}.
 *
 * val = Value where g_{3/2} is to be inverted.
 *
 * Returns the inverted value.
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
 * Classical ideal gas. Free energy / volume derivative with respect to rho: d(A/V) / drho
 *
 * rhop = Gas density (REAL).
 *
 * Returns the free energy / volume derivative.
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
 * Classical ideal gas. NVT free energy / volume (i.e., A/V, A = U - TS).
 *
 * rhop = Gas density (REAL).
 *
 * Returns free energy / volume.
 *
 */

EXPORT REAL dft_common_classical_idealgas_energy(REAL rhop, void *params) {

  REAL l3;
  dft_ot_functional *otf = (dft_ot_functional *) params;

  l3 = dft_common_lwl3(otf->mass, otf->temp);
  return -rhop * GRID_AUKB * otf->temp * (1.0 - LOG(rhop*l3 + 1E-6*EPS));
}

/*
 * Ideal bose gas. NVT free energy / volume (i.e., A/V, A = U - TS).
 *
 * rhop = Gas density (REAL).
 *
 * Returns free energy / volume.
 */

EXPORT REAL dft_common_bose_idealgas_energy(REAL rhop, void *params) {

  REAL z, l3;
  dft_ot_functional *otf = (dft_ot_functional *) params;

  l3 = dft_common_lwl3(otf->mass, otf->temp);
  z = dft_common_fit_z(rhop * l3);
  return (otf->c4 * GRID_AUKB * otf->temp * (rhop * LOG(z) - dft_common_fit_g52(z) / l3));
}

/*
 * Ideal bose gas. Derivative of energy / volume with respect to rho.
 *
 * rhop = Gas density (REAL).
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
 * Numerical potential routines. Equidistant steps for potential required!
 *
 * file = Filename (char *).
 * pot  = Place the potential in this structure (dft_extpot *).
 *
 * No return value.
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
 * External potential suitable for grid map() routines.
 *
 * arg = Potential (dft_extpot_set *).
 * x   = x-coordinate (REAL).
 * y   = y-coordinate (REAL).
 * z   = z-coordinate (REAL).
 *
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
 * Map a potential given by an ascii file into a grid.
 *
 * average = 0: no averaging, 1 = average XY, 2 = average YZ, 3 = average XZ,
 *           4 = average XYZ.
 * file_x  = Potential along x axis (char *).
 * file_y  = Potential along y axis (char *).
 * file_z  = Potential along z axis (char *).
 * grid    = Output potential grid (cgrid3d *).
 * theta0  = Rotation angle theta.
 * phi0    = Rotation angle phi.
 * x0      = New origin x.
 * y0      = New origin y.
 * z0      = New origin z.
 * 
 * No return value.
 *
 * TODO: Change tilt to rotate.
 *
 */
	
EXPORT void dft_common_potential_map_tilt_shift(char average, char *filex, char *filey, char *filez, rgrid3d *potential, REAL theta0, REAL phi0, REAL x0, REAL y0, REAL z0) {

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
  rgrid3d_map(potential, dft_common_extpot, (void *) &set);
  fprintf(stderr, "done.\n");
}

/*
 * Special case of the above with x0 = y0 = z0 = theta0 = phi0 = 0 for backwards compatibility.
 * 
 */

EXPORT void dft_common_potential_map(char average, char *filex, char *filey, char *filez, rgrid3d *potential) {
  
  dft_common_potential_map_tilt_shift(average, filex, filey, filez, potential, 0.0, 0.0, 0.0, 0.0, 0.0);
}

/*
 * Special case of the above with x0 = y0 = z0 = 0 for backwards compatibility.
 *
 */

EXPORT void dft_common_potential_map_tilt(char average, char *filex, char *filey, char *filez, rgrid3d *potential, REAL theta, REAL phi) {
  
  dft_common_potential_map_tilt_shift(average, filex, filey, filez, potential, theta, phi, 0.0, 0.0, 0.0);
}

/*
 * Nonperiodic version of dft_common_potential_map.
 *
 * No need for this, just set x0, y0, z0. Left here only for compatibility.
 *
 */

EXPORT void dft_common_potential_map_nonperiodic(char average, char *filex, char *filey, char *filez, rgrid3d *potential) {

  rgrid3d_set_origin(potential, -((REAL) (potential->nx / 2)) * potential->step, -((REAL) (potential->ny / 2)) * potential->step, -((REAL) (potential->nz / 2)) * potential->step);
  dft_common_potential_map(average, filex, filey, filez, potential);
}

/*
 * Map a potential given by an ascii file into a grid (with smoothing).
 *
 * average = 0: no averaging, 1 = average XY, 2 = average YZ, 3 = average XZ,
 *           4 = average XYZ.
 * file_x  = Potential along x axis (char *).
 * file_y  = Potential along y axis (char *).
 * file_z  = Potential along z axis (char *).
 * grid    = Output potential grid (cgrid3d *).
 * theta0  = Rotation angle theta.
 * phi0    = Rotation angle phi.
 * x0      = New origin x.
 * y0      = New origin y.
 * z0      = New origin z.
 * 
 * No return value.
 *
 * TODO: Change tilt to rotate.
 *
 */
	
EXPORT void dft_common_potential_smap_tilt_shift(char average, char *filex, char *filey, char *filez, rgrid3d *potential, REAL theta0, REAL phi0, REAL x0, REAL y0, REAL z0) {

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
  rgrid3d_smooth_map(potential, dft_common_extpot, (void *) &set, 10); /* TODO allow changing this */
  fprintf(stderr, "done.\n");
}

/*
 * Special case of the above with x0 = y0 = z0 = theta0 = phi0 = 0 for backwards compatibility.
 *
 */

EXPORT void dft_common_potential_smap(char average, char *filex, char *filey, char *filez, rgrid3d *potential) {
  
  dft_common_potential_smap_tilt_shift(average, filex, filey, filez, potential, 0.0, 0.0, 0.0, 0.0, 0.0);
}

/* 
 * Special of the above case with x0 = y0 = z0 = 0 for backwards compatibility.
 *
 */

EXPORT void dft_common_potential_smap_tilt(char average, char *filex, char *filey, char *filez, rgrid3d *potential, REAL theta, REAL phi) {
  
  dft_common_potential_smap_tilt_shift(average, filex, filey, filez, potential, theta, phi, 0.0, 0.0, 0.0);
}

/*
 * Nonperiodic version of dft_common_potential_map. No need for this, just set x0,y0,z0.
 * Left here only for compatibility.
 *
 */

EXPORT void dft_common_potential_smap_nonperiodic(char average, char *filex, char *filey, char *filez, rgrid3d *potential) {

  rgrid3d_set_origin(potential, -((REAL) (potential->nx / 2)) * potential->step, -((REAL) (potential->ny / 2)) * potential->step, -((REAL) (potential->nz / 2)) * potential->step);
  dft_common_potential_smap(average, filex, filey, filez, potential);
}

/*
 *
 * This is an auxiliary routine and should not be called by users (hence not exported).
 *
 */

static rgrid3d *dft_common_pot_interpolate_read(INT n, char **files) {

  REAL r, pot_begin, pot_step;
  INT i, j, k;
  INT nr, nphi, pot_length;
  rgrid3d *cyl;
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

  cyl = rgrid3d_alloc(nr, nphi, 1, pot.step, RGRID3D_PERIODIC_BOUNDARY, NULL, "cyl");
  
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
	cyl->value[i * nphi + j] = pot.points[0];
      else
	cyl->value[i * nphi + j] = pot.points[k++];
    }
  }

  return cyl;
}

/*
 *
 * This is an auxiliary routine and should not be called by users (hence not exported).
 *
 */

inline REAL eval_value_at_index_cyl(rgrid3d *grid, INT i, INT j, INT k) {

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
  return grid->value[(i*nphi + j)*nz + k];
}

/*
 *
 * This is an auxiliary routine and should not be called by users (hence not exported).
 *
 */

EXPORT inline REAL eval_value_cyl(rgrid3d *grid, REAL r, REAL phi, REAL z) {

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

static inline REAL dft_common_interpolate_value(rgrid3d *grid, REAL r, REAL phi, REAL *x, REAL *y) {

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

static inline REAL dft_common_spline_value(rgrid3d *grid, REAL r, REAL phi, REAL *x, REAL *y, REAL *y2) {

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
 * Produce interpolated 3-D potential energy surface
 * from n 1-D cuts along phi = 0, pi/n, ..., pi directions.
 * (symmetric for ]Pi,2Pi[)
 * Requires cylindrical symmetry for the overall potential (i.e., linear molecule).
 * All potentials must have the same range, steps, number of points.
 *
 * n     = Number of potentials along n different angles (INT).
 * files = Array of strings for file names containing the potentials (char **).
 * out   = 3-D cartesian grid containing the angular interpolated potential data. 
 *         (must be allocated before calling)
 *
 */

EXPORT void dft_common_pot_interpolate(INT n, char **files, rgrid3d *out) {

  REAL x, y, z, r, phi, step = out->step, x0 = out->x0, y0 = out->y0, z0 = out->z0;
  INT nx = out->nx, ny = out->ny, nz = out->nz, nynz = ny * nz, i, j, k;
  REAL *tmp1, *tmp2;
  rgrid3d *cyl;

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
	out->value[i * nynz + j * nz + k] = dft_common_interpolate_value(cyl, r, phi, tmp1, tmp2);
      }
    }
  }
  rgrid3d_free(cyl);
  free(tmp1);
  free(tmp2);
}

/*
 * Produce interpolated spline 3-D potential energy surface
 * from n 1-D cuts along phi = 0, pi/n, ..., pi directions.
 * (symmetric for ]Pi,2Pi[)
 * Requires cylindrical symmetry for the overall potential (i.e., linear molecule).
 * All potentials must have the same range, steps, number of points.
 *
 * n     = Number of potentials along n different angles (INT).
 * files = Array of strings for file names containing the potentials (char **).
 * out   = 3-D cartesian grid containing the angular interpolated potential data. 
 *         (must be allocated before calling)
 *
 */

EXPORT void dft_common_pot_spline(INT n, char **files, rgrid3d *out) {

  REAL x, y, z, r, phi, step = out->step, x0 = out->x0, y0 = out->y0, z0 = out->z0;
  INT nx = out->nx, ny = out->ny, nz = out->nz, nynz = ny * nz, i, j, k;
  REAL *tmp1, *tmp2 , *tmp3;
  rgrid3d *cyl;

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
	out->value[i * nynz + j * nz + k] = dft_common_spline_value(cyl, r, phi, tmp1, tmp2, tmp3) ;
      }
    }
  }
  rgrid3d_free(cyl);
  free(tmp1);
  free(tmp2);
}

/*
 * Compute the numerical second derivative with respect to the angle of the potential.
 * Computed via second-order finite difference formula,
 *  
 *  f''(i) = h**2 * ( f(i-1) - 2*f(i) + f(i+1) )
 *
 *  in this case h = 2*pi / nx .
 *
 * TODO: There's a better way to do this now that spline is implemented.
 * spline generates the second derivative as by-product.
 *
 */

EXPORT void dft_common_pot_angularderiv(INT n, char **files, rgrid3d *out) {

  REAL x, y, z, r, phi, step_cyl, step = out->step, x0 = out->x0, y0 = out->y0, z0 = out->z0;
  INT nx = out->nx, ny = out->ny, nz = out->nz, i, j, k, nynz = ny * nz;
  INT nphi , nr;
  rgrid3d *cyl_pot, *cyl_k;

  cyl_pot = dft_common_pot_interpolate_read(n, files);    /* cyl_pot allocated */
  nr = cyl_pot->nx;
  nphi = cyl_pot->ny;
  step_cyl = cyl_pot->step;

  cyl_k = rgrid3d_alloc(nr, nphi, 1, step_cyl, RGRID3D_PERIODIC_BOUNDARY, NULL, "cyl_k"); 

  /* second derivative respect to theta */
  REAL inv_step2 = ((REAL) (nphi * nphi)) / (2.0 * 2.0 * M_PI * M_PI);
  for (i = 0; i < nr; i++) {
    for (j = 0; j < nphi ; j++) {
	      cyl_k->value[i * nphi + j ] = inv_step2 * (
			         rgrid3d_value_at_index(cyl_pot, i, j-1, 0)
			  -2.0 * rgrid3d_value_at_index(cyl_pot, i, j  , 0)
			  +      rgrid3d_value_at_index(cyl_pot, i, j+1, 0));
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
	out->value[i * nynz + j * nz + k] = eval_value_cyl(cyl_k, r, phi, 0.0);
      }
    }
  }

  rgrid3d_free(cyl_pot);
  rgrid3d_free(cyl_k);
}

/*
 * Spherically averaged 1-D potential energy surface
 * from n 1-D cuts along phi = 0, pi/n, ..., pi directions.
 * 
 * NOTE this is different than in dft_common_pot_interpolate where you must
 * give potentials for the whole 2pi.
 *
 * Requires cylindrical symmetry for the overall potential (i.e., linear molecule).
 * All potentials must have the same range, steps, number of points.
 *
 * n     = Number of potentials along n different angles (INT).
 * files = Array of strings for file names containing the potentials (char **).
 * ni    = Number of intermediate angular points used in interpolation (int)
 *         (if zero, will be set to 10 * n, which works well).
 * out   = 3-D grid containing the angular interpolated potential grid. 
 * 
 */
EXPORT void dft_common_pot_average(INT n, char **files, rgrid3d *out) {
  
  REAL x, y, z, r, step = out->step, pot_begin, pot_step, x0 = out->x0, y0 = out->y0, z0 = out->z0;
  INT nx = out->nx, ny = out->ny, nz = out->nz, i, j, k, nr, pot_length, nynz = ny * nz;
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
	  out->value[i * nynz + j * nz + k] = pot_ave.points[0];
	else {
	  nr = (INT) ((r - pot_begin) / pot_step);
	  if(nr < pot_ave.length)
	    out->value[i * nynz + j * nz + k] = pot_ave.points[nr];
	  else
	    out->value[i * nynz + j * nz + k] = 0.0;
	}
      }
    }
  }
}
