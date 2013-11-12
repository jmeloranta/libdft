/*
 * Common routines.
 *
 */

#include <stdio.h>
#include <math.h>
#include <grid/grid.h>
#include <grid/au.h>
#include "dft.h"

/*
 * Tunable numerical parameters.
 *
 */

#define EPS 1.0E-8
#define CUTOFF (50.0 / GRID_AUTOK)

/* 
 * pow() function that uses integer exponents (fast).
 *
 * TODO: Change things back everywhere so that ipow is mostly used (except for DR).
 *
 */

EXPORT inline double dft_common_ipow(double x, int y) {
  
  int i;
  double v = 1.0;

  for (i = 0; i < y; i++)
    v *= x;

  return v;
}

/* NOTE: r is actually r^2 */
EXPORT inline double dft_common_lj_func(double r2, double sig, double eps) {

   /* s = (sigma/r)^6 */
  r2 = sig * sig / r2; /* r already squared elsewhere */
  r2 = r2 * r2 * r2;
  
  /* Vlj = 4 * eps ( (sigma/r)^12 - (sigma/r)^6 ) */ 
  return 4.0 * eps * r2 * (r2 - 1.0);
}

/*
 * Lennard-Jones potential to be used with grid map() routines (3D)x
 * Note that the LJ potential has zero core when r < h.
 *
 * The potential paramegers are passed in arg (ot_common_lh data type).
 *
 */

EXPORT inline double dft_common_lennard_jones(void *arg, double x, double y, double z) {

  double h = ((dft_common_lj *) arg)->h;
  double sig = ((dft_common_lj *) arg)->sigma;
  double eps = ((dft_common_lj *) arg)->epsilon;
  double r2;

  r2 = x * x + y * y + z * z;
  
  if (r2 <= h * h) return 0.0;
  
  return dft_common_lj_func(r2, sig, eps);
}

/*
 * Lennard-Jones potential to be used with grid map() routines (2D).
 * Note that the LJ potential has zero core when r < h.
 *
 * The potential paramegers are passed in arg (ot_common_lh data type).
 *
 */

EXPORT inline double dft_common_lennard_jones_2d(void *arg, double z, double r) {

  double h = ((dft_common_lj *) arg)->h;
  double sig = ((dft_common_lj *) arg)->sigma;
  double eps = ((dft_common_lj *) arg)->epsilon;
  double r2;

  r2 = z * z + r * r;
  
  if (r2 <= h * h) return 0.0;
  
  return dft_common_lj_func(r2, sig, eps);
}

/*
 * Lennard-Jones potential with smoothed core to be used with grid map()
 * routines. Parameters passed in arg (see the regular LJ above).
 *
 */

EXPORT inline double dft_common_lennard_jones_smooth(void *arg, double x, double y, double z) {

  double h = ((dft_common_lj *) arg)->h;
  double sig = ((dft_common_lj *) arg)->sigma;
  double eps = ((dft_common_lj *) arg)->epsilon;
  double r2, h2 = h * h;

  r2 = x * x + y * y + z * z;
  
  /* Ul(h) * (r/h)^4 */
  if (r2 < h2) return dft_common_lj_func(h2, sig, eps) * r2 * r2 / (h2 * h2);
  
  return dft_common_lj_func(r2, sig, eps);
}

/*
 * Lennard-Jones potential with smoothed core to be used with grid map() 
 * routines (2D). Pameters passed in arg (see the regular LJ above).
 *
 */

EXPORT inline double dft_common_lennard_jones_smooth_2d(void *arg, double z, double r) {

  double h = ((dft_common_lj *) arg)->h;
  double sig = ((dft_common_lj *) arg)->sigma;
  double eps = ((dft_common_lj *) arg)->epsilon;
  double r2, h2 = h * h;

  r2 = z * z + r * r;
  
  /* Ul(h) * (r/h)^4 */
  if (r2 < h2) return dft_common_lj_func(h2, sig, eps) * r2 * r2 / (h2 * h2);
  
  return dft_common_lj_func(r2, sig, eps);
}

/*
 * Spherical average function to be used with grid map() routines (3D).
 * The sphere radius is passed in arg.
 *
 */

EXPORT inline double dft_common_spherical_avg(void *arg, double x, double y, double z) {

  double h = *((double *) arg), h2 = h * h;
  if (x*x + y*y + z*z <= h2)
    return 3.0 / (4.0 * M_PI * h * h2);
  return 0.0;
}


/*
 * Spherical average function IN MOMENTUM SPACE to be used with grid map() routines (3D).
 * The sphere radius is passed in arg.
 *
 */

EXPORT inline double dft_common_spherical_avg_k(void *arg, double kx, double ky, double kz) {

  double hk = *((double *) arg) * sqrt(kx*kx + ky*ky + kz*kz) ;
  if(hk < 1.e-5)
	return 1.0 - 0.1*hk*hk; /* second order Taylor expansion */
  return 3.0 * (sin(hk) - hk * cos(hk) ) / (hk*hk*hk) ;
}

/*
 * Spherical average function to be used with grid map() routines (2D).
 * The sphere radius is passed in arg.
 *
 */

EXPORT inline double dft_common_spherical_avg_2d(void *arg, double z, double r) {

  double h = *((double *) arg), h2 = h * h;
  if (z * z + r * r <= h2)
    return 3.0 / (4.0 * M_PI * h * h2);
  return 0.0;
}

/*
 * Gaussian function to be used with grid map() functions (3D).
 * The gaussian is centered at (0,0,0) and width is given in arg.
 *
 */ 

EXPORT inline double dft_common_gaussian(void *arg, double x, double y, double z) {

  double inv_width = *((double *) arg);
  double norm = 0.5 * M_2_SQRTPI * inv_width;

  norm = norm * norm * norm;
  return norm * exp(-(x * x + y * y + z * z) * inv_width * inv_width);
}

/*
 * Gaussian function to be used with grid map() functions (2D.
 * The gaussian is centered at (0,0,0) and width is given in arg.
 *
 */ 

EXPORT inline double dft_common_gaussian_2d(void *arg, double z, double r) {

  double inv_width = *((double *) arg);
  double norm = 0.5 * M_2_SQRTPI * inv_width;

  norm = norm * norm * norm;
  return norm * exp(-(z * z + r * r) * inv_width * inv_width);
}

/*
 * Functions for polylogarithms (using polynomial fits).
 *
 */

/* Li_{1/2}(z) fit parameters */
#define LENA 9
static double A[LENA] = {-0.110646, 5.75395, 37.2433, 123.411, 228.972, 241.484, 143.839, 45.0107, 5.74587};

/* Li_{3/2}(z) fit parameters */
#define LENB 10
static double B[LENB] = {0.0025713, 1.0185, 0.2307, -0.15674, 1.0025, 1.8146, -1.8981, -3.0619, 1.438, 1.9332};

/* Li_{5/2}(z) fit parameters */
#define LENC 7
static double C[LENC] = {-0.00022908, 1.0032, 0.18414, 0.03958, -0.0024563, 0.057067, 0.052367};

/* parameters for g_{3/2}(z) = \rho\lambda^3 */
#define LEND 5
static double D[LEND] = {5.913E-5, 0.99913, -0.35069, 0.053981, -0.0038613};

/* Boltzmann constant in au */
#define DFT_KB 3.1668773658e-06

/*
 * Return thermal wavelength for a particle with mass "mass" at temperature
 * "temp".
 *
 */

EXPORT inline double dft_common_lwl3(double mass, double temp) {

  double lwl;

  /* hbar = 1 */
  lwl = sqrt(2.0 * M_PI / (mass * DFT_KB * temp));
  return lwl * lwl * lwl;
}

/*
 * Evaluate z(rho, T) (polylog).
 *
 */

EXPORT inline double dft_common_fit_z(double val) {

  int i;
  double rv = 0.0, e = 1.0;

  if(val >= 2.583950) return 1.0; /* g_{3/2}(1) */
  for (i = 0; i < LEND; i++) {
    rv += D[i] * e;
    e *= val;
  }
  if(rv <= 0.0) rv = 1E-6;
  return rv;
}

/* 
 * Evaluate g_{1/2} (polylog).
 *
 */

EXPORT inline double dft_common_fit_g12(double z) {

  int i;
  double rv = 0.0, e = 1.0;

  if(fabs(z) > 1.0) fprintf(stderr, "polylog: warning |z| > 1.\n");
  z -= 1.0;
  rv = A[0] / (z + EPS);
  for (i = 0; i < LENA; i++) {
    rv += A[i+1] * e;
    e *= z;
  }
  if(z + 1.0 > 0.0 && rv < 0.0) rv = 0.0; /* small values may give wrong sign */
  return rv;
}

/*
 * Evaluate g_{3/2} (polylog)
 *
 */

EXPORT inline double dft_common_fit_g32(double z) {

  int i;
  double rv = 0.0, e = 1.0;

  if(fabs(z) > 1.0) fprintf(stderr, "polylog: warning |z| > 1.\n");
  for (i = 0; i < LENB; i++) {
    rv += B[i] * e;
    e *= z;
  }
  return rv;
}

/*
 * Evaluate g_{5/2} (polylog)
 *
 */

EXPORT inline double dft_common_fit_g52(double z) {

  int i;
  double rv = 0.0, e = 1.0;

  if(fabs(z) > 1.0) fprintf(stderr, "polylog: warning |z| > 1.\n");
  for (i = 0; i < LENC; i++) {
    rv += C[i] * e;
    e *= z;
  }
  return rv;
}

/*
 * Ideal gas contribution routines. Call the params routine first to
 * set temperature, mass and the constant (multiplier) c4. Then use
 * the grid map() functions.
 *
 */

static double XXX_temp, XXX_mass, XXX_c4;

EXPORT void dft_common_idealgas_params(double temp, double mass, double c4) {

  XXX_temp = temp;
  XXX_mass = mass;
  XXX_c4 = c4;
}

EXPORT double dft_common_idealgas_op(double rhop) {

  double l3, z0, rl3, g12, g32;
  double tmp;

  l3 = dft_common_lwl3(XXX_mass, XXX_temp);
  rl3 = rhop * l3;
  if(rl3 >= 2.583950) return 0.0;
  z0 = dft_common_fit_z(rl3);
  g12 = dft_common_fit_g12(z0);
  g32 = dft_common_fit_g32(z0);

  /* note g12 may be zero too - avoid NaNs... */
  tmp = XXX_c4 * DFT_KB * XXX_temp * (log(z0 + EPS) + rl3 / (g12 + EPS) - g32 / (g12 + EPS));
  /* The above term is ill behaved at low densities */
  if(fabs(tmp) > CUTOFF) {
    if(tmp < 0.0) return -CUTOFF;
    else return CUTOFF;
  } else return tmp;
}

EXPORT double dft_common_idealgas_energy_op(double rhop) {

  double z, l3;

  l3 = dft_common_lwl3(XXX_mass, XXX_temp);
  z = dft_common_fit_z(rhop * l3);
  return (XXX_c4 * DFT_KB * XXX_temp * (rhop * log(z) - dft_common_fit_g52(z) / l3));
}

/*
 * Numerical potential routines. Equidistant steps for potential required!
 *
 */

EXPORT void dft_common_read_pot(char *file, dft_extpot *pot) {

  FILE *fp;
  int i;
  double b = 0.0, s = 0.0, x = 0.0, px;

  if(!(fp = fopen(file, "r"))) {
    fprintf(stderr, "libdft: Can't open %s.\n", file);
    exit(1);
  }

  for (i = 0; i < DFT_MAX_POTENTIAL_POINTS; i++) {
    px = x;
    if(fscanf(fp, " %le %le", &x, &(pot->points[i])) != 2) break;
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
    if(fabs(x - px - s) > s/10.0) {
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
 */

EXPORT double dft_common_extpot(void *arg, double x, double y, double z) {
   
  double r = sqrt(x * x + y * y + z * z), px, py, pz, tmp;
  double theta, phi, sin_theta, cos_theta, sin_phi, cos_phi;
  long i;
  dft_extpot_set *set = (dft_extpot_set *) arg;
  double *pot_x = set->x->points, *pot_y = set->y->points, *pot_z = set->z->points;
  double bx = set->x->begin, by = set->y->begin, bz = set->z->begin;
  double sx = set->x->step, sy = set->y->step, sz = set->z->step;
  long lx = set->x->length, ly = set->y->length, lz = set->z->length;
  int aver = set->average;

  /* x */
  i = (long) ((r - bx) / sx);
  if(i < 0) {
    /*    px = pot_x[0] + 1.0E-3 * pot_x[0] * (double) labs(i); */
    px = pot_x[0];
  } else if (i > lx-1) {
    px =  0.0;    /* was pot_x[lx-1] */
  } else px = pot_x[i];

  /* y */
  i = (long) ((r - by) / sy);
  if(i < 0) {
    /*    py = pot_y[0] + 1.0E-3 * pot_y[0] * (double) labs(i); */
    py = pot_y[0];
  } else if (i > ly-1) {
    py =  0.0; /* was pot_y[ly-1] */
  } else py = pot_y[i];

  /* z */
  i = (long) ((r - bz) / sz);
  if(i < 0) {
    /*    pz = pot_z[0] + 1.0E-3 * pot_z[0] * (double) labs(i); */
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

  theta = acos(z / (r + 1E-3));
  phi = atan(y / (x + 1E-3));
  sin_theta = sin(theta);
  sin_theta *= sin_theta;
  cos_theta = cos(theta);
  cos_theta *= cos_theta;
  sin_phi = sin(phi);
  sin_phi *= sin_phi;
  cos_phi = cos(phi);
  cos_phi *= cos_phi;
  return px * sin_theta * cos_phi + py * sin_theta * sin_phi + pz * cos_theta;
}

EXPORT double dft_common_extpot_cyl(void *arg, double r, double phi, double z) {

  double x, y;

  x = r * cos(phi);
  y = r * sin(phi);
  return dft_common_extpot(arg, x, y, z);
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
 * 
 * No return value.
 *
 */

EXPORT void dft_common_potential_map(int average, char *filex, char *filey, char *filez, rgrid3d *potential) {

  dft_extpot x, y, z;
  dft_extpot_set set;
  
  set.x = &x;
  set.y = &y;
  set.z = &z;
  set.average = average;

  fprintf(stderr, "libdft: Mapping potential file with x = %s, y = %s, z = %s. Average = %d\n", filex, filey, filez, average);
  dft_common_read_pot(filex, &x);
  dft_common_read_pot(filey, &y);
  dft_common_read_pot(filez, &z);
  rgrid3d_map(potential, dft_common_extpot, (void *) &set);
}

/*
 * Mapping function for a potential in cylindrical (2D) coordinates.
 *
 */
 
EXPORT double dft_common_extpot_2d(void *arg, double z, double r) {
   
  double rr = sqrt(z * z + r * r), pz, pr, tmp;
  double phi, sin_phi, cos_phi;
  long i;
  dft_extpot_set_2d *set = (dft_extpot_set_2d *) arg;
  double *pot_z = set->z->points, *pot_r = set->r->points;
  double bz = set->z->begin, br = set->r->begin;
  double sz = set->z->step, sr = set->r->step;
  long lz = set->z->length, lr = set->r->length;
  int aver = set->average;

  /* z */
  i = (long) ((rr - bz) / sz);
  if(i < 0) {
    /*    pz = pot_z[0] + 1.0E-3 * pot_z[0] * (double) labs(i); */
    pz = pot_z[0];
  } else if (i > lz-1) {
    pz =  0.0;    /* was pot_z[lz-1] */
  } else pz = pot_z[i];

  /* r */
  i = (long) ((rr - br) / sr);
  if(i < 0) {
    /*    pr = pot_r[0] + 1.0E-3 * pot_r[0] * (double) labs(i); */
    pr = pot_r[0];
  } else if (i > lr-1) {
    pr =  0.0; /* was pot_r[lr-1] */
  } else pr = pot_r[i];

  switch(aver) {
  case 0: /* no averaging */
    break;
  case 1: /* ZR average */
    tmp = (pz + 2.0 * pr) / 3.0;
    pz = pr = tmp;
    break;
  }

  phi = atan(r / (z + 1E-3));
  sin_phi = sin(phi);
  sin_phi *= sin_phi;
  cos_phi = cos(phi);
  cos_phi *= cos_phi;
  return pz * sin_phi + pr * cos_phi;
}

/*
 * Map a potential given by an ascii file onto a cylindrical grid.
 *
 * average = 0: no averaging, 1 = average zr.
 * file_z  = Potential along z axis (char *).
 * file_r  = Potential along r axis (char *).
 * grid    = Output potential grid (cgrid3d *).
 * 
 * No return value.
 *
 */

EXPORT void dft_common_potential_map_2d(int average, char *filez, char *filer, rgrid2d *potential) {

  dft_extpot z, r;
  dft_extpot_set_2d set;
  
  set.z = &z;
  set.r = &r;
  set.average = average;

  fprintf(stderr, "libdft: Mapping potential file with z = %s, r = %s. Average = %d\n", filez, filer, average);
  dft_common_read_pot(filez, &z);
  dft_common_read_pot(filer, &r);
  rgrid2d_map_cyl(potential, dft_common_extpot_2d, (void *) &set);
}

/*
 * Map a potential given by an ascii file into a 3D cylindrical grid.
 *
 * average = 0: no averaging, 1 = average XY, 2 = average YZ, 3 = average XZ,
 *           4 = average XYZ.
 * file_x  = Potential along x axis (char *).
 * file_y  = Potential along y axis (char *).
 * file_z  = Potential along z axis (char *).
 * grid    = Output potential grid (cgrid3d *).
 * 
 * No return value.
 *
 */

EXPORT void dft_common_potential_map_cyl(int average, char *filex, char *filey, char *filez, rgrid3d *potential) {

  dft_extpot x, y, z;
  dft_extpot_set set;
  
  set.x = &x;
  set.y = &y;
  set.z = &z;
  set.average = average;

  fprintf(stderr, "libdft: Mapping potential file with x = %s, y = %s, z = %s. Average = %d\n", filex, filey, filez, average);
  dft_common_read_pot(filex, &x);
  dft_common_read_pot(filey, &y);
  dft_common_read_pot(filez, &z);
  rgrid3d_map_cyl(potential, dft_common_extpot_cyl, (void *) &set);
}
