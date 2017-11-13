#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <grid/grid.h>
#include <grid/au.h>
#include "dft.h"
#include "ot.h"

/* Step size used for finite difference derivative. */
#define LOCAL_EPS 1E-7

/*
 * Energy density in uniform bulk.
 *
 * otf = OT functional (dft_ot_functional *; input).
 * rho = Bulk density (double; input).
 *
 * Returns bulk liquid energy density (i.e., energy / volume).
 *
 */

EXPORT double dft_ot_bulk_energy(dft_ot_functional *otf, double rho) {

  double tmp;

  if(otf->model & DFT_ZERO) return 0.0;

  if(otf->model & DFT_GP) return otf->mu0 * rho;

  tmp = (1.0 / 2.0) * otf->b * rho * rho + (1.0 / 2.0) * otf->c2 * rho * rho * rho + (1.0 / 3.0) * otf->c3 * rho * rho * rho * rho;
  if(otf->c4 != 0.0) {
    dft_common_idealgas_params(otf->temp, otf->mass, otf->c4);
    tmp += dft_common_bose_idealgas_energy(rho); /* includes c4 */
  }
  return tmp;
}

/* 
 * Derivative of energy with respect to density in uniform bulk. 
 * In equilibirum, this is equal to the chemical potential.
 *
 * otf = OT functional (df_ot_functional *; input).
 * rho = Bulk density (double; *).
 *
 * Returns (dE/drho)(rho).
 *
 */

EXPORT double dft_ot_bulk_dEdRho(dft_ot_functional *otf, double rho) {

  double tmp;

  if(otf->model & DFT_ZERO) return 0.0;

  if(otf->model & DFT_GP) return otf->mu0;

  tmp = otf->b * rho + (3.0 / 2.0) * otf->c2 * rho * rho + (4.0 / 3.0) * otf->c3 * rho * rho * rho;
  if(otf->c4 != 0.0) {
    dft_common_idealgas_params(otf->temp, otf->mass, otf->c4);
    tmp += dft_common_bose_idealgas_dEdRho(rho);
  }
  return tmp;
}

/*
 * Equilibrium density for uniform bulk with no pressure applied.
 *
 * In general, the eq. density is obtained by solving:
 * 	Pressure = dEdRho(rho0)*rho0 - bulk_energy(rho0) 
 * For the OT functional with Pressure = 0, the solution is analytical
 * but we use the general routine here.
 *
 * otf = OT functional (dft_ot_functional *; input).
 *
 * Returns the bulk density when P = 0.
 * 
 */

EXPORT double dft_ot_bulk_density(dft_ot_functional *otf) {

  if(otf->model & DFT_ZERO) return 0.0;
  if(otf->model & DFT_GP) return otf->rho0;

  return dft_ot_bulk_density_pressurized(otf, 0.0);
}

/*
 * Chemical potential of the uniform bulk. If this quantity
 * is substracted from the external potential then the imaginary time
 * converges to a solution with the equilibrium density in the borders
 * of the box - no need for rescaling during imaginary time propagation.
 *
 * otf = OT functional (dft_ot_functional *; input).
 *
 * Returns the chemical potential at bulk density (P = 0).
 *
 */

EXPORT double dft_ot_bulk_chempot(dft_ot_functional *otf) {

  if(otf->model & DFT_GP) return otf->mu0;
  return dft_ot_bulk_dEdRho(otf, dft_ot_bulk_density(otf));
}

/*
 * Chemical potential of the uniform bulk. If this quantity
 * is substracted from the external potential, then the imaginary time
 * converges to a solution with the equilibrium density in the borders
 * of the box - no need for rescaling in ITP. The only difference compare
 * to the above is that the chemical potential is computed for the bulk density
 * provided in otf rather than the saturated vapor pressure.
 *
 * otf = OT functional (dft_ot_functional *; input).
 *
 * Returns chemical potential at otf->rho0.
 *
 */

EXPORT double dft_ot_bulk_chempot2(dft_ot_functional *otf) {

  if(otf->model & DFT_GP) return otf->mu0;
  return dft_ot_bulk_dEdRho(otf, otf->rho0);
}

/*
 * Chemical potential of the uniform bulk. If this quantity
 * is substracted from the external potential, then the imaginary time
 * converges to a solution with the equilibrium density in the borders
 * of the box - no need for rescaling in ITP. The only difference compare
 * to the above is that the chemical potential is computed for the bulk density
 * provided in otf rather than the saturated vapor pressure.
 *
 * otf  = OT functional (dft_ot_functional *; input).
 * rho0 = liquid density (double; input).
 *
 * Returns chemical potential at otf->rho0.
 *
 */

EXPORT double dft_ot_bulk_chempot3(dft_ot_functional *otf, double rho0) {

  if(otf->model & DFT_GP) return otf->mu0;
  return dft_ot_bulk_dEdRho(otf, rho0);
}

/*
 * Pressure of uniform bulk at given certain density.
 *
 * otf = OT functional (dft_ot_functional *; input).
 * rho = Bulk density (double; input).
 *
 * Returns the external pressure corresponding to density rho.
 *
 */

EXPORT double dft_ot_bulk_pressure(dft_ot_functional *otf, double rho) {

  return rho * dft_ot_bulk_dEdRho(otf, rho) - dft_ot_bulk_energy(otf, rho);
}

/* 
 * Derivate of pressure with respect to density in uniform bulk.
 *
 * otf = OT functional (dft_ot_functional *; input).
 * rho = bulk density where derivative is evaluated (double; input).
 *
 * Returns (dP/dRho) evaluated at rho.
 *
 */ 

EXPORT double dft_ot_bulk_dPdRho(dft_ot_functional *otf, double rho) {

#if 0
  double tmp, z, l3;

  tmp = otf->b * rho + 3.0 * otf->c2 * rho * rho + 4.0 * otf->c3 * rho * rho * rho;
  if(otf->c4 != 0.0) {
   l3 = dft_common_lwl3(otf->mass, otf->temp);
   z = dft_common_fit_z(rho * l3);
   tmp += GRID_AUKB * otf->temp * dft_common_fit_g32(z) / dft_common_fit_g12(z);
  }
  return tmp;
#else
  return (dft_ot_bulk_pressure(otf, rho+LOCAL_EPS) - dft_ot_bulk_pressure(otf, rho-LOCAL_EPS)) / (2.0 * LOCAL_EPS);
#endif
}

/*
 * Equilibrium density for pressurized uniform bulk.
 * The density is obtained by solving:
 * 	Pressure = dEdRho(rho0)*rho0 - bulk_energy(rho0) 
 *
 * otf      = OT functional (dft_ot_functional *; input).
 * pressure = External pressure (double; input).
 *
 * Returns equilibrium bulk density at given pressure.
 *
 */

EXPORT double dft_ot_bulk_density_pressurized(dft_ot_functional *otf, double pressure) {

  double rho0 = 1.0;
  double misP = dft_ot_bulk_pressure(otf, rho0) - pressure;
  double tol2 = 1.0E-12;
  int i, maxiter = 1000;

  if(otf->model & DFT_ZERO) return 0.0;

  if(otf->model & DFT_GP) return otf->rho0;  // no density dep.
  
  /*
   * Newton-Rapson to solve for rho:
   * Pressure = bulk_dEdRho * rho - bulk_ener
   *
   */
  for(i = 0; i < maxiter; i++) {
    //    if(misP * misP / (pressure * pressure) < tol2) return rho0;
    // printf("rho0 = %le, misP = %le, tol2 = %le\n", rho0, misP, tol2);
    if(fabs(misP) < tol2) return rho0;
    rho0 -= misP / dft_ot_bulk_dPdRho(otf, rho0);
    misP = dft_ot_bulk_pressure(otf, rho0) - pressure;
  }
  fprintf(stderr, "libdft: Error in dft_ot_bulk_density_pressurized - Newton-Raphson did not converge.\n");
  abort();
  return NAN;
}

/*
 * Chemical potential for pressurized uniform bulk.
 *
 * otf      = OT functional (dft_ot_functional *; input).
 * pressure = External pressure where to evaluate chem.pot. (double; input).
 *
 * Returns chemical potential at the given pressure.
 *
 */

EXPORT double dft_ot_bulk_chempot_pressurized(dft_ot_functional *otf, double pressure) {

  return dft_ot_bulk_dEdRho(otf, dft_ot_bulk_density_pressurized(otf, pressure));
}


/*
 * Isothermal compressibility: (1/rho) (drho / dP) = 1 / (rho dP/drho) evaluated at given rho.
 *
 */

EXPORT double dft_ot_bulk_compressibility(dft_ot_functional *otf, double rho) {

  return 1.0 / (rho * dft_ot_bulk_dPdRho(otf, rho));
}

/*
 * Speed of sound: c = 1 / sqrt(M * kappa * rho) at given rho.
 *
 */

EXPORT double dft_ot_bulk_sound_speed(dft_ot_functional *otf, double rho) {

  return 1.0 / sqrt(otf->mass * rho * dft_ot_bulk_compressibility(otf, rho));
}

/*
 * Calculate bulk dispersion relation (omega vs. k). Numerical solution.
 *
 * otf  = functional (dft_ot_functional *; input).
 * k    = momentum (double *; input/output). On output, contains the actual value of k used for computing omega.
 * rho0 = bulk density (double; input).
 *
 * Returns energy (omega; a.u.).
 *
 * NOTES: This uses driver3d and overwrites the otf structure there. So, do not use
 *        this routine at the same time when running simulations that employ driver3d.
 *
 *        k to Angs^-1: k / GRID_AUTOANG
 *        omega to K: omega * GRID_AUTOK
 *
 *        If rho0 < 0, the routine just frees the allocated memory.
 *
 *        This requires fairly dense grid to get the correct dispersion relation when KC/BF present.
 * 
 */

typedef struct sWaveParams_struct {
  double kx, ky, kz;
  double a, rho;
} sWaveParams;

static cgrid3d *Apotential_store = NULL;
static wf3d *Agwf = NULL, *Agwfp = NULL;
static rgrid3d *Adensity = NULL, *Apot = NULL;

static double complex Awave(void *arg, double x, double y, double z) {

  double kx = ((sWaveParams *) arg)->kx;
  double ky = ((sWaveParams *) arg)->ky;
  double kz = ((sWaveParams *) arg)->kz;
  double a = ((sWaveParams *) arg)->a;
  double psi = sqrt(((sWaveParams *) arg)->rho);
  
  return psi + 0.5 * a * psi * (cexp(I * (kx * x + ky * y + kz * z)) + cexp(-I*(kx * x + ky * y + kz * z)));
}

double dft_ot_bulk_TS = 50.0; /* fs */
double dft_ot_bulk_AMP = 1.0E-3; /* amplitude of the excitation */
double dft_ot_bulk_NX = 128; /* Bohr */
double dft_ot_bulk_NY = 32;  /* Bohr */
double dft_ot_bulk_NZ = 32;  /* Bohr */
double dft_ot_bulk_STEP = 1.0; /* Bohr */
long dft_ot_bulk_THR = 0;    /* Number of threads */

static double prev_step = 0.0;
extern int dft_driver_verbose;

EXPORT double dft_ot_dispersion(dft_ot_functional *otf, double *k, double rho0) {

  double tmp, pval, mu0, omega;   /* TS in fs */
  sWaveParams wave_params;
  long l;

  dft_driver_verbose = 0;
  if(dft_ot_bulk_STEP != prev_step) {
    dft_ot_bulk_NX = 512;
    prev_step = dft_ot_bulk_STEP;
    if(Apotential_store) cgrid3d_free(Apotential_store);
    if(Apot) rgrid3d_free(Apot);
    if(Agwf) grid3d_wf_free(Agwf);
    if(Agwfp) grid3d_wf_free(Agwfp);
    if(Adensity) rgrid3d_free(Adensity);
    if(rho0 < 0.0) return 0.0; /* just free the memeory */
    dft_driver_setup_grid(dft_ot_bulk_NX, dft_ot_bulk_NY, dft_ot_bulk_NZ, dft_ot_bulk_STEP, dft_ot_bulk_THR);
    dft_driver_setup_model(otf->model, DFT_DRIVER_REAL_TIME, rho0);
    dft_driver_setup_boundary_type(DFT_DRIVER_BOUNDARY_REGULAR, 0.0, 0.0);
    dft_driver_setup_normalization(DFT_DRIVER_DONT_NORMALIZE, 0, 0.0, 0);
    dft_driver_setup_boundary_condition(DFT_DRIVER_BC_NORMAL);
    dft_driver_initialize();
    Apotential_store = dft_driver_alloc_cgrid();
    Adensity = dft_driver_alloc_rgrid();           /* avoid allocating separate space for density */
    Apot = dft_driver_alloc_rgrid();
    Agwf = dft_driver_alloc_wavefunction(otf->mass);
    Agwfp = dft_driver_alloc_wavefunction(otf->mass);
  }
  /* Update driver otf structure - parameters in the given otf may have changed from last call */
  otf->lennard_jones = dft_driver_otf->lennard_jones;
  otf->spherical_avg = dft_driver_otf->spherical_avg;
  otf->gaussian_tf = dft_driver_otf->gaussian_tf;
  otf->gaussian_x_tf = dft_driver_otf->gaussian_x_tf;
  otf->gaussian_y_tf = dft_driver_otf->gaussian_y_tf;
  otf->gaussian_z_tf = dft_driver_otf->gaussian_z_tf;
  otf->backflow_pot = dft_driver_otf->backflow_pot;
  bcopy(otf, dft_driver_otf, sizeof(dft_ot_functional));
  mu0 = dft_ot_bulk_chempot2(otf);
  rgrid3d_constant(Apot, -mu0);

  tmp = 2.0 * M_PI / (dft_ot_bulk_NX * dft_ot_bulk_STEP);
  wave_params.kx = ((long) (0.5 + *k / tmp)) * tmp; // round to nearest k with the grid - should we return this also?
  *k = wave_params.kx;
  if(*k == 0.0) return 0.0;
  wave_params.ky = 0.0;
  wave_params.kz = 0.0;
  wave_params.a = dft_ot_bulk_AMP;
  wave_params.rho = rho0;
  grid3d_wf_map(Agwf, Awave, &wave_params);
  pval = 1E99;
  for(l = 0; ; l++) {
    dft_driver_propagate_predict(DFT_DRIVER_PROPAGATE_HELIUM, Apot, Agwf, Agwfp, Apotential_store, dft_ot_bulk_TS, l);
    dft_driver_propagate_correct(DFT_DRIVER_PROPAGATE_HELIUM, Apot, Agwf, Agwfp, Apotential_store, dft_ot_bulk_TS, l);
    grid3d_wf_density(Agwf, Adensity);
    if(rgrid3d_value_at_index(Adensity, dft_ot_bulk_NX/2, dft_ot_bulk_NY/2, dft_ot_bulk_NZ/2) > pval) {
      l--;
      break;
    }
    pval = rgrid3d_value_at_index(Adensity, dft_ot_bulk_NX/2, dft_ot_bulk_NY/2, dft_ot_bulk_NZ/2);
  }
  dft_driver_verbose = 1;
  omega = (1.0 / (2.0 * l * dft_ot_bulk_TS / GRID_AUTOFS));
  return (omega / GRID_AUTOS) * GRID_HZTOCM1 * 1.439 /* cm-1 to K */ / GRID_AUTOK;
}

/*
 * Calculate bulk dispersion relation (omega vs. k). Semi-analytic solution.
 *
 * otf  = functional (dft_ot_functional *; input).
 * k    = momentum (double *; input/output). On output, contains the actual value of k used for computing omega.
 * rho0 = bulk density (double; input).
 *
 * Returns energy (omega; a.u.).
 *
 * NOTES: This uses driver3d and overwrites the otf structure there. So, do not use
 *        this routine at the same time when running simulations that employ driver3d.
 *
 *        k to Angs^-1: k / GRID_AUTOANG
 *        omega to K: (omega / GRID_AUTOS) * GRID_HZTOCM1 * 1.439    (<- cm-1 to K)
 *
 *        If rho0 < 0, the routine just frees the allocated memory.
 *
 */

#define FT_UL 1E3
#define FT_STEP 1.0E-2

static double ft_lj(dft_ot_functional *otf, double k) {

  double sigma = otf->lj_params.sigma, epsilon = otf->lj_params.epsilon, h = otf->lj_params.h;
  double val = 0.0, x, ks = k * sigma;

  for (x = h / sigma; x < FT_UL; x += FT_STEP) 
    val += sin(ks * x) * (pow(x, -11.0) - pow(x, -5.0));
  val *= FT_STEP * 16.0 * M_PI * epsilon * sigma * sigma * sigma / ks;
  return val;
}

static double ft_pi(dft_ot_functional *otf, double k) {

  double h = otf->lj_params.h, kh = k * h;

  return (3.0 / (kh * kh * kh)) * (sin(kh) - kh * cos(kh));
}

static double ft_vj(dft_ot_functional *otf, double k) {

  double g11 = otf->bf_params.g11, g12 = otf->bf_params.g12, g21 = otf->bf_params.g21, g22 = otf->bf_params.g22;
  double a1 = otf->bf_params.a1, a2 = otf->bf_params.a2;
  double ea1, ea2, k2 = k * k, val, mpi32 = pow(M_PI, 3.0 / 2.0);

  ea1 = exp(-k2 / (4.0 * a1));
  ea2 = exp(-k2 / (4.0 * a2));

  val = (g11 * pow(M_PI/a1, 3.0/2.0) + g12 * (6.0 * a1 - k2) * mpi32 / (4.0 * pow(a1, 7.0/2.0))) * ea1
    + (g21 * pow(M_PI/a2, 3.0/2.0) + g22 * (6.0 * a2 - k2) * mpi32 / (4.0 * pow(a2, 7.0/2.0))) * ea2;
  return val;
}

EXPORT double dft_ot_bulk_dispersion(dft_ot_functional *otf, double *k, double rho0) {

  double tmp, lj, pi, vj, vj0, ikai, tk;

  if(rho0 < 0.0) return 0.0;
  tk = *k;
  if(tk < 1E-3) return 0.0;
  lj = ft_lj(otf, tk);
  pi = ft_pi(otf, tk);

  tmp = HBAR * HBAR * tk * tk / otf->mass;
  ikai = tmp / 4.0 + rho0 * lj 
    + otf->c2 * (2.0 * pi + pi * pi) * rho0 * rho0 
    + 2.0 * otf->c3 * (pi + pi * pi) * rho0 * rho0 * rho0;
  if(otf->model & DFT_OT_KC)
    ikai += -(tmp/2.0) * otf->alpha_s * rho0 * (1.0 - rho0 / otf->rho_0s) * (1.0 - rho0 / otf->rho_0s) 
           * exp(-tk * tk * otf->l_g * otf->l_g / 4.0);
  if(otf->model & DFT_OT_BACKFLOW) {
    vj = ft_vj(otf, tk);
    vj0 = ft_vj(otf, 0.0);
    return sqrt(tmp * ikai * (1.0 - rho0 * (vj0 - vj)));
  }
  return sqrt(tmp * ikai);
}

/*
 * Calculation of the static structure factor X(q).
 *
 * otf  = functional (dft_ot_functional *; input).
 * k    = momentum (double *; input).
 *
 * Returns -1/X(q)
 * 
 */

EXPORT double dft_ot_bulk_istatic(dft_ot_functional *otf, double *k, double rho0) {

  double lj, pi, ikai, tk;

  if(rho0 < 0.0) return 0.0;
  tk = *k;
  if(tk < 1E-2) return tk = 1E-2;
  lj = ft_lj(otf, tk);
  pi = ft_pi(otf, tk);

  ikai = HBAR * HBAR * tk * tk / (4.0 * otf->mass) + rho0 * lj 
    + otf->c2 * (2.0 * pi + pi * pi) * rho0 * rho0 
    + 2.0 * otf->c3 * (pi + pi * pi) * rho0 * rho0 * rho0;
  if(otf->model & DFT_OT_KC)
    ikai += -(HBAR * HBAR / (2.0 * otf->mass)) * otf->alpha_s * rho0 * (1.0 - rho0 / otf->rho_0s) * (1.0 - rho0 / otf->rho_0s) 
           * exp(-tk * tk * otf->l_g * otf->l_g / 4.0);

  return ikai;
}

/*
 * Calculate free (flat) surface tension.
 *
 */

double dft_ot_bulk_slab_width = 80.0; /* Bohr */

static double complex Aslab(void *NA, double x, double y, double z) {

  if(fabs(x) < dft_ot_bulk_slab_width/2.0) return 1.0;
  else return 0.0;
}

EXPORT double dft_ot_bulk_surface_tension(dft_ot_functional *otf, double rho0) {

  double mu0, stens, prev_stens;
  long i;

  dft_driver_verbose = 0;
  if(dft_ot_bulk_STEP != prev_step) {
    dft_ot_bulk_NX = 128;
    prev_step = dft_ot_bulk_STEP;
    if(Apotential_store) cgrid3d_free(Apotential_store);
    if(Apot) rgrid3d_free(Apot);
    if(Agwf) grid3d_wf_free(Agwf);
    if(Agwfp) grid3d_wf_free(Agwfp);
    if(Adensity) rgrid3d_free(Adensity); // not needed - remove
    if(rho0 < 0.0) return 0.0; /* just free the memeory */
    dft_driver_setup_grid(dft_ot_bulk_NX, dft_ot_bulk_NY, dft_ot_bulk_NZ, dft_ot_bulk_STEP, dft_ot_bulk_THR);
    dft_driver_setup_model(otf->model, DFT_DRIVER_IMAG_TIME, rho0);
    dft_driver_setup_boundary_type(DFT_DRIVER_BOUNDARY_REGULAR, 0.0, 0.0);
    dft_driver_setup_normalization(DFT_DRIVER_DONT_NORMALIZE, 0, 0.0, 0);
    dft_driver_setup_boundary_condition(DFT_DRIVER_BC_NORMAL);
    dft_driver_initialize();
    Apotential_store = dft_driver_alloc_cgrid();
    Adensity = dft_driver_alloc_rgrid();           /* avoid allocating separate space for density */
    Apot = dft_driver_alloc_rgrid();
    Agwf = dft_driver_alloc_wavefunction(otf->mass);
    Agwfp = dft_driver_alloc_wavefunction(otf->mass);
    /* setup a free surface (slab around x = 0) */
    grid3d_wf_map(Agwf, &Aslab, NULL);
    cgrid3d_multiply(Agwf->grid, sqrt(rho0));
  }
  /* Update driver otf structure - parameters in the given otf may have changed from last call */
  otf->lennard_jones = dft_driver_otf->lennard_jones;
  otf->spherical_avg = dft_driver_otf->spherical_avg;
  otf->gaussian_tf = dft_driver_otf->gaussian_tf;
  otf->gaussian_x_tf = dft_driver_otf->gaussian_x_tf;
  otf->gaussian_y_tf = dft_driver_otf->gaussian_y_tf;
  otf->gaussian_z_tf = dft_driver_otf->gaussian_z_tf;
  otf->backflow_pot = dft_driver_otf->backflow_pot;
  bcopy(otf, dft_driver_otf, sizeof(dft_ot_functional));
  mu0 = dft_ot_bulk_chempot2(otf);
  rgrid3d_constant(Apot, -mu0);
  prev_stens = 1E99;
  for(i = 1; ; i++) {
    dft_driver_propagate_predict(DFT_DRIVER_PROPAGATE_HELIUM, Apot, Agwf, Agwfp, Apotential_store, dft_ot_bulk_TS, i);
    dft_driver_propagate_correct(DFT_DRIVER_PROPAGATE_HELIUM, Apot, Agwf, Agwfp, Apotential_store, dft_ot_bulk_TS, i);
    stens = dft_driver_energy(Agwf, Apot) / (2.0 * dft_ot_bulk_NY * dft_ot_bulk_NZ * dft_ot_bulk_STEP * dft_ot_bulk_STEP);
    if(fabs(stens - prev_stens) / stens < 0.03) break;
    prev_stens = stens;
  }
  dft_driver_verbose = 1;
  return stens;
}

