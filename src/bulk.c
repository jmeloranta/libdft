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
 * rho = Bulk density (REAL; input).
 *
 * Returns bulk liquid energy density (i.e., energy / volume).
 *
 */

EXPORT REAL dft_ot_bulk_energy(dft_ot_functional *otf, REAL rho) {

  REAL tmp;

  if(otf->model & DFT_ZERO) return 0.0;

  if((otf->model & DFT_GP) || (otf->model & DFT_GP2)) return otf->mu0 * rho;

  tmp = (1.0 / 2.0) * otf->b * rho * rho + (1.0 / 2.0) * otf->c2 * rho * rho * rho + (1.0 / 3.0) * otf->c3 * rho * rho * rho * rho;
  if(otf->c4 != 0.0)
    tmp += dft_common_bose_idealgas_energy(rho, (void *) otf); /* includes c4 */
  return tmp;
}

/* 
 * Derivative of energy with respect to density in uniform bulk. 
 * At equilibirum, this is equal to the chemical potential.
 *
 * otf = OT functional (df_ot_functional *; input).
 * rho = Bulk density (REAL; *).
 *
 * Returns (dE/drho)(rho).
 *
 */

EXPORT REAL dft_ot_bulk_dEdRho(dft_ot_functional *otf, REAL rho) {

  REAL tmp;

  if(otf->model & DFT_ZERO) return 0.0;

  if((otf->model & DFT_GP) || (otf->model & DFT_GP2)) return otf->mu0;

  tmp = otf->b * rho + (3.0 / 2.0) * otf->c2 * rho * rho + (4.0 / 3.0) * otf->c3 * rho * rho * rho;
  if(otf->c4 != 0.0)
    tmp += dft_common_bose_idealgas_dEdRho(rho, (void *) otf);
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

EXPORT REAL dft_ot_bulk_density(dft_ot_functional *otf) {

  if(otf->model & DFT_ZERO) return 0.0;
  if((otf->model & DFT_GP) || (otf->model & DFT_GP2)) return otf->rho0;

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

EXPORT REAL dft_ot_bulk_chempot(dft_ot_functional *otf) {

  if((otf->model & DFT_GP) || (otf->model & DFT_GP2)) return otf->mu0;
  return dft_ot_bulk_dEdRho(otf, dft_ot_bulk_density(otf));
}

/*
 * Chemical potential of the uniform bulk. If this quantity
 * is substracted from the external potential, then the imaginary time
 * converges to a solution with the equilibrium density in the borders
 * of the box - no need for rescaling in ITP. The only difference compared
 * to the above is that the chemical potential is computed for the bulk density
 * provided in otf rather than the saturated vapor pressure.
 *
 * otf = OT functional (dft_ot_functional *; input).
 *
 * Returns chemical potential at otf->rho0.
 *
 */

EXPORT REAL dft_ot_bulk_chempot2(dft_ot_functional *otf) {

  if((otf->model & DFT_GP) || (otf->model & DFT_GP2)) return otf->mu0;
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
 * rho0 = liquid density (REAL; input).
 *
 * Returns chemical potential at otf->rho0.
 *
 */

EXPORT REAL dft_ot_bulk_chempot3(dft_ot_functional *otf, REAL rho0) {

  if((otf->model & DFT_GP) || (otf->model & DFT_GP2)) return otf->mu0;
  return dft_ot_bulk_dEdRho(otf, rho0);
}

/*
 * Pressure of uniform bulk at given certain density.
 *
 * otf = OT functional (dft_ot_functional *; input).
 * rho = Bulk density (REAL; input).
 *
 * Returns the external pressure corresponding to density rho.
 *
 */

EXPORT REAL dft_ot_bulk_pressure(dft_ot_functional *otf, REAL rho) {

  return rho * dft_ot_bulk_dEdRho(otf, rho) - dft_ot_bulk_energy(otf, rho);
}

/* 
 * Derivate of pressure with respect to density in uniform bulk.
 *
 * otf = OT functional (dft_ot_functional *; input).
 * rho = bulk density where derivative is evaluated (REAL; input).
 *
 * Returns (dP/dRho) evaluated at rho.
 *
 */ 

EXPORT REAL dft_ot_bulk_dPdRho(dft_ot_functional *otf, REAL rho) {

#if 0
  REAL tmp, z, l3;

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
 * pressure = External pressure (REAL; input).
 *
 * Returns equilibrium bulk density at given pressure.
 *
 */

EXPORT REAL dft_ot_bulk_density_pressurized(dft_ot_functional *otf, REAL pressure) {

  REAL rho0 = 1.0;
  REAL misP = dft_ot_bulk_pressure(otf, rho0) - pressure;
  REAL tol2 = 1.0E-12;
  int i, maxiter = 1000;

  if(otf->model & DFT_ZERO) return 0.0;

  if((otf->model & DFT_GP) || (otf->model & DFT_GP2)) return otf->rho0;  // no density dep.
  
  /*
   * Newton-Rapson to solve for rho:
   * Pressure = bulk_dEdRho * rho - bulk_ener
   *
   */
  for(i = 0; i < maxiter; i++) {
    //    if(misP * misP / (pressure * pressure) < tol2) return rho0;
    // printf("rho0 = %le, misP = %le, tol2 = %le\n", rho0, misP, tol2);
    if(FABS(misP) < tol2) return rho0;
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
 * pressure = External pressure where to evaluate chem.pot. (REAL; input).
 *
 * Returns chemical potential at the given pressure.
 *
 */

EXPORT REAL dft_ot_bulk_chempot_pressurized(dft_ot_functional *otf, REAL pressure) {

  return dft_ot_bulk_dEdRho(otf, dft_ot_bulk_density_pressurized(otf, pressure));
}

/*
 * Isothermal compressibility: (1/rho) (drho / dP) = 1 / (rho dP/drho) evaluated at given rho.
 *
 */

EXPORT REAL dft_ot_bulk_compressibility(dft_ot_functional *otf, REAL rho) {

  return 1.0 / (rho * dft_ot_bulk_dPdRho(otf, rho));
}

/*
 * Speed of sound: c = 1 / sqrt(M * kappa * rho) at given rho.
 *
 */

EXPORT REAL dft_ot_bulk_sound_speed(dft_ot_functional *otf, REAL rho) {

  return 1.0 / SQRT(otf->mass * rho * dft_ot_bulk_compressibility(otf, rho));
}

/*
 * Calculate bulk dispersion relation (omega vs. k). Numerical solution for current otf.
 *
 * wf   = Wave function (wf *; input/output).
 * otf  = Orsay-Trento functional pointer (dft_ot_functional *; input).
 * ts   = Time step in au (REAL; input).
 * k    = Requested wavenumber in a.u. (REAL; input/output). 
 * amp  = Amplitude relative to rho0 (REAL; input).
 * pred = 1 = Use predict-correct, 0 = no predict-correct (char; input).
 * dir  = Direction for plane wave excitation (0 = X, 1 = Y, 2 = Z; char).
 *
 * Returns energy (omega; a.u.).
 *
 */

EXPORT REAL dft_ot_dispersion(wf *gwf, dft_ot_functional *otf, REAL ts, REAL *k, REAL amp, char pred, char dir) {

  REAL step = gwf->grid->step, rho0, tmp, ival, mu0, omega;   /* TS in fs */
  dft_plane_wave wave_params;
  INT l, nx = gwf->grid->nx, ny = gwf->grid->ny, nz = gwf->grid->nz;
  wf *gwfp;
  cgrid *potential;
  grid_timer timer;
  
  if(pred) gwfp = grid_wf_clone(gwf, "gwfp for dft_ot_dispersion");
  potential = cgrid_clone(gwf->cworkspace, "potential for dft_ot_dispersion");
  rho0 = otf->rho0;
  mu0 = dft_ot_bulk_chempot2(otf);
  
  switch(dir) {
    case 0: /* X direction */
      if(nx == 1) {
        fprintf(stderr, "libdft: The plane wave is along X but only one point allocated in that direction.\n");
        exit(1);
      }
      tmp = 2.0 * M_PI / (((REAL) nx) * step);
      wave_params.kx = ((REAL) (((INT) (0.5 + *k / tmp)))) * tmp; // round to nearest k with the grid - should we return this also?
      *k = wave_params.kx;
      if(*k == 0.0) return 0.0;
      wave_params.ky = 0.0;
      wave_params.kz = 0.0;
      break;
    case 1: /* Y direction */
      if(ny == 1) {
        fprintf(stderr, "libdft: The plane wave is along Y but only one point allocated in that direction.\n");
        exit(1);
      }
      tmp = 2.0 * M_PI / (((REAL) ny) * step);
      wave_params.ky = ((REAL) (((INT) (0.5 + *k / tmp)))) * tmp; // round to nearest k with the grid - should we return this also?
      *k = wave_params.ky;
      if(*k == 0.0) return 0.0;
      wave_params.kx = 0.0;
      wave_params.kz = 0.0;
      break;
    case 2: /* Z direction */
      if(nz == 1) {
        fprintf(stderr, "libdft: The plane wave is along Z but only one point allocated in that direction.\n");
        exit(1);
      }
      tmp = 2.0 * M_PI / (((REAL) nz) * step);
      wave_params.kz = ((REAL) (((INT) (0.5 + *k / tmp)))) * tmp; // round to nearest k with the grid - should we return this also?
      *k = wave_params.kz;
      if(*k == 0.0) return 0.0;
      wave_params.kx = 0.0;
      wave_params.ky = 0.0;
      break;
    default:
      fprintf(stderr, "libdft: Error in pane wave direction (only 0, 1, 2 allowed).\n");
      exit(1);
  }
  wave_params.a = amp;
  wave_params.rho = otf->rho0;
  grid_wf_map(gwf, dft_common_planewave, &wave_params);
  ival = 0.0;
  for(l = 0; ; l++) {
    grid_timer_start(&timer);
    cgrid_zero(potential);
    if(pred) {
      cgrid_copy(gwfp->grid, gwf->grid);
      dft_ot_potential(otf, potential, gwf);
      cgrid_add(potential, -mu0);
      grid_wf_propagate_predict(gwfp, potential, ts);
      dft_ot_potential(otf, potential, gwfp);
      cgrid_add(potential, -mu0);
      cgrid_multiply(potential, 0.5);
      grid_wf_propagate_correct(gwf, potential, ts);
    } else {
      dft_ot_potential(otf, potential, gwf);
      grid_wf_propagate(gwf, potential, ts);
    }
    printf("Iteration " FMT_I " - Wall clock time = " FMT_R " seconds.\n", l, grid_timer_wall_clock_time(&timer));
    ival += (POW(CABS(cgrid_value_at_index(gwf->grid, nx/2, ny/2, nz/2)), 2.0) - rho0);
    if(ival < 0.0) {
      l--;
      break;
    }
  }
  omega = (1.0 / (2.0 * ((REAL) l) * ts / GRID_AUTOFS));
  grid_wf_free(gwf);
  if(pred) {
    grid_wf_free(gwfp);
    cgrid_free(potential);
  }
  return (omega / GRID_AUTOS) * GRID_HZTOCM1 * 1.439 /* cm-1 to K */ / GRID_AUTOK;
}

/*
 * Calculate bulk dispersion relation (omega vs. k). Semi-analytic solution.
 *
 * otf  = functional (dft_ot_functional *; input).
 * k    = momentum (REAL *; input/output). On output, contains the actual value of k used for computing omega.
 * rho0 = bulk density (REAL; input).
 *
 * Returns energy (omega; a.u.).
 *
 * NOTES: This uses driver and overwrites the otf structure there. So, do not use
 *        this routine at the same time when running simulations that employ driver.
 *
 *        k to Angs^-1: k / GRID_AUTOANG
 *        omega to K: (omega / GRID_AUTOS) * GRID_HZTOCM1 * 1.439    (<- cm-1 to K)
 *
 *        If rho0 < 0, the routine just frees the allocated memory.
 *
 */

#define FT_UL 1E3
#define FT_STEP 1.0E-2

static REAL ft_lj(dft_ot_functional *otf, REAL k) {

  REAL sigma = otf->lj_params.sigma, epsilon = otf->lj_params.epsilon, h = otf->lj_params.h;
  REAL val = 0.0, x, ks = k * sigma;

  for (x = h / sigma; x < FT_UL; x += FT_STEP) 
    val += SIN(ks * x) * (POW(x, -11.0) - POW(x, -5.0));
  val *= FT_STEP * 16.0 * M_PI * epsilon * sigma * sigma * sigma / ks;
  return val;
}

static REAL ft_pi(dft_ot_functional *otf, REAL k) {

  REAL h = otf->lj_params.h, kh = k * h;

  return (3.0 / (kh * kh * kh)) * (SIN(kh) - kh * COS(kh));
}

static REAL ft_vj(dft_ot_functional *otf, REAL k) {

  REAL g11 = otf->bf_params.g11, g12 = otf->bf_params.g12, g21 = otf->bf_params.g21, g22 = otf->bf_params.g22;
  REAL a1 = otf->bf_params.a1, a2 = otf->bf_params.a2;
  REAL ea1, ea2, k2 = k * k, val, mpi32 = POW(M_PI, 3.0 / 2.0);

  ea1 = EXP(-k2 / (4.0 * a1));
  ea2 = EXP(-k2 / (4.0 * a2));

  val = (g11 * POW(M_PI/a1, 3.0/2.0) + g12 * (6.0 * a1 - k2) * mpi32 / (4.0 * POW(a1, 7.0/2.0))) * ea1
    + (g21 * POW(M_PI/a2, 3.0/2.0) + g22 * (6.0 * a2 - k2) * mpi32 / (4.0 * POW(a2, 7.0/2.0))) * ea2;
  return val;
}

EXPORT REAL dft_ot_bulk_dispersion(dft_ot_functional *otf, REAL *k, REAL rho0) {

  REAL tmp, lj, pi, vj, vj0, ikai, tk;

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
           * EXP(-tk * tk * otf->l_g * otf->l_g / 4.0);
  if(otf->model & DFT_OT_BACKFLOW) {
    vj = ft_vj(otf, tk);
    vj0 = ft_vj(otf, 0.0);
    return SQRT(tmp * ikai * (1.0 - rho0 * (vj0 - vj)));
  }
  return SQRT(tmp * ikai);
}

/*
 * Calculation of the static structure factor X(q).
 *
 * otf  = functional (dft_ot_functional *; input).
 * k    = momentum (REAL *; input).
 *
 * Returns -1/X(q)
 * 
 */

EXPORT REAL dft_ot_bulk_istatic(dft_ot_functional *otf, REAL *k, REAL rho0) {

  REAL lj, pi, ikai, tk;

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
           * EXP(-tk * tk * otf->l_g * otf->l_g / 4.0);

  return ikai;
}

/*
 * Calculate free (flat) surface tension.
 *
 * TODO: Is this correct? propagate without potential??
 * 
 */

static REAL complex Aslab(void *param, REAL x, REAL y, REAL z) {

  REAL width = *((REAL *) param);

  if(FABS(x) < width/2.0) return 1.0;
  else return 0.0;
}

EXPORT REAL dft_ot_bulk_surface_tension(wf *gwf, dft_ot_functional *otf, REAL ts, REAL width) {

  REAL mu0, stens, prev_stens, rho0 = otf->rho0;
  INT i, ny = gwf->grid->ny, nz = gwf->grid->nz;
  rgrid *density;
  cgrid *potential;
  REAL step = gwf->grid->step;

  potential = cgrid_clone(gwf->grid, "Surface tension potential");
  density = rgrid_clone(otf->density, "Surface tension density");

  mu0 = dft_ot_bulk_chempot2(otf);
  grid_wf_map(gwf, &Aslab, NULL);
  cgrid_multiply(gwf->grid, SQRT(rho0));

  prev_stens = 1E11;
  for(i = 1; ; i++) {
    cgrid_zero(potential);
    dft_ot_potential(otf, potential, gwf);
    cgrid_add(potential, -mu0);
    grid_wf_propagate(gwf, potential, ts);

    dft_ot_energy_density(otf, density, gwf);
    stens = (grid_wf_energy(gwf, NULL) + rgrid_integral(density)) 
       / (2.0 * ((REAL) ny) * ((REAL) nz) * step * step);
    if(FABS(stens - prev_stens) / stens < 0.03) break;
    prev_stens = stens;
  }
  cgrid_free(potential);
  rgrid_free(density);
  return stens;
}

