#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <grid/grid.h>
#include <grid/au.h>
#include "dft.h"
#include "ot.h"

/* Step size used for finite difference derivative. */
#define LOCAL_EPS 1E-7

/* TODO: FIX ME - there should be a way not to repeat everything here (instead of otf, use the variables) */
/* also update comments */

/*
 * Energy density in uniform bulk.
 */

EXPORT double dft_ot_bulk_energy_2d(dft_ot_functional_2d *otf, double rho) {

  if(otf->model & DFT_ZERO) return 0.0;

  if(otf->model & DFT_GP) return otf->mu0 * rho;

  if(otf->c4 != 0.0) {
    dft_common_idealgas_params(otf->temp, otf->mass, otf->c4);
    return otf->b * rho*rho/2.0 + otf->c2 * rho*rho*rho/2.0 + otf->c3 * rho*rho*rho*rho/3.0 + dft_common_bose_idealgas_energy(rho);
  }
  return otf->b * rho*rho/2.0 + otf->c2 * rho*rho*rho/2.0 + otf->c3 * rho*rho*rho*rho/3.0;
}

/* 
 * Derivate of energy with respect to density in uniform bulk. 
 * In equilibirum, this quantity is equal to the chemical potential.
 *
 */

EXPORT double dft_ot_bulk_dEdRho_2d(dft_ot_functional_2d *otf, double rho) {

  if(otf->model & DFT_ZERO) return 0.0;

  if(otf->model & DFT_GP) return otf->mu0;

  //return otf->b * rho + otf->c2 * rho*rho*1.5 + otf->c3 * rho*rho*rho*4.0/3.0;
  return (dft_ot_bulk_energy_2d(otf, rho + LOCAL_EPS) - dft_ot_bulk_energy_2d(otf, rho - LOCAL_EPS) ) / (2.0 * LOCAL_EPS);
}

/*
 * Equilibrium density for the uniform bulk with no pressure applied.
 * In general, the eq. density is obtained by solving:
 * 	Pressure = dEdRho(rho0)*rho0 - bulk_energy(rho0) 
 * For the OT functional with Pressure = 0, the solution is analytical.
 */

EXPORT double dft_ot_bulk_density_2d(dft_ot_functional_2d *otf) {

  //double Bo2A = otf->c2/(2.0 * otf->c3);
  //double Co2A = otf->b/(2.0 * otf->c3);
  
  if(otf->model & DFT_ZERO) return 0.0;
  if(otf->model & DFT_GP) return otf->rho0;

  //return sqrt(Bo2A * Bo2A - Co2A) - Bo2A;
  return dft_ot_bulk_density_pressurized_2d(otf, 0.0);
}

/*
 * Chemical potential of the uniform bulk. If this quantity
 * is substracted from the external potential, then the imaginary time
 * converges to a solution with the equilibrium density in the borders
 * of the box -no need for scaling.
 */

EXPORT double dft_ot_bulk_chempot_2d(dft_ot_functional_2d *otf) {

  return dft_ot_bulk_dEdRho_2d(otf, dft_ot_bulk_density_2d(otf));
}

/*
 * Chemical potential of the uniform bulk. If this quantity
 * is substracted from the external potential, then the imaginary time
 * converges to a solution with the equilibrium density in the borders
 * of the box -no need for scaling.
 *
 * The only difference compare to the above is that the 
 * chemical potential is computed for the bulk density
 * provided in otf rather than the saturated vapor pressure.
 *
 */

EXPORT double dft_ot_bulk_chempot2_2d(dft_ot_functional_2d *otf) {

  return dft_ot_bulk_dEdRho_2d(otf, otf->rho0);
}

/*
 * Pressure of the uniform bulk at a certain density. To use this
 * as control parameter, the right chemical potential must be computed.
 */

EXPORT double dft_ot_bulk_pressure_2d(dft_ot_functional_2d *otf, double rho) {

  return rho * dft_ot_bulk_dEdRho_2d(otf, rho) - dft_ot_bulk_energy_2d(otf, rho);
}

/* 
 * Derivate of pressure with respect to density in uniform bulk.
 * Used only for the Newton-Raphson on bulk_density (for P!=0) 
 */ 

EXPORT double dft_ot_bulk_dPdRho_2d(dft_ot_functional_2d *otf, double rho) {

  return (dft_ot_bulk_pressure_2d(otf, rho + LOCAL_EPS) - dft_ot_bulk_pressure_2d(otf, rho - LOCAL_EPS)) / (2.0 * LOCAL_EPS);
  //return otf->b * rho + otf->c2 * rho*rho*3.0 + otf->c3 * rho*rho*rho*4.0;
}

/*
 * Equilibrium density for the pressurized uniform bulk.
 * In general, the eq. density is obtained by solving:
 * 	Pressure = dEdRho(rho0)*rho0 - bulk_energy(rho0) 
 */

EXPORT double dft_ot_bulk_density_pressurized_2d(dft_ot_functional_2d *otf, double pressure) {

  //  double rho0 = dft_ot_bulk_density_2d(otf);
  double rho0 = 1.0;
  double misP = dft_ot_bulk_pressure_2d(otf, rho0) - pressure;
  double tol2 = 1.0E-12;
  int i, maxiter = 1000;

  if(otf->model & DFT_ZERO) return 0.0;

  if(otf->model & DFT_GP) return otf->rho0;
  
  //  if(pressure==0.0) return dft_ot_bulk_density_2d(otf);

  /*
   * Newton-Rapson to solve for rho:
   * Pressure = bulk_dEdRho * rho - bulk_ener
   *
   */
  for(i = 0; i < maxiter; i++) {
    //    if(misP*misP / (pressure*pressure) < tol2) return rho0;
    // printf("rho0 = %le, misP = %le, tol2 = %le\n", rho0, misP, tol2);
    if(fabs(misP) < tol2) return rho0;
    rho0 -= misP / dft_ot_bulk_dPdRho_2d(otf, rho0);
    misP = dft_ot_bulk_pressure_2d(otf, rho0) - pressure;
  }
  fprintf(stderr, "libdft: Error in dft_ot_bulk_density_2d - Newton-Raphson did not converge.\n");
  abort();
  return NAN;
}

/*
 * Chemical potential for the pressurized uniform bulk.
 */

EXPORT double dft_ot_bulk_chempot_pressurized_2d(dft_ot_functional_2d *otf, double pressure) {

  return dft_ot_bulk_dEdRho_2d(otf, dft_ot_bulk_density_pressurized_2d(otf, pressure));
}
