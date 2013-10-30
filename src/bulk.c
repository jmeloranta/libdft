#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <grid/grid.h>
#include <grid/au.h>
#include "dft.h"
#include "ot.h"

/*
 * Energy density in uniform bulk.
 */

EXPORT double bulk_energy(dft_ot_functional *otf, double rho){
	if(otf->model & DFT_ZERO)
		return 0. ;
	if(otf->model & DFT_GP)
		return 0.5 * rho*rho * otf->mu0 / otf->rho0 ;
	return otf->b * rho*rho/2. + otf->c2 * rho*rho*rho/2.+ otf->c3 * rho*rho*rho*rho/3. ;
}

/* 
 * Derivate of energy with respect to density in uniform bulk. 
 * In equilibirum, this quantity is equal to the chemical potential.
 *
 */

EXPORT double bulk_dEdRho(dft_ot_functional *otf, double rho){
	if(otf->model & DFT_ZERO)
		return 0. ;
	if(otf->model & DFT_GP)
		return rho * otf->mu0 / otf->rho0 ;
	return otf->b * rho + otf->c2 * rho*rho*1.5+ otf->c3 * rho*rho*rho*4./3. ;
	/* if general implementation needed: */
	//return ( bulk_energy(rho*1.0001) - bulk_energy(rho*0.9999) ) / (0.0002*rho)
}

/*
 * Equilibrium density for the uniform bulk with no pressure applied.
 * In general, the eq. density is obtained by solving:
 * 	Pressure = dEdRho(rho0)*rho0 - bulk_energy(rho0) 
 * For the OT functional with Pressure = 0, the solution is analytical.
 */

EXPORT double bulk_density(dft_ot_functional *otf){
	if(otf->model & DFT_ZERO)
		return 0. ;
	if(otf->model & DFT_GP)
		return 0. ;
	double Bo2A = otf->c2/( 2. * otf->c3) ;
	double Co2A = otf->b/( 2. * otf->c3) ;
	return sqrt( Bo2A*Bo2A - Co2A) - Bo2A ;
}

/*
 * Chemical potential of the uniform bulk. If this quantity
 * is substracted from the external potential, then the imaginary time
 * converges to a solution with the equilibrium density in the borders
 * of the box -no need for scaling.
 */

EXPORT double bulk_chempot(dft_ot_functional *otf){
	return bulk_dEdRho(otf, bulk_density(otf) ) ;
}


/*
 * Pressure of the uniform bulk at a certain density. To use this
 * as control parameter, the right chemical potential must be computed.
 */
EXPORT double bulk_pressure(dft_ot_functional *otf, double rho){
	return rho * bulk_dEdRho(otf, rho) - bulk_energy(otf, rho) ;
}

/* 
 * Derivate of pressure with respect to density in uniform bulk.
 * Used only for the Newton-Raphson on bulk_density (for P!=0) 
 */ 

EXPORT double bulk_dPdRho(dft_ot_functional *otf, double rho){
	return otf->b * rho + otf->c2 * rho*rho*3.0 + otf->c3 * rho*rho*rho*4.0 ;
}

/*
 * Equilibrium density for the pressurized uniform bulk.
 * In general, the eq. density is obtained by solving:
 * 	Pressure = dEdRho(rho0)*rho0 - bulk_energy(rho0) 
 */

EXPORT double bulk_density_pressurized(dft_ot_functional *otf, double pressure){
	if(otf->model & DFT_ZERO)
		return 0. ;
	if(otf->model & DFT_GP)
		return sqrt(2.0 * pressure * otf->rho0 / otf->mu0) ;

	if(pressure==0.)
		return bulk_density(otf) ;
	/* Newton-Rapson to solve for rho:
	 *  Pressure = bulk_dEdRho * rho - bulk_ener
	 *
	 */
	double rho0 = bulk_density(otf) ;
	double misP = bulk_pressure(otf, rho0) - pressure ;
	double tol2 = 1.e-12;
	int i, maxiter = 1000;
	for(i=0; i<maxiter; i++){
		if( misP*misP/(pressure*pressure) < tol2){
			return rho0 ;
		}
		rho0 -= misP/bulk_dPdRho(otf, rho0) ;
		misP = bulk_pressure(otf, rho0) - pressure ;
	}
	fprintf(stderr, "libdft: Error in bulk_density: Newton-Raphson did not converge for the given pressure.\n");
	abort();

}

/*
 * Chemical potential for the pressurized uniform bulk.
 */

EXPORT double bulk_chempot_pressurized(dft_ot_functional *otf, double pressure){
	return bulk_dEdRho(otf, bulk_density_pressurized(otf, pressure) ) ;
}
