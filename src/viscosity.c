/*
 * Viscous potential obtained from Navier-Stokes eq. by Madelung transformation.
 *
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <grid/grid.h>
#include <grid/au.h>
#include "dft.h"
#include "ot.h"
#include "git-version.h"

struct visc_func_param {
  REAL rho0;
  REAL viscosity;
  REAL viscosity_alpha;
};

static REAL visc_func(REAL rho, void *prm) {

  struct visc_func_param *params = (struct visc_func_param *) prm;

  return POW(rho / params->rho0, params->viscosity_alpha) * params->viscosity;  // viscosity_alpha > 0
}

/*
 * Add viscous potential (from Navier-Stokes).
 *
 * gwf             = Wave function (wf *; input).
 * otf             = Orsay-Trento functional pointer (dft_ot_functional *; input).
 * viscosity       = Viscosity in atomic units (REAL; input).
 * viscosity_alpha = Exponent for scaling viscosity at interface (see above) (REAL; input).
 *
 * No return value.
 *
 */

EXPORT void dft_viscous_potential(wf *gwf, dft_ot_functional *otf, cgrid *pot, REAL viscosity, REAL viscosity_alpha) {

  rgrid *workspace1 = otf->workspace1;
  rgrid *workspace2 = otf->workspace2;
  rgrid *workspace3 = otf->workspace3;
  rgrid *workspace4 = otf->workspace4;
  rgrid *workspace5 = otf->workspace5;
  rgrid *workspace6 = otf->workspace6;
  rgrid *workspace7 = otf->workspace7;
  rgrid *workspace8 = otf->workspace8;
  REAL rho0 = otf->rho0;
  struct visc_func_param prm;  
  
  if(!otf->workspace1) workspace1 = otf->workspace1 = rgrid_clone(otf->density, "OTF workspace 1");
  if(!otf->workspace2) workspace2 = otf->workspace2 = rgrid_clone(otf->density, "OTF workspace 2");
  if(!otf->workspace3) workspace3 = otf->workspace3 = rgrid_clone(otf->density, "OTF workspace 3");
  if(!otf->workspace4) workspace4 = otf->workspace4 = rgrid_clone(otf->density, "OTF workspace 4");
  if(!otf->workspace5) workspace5 = otf->workspace5 = rgrid_clone(otf->density, "OTF workspace 5");
  if(!otf->workspace6) workspace6 = otf->workspace6 = rgrid_clone(otf->density, "OTF workspace 6");
  if(!otf->workspace7) workspace7 = otf->workspace7 = rgrid_clone(otf->density, "OTF workspace 7");
  if(!otf->workspace8) workspace8 = otf->workspace8 = rgrid_clone(otf->density, "OTF workspace 8");

  prm.rho0 = rho0;
  prm.viscosity = viscosity;
  prm.viscosity_alpha = viscosity_alpha;

  // was 1e-8   (crashes, 5E-8 done, test 1E-7)
  // we have to worry about 1 / rho....
#define POISSON_EPS 1E-7

  /* Stress tensor elements (without viscosity) */
  /* 1 (diagonal; workspace2) */
  grid_wf_velocity_x(gwf, workspace8, otf->veloc_cutoff);
  RGRID_GRADIENT_X(workspace8, workspace2);
  rgrid_multiply(workspace2, 4.0/3.0);
  grid_wf_velocity_y(gwf, workspace8, otf->veloc_cutoff);
  RGRID_GRADIENT_Y(workspace8, workspace1);
  rgrid_multiply(workspace1, -2.0/3.0);
  rgrid_sum(workspace2, workspace2, workspace1);
  grid_wf_velocity_z(gwf, workspace8, otf->veloc_cutoff);
  RGRID_GRADIENT_Z(workspace8, workspace1);
  rgrid_multiply(workspace1, -2.0/3.0);
  rgrid_sum(workspace2, workspace2, workspace1);

  /* 2 = 4 (symmetry; workspace3) */
  grid_wf_velocity_y(gwf, workspace8, otf->veloc_cutoff);
  RGRID_GRADIENT_X(workspace8, workspace3);
  grid_wf_velocity_x(gwf, workspace8, otf->veloc_cutoff);
  RGRID_GRADIENT_Y(workspace8, workspace1);
  rgrid_sum(workspace3, workspace3, workspace1);
  
  /* 3 = 7 (symmetry; workspace4) */
  grid_wf_velocity_z(gwf, workspace8, otf->veloc_cutoff);
  RGRID_GRADIENT_X(workspace8, workspace4);
  grid_wf_velocity_x(gwf, workspace8, otf->veloc_cutoff);
  RGRID_GRADIENT_Z(workspace8, workspace1);
  rgrid_sum(workspace4, workspace4, workspace1);

  /* 5 (diagonal; workspace5) */
  grid_wf_velocity_y(gwf, workspace8, otf->veloc_cutoff);
  RGRID_GRADIENT_Y(workspace8, workspace5);
  rgrid_multiply(workspace5, 4.0/3.0);
  grid_wf_velocity_x(gwf, workspace8, otf->veloc_cutoff);
  RGRID_GRADIENT_X(workspace8, workspace1);
  rgrid_multiply(workspace1, -2.0/3.0);
  rgrid_sum(workspace5, workspace5, workspace1);
  grid_wf_velocity_z(gwf, workspace8, otf->veloc_cutoff);
  RGRID_GRADIENT_Z(workspace8, workspace1);
  rgrid_multiply(workspace1, -2.0/3.0);
  rgrid_sum(workspace5, workspace5, workspace1);
  
  /* 6 = 8 (symmetryl workspace6) */
  grid_wf_velocity_z(gwf, workspace8, otf->veloc_cutoff);
  RGRID_GRADIENT_Y(workspace8, workspace6);
  grid_wf_velocity_y(gwf, workspace8, otf->veloc_cutoff);
  RGRID_GRADIENT_Z(workspace8, workspace1);
  rgrid_sum(workspace6, workspace6, workspace1);

  /* 9 = (diagonal; workspace7) */
  grid_wf_velocity_z(gwf, workspace8, otf->veloc_cutoff);
  RGRID_GRADIENT_Z(workspace8, workspace7);
  rgrid_multiply(workspace7, 4.0/3.0);
  grid_wf_velocity_x(gwf, workspace8, otf->veloc_cutoff);
  RGRID_GRADIENT_X(workspace8, workspace1);
  rgrid_multiply(workspace1, -2.0/3.0);
  rgrid_sum(workspace7, workspace7, workspace1);
  grid_wf_velocity_y(gwf, workspace8, otf->veloc_cutoff);
  RGRID_GRADIENT_Y(workspace8, workspace1);
  rgrid_multiply(workspace1, -2.0/3.0);
  rgrid_sum(workspace7, workspace7, workspace1);

  /* factor in viscosity */
  grid_wf_density(gwf, workspace8);
  rgrid_operate_one(workspace8, workspace8, visc_func, &prm);
  rgrid_product(workspace2, workspace2, workspace8);
  rgrid_product(workspace3, workspace3, workspace8);
  rgrid_product(workspace4, workspace4, workspace8);
  rgrid_product(workspace5, workspace5, workspace8);
  rgrid_product(workspace6, workspace6, workspace8);
  rgrid_product(workspace7, workspace7, workspace8);
  
  /* x component of divergence (workspace1) */
  rgrid_div(workspace1, workspace2, workspace3, workspace4); // (d/dx) 1(wrk2) + (d/dy) 2(wrk3) + (d/dz) 3(wrk4)
  /* y component of divergence (workspace2) */
  rgrid_div(workspace2, workspace3, workspace5, workspace6); // (d/dx) 2(wrk3) + (d/dy) 5(wrk5) + (d/dz) 6(wrk6)
  /* x component of divergence (workspace3) */
  rgrid_div(workspace3, workspace4, workspace6, workspace7); // (d/dx) 3(wrk4) + (d/dy) 6(wrk6) + (d/dz) 9(wrk7)

  /* divide by -rho */
  grid_wf_density(gwf, workspace8);
  rgrid_multiply(workspace8, -1.0);
  rgrid_division_eps(workspace1, workspace1, workspace8, POISSON_EPS);
  rgrid_division_eps(workspace2, workspace2, workspace8, POISSON_EPS);
  rgrid_division_eps(workspace3, workspace3, workspace8, POISSON_EPS);
  
  /* the final divergence */
  rgrid_div(workspace8, workspace1, workspace2, workspace3);
  
  // Solve the Poisson equation to get the viscous potential
#ifdef GRID_MGPU
//  New libgrid does not include fft's automatically!
  rgrid_fft(workspace8);
  rgrid_poisson(workspace8);
  rgrid_inverse_fft(workspace8);
#else
  rgrid_poisson(workspace8);
#endif
  grid_add_real_to_complex_re(pot, workspace8);
}
