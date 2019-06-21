/*
 * Orsay-Trento functional for superfluid helium. Functional derivative of energy.
 *
 * NOTE: This code uses FFT for evaluating all the integrals, which
 *       implies periodic boundary conditions!
 *
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <grid/grid.h>
#include <grid/au.h>
#include "dft.h"
#include "ot.h"
#include "ot-private.h"
#include "git-version.h"

/* Local functions */

static void dft_ot_add_nonlocal_correlation_potential(dft_ot_functional *otf, cgrid *potential, rgrid *rho, rgrid *rho_tf, rgrid *workspace1, rgrid *workspace2, rgrid *workspace3, rgrid *workspace4, rgrid *workspace5);
static void dft_ot_add_nonlocal_correlation_potential_x(dft_ot_functional *otf, cgrid *potential, rgrid *rho, rgrid *rho_tf, rgrid *rho_st, rgrid *workspace1, rgrid *workspace2, rgrid *workspace3, rgrid *workspace4);
static void dft_ot_add_nonlocal_correlation_potential_y(dft_ot_functional *otf, cgrid *potential, rgrid *rho, rgrid *rho_tf, rgrid *rho_st, rgrid *workspace1, rgrid *workspace2, rgrid *workspace3, rgrid *workspace4);
static void dft_ot_add_nonlocal_correlation_potential_z(dft_ot_functional *otf, cgrid *potential, rgrid *rho, rgrid *rho_tf, rgrid *rho_st, rgrid *workspace1, rgrid *workspace2, rgrid *workspace3, rgrid *workspace4);
static void dft_ot_add_barranco(dft_ot_functional *otf, cgrid *potential, rgrid *rho, rgrid *workspace1);
static void dft_ot_add_ancilotto(dft_ot_functional *otf, cgrid *potential, rgrid *rho, rgrid *workspace1);

/*
 * Allocate OT functional. This must be called first.
 * 
 * model = which OT functional variant to use:
 *         DFT_OT_KC       Include the non-local kinetic energy correlation.
 *         DFT_OT_HD       Include Barranco's high density correction (original h for sp. ave).
 *         DFT_OT_HD2      Include Barranco's high density correction (new h for sp. ave).
 *         DFT_OT_BACKFLOW Include the backflow potential (dynamics).
 *         DFT_OT_T0MK     Thermal model 0.0 K (i.e. just new parametrization)
 *         DFT_OT_T400MK   Thermal model 0.4 K
 *         DFT_OT_T600MK   Thermal model 0.6 K
 *         DFT_OT_T800MK   Thermal model 0.8 K
 *         DFT_OT_T1200MK  Thermal model 1.2 K
 *         DFT_OT_T1400MK  Thermal model 1.4 K
 *         DFT_OT_T1600MK  Thermal model 1.6 K
 *         DFT_OT_T1800MK  Thermal model 1.8 K
 *         DFT_OT_T2000MK  Thermal model 2.0 K
 *         DFT_OT_T2100MK  Thermal model 2.1 K
 *         DFT_OT_T2200MK  Thermal model 2.2 K
 *         DFT_OT_T2400MK  Thermal model 2.4 K
 *         DFT_OT_T2600MK  Thermal model 2.6 K
 *         DFT_OT_T2800MK  Thermal model 2.8 K
 *         DFT_OT_T3000MK  Thermal model 3.0 K
 *         DFT_GP          Gross-Pitaevskii equation  ("works" for ions)
 *         DFT_GP2         Gross-Pitaevskii equation  (gives the correct speed of sound)
 *         DFT_ZERO        No potential
 *                If multiple options are needed, use bitwise and operator (&).
 * wf           = Wavefunction to be used with this OT (wf *; input).
 * min_substeps = minimum substeps for function smoothing over the grid.
 * max_substeps = maximum substeps for function smoothing over the grid.
 *
 * Return value: pointer to the allocated OT DFT structure.
 *
 * Basic OT allocates 2 real grids
 *       KC adds 4 real grids
 *       BF adds 3 real grids
 *
 */

EXPORT dft_ot_functional *dft_ot_alloc(INT model, wf *gwf, INT min_substeps, INT max_substeps) {

  REAL radius, inv_width;
  dft_ot_functional *otf;
  REAL x0 = gwf->grid->x0, y0 = gwf->grid->y0, z0 = gwf->grid->z0;
  REAL step = gwf->grid->step;
  INT nx = gwf->grid->nx, ny = gwf->grid->ny, nz = gwf->grid->nz;
  
  otf = (dft_ot_functional *) malloc(sizeof(dft_ot_functional));
  otf->model = model;
  if (!otf) {
    fprintf(stderr, "libdft: Error in dft_ot_alloc(): Could not allocate memory for dft_ot_functional.\n");
    return NULL;
  }
 
  fprintf(stderr, "libdft: GIT version ID %s\n", VERSION);
  fprintf(stderr, "libdft: Grid " FMT_I " x " FMT_I " x " FMT_I " with step " FMT_R " Bohr.\n", nx, ny, nz, step);
  fprintf(stderr, "libdft: Functional = " FMT_I ".\n", model);

  /* TODO: There is code for other BCs too */

//  if(gwf->grid->value_outside != CGRID_PERIODIC_BOUNDARY) {
//    fprintf(stderr, "libdft: Only periodic boundaries supported.\n");
//    exit(1);
//  }

  dft_ot_temperature(otf, model);

  // TODO: Periodic BC hardcoded at the moment.

  /* these grids are not needed for GP */
  if(!(model & DFT_GP) && !(model & DFT_ZERO) && !(model & DFT_GP2)) {
    otf->lennard_jones = rgrid_alloc(nx, ny, nz, step, RGRID_PERIODIC_BOUNDARY, 0, "OT Lennard-Jones");
    rgrid_set_origin(otf->lennard_jones, x0, y0, z0);
    otf->spherical_avg = rgrid_alloc(nx, ny, nz, step, RGRID_PERIODIC_BOUNDARY, 0, "OT Sph. average");
    rgrid_set_origin(otf->spherical_avg, x0, y0, z0);

    if(model & DFT_OT_KC) {
      otf->gaussian_tf = rgrid_alloc(nx, ny, nz, step, RGRID_PERIODIC_BOUNDARY, 0, "OT KC Gauss TF");
      if(!otf->gaussian_tf) {
        fprintf(stderr, "libdft: Error in dft_ot_alloc(): Could not allocate memory for gaussian.\n");
        return 0;
      }
      rgrid_set_origin(otf->gaussian_tf, x0, y0, z0);
      if(nx != 1 || ny != 1) {
        otf->gaussian_x_tf = rgrid_alloc(nx, ny, nz, step, RGRID_PERIODIC_BOUNDARY, 0, "OT KC Gauss TF_x");
        otf->gaussian_y_tf = rgrid_alloc(nx, ny, nz, step, RGRID_PERIODIC_BOUNDARY, 0, "OT KC Gauss TF_y");
        if(!otf->gaussian_x_tf || !otf->gaussian_y_tf) {
  	  fprintf(stderr, "libdft: Error in dft_ot_alloc(): Could not allocate memory for gaussian.\n");
	  return 0;
        }
        rgrid_set_origin(otf->gaussian_x_tf, x0, y0, z0);
        rgrid_set_origin(otf->gaussian_y_tf, x0, y0, z0);
      } else otf->gaussian_x_tf = otf->gaussian_y_tf = NULL;
      otf->gaussian_z_tf = rgrid_alloc(nx, ny, nz, step, RGRID_PERIODIC_BOUNDARY, 0, "OT KC Gauss TF_z");
      if(!otf->gaussian_z_tf) {
        fprintf(stderr, "libdft: Error in dft_ot_alloc(): Could not allocate memory for gaussian.\n");
        return 0;
      }
      rgrid_set_origin(otf->gaussian_z_tf, x0, y0, z0);
    } else otf->gaussian_x_tf = otf->gaussian_y_tf = otf->gaussian_z_tf = otf->gaussian_tf = NULL;
  
    if(model & DFT_OT_BACKFLOW) {
      otf->backflow_pot = rgrid_alloc(nx, ny, nz, step, RGRID_PERIODIC_BOUNDARY, 0, "OT Backflow");
      if(!otf->backflow_pot) {
	fprintf(stderr, "libdft: Error in dft_ot_alloc(): Could not allocate memory for backflow_pot.\n");
	return 0;
      }
    } else otf->backflow_pot = NULL;

#ifdef DFT_OT_1D
    if(nx == 1 && ny == 1)
      fprintf(stderr, "libdft: Using 1-D model with effective 3-D potential.\n");
#endif
  
    /* pre-calculate */
    if(otf->model & DFT_DR) {
#ifdef DFT_OT_1D
      if(nx == 1 && ny == 1) {
        fprintf(stderr, "libdft: DFT_DR not implemented for 1-D.\n");
        exit(1);
      }
#endif
      fprintf(stderr, "libdft: LJ according to DR - ");
      rgrid_adaptive_map(otf->lennard_jones, dft_common_lennard_jones_smooth, &(otf->lj_params), min_substeps, max_substeps, 0.01 / GRID_AUTOK);
      rgrid_fft(otf->lennard_jones);
    } else {
      fprintf(stderr, "libdft: LJ according to OT - ");
#ifdef DFT_OT_1D
      if(nx == 1 && ny == 1)
        rgrid_adaptive_map(otf->lennard_jones, dft_common_lennard_jones_1d, &(otf->lj_params), min_substeps, max_substeps, 0.01 / GRID_AUTOK);
      else 
#endif
        rgrid_adaptive_map(otf->lennard_jones, dft_common_lennard_jones, &(otf->lj_params), min_substeps, max_substeps, 0.01 / GRID_AUTOK);      
      rgrid_fft(otf->lennard_jones);
      /* Scaling of LJ so that the integral is exactly b */
      if(nx != 1 || ny != 1)      
        rgrid_multiply_fft(otf->lennard_jones, otf->b / (step * step * step * (REAL) rgrid_cvalue_at_index(otf->lennard_jones, 0, 0, 0)));
    }
    fprintf(stderr, "Done.\n");

    if(otf->model & DFT_OT_HD2) {
      radius = otf->lj_params.h * 1.065; /* PRB 72, 214522 (2005) */
      fprintf(stderr, "libdft: Spherical average (new) - ");
    } else {
      radius = otf->lj_params.h;
      fprintf(stderr, "libdft: Spherical average (original) - ");
    }

#ifdef DFT_OT_1D
    if(nx == 1 && ny == 1)
      rgrid_adaptive_map(otf->spherical_avg, dft_common_spherical_avg_1d, &radius, min_substeps, max_substeps, 0.01 / GRID_AUTOK);
    else 
#endif
      rgrid_adaptive_map(otf->spherical_avg, dft_common_spherical_avg, &radius, min_substeps, max_substeps, 0.01 / GRID_AUTOK);
    rgrid_fft(otf->spherical_avg);
    /* Scaling of sph. avg. so that the integral is exactly 1 */
    if(nx != 1 || ny != 1)      
      rgrid_multiply_fft(otf->spherical_avg, 1.0 / (step * step * step * (REAL) rgrid_cvalue_at_index(otf->spherical_avg, 0, 0, 0)));
    fprintf(stderr, "Done.\n");
    
    if(model & DFT_OT_KC) {
      fprintf(stderr, "libdft: Kinetic correlation - ");	
      inv_width = 1.0 / otf->l_g;
#ifdef DFT_OT_1D
      if(nx == 1 && ny == 1) {
        rgrid_adaptive_map(otf->gaussian_tf, dft_common_gaussian_1d, &inv_width, min_substeps, max_substeps, 0.01 / GRID_AUTOK);
        RGRID_GRADIENT_Z(otf->gaussian_tf, otf->gaussian_z_tf);
        rgrid_fft(otf->gaussian_z_tf);
        rgrid_fft(otf->gaussian_tf);
      } else {
#endif
        rgrid_adaptive_map(otf->gaussian_tf, dft_common_gaussian, &inv_width, min_substeps, max_substeps, 0.01 / GRID_AUTOK);
        RGRID_GRADIENT_X(otf->gaussian_tf, otf->gaussian_x_tf);
        RGRID_GRADIENT_Y(otf->gaussian_tf, otf->gaussian_y_tf);
        RGRID_GRADIENT_Z(otf->gaussian_tf, otf->gaussian_z_tf);
        rgrid_fft(otf->gaussian_x_tf);
        rgrid_fft(otf->gaussian_y_tf);
        rgrid_fft(otf->gaussian_z_tf);
        rgrid_fft(otf->gaussian_tf);
#ifdef DFT_OT_1D
      }
#endif
      fprintf(stderr, "Done.\n");
    }
    
    if(model & DFT_OT_BACKFLOW) {
      fprintf(stderr, "libdft: Backflow - ");
#ifdef DFT_OT_1D
      if(nx == 1 && ny == 1)
        rgrid_adaptive_map(otf->backflow_pot, dft_ot_backflow_pot_1d, &(otf->bf_params), min_substeps, max_substeps, 0.01 / GRID_AUTOK);
      else
#endif
        rgrid_adaptive_map(otf->backflow_pot, dft_ot_backflow_pot, &(otf->bf_params), min_substeps, max_substeps, 0.01 / GRID_AUTOK);
      rgrid_fft(otf->backflow_pot);
      fprintf(stderr, "Done.\n");
    }
  }

  /* Allocate workspaces based on the functional */
  otf->density = rgrid_alloc(nx, ny, nz, step, RGRID_PERIODIC_BOUNDARY, 0, "OT Density");
  otf->workspace1 = NULL;
  otf->workspace2 = NULL;
  otf->workspace3 = NULL;
  otf->workspace4 = NULL;
  otf->workspace5 = NULL;
  otf->workspace6 = NULL;
  otf->workspace7 = NULL;
  otf->workspace8 = NULL;
  otf->workspace9 = NULL;

  if(model & DFT_ZERO) return otf; /* No workspaces needed */
  otf->workspace1 = rgrid_alloc(nx, ny, nz, step, RGRID_PERIODIC_BOUNDARY, 0, "OT Workspace 1");
  if((model & DFT_GP) || (model & DFT_GP2)) return otf;
  otf->workspace2 = rgrid_alloc(nx, ny, nz, step, RGRID_PERIODIC_BOUNDARY, 0, "OT Workspace 2");
  otf->workspace3 = rgrid_alloc(nx, ny, nz, step, RGRID_PERIODIC_BOUNDARY, 0, "OT Workspace 3");
  if(!(model & DFT_OT_KC) && !(model & DFT_OT_BACKFLOW)) return otf; // plain OT or DR
  otf->workspace4 = rgrid_alloc(nx, ny, nz, step, RGRID_PERIODIC_BOUNDARY, 0, "OT Workspace 4");
  otf->workspace5 = rgrid_alloc(nx, ny, nz, step, RGRID_PERIODIC_BOUNDARY, 0, "OT Workspace 5");
  otf->workspace6 = rgrid_alloc(nx, ny, nz, step, RGRID_PERIODIC_BOUNDARY, 0, "OT Workspace 6");
  if(!(model & DFT_OT_BACKFLOW)) return otf;
  otf->workspace7 = rgrid_alloc(nx, ny, nz, step, RGRID_PERIODIC_BOUNDARY, 0, "OT Workspace 7");
  otf->workspace8 = rgrid_alloc(nx, ny, nz, step, RGRID_PERIODIC_BOUNDARY, 0, "OT Workspace 8");
  otf->workspace9 = rgrid_alloc(nx, ny, nz, step, RGRID_PERIODIC_BOUNDARY, 0, "OT Workspace 9");

  return otf;
}

/*
 * Free OT functional structure.
 *
 * otf = functional structure to be freed (allocated previously
 *       by dft_ot_alloc()).
 *
 * No return value.
 *
 */

EXPORT void dft_ot_free(dft_ot_functional *otf) {

  if (otf) {
    if (otf->lennard_jones) rgrid_free(otf->lennard_jones);
    if (otf->spherical_avg) rgrid_free(otf->spherical_avg);
    if (otf->gaussian_tf) rgrid_free(otf->gaussian_tf);
    if (otf->gaussian_x_tf) rgrid_free(otf->gaussian_x_tf);
    if (otf->gaussian_y_tf) rgrid_free(otf->gaussian_y_tf);
    if (otf->gaussian_z_tf) rgrid_free(otf->gaussian_z_tf);
    if (otf->backflow_pot) rgrid_free(otf->backflow_pot);
    if (otf->density) rgrid_free(otf->density);
    if (otf->workspace1) rgrid_free(otf->workspace1);
    if (otf->workspace2) rgrid_free(otf->workspace2);
    if (otf->workspace3) rgrid_free(otf->workspace3);
    if (otf->workspace4) rgrid_free(otf->workspace4);
    if (otf->workspace5) rgrid_free(otf->workspace5);
    if (otf->workspace6) rgrid_free(otf->workspace6);
    if (otf->workspace7) rgrid_free(otf->workspace7);
    if (otf->workspace8) rgrid_free(otf->workspace8);
    if (otf->workspace9) rgrid_free(otf->workspace9);
    free(otf);
  }
}

/*
 * Calculate the non-linear potential grid.
 *
 * otf        = OT functional structure (input; dft_ot_functional *).
 * potential  = Potential grid where the result will be stored (output; cgrid *).
 *              NOTE: the potential will be added to this (may want to zero it first)
 * wf         = Wavefunction (input; wf *).
 *
 * No return value.
 *
 * Grid usage in otf structure:
 * density    = Liquid helium density grid (all functionals).
 * workspace1 = Workspace grid. GP access up to this point.
 * workspace2 = Workspace grid.
 * workspace3 = Workspace grid. Basic OT access up to this point.
 * workspace4 = Workspace grid.
 * workspace5 = Workspace grid.
 * workspace6 = Workspace grid. KC access up to this point.
 * workspace7 = Workspace grid. 
 * workspace8 = Workspace grid. 
 * workspace9 = Workspace grid. BF access up to this point.
 *
 */

EXPORT void dft_ot_potential(dft_ot_functional *otf, cgrid *potential, wf *wf) {

  rgrid *workspace1, *workspace2, *workspace3, *workspace4, *workspace5, *workspace6, *workspace7, *workspace8, *workspace9;
  rgrid *density;

  density = otf->density;
  grid_wf_density(wf, density);
  workspace1 = otf->workspace1;
  workspace2 = otf->workspace2;
  workspace3 = otf->workspace3;
  workspace4 = otf->workspace4;
  workspace5 = otf->workspace5;
  workspace6 = otf->workspace6;
  workspace7 = otf->workspace7;
  workspace8 = otf->workspace8;
  workspace9 = otf->workspace9;

  if(otf->model & DFT_ZERO) {
    fprintf(stderr, "libdft: Warning - zero potential used.\n");
    cgrid_zero(potential);
    return;
  }

  if((otf->model & DFT_GP) || (otf->model & DFT_GP2)) {
    rgrid_claim(workspace1);
    rgrid_copy(workspace1, density);
    rgrid_multiply(workspace1, otf->mu0 / otf->rho0); // positive value
    grid_add_real_to_complex_re(potential, workspace1);
    rgrid_release(workspace1);
    return;
  }

  /* Lennard-Jones */  
  /* int rho(r') Vlj(r-r') dr' */
  /* workspace1 = FFT of density */
  rgrid_claim(workspace1);
  rgrid_claim(workspace2);
  dft_ot_add_lennard_jones_potential(otf, potential, density, workspace1 /* rho_tf */, workspace2);
  rgrid_release(workspace2);

  /* Non-linear local correlation */
  /* note workspace1 = fft of \rho */
  rgrid_claim(workspace2);
  rgrid_claim(workspace3);
  dft_ot_add_local_correlation_potential(otf, potential, density, workspace1 /* rho_tf */, workspace2, workspace3);
  rgrid_release(workspace2);
  rgrid_release(workspace3);

  /* Non-local correlation for kinetic energy (workspace1 = FFT(rho)) */
  if(otf->model & DFT_OT_KC) {
    rgrid_claim(workspace2); rgrid_claim(workspace3); rgrid_claim(workspace4);
    rgrid_claim(workspace5); rgrid_claim(workspace6);
    dft_ot_add_nonlocal_correlation_potential(otf, potential, density, workspace1 /* rho_tf */, workspace2, workspace3, workspace4, workspace5, workspace6);
    rgrid_release(workspace2); rgrid_release(workspace3); rgrid_release(workspace4);
    rgrid_release(workspace5); rgrid_release(workspace6);
  }
  /* workspace1 no longer needed */
  rgrid_release(workspace1);

  /* Barranco's penalty term */
  if((otf->model & DFT_OT_HD) || (otf->model & DFT_OT_HD2)) {
    rgrid_claim(workspace1);
    dft_ot_add_barranco(otf, potential, density, workspace1);
    rgrid_release(workspace1);
  }

  if(otf->model & DFT_OT_BACKFLOW) {
    /* wf, veloc_x(1), veloc_y(2), veloc_z(3), wrk(4) */
    // grid_wf_momentum(wf, workspace1, workspace2, workspace3, ...);   But can't do since we don't have cmplx workspaces
    rgrid_claim(workspace1); rgrid_claim(workspace2); rgrid_claim(workspace3);
    rgrid_claim(workspace4); rgrid_claim(workspace5); rgrid_claim(workspace6);
    rgrid_claim(workspace7); rgrid_claim(workspace8); rgrid_claim(workspace9);
#ifdef DFT_OT_1D
    if(density->nx == 1 && density->ny == 1) 
      grid_wf_velocity_z(wf, workspace3, otf->veloc_cutoff);
    else
#endif
      grid_wf_velocity(wf, workspace1, workspace2, workspace3, otf->veloc_cutoff);
    dft_ot_backflow_potential(otf, potential, density, workspace1 /* veloc_x */, workspace2 /* veloc_y */, workspace3 /* veloc_z */, workspace4, workspace5, workspace6, workspace7, workspace8, workspace9);
    rgrid_release(workspace1); rgrid_release(workspace2); rgrid_release(workspace3);
    rgrid_release(workspace4); rgrid_release(workspace5); rgrid_release(workspace6);
    rgrid_release(workspace7); rgrid_release(workspace8); rgrid_release(workspace9);
  }

  if(otf->model >= DFT_OT_T400MK && !(otf->model & DFT_DR)) {
    /* include the ideal gas contribution */
    rgrid_claim(workspace1);
    dft_ot_add_ancilotto(otf, potential, density, workspace1);
    rgrid_release(workspace1);
  }
}

/*
 * Lennard-Jones potential.
 *
 */

EXPORT inline void dft_ot_add_lennard_jones_potential(dft_ot_functional *otf, cgrid *potential, rgrid *density, rgrid *workspace1, rgrid *workspace2) {

  rgrid_copy(workspace1, density);
  rgrid_fft(workspace1);

  rgrid_fft_convolute(workspace2, workspace1, otf->lennard_jones);  // Don't overwrite workspace1 - needed later
  rgrid_inverse_fft(workspace2);
  grid_add_real_to_complex_re(potential, workspace2);
  /* leave FFT(rho) in workspace1 (used later in local correlation potential as rho_tf) */
}

/*
 * Local correlation potential.
 *
 */

EXPORT inline void dft_ot_add_local_correlation_potential(dft_ot_functional *otf, cgrid *potential, rgrid *rho, rgrid *rho_tf, rgrid *workspace1, rgrid *workspace2) {

  /* workspace1 = \bar{\rho} */
  rgrid_fft_convolute(workspace1, rho_tf, otf->spherical_avg);
  rgrid_inverse_fft(workspace1); 

  /* C2.1 */
  if(otf->model & DFT_DR)
    rgrid_power(workspace2, workspace1, otf->c2_exp);
  else
    rgrid_ipower(workspace2, workspace1, (INT) otf->c2_exp);
  rgrid_multiply(workspace2, otf->c2 / 2.0);
  grid_add_real_to_complex_re(potential, workspace2);

  /* C3.1 */
  if(otf->model & DFT_DR)
    rgrid_power(workspace2, workspace1, otf->c3_exp);
  else
    rgrid_ipower(workspace2, workspace1, (INT) otf->c3_exp);
  rgrid_multiply(workspace2, otf->c3 / 3.0);
  grid_add_real_to_complex_re(potential, workspace2);

  /* C2.2 & C3.2 */
  if(otf->model & DFT_DR)  {
    rgrid_power(workspace2, workspace1, otf->c2_exp - 1.0);
    rgrid_power(workspace1, workspace1, otf->c3_exp - 1.0);
  } else {  
    rgrid_ipower(workspace2, workspace1, (INT) (otf->c2_exp - 1.0));
    rgrid_ipower(workspace1, workspace1, (INT) (otf->c3_exp - 1.0));
  }   
  rgrid_multiply(workspace2, otf->c2 * otf->c2_exp / 2.0);  // For OT, c2_exp / 2 = 1
  rgrid_multiply(workspace1, otf->c3 * otf->c3_exp / 3.0);  // For OT, c3_exp / 3 = 1
  rgrid_sum(workspace2, workspace2, workspace1);

  rgrid_product(workspace2, workspace2, rho);
  rgrid_fft(workspace2);
  rgrid_fft_convolute(workspace2, workspace2, otf->spherical_avg);
  rgrid_inverse_fft(workspace2);
  grid_add_real_to_complex_re(potential, workspace2);
}

/* 
 * Nonlocal correlation potential.
 *
 */

static inline void dft_ot_add_nonlocal_correlation_potential(dft_ot_functional *otf, cgrid *potential, rgrid *rho, rgrid *rho_tf, rgrid *workspace1, rgrid *workspace2, rgrid *workspace3, rgrid *workspace4, rgrid *workspace5) {

  /* rho^tilde(r) = int F(r-r') rho(r') dr' */
  /* NOTE: rho_tf from LJ (workspace1 there). */
  rgrid_fft_convolute(workspace1, otf->gaussian_tf, rho_tf);
  rgrid_inverse_fft(workspace1);
  /* workspace1 = rho_st = 1 - 1/\tilde{\rho}/\rho_{0s} */
  rgrid_multiply(workspace1, -1.0 / otf->rho_0s);
  rgrid_add(workspace1, 1.0);

  if(rho->nx > 1) dft_ot_add_nonlocal_correlation_potential_x(otf, potential, rho, rho_tf, workspace1 /* rho_st */, workspace2, workspace3, workspace4, workspace5);
  if(rho->ny > 1) dft_ot_add_nonlocal_correlation_potential_y(otf, potential, rho, rho_tf, workspace1 /* rho_st */, workspace2, workspace3, workspace4, workspace5);
  dft_ot_add_nonlocal_correlation_potential_z(otf, potential, rho, rho_tf, workspace1 /* rho_st */, workspace2, workspace3, workspace4, workspace5);
}

/*
 * X component to nonlocal correlation potential.
 *
 */

static inline void dft_ot_add_nonlocal_correlation_potential_x(dft_ot_functional *otf, cgrid *potential, rgrid *rho, rgrid *rho_tf, rgrid *rho_st, rgrid *workspace1, rgrid *workspace2, rgrid *workspace3, rgrid *workspace4) {

  REAL c;

  // All workspaces already claimed above
  c = otf->alpha_s / (2.0 * otf->mass);

  /* workspace1 = (d/dx) rho */
  RGRID_GRADIENT_X(rho, workspace1);

  /*** 1st term ***/

  /* Construct workspace2 = FFT(G) = FFT((d/dx) \rho(r_1) * (1 - \tilde{\rho(r_1)} / \rho_{0s})) */
  rgrid_product(workspace2, workspace1, rho_st); // rho_st = (1 - \tilde{\rho(r_1)} / \rho_{0s})
  rgrid_fft(workspace2);

  /* 1st term: c convolute [((d/dx) F) . G] */
  rgrid_fft_convolute(workspace3, otf->gaussian_x_tf, workspace2);
  rgrid_inverse_fft(workspace3);
  rgrid_product(workspace3, workspace3, rho_st);
  rgrid_multiply(workspace3, c);
  grid_add_real_to_complex_re(potential, workspace3);  

  /* in use: workspace1 (grad rho), workspace2 (FFT(G)) */

  /*** 2nd term ***/
  
  /* Construct workspace3 = J = convolution(F G) */
  rgrid_fft_convolute(workspace3, otf->gaussian_tf, workspace2);
  rgrid_inverse_fft(workspace3);

  /* Construct workspace4 = FFT(H) = FFT((d/dx) \rho * J) */
  rgrid_copy(workspace4, workspace3);
  rgrid_product(workspace4, workspace1, workspace4);
  rgrid_fft(workspace4);

  /* 2nd term: c convolute(F H) */
  rgrid_fft_convolute(workspace4, otf->gaussian_tf, workspace4);
  rgrid_inverse_fft(workspace4);
  rgrid_multiply(workspace4, c);
  grid_add_real_to_complex_re(potential, workspace4);

  /* workspace1 (grad rho), workspace3 (J) */

  /*** 3rd term ***/
  
  /* -c J . convolute((d/dx)F \rho) */
  rgrid_fft_convolute(workspace2, otf->gaussian_x_tf, rho_tf);
  rgrid_inverse_fft(workspace2);
  rgrid_product(workspace2, workspace2, workspace3);
  rgrid_multiply(workspace2, -c);
  grid_add_real_to_complex_re(potential, workspace2);
}

/*
 * Y component to nonlocal correlation potential.
 *
 */

static inline void dft_ot_add_nonlocal_correlation_potential_y(dft_ot_functional *otf, cgrid *potential, rgrid *rho, rgrid *rho_tf, rgrid *rho_st, rgrid *workspace1, rgrid *workspace2, rgrid *workspace3, rgrid *workspace4) {

  REAL c;

  // All workspaces already claimed above
  c = otf->alpha_s / (2.0 * otf->mass);

  /* workspace1 = (d/dy) rho */
  RGRID_GRADIENT_Y(rho, workspace1);

  /*** 1st term ***/

  /* Construct workspace2 = FFT(G) = FFT((d/dy) \rho(r_1) * (1 - \tilde{\rho(r_1)} / \rho_{0s})) */
  rgrid_product(workspace2, workspace1, rho_st);
  rgrid_fft(workspace2);

  /* 1st term: c convolute [((d/dy) F) . G] */
  rgrid_fft_convolute(workspace3, otf->gaussian_y_tf, workspace2);
  rgrid_inverse_fft(workspace3);
  rgrid_product(workspace3, workspace3, rho_st);
  rgrid_multiply(workspace3, c);
  grid_add_real_to_complex_re(potential, workspace3);  

  /* in use: workspace1 (grad rho), workspace2 (FFT(G)) */

  /*** 2nd term ***/
  
  /* Construct workspace3 = J = convolution(F G) */
  rgrid_fft_convolute(workspace3, otf->gaussian_tf, workspace2);
  rgrid_inverse_fft(workspace3);

  /* Construct workspace4 = FFT(H) = FFT((d/dy) \rho * J) */
  rgrid_copy(workspace4, workspace3);
  rgrid_product(workspace4, workspace1, workspace4);
  rgrid_fft(workspace4);

  /* 2nd term: c convolute(F H) */
  rgrid_fft_convolute(workspace4, otf->gaussian_tf, workspace4);
  rgrid_inverse_fft(workspace4);
  rgrid_multiply(workspace4, c);
  grid_add_real_to_complex_re(potential, workspace4);

  /* workspace1 (grad rho), workspace3 (J) */

  /*** 3rd term ***/
  
  /* -c J . convolute((d/dy)F \rho) */

  rgrid_fft_convolute(workspace2, otf->gaussian_y_tf, rho_tf);
  rgrid_inverse_fft(workspace2);
  rgrid_product(workspace2, workspace2, workspace3);
  rgrid_multiply(workspace2, -c);
  grid_add_real_to_complex_re(potential, workspace2);
}

/*
 * Z component to nonlocal correlation potential.
 *
 */

static inline void dft_ot_add_nonlocal_correlation_potential_z(dft_ot_functional *otf, cgrid *potential, rgrid *rho, rgrid *rho_tf, rgrid *rho_st, rgrid *workspace1, rgrid *workspace2, rgrid *workspace3, rgrid *workspace4) {

  REAL c;

  // All workspaces already claimed above
  c = otf->alpha_s / (2.0 * otf->mass);

  /* workspace1 = (d/dz) rho */
  RGRID_GRADIENT_Z(rho, workspace1);

  /*** 1st term ***/

  /* Construct workspace2 = FFT(G) = FFT((d/dz) \rho(r_1) * (1 - \tilde{\rho(r_1)} / \rho_{0s})) */
  rgrid_product(workspace2, workspace1, rho_st);
  rgrid_fft(workspace2);

  /* 1st term: c convolute [((d/dz) F) . G] */
  rgrid_fft_convolute(workspace3, otf->gaussian_z_tf, workspace2);
  rgrid_inverse_fft(workspace3);
  rgrid_product(workspace3, workspace3, rho_st);
  rgrid_multiply(workspace3, c);

  grid_add_real_to_complex_re(potential, workspace3);  

  /* in use: workspace1 (grad rho), workspace2 (FFT(G)) */

  /*** 2nd term ***/
  
  /* Construct workspace3 = J = convolution(F G) */
  rgrid_fft_convolute(workspace3, otf->gaussian_tf, workspace2);
  rgrid_inverse_fft(workspace3);

  /* Construct workspace4 = FFT(H) = FFT((d/dz) \rho * J) */
  rgrid_copy(workspace4, workspace3);
  rgrid_product(workspace4, workspace1, workspace4);
  rgrid_fft(workspace4);

  /* 2nd term: c convolute(F H) */
  rgrid_fft_convolute(workspace4, otf->gaussian_tf, workspace4);
  rgrid_inverse_fft(workspace4);
  rgrid_multiply(workspace4, c);
  grid_add_real_to_complex_re(potential, workspace4);

  /* workspace1 (grad rho), workspace3 (J) */

  /*** 3rd term ***/
  
  /* -c J . convolute((d/dz)F \rho) */

  rgrid_fft_convolute(workspace2, otf->gaussian_z_tf, rho_tf);
  rgrid_inverse_fft(workspace2);
  rgrid_product(workspace2, workspace2, workspace3);
  rgrid_multiply(workspace2, -c);
  grid_add_real_to_complex_re(potential, workspace2);
}

/* 
 * Thermal DFT.
 *
 */

static inline void dft_ot_add_ancilotto(dft_ot_functional *otf, cgrid *potential, rgrid *rho, rgrid *workspace1) {

  grid_func6a_operate_one(workspace1, rho, otf->mass, otf->temp, otf->c4);
  grid_add_real_to_complex_re(potential, workspace1);
}

/* 
 * High density correction.
 *
 */

static inline void dft_ot_add_barranco(dft_ot_functional *otf, cgrid *potential, rgrid *rho, rgrid *workspace1) {

  grid_func4_operate_one(workspace1, rho, otf->beta, otf->rhom, otf->C);
  grid_add_real_to_complex_re(potential, workspace1);
}

/*
 * Add the backflow non-linear potential.
 *
 * otf            = OT functional structure (dft_ot_functional *).
 * potential      = potential grid (output). Not cleared (cgrid *).
 * density        = liquid density grid (input; rgrid *).
 * veloc_x        = velocity grid (with respect to X; input - overwritten on exit; rgrid *).
 * veloc_y        = velocity grid (with respect to Y; input - overwritten on exit; rgrid *).
 * veloc_z        = velocity grid (with respect to Z; input - overwritten on exit; rgrid *).
 * workspace1     = Workspace grid (must be allocated by the user; rgrid *).
 * workspace2     = Workspace grid (must be allocated by the user; rgrid *).
 * workspace3     = Workspace grid (must be allocated by the user; rgrid *).
 * workspace4     = Workspace grid (must be allocated by the user; rgrid *).
 * workspace5     = Workspace grid (must be allocated by the user; rgrid *).
 * workspace6     = Workspace grid (must be allocated by the user; rgrid *).
 *
 * No return value.
 *
 */

EXPORT void dft_ot_backflow_potential(dft_ot_functional *otf, cgrid *potential, rgrid *density, rgrid *veloc_x, rgrid *veloc_y, rgrid *veloc_z, rgrid *workspace1, rgrid *workspace2, rgrid *workspace3, rgrid *workspace4, rgrid *workspace5, rgrid *workspace6) {

  /* Calculate A (workspace1) [scalar] */
  if((otf->model & DFT_OT_HD) || (otf->model & DFT_OT_HD2))
    grid_func2_operate_one(workspace1, density, otf->xi, otf->rhobf); /* rho -> g rho */
  else
    rgrid_copy(workspace1, density);   /* Original BF code (without the MM density cutoff), just rho */
  rgrid_fft(workspace1);
  rgrid_fft_convolute(workspace1, workspace1, otf->backflow_pot);
  rgrid_inverse_fft(workspace1);

  /* Calculate C (workspace2) [scalar] */
  rgrid_product(workspace2, veloc_z, veloc_z);
  if(density->nx != 1 || density->ny != 1) { // 1-D x & y velocity components zero
    rgrid_add_scaled_product(workspace2, 1.0, veloc_y, veloc_y);
    rgrid_add_scaled_product(workspace2, 1.0, veloc_x, veloc_x);
  }

  if((otf->model & DFT_OT_HD) || (otf->model & DFT_OT_HD2))
    grid_func2_operate_one_product(workspace2, workspace2, density, otf->xi, otf->rhobf);  /* multiply by g rho */
  else
    rgrid_product(workspace2, workspace2, density);  /* orignal: multiply by just rho */
  rgrid_fft(workspace2);
  rgrid_fft_convolute(workspace2, workspace2, otf->backflow_pot);
  rgrid_inverse_fft(workspace2);

  /* Calculate B (workspace3 (B_x), workspace4 (B_y), workspace5 (B_z)) [vector] */
  if(density->nx != 1 || density->ny != 1) {
    /* B_X */
    if((otf->model & DFT_OT_HD) || (otf->model & DFT_OT_HD2))
      grid_func2_operate_one_product(workspace3, veloc_x, density, otf->xi, otf->rhobf); /* MM: g rho */
    else
      rgrid_product(workspace3, veloc_x, density); /* original: just rho */
    rgrid_fft(workspace3);
    rgrid_fft_convolute(workspace3, workspace3, otf->backflow_pot);
    rgrid_inverse_fft(workspace3);
  
    /* B_Y */
    if((otf->model & DFT_OT_HD) || (otf->model & DFT_OT_HD2))
      grid_func2_operate_one_product(workspace4, veloc_y, density, otf->xi, otf->rhobf); /* MM: g rho */
    else
      rgrid_product(workspace4, veloc_y, density); /* original: just rho */
    rgrid_fft(workspace4);
    rgrid_fft_convolute(workspace4, workspace4, otf->backflow_pot);
    rgrid_inverse_fft(workspace4);
  }

  /* B_Z */
  if((otf->model & DFT_OT_HD) || (otf->model & DFT_OT_HD2))
    grid_func2_operate_one_product(workspace5, veloc_z, density, otf->xi, otf->rhobf); /* MM: g rho */
  else
    rgrid_product(workspace5, veloc_z, density); /* original: just rho */
  rgrid_fft(workspace5);
  rgrid_fft_convolute(workspace5, workspace5, otf->backflow_pot);
  rgrid_inverse_fft(workspace5);

  /* 1. Calculate the real part of the potential */

  /* -(m/2) (v(r) . (v(r)A(r) - 2B(r)) + C(r))  = -(m/2) [ A(v_x^2 + v_y^2 + v_z^2) - 2v_x B_x - 2v_y B_y - 2v_z B_z + C] */
  rgrid_product(workspace6, veloc_z, veloc_z);
  if(density->nx != 1 || density->ny != 1) {
    rgrid_add_scaled_product(workspace6, 1.0, veloc_y, veloc_y);
    rgrid_add_scaled_product(workspace6, 1.0, veloc_x, veloc_x);
  }
  rgrid_product(workspace6, workspace6, workspace1);
  if(density->nx != 1 || density->ny != 1) {
    rgrid_add_scaled_product(workspace6, -2.0, veloc_x, workspace3);
    rgrid_add_scaled_product(workspace6, -2.0, veloc_y, workspace4);
  }
  rgrid_add_scaled_product(workspace6, -2.0, veloc_z, workspace5);
  rgrid_sum(workspace6, workspace6, workspace2);
  if((otf->model & DFT_OT_HD) || (otf->model & DFT_OT_HD2)) /* multiply by [rho x (dG/drho)(rho) + G(rho)] */
    grid_func3_operate_one_product(workspace6, workspace6, density, otf->xi, otf->rhobf);
  rgrid_multiply(workspace6, -0.5 * otf->mass);

  grid_add_real_to_complex_re(potential, workspace6);

  /* workspace2 (C), workspace6 not used after this point */

  /* 2. Calculate the imaginary part of the potential */

  rgrid_zero(workspace6);
  
  /* v_x -> v_xA - B_x, v_y -> v_yA - B_y, v_z -> v_zA - B_z (velocities are overwritten here) */
  if(density->nx != 1 || density->ny != 1) {
    rgrid_product(veloc_x, veloc_x, workspace1);
    rgrid_add_scaled(veloc_x, -1.0, workspace3);
    rgrid_product(veloc_y, veloc_y, workspace1);
    rgrid_add_scaled(veloc_y, -1.0, workspace4);
  }
  rgrid_product(veloc_z, veloc_z, workspace1);
  rgrid_add_scaled(veloc_z, -1.0, workspace5);

  /* 1.1 (1/2) (drho/dx)/rho * (v_xA - B_x) */  
  if(density->nx != 1 || density->ny != 1) {
    if((otf->model & DFT_OT_HD) || (otf->model & DFT_OT_HD2)) {
      grid_func2_operate_one(workspace1, density, otf->xi, otf->rhobf);
      RGRID_GRADIENT_X(workspace1, workspace2);
    } else
      RGRID_GRADIENT_X(density, workspace2);
    rgrid_division_eps(workspace2, workspace2, density, otf->div_epsilon);
    rgrid_add_scaled_product(workspace6, 0.5, workspace2, veloc_x);
  }

  /* 1.2 (1/2) (drho/dy)/rho * (v_yA - B_y) */
  if(density->nx != 1 || density->ny != 1) {
    if((otf->model & DFT_OT_HD) || (otf->model & DFT_OT_HD2)) {
      /* grid_func2_operate_one(workspace1, density, otf->xi, otf->rhobf); */ /* done above */
      RGRID_GRADIENT_Y(workspace1, workspace2);  
    } else
      RGRID_GRADIENT_Y(density, workspace2);
    rgrid_division_eps(workspace2, workspace2, density, otf->div_epsilon);
    rgrid_add_scaled_product(workspace6, 0.5, workspace2, veloc_y);
  }

  /* 1.3 (1/2) (drho/dz)/rho * (v_zA - B_z) */
  if((otf->model & DFT_OT_HD) || (otf->model & DFT_OT_HD2)) {
    if(density->nx == 1 && density->ny == 1) grid_func2_operate_one(workspace1, density, otf->xi, otf->rhobf); // May have been done above
    RGRID_GRADIENT_Z(workspace1, workspace2);  
  } else RGRID_GRADIENT_Z(density, workspace2);
  rgrid_division_eps(workspace2, workspace2, density, otf->div_epsilon);
  rgrid_add_scaled_product(workspace6, 0.5, workspace2, veloc_z);

  /* 2.1 (1/2) (d/dx) (v_xA - B_x) */
  if(density->nx != 1 || density->ny != 1) {
    RGRID_GRADIENT_X(veloc_x, workspace2);
    if((otf->model & DFT_OT_HD) || (otf->model & DFT_OT_HD2))
      grid_func1_operate_one_product(workspace2, workspace2, density, otf->xi, otf->rhobf);   /* multiply by g */
    rgrid_add_scaled(workspace6, 0.5, workspace2);
  }

  /* 2.2 (1/2) (d/dy) (v_yA - B_y) */
  if(density->nx != 1 || density->ny != 1) {
    RGRID_GRADIENT_Y(veloc_y, workspace2);
    if((otf->model & DFT_OT_HD) || (otf->model & DFT_OT_HD2))
      grid_func1_operate_one_product(workspace2, workspace2, density, otf->xi, otf->rhobf);   /* multiply by g */
    rgrid_add_scaled(workspace6, 0.5, workspace2);
  }

  /* 2.3 (1/2) (d/dz) (v_zA - B_z) */
  RGRID_GRADIENT_Z(veloc_z, workspace2);
  if((otf->model & DFT_OT_HD) || (otf->model & DFT_OT_HD2))
    grid_func1_operate_one_product(workspace2, workspace2, density, otf->xi, otf->rhobf);   /* multiply by g */
  rgrid_add_scaled(workspace6, 0.5, workspace2);

  grid_add_real_to_complex_im(potential, workspace6);
}

/*
 * Initialize the OTF structure.
 *
 */

EXPORT inline void dft_ot_temperature(dft_ot_functional *otf, INT model) {

  if((otf->model & DFT_OT_HD) || (otf->model & DFT_OT_HD2)) { /* high density penalty */
    otf->beta = (40.0 / (GRID_AUTOANG * GRID_AUTOANG * GRID_AUTOANG));
    otf->rhom = (0.37 * GRID_AUTOANG * GRID_AUTOANG * GRID_AUTOANG);
    otf->C = 0.1; /* a.u. */
  } else {
    otf->beta = 0.0;
    otf->rhom = 0.0;
    otf->C = 0.0;
  }

  if(model & DFT_DR) { /* Dupont-Roc */
    otf->b = 0.0;
    otf->c2 = 10455400.0; /* K Angs^(3 + 3\gamma) */
    otf->c2_exp = 1.0 + 2.8; /* 1 + gamma */
    otf->c3 = 0.0; /* not used */
    otf->c3_exp = 1.0;
    otf->c4 = 0.0;  
    otf->temp = 0.0;
    otf->rho0 = 0.0218360;
    otf->lj_params.h = 2.377;    /* Angs */
  }                                                                                                                                    
  
  if(model < DFT_OT_T0MK) { /* 0 */
    otf->b = -718.99;
    otf->c2 = -2.411857E4;
    otf->c2_exp = 2;
    otf->c3 = 1.858496E6;
    otf->c3_exp = 3;
    otf->c4 = 0.0;
    otf->temp = 0.0;
    otf->rho0 = 0.0218360;     /* Angs^-3 */
    otf->lj_params.h = 2.1903; /* Angs */
  }

  if(model & DFT_OT_T0MK) { /* 1 */
    otf->b = -719.2435;
    otf->c2 = -24258.88;
    otf->c2_exp = 2;
    otf->c3 = 1865257.0;
    otf->c3_exp = 3;
    otf->c4 = 0.0;
    otf->temp = 0.0;
    otf->rho0 = 0.0218354;
    otf->lj_params.h = 2.19035;
  }

  if(model & DFT_OT_T400MK) { /* 2 */
    otf->b = -714.2174;
    otf->c2 = -24566.29;
    otf->c2_exp = 2;
    otf->c3 = 1873203.0;
    otf->c3_exp = 3;
    otf->c4 = 0.98004;
    otf->temp = 0.4;
    otf->rho0 = 0.0218351;
    otf->lj_params.h = 2.18982;
  }

  if(model & DFT_OT_T600MK) { /* 3 */
    otf->b = -705.1319;
    otf->c2 = -25124.17;
    otf->c2_exp = 2;
    otf->c3 = 1887707.0;
    otf->c3_exp = 3;
    otf->c4 = 0.99915;
    otf->temp = 0.6;
    otf->rho0 = 0.0218346;
    otf->lj_params.h = 2.18887;
  }

  if(model & DFT_OT_T800MK) { /* 4 */
    otf->b = -690.4745;
    otf->c2 = -26027.12;
    otf->c2_exp = 2;
    otf->c3 = 1911283.0;
    otf->c3_exp = 3;
    otf->c4 = 0.99548;
    otf->temp = 0.8;
    otf->rho0 = 0.0218331;
    otf->lj_params.h = 2.18735;
  }

  if(model & DFT_OT_T1200MK) { /* 5 */
    otf->b = -646.5135;
    otf->c2 = -28582.81;
    otf->c2_exp = 2;
    otf->c3 = 1973737.0;
    otf->c3_exp = 3;
    otf->c4 = 0.99666;
    otf->temp = 1.2;
    otf->rho0 = 0.0218298;
    otf->lj_params.h = 2.18287;
  }

  if(model & DFT_OT_T1400MK) { /* 6 */
    otf->b = -625.8123;
    otf->c2 = -29434.03;
    otf->c2_exp = 2;
    otf->c3 = 1984068.0;
    otf->c3_exp = 3;
    otf->c4 = 0.99829;
    otf->temp = 1.4;
    otf->rho0 = 0.0218332;
    otf->lj_params.h = 2.18080;
  }

  if(model & DFT_OT_T1600MK) { /* 7 */
    otf->b = -605.9788;
    otf->c2 = -30025.96;
    otf->c2_exp = 2;
    otf->c3 = 1980898.0;
    otf->c3_exp = 3;
    otf->c4 = 1.00087;
    otf->temp = 1.6;
    otf->rho0 = 0.0218453;
    otf->lj_params.h = 2.17885;
  }

  if(model & DFT_OT_T1800MK) { /* 8 */
    otf->b = -593.8289;
    otf->c2 = -29807.56;
    otf->c2_exp = 2;
    otf->c3 = 1945685.0;
    otf->c3_exp = 3;
    otf->c4 = 1.00443;
    otf->temp = 1.8;
    otf->rho0 = 0.0218703;
    otf->lj_params.h = 2.17766;
  }

  if(model & DFT_OT_T2000MK) { /* 9 */
    otf->b = -600.8313;
    otf->c2 = -27850.96;
    otf->c2_exp = 2;
    otf->c3 = 1847407.0;
    otf->c3_exp = 3;
    otf->c4 = 1.00919;
    otf->temp = 2.0;
    otf->rho0 = 0.0219153;
    otf->lj_params.h = 2.17834;
  }

  if(model & DFT_OT_T2100MK) { /* 10 */
    otf->b = -620.9129;
    otf->c2 = -25418.15;
    otf->c2_exp = 2;
    otf->c3 = 1747494.0;
    otf->c3_exp = 3;
    otf->c4 = 1.01156;
    otf->temp = 2.1;
    otf->rho0 = 0.0219500;
    otf->lj_params.h = 2.18032;
  }

  if(model & DFT_OT_T2200MK) { /* 11 */
    otf->b = -619.2016;
    otf->c2 = -25096.68;
    otf->c2_exp = 2;
    otf->c3 = 1720802.0;
    otf->c3_exp = 3;
    otf->c4 = 1.01436;
    otf->temp = 2.2;
    otf->rho0 = 0.0219859;
    otf->lj_params.h = 2.18015;
  }

  if(model & DFT_OT_T2400MK) { /* 12 */
    otf->b = -609.0757;
    otf->c2 = -26009.98;
    otf->c2_exp = 2;
    otf->c3 = 1747943.0;
    otf->c3_exp = 3;
    otf->c4 = 1.02130;
    otf->temp = 2.4;
    otf->rho0 = 0.0218748;
    otf->lj_params.h = 2.17915;
  }

  if(model & DFT_OT_T2600MK) { /* 13 */
    otf->b = -634.0664;
    otf->c2 = -23790.66;
    otf->c2_exp = 2;
    otf->c3 = 1670707.0;
    otf->c3_exp = 3;
    otf->c4 = 1.02770;
    otf->temp = 2.6;
    otf->rho0 = 0.0217135;
    otf->lj_params.h = 2.18162;
  }

  if(model & DFT_OT_T2800MK) { /* 14 */
    otf->b = -663.9942;
    otf->c2 = -21046.37;
    otf->c2_exp = 2;
    otf->c3 = 1574611.0;
    otf->c3_exp = 3;
    otf->c4 = 1.03429;
    otf->temp = 2.8;
    otf->rho0 = 0.0215090;
    otf->lj_params.h = 2.18463;
  }

  if(model & DFT_OT_T3000MK) { /* 15 */
    otf->b = -673.6543;
    otf->c2 = -20022.76;
    otf->c2_exp = 2;
    otf->c3 = 1535887.0;
    otf->c3_exp = 3;
    otf->c4 = 1.04271;
    otf->temp = 3.0;
    otf->rho0 = 0.0212593;
    otf->lj_params.h = 2.18562;
  }

  if((model & DFT_GP) || (model & DFT_ZERO)) {  // GP tweaked to give the "right answer" for ions
    otf->temp = 0.0;
    otf->rho0 = 0.021816;
    otf->mu0 = 6.0575 / GRID_AUTOK;
    otf->c2 = otf->c2_exp = otf->c3 = otf->c3_exp = 0.0;
    /* most of the parameters are unused */
  }

  if(model & DFT_GP2) {   // GP with the correct speed of sound
    otf->temp = 0.0;
    otf->rho0 = 0.021816;
    otf->mu0 = 27.2613 / GRID_AUTOK;
    otf->c2 = otf->c2_exp = otf->c3 = otf->c3_exp = 0.0;
    /* most of the parameters are unused */
  }

  /* Backflow fix from M & M */
  otf->xi = 1E4 / (GRID_AUTOANG * GRID_AUTOANG * GRID_AUTOANG);
  otf->rhobf = 0.033 * (GRID_AUTOANG * GRID_AUTOANG * GRID_AUTOANG);

  fprintf(stderr,"libdft: Temperature = " FMT_R " K.\n", otf->temp); 

  otf->b /= GRID_AUTOK * (GRID_AUTOANG * GRID_AUTOANG * GRID_AUTOANG);
  otf->c2 /= GRID_AUTOK * POW(GRID_AUTOANG, 3.0 * otf->c2_exp);
  otf->c3 /= GRID_AUTOK * POW(GRID_AUTOANG, 3.0 * otf->c3_exp);
  otf->rho0 *= GRID_AUTOANG * GRID_AUTOANG * GRID_AUTOANG;
  otf->lj_params.h /= GRID_AUTOANG;
  otf->lj_params.sigma   = 2.556 / GRID_AUTOANG;
  otf->lj_params.epsilon = 10.22 / GRID_AUTOK;
  otf->lj_params.cval = 0.0;
  otf->rho_0s = 0.04 * GRID_AUTOANG * GRID_AUTOANG * GRID_AUTOANG;
  otf->alpha_s = 54.31 / (GRID_AUTOANG * GRID_AUTOANG * GRID_AUTOANG);
  otf->l_g = 1.0 / GRID_AUTOANG;
  otf->mass = 4.0026 / GRID_AUTOAMU;
  otf->bf_params.g11 = -19.7544;
  otf->bf_params.g12 = 12.5616 * GRID_AUTOANG * GRID_AUTOANG;
  otf->bf_params.a1  = 1.023 * GRID_AUTOANG * GRID_AUTOANG;
  otf->bf_params.g21 = -0.2395;
  otf->bf_params.g22 = 0.0312 * GRID_AUTOANG * GRID_AUTOANG;
  otf->bf_params.a2  = 0.14912 * GRID_AUTOANG * GRID_AUTOANG;

  fprintf(stderr, "libdft: C2 = " FMT_R " K Angs^" FMT_R "\n",
	  otf->c2 * GRID_AUTOK * POW(GRID_AUTOANG, 3.0 * otf->c2_exp),
	  3.0 * otf->c2_exp);
  
  fprintf(stderr, "libdft: C3 = " FMT_R " K Angs^" FMT_R "\n", 
	  otf->c3 * GRID_AUTOK * POW(GRID_AUTOANG, 3.0 * otf->c3_exp),
	  3.0 * otf->c3_exp);
  
  otf->model = model;
  otf->veloc_cutoff = 300.0 / GRID_AUTOMPS;  // Default 300 m/s
  otf->div_epsilon = 1E-5;
}
