/*
 * Orsay-Trento functional for superfluid helium in 3D.
 *
 * NOTE: This code uses FFT for evaluating all the integrals, which
 *       implies periodic boundary conditions!
 *       The 1D and 2D codes do not and any boundary conditions can be used.
 * 
 *
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <grid/grid.h>
#include <grid/au.h>
#include "dft.h"
#include "ot.h"

static void dft_ot3d_add_local_correlation_potential(dft_ot_functional *otf, cgrid3d *potential, const rgrid3d *rho, const rgrid3d *rho_tf, rgrid3d *workspace1, rgrid3d *workspace2);
static void dft_ot3d_add_nonlocal_correlation_potential(dft_ot_functional *otf, cgrid3d *potential, const rgrid3d *rho, rgrid3d *rho_tf, rgrid3d *workspace1, rgrid3d *workspace2, rgrid3d *workspace3, rgrid3d *workspace4, rgrid3d *workspace5);
static void dft_ot3d_add_nonlocal_correlation_potential_x(dft_ot_functional *otf, cgrid3d *potential, const rgrid3d *rho, rgrid3d *rho_tf, rgrid3d *rho_st, rgrid3d *workspace1, rgrid3d *workspace2, rgrid3d *workspace3, rgrid3d *workspace4);
static void dft_ot3d_add_nonlocal_correlation_potential_y(dft_ot_functional *otf, cgrid3d *potential, const rgrid3d *rho, rgrid3d *rho_tf, rgrid3d *rho_st, rgrid3d *workspace1, rgrid3d *workspace2, rgrid3d *workspace3, rgrid3d *workspace4);
static void dft_ot3d_add_nonlocal_correlation_potential_z(dft_ot_functional *otf, cgrid3d *potential, const rgrid3d *rho, rgrid3d *rho_tf, rgrid3d *rho_st, rgrid3d *workspace1, rgrid3d *workspace2, rgrid3d *workspace3, rgrid3d *workspace4);
static void dft_ot3d_add_barranco(dft_ot_functional *otf, cgrid3d *potential, const rgrid3d *rho, rgrid3d *workspace1);
static void dft_ot3d_add_ancilotto(dft_ot_functional *otf, cgrid3d *potential, const rgrid3d *rho, rgrid3d *workspace1);

static double XXX_beta, XXX_rhom, XXX_C;

/* local function */
static inline double dft_ot3d_barranco_op(double rhop) {

  double stanh = tanh(XXX_beta * (rhop - XXX_rhom));

  return (XXX_C * ((1.0 + stanh) + XXX_beta * rhop * (1.0 - stanh * stanh)));
}

/* local function */
static inline double dft_ot3d_barranco_energy_op(double rhop) {

  return (XXX_C * (1.0 + tanh(XXX_beta * (rhop - XXX_rhom))) * rhop);
}

static inline double dft_ot_backflow_pot(void *arg, double x, double y, double z) {

  double g11 = ((dft_ot_bf *) arg)->g11;
  double g12 = ((dft_ot_bf *) arg)->g12;
  double g21 = ((dft_ot_bf *) arg)->g21;
  double g22 = ((dft_ot_bf *) arg)->g22;
  double a1 = ((dft_ot_bf *) arg)->a1;
  double a2 = ((dft_ot_bf *) arg)->a2;
  double r2 = x * x + y * y + z * z;
  
  return (g11 + g12 * r2) * exp(-a1 * r2) + (g21 + g22 * r2) * exp(-a2 * r2);
}

/*
 * Allocate 3D OT functional. This must be called first.
 * 
 * model = which OT functional variant to use:
 *         DFT_OT_KC       Include the non-local kinetic energy correlation.
 *         DFT_OT_HD       Include Barranco's high density correction.
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
 *         DFT_GP          Gross-Pitaevskii equation
 *         DFT_DR          Dupont-Roc functional
 *         DFT_ZERO        No potential
 *
 *           If multiple options are needed, use and (&).
 * nx    = Number of grid points along X-axis.
 * ny    = Number of grid points along Y-axis.
 * nz    = Number of grid points along Z-axis.
 * step  = Spatial grid step (same along all axis).
 * bc    = Boundary condition: DFT_DRIVER_BC_X.
 * min_substeps = minimum substeps for function smoothing over the grid.
 * max_substeps = maximum substeps for function smoothing over the grid.
 *
 * Return value: pointer to the allocated OT DFT structure.
 *
 */

EXPORT dft_ot_functional *dft_ot3d_alloc(long model, long nx, long ny, long nz, double step, int bc, int min_substeps, int max_substeps) {

  double radius, inv_width;
  dft_ot_functional *otf;
  double (*grid_type)(const rgrid3d *, long, long, long);
  
  otf = (dft_ot_functional *) malloc(sizeof(dft_ot_functional));
  otf->model = model;
  if (!otf) {
    fprintf(stderr, "libdft: Error in dft_ot3d_alloc(): Could not allocate memory for dft_ot_functional.\n");
    return 0;
  }
  
  switch(bc) {
  case DFT_DRIVER_BC_NORMAL: 
    grid_type = RGRID3D_PERIODIC_BOUNDARY;
    break;
  case DFT_DRIVER_BC_X:
  case DFT_DRIVER_BC_Y:
  case DFT_DRIVER_BC_Z:
  case DFT_DRIVER_BC_NEUMANN:
    grid_type = RGRID3D_NEUMANN_BOUNDARY;
    break;
  default:
    fprintf(stderr, "libdft: Illegal boundary type.\n");
    exit(1);
  }


  dft_ot_temperature(otf, model);
  /* these grids are not needed for GP */
  if(!(model & DFT_GP) && !(model & DFT_ZERO)) {
    otf->lennard_jones = rgrid3d_alloc(nx, ny, nz, step, grid_type, 0);
    otf->spherical_avg = rgrid3d_alloc(nx, ny, nz, step, grid_type, 0);

    if(model & DFT_OT_KC) {
      otf->gaussian_tf = rgrid3d_alloc(nx, ny, nz, step, grid_type, 0);
      otf->gaussian_x_tf = rgrid3d_alloc(nx, ny, nz, step, grid_type, 0);
      otf->gaussian_y_tf = rgrid3d_alloc(nx, ny, nz, step, grid_type, 0);
      otf->gaussian_z_tf = rgrid3d_alloc(nx, ny, nz, step, grid_type, 0);
      if(!otf->gaussian_x_tf || !otf->gaussian_y_tf || !otf->gaussian_z_tf || !otf->gaussian_tf) {
	fprintf(stderr, "libdft: Error in dft_ot3d_alloc(): Could not allocate memory for gaussian.\n");
	return 0;
      }
    } else otf->gaussian_x_tf = otf->gaussian_y_tf = otf->gaussian_z_tf = otf->gaussian_tf = NULL;
  
    if(model & DFT_OT_BACKFLOW) {
      otf->backflow_pot = rgrid3d_alloc(nx, ny, nz, step, grid_type, 0);
      if(!otf->backflow_pot) {
	fprintf(stderr, "libdft: Error in dft_ot3d_alloc(): Could not allocate memory for backflow_pot.\n");
	return 0;
      }
    } else otf->backflow_pot = NULL;
  
    /* pre-calculate */
    if(model & DFT_DR) {
      fprintf(stderr, "libdft: LJ according to DR.\n");
      if( otf->lennard_jones->value_outside == RGRID3D_PERIODIC_BOUNDARY )
        rgrid3d_adaptive_map(otf->lennard_jones, dft_common_lennard_jones_smooth, &(otf->lj_params), min_substeps, max_substeps, 0.01 / GRID_AUTOK);
      else
        rgrid3d_adaptive_map_nonperiodic(otf->lennard_jones, dft_common_lennard_jones_smooth, &(otf->lj_params), min_substeps, max_substeps, 0.01 / GRID_AUTOK);
    } else {
      if( otf->lennard_jones->value_outside == RGRID3D_PERIODIC_BOUNDARY ){
	fprintf(stderr, "libdft: LJ according to SD (periodic).\n");
        rgrid3d_adaptive_map(otf->lennard_jones, dft_common_lennard_jones, &(otf->lj_params), min_substeps, max_substeps, 0.01 / GRID_AUTOK);
        /* Scaling of LJ so that the integral is exactly b */
        rgrid3d_multiply( otf->lennard_jones , otf->b / rgrid3d_integral(otf->lennard_jones) ) ;
        rgrid3d_fft(otf->lennard_jones);
      } else {
	fprintf(stderr, "libdft: LJ according to SD (nonperiodic).\n");
        rgrid3d_adaptive_map_nonperiodic(otf->lennard_jones, dft_common_lennard_jones, &(otf->lj_params), min_substeps, max_substeps, 0.01 / GRID_AUTOK);
        rgrid3d_fft(otf->lennard_jones);
        /* Scaling of LJ so that the integral is exactly b */
        rgrid3d_multiply(otf->lennard_jones , otf->b / ( step * step * step * otf->lennard_jones->value[0]));
      }
    }

    radius = otf->lj_params.h;
    if( otf->spherical_avg->value_outside == RGRID3D_PERIODIC_BOUNDARY ){
      fprintf(stderr, "libdft: Spherical average (periodic).\n");
      rgrid3d_adaptive_map(otf->spherical_avg, dft_common_spherical_avg, &radius, min_substeps, max_substeps, 0.01 / GRID_AUTOK);
      /* Scaling of sph. avg. so that the integral is exactly 1 */
      rgrid3d_multiply(otf->spherical_avg, 1.0 / rgrid3d_integral(otf->spherical_avg));
      rgrid3d_fft(otf->spherical_avg);
    } else {
      fprintf(stderr, "libdft: Spherical average (nonperiodic).\n");
      rgrid3d_adaptive_map_nonperiodic(otf->spherical_avg, dft_common_spherical_avg, &radius, min_substeps, max_substeps, 0.01 / GRID_AUTOK);
      rgrid3d_fft(otf->spherical_avg);
      /* Scaling of sph. avg. so that the integral is exactly 1 */
      rgrid3d_multiply(otf->spherical_avg, 1.0 / (step * step * step * otf->spherical_avg->value[0]));
    }
    
    if(model & DFT_OT_KC) {
      inv_width = 1.0 / otf->l_g;
      if(otf->gaussian_tf->value_outside == RGRID3D_PERIODIC_BOUNDARY) {
	fprintf(stderr, "libdft: Kinetic correlation (periodic).\n");	
        rgrid3d_adaptive_map(otf->gaussian_tf, dft_common_gaussian, &inv_width, min_substeps, max_substeps, 0.01 / GRID_AUTOK);
      } else {  
	fprintf(stderr, "libdft: Kinetic correlation (periodic).\n");	
        rgrid3d_adaptive_map_nonperiodic(otf->gaussian_tf, dft_common_gaussian, &inv_width, min_substeps, max_substeps, 0.01 / GRID_AUTOK);
      }
      rgrid3d_fd_gradient_x(otf->gaussian_tf, otf->gaussian_x_tf);
      rgrid3d_fd_gradient_y(otf->gaussian_tf, otf->gaussian_y_tf);
      rgrid3d_fd_gradient_z(otf->gaussian_tf, otf->gaussian_z_tf);
      rgrid3d_fft(otf->gaussian_x_tf);
      rgrid3d_fft(otf->gaussian_y_tf);
      rgrid3d_fft(otf->gaussian_z_tf);
      rgrid3d_fft(otf->gaussian_tf);
    }
    
    if(model & DFT_OT_BACKFLOW) {
      if(otf->backflow_pot->value_outside == RGRID3D_PERIODIC_BOUNDARY) {
	fprintf(stderr, "libdft: Backflow (periodic).\n");	
        rgrid3d_adaptive_map(otf->backflow_pot, dft_ot_backflow_pot, &(otf->bf_params), min_substeps, max_substeps, 0.01 / GRID_AUTOK);
      } else {
	fprintf(stderr, "libdft: Backflow (nonperiodic).\n");	
        rgrid3d_adaptive_map_nonperiodic(otf->backflow_pot, dft_ot_backflow_pot, &(otf->bf_params), min_substeps, max_substeps, 0.01 / GRID_AUTOK);
      }
      rgrid3d_fft(otf->backflow_pot);
    }
  }

  return otf;
}

/*
 * Free OT 3D functional structure.
 *
 * otf = functional structure to be freed (allocated previously
 *       by dft_ot3d_alloc()).
 *
 * No return value.
 *
 */

EXPORT void dft_ot3d_free(dft_ot_functional *otf) {

  if (otf) {
    if (otf->lennard_jones) rgrid3d_free(otf->lennard_jones);
    if (otf->spherical_avg) rgrid3d_free(otf->spherical_avg);
    if (otf->gaussian_tf) rgrid3d_free(otf->gaussian_tf);
    if (otf->gaussian_x_tf) rgrid3d_free(otf->gaussian_tf);
    if (otf->gaussian_y_tf) rgrid3d_free(otf->gaussian_tf);
    if (otf->gaussian_z_tf) rgrid3d_free(otf->gaussian_tf);
    if (otf->backflow_pot) rgrid3d_free(otf->backflow_pot);
    free(otf);
  }
}

/*
 * Calculate the non-linear potential grid (including backflow).
 *
 * otf        = OT 3D functional structure.
 * potential  = Potential grid where the result will be stored (output). NOTE: the potential will be added to this (may want to zero it first)
 * wf         = Wavefunction (input; used only for backflow).
 * density    = Liquid helium density grid (input).
 * workspace1 = Workspace grid (must be allocated by the user).
 * workspace2 = Workspace grid (must be allocated by the user).
 * workspace3 = Workspace grid (must be allocated by the user).
 * workspace4 = Workspace grid (must be allocated by the user).
 * workspace5 = Workspace grid (must be allocated by the user).
 * workspace6 = Workspace grid (must be allocated by the user).
 * workspace7 = Workspace grid (must be allocated by the user; accessed only with backflow).
 * workspace8 = Workspace grid (must be allocated by the user; accessed only with backflow).
 * workspace9 = Workspace grid (must be allocated by the user; accessed only with backflow).
 *
 * No return value.
 *
 */

EXPORT void dft_ot3d_potential(dft_ot_functional *otf, cgrid3d *potential, wf3d *wf, rgrid3d *density, rgrid3d *workspace1, rgrid3d *workspace2, rgrid3d *workspace3, rgrid3d *workspace4, rgrid3d *workspace5, rgrid3d *workspace6, rgrid3d *workspace7, rgrid3d *workspace8, rgrid3d *workspace9) {

  if(otf->model & DFT_ZERO) {
    fprintf(stderr, "libdft: Warning - zero potential used.\n");
    cgrid3d_zero(potential);
    return;
  }

  if(otf->model & DFT_GP) {
    rgrid3d_copy(workspace1, density);
    rgrid3d_multiply(workspace1, otf->mu0 / otf->rho0);
    grid3d_add_real_to_complex_re(potential, workspace1);
    return;
  }

  /* transform of rho (wrk1) (TODO: this could be passed along further below to avoid transforming density again) */
  rgrid3d_copy(workspace1, density);
  rgrid3d_fft(workspace1);

  /* Lennard-Jones */  
  /* int rho(r') Vlj(r-r') dr' */
  rgrid3d_fft_convolute(workspace2, workspace1, otf->lennard_jones);
  rgrid3d_inverse_fft(workspace2);
  grid3d_add_real_to_complex_re(potential, workspace2);

  /* Non-linear local correlation */
  /* note workspace1 = fft of \rho */
  dft_ot3d_add_local_correlation_potential(otf, potential, density, workspace1 /* rho_tf */, workspace2, workspace3);

  if(otf->model & DFT_OT_KC)
    /* Non-local correlation for kinetic energy */
    dft_ot3d_add_nonlocal_correlation_potential(otf, potential, density, workspace1, workspace2, workspace3, workspace4, workspace5, workspace6);

  if(otf->model & DFT_OT_HD)
    /* Barranco's penalty term */
    dft_ot3d_add_barranco(otf, potential, density, workspace1);

  if(otf->model & DFT_OT_BACKFLOW) {
    /* wf, veloc_x(1), veloc_y(2), veloc_z(3), wrk(4) */
    grid3d_wf_probability_flux(wf, workspace1, workspace2, workspace3);
    rgrid3d_division(workspace1, workspace1, density);  /* velocity = flux / rho */
    rgrid3d_division(workspace2, workspace2, density);
    rgrid3d_division(workspace3, workspace3, density);
    dft_ot3d_backflow_potential(otf, potential, density, workspace1 /* veloc_x */, workspace2 /* veloc_y */, workspace3 /* veloc_z */, workspace4, workspace5, workspace6, workspace7, workspace8, workspace9);
  }

  if(otf->model >= DFT_OT_T400MK && !(otf->model & DFT_DR))
    /* include the ideal gas contribution */
    dft_ot3d_add_ancilotto(otf, potential, density, workspace1);
}

/* local function */
static void dft_ot3d_add_local_correlation_potential(dft_ot_functional *otf, cgrid3d *potential, const rgrid3d *rho, const rgrid3d *rho_tf, rgrid3d *workspace1, rgrid3d *workspace2) {

  /* C2 */
  /* wrk1 = \bar{\rho} */
  rgrid3d_fft_convolute(workspace1, rho_tf, otf->spherical_avg);
  rgrid3d_inverse_fft(workspace1); 

  rgrid3d_power(workspace2, workspace1, otf->c2_exp);
  rgrid3d_multiply(workspace2, otf->c2 / 2.0);
  grid3d_add_real_to_complex_re(potential, workspace2);
  
  rgrid3d_power(workspace2, workspace1, otf->c2_exp - 1.0);
  rgrid3d_product(workspace2, workspace2, rho);
  rgrid3d_fft(workspace2);
  rgrid3d_fft_convolute(workspace2, workspace2, otf->spherical_avg);
  rgrid3d_inverse_fft(workspace2);
  rgrid3d_multiply(workspace2, otf->c2 * otf->c2_exp / 2.0);
  grid3d_add_real_to_complex_re(potential, workspace2);

  /* C3 */
  rgrid3d_power(workspace2, workspace1, otf->c3_exp);
  rgrid3d_multiply(workspace2, otf->c3 / 3.0);
  grid3d_add_real_to_complex_re(potential, workspace2);
  
  rgrid3d_power(workspace2, workspace1, otf->c3_exp - 1.0);
  rgrid3d_product(workspace2, workspace2, rho);
  rgrid3d_fft(workspace2);
  rgrid3d_fft_convolute(workspace2, workspace2, otf->spherical_avg);
  rgrid3d_inverse_fft(workspace2);
  rgrid3d_multiply(workspace2, otf->c3 * otf->c3_exp / 3.0);
  grid3d_add_real_to_complex_re(potential, workspace2);
}

/* local function */
static void dft_ot3d_add_nonlocal_correlation_potential(dft_ot_functional *otf, cgrid3d *potential, const rgrid3d *rho, rgrid3d *rho_tf, rgrid3d *workspace1, rgrid3d *workspace2, rgrid3d *workspace3, rgrid3d *workspace4, rgrid3d *workspace5) {

  /* rho^tilde(r) = int F(r-r') rho(r') dr' */
  rgrid3d_fft_convolute(workspace1, otf->gaussian_tf, rho_tf);
  rgrid3d_inverse_fft(workspace1);
  /* workspace1 = rho_st = 1 - 1/\tilde{\rho}/\rho_{0s} */
  rgrid3d_multiply(workspace1, -1.0 / otf->rho_0s);
  rgrid3d_add(workspace1, 1.0);

  dft_ot3d_add_nonlocal_correlation_potential_x(otf, potential, rho, rho_tf, workspace1 /* rho_st */, workspace2, workspace3, workspace4, workspace5);
  dft_ot3d_add_nonlocal_correlation_potential_y(otf, potential, rho, rho_tf, workspace1 /* rho_st */, workspace2, workspace3, workspace4, workspace5);
  dft_ot3d_add_nonlocal_correlation_potential_z(otf, potential, rho, rho_tf, workspace1 /* rho_st */, workspace2, workspace3, workspace4, workspace5);
}

/* local function */
static void dft_ot3d_add_nonlocal_correlation_potential_x(dft_ot_functional *otf, cgrid3d *potential, const rgrid3d *rho, rgrid3d *rho_tf, rgrid3d *rho_st, rgrid3d *workspace1, rgrid3d *workspace2, rgrid3d *workspace3, rgrid3d *workspace4) {

  double c;

  c = otf->alpha_s / (2.0 * otf->mass);

  /* workspace1 = (d/dx) rho */
  rgrid3d_fd_gradient_x(rho, workspace1);

  /*** 1st term ***/

  /* Construct workspace2 = FFT(G) = FFT((d/dx) \rho(r_1) * (1 - \tilde{\rho(r_1)} / \rho_{0s})) */
  rgrid3d_product(workspace2, workspace1, rho_st); // rho_st = (1 - \tilde{\rho(r_1)} / \rho_{0s})
  rgrid3d_fft(workspace2);

  /* 1st term: c convolute [((d/dx) F) . G] */
  rgrid3d_fft_convolute(workspace3, otf->gaussian_x_tf, workspace2);
  rgrid3d_inverse_fft(workspace3);
  rgrid3d_product(workspace3, workspace3, workspace4);
  rgrid3d_multiply(workspace3, c);
  grid3d_add_real_to_complex_re(potential, workspace3);  

  /* in use: workspace1 (grad rho), workspace2 (FFT(G)) */

  /*** 2nd term ***/
  
  /* Construct workspace3 = J = convolution(F G) */
  rgrid3d_fft_convolute(workspace3, otf->gaussian_tf, workspace2);
  rgrid3d_inverse_fft(workspace3);

  /* Construct workspace4 = FFT(H) = FFT((d/dx) \rho * J) */
  rgrid3d_copy(workspace4, workspace3);
  rgrid3d_product(workspace4, workspace1, workspace4);
  rgrid3d_fft(workspace4);

  /* 2nd term: c convolute(F H) */
  rgrid3d_fft_convolute(workspace4, otf->gaussian_tf, workspace4);
  rgrid3d_inverse_fft(workspace4);
  rgrid3d_multiply(workspace4, c);
  grid3d_add_real_to_complex_re(potential, workspace4);

  /* workspace1 (grad rho), workspace3 (J) */

  /*** 3rd term ***/
  
  /* -c J . convolute((d/dx)F \rho) */
  rgrid3d_fft_convolute(workspace2, otf->gaussian_x_tf, rho_tf);
  rgrid3d_inverse_fft(workspace2);
  rgrid3d_product(workspace2, workspace2, workspace3);
  rgrid3d_multiply(workspace2, -c);
  grid3d_add_real_to_complex_re(potential, workspace2);
}

/* local function */
static void dft_ot3d_add_nonlocal_correlation_potential_y(dft_ot_functional *otf, cgrid3d *potential, const rgrid3d *rho, rgrid3d *rho_tf, rgrid3d *rho_st, rgrid3d *workspace1, rgrid3d *workspace2, rgrid3d *workspace3, rgrid3d *workspace4) {

  double c;

  c = otf->alpha_s / (2.0 * otf->mass);

  /* workspace1 = (d/dy) rho */
  rgrid3d_fd_gradient_y(rho, workspace1);

  /*** 1st term ***/

  /* Construct workspace2 = FFT(G) = FFT((d/dy) \rho(r_1) * (1 - \tilde{\rho(r_1)} / \rho_{0s})) */
  rgrid3d_product(workspace2, workspace1, rho_st);
  rgrid3d_fft(workspace2);

  /* 1st term: c convolute [((d/dy) F) . G] */
  rgrid3d_fft_convolute(workspace3, otf->gaussian_y_tf, workspace2);
  rgrid3d_inverse_fft(workspace3);
  rgrid3d_product(workspace3, workspace3, workspace4);
  rgrid3d_multiply(workspace3, c);
  grid3d_add_real_to_complex_re(potential, workspace3);  

  /* in use: workspace1 (grad rho), workspace2 (FFT(G)) */

  /*** 2nd term ***/
  
  /* Construct workspace3 = J = convolution(F G) */
  rgrid3d_fft_convolute(workspace3, otf->gaussian_tf, workspace2);
  rgrid3d_inverse_fft(workspace3);

  /* Construct workspace4 = FFT(H) = FFT((d/dy) \rho * J) */
  rgrid3d_copy(workspace4, workspace3);
  rgrid3d_product(workspace4, workspace1, workspace4);
  rgrid3d_fft(workspace4);

  /* 2nd term: c convolute(F H) */
  rgrid3d_fft_convolute(workspace4, otf->gaussian_tf, workspace4);
  rgrid3d_inverse_fft(workspace4);
  rgrid3d_multiply(workspace4, c);
  grid3d_add_real_to_complex_re(potential, workspace4);

  /* workspace1 (grad rho), workspace3 (J) */

  /*** 3rd term ***/
  
  /* -c J . convolute((d/dy)F \rho) */

  rgrid3d_fft_convolute(workspace2, otf->gaussian_y_tf, rho_tf);
  rgrid3d_inverse_fft(workspace2);
  rgrid3d_product(workspace2, workspace2, workspace3);
  rgrid3d_multiply(workspace2, -c);
  grid3d_add_real_to_complex_re(potential, workspace2);
}

/* local function */
static void dft_ot3d_add_nonlocal_correlation_potential_z(dft_ot_functional *otf, cgrid3d *potential, const rgrid3d *rho, rgrid3d *rho_tf, rgrid3d *rho_st, rgrid3d *workspace1, rgrid3d *workspace2, rgrid3d *workspace3, rgrid3d *workspace4) {

  double c;

  c = otf->alpha_s / (2.0 * otf->mass);

  /* workspace1 = (d/dz) rho */
  rgrid3d_fd_gradient_z(rho, workspace1);

  /*** 1st term ***/

  /* Construct workspace2 = FFT(G) = FFT((d/dz) \rho(r_1) * (1 - \tilde{\rho(r_1)} / \rho_{0s})) */
  rgrid3d_product(workspace2, workspace1, rho_st);
  rgrid3d_fft(workspace2);

  /* 1st term: c convolute [((d/dz) F) . G] */
  rgrid3d_fft_convolute(workspace3, otf->gaussian_z_tf, workspace2);
  rgrid3d_inverse_fft(workspace3);
  rgrid3d_product(workspace3, workspace3, workspace4);
  rgrid3d_multiply(workspace3, c);
  grid3d_add_real_to_complex_re(potential, workspace3);  

  /* in use: workspace1 (grad rho), workspace2 (FFT(G)) */

  /*** 2nd term ***/
  
  /* Construct workspace3 = J = convolution(F G) */
  rgrid3d_fft_convolute(workspace3, otf->gaussian_tf, workspace2);
  rgrid3d_inverse_fft(workspace3);

  /* Construct workspace4 = FFT(H) = FFT((d/dz) \rho * J) */
  rgrid3d_copy(workspace4, workspace3);
  rgrid3d_product(workspace4, workspace1, workspace4);
  rgrid3d_fft(workspace4);

  /* 2nd term: c convolute(F H) */
  rgrid3d_fft_convolute(workspace4, otf->gaussian_tf, workspace4);
  rgrid3d_inverse_fft(workspace4);
  rgrid3d_multiply(workspace4, c);
  grid3d_add_real_to_complex_re(potential, workspace4);

  /* workspace1 (grad rho), workspace3 (J) */

  /*** 3rd term ***/
  
  /* -c J . convolute((d/dz)F \rho) */

  rgrid3d_fft_convolute(workspace2, otf->gaussian_z_tf, rho_tf);
  rgrid3d_inverse_fft(workspace2);
  rgrid3d_product(workspace2, workspace2, workspace3);
  rgrid3d_multiply(workspace2, -c);
  grid3d_add_real_to_complex_re(potential, workspace2);
}

/* local function */
static void dft_ot3d_add_ancilotto(dft_ot_functional *otf, cgrid3d *potential, const rgrid3d *rho, rgrid3d *workspace1) {

  dft_common_idealgas_params(otf->temp, otf->mass, otf->c4);
  rgrid3d_operate_one(workspace1, rho, dft_common_idealgas_op);
  grid3d_add_real_to_complex_re(potential, workspace1);
}

/* local function */
static void dft_ot3d_add_barranco(dft_ot_functional *otf, cgrid3d *potential, const rgrid3d *rho, rgrid3d *workspace1) {

  XXX_beta = otf->beta;
  XXX_rhom = otf->rhom;
  XXX_C = otf->C;
  rgrid3d_operate_one(workspace1, rho, dft_ot3d_barranco_op);
  grid3d_add_real_to_complex_re(potential, workspace1);
}


/*
 * Evaluate the potential part to the energy density.
 * Note: the single particle kinetic portion is NOT included.
 *       (use grid3d_wf_kinetic_energy() to calculate this separately)
 *
 * otf            = OT 3D functional structure.
 * energy_density = energy density grid (output).
 * density        = liquid density grid (input).
 * workspace1 = Workspace grid (must be allocated by the user).
 * workspace2 = Workspace grid (must be allocated by the user).
 * workspace3 = Workspace grid (must be allocated by the user).
 * workspace4 = Workspace grid (must be allocated by the user).
 * workspace5 = Workspace grid (must be allocated by the user).
 * workspace6 = Workspace grid (must be allocated by the user).
 * workspace7 = Workspace grid (must be allocated by the user).
 * workspace8 = Workspace grid (must be allocated by the user).
 *
 * No return value.
 *
 */

EXPORT void dft_ot3d_energy_density(dft_ot_functional *otf, rgrid3d *energy_density, const rgrid3d *density, rgrid3d *workspace1, rgrid3d *workspace2, rgrid3d *workspace3, rgrid3d *workspace4, rgrid3d *workspace5, rgrid3d *workspace6, rgrid3d *workspace7, rgrid3d *workspace8) {
  
  rgrid3d_zero(energy_density);

  if(otf->model & DFT_ZERO) {
    fprintf(stderr, "libdft: Warning - zero potential used.\n");
    return;
  }

  if(otf->model & DFT_GP) {
    rgrid3d_copy(energy_density, density);
    rgrid3d_product(energy_density, energy_density, density);
    rgrid3d_multiply(energy_density, 0.5 * otf->mu0 / otf->rho0);
    return;
  }

  /* transform rho (wrk1) */
  rgrid3d_copy(workspace1, density);
  rgrid3d_fft(workspace1);

  /* Lennard-Jones */  
  /* (1/2) rho(r) int V_lj(|r-r'|) rho(r') dr' */
  rgrid3d_fft_convolute(workspace2, workspace1, otf->lennard_jones);
  rgrid3d_inverse_fft(workspace2);
  rgrid3d_add_scaled_product(energy_density, 0.5, density, workspace2);

  /* local correlation */
  /* wrk1 = \bar{\rho} */
  rgrid3d_fft_convolute(workspace1, workspace1, otf->spherical_avg);
  rgrid3d_inverse_fft(workspace1);

  /* C2 */
  rgrid3d_power(workspace2, workspace1, otf->c2_exp);
  rgrid3d_product(workspace2, workspace2, density);
  rgrid3d_add_scaled(energy_density, otf->c2 / 2.0, workspace2);

  /* C3 */
  rgrid3d_power(workspace2, workspace1, otf->c3_exp);
  rgrid3d_product(workspace2, workspace2, density);
  rgrid3d_add_scaled(energy_density, otf->c3 / 3.0, workspace2);

  /* Barranco's contribution (high density) */
  if(otf->model & DFT_OT_HD) {
    XXX_beta = otf->beta;
    XXX_rhom = otf->rhom;
    XXX_C = otf->C;
    rgrid3d_operate_one(workspace1, density, dft_ot3d_barranco_energy_op);
    rgrid3d_sum(energy_density, energy_density, workspace1);
  }

  /* Ideal gas contribution (thermal) */
  if(otf->model >= DFT_OT_T400MK) {
    dft_common_idealgas_params(otf->temp, otf->mass, otf->c4);
    rgrid3d_operate_one(workspace1, density, dft_common_idealgas_energy_op);
    rgrid3d_sum(energy_density, energy_density, workspace1);
  }
  
  /* begin kinetic energy correlation energy density */
  if(otf->model & DFT_OT_KC) { /* new code */
    /* 1. convolute density with F to get \tilde{\rho} (wrk1) */
    rgrid3d_copy(workspace2, density);
    rgrid3d_fft(workspace2);
    rgrid3d_fft_convolute(workspace1, workspace2, otf->gaussian_tf);   /* otf->gaussian_tf is already in Fourier space */    
    rgrid3d_inverse_fft(workspace1);

    /* 2. modify wrk1 from \tilde{\rho} to (1 - \tilde{\rho}/\rho_{0s} */
    rgrid3d_multiply_and_add(workspace1, -1.0/otf->rho_0s, 1.0);

    /* 3. gradient \rho to wrk3 (x), wrk4 (y), wrk5 (z) */
    rgrid3d_fd_gradient_x(density, workspace3);
    rgrid3d_fd_gradient_y(density, workspace4);
    rgrid3d_fd_gradient_z(density, workspace5);
    
    /* 4. X component: wrk6 = wrk3 * wrk1 (wrk1 = (d/dx)\rho_x * (1 - \tilde{\rho}/\rho_{0s}) */
    /*    Y component: wrk7 = wrk4 * wrk1 (wrk1 = (d/dy)\rho_y * (1 - \tilde{\rho}/\rho_{0s}) */
    /*    Z component: wrk8 = wrk5 * wrk1 (wrk1 = (d/dz)\rho_z * (1 - \tilde{\rho}/\rho_{0s}) */
    rgrid3d_product(workspace6, workspace3, workspace1);
    rgrid3d_product(workspace7, workspace4, workspace1);
    rgrid3d_product(workspace8, workspace5, workspace1);

    /* 5. convolute (X): wrk6 = convolution(otf->gaussian * wrk6). */
    /*    convolute (Y): wrk7 = convolution(otf->gaussian * wrk7). */
    /*    convolute (Z): wrk8 = convolution(otf->gaussian * wrk8). */
    rgrid3d_fft(workspace6);
    rgrid3d_fft(workspace7);
    rgrid3d_fft(workspace8);
    rgrid3d_fft_convolute(workspace6, workspace6, otf->gaussian_tf);
    rgrid3d_fft_convolute(workspace7, workspace7, otf->gaussian_tf);
    rgrid3d_fft_convolute(workspace8, workspace8, otf->gaussian_tf);
    rgrid3d_inverse_fft(workspace6);
    rgrid3d_inverse_fft(workspace7);
    rgrid3d_inverse_fft(workspace8);

    /* 6. X: wrk6 = wrk6 * wrk3 * wrk1 */
    /*    Y: wrk7 = wrk7 * wrk4 * wrk1 */
    /*    Z: wrk8 = wrk8 * wrk5 * wrk1 */
    rgrid3d_product(workspace6, workspace6, workspace3);
    rgrid3d_product(workspace6, workspace6, workspace1);
    rgrid3d_product(workspace7, workspace7, workspace4);
    rgrid3d_product(workspace7, workspace7, workspace1);
    rgrid3d_product(workspace8, workspace8, workspace5);
    rgrid3d_product(workspace8, workspace8, workspace1);
    
    /* 7. add wrk6 + wrk7 + wrk8 (components from the dot product) */
    /* 8. multiply by -\hbar^2\alpha_s/(4M_{He}) */    
    rgrid3d_add_scaled(energy_density, -otf->alpha_s / (4.0 * otf->mass), workspace6);
    rgrid3d_add_scaled(energy_density, -otf->alpha_s / (4.0 * otf->mass), workspace7);
    rgrid3d_add_scaled(energy_density, -otf->alpha_s / (4.0 * otf->mass), workspace8);
  }
}

/*
 * Add the backflow non-linear potential.
 *
 * otf            = OT 3D functional structure.
 * potential      = potential grid (output). Not cleared.
 * density        = liquid density grid (input).
 * veloc_x        = veloc grid (with respect to X; input - overwritten on exit).
 * veloc_y        = veloc grid (with respect to Y; input - overwritten on exit).
 * veloc_z        = veloc grid (with respect to Z; input - overwritten on exit).
 * workspace1     = Workspace grid (must be allocated by the user).
 * workspace2     = Workspace grid (must be allocated by the user).
 * workspace3     = Workspace grid (must be allocated by the user).
 * workspace4     = Workspace grid (must be allocated by the user).
 * workspace5     = Workspace grid (must be allocated by the user).
 * workspace6     = Workspace grid (must be allocated by the user).
 *
 * No return value.
 *
 */

EXPORT void dft_ot3d_backflow_potential(dft_ot_functional *otf, cgrid3d *potential, const rgrid3d *density, rgrid3d *veloc_x, rgrid3d *veloc_y, rgrid3d *veloc_z, rgrid3d *workspace1, rgrid3d *workspace2, rgrid3d *workspace3, rgrid3d *workspace4, rgrid3d *workspace5, rgrid3d *workspace6) {
  
  /* Calculate A (workspace1) [scalar] */
  rgrid3d_copy(workspace1, density);
  rgrid3d_fft(workspace1);
  rgrid3d_fft_convolute(workspace1, workspace1, otf->backflow_pot);
  rgrid3d_inverse_fft(workspace1);

  /* Calculate C (workspace2) [scalar] */
  rgrid3d_product(workspace2, veloc_x, veloc_x);
  rgrid3d_add_scaled_product(workspace2, 1.0, veloc_y, veloc_y);
  rgrid3d_add_scaled_product(workspace2, 1.0, veloc_z, veloc_z);
  rgrid3d_product(workspace2, workspace2, density);
  rgrid3d_fft(workspace2);
  rgrid3d_fft_convolute(workspace2, workspace2, otf->backflow_pot);
  rgrid3d_inverse_fft(workspace2);

  /* Calculate B (workspace3 (x), workspace4 (y), workspace5 (z)) [vector] */
  rgrid3d_product(workspace3, density, veloc_x);
  rgrid3d_fft(workspace3);
  rgrid3d_fft_convolute(workspace3, workspace3, otf->backflow_pot);
  rgrid3d_inverse_fft(workspace3);

  rgrid3d_product(workspace4, density, veloc_y);
  rgrid3d_fft(workspace4);
  rgrid3d_fft_convolute(workspace4, workspace4, otf->backflow_pot);
  rgrid3d_inverse_fft(workspace4);

  rgrid3d_product(workspace5, density, veloc_z);
  rgrid3d_fft(workspace5);
  rgrid3d_fft_convolute(workspace5, workspace5, otf->backflow_pot);
  rgrid3d_inverse_fft(workspace5);

  /* 1. Calculate the real part of the potential */

  /* -(m/2) (v(r) . (v(r)A(r) - 2B(r)) + C(r))  = -(m/2) [ A(v_x^2 + v_y^2 + v_z^2) - 2v_x B_x - 2v_y B_y - 2v_z B_z + C] */
  rgrid3d_product(workspace6, veloc_x, veloc_x);
  rgrid3d_add_scaled_product(workspace6, 1.0, veloc_y, veloc_y);
  rgrid3d_add_scaled_product(workspace6, 1.0, veloc_z, veloc_z);
  rgrid3d_product(workspace6, workspace6, workspace1);
  rgrid3d_add_scaled_product(workspace6, -2.0, veloc_x, workspace3);
  rgrid3d_add_scaled_product(workspace6, -2.0, veloc_y, workspace4);
  rgrid3d_add_scaled_product(workspace6, -2.0, veloc_z, workspace5);
  rgrid3d_sum(workspace6, workspace6, workspace2);
  rgrid3d_multiply(workspace6, -0.5 * otf->mass);
  grid3d_add_real_to_complex_re(potential, workspace6);

  /* workspace2 (C), workspace6 not used after this point */

  /* 2. Calculate the imaginary part of the potential */

  rgrid3d_zero(workspace6);
  
  /* v_x -> v_xA - B_x, v_y -> v_yA - B_y, v_z -> v_zA - B_z (velocities are overwritten here) */
  rgrid3d_product(veloc_x, veloc_x, workspace1);
  rgrid3d_add_scaled(veloc_x, -1.0, workspace3);
  rgrid3d_product(veloc_y, veloc_y, workspace1);
  rgrid3d_add_scaled(veloc_y, -1.0, workspace4);
  rgrid3d_product(veloc_z, veloc_z, workspace1);
  rgrid3d_add_scaled(veloc_z, -1.0, workspace5);

  /* 1.1 (1/2) (drho/dx)/rho * (v_xA - B_x) */
  rgrid3d_fd_gradient_x(density, workspace2);
  rgrid3d_division(workspace2, workspace2, density);
  rgrid3d_add_scaled_product(workspace6, 0.5, workspace2, veloc_x);

  /* 1.2 (1/2) (drho/dy)/rho * (v_yA - B_y) */
  rgrid3d_fd_gradient_y(density, workspace2);
  rgrid3d_division(workspace2, workspace2, density);
  rgrid3d_add_scaled_product(workspace6, 0.5, workspace2, veloc_y);

  /* 1.3 (1/2) (drho/dz)/rho * (v_zA - B_z) */
  rgrid3d_fd_gradient_z(density, workspace2);
  rgrid3d_division(workspace2, workspace2, density);
  rgrid3d_add_scaled_product(workspace6, 0.5, workspace2, veloc_z);

  /* 2.1 (1/2) (d/dx) (v_xA - B_x) */
  rgrid3d_fd_gradient_x(veloc_x, workspace2);
  rgrid3d_add_scaled(workspace6, 0.5, workspace2);

  /* 2.2 (1/2) (d/dy) (v_yA - B_y) */
  rgrid3d_fd_gradient_y(veloc_y, workspace2);
  rgrid3d_add_scaled(workspace6, 0.5, workspace2);

  /* 2.3 (1/2) (d/dz) (v_zA - B_z) */
  rgrid3d_fd_gradient_z(veloc_z, workspace2);
  rgrid3d_add_scaled(workspace6, 0.5, workspace2);

  grid3d_add_real_to_complex_im(potential, workspace6);
}

/*
 * Evaluate the backflow energy density.
 *
 * NOT IMPLEMENTED YET.
 *
 */

EXPORT void dft_ot3d_backflow_energy_density(dft_ot_functional *otf, rgrid3d *energy_density, const wf3d *wavefunc, rgrid3d *workspace1, rgrid3d *workspace2, rgrid3d *workspace3, rgrid3d *workspace4) {

  fprintf(stderr, "Energy density for Backflow not implemented.\n");
  abort();
}


EXPORT inline void dft_ot_temperature(dft_ot_functional *otf, long model) {

  fprintf(stderr, "libdft: Model = %ld\n", model);

  if(otf->model & DFT_OT_HD) { /* high density penalty */
    otf->beta = (40.0 / (GRID_AUTOANG * GRID_AUTOANG * GRID_AUTOANG));
    otf->rhom = (0.37 * GRID_AUTOANG * GRID_AUTOANG * GRID_AUTOANG);
    otf->C = 0.1; /* a.u. */
  }

  if(model & DFT_DR) { /* Dupont-Roc */
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
    otf->c2_exp = 2.0;
    otf->c3 = 1.858496E6;
    otf->c3_exp = 3.0;
    otf->c4 = 0.0;
    otf->temp = 0.0;
    otf->rho0 = 0.0218360;
    otf->lj_params.h = 2.1903;
  }

  if(model & DFT_OT_T0MK) { /* 1 */
    otf->b = -719.2435;
    otf->c2 = -24258.88;
    otf->c2_exp = 2.0;
    otf->c3 = 1865257.0;
    otf->c3_exp = 3.0;
    otf->c4 = 0.0;
    otf->temp = 0.0;
    otf->rho0 = 0.0218354;
    otf->lj_params.h = 2.19035;
  }

  if(model & DFT_OT_T400MK) { /* 2 */
    otf->b = -714.2174;
    otf->c2 = -24566.29;
    otf->c2_exp = 2.0;
    otf->c3 = 1873203.0;
    otf->c3_exp = 3.0;
    otf->c4 = 0.98004;
    otf->temp = 0.4;
    otf->rho0 = 0.0218351;
    otf->lj_params.h = 2.18982;
  }

  if(model & DFT_OT_T600MK) { /* 3 */
    otf->b = -705.1319;
    otf->c2 = -25124.17;
    otf->c2_exp = 2.0;
    otf->c3 = 1887707.0;
    otf->c3_exp = 3.0;
    otf->c4 = 0.99915;
    otf->temp = 0.6;
    otf->rho0 = 0.0218346;
    otf->lj_params.h = 2.18887;
  }

  if(model & DFT_OT_T800MK) { /* 4 */
    otf->b = -690.4745;
    otf->c2 = -26027.12;
    otf->c2_exp = 2.0;
    otf->c3 = 1911283.0;
    otf->c3_exp = 3.0;
    otf->c4 = 0.99548;
    otf->temp = 0.8;
    otf->rho0 = 0.0218331;
    otf->lj_params.h = 2.18735;
  }

  if(model & DFT_OT_T1200MK) { /* 5 */
    otf->b = -646.5135;
    otf->c2 = -28582.81;
    otf->c2_exp = 2.0;
    otf->c3 = 1973737.0;
    otf->c3_exp = 3.0;
    otf->c4 = 0.99666;
    otf->temp = 1.2;
    otf->rho0 = 0.0218298;
    otf->lj_params.h = 2.18287;
  }

  if(model & DFT_OT_T1400MK) { /* 6 */
    otf->b = -625.8123;
    otf->c2 = -29434.03;
    otf->c2_exp = 2.0;
    otf->c3 = 1984068.0;
    otf->c3_exp = 3.0;
    otf->c4 = 0.99829;
    otf->temp = 1.4;
    otf->rho0 = 0.0218332;
    otf->lj_params.h = 2.18080;
  }

  if(model & DFT_OT_T1600MK) { /* 7 */
    otf->b = -605.9788;
    otf->c2 = -30025.96;
    otf->c2_exp = 2.0;
    otf->c3 = 1980898.0;
    otf->c3_exp = 3.0;
    otf->c4 = 1.00087;
    otf->temp = 1.6;
    otf->rho0 = 0.0218453;
    otf->lj_params.h = 2.17885;
  }

  if(model & DFT_OT_T1800MK) { /* 8 */
    otf->b = -593.8289;
    otf->c2 = -29807.56;
    otf->c2_exp = 2.0;
    otf->c3 = 1945685.0;
    otf->c3_exp = 3.0;
    otf->c4 = 1.00443;
    otf->temp = 1.8;
    otf->rho0 = 0.0218703;
    otf->lj_params.h = 2.17766;
  }

  if(model & DFT_OT_T2000MK) { /* 9 */
    otf->b = -600.8313;
    otf->c2 = -27850.96;
    otf->c2_exp = 2.0;
    otf->c3 = 1847407.0;
    otf->c3_exp = 3.0;
    otf->c4 = 1.00919;
    otf->temp = 2.0;
    otf->rho0 = 0.0219153;
    otf->lj_params.h = 2.17834;
  }

  if(model & DFT_OT_T2100MK) { /* 10 */
    otf->b = -620.9129;
    otf->c2 = -25418.15;
    otf->c2_exp = 2.0;
    otf->c3 = 1747494.0;
    otf->c3_exp = 3.0;
    otf->c4 = 1.01156;
    otf->temp = 2.1;
    otf->rho0 = 0.0219500;
    otf->lj_params.h = 2.18032;
  }

  if(model & DFT_OT_T2200MK) { /* 11 */
    otf->b = -619.2016;
    otf->c2 = -25096.68;
    otf->c2_exp = 2.0;
    otf->c3 = 1720802.0;
    otf->c3_exp = 3.0;
    otf->c4 = 1.01436;
    otf->temp = 2.2;
    otf->rho0 = 0.0219859;
    otf->lj_params.h = 2.18015;
  }

  if(model & DFT_OT_T2400MK) { /* 12 */
    otf->b = -609.0757;
    otf->c2 = -26009.98;
    otf->c2_exp = 2.0;
    otf->c3 = 1747943.0;
    otf->c3_exp = 3.0;
    otf->c4 = 1.02130;
    otf->temp = 2.4;
    otf->rho0 = 0.0218748;
    otf->lj_params.h = 2.17915;
  }

  if(model & DFT_OT_T2600MK) { /* 13 */
    otf->b = -634.0664;
    otf->c2 = -23790.66;
    otf->c2_exp = 2.0;
    otf->c3 = 1670707.0;
    otf->c3_exp = 3.0;
    otf->c4 = 1.02770;
    otf->temp = 2.6;
    otf->rho0 = 0.0217135;
    otf->lj_params.h = 2.18162;
  }

  if(model & DFT_OT_T2800MK) { /* 14 */
    otf->b = -663.9942;
    otf->c2 = -21046.37;
    otf->c2_exp = 2.0;
    otf->c3 = 1574611.0;
    otf->c3_exp = 3.0;
    otf->c4 = 1.03429;
    otf->temp = 2.8;
    otf->rho0 = 0.0215090;
    otf->lj_params.h = 2.18463;
  }

  if(model & DFT_OT_T3000MK) { /* 15 */
    otf->b = -673.6543;
    otf->c2 = -20022.76;
    otf->c2_exp = 2.0;
    otf->c3 = 1535887.0;
    otf->c3_exp = 3.0;
    otf->c4 = 1.04271;
    otf->temp = 3.0;
    otf->rho0 = 0.0212593;
    otf->lj_params.h = 2.18562;
  }

  if((model & DFT_GP) || (model & DFT_ZERO)) {
    otf->temp = 0.0;
    otf->rho0 = 0.0212593;
    otf->mu0 = 7.0 / GRID_AUTOK;
    /* most of the parameters are unused */
  }

  fprintf(stderr,"libdft: Temperature = %le K.\n", otf->temp); 

  otf->b /= GRID_AUTOK * (GRID_AUTOANG * GRID_AUTOANG * GRID_AUTOANG);
  otf->c2 /= GRID_AUTOK * pow(GRID_AUTOANG, 3.0 * otf->c2_exp);
  otf->c3 /= GRID_AUTOK * pow(GRID_AUTOANG, 3.0 * otf->c3_exp);
  otf->rho0 *= GRID_AUTOANG * GRID_AUTOANG * GRID_AUTOANG;
  otf->lj_params.h /= GRID_AUTOANG;
  otf->lj_params.sigma   = 2.556  / GRID_AUTOANG;
  otf->lj_params.epsilon = 10.22  / GRID_AUTOK;
  otf->rho_0s = 0.04 * (GRID_AUTOANG * GRID_AUTOANG * GRID_AUTOANG);
  otf->alpha_s = 54.31 / (GRID_AUTOANG * GRID_AUTOANG * GRID_AUTOANG);
  otf->l_g = 1.0 / GRID_AUTOANG;
  otf->mass = 4.0026 / GRID_AUTOAMU;
  otf->rho_eps = 1E-7 * 0.0218360 * (GRID_AUTOANG * GRID_AUTOANG * GRID_AUTOANG);
  otf->bf_params.g11 = -19.7544;
  otf->bf_params.g12 = 12.5616 * (GRID_AUTOANG*GRID_AUTOANG);
  otf->bf_params.a1  = 1.023 * (GRID_AUTOANG*GRID_AUTOANG);
  otf->bf_params.g21 = -0.2395;
  otf->bf_params.g22 = 0.0312 * (GRID_AUTOANG*GRID_AUTOANG);
  otf->bf_params.a2  = 0.14912 * (GRID_AUTOANG*GRID_AUTOANG);

  fprintf(stderr, "libdft: C2 = %le K Angs^%le\n", 
	  otf->c2 * GRID_AUTOK * pow(GRID_AUTOANG, 3.0 * otf->c2_exp),
	  3.0 * otf->c2_exp);
  
  fprintf(stderr, "libdft: C3 = %le K Angs^%le\n", 
	  otf->c3 * GRID_AUTOK * pow(GRID_AUTOANG, 3.0 * otf->c3_exp),
	  3.0 * otf->c3_exp);
  
  otf->model = model;
}
