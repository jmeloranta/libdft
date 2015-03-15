/*
 * Orsay-Trento functional for superfluid helium in 2D (cylindrical coords).
 *
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <grid/grid.h>
#include <grid/au.h>
#include "dft.h"
#include "ot.h"

long dft_ot2d_hankel_pad = 0;

static void dft_ot2d_add_local_correlation_potential(dft_ot_functional_2d *otf, cgrid2d *potential, const rgrid2d *rho, const rgrid2d *rho_tf, rgrid2d *workspace1, rgrid2d *workspace2);
static void dft_ot2d_add_nonlocal_correlation_potential(dft_ot_functional_2d *otf, cgrid2d *potential, const rgrid2d *rho, rgrid2d *rho_tf, rgrid2d *workspace1, rgrid2d *workspace2, rgrid2d *workspace3, rgrid2d *workspace4, rgrid2d *workspace5);
static void dft_ot2d_add_nonlocal_correlation_potential_z(dft_ot_functional_2d *otf, cgrid2d *potential, const rgrid2d *rho, rgrid2d *rho_tf, rgrid2d *rho_st, rgrid2d *workspace1, rgrid2d *workspace2, rgrid2d *workspace3, rgrid2d *workspace4);
static void dft_ot2d_add_nonlocal_correlation_potential_r(dft_ot_functional_2d *otf, cgrid2d *potential, const rgrid2d *rho, rgrid2d *rho_tf, rgrid2d *rho_st, rgrid2d *workspace1, rgrid2d *workspace2, rgrid2d *workspace3, rgrid2d *workspace4);
static void dft_ot2d_add_barranco(dft_ot_functional_2d *otf, cgrid2d *potential, const rgrid2d *rho, rgrid2d *workspace1);
static void dft_ot2d_add_ancilotto(dft_ot_functional_2d *otf, cgrid2d *potential, const rgrid2d *rho, rgrid2d *workspace1);

/* Tunable parameters */
#define EPS 1E-8
#define CUTOFF (50.0 / GRID_AUTOK)

static double XXX_beta, XXX_rhom, XXX_C;

/* local function */
static inline double dft_ot2d_barranco_op(double rhop) {

  double stanh = tanh(XXX_beta * (rhop - XXX_rhom));

  return (XXX_C * ((1.0 + stanh) + XXX_beta * rhop * (1.0 - stanh * stanh)));
}

static inline double dft_ot2d_barranco_energy_op(double rhop) {

  return (XXX_C * (1.0 + tanh(XXX_beta * (rhop - XXX_rhom))) * rhop);
}

static inline double dft_ot_backflow_pot_2d(void *arg, double z, double r) {

  double g11 = ((dft_ot_bf *) arg)->g11;
  double g12 = ((dft_ot_bf *) arg)->g12;
  double g21 = ((dft_ot_bf *) arg)->g21;
  double g22 = ((dft_ot_bf *) arg)->g22;
  double a1 = ((dft_ot_bf *) arg)->a1;
  double a2 = ((dft_ot_bf *) arg)->a2;
  double r2 = z * z + r * r;
  
  return (g11 + g12 * r2) * exp(-a1 * r2) + (g21 + g22 * r2) * exp(-a2 * r2);
}

/*
 * Allocate 2D OT functional. This must be called first.
 * 
 * model = which OT functional variant to use:
 *         DFT_OT_KC       Include the non-local kinetic energy correlation.
 *         DFT_OT_HD       Include Barranco's high density correction (original h for sp. ave).
 *         DFT_OT_HD2       Include Barranco's high density correction (new h for sp. ave).
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
 *         DFT_ZERO        Zero potential.
 *
 *           If multiple options are needed, use and (&).
 * nz    = Number of grid points along Z-axis.
 * nr    = Number of grid points along R-axis.
 * step  = Spatial grid step (same along all axis).
 * min_substeps = minimum substeps for function smoothing over the grid.
 * max_substeps = maximum substeps for function smoothing over the grid.
 *
 * Return value: pointer to the allocated OT DFT structure.
 *
 */

EXPORT dft_ot_functional_2d *dft_ot2d_alloc(long model, long nz, long nr, double step, int min_substeps, int max_substeps) {

  double radius, inv_width;
  dft_ot_functional_2d *otf;
  
  dft_ot2d_hankel_pad = nr / 3; /* 1/3 of last points the be cleaned up */
  otf = (dft_ot_functional_2d *) malloc(sizeof(dft_ot_functional_2d));
  otf->model = model;
  if (!otf) {
    fprintf(stderr, "libdft: Error in dft_ot2d_alloc(): Could not allocate memory for dft_ot_functional_2d.\n");
    return 0;
  }
  
  dft_ot_temperature_2d(otf, model);
  /* these grids are not needed for GP */
  if(!(model & DFT_GP) && !(model & DFT_ZERO)) {
    otf->lennard_jones = rgrid2d_alloc(nz, nr, step, RGRID2D_NEUMANN_BOUNDARY, 0);
    otf->spherical_avg = rgrid2d_alloc(nz, nr, step, RGRID2D_NEUMANN_BOUNDARY, 0);

    if(model & DFT_OT_KC) {
      otf->gaussian_tf = rgrid2d_alloc(nz, nr, step, RGRID2D_NEUMANN_BOUNDARY, 0);
      otf->gaussian_z_tf = rgrid2d_alloc(nz, nr, step, RGRID2D_NEUMANN_BOUNDARY, 0);
      otf->gaussian_r_tf = rgrid2d_alloc(nz, nr, step, RGRID2D_NEUMANN_BOUNDARY, 0);
      if(!otf->gaussian_tf || !otf->gaussian_z_tf || !otf->gaussian_r_tf) {
	fprintf(stderr, "libdft: Error in dft_ot2d_alloc(): Could not allocate memory for gaussian.\n");
	return 0;
      }
    } else otf->gaussian_tf = otf->gaussian_z_tf = otf->gaussian_r_tf = NULL;
  
    if(model & DFT_OT_BACKFLOW) {
      otf->backflow_pot = rgrid2d_alloc(nz, nr, step, RGRID2D_NEUMANN_BOUNDARY, 0);
      if(!otf->backflow_pot) {
	fprintf(stderr, "libdft: Error in dft_ot2d_alloc(): Could not allocate memory for backflow_pot.\n");
	return 0;
      }
    } else otf->backflow_pot = NULL;
  
    /* pre-calculate */
    if(model & DFT_DR) {
      fprintf(stderr, "libdft: LJ according to DR.\n");
      rgrid2d_adaptive_map_cyl(otf->lennard_jones, dft_common_lennard_jones_smooth_2d, &(otf->lj_params), min_substeps, max_substeps, 0.01 / GRID_AUTOK);
    } else {
      fprintf(stderr, "libdft: LJ according to SD.\n");
      rgrid2d_adaptive_map_cyl(otf->lennard_jones, dft_common_lennard_jones_2d, &(otf->lj_params), min_substeps, max_substeps, 0.01 / GRID_AUTOK);
      /* Scaling of LJ so that the integral is exactly b */
      rgrid2d_multiply(otf->lennard_jones, otf->b / rgrid2d_integral_cyl(otf->lennard_jones));
    }
    rgrid2d_fft_cylindrical(otf->lennard_jones);

    if(otf->model & DFT_OT_HD2) {
      radius = otf->lj_params.h * 1.065;
      fprintf(stderr, "libdft: Spherical average (new).\n");
    } else {
      radius = otf->lj_params.h;
      fprintf(stderr, "libdft: Spherical average (original).\n");
    }
    rgrid2d_adaptive_map_cyl(otf->spherical_avg, dft_common_spherical_avg_2d, &radius, min_substeps, max_substeps, 0.01 / GRID_AUTOK);
    /* Scaling of sph. avg. so that the integral is exactly 1 */
    rgrid2d_multiply(otf->spherical_avg, 1.0 / rgrid2d_integral_cyl(otf->spherical_avg));
    rgrid2d_fft_cylindrical(otf->spherical_avg);
    
    if(model & DFT_OT_KC) {
      inv_width = 1.0 / otf->l_g;
      fprintf(stderr, "libdft: Kinetic correlation.\n");	
      rgrid2d_adaptive_map_cyl(otf->gaussian_tf, dft_common_gaussian_2d, &inv_width, min_substeps, max_substeps, 0.01 / GRID_AUTOK);
      rgrid2d_fd_gradient_cyl_z(otf->gaussian_tf, otf->gaussian_z_tf);
      rgrid2d_fd_gradient_cyl_r(otf->gaussian_tf, otf->gaussian_r_tf);
      rgrid2d_fft_cylindrical(otf->gaussian_tf);
      rgrid2d_fft_cylindrical(otf->gaussian_z_tf);
      rgrid2d_fft_cylindrical(otf->gaussian_r_tf);
    }
  
    if(model & DFT_OT_BACKFLOW) {
      fprintf(stderr, "libdft: Backflow.\n");	
      rgrid2d_adaptive_map_cyl(otf->backflow_pot, dft_ot_backflow_pot_2d, &(otf->bf_params), min_substeps, max_substeps, 0.01 / GRID_AUTOK);
      rgrid2d_fft_cylindrical(otf->backflow_pot);
    }
  }

  return otf;
}

/*
 * Free OT 2D functional structure.
 *
 * otf = functional structure to be freed (allocated previously
 *       by dft_ot2d_alloc()).
 *
 * No return value.
 *
 */

EXPORT void dft_ot2d_free(dft_ot_functional_2d *otf) {

  if (otf) {
    if(!(otf->model & DFT_GP) && !(otf->model & DFT_ZERO)) {
      if (otf->lennard_jones) rgrid2d_free(otf->lennard_jones);
      if (otf->spherical_avg) rgrid2d_free(otf->spherical_avg);
      if (otf->gaussian_tf) rgrid2d_free(otf->gaussian_tf);
      if (otf->gaussian_z_tf) rgrid2d_free(otf->gaussian_z_tf);
      if (otf->gaussian_r_tf) rgrid2d_free(otf->gaussian_r_tf);
      if (otf->backflow_pot) rgrid2d_free(otf->backflow_pot);
    }
    free(otf);
  }
}

/*
 * Calculate the non-linear potential grid (including backflow).
 *
 * otf        = OT 2D functional structure.
 * potential  = Potential grid where the result will be stored (output).
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

EXPORT void dft_ot2d_potential(dft_ot_functional_2d *otf, cgrid2d *potential, wf2d *wf, rgrid2d *density, rgrid2d *workspace1, rgrid2d *workspace2, rgrid2d *workspace3, rgrid2d *workspace4, rgrid2d *workspace5, rgrid2d *workspace6, rgrid2d *workspace7, rgrid2d *workspace8, rgrid2d *workspace9) {


  if(otf->model & DFT_ZERO) {
    fprintf(stderr, "libdft: Warning - zero potential used.\n");
    cgrid2d_zero(potential);
    return;
  }

  if(otf->model & DFT_GP) {
    rgrid2d_copy(workspace1, density);
    rgrid2d_multiply(workspace1, otf->mu0 / otf->rho0);
    grid2d_add_real_to_complex_re(potential, workspace1);
    return;
  }

  /* transform of rho (wrk1) */
  rgrid2d_copy(workspace1, density);
  rgrid2d_fft_cylindrical(workspace1);

  /* Lennard-Jones */  
  /* int rho(r') Vlj(r-r') dr' */
  rgrid2d_fft_cylindrical_convolute(workspace2, workspace1, otf->lennard_jones);
  rgrid2d_inverse_fft_cylindrical(workspace2);
  /* kill spurious Hankel contribution at end */
  rgrid2d_fft_cylindrical_cleanup(workspace2, dft_ot2d_hankel_pad);
  grid2d_add_real_to_complex_re(potential, workspace2);

  /* Non-linear local correlation */
  /* note workspace1 = fft of \rho */
  dft_ot2d_add_local_correlation_potential(otf, potential, density, workspace1 /* rho_tf */, workspace2, workspace3);

  if(otf->model & DFT_OT_KC)
    /* Non-local correlation for kinetic energy */
    dft_ot2d_add_nonlocal_correlation_potential(otf, potential, density, workspace1, workspace2, workspace3, workspace4, workspace5, workspace6);

  if((otf->model & DFT_OT_HD) || (otf->model & DFT_OT_HD2))
    /* Barranco's penalty term */
    dft_ot2d_add_barranco(otf, potential, density, workspace1);

  if(otf->model & DFT_OT_BACKFLOW) {
    /* wf, veloc_z(1), veloc_r(2), wrk(3) */
    grid2d_wf_probability_flux(wf, workspace1, workspace2);
    // grid2d_wf_momentum(wf, workspace1, workspace2, workspace3);   /* this would imply FFT boundaries */
    rgrid2d_division_eps(workspace1, workspace1, density, DFT_BF_EPS); /* velocity = flux / rho */
    rgrid2d_division_eps(workspace2, workspace2, density, DFT_BF_EPS);
    dft_ot2d_backflow_potential(otf, potential, density, workspace1 /* veloc_z */, workspace2 /* veloc_r */, workspace3, workspace4, workspace5, workspace6, workspace7);
  }

  if(otf->model >= DFT_OT_T400MK && !(otf->model & DFT_DR))
    /* include the ideal gas contribution */
    dft_ot2d_add_ancilotto(otf, potential, density, workspace1);
}

/* local function */
static void dft_ot2d_add_local_correlation_potential(dft_ot_functional_2d *otf, cgrid2d *potential, const rgrid2d *rho, const rgrid2d *rho_tf, rgrid2d *workspace1, rgrid2d *workspace2) {

  /* wrk1 = \bar{\rho} */
  rgrid2d_fft_cylindrical_convolute(workspace1, (rgrid2d *) rho_tf, otf->spherical_avg);
  rgrid2d_inverse_fft_cylindrical(workspace1);

  /* c2 */
  rgrid2d_abs_power(workspace2, workspace1, otf->c2_exp);
  rgrid2d_multiply(workspace2, otf->c2 / 2.0);
  rgrid2d_fft_cylindrical_cleanup(workspace2, dft_ot2d_hankel_pad);
  grid2d_add_real_to_complex_re(potential, workspace2);
  
  rgrid2d_abs_power(workspace2, workspace1, otf->c2_exp - 1.0);
  rgrid2d_product(workspace2, workspace2, rho);
  rgrid2d_fft_cylindrical(workspace2);
  rgrid2d_fft_cylindrical_convolute(workspace2, workspace2, otf->spherical_avg);
  rgrid2d_inverse_fft_cylindrical(workspace2);
  rgrid2d_multiply(workspace2, otf->c2 * otf->c2_exp / 2.0);
  rgrid2d_fft_cylindrical_cleanup(workspace2, dft_ot2d_hankel_pad);
  grid2d_add_real_to_complex_re(potential, workspace2);

  /* c3 */
  rgrid2d_abs_power(workspace2, workspace1, otf->c3_exp);
  rgrid2d_multiply(workspace2, otf->c3 / 3.0);
  rgrid2d_fft_cylindrical_cleanup(workspace2, dft_ot2d_hankel_pad);
  grid2d_add_real_to_complex_re(potential, workspace2);
  
  rgrid2d_abs_power(workspace2, workspace1, otf->c3_exp - 1.0);
  rgrid2d_product(workspace2, workspace2, rho);
  rgrid2d_fft_cylindrical(workspace2);
  rgrid2d_fft_cylindrical_convolute(workspace2, workspace2, otf->spherical_avg);
  rgrid2d_inverse_fft_cylindrical(workspace2);
  rgrid2d_multiply(workspace2, otf->c3 * otf->c3_exp / 3.0);
  rgrid2d_fft_cylindrical_cleanup(workspace2, dft_ot2d_hankel_pad);
  grid2d_add_real_to_complex_re(potential, workspace2);
}

/* local function */
static void dft_ot2d_add_nonlocal_correlation_potential(dft_ot_functional_2d *otf, cgrid2d *potential, const rgrid2d *rho, rgrid2d *rho_tf, rgrid2d *workspace1, rgrid2d *workspace2, rgrid2d *workspace3, rgrid2d *workspace4, rgrid2d *workspace5) {

  /* \tilde{rho}(r) = int F(r-r') rho(r') dr' */
  rgrid2d_fft_cylindrical_convolute(workspace1, otf->gaussian_tf, rho_tf);
  rgrid2d_inverse_fft_cylindrical(workspace1);
  /* workspace1 = rho_st = 1 - 1/\tilde{\rho}/\rho_{0s} */
  rgrid2d_fft_cylindrical_cleanup(workspace1, dft_ot2d_hankel_pad);
  rgrid2d_multiply(workspace1, -1.0 / otf->rho_0s);
  rgrid2d_add(workspace1, 1.0);

  dft_ot2d_add_nonlocal_correlation_potential_z(otf, potential, rho, rho_tf, workspace1 /* rho_st */, workspace2, workspace3, workspace4, workspace5);
  dft_ot2d_add_nonlocal_correlation_potential_r(otf, potential, rho, rho_tf, workspace1 /* rho_st */, workspace2, workspace3, workspace4, workspace5);
}

/* local function */
static void dft_ot2d_add_nonlocal_correlation_potential_z(dft_ot_functional_2d *otf, cgrid2d *potential, const rgrid2d *rho, rgrid2d *rho_tf, rgrid2d *rho_st, rgrid2d *workspace1, rgrid2d *workspace2, rgrid2d *workspace3, rgrid2d *workspace4) {

  double c;

  c = otf->alpha_s / (2.0 * otf->mass);

  /* workspace1 = (d/dz) rho */
  rgrid2d_fd_gradient_cyl_z(rho, workspace1);

  /*** 1st term ***/

  /* Construct workspace2 = FFT(G) = FFT((d/dz) \rho(r_1) * (1 - \tilde{\rho(r_1)} / \rho_{0s})) */
  rgrid2d_product(workspace2, workspace1, rho_st);  // rho_st = (1 - \tilde{\rho} / \rho_{0s})
  rgrid2d_fft_cylindrical(workspace2);

  /* 1st term: c convolute [((d/dz) F) . G] */
  rgrid2d_fft_cylindrical_convolute(workspace3, otf->gaussian_z_tf, workspace2);
  rgrid2d_inverse_fft_cylindrical(workspace3);
  rgrid2d_fft_cylindrical_cleanup(workspace3, dft_ot2d_hankel_pad);
  rgrid2d_product(workspace3, workspace3, rho_st);
  rgrid2d_multiply(workspace3, c);
  grid2d_add_real_to_complex_re(potential, workspace3);  

  /* in use: workspace1 (grad rho), workspace2 (FFT(G)) */

  /*** 2nd term ***/
  
  /* Construct workspace3 = J = convolution(F G) */
  rgrid2d_fft_cylindrical_convolute(workspace3, otf->gaussian_tf, workspace2);
  rgrid2d_inverse_fft_cylindrical(workspace3);
  rgrid2d_fft_cylindrical_cleanup(workspace3, dft_ot2d_hankel_pad);

  /* Construct workspace4 = FFT(H) = FFT((d/dz) \rho * J) */
  rgrid2d_copy(workspace4, workspace3);
  rgrid2d_product(workspace4, workspace1, workspace4);
  rgrid2d_fft_cylindrical(workspace4);

  /* 2nd term: c convolute(F H) */
  rgrid2d_fft_cylindrical_convolute(workspace4, otf->gaussian_tf, workspace4);
  rgrid2d_inverse_fft_cylindrical(workspace4);
  rgrid2d_fft_cylindrical_cleanup(workspace4, dft_ot2d_hankel_pad);
  rgrid2d_multiply(workspace4, c);
  grid2d_add_real_to_complex_re(potential, workspace4);

  /* workspace1 (grad rho), workspace3 (J) */

  /*** 3rd term ***/
  
  /* -c J . convolute((d/dz)F \rho) */
  rgrid2d_fft_cylindrical_convolute(workspace2, otf->gaussian_z_tf, rho_tf);
  rgrid2d_inverse_fft_cylindrical(workspace2);
  rgrid2d_fft_cylindrical_cleanup(workspace2, dft_ot2d_hankel_pad);
  rgrid2d_product(workspace2, workspace2, workspace3);
  rgrid2d_multiply(workspace2, -c);
  grid2d_add_real_to_complex_re(potential, workspace2);
}

/* local function */
/* There are even / odd function issues here (with respect to r). For example F is even around r = 0 but (d/dr)F is not! */
static void dft_ot2d_add_nonlocal_correlation_potential_r(dft_ot_functional_2d *otf, cgrid2d *potential, const rgrid2d *rho, rgrid2d *rho_tf, rgrid2d *rho_st, rgrid2d *workspace1, rgrid2d *workspace2, rgrid2d *workspace3, rgrid2d *workspace4) {

  double c;

  /* WARNING: We are cheating here! (d/dr) \rho is not cylindrically symmetric! However, it seems that this is not necessarily a big issue in practice. */
  /* It is interesting that (d/dr) F definitely has to be evaluated such that d/dr is taken AFTER then convolution (F is symmetric), otherwise the convolution */
  /* would fail completely. So apparently it can somehow deal with one anti-symmetric function but two is too much! */
  /* This needs more attention ! */
  /* The r component is included twice as the dot products are evaluated in catesian coordinates (r = {x,y}) */
  
  c = otf->alpha_s / (2.0 * otf->mass);

  /* workspace1 = (d/dr) rho */
  rgrid2d_fd_gradient_cyl_r(rho, workspace1);

  /*** 1st term ***/

  /* Construct workspace2 = FFT(G) = FFT((d/dr) \rho(r_1) * (1 - \tilde{\rho(r_1)} / \rho_{0s})) */
  /* workspace4 = (1 - \tilde{\rho(r_1)} / \rho_{0s})) */
  rgrid2d_product(workspace2, workspace1, rho_st);  // rho_st = (1 - \tilde{\rho(r_1)} / \rho_{0s}))
  rgrid2d_fft_cylindrical(workspace2);

  /* 1st term: c convolute [((d/dr) F) . G] -- tricky business with symmetric and anti-symmetric functions (with respect to r = 0) */
  /* It seems that taking derivative outside the integral produces better results */
  rgrid2d_fft_cylindrical_convolute(workspace4, otf->gaussian_tf, workspace2);
  rgrid2d_inverse_fft_cylindrical(workspace4);
  rgrid2d_fft_cylindrical_cleanup(workspace4, dft_ot2d_hankel_pad);
  rgrid2d_fd_gradient_cyl_r(workspace4, workspace3);
  rgrid2d_product(workspace3, workspace3, rho_st);
  rgrid2d_multiply(workspace3, 2.0 * c);      /* cartesian dot product, 2 X */
  grid2d_add_real_to_complex_re(potential, workspace3);  

  /* in use: workspace1 (grad rho), workspace2 (FFT(G)) */

  /*** 2nd term ***/
  
  /* Construct workspace3 = J = convolution(F G) */
  rgrid2d_fft_cylindrical_convolute(workspace3, otf->gaussian_tf, workspace2);
  rgrid2d_inverse_fft_cylindrical(workspace3);
  rgrid2d_fft_cylindrical_cleanup(workspace3, dft_ot2d_hankel_pad);

  /* Construct workspace4 = FFT(H) = FFT((d/dr) \rho * J) */
  rgrid2d_copy(workspace4, workspace3);
  rgrid2d_product(workspace4, workspace1, workspace4);
  rgrid2d_fft_cylindrical(workspace4);

  /* 2nd term: c convolute(F H) */
  rgrid2d_fft_cylindrical_convolute(workspace4, otf->gaussian_tf, workspace4);
  rgrid2d_inverse_fft_cylindrical(workspace4);
  rgrid2d_fft_cylindrical_cleanup(workspace4, dft_ot2d_hankel_pad);
  rgrid2d_multiply(workspace4, 2.0 * c);    /* Cartesian dot product, 2 X */
  grid2d_add_real_to_complex_re(potential, workspace4);

  /* workspace1 (grad rho), workspace3 (J) */

  /*** 3rd term ***/
  
  /* -c J . convolute((d/dr)F \rho)  = -c J . (d/dr) convolute(F \rho) */
  rgrid2d_fft_cylindrical_convolute(workspace2, otf->gaussian_tf, rho_tf);
  rgrid2d_inverse_fft_cylindrical(workspace2);
  rgrid2d_fft_cylindrical_cleanup(workspace2, dft_ot2d_hankel_pad);
  rgrid2d_fd_gradient_cyl_r(workspace2, workspace4);
  rgrid2d_product(workspace2, workspace4, workspace3);
  rgrid2d_multiply(workspace2, -2.0 * c);    /* Cartesian dot product, 2 X */
  grid2d_add_real_to_complex_re(potential, workspace2);
}

/* local function */
static void dft_ot2d_add_ancilotto(dft_ot_functional_2d *otf, cgrid2d *potential, const rgrid2d *rho, rgrid2d *workspace1) {

  dft_common_idealgas_params(otf->temp, otf->mass, otf->c4);
  rgrid2d_operate_one(workspace1, rho, dft_common_idealgas_op);
  grid2d_add_real_to_complex_re(potential, workspace1);
}

/* local function */
static void dft_ot2d_add_barranco(dft_ot_functional_2d *otf, cgrid2d *potential, const rgrid2d *rho, rgrid2d *workspace1) {

  XXX_beta = otf->beta;
  XXX_rhom = otf->rhom;
  XXX_C = otf->C;
  rgrid2d_operate_one(workspace1, rho, dft_ot2d_barranco_op);
  grid2d_add_real_to_complex_re(potential, workspace1);
}

/*
 * Evaluate the potential part to the energy density.
 * Note: the single particle kinetic portion is NOT included.
 *       (use grid2d_wf_kinetic_energy() to calculate this separately)
 *
 * otf            = OT 2D functional structure.
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

EXPORT void dft_ot2d_energy_density(dft_ot_functional_2d *otf, rgrid2d *energy_density, wf2d *wf, const rgrid2d *density, rgrid2d *workspace1, rgrid2d *workspace2, rgrid2d *workspace3, rgrid2d *workspace4, rgrid2d *workspace5, rgrid2d *workspace6, rgrid2d *workspace7, rgrid2d *workspace8) {
  
  if(otf->model & DFT_GP) {
    fprintf(stderr, "Energy density for DFT_GP not implemented (TODO).\n");
    exit(1);
  }

  rgrid2d_zero(energy_density);

  /* wrk1 = FFT(\rho) */
  rgrid2d_copy(workspace1, density);
  rgrid2d_fft_cylindrical(workspace1);

  /* Lennard-Jones */  
  /* (1/2) rho(r) int V_lj(|r-r'|) rho(r') dr' */
  rgrid2d_fft_cylindrical_convolute(workspace2, workspace1, otf->lennard_jones);
  rgrid2d_inverse_fft_cylindrical(workspace2);
  rgrid2d_fft_cylindrical_cleanup(workspace2, dft_ot2d_hankel_pad);
  rgrid2d_add_scaled_product(energy_density, 0.5, density, workspace2);

  /* local correlation */
  /* wrk1 = \bar{\rho} */
  rgrid2d_fft_cylindrical_convolute(workspace1, workspace1, otf->spherical_avg);
  rgrid2d_inverse_fft_cylindrical(workspace1);
  rgrid2d_fft_cylindrical_cleanup(workspace1, dft_ot2d_hankel_pad);

  /* (C2/2) * \rho * \bar{\rho}^2 */
  rgrid2d_abs_power(workspace2, workspace1, otf->c2_exp);
  rgrid2d_product(workspace2, workspace2, density);
  rgrid2d_add_scaled(energy_density, otf->c2 / 2.0, workspace2);

  /* (C3/3) * \rho * \bar{\rho}^3 */
  rgrid2d_abs_power(workspace2, workspace1, otf->c3_exp);
  rgrid2d_product(workspace2, workspace2, density);
  rgrid2d_add_scaled(energy_density, otf->c3 / 3.0, workspace2);

  /* Barranco's contribution (high density) */
  if((otf->model & DFT_OT_HD) || (otf->model & DFT_OT_HD2)) {
    XXX_beta = otf->beta;
    XXX_rhom = otf->rhom;
    XXX_C = otf->C;
    rgrid2d_operate_one(workspace1, density, dft_ot2d_barranco_energy_op);
    rgrid2d_sum(energy_density, energy_density, workspace1);
  }

  /* Ancilotto's contribution (thermal) */
  if(otf->model >= DFT_OT_T400MK) {
    dft_common_idealgas_params(otf->temp, otf->mass, otf->c4);
    rgrid2d_operate_one(workspace1, density, dft_common_idealgas_energy_op);
    rgrid2d_sum(energy_density, energy_density, workspace1);
  }
  
  /* begin kinetic energy correlation energy density */
  if(otf->model & DFT_OT_KC) { /* new code */
    /* 1. convolute density with F to get \tilde{\rho} (wrk1) */
    rgrid2d_copy(workspace2, density);
    rgrid2d_fft_cylindrical(workspace2);
    rgrid2d_fft_cylindrical_convolute(workspace1, workspace2, otf->gaussian_tf);   /* otf->gaussian_tf is already in Fourier space */
    rgrid2d_inverse_fft_cylindrical(workspace1);
    rgrid2d_fft_cylindrical_cleanup(workspace1, dft_ot2d_hankel_pad);

    /* 2. modify wrk1 from \tilde{\rho} to (1 - \tilde{\rho}/\rho_{0s} */
    rgrid2d_multiply_and_add(workspace1, -1.0/otf->rho_0s, 1.0);

    /* 3. gradient \rho to wrk3 (z), wrk4 (r) */
    rgrid2d_fd_gradient_cyl_z(density, workspace3); /* d/dz (cartesian) */
    rgrid2d_fd_gradient_cyl_r(density, workspace4); /* d/dr (cartesian) */
    
    /* 4. Z component: wrk6 = wrk3 * wrk1 (wrk1 = (d/dz)\rho_z * (1 - \tilde{\rho}/\rho_{0s})) */
    /*    R component: wrk7 = wrk4 * wrk1 (wrk1 = (d/dr)\rho_r * (1 - \tilde{\rho}/\rho_{0s})) (include twice in dot product) */
    /* workspace 5 is now free */
    rgrid2d_product(workspace6, workspace3, workspace1);
    rgrid2d_product(workspace7, workspace4, workspace1);

    /* 5. convolute (Z): wrk6 = convolution(otf->gaussian * wrk6). */
    /*    convolute (R): wrk7 = convolution(otf->gaussian * wrk7). */
    rgrid2d_fft_cylindrical(workspace6);
    rgrid2d_fft_cylindrical(workspace7);
    rgrid2d_fft_cylindrical_convolute(workspace6, workspace6, otf->gaussian_tf);
    rgrid2d_fft_cylindrical_convolute(workspace7, workspace7, otf->gaussian_tf);
    rgrid2d_inverse_fft_cylindrical(workspace6);
    rgrid2d_fft_cylindrical_cleanup(workspace6, dft_ot2d_hankel_pad);
    rgrid2d_inverse_fft_cylindrical(workspace7);
    rgrid2d_fft_cylindrical_cleanup(workspace7, dft_ot2d_hankel_pad);

    /* 6. Z: wrk6 = wrk6 * wrk3 * wrk1 */
    /*    R: wrk7 = wrk7 * wrk4 * wrk1 */
    rgrid2d_product(workspace6, workspace6, workspace3);
    rgrid2d_product(workspace6, workspace6, workspace1);
    rgrid2d_product(workspace7, workspace7, workspace4);
    rgrid2d_product(workspace7, workspace7, workspace1);
    
    /* 7. add wrk6 + 2 x wrk7 (components from the CARTESIAN dot product) */
    /* 8. multiply by -\hbar^2\alpha_s/(4M_{He}) */    
    rgrid2d_add_scaled(energy_density, -otf->alpha_s / (4.0 * otf->mass), workspace6);
    rgrid2d_add_scaled(energy_density, -2.0 * otf->alpha_s / (4.0 * otf->mass), workspace7);  // both X, Y compinents -> 2x
  }
  if(otf->model & DFT_OT_BACKFLOW) {  /* copied from 3D, workspace3 not in use */
    grid2d_wf_probability_flux(wf, workspace1, workspace2);    /* finite difference */
    // grid2d_wf_momentum(wf, workspace1, workspace2, workspace4);   /* this would imply FFT boundaries */
    rgrid2d_division_eps(workspace1, workspace1, density, DFT_BF_EPS);  /* velocity = flux / rho, v_z */
    rgrid2d_division_eps(workspace2, workspace2, density, DFT_BF_EPS);  /* v_r */
    rgrid2d_product(workspace4, workspace1, workspace1);   /* v_z^2 */
    rgrid2d_product(workspace5, workspace2, workspace2);   /* v_r^2 */
    rgrid2d_sum(workspace4, workspace4, workspace5);       /* wrk4 = v_z^2 + v_r^2 */
    rgrid2d_sum(workspace4, workspace4, workspace5);       /* evaluated in cartesian coords: r appears twice */

    /* Term 1: -(M/4) * rho(r) * v(r)^2 \int U_j(|r - r'|) * rho(r') d3r' */
    rgrid2d_copy(workspace5, density);
    rgrid2d_fft_cylindrical(workspace5);     /* This was done before - TODO: save previous rho FFT and reuse here */
    rgrid2d_fft_cylindrical_convolute(workspace6, otf->backflow_pot, workspace5);
    rgrid2d_inverse_fft_cylindrical(workspace6);
    rgrid2d_fft_cylindrical_cleanup(workspace6, dft_ot2d_hankel_pad);
    rgrid2d_product(workspace6, workspace6, workspace4);
    rgrid2d_product(workspace6, workspace6, density);
    rgrid2d_add_scaled(energy_density, -otf->mass / 4.0, workspace6);
    /* Term 2: +(M/2) * rho(r) v(r) . \int U_j(|r - r'|) * rho(r') v(r') d3r' */
    /* z contribution */
    rgrid2d_product(workspace5, density, workspace1);   /* rho(r') * v_z(r') */
    rgrid2d_fft_cylindrical(workspace5);
    rgrid2d_fft_cylindrical_convolute(workspace6, otf->backflow_pot, workspace5);
    rgrid2d_inverse_fft_cylindrical(workspace6);
    rgrid2d_fft_cylindrical_cleanup(workspace6, dft_ot2d_hankel_pad);
    rgrid2d_product(workspace6, workspace6, density);
    rgrid2d_product(workspace6, workspace6, workspace1);
    rgrid2d_add_scaled(energy_density, otf->mass / 2.0, workspace6);
    /* r contribution */
    rgrid2d_product(workspace5, density, workspace2);   /* rho(r') * v_r(r') */
    rgrid2d_fft_cylindrical(workspace5);
    rgrid2d_fft_cylindrical_convolute(workspace6, otf->backflow_pot, workspace5);
    rgrid2d_inverse_fft_cylindrical(workspace6);
    rgrid2d_fft_cylindrical_cleanup(workspace6, dft_ot2d_hankel_pad);
    rgrid2d_product(workspace6, workspace6, density);
    rgrid2d_product(workspace6, workspace6, workspace2);
    rgrid2d_add_scaled(energy_density, 2.0 * otf->mass / 2.0, workspace6); /* in cartesian: 2 X r contrib */
    /* Term 3: -(M/4) rho(r) \int U_j(|r - r'|) rho(r') v^2(r') d3r' */
    rgrid2d_product(workspace5, density, workspace4);
    rgrid2d_fft_cylindrical(workspace5);
    rgrid2d_fft_cylindrical_convolute(workspace6, otf->backflow_pot, workspace5);
    rgrid2d_inverse_fft_cylindrical(workspace6);
    rgrid2d_fft_cylindrical_cleanup(workspace6, dft_ot2d_hankel_pad);
    rgrid2d_product(workspace6, workspace6, density);
    rgrid2d_add_scaled(energy_density, -otf->mass / 4.0, workspace6);
  }
}

/*
 * Add the backflow non-linear potential.
 *
 * otf            = OT 2D functional structure.
 * potential      = potential grid (output). Not cleared.
 * density        = liquid density grid (input).
 * veloc_z        = veloc grid (with respect to Z; input - overwritten on exit).
 * veloc_r        = veloc grid (with respect to R; input - overwritten on exit).
 * workspace1     = Workspace grid (must be allocated by the user).
 * workspace2     = Workspace grid (must be allocated by the user).
 * workspace3     = Workspace grid (must be allocated by the user).
 * workspace4     = Workspace grid (must be allocated by the user).
 * workspace5     = Workspace grid (must be allocated by the user).
 *
 * No return value.
 *
 */

EXPORT void dft_ot2d_backflow_potential(dft_ot_functional_2d *otf, cgrid2d *potential, const rgrid2d *density, rgrid2d *veloc_z, rgrid2d *veloc_r, rgrid2d *workspace1, rgrid2d *workspace2, rgrid2d *workspace3, rgrid2d *workspace4, rgrid2d *workspace5) {
  
  /* Calculate A (workspace1) [scalar] */
  rgrid2d_copy(workspace1, density);
  rgrid2d_fft_cylindrical(workspace1);
  rgrid2d_fft_cylindrical_convolute(workspace1, workspace1, otf->backflow_pot);
  rgrid2d_inverse_fft_cylindrical(workspace1);
  rgrid2d_fft_cylindrical_cleanup(workspace1, dft_ot2d_hankel_pad);

  /* Calculate C (workspace2) [scalar] */
  rgrid2d_product(workspace2, veloc_z, veloc_z);
  rgrid2d_add_scaled_product(workspace2, 1.0, veloc_r, veloc_r);
  rgrid2d_product(workspace2, workspace2, density);
  rgrid2d_fft_cylindrical(workspace2);
  rgrid2d_fft_cylindrical_convolute(workspace2, workspace2, otf->backflow_pot);
  rgrid2d_inverse_fft_cylindrical(workspace2);
  rgrid2d_fft_cylindrical_cleanup(workspace2, dft_ot2d_hankel_pad);

  /* Calculate B (workspace3 (z), workspace4 (r)) [vector] */
  rgrid2d_product(workspace3, density, veloc_z);
  rgrid2d_fft_cylindrical(workspace3);
  rgrid2d_fft_cylindrical_convolute(workspace3, workspace3, otf->backflow_pot);
  rgrid2d_inverse_fft_cylindrical(workspace3);
  rgrid2d_fft_cylindrical_cleanup(workspace3, dft_ot2d_hankel_pad);

  rgrid2d_product(workspace4, density, veloc_r);
  rgrid2d_fft_cylindrical(workspace4);
  rgrid2d_fft_cylindrical_convolute(workspace4, workspace4, otf->backflow_pot);
  rgrid2d_inverse_fft_cylindrical(workspace4);
  rgrid2d_fft_cylindrical_cleanup(workspace4, dft_ot2d_hankel_pad);

  /* 1. Calculate the real part of the potential */

  /* -(m/2) (v(r) . (v(r)A(r) - 2B(r)) + C(r))  = -(m/2) [ A(v_x^2 + v_y^2 + v_z^2) - 2v_x B_x - 2v_y B_y - 2v_z B_z + C] */
  rgrid2d_product(workspace5, veloc_z, veloc_z);
  rgrid2d_add_scaled_product(workspace5, 1.0, veloc_r, veloc_r);
  rgrid2d_product(workspace5, workspace5, workspace1);
  rgrid2d_add_scaled_product(workspace5, -2.0, veloc_z, workspace3);
  rgrid2d_add_scaled_product(workspace5, -2.0, veloc_r, workspace4);
  rgrid2d_sum(workspace5, workspace5, workspace2);
  rgrid2d_multiply(workspace5, -0.5 * otf->mass);
  grid2d_add_real_to_complex_re(potential, workspace5);

  /* workspace2 (C), workspace5 not used after this point */

  /* 2. Calculate the imaginary part of the potential */

  rgrid2d_zero(workspace5);
  
  /* v_z -> v_zA - B_z, v_r -> v_rA - B_r (velocities are overwritten here) */
  rgrid2d_product((rgrid2d *) veloc_z, veloc_z, workspace1);
  rgrid2d_add_scaled((rgrid2d *) veloc_z, -1.0, workspace3);
  rgrid2d_product((rgrid2d *) veloc_r, veloc_r, workspace1);
  rgrid2d_add_scaled((rgrid2d *) veloc_r, -1.0, workspace4);

  /* 1.1 (drho/dz)/rho * (v_zA - B_z) */
  rgrid2d_fd_gradient_cyl_z(density, workspace2);
  rgrid2d_division_eps(workspace2, workspace2, density, DFT_BF_EPS);
  rgrid2d_add_scaled_product(workspace5, 0.5, workspace2, veloc_z);

  /* 1.2 (drho/dr)/rho * (v_rA - B_r) */
  rgrid2d_fd_gradient_cyl_r(density, workspace2);
  rgrid2d_division_eps(workspace2, workspace2, density, DFT_BF_EPS);
  /* dot product: 2 X r */
  rgrid2d_add_scaled_product(workspace5, 0.5, workspace2, veloc_r);

  /* 2.1 (d/dz) (v_zA - B_z) */
  rgrid2d_fd_gradient_cyl_z(veloc_z, workspace2);
  rgrid2d_add_scaled(workspace5, 0.5, workspace2);

  /* 2.2 (d/dr) (v_rA - B_r) */
  rgrid2d_fd_gradient_cyl_r(veloc_r, workspace2);
  rgrid2d_add_scaled(workspace5, 0.5, workspace2);

  grid2d_add_real_to_complex_im(potential, workspace5);
}

EXPORT inline void dft_ot_temperature_2d(dft_ot_functional_2d *otf, long model) {

  fprintf(stderr, "libdft: Model = %ld\n", model);

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

  if(model & DFT_GP) {
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
