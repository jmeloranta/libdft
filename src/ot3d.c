/*
 * Orsay-Trento functional for superfluid helium in 3D.
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

#define MM_BACKFLOW   /* Include Pi & Barranco cutoff correction to backflow at high densities (where BF is unstable otherwise) */

static void dft_ot3d_add_nonlocal_correlation_potential(dft_ot_functional *otf, cgrid3d *potential, rgrid3d *rho, rgrid3d *rho_tf, rgrid3d *workspace1, rgrid3d *workspace2, rgrid3d *workspace3, rgrid3d *workspace4, rgrid3d *workspace5);
static void dft_ot3d_add_nonlocal_correlation_potential_x(dft_ot_functional *otf, cgrid3d *potential, rgrid3d *rho, rgrid3d *rho_tf, rgrid3d *rho_st, rgrid3d *workspace1, rgrid3d *workspace2, rgrid3d *workspace3, rgrid3d *workspace4);
static void dft_ot3d_add_nonlocal_correlation_potential_y(dft_ot_functional *otf, cgrid3d *potential, rgrid3d *rho, rgrid3d *rho_tf, rgrid3d *rho_st, rgrid3d *workspace1, rgrid3d *workspace2, rgrid3d *workspace3, rgrid3d *workspace4);
static void dft_ot3d_add_nonlocal_correlation_potential_z(dft_ot_functional *otf, cgrid3d *potential, rgrid3d *rho, rgrid3d *rho_tf, rgrid3d *rho_st, rgrid3d *workspace1, rgrid3d *workspace2, rgrid3d *workspace3, rgrid3d *workspace4);
static void dft_ot3d_add_barranco(dft_ot_functional *otf, cgrid3d *potential, rgrid3d *rho, rgrid3d *workspace1);
static void dft_ot3d_add_ancilotto(dft_ot_functional *otf, cgrid3d *potential, rgrid3d *rho, rgrid3d *workspace1);

/* local functions */
static REAL XXX_xi, XXX_rhobf;
static inline REAL dft_ot3d_bf_pi_energy_op2(REAL rhop) {

  // TODO: there are indications that tanh() is really slow (possibly bug in glibc)
  return 0.5 * (1.0 - TANH(XXX_xi * (rhop - XXX_rhobf)));    /* G(rho) */
}

static inline REAL dft_ot3d_bf_pi_energy_op(REAL rhop) {

  return rhop * dft_ot3d_bf_pi_energy_op2(rhop);    /* rho * G(rho) */
}

static inline REAL dft_ot3d_bf_pi_op(REAL rhop) {   /* rho * dG/drho + G */

  REAL tmp = 1.0 / (COSH((rhop - XXX_rhobf) * XXX_xi) + DFT_BF_EPS);
  
  return rhop * (-0.5 * XXX_xi * tmp * tmp) + dft_ot3d_bf_pi_energy_op2(rhop);
}

/* local function */
static REAL XXX_beta, XXX_rhom, XXX_C;
static inline REAL dft_ot3d_barranco_op(REAL rhop) {

  REAL stanh = TANH(XXX_beta * (rhop - XXX_rhom));

  return (XXX_C * ((1.0 + stanh) + XXX_beta * rhop * (1.0 - stanh * stanh)));
}

/* local function */
static inline REAL dft_ot3d_barranco_energy_op(REAL rhop) {

  return (XXX_C * (1.0 + TANH(XXX_beta * (rhop - XXX_rhom))) * rhop);
}

EXPORT REAL dft_ot_backflow_pot(void *arg, REAL x, REAL y, REAL z) {

  REAL g11 = ((dft_ot_bf *) arg)->g11;
  REAL g12 = ((dft_ot_bf *) arg)->g12;
  REAL g21 = ((dft_ot_bf *) arg)->g21;
  REAL g22 = ((dft_ot_bf *) arg)->g22;
  REAL a1 = ((dft_ot_bf *) arg)->a1;
  REAL a2 = ((dft_ot_bf *) arg)->a2;
  REAL r2 = x * x + y * y + z * z;
  
  return (g11 + g12 * r2) * EXP(-a1 * r2) + (g21 + g22 * r2) * EXP(-a2 * r2);
}

/*
 * Allocate 3D OT functional. This must be called first.
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
 *         DFT_GP          Gross-Pitaevskii equation
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

EXPORT dft_ot_functional *dft_ot3d_alloc(INT model, INT nx, INT ny, INT nz, REAL step, char bc, INT min_substeps, INT max_substeps) {

  REAL radius, inv_width;
  dft_ot_functional *otf;
  REAL (*grid_type)(rgrid3d *, INT, INT, INT);
  REAL x0, y0, z0;
  
  otf = (dft_ot_functional *) malloc(sizeof(dft_ot_functional));
  otf->model = model;
  if (!otf) {
    fprintf(stderr, "libdft: Error in dft_ot3d_alloc(): Could not allocate memory for dft_ot_functional.\n");
    return 0;
  }
 
  /*
   * If we have symmetric/antisymmetric b.c. the 'kernel' grids
   * (those used for convolution) must be mapped from [0,L]
   * instead of [-L/2,L/2]. This can be done by setting the
   * appropiate origin x0,y0,z0
   */ 
  switch(bc) {
  case DFT_DRIVER_BC_NORMAL: 
  case DFT_DRIVER_BC_NEUMANN:   /* TODO: This should belong to the case below */
    grid_type = RGRID3D_PERIODIC_BOUNDARY;
    x0 = y0 = z0 = 0.0;
    break;
  case DFT_DRIVER_BC_X:
  case DFT_DRIVER_BC_Y:
  case DFT_DRIVER_BC_Z:
    /*  case DFT_DRIVER_BC_NEUMANN: */   /* See above */
    grid_type = RGRID3D_NEUMANN_BOUNDARY;
    x0 = -(((REAL) nx/2) + 0.5) * step;
    y0 = -(((REAL) ny/2) + 0.5) * step;
    z0 = -(((REAL) nz/2) + 0.5) * step;
    break;
  default:
    fprintf(stderr, "libdft: Illegal boundary type.\n");
    exit(1);
  }

  dft_ot_temperature(otf, model);
  /* these grids are not needed for GP */
  if(!(model & DFT_GP) && !(model & DFT_ZERO)) {
    otf->lennard_jones = rgrid3d_alloc(nx, ny, nz, step, grid_type, 0, "OT Lennard-Jones");
    rgrid3d_set_origin(otf->lennard_jones, x0, y0, z0);
    otf->spherical_avg = rgrid3d_alloc(nx, ny, nz, step, grid_type, 0, "OT Sph. average");
    rgrid3d_set_origin(otf->spherical_avg, x0, y0, z0);

    if(model & DFT_OT_KC) {
      otf->gaussian_tf = rgrid3d_alloc(nx, ny, nz, step, grid_type, 0, "OT KC Gauss TF");
      otf->gaussian_x_tf = rgrid3d_alloc(nx, ny, nz, step, grid_type, 0, "OT KC Gauss TF_x");
      otf->gaussian_y_tf = rgrid3d_alloc(nx, ny, nz, step, grid_type, 0, "OT KC Gauss TF_y");
      otf->gaussian_z_tf = rgrid3d_alloc(nx, ny, nz, step, grid_type, 0, "OT KC Gauss TF_z");
      rgrid3d_set_origin(otf->gaussian_tf, x0, y0, z0);
      rgrid3d_set_origin(otf->gaussian_x_tf, x0, y0, z0);
      rgrid3d_set_origin(otf->gaussian_y_tf, x0, y0, z0);
      rgrid3d_set_origin(otf->gaussian_z_tf, x0, y0, z0);

      if(!otf->gaussian_x_tf || !otf->gaussian_y_tf || !otf->gaussian_z_tf || !otf->gaussian_tf) {
	fprintf(stderr, "libdft: Error in dft_ot3d_alloc(): Could not allocate memory for gaussian.\n");
	return 0;
      }
    } else otf->gaussian_x_tf = otf->gaussian_y_tf = otf->gaussian_z_tf = otf->gaussian_tf = NULL;
  
    if(model & DFT_OT_BACKFLOW) {
      otf->backflow_pot = rgrid3d_alloc(nx, ny, nz, step, grid_type, 0, "OT Backflow");
      if(!otf->backflow_pot) {
	fprintf(stderr, "libdft: Error in dft_ot3d_alloc(): Could not allocate memory for backflow_pot.\n");
	return 0;
      }
    } else otf->backflow_pot = NULL;
  
    /* pre-calculate */
    fprintf(stderr, "libdft: LJ according to SD - ");
    rgrid3d_adaptive_map(otf->lennard_jones, dft_common_lennard_jones, &(otf->lj_params), min_substeps, max_substeps, 0.01 / GRID_AUTOK);
    rgrid3d_fft(otf->lennard_jones);
    /* Scaling of LJ so that the integral is exactly b */
    rgrid3d_multiply_fft(otf->lennard_jones, otf->b / (step * step * step * (REAL) rgrid3d_cvalue_at_index(otf->lennard_jones, 0, 0, 0)));
    fprintf(stderr, "Done.\n");

    if(otf->model & DFT_OT_HD2) {
      radius = otf->lj_params.h * 1.065; /* PRB 72, 214522 (2005) */
      fprintf(stderr, "libdft: Spherical average (new) - ");
    } else {
      radius = otf->lj_params.h;
      fprintf(stderr, "libdft: Spherical average (original) - ");
    }

    rgrid3d_adaptive_map(otf->spherical_avg, dft_common_spherical_avg, &radius, min_substeps, max_substeps, 0.01 / GRID_AUTOK);
    rgrid3d_fft(otf->spherical_avg);
    /* Scaling of sph. avg. so that the integral is exactly 1 */
    rgrid3d_multiply_fft(otf->spherical_avg, 1.0 / (step * step * step * (REAL) rgrid3d_cvalue_at_index(otf->spherical_avg, 0, 0, 0)));
    fprintf(stderr, "Done.\n");
    
    if(model & DFT_OT_KC) {
      inv_width = 1.0 / otf->l_g;
      fprintf(stderr, "libdft: Kinetic correlation - ");	
      rgrid3d_adaptive_map(otf->gaussian_tf, dft_common_gaussian, &inv_width, min_substeps, max_substeps, 0.01 / GRID_AUTOK);
      rgrid3d_fd_gradient_x(otf->gaussian_tf, otf->gaussian_x_tf);
      rgrid3d_fd_gradient_y(otf->gaussian_tf, otf->gaussian_y_tf);
      rgrid3d_fd_gradient_z(otf->gaussian_tf, otf->gaussian_z_tf);
      rgrid3d_fft(otf->gaussian_x_tf);
      rgrid3d_fft(otf->gaussian_y_tf);
      rgrid3d_fft(otf->gaussian_z_tf);
      rgrid3d_fft(otf->gaussian_tf);
      fprintf(stderr, "Done.\n");
    }
    
    if(model & DFT_OT_BACKFLOW) {
      fprintf(stderr, "libdft: Backflow - ");
      rgrid3d_adaptive_map(otf->backflow_pot, dft_ot_backflow_pot, &(otf->bf_params), min_substeps, max_substeps, 0.01 / GRID_AUTOK);
      rgrid3d_fft(otf->backflow_pot);
      fprintf(stderr, "Done.\n");
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
    //cgrid3d_zero(potential);
    return;
  }

  if(otf->model & DFT_GP) {
    /* the potential part is: \lambda \left|\psi\right|^2\psi - \mu\psi */
    /* with \lambda < 0 and \mu < 0. For bulk: \lambda\rho_0 - \mu = 0 and \lambda = \frac{\mu}{\rho_0} (which is < 0) */
    rgrid3d_copy(workspace1, density);
    rgrid3d_multiply(workspace1, otf->mu0 / otf->rho0);   // test - sign
    grid3d_add_real_to_complex_re(potential, workspace1);
    if((otf->model & DFT_OT_HD) || (otf->model & DFT_OT_HD2)) {
      /* Barranco's penalty term */
      rgrid3d_copy(workspace1, density);
      rgrid3d_fft(workspace1);
      dft_ot3d_add_barranco(otf, potential, density, workspace1);
    }
    return;
  }

  /* Lennard-Jones */  
  /* int rho(r') Vlj(r-r') dr' */
  dft_ot3d_add_lennard_jones_potential(otf, potential, density, workspace1, workspace2);
  /* workspace1 = FFT of density */

  /* Non-linear local correlation */
  /* note workspace1 = fft of \rho */
  dft_ot3d_add_local_correlation_potential(otf, potential, density, workspace1 /* rho_tf */, workspace2, workspace3, workspace4);

  /* Non-local correlation for kinetic energy (workspace1 = FFT(rho)) */
  if(otf->model & DFT_OT_KC)
    dft_ot3d_add_nonlocal_correlation_potential(otf, potential, density, workspace1 /* rho_tf */, workspace2, workspace3, workspace4, workspace5, workspace6);

  /* Barranco's penalty term */
  if((otf->model & DFT_OT_HD) || (otf->model & DFT_OT_HD2))
    dft_ot3d_add_barranco(otf, potential, density, workspace1);

  if(otf->model & DFT_OT_BACKFLOW) {
    /* wf, veloc_x(1), veloc_y(2), veloc_z(3), wrk(4) */
    grid3d_wf_probability_flux(wf, workspace1, workspace2, workspace3);    /* finite difference */
    // grid3d_wf_momentum(wf, workspace1, workspace2, workspace3, workspace4);   /* this would imply FFT boundaries */
    rgrid3d_division_eps(workspace1, workspace1, density, DFT_BF_EPS);  /* velocity = flux / rho */
    rgrid3d_division_eps(workspace2, workspace2, density, DFT_BF_EPS);
    rgrid3d_division_eps(workspace3, workspace3, density, DFT_BF_EPS);
    dft_ot3d_backflow_potential(otf, potential, density, workspace1 /* veloc_x */, workspace2 /* veloc_y */, workspace3 /* veloc_z */, workspace4, workspace5, workspace6, workspace7, workspace8, workspace9);
  }

  if(otf->model >= DFT_OT_T400MK && !(otf->model & DFT_DR))
    /* include the ideal gas contribution */
    dft_ot3d_add_ancilotto(otf, potential, density, workspace1);
}

EXPORT inline void dft_ot3d_add_lennard_jones_potential(dft_ot_functional *otf, cgrid3d *potential, rgrid3d *density, rgrid3d *workspace1, rgrid3d *workspace2) {

  rgrid3d_copy(workspace1, density);
  rgrid3d_fft(workspace1);

  rgrid3d_fft_convolute(workspace2, workspace1, otf->lennard_jones);  // Don't overwrite workspace1 - needed later
  rgrid3d_inverse_fft(workspace2);
  grid3d_add_real_to_complex_re(potential, workspace2);
  /* leave FFT(rho) in workspace1 (used later in local correlation potential as rho_tf) */
}

EXPORT inline void dft_ot3d_add_local_correlation_potential(dft_ot_functional *otf, cgrid3d *potential, rgrid3d *rho, rgrid3d *rho_tf, rgrid3d *workspace1, rgrid3d *workspace2, rgrid3d *workspace3) {

  /* workspace1 = \bar{\rho} */
  rgrid3d_fft_convolute(workspace1, rho_tf, otf->spherical_avg);
  rgrid3d_inverse_fft(workspace1); 

  /* C2.1 */
  rgrid3d_ipower(workspace2, workspace1, (INT) otf->c2_exp);
  rgrid3d_multiply(workspace2, otf->c2 / 2.0);
  grid3d_add_real_to_complex_re(potential, workspace2);

  /* C3.1 */
  rgrid3d_ipower(workspace2, workspace1, (INT) otf->c3_exp);
  rgrid3d_multiply(workspace2, otf->c3 / 3.0);
  grid3d_add_real_to_complex_re(potential, workspace2);

  /* C2.2 & C3.2 */
  rgrid3d_ipower(workspace2, workspace1, (INT) (otf->c2_exp - 1));
  rgrid3d_ipower(workspace3, workspace1, (INT) (otf->c3_exp - 1));
  rgrid3d_multiply(workspace2, otf->c2 * otf->c2_exp / 2.0);  // For OT, c2_exp / 2 = 1
  rgrid3d_multiply(workspace3, otf->c3 * otf->c3_exp / 3.0);  // For OT, c3_exp / 3 = 1
  rgrid3d_sum(workspace2, workspace2, workspace3);
  rgrid3d_product(workspace2, workspace2, rho);
  rgrid3d_fft(workspace2);
  rgrid3d_fft_convolute(workspace2, workspace2, otf->spherical_avg);
  rgrid3d_inverse_fft(workspace2);
  grid3d_add_real_to_complex_re(potential, workspace2);
}

/* local function */
static inline void dft_ot3d_add_nonlocal_correlation_potential(dft_ot_functional *otf, cgrid3d *potential, rgrid3d *rho, rgrid3d *rho_tf, rgrid3d *workspace1, rgrid3d *workspace2, rgrid3d *workspace3, rgrid3d *workspace4, rgrid3d *workspace5) {

  /* rho^tilde(r) = int F(r-r') rho(r') dr' */
  /* NOTE: rho_tf from LJ (workspace1 there). */
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
static inline void dft_ot3d_add_nonlocal_correlation_potential_x(dft_ot_functional *otf, cgrid3d *potential, rgrid3d *rho, rgrid3d *rho_tf, rgrid3d *rho_st, rgrid3d *workspace1, rgrid3d *workspace2, rgrid3d *workspace3, rgrid3d *workspace4) {

  REAL c;

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
  rgrid3d_product(workspace3, workspace3, rho_st);
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
static inline void dft_ot3d_add_nonlocal_correlation_potential_y(dft_ot_functional *otf, cgrid3d *potential, rgrid3d *rho, rgrid3d *rho_tf, rgrid3d *rho_st, rgrid3d *workspace1, rgrid3d *workspace2, rgrid3d *workspace3, rgrid3d *workspace4) {

  REAL c;

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
  rgrid3d_product(workspace3, workspace3, rho_st);
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
static inline void dft_ot3d_add_nonlocal_correlation_potential_z(dft_ot_functional *otf, cgrid3d *potential, rgrid3d *rho, rgrid3d *rho_tf, rgrid3d *rho_st, rgrid3d *workspace1, rgrid3d *workspace2, rgrid3d *workspace3, rgrid3d *workspace4) {

  REAL c;

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
  rgrid3d_product(workspace3, workspace3, rho_st);
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

/* local function (potential) */
static inline void dft_ot3d_add_ancilotto(dft_ot_functional *otf, cgrid3d *potential, rgrid3d *rho, rgrid3d *workspace1) {

  dft_common_idealgas_params(otf->temp, otf->mass, otf->c4);
  rgrid3d_operate_one(workspace1, rho, dft_common_bose_idealgas_dEdRho);
  grid3d_add_real_to_complex_re(potential, workspace1);
}

/* local function */
static inline void dft_ot3d_add_barranco(dft_ot_functional *otf, cgrid3d *potential, rgrid3d *rho, rgrid3d *workspace1) {

  XXX_beta = otf->beta;
  XXX_rhom = otf->rhom;
  XXX_C = otf->C;
  rgrid3d_operate_one(workspace1, rho, dft_ot3d_barranco_op);
  grid3d_add_real_to_complex_re(potential, workspace1);
}

/*
 * Evaluate the potential part to the energy density. Integrate to get the total energy.
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

EXPORT void dft_ot3d_energy_density(dft_ot_functional *otf, rgrid3d *energy_density, wf3d *wf, rgrid3d *density, rgrid3d *workspace1, rgrid3d *workspace2, rgrid3d *workspace3, rgrid3d *workspace4, rgrid3d *workspace5, rgrid3d *workspace6, rgrid3d *workspace7, rgrid3d *workspace8) {
  
  rgrid3d_zero(energy_density);

  if(otf->model & DFT_ZERO) {
    fprintf(stderr, "libdft: Warning - zero potential used.\n");
    return;
  }

  if(otf->model & DFT_GP) {
    rgrid3d_copy(energy_density, density);
    rgrid3d_product(energy_density, energy_density, density);
    /* the energy functional is: (\lambda/2)\int \left|\psi\right|^4 d\tau */
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
  rgrid3d_ipower(workspace2, workspace1, (INT) otf->c2_exp);
  rgrid3d_product(workspace2, workspace2, density);
  rgrid3d_add_scaled(energy_density, otf->c2 / 2.0, workspace2);

  /* C3 */
  rgrid3d_ipower(workspace2, workspace1, (INT) otf->c3_exp);
  rgrid3d_product(workspace2, workspace2, density);
  rgrid3d_add_scaled(energy_density, otf->c3 / 3.0, workspace2);

  /* Barranco's contribution (high density) */
  if((otf->model & DFT_OT_HD) || (otf->model & DFT_OT_HD2)) {
    XXX_beta = otf->beta;
    XXX_rhom = otf->rhom;
    XXX_C = otf->C;
    rgrid3d_operate_one(workspace1, density, dft_ot3d_barranco_energy_op);
    rgrid3d_sum(energy_density, energy_density, workspace1);
  }

  /* Ideal gas contribution (thermal) */
  if(otf->model >= DFT_OT_T400MK && otf->model < DFT_GP) { /* do not add this for DR */
    dft_common_idealgas_params(otf->temp, otf->mass, otf->c4);
    rgrid3d_operate_one(workspace1, density, dft_common_bose_idealgas_energy);
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

  if(otf->model & DFT_OT_BACKFLOW) {
    if((otf->model & DFT_OT_HD) || (otf->model & DFT_OT_HD2)) {
      /* Modified BF (Marti & Manuel) - screening of density. */
      XXX_xi = 1E4 * (GRID_AUTOANG * GRID_AUTOANG * GRID_AUTOANG);
      XXX_rhobf = 0.033 / (GRID_AUTOANG * GRID_AUTOANG * GRID_AUTOANG);
      rgrid3d_operate_one(workspace7, density, dft_ot3d_bf_pi_energy_op);
    } else {
      /* Original BF */
      rgrid3d_copy(workspace7, density);
    }
    // workspace7 = density from this on
    
    // grid3d_wf_momentum(wf, workspace1, workspace2, workspace3, workspace4);   /* this would imply FFT boundaries */
    grid3d_wf_probability_flux(wf, workspace1, workspace2, workspace3);    /* finite difference */
    rgrid3d_division_eps(workspace1, workspace1, workspace7, DFT_BF_EPS);  /* velocity = flux / rho, v_x */
    rgrid3d_division_eps(workspace2, workspace2, workspace7, DFT_BF_EPS);  /* v_y */
    rgrid3d_division_eps(workspace3, workspace3, workspace7, DFT_BF_EPS);  /* v_z */
    rgrid3d_product(workspace4, workspace1, workspace1);   /* v_x^2 */
    rgrid3d_product(workspace5, workspace2, workspace2);   /* v_y^2 */
    rgrid3d_sum(workspace4, workspace4, workspace5);
    rgrid3d_product(workspace5, workspace3, workspace3);   /* v_z^2 */
    rgrid3d_sum(workspace4, workspace4, workspace5);       /* wrk4 = v_x^2 + v_y^2 + v_z^2 */

    /* Term 1: -(M/4) * rho(r) * v(r)^2 \int U_j(|r - r'|) * rho(r') d3r' */
    rgrid3d_copy(workspace5, workspace7);
    rgrid3d_fft(workspace5);                        /* This was done before - TODO: save previous rho FFT and reuse here */
    rgrid3d_fft_convolute(workspace6, otf->backflow_pot, workspace5);
    rgrid3d_inverse_fft(workspace6);
    rgrid3d_product(workspace6, workspace6, workspace4);
    rgrid3d_product(workspace6, workspace6, workspace7);
    rgrid3d_add_scaled(energy_density, -otf->mass / 4.0, workspace6);
    /* Term 2: +(M/2) * rho(r) v(r) . \int U_j(|r - r'|) * rho(r') v(r') d3r' */
    /* x contribution */
    rgrid3d_product(workspace5, workspace7, workspace1);   /* rho(r') * v_x(r') */
    rgrid3d_fft(workspace5);
    rgrid3d_fft_convolute(workspace6, otf->backflow_pot, workspace5);
    rgrid3d_inverse_fft(workspace6);
    rgrid3d_product(workspace6, workspace6, workspace7);
    rgrid3d_product(workspace6, workspace6, workspace1);
    rgrid3d_add_scaled(energy_density, otf->mass / 2.0, workspace6);
    /* y contribution */
    rgrid3d_product(workspace5, workspace7, workspace2);   /* rho(r') * v_y(r') */
    rgrid3d_fft(workspace5);
    rgrid3d_fft_convolute(workspace6, otf->backflow_pot, workspace5);
    rgrid3d_inverse_fft(workspace6);
    rgrid3d_product(workspace6, workspace6, workspace7);
    rgrid3d_product(workspace6, workspace6, workspace2);
    rgrid3d_add_scaled(energy_density, otf->mass / 2.0, workspace6);
    /* z contribution */
    rgrid3d_product(workspace5, workspace7, workspace3);   /* rho(r') * v_z(r') */
    rgrid3d_fft(workspace5);
    rgrid3d_fft_convolute(workspace6, otf->backflow_pot, workspace5);
    rgrid3d_inverse_fft(workspace6);
    rgrid3d_product(workspace6, workspace6, workspace7);
    rgrid3d_product(workspace6, workspace6, workspace3);
    rgrid3d_add_scaled(energy_density, otf->mass / 2.0, workspace6);
    /* Term 3: -(M/4) rho(r) \int U_j(|r - r'|) rho(r') v^2(r') d3r' */
    rgrid3d_product(workspace5, workspace7, workspace4);
    rgrid3d_fft(workspace5);
    rgrid3d_fft_convolute(workspace6, otf->backflow_pot, workspace5);
    rgrid3d_inverse_fft(workspace6);
    rgrid3d_product(workspace6, workspace6, workspace7);
    rgrid3d_add_scaled(energy_density, -otf->mass / 4.0, workspace6);
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

EXPORT void dft_ot3d_backflow_potential(dft_ot_functional *otf, cgrid3d *potential, rgrid3d *density, rgrid3d *veloc_x, rgrid3d *veloc_y, rgrid3d *veloc_z, rgrid3d *workspace1, rgrid3d *workspace2, rgrid3d *workspace3, rgrid3d *workspace4, rgrid3d *workspace5, rgrid3d *workspace6) {

  /* Calculate A (workspace1) [scalar] */
  if((otf->model & DFT_OT_HD) || (otf->model & DFT_OT_HD2)) {
    XXX_xi = 1E4 * (GRID_AUTOANG * GRID_AUTOANG * GRID_AUTOANG);
    XXX_rhobf = 0.033 / (GRID_AUTOANG * GRID_AUTOANG * GRID_AUTOANG);
    rgrid3d_operate_one(workspace1, density, dft_ot3d_bf_pi_energy_op); /* g rho */
  } else {
    /* Original BF code (without the MM density cutoff) */
    rgrid3d_copy(workspace1, density);   /* just rho */
  }
  rgrid3d_fft(workspace1);
  rgrid3d_fft_convolute(workspace1, workspace1, otf->backflow_pot);
  rgrid3d_inverse_fft(workspace1);

  /* Calculate C (workspace2) [scalar] */
  rgrid3d_product(workspace2, veloc_x, veloc_x);
  rgrid3d_add_scaled_product(workspace2, 1.0, veloc_y, veloc_y);
  rgrid3d_add_scaled_product(workspace2, 1.0, veloc_z, veloc_z);
  if((otf->model & DFT_OT_HD) || (otf->model & DFT_OT_HD2)) { 
    rgrid3d_operate_one_product(workspace2, workspace2, density, dft_ot3d_bf_pi_energy_op); /* g rho */
  } else {
    rgrid3d_product(workspace2, workspace2, density);
  }
  rgrid3d_fft(workspace2);
  rgrid3d_fft_convolute(workspace2, workspace2, otf->backflow_pot);
  rgrid3d_inverse_fft(workspace2);

  /* Calculate B (workspace3 (x), workspace4 (y), workspace5 (z)) [vector] */
  /* B_X */
  if((otf->model & DFT_OT_HD) || (otf->model & DFT_OT_HD2)) { 
    rgrid3d_operate_one_product(workspace3, veloc_x, density, dft_ot3d_bf_pi_energy_op); /* g rho */
  } else {
    rgrid3d_product(workspace3, veloc_x, density);
  }
  rgrid3d_fft(workspace3);
  rgrid3d_fft_convolute(workspace3, workspace3, otf->backflow_pot);
  rgrid3d_inverse_fft(workspace3);

  /* B_Y */
  if((otf->model & DFT_OT_HD) || (otf->model & DFT_OT_HD2)) { 
    rgrid3d_operate_one_product(workspace4, veloc_y, density, dft_ot3d_bf_pi_energy_op); /* g rho */
  } else {
    rgrid3d_product(workspace4, veloc_y, density);
  }
  rgrid3d_fft(workspace4);
  rgrid3d_fft_convolute(workspace4, workspace4, otf->backflow_pot);
  rgrid3d_inverse_fft(workspace4);

  /* B_Z */
  if((otf->model & DFT_OT_HD) || (otf->model & DFT_OT_HD2)) { 
    rgrid3d_operate_one_product(workspace5, veloc_z, density, dft_ot3d_bf_pi_energy_op); /* g rho */
  } else {
    rgrid3d_product(workspace5, veloc_z, density);
  }
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
  if((otf->model & DFT_OT_HD) || (otf->model & DFT_OT_HD2)) { 
    /* multiply by [rho x (dG/drho)(rho) + G(rho)] (dft_ot3d_bf_pi_op) */
    rgrid3d_operate_one_product(workspace6, workspace6, density, dft_ot3d_bf_pi_op);
  }
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
  if((otf->model & DFT_OT_HD) || (otf->model & DFT_OT_HD2)) { 
    rgrid3d_operate_one(workspace1, density, dft_ot3d_bf_pi_energy_op);
    rgrid3d_fd_gradient_x(workspace1, workspace2);  
  } else {
    rgrid3d_fd_gradient_x(density, workspace2);
  }
  rgrid3d_division_eps(workspace2, workspace2, density, DFT_BF_EPS);
  rgrid3d_add_scaled_product(workspace6, 0.5, workspace2, veloc_x);

  /* 1.2 (1/2) (drho/dy)/rho * (v_yA - B_y) */
  if((otf->model & DFT_OT_HD) || (otf->model & DFT_OT_HD2)) { 
    /* rgrid3d_operate_one(workspace1, density, dft_ot3d_bf_pi_energy_op); */ /* already done above */
    rgrid3d_fd_gradient_y(workspace1, workspace2);  
  } else {
    rgrid3d_fd_gradient_y(density, workspace2);
  }
  rgrid3d_division_eps(workspace2, workspace2, density, DFT_BF_EPS);
  rgrid3d_add_scaled_product(workspace6, 0.5, workspace2, veloc_y);

  /* 1.3 (1/2) (drho/dz)/rho * (v_zA - B_z) */
  if((otf->model & DFT_OT_HD) || (otf->model & DFT_OT_HD2)) { 
    /* rgrid3d_operate_one(workspace1, density, dft_ot3d_bf_pi_energy_op); */ /* already done above */
    rgrid3d_fd_gradient_z(workspace1, workspace2);  
  } else {
    rgrid3d_fd_gradient_z(density, workspace2);
  }
  rgrid3d_division_eps(workspace2, workspace2, density, DFT_BF_EPS);
  rgrid3d_add_scaled_product(workspace6, 0.5, workspace2, veloc_z);

  /* 2.1 (1/2) (d/dx) (v_xA - B_x) */
  rgrid3d_fd_gradient_x(veloc_x, workspace2);
  if((otf->model & DFT_OT_HD) || (otf->model & DFT_OT_HD2)) { 
    rgrid3d_operate_one_product(workspace2, workspace2, density, dft_ot3d_bf_pi_energy_op2); /* multiply by g */
  }
  rgrid3d_add_scaled(workspace6, 0.5, workspace2);

  /* 2.2 (1/2) (d/dy) (v_yA - B_y) */
  rgrid3d_fd_gradient_y(veloc_y, workspace2);
  if((otf->model & DFT_OT_HD) || (otf->model & DFT_OT_HD2)) { 
    rgrid3d_operate_one_product(workspace2, workspace2, density, dft_ot3d_bf_pi_energy_op2);  /* multiply by g */
  }
  rgrid3d_add_scaled(workspace6, 0.5, workspace2);

  /* 2.3 (1/2) (d/dz) (v_zA - B_z) */
  rgrid3d_fd_gradient_z(veloc_z, workspace2);
  if((otf->model & DFT_OT_HD) || (otf->model & DFT_OT_HD2)) { 
    rgrid3d_operate_one_product(workspace2, workspace2, density, dft_ot3d_bf_pi_energy_op2);  /* multiply by g */
  }
  rgrid3d_add_scaled(workspace6, 0.5, workspace2);

  grid3d_add_real_to_complex_im(potential, workspace6);
}

EXPORT inline void dft_ot_temperature(dft_ot_functional *otf, INT model) {

  fprintf(stderr, "libdft: Model = " FMT_I "\n", model);

  if((otf->model & DFT_OT_HD) || (otf->model & DFT_OT_HD2)) { /* high density penalty */
    otf->beta = (40.0 / (GRID_AUTOANG * GRID_AUTOANG * GRID_AUTOANG));
    otf->rhom = (0.37 * GRID_AUTOANG * GRID_AUTOANG * GRID_AUTOANG);
    otf->C = 0.1; /* a.u. */
  } else {
    otf->beta = 0.0;
    otf->rhom = 0.0;
    otf->C = 0.0;
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

  if((model & DFT_GP) || (model & DFT_ZERO)) {
    otf->temp = 0.0;
    otf->rho0 = 0.0218360;
    otf->mu0 = 7.0 / GRID_AUTOK;  // DEBUG: sign
    otf->c2 = otf->c2_exp = otf->c3 = otf->c3_exp = 0.0;
    /* most of the parameters are unused */
  }

  fprintf(stderr,"libdft: Temperature = " FMT_R " K.\n", otf->temp); 

  otf->b /= GRID_AUTOK * (GRID_AUTOANG * GRID_AUTOANG * GRID_AUTOANG);
  otf->c2 /= GRID_AUTOK * POW(GRID_AUTOANG, 3.0 * otf->c2_exp);
  otf->c3 /= GRID_AUTOK * POW(GRID_AUTOANG, 3.0 * otf->c3_exp);
  otf->rho0 *= GRID_AUTOANG * GRID_AUTOANG * GRID_AUTOANG;
  otf->lj_params.h /= GRID_AUTOANG;
  otf->lj_params.sigma   = 2.556 / GRID_AUTOANG;
  otf->lj_params.epsilon = 10.22 / GRID_AUTOK;
  otf->lj_params.cval = 0.0;
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

  fprintf(stderr, "libdft: C2 = " FMT_R " K Angs^" FMT_R "\n",
	  otf->c2 * GRID_AUTOK * ipow(GRID_AUTOANG, (INT) (3.0 * otf->c2_exp)),
	  3.0 * otf->c2_exp);
  
  fprintf(stderr, "libdft: C3 = " FMT_R " K Angs^" FMT_R "\n", 
	  otf->c3 * GRID_AUTOK * ipow(GRID_AUTOANG, (INT) (3.0 * otf->c3_exp)),
	  3.0 * otf->c3_exp);
  
  otf->model = model;
}
