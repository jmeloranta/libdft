/*
 * Simple driver routines to propagate the liquid (3D).
 *
 * TODO: Add comments to show which internal workspaces are used by 
 * each function.
 *
 * TODO: DFT_DRIVER_BC_NEUMANN generates problems with DFT_DRIVER_KINETIC_CN_NBC_ROT .
 *
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <grid/grid.h>
#include <grid/au.h>
#include "dft.h"
#include "ot.h"

/*
 * Parameters for evaluating the potential grid (no need to play with these).
 *
 */

#define MIN_SUBSTEPS 4
#define MAX_SUBSTEPS 32

/* End of tunable parameters */

/* Global user accessible variables */
char dft_driver_verbose = 1;   /* set to zero to eliminate informative print outs */
dft_ot_functional *dft_driver_otf = 0;
char dft_driver_init_wavefunction = 1;
char dft_driver_kinetic = 0; /* default FFT propagation for kinetic, TODO: FFT gives some numerical hash - bug? */
REAL complex (*dft_driver_bc_function)(void *, REAL complex, INT, INT, INT) = NULL; /* User specified function for absorbing boundaries */

static INT driver_nx = 0, driver_ny = 0, driver_nz = 0, driver_nx2 = 0, driver_ny2 = 0, driver_nz2 = 0, driver_threads = 0, driver_dft_model = 0, driver_iter_mode = 0, driver_boundary_type = 0;
static INT driver_norm_type = 0, driver_nhe = 0, center_release = 0, driver_rels = 0;
static char driver_bc = 0;
static REAL driver_frad = 0.0, driver_omega = 0.0, driver_damp = 0.2, driver_width_x = 1.0, driver_width_y = 1.0, driver_width_z = 1.0;
static REAL viscosity = 0.0, viscosity_alpha = 1.0;
static REAL driver_step = 0.0, driver_rho0 = 0.0;
static REAL driver_x0 = 0.0, driver_y0 = 0.0, driver_z0 = 0.0, driver_bx = 0.0, driver_by = 0.0, driver_bz = 0.0;
static REAL driver_kx0 = 0.0, driver_ky0 = 0.0,driver_kz0 = 0.0;
static rgrid3d *density = 0, *workspace1 = 0, *workspace2 = 0, *workspace3 = 0, *workspace4 = 0, *workspace5 = 0, *workspace6 = 0;
static rgrid3d *workspace7 = 0, *workspace8 = 0, *workspace9 = 0;
static cgrid3d *cworkspace = 0;
static grid_timer timer;

/*
 * Return default wisdom file name.
 *
 */

static char *dft_driver_wisfile() {

  char hn[128];
  static char *buf = NULL;

  if(buf == NULL && !(buf = (char *) malloc(128))) {
    fprintf(stderr, "libdft: memory allocation failure (wisfile).\n");
    exit(1);
  }
  gethostname(hn, sizeof(hn));
  sprintf(buf, "fftw-%s.wis", hn);  
  fprintf(stderr, "libdft: Wisdom file = %s.\n", buf);
  return buf;
}

/*
 * Wave function normalization (for imaginary time).
 *
 */

int dft_driver_temp_disable_other_normalization = 0;

inline static void scale_wf(char what, wf3d *gwf) {

  INT i, j, k;
  REAL x, y, z;
  REAL complex norm;

  if(what >= DFT_DRIVER_PROPAGATE_OTHER) { /* impurity */
    if(!dft_driver_temp_disable_other_normalization) grid3d_wf_normalize(gwf);
    return;
  }
  
  /* liquid helium */
  switch(driver_norm_type) {
  case DFT_DRIVER_NORMALIZE_BULK: /*bulk normalization */
    norm = SQRT(driver_rho0) / CABS(cgrid3d_value_at_index(gwf->grid, 0, 0, 0));
    cgrid3d_multiply(gwf->grid, norm);
    break;
  case DFT_DRIVER_NORMALIZE_ZEROB:
    i = driver_nx / driver_nhe;
    j = driver_ny / driver_nhe;
    k = driver_nz / driver_nhe;
    norm = SQRT(driver_rho0) / CABS(cgrid3d_value_at_index(gwf->grid, i, j, k));
    cgrid3d_multiply(gwf->grid, norm);
    break;
  case DFT_DRIVER_NORMALIZE_DROPLET: /* helium droplet */
    if(!center_release) {
      REAL sq;
      sq = SQRT(3.0*driver_rho0/4.0);
      for (i = 0; i < driver_nx; i++) {
	x = ((REAL) (i - driver_nx2)) * driver_step;
	for (j = 0; j < driver_ny; j++) {
	  y = ((REAL) (j - driver_ny2)) * driver_step;
	  for (k = 0; k < driver_nz; k++) {
	    z = ((REAL) (k - driver_nz2)) * driver_step;
	    if(SQRT(x*x + y*y + z*z) < driver_frad && CABS(cgrid3d_value_at_index(gwf->grid, i, j, k)) < sq)
              cgrid3d_value_to_index(gwf->grid, i, j, k, sq);
	  }
	}
      }
    }
    grid3d_wf_normalize(gwf);
    cgrid3d_multiply(gwf->grid, SQRT((REAL) driver_nhe));
    break;
  case DFT_DRIVER_NORMALIZE_COLUMN: /* column along y */
    if(!center_release) {
      REAL sq;
      sq = SQRT(3.0*driver_rho0/4.0);
      for (i = 0; i < driver_nx; i++) {
	x = ((REAL) (i - driver_nx2)) * driver_step;
	for (j = 0; j < driver_ny; j++) {
	  y = ((REAL) (j - driver_ny2)) * driver_step;
	  for (k = 0; k < driver_nz; k++) {
	    z = ((REAL) (k - driver_nz2)) * driver_step;
	    if(SQRT(x * x + z * z) < driver_frad && CABS(cgrid3d_value_at_index(gwf->grid, i, j, k)) < sq)
              cgrid3d_value_to_index(gwf->grid, i, j, k, sq);
	  }
	}
      }
    }
    grid3d_wf_normalize(gwf);
    cgrid3d_multiply(gwf->grid, SQRT((REAL) driver_nhe));
    break;
  case DFT_DRIVER_NORMALIZE_SURFACE:   /* in (x,y) plane starting at z = 0 */
    if(!center_release) {
      for (i = 0; i < driver_nx; i++)
	for (j = 0; j < driver_ny; j++)
	  for (k = 0; k < driver_nz; k++) {
	    z = ((REAL) (k - driver_nz2)) * driver_step;
	    if(FABS(z) < driver_frad)
              cgrid3d_value_to_index(gwf->grid, i, j, k, 0.0);
	  }
    }
    grid3d_wf_normalize(gwf);
    cgrid3d_multiply(gwf->grid, SQRT((REAL) driver_nhe));
    break;
  case DFT_DRIVER_NORMALIZE_N:
    grid3d_wf_normalize(gwf);
    cgrid3d_multiply(gwf->grid, SQRT((REAL) driver_nhe));
    break;    
  case DFT_DRIVER_DONT_NORMALIZE:
    break;
  default:
    fprintf(stderr, "libdft: Unknown normalization method.\n");
    exit(1);
  }
}

/*
 * FFTW Wisdom interface - import wisdom.
 *
 * file = file name for reading wisdom data (input; char *).
 *
 */

EXPORT void dft_driver_read_wisdom(char *file) {

  /* Attempt to use wisdom (FFTW) from previous runs */
#ifdef SINGLE_PREC
  if(fftwf_import_wisdom_from_filename(file) == 1) {
#else
  if(fftw_import_wisdom_from_filename(file) == 1) {
#endif
    if(dft_driver_verbose) fprintf(stderr, "libdft: Using wisdom stored in %s.\n", file);
  } else {
    if(dft_driver_verbose) fprintf(stderr, "libdft: No existing wisdom file.\n");
  }
}

/*
 * FFTW Wisdom interface - export wisdom.
 *
 * file = file name for saving wisdom data (input, char *).
 *
 */

EXPORT void dft_driver_write_wisdom(char *file) {

#ifdef SINGLE_PREC
  fftwf_export_wisdom_to_filename(file);
#else
  fftw_export_wisdom_to_filename(file);
#endif
}

/*
 * Initialize dft_driver routines. This must always be called after the
 * parameters have been set.
 *
 * No return value.
 *
 */

EXPORT void dft_driver_initialize() {

  if(driver_nx == 0) {
    fprintf(stderr, "libdft: dft_driver not properly initialized.\n");
    exit(1);
  }

  if(workspace1) rgrid3d_free(workspace1);
  if(workspace2) rgrid3d_free(workspace2);
  if(workspace3) rgrid3d_free(workspace3);
  if(workspace4) rgrid3d_free(workspace4);
  if(workspace5) rgrid3d_free(workspace5);
  if(workspace6) rgrid3d_free(workspace6);
  if(workspace7) rgrid3d_free(workspace7);
  if(workspace8) rgrid3d_free(workspace8);
  if(workspace9) rgrid3d_free(workspace9);
  if(density) rgrid3d_free(density);
  if(dft_driver_otf) dft_ot3d_free(dft_driver_otf);

  grid_timer_start(&timer);
  grid_threads_init(driver_threads);
  dft_driver_read_wisdom(dft_driver_wisfile());
  workspace1 = dft_driver_alloc_rgrid("DR workspace1");
  workspace2 = dft_driver_alloc_rgrid("DR workspace2");
  workspace3 = dft_driver_alloc_rgrid("DR workspace3");
  workspace4 = dft_driver_alloc_rgrid("DR workspace4");
  workspace5 = dft_driver_alloc_rgrid("DR workspace5");
  workspace6 = dft_driver_alloc_rgrid("DR workspace6");
  workspace7 = dft_driver_alloc_rgrid("DR workspace7");
  if(driver_dft_model & DFT_OT_BACKFLOW) {
    workspace8 = dft_driver_alloc_rgrid("DR workspace8");
    workspace9 = dft_driver_alloc_rgrid("DR workspace9");
  }
  density = dft_driver_alloc_rgrid("DR density");
  dft_driver_otf = dft_ot3d_alloc(driver_dft_model, driver_nx, driver_ny, driver_nz, driver_step, driver_bc, MIN_SUBSTEPS, MAX_SUBSTEPS);
  if(driver_rho0 == 0.0) {
    if(dft_driver_verbose) fprintf(stderr, "libdft: Setting driver_rho0 to " FMT_R "\n", dft_driver_otf->rho0);
    driver_rho0 = dft_driver_otf->rho0;
  } else {
    if(dft_driver_verbose) fprintf(stderr, "libdft: Overwritting dft_driver_otf->rho0 to " FMT_R ".\n", driver_rho0);
    dft_driver_otf->rho0 = driver_rho0;
  }
  if(dft_driver_verbose) fprintf(stderr, "libdft: rho0 = " FMT_R " Angs^-3.\n", driver_rho0 / (GRID_AUTOANG * GRID_AUTOANG * GRID_AUTOANG));
  if(dft_driver_verbose) fprintf(stderr, "libdft: " FMT_R " wall clock seconds for initialization.\n", grid_timer_wall_clock_time(&timer));
}

/*
 * Set up the DFT calculation grid.
 *
 * nx      = number of grid points along x (INT).
 * ny      = number of grid points along y (INT).
 * nz      = number of grid points along z (INT).
 * threads = number of parallel execution threads (INT).
 *
 * No return value.
 *
 */

EXPORT void dft_driver_setup_grid(INT nx, INT ny, INT nz, REAL step, INT threads) {
  
  // TODO: fixme
  //  if((nx % 2) || (ny % 2) || (nz % 2)) {
  //    fprintf(stderr, "libdft: Currently works only with array sizes of multiples of two.\n");
  //    exit(1);
  //  }

  driver_nx = nx; driver_nx2 = driver_nx / 2;
  driver_ny = ny; driver_ny2 = driver_ny / 2;
  driver_nz = nz; driver_nz2 = driver_nz / 2;
  driver_step = step;
  // Set the origin to its default value if it is not defined
  if(dft_driver_verbose) fprintf(stderr, "libdft: Grid size = (" FMT_I "," FMT_I "," FMT_I ") with step = " FMT_R ".\n", nx, ny, nz, step);
  driver_threads = threads;
}

/*
 * Set up grid origin.
 * Can be overwritten for a particular grid calling (r/c)grid3d_set_origin
 *
 * x0 = X coordinate for the new origin (input, REAL).
 * y0 = Y coordinate for the new origin (input, REAL).
 * z0 = Z coordinate for the new origin (input, REAL).
 *
 */

EXPORT void dft_driver_setup_origin(REAL x0, REAL y0, REAL z0) {

  driver_x0 = x0;
  driver_y0 = y0;
  driver_z0 = z0;
  if(dft_driver_verbose) fprintf(stderr, "libdft: Origin of coordinates set at (" FMT_R "," FMT_R "," FMT_R ")\n", x0, y0, z0);
}

/*
 * Set up grid momentum frame of reference, i.e. a background velocity.
 * Can be overwritten for a particular grid calling (r/c)grid3d_set_momentum
 *
 * kx0 = kx for new momentum frame (input, REAL).
 * ky0 = ky for new momentum frame (input, REAL).
 * kz0 = kz for new momentum frame (input, REAL).
 *
 */

EXPORT void dft_driver_setup_momentum(REAL kx0, REAL ky0, REAL kz0) {

  driver_kx0 = kx0;
  driver_ky0 = ky0;
  driver_kz0 = kz0;
  if(dft_driver_verbose) fprintf(stderr, "libdft: Frame of reference momentum = (" FMT_R "," FMT_R "," FMT_R ")\n", kx0, ky0, kz0);
}

/*
 * Set effective visocisty.
 *
 * visc  = Viscosity of bulk liquid in Pa s (SI) units. This is typically the normal fraction x normal fluid viscosity.
 *         (default value 0.0; input, REAL)
 * alpha = exponent for visc * (rho/rho0)^alpha  for calculating the interfacial viscosity (input, REAL).
 *
 */

EXPORT void dft_driver_setup_viscosity(REAL visc, REAL alpha) {

  viscosity = (visc / GRID_AUTOPAS);
  viscosity_alpha = alpha;
  if(dft_driver_verbose) fprintf(stderr, "libdft: Effective viscosity set to " FMT_R " a.u, alpha = " FMT_R ".\n", visc / GRID_AUTOPAS, alpha);
}

/*
 * Set up the DFT calculation model.
 *
 * dft_model = specify the DFT Hamiltonian to use (see ot.h) (input, INT).
 * iter_mode = iteration mode: 1 = imaginary time, 0 = real time (input, INT).
 * rho0      = equilibrium density for the liquid (in a.u.; input, REAL).
 *             if 0.0, the equilibrium density will be used
 *             when dft_driver_initialize is called.
 *
 * No return value.
 *
 */

static int bh = 0;

EXPORT void dft_driver_setup_model(INT dft_model, INT iter_mode, REAL rho0) {

  driver_dft_model = dft_model;
  driver_iter_mode = iter_mode;
  if(dft_driver_verbose) fprintf(stderr, "libdft: %s time calculation.\n", iter_mode?"imaginary":"real");
  if(bh && dft_driver_verbose) fprintf(stderr,"libdft: WARNING -- Overwritting driver_rho0 to " FMT_R "\n", rho0);
  driver_rho0 = rho0;
  bh = 1;
}

/*
 * Define boundary type.
 *
 * type    = Boundary type: 0 = regular, 1 = absorbing (imag time) 
 *           (input, INT).
 * damp    = Daping constant (input, REAL). Usually between 0.1 and 1.0. Only when type = 1.
 * width_x = Width of the absorbing region along x. Only when type = 1.
 * width_y = Width of the absorbing region along y. Only when type = 1.
 * width_z = Width of the absorbing region along z. Only when type = 1.
 * 
 * NOTE: For the absorbing BC to work, one MUST use DFT_DRIVER_DONT_NORMALIZE and include the chemical potential!
 *       (in both imaginary & real time propagation). Otherwise, you will find issues at the boundary (due to the 
 *       constraint no longer present in RT simulations).
 *
 * No return value.
 *
 */

EXPORT void dft_driver_setup_boundary_type(INT boundary_type, REAL damp, REAL width_x, REAL width_y, REAL width_z) {

  driver_boundary_type = boundary_type;
  if(driver_boundary_type == DFT_DRIVER_BOUNDARY_ITIME) {
    if(dft_driver_verbose) fprintf(stderr, "libdft: ITIME absorbing boundary.\n");
    if(dft_driver_kinetic == DFT_DRIVER_KINETIC_CN_NBC_ROT) {
      fprintf(stderr, "libdft: Absorbing boundaries with rotating liquid (not implemented).\n");
      exit(1);
    }
  }
  driver_damp = damp;
  driver_width_x = width_x;
  driver_width_y = width_y;
  driver_width_z = width_z;
  driver_bx = driver_step * (REAL) driver_nx2 - driver_width_x;
  driver_by = driver_step * (REAL) driver_ny2 - driver_width_y;
  driver_bz = driver_step * (REAL) driver_nz2 - driver_width_z;
}

/*
 * Impose normal or vortex compatible boundaries.
 *
 * bc = Boundary type:
 *           Normal (DFT_DRIVER_BC_NORMAL), Vortex along X (DFT_DRIVER_BC_X), Vortex along Y (DFT_DRIVER_BC_Y), 
 *           Vortex along Z (DFT_DRIVER_BC_Z), Neumann (DFT_DRIVER_BC_NEUMANN) (input, int).
 *
 */

EXPORT void dft_driver_setup_boundary_condition(char bc) {

  driver_bc = bc;
}

/*
 * Set up normalization method for imaginary time propagation.
 *
 * type = how to renormalize the wavefunction: DFT_DRIVER_NORMALIZE_BULK = bulk; DFT_DRIVER_NORMALIZE_DROPLET = droplet
 *        placed at the origin; DFT_DRIVER_NORMALIZE_COLUMN = column placed at x = 0 (input, INT); DFT_DRIVER_DONT_NORMALIZE
 *        = no normalization (must use the correct chemical potential).
 * nhe  = desired # of He atoms for types 1 & 2 above (input, INT).
 * frad = fixed volume radius (REAL). Liquid within this radius
 *        will be fixed to rho0 to converge to droplet or column (input REAL).
 * rels = iteration after which the fixing condition will be released (input INT).
 *        This should be done for the last few iterations to avoid
 *        artifacts arising from the fixing constraint. Set to zero to disable.
 * 
 */

EXPORT void dft_driver_setup_normalization(INT norm_type, INT nhe, REAL frad, INT rels) {

  driver_norm_type = norm_type;
  driver_nhe = nhe;
  driver_rels = rels;
  driver_frad = frad;
}

/*
 * Modify the value of the angular velocity omega (rotating liquid).
 *
 * omega = angular velocity (input, REAL);
 *
z */

EXPORT void dft_driver_setup_rotation_omega(REAL omega) {

  driver_omega = omega;
  dft_driver_kinetic = DFT_DRIVER_KINETIC_CN_NBC_ROT;
  if(dft_driver_verbose) fprintf(stderr, "libdft: Using CN for kinetic energy propagation. Set BC to Neumann to also evaluate kinetic energy using CN.\n");
}

/*
 * Impose imaginary time within the absorbing boundary region.
 *
 * Called back from grid3d_wf_propagate_kinetic_cn_nbc2() and grid3d_wf_propagate_potential() routines.
 *
 */

struct priv_data {
  INT nx2, ny2, nz2;
  REAL step;
  REAL width_x, width_y, width_z;
  REAL bx, by, bz;
  REAL damp;
};

/*
 * For some reason tanh() is really slow in glibc. We need just approx tanh() -> use lookup table.
 *
 */

static REAL fast_tanh_table[256] = { -0.96402758, -0.96290241, -0.96174273, -0.96054753, -0.95931576,
                           -0.95804636, -0.95673822, -0.95539023, -0.95400122, -0.95257001,
                           -0.95109539, -0.9495761 , -0.94801087, -0.94639839, -0.94473732,
                           -0.94302627, -0.94126385, -0.93944862, -0.93757908, -0.93565374,
                           -0.93367104, -0.93162941, -0.92952723, -0.92736284, -0.92513456,
                           -0.92284066, -0.92047938, -0.91804891, -0.91554743, -0.91297305,
                           -0.91032388, -0.90759795, -0.9047933 , -0.90190789, -0.89893968,
                           -0.89588656, -0.89274642, -0.88951709, -0.88619637, -0.88278203,
                           -0.87927182, -0.87566342, -0.87195453, -0.86814278, -0.86422579,
                           -0.86020115, -0.85606642, -0.85181914, -0.84745683, -0.84297699,
                           -0.83837709, -0.83365461, -0.82880699, -0.82383167, -0.81872609,
                           -0.81348767, -0.80811385, -0.80260204, -0.7969497 , -0.79115425,
                           -0.78521317, -0.77912392, -0.772884  , -0.76649093, -0.75994227,
                           -0.75323562, -0.74636859, -0.73933889, -0.73214422, -0.7247824 ,
                           -0.71725127, -0.70954876, -0.70167287, -0.6936217 , -0.68539341,
                           -0.67698629, -0.66839871, -0.65962916, -0.65067625, -0.64153871,
                           -0.6322154 , -0.62270534, -0.61300768, -0.60312171, -0.59304692,
                           -0.58278295, -0.57232959, -0.56168685, -0.55085493, -0.53983419,
                           -0.52862523, -0.51722883, -0.50564601, -0.49387799, -0.48192623,
                           -0.46979241, -0.45747844, -0.44498647, -0.4323189 , -0.41947836,
                           -0.40646773, -0.39329014, -0.37994896, -0.36644782, -0.35279057,
                           -0.33898135, -0.32502449, -0.31092459, -0.2966865 , -0.28231527,
                           -0.26781621, -0.25319481, -0.23845682, -0.22360817, -0.208655  ,
                           -0.19360362, -0.17846056, -0.16323249, -0.14792623, -0.13254879,
                           -0.11710727, -0.10160892, -0.08606109, -0.07047123, -0.05484686,
                           -0.0391956 , -0.02352507, -0.00784298,  0.00784298,  0.02352507,
                           0.0391956 ,  0.05484686,  0.07047123,  0.08606109,  0.10160892,
                           0.11710727,  0.13254879,  0.14792623,  0.16323249,  0.17846056,
                           0.19360362,  0.208655  ,  0.22360817,  0.23845682,  0.25319481,
                           0.26781621,  0.28231527,  0.2966865 ,  0.31092459,  0.32502449,
                           0.33898135,  0.35279057,  0.36644782,  0.37994896,  0.39329014,
                           0.40646773,  0.41947836,  0.4323189 ,  0.44498647,  0.45747844,
                           0.46979241,  0.48192623,  0.49387799,  0.50564601,  0.51722883,
                           0.52862523,  0.53983419,  0.55085493,  0.56168685,  0.57232959,
                           0.58278295,  0.59304692,  0.60312171,  0.61300768,  0.62270534,
                           0.6322154 ,  0.64153871,  0.65067625,  0.65962916,  0.66839871,
                           0.67698629,  0.68539341,  0.6936217 ,  0.70167287,  0.70954876,
                           0.71725127,  0.7247824 ,  0.73214422,  0.73933889,  0.74636859,
                           0.75323562,  0.75994227,  0.76649093,  0.772884  ,  0.77912392,
                           0.78521317,  0.79115425,  0.7969497 ,  0.80260204,  0.80811385,
                           0.81348767,  0.81872609,  0.82383167,  0.82880699,  0.83365461,
                           0.83837709,  0.84297699,  0.84745683,  0.85181914,  0.85606642,
                           0.86020115,  0.86422579,  0.86814278,  0.87195453,  0.87566342,
                           0.87927182,  0.88278203,  0.88619637,  0.88951709,  0.89274642,
                           0.89588656,  0.89893968,  0.90190789,  0.9047933 ,  0.90759795,
                           0.91032388,  0.91297305,  0.91554743,  0.91804891,  0.92047938,
                           0.92284066,  0.92513456,  0.92736284,  0.92952723,  0.93162941,
                           0.93367104,  0.93565374,  0.93757908,  0.93944862,  0.94126385,
                           0.94302627,  0.94473732,  0.94639839,  0.94801087,  0.9495761 ,
                           0.95109539,  0.95257001,  0.95400122,  0.95539023,  0.95673822,
                           0.95804636,  0.95931576,  0.96054753,  0.96174273,  0.96290241,
                           0.96402758 };

static inline REAL fast_tanh(REAL x) {

  if(x > 2.0) return 1.0;
  else if(x <= -2.0) return -1.0;
  else {
    INT index = 128 + (INT) (64.0 * x);
    return fast_tanh_table[index];
  }
}

REAL complex dft_driver_itime_abs(void *data, REAL complex tstep, INT i, INT j, INT k) {

  REAL x, y, z, tmp;
  struct priv_data *asd = (struct priv_data *) data;  
  INT nx2 = asd->nx2, ny2 = asd->ny2, nz2 = asd->nz2;
  REAL step = asd->step, width_x = asd->width_x, width_y = asd->width_y, width_z = asd->width_z;
  REAL bx = asd->bx, by = asd->by, bz = asd->bz;
  REAL damp = asd->damp;

  // current position
  x = FABS((REAL) (i - nx2) * step);
  y = FABS((REAL) (j - ny2) * step);
  z = FABS((REAL) (k - nz2) * step);

  if(x < bx && y < by && z < bz) return tstep;

  tmp = 0.0;
  if(x >= bx) tmp += (x - bx) / width_x;
  if(y >= by) tmp += (y - by) / width_y;
  if(z >= bz) tmp += (z - bz) / width_z;
  // TODO: fast_tanh does not work with single precision
#ifdef SINGLE_PREC
  tmp = TANH(tmp) * damp;  // Does not work for single precision???
#else
  tmp = fast_tanh(tmp) * damp;   // tanh() is slow - use lookup table
#endif
  return (1.0 - I * tmp) * CABS(tstep);
}

/*
 * Propagate kinetic (1st half).
 *
 * what = DFT_DRIVER_PROPAGATE_HELIUM  or DFT_DRIVER_PROPAGATE_OTHER (char; input).
 * gwf = wavefunction (wf3d *; input).
 * tstep = time step (REAL; input).
 *
 */

EXPORT void dft_driver_propagate_kinetic_first(char what, wf3d *gwf, REAL tstep) {

  REAL complex htime;
 
  if(what == DFT_DRIVER_PROPAGATE_OTHER_ONLYPOT) return;   /* skip kinetic */

  tstep /= GRID_AUTOFS;

  if(driver_iter_mode == DFT_DRIVER_REAL_TIME) htime = tstep / 2.0;
  else htime = -I * tstep / 2.0;

  /* 1/2 x kinetic */
  switch(dft_driver_kinetic) {
  case DFT_DRIVER_KINETIC_FFT: /* this works for absorbing boundaries too ! -- even it is real time there! */
    // NOTE: FFT only takes the time step from the (0, 0, 0) position of the grid only (allows time dependent real / imag switching)
    if(dft_driver_bc_function) {
      struct priv_data data;
      data.nx2 = driver_nx2; data.ny2 = driver_ny2; data.nz2 = driver_nz2;
      data.step = driver_step;
      data.width_x = driver_width_x; data.width_y = driver_width_y; data.width_z = driver_width_z;
      data.bx = driver_bx; data.by = driver_by; data.bz = driver_bz;
      data.damp = driver_damp;
      htime = (*dft_driver_bc_function)((void *) &data, tstep / 2.0, 0, 0, 0); // else use htime
    }
    grid3d_wf_propagate_kinetic_fft(gwf, htime);
    break;
  case DFT_DRIVER_KINETIC_CN_DBC:
    if(!cworkspace)
      cworkspace = dft_driver_alloc_cgrid("DR cworkspace");
    if(driver_boundary_type == DFT_DRIVER_BOUNDARY_ITIME && driver_iter_mode == DFT_DRIVER_REAL_TIME)
      fprintf(stderr, "libdft: CN_DBC absorbing boundary not implemented.\n");
    grid3d_wf_propagate_kinetic_cn_dbc(gwf, htime, cworkspace);
    break;
  case DFT_DRIVER_KINETIC_CN_NBC:
    if(!cworkspace)
      cworkspace = dft_driver_alloc_cgrid("DR cworkspace");
    if(driver_boundary_type == DFT_DRIVER_BOUNDARY_ITIME && driver_iter_mode == DFT_DRIVER_REAL_TIME) { // do not apply in imag time
      struct priv_data data;
      data.nx2 = driver_nx2; data.ny2 = driver_ny2; data.nz2 = driver_nz2;
      data.step = driver_step;
      data.width_x = driver_width_x; data.width_y = driver_width_y; data.width_z = driver_width_z;
      data.bx = driver_bx; data.by = driver_by; data.bz = driver_bz;
      data.damp = driver_damp;
      if(dft_driver_bc_function)
        grid3d_wf_propagate_kinetic_cn_nbc2(gwf, dft_driver_bc_function, htime, (void *) &data, cworkspace);
      else
        grid3d_wf_propagate_kinetic_cn_nbc2(gwf, dft_driver_itime_abs, htime, (void *) &data, cworkspace);
    } else grid3d_wf_propagate_kinetic_cn_nbc(gwf, htime, cworkspace);
    break;
  case DFT_DRIVER_KINETIC_CN_NBC_ROT:
    if(!cworkspace)
      cworkspace = dft_driver_alloc_cgrid("DR cworkspace");
    if(driver_boundary_type == DFT_DRIVER_BOUNDARY_ITIME && driver_iter_mode == DFT_DRIVER_REAL_TIME)
      fprintf(stderr, "libdft: CN_DBC absorbing boundary not implemented.\n");
    grid3d_wf_propagate_kinetic_cn_nbc_rot(gwf, htime, driver_omega, cworkspace);
    break;
  case DFT_DRIVER_KINETIC_CN_PBC:
    if(!cworkspace)
      cworkspace = dft_driver_alloc_cgrid("DR cworkspace");
    if(driver_boundary_type == DFT_DRIVER_BOUNDARY_ITIME && driver_iter_mode == DFT_DRIVER_REAL_TIME)
      fprintf(stderr, "libdft: CN_DBC absorbing boundary not implemented.\n");
    grid3d_wf_propagate_kinetic_cn_pbc(gwf, htime, cworkspace);
    break;
#if 0
  case DFT_DRIVER_KINETIC_CN_APBC:
    if(!cworkspace)
      cworkspace = dft_driver_alloc_cgrid("DR cworkspace");
    grid3d_wf_propagate_kinetic_cn_apbc(gwf, htime, cworkspace);
    break;
#endif
  default:
    fprintf(stderr, "libdft: Unknown BC for kinetic energy propagation.\n");
    exit(1);
  }
  if(driver_iter_mode == DFT_DRIVER_IMAG_TIME) scale_wf(what, gwf);
}

/*
 * Propagate kinetic (2nd half).
 *
 * what = DFT_DRIVER_PROPAGATE_HELIUM or DFT_DRIVER_PROPAGATE_OTHER (char; input).
 * gwf = wavefunction (wf3d *; input/output).
 * tstep = time step (REAL; input).
 *
 */

EXPORT void dft_driver_propagate_kinetic_second(char what, wf3d *gwf, REAL tstep) {

  static char local_been_here = 0;
  
  if(what == DFT_DRIVER_PROPAGATE_OTHER_ONLYPOT) return;   /* skip kinetic */

  dft_driver_propagate_kinetic_first(what, gwf, tstep);

  if(!local_been_here) {
    local_been_here = 1;
    dft_driver_write_wisdom(dft_driver_wisfile()); // we have done many FFTs at this point
  }
}

/*
 * Calculate OT-DFT potential.
 *
 * gwf = wavefunction (wf3d *; input).
 * pot = complex potential (cgrid3d *; output).
 *
 */

EXPORT void dft_driver_ot_potential(wf3d *gwf, cgrid3d *pot) {

  grid3d_wf_density(gwf, density);

  dft_ot3d_potential(dft_driver_otf, pot, gwf, density, workspace1, workspace2, workspace3, workspace4, workspace5, workspace6, workspace7, workspace8, workspace9);
}

/*
 * Compute the viscous potential (Navier-Stokes).
 *
 * gwf = wavefunction (wf3d *; input).
 * pot = potential (cgrid3d *; output).
 *
 */

#define POISSON /* Solve Poisson eq for the viscous potential */

static REAL visc_func(REAL rho) {

  return POW(rho / driver_rho0, viscosity_alpha) * viscosity;  // viscosity_alpha > 0
}

EXPORT void dft_driver_viscous_potential(wf3d *gwf, cgrid3d *pot) {

#ifdef POISSON
  // was 1e-8   (crashes, 5E-8 done, test 1E-7)
  // we have to worry about 1 / rho....
#define POISSON_EPS 1E-7
  if(!workspace8) workspace8 = dft_driver_alloc_rgrid("DR workspace8");

  /* Stress tensor elements (without viscosity) */
  /* 1 (diagonal; workspace2) */
  dft_driver_veloc_field_x_eps(gwf, workspace8, POISSON_EPS); // Watch out! workspace1 used by veloc_field
  rgrid3d_fd_gradient_x(workspace8, workspace2);
  rgrid3d_multiply(workspace2, 4.0/3.0);
  dft_driver_veloc_field_y_eps(gwf, workspace8, POISSON_EPS);
  rgrid3d_fd_gradient_y(workspace8, workspace1);
  rgrid3d_multiply(workspace1, -2.0/3.0);
  rgrid3d_sum(workspace2, workspace2, workspace1);
  dft_driver_veloc_field_z_eps(gwf, workspace8, POISSON_EPS);
  rgrid3d_fd_gradient_z(workspace8, workspace1);
  rgrid3d_multiply(workspace1, -2.0/3.0);
  rgrid3d_sum(workspace2, workspace2, workspace1);

  /* 2 = 4 (symmetry; workspace3) */
  dft_driver_veloc_field_y_eps(gwf, workspace8, POISSON_EPS); // Watch out! workspace1 used by veloc_field
  rgrid3d_fd_gradient_x(workspace8, workspace3);
  dft_driver_veloc_field_x_eps(gwf, workspace8, POISSON_EPS);
  rgrid3d_fd_gradient_y(workspace8, workspace1);
  rgrid3d_sum(workspace3, workspace3, workspace1);
  
  /* 3 = 7 (symmetry; workspace4) */
  dft_driver_veloc_field_z_eps(gwf, workspace8, POISSON_EPS); // Watch out! workspace1 used by veloc_field
  rgrid3d_fd_gradient_x(workspace8, workspace4);
  dft_driver_veloc_field_x_eps(gwf, workspace8, POISSON_EPS);
  rgrid3d_fd_gradient_z(workspace8, workspace1);
  rgrid3d_sum(workspace4, workspace4, workspace1);

  /* 5 (diagonal; workspace5) */
  dft_driver_veloc_field_y_eps(gwf, workspace8, POISSON_EPS); // Watch out! workspace1 used by veloc_field
  rgrid3d_fd_gradient_y(workspace8, workspace5);
  rgrid3d_multiply(workspace5, 4.0/3.0);
  dft_driver_veloc_field_x_eps(gwf, workspace8, POISSON_EPS);
  rgrid3d_fd_gradient_x(workspace8, workspace1);
  rgrid3d_multiply(workspace1, -2.0/3.0);
  rgrid3d_sum(workspace5, workspace5, workspace1);
  dft_driver_veloc_field_z_eps(gwf, workspace8, POISSON_EPS);
  rgrid3d_fd_gradient_z(workspace8, workspace1);
  rgrid3d_multiply(workspace1, -2.0/3.0);
  rgrid3d_sum(workspace5, workspace5, workspace1);
  
  /* 6 = 8 (symmetryl workspace6) */
  dft_driver_veloc_field_z_eps(gwf, workspace8, POISSON_EPS); // Watch out! workspace1 used by veloc_field
  rgrid3d_fd_gradient_y(workspace8, workspace6);
  dft_driver_veloc_field_y_eps(gwf, workspace8, POISSON_EPS);
  rgrid3d_fd_gradient_z(workspace8, workspace1);
  rgrid3d_sum(workspace6, workspace6, workspace1);

  /* 9 = (diagonal; workspace7) */
  dft_driver_veloc_field_z_eps(gwf, workspace8, POISSON_EPS); // Watch out! workspace1 used by veloc_field
  rgrid3d_fd_gradient_z(workspace8, workspace7);
  rgrid3d_multiply(workspace7, 4.0/3.0);
  dft_driver_veloc_field_x_eps(gwf, workspace8, POISSON_EPS);
  rgrid3d_fd_gradient_x(workspace8, workspace1);
  rgrid3d_multiply(workspace1, -2.0/3.0);
  rgrid3d_sum(workspace7, workspace7, workspace1);
  dft_driver_veloc_field_y_eps(gwf, workspace8, POISSON_EPS);
  rgrid3d_fd_gradient_y(workspace8, workspace1);
  rgrid3d_multiply(workspace1, -2.0/3.0);
  rgrid3d_sum(workspace7, workspace7, workspace1);

  /* factor in viscosity */
  grid3d_wf_density(gwf, workspace8);
  rgrid3d_operate_one(workspace8, workspace8, visc_func);
  rgrid3d_product(workspace2, workspace2, workspace8);
  rgrid3d_product(workspace3, workspace3, workspace8);
  rgrid3d_product(workspace4, workspace4, workspace8);
  rgrid3d_product(workspace5, workspace5, workspace8);
  rgrid3d_product(workspace6, workspace6, workspace8);
  rgrid3d_product(workspace7, workspace7, workspace8);
  
  /* x component of divergence (workspace1) */
  rgrid3d_div(workspace1, workspace2, workspace3, workspace4); // (d/dx) 1(wrk2) + (d/dy) 2(wrk3) + (d/dz) 3(wrk4)
  /* y component of divergence (workspace2) */
  rgrid3d_div(workspace2, workspace3, workspace5, workspace6); // (d/dx) 2(wrk3) + (d/dy) 5(wrk5) + (d/dz) 6(wrk6)
  /* x component of divergence (workspace3) */
  rgrid3d_div(workspace3, workspace4, workspace6, workspace7); // (d/dx) 3(wrk4) + (d/dy) 6(wrk6) + (d/dz) 9(wrk7)

  /* divide by -rho */
  grid3d_wf_density(gwf, workspace8);
  rgrid3d_multiply(workspace8, -1.0);
  rgrid3d_division_eps(workspace1, workspace1, workspace8, POISSON_EPS);
  rgrid3d_division_eps(workspace2, workspace2, workspace8, POISSON_EPS);
  rgrid3d_division_eps(workspace3, workspace3, workspace8, POISSON_EPS);
  
  /* the final divergence */
  rgrid3d_div(workspace8, workspace1, workspace2, workspace3);
  
  // Solve the Poisson equation to get the viscous potential
  rgrid3d_poisson(workspace8);
  grid3d_add_real_to_complex_re(pot, workspace8);
#else
  // NOT IN USE
  REAL tot = -(4.0 / 3.0) * viscosity / driver_rho0;
  
  dft_driver_veloc_field_eps(gwf, workspace2, workspace3, workspace4, DFT_BF_EPS); // Watch out! workspace1 used by veloc_field
  rgrid3d_div(workspace1, workspace2, workspace3, workspace4);  
  rgrid3d_multiply(workspace1, tot);
  grid3d_add_real_to_complex_re(pot, workspace1);
#endif
}

/*
 * Propagate potential.
 *
 * what  = DFT_DRIVER_PROPAGATE_HELIUM  or DFT_DRIVER_PROPAGATE_OTHER (char; input).
 * gwf   = wavefunction (wf3d *; input/output).
 * pot   = potential (cgrid3d *; input).
 * tstep = time step (REAL, input).
 *
 */

EXPORT void dft_driver_propagate_potential(char what, wf3d *gwf, cgrid3d *pot, REAL tstep) {

  REAL complex time;

  tstep /= GRID_AUTOFS;
  if(driver_iter_mode == DFT_DRIVER_REAL_TIME) time = tstep;
  else time = -I * tstep;

  if(driver_boundary_type == DFT_DRIVER_BOUNDARY_ITIME && driver_iter_mode == DFT_DRIVER_REAL_TIME) {
    struct priv_data data;
    data.nx2 = driver_nx2; data.ny2 = driver_ny2; data.nz2 = driver_nz2;
    data.step = driver_step;
    data.width_x = driver_width_x; data.width_y = driver_width_y; data.width_z = driver_width_z;
    data.bx = driver_bx; data.by = driver_by; data.bz = driver_bz;
    data.damp = driver_damp;
    if(dft_driver_bc_function)
      grid3d_wf_propagate_potential2(gwf, pot, dft_driver_bc_function, time, (void *) &data);
    else
      grid3d_wf_propagate_potential2(gwf, pot, dft_driver_itime_abs, time, (void *) &data);
  } else grid3d_wf_propagate_potential(gwf, pot, time);

  if(driver_iter_mode == DFT_DRIVER_IMAG_TIME) scale_wf(what, gwf);
}

/*
 * Predict step: propagate the given wf in time.
 *
 * what      = DFT_DRIVER_PROPAGATE_HELIUM  or DFT_DRIVER_PROPAGATE_OTHER (char; input).
 * ext_pot   = present external potential grid (rgrid3d *; input) (NULL = no ext. pot).
 * gwf       = liquid wavefunction to propagate (wf3d *; input).
 *             Note that gwf is NOT changed by this routine.
 * gwfp      = predicted wavefunction (wf3d *; output).
 * potential = storage space for the potential (cgrid3d *; output).
 *             Do not overwrite this before calling the correct routine.
 * tstep     = time step in FS (REAL; input).
 * iter      = current iteration (INT; input).
 *
 * If what == DFT_DRIVER_PROPAGATE_HELIUM, the liquid potential is added automatically. Both kinetic and potential propagated.
 * If what == DFT_DIRVER_PROPAGATE_OTHER, propagate only with the external potential (i.e., impurity). Both kin + ext pot. propagated.
 * If what == DFT_DRIVER_PROPAGATE_OTHER_ONLYPOT, propagate only with the external potential - no kinetic energy (Lappacian).
 *
 * No return value.
 *
 */

EXPORT inline void dft_driver_propagate_predict(char what, rgrid3d *ext_pot, wf3d *gwf, wf3d *gwfp, cgrid3d *potential, REAL tstep, INT iter) {

  grid_timer_start(&timer);  

  if(driver_nx == 0) {
    fprintf(stderr, "libdft: dft_driver not setup.\n");
    exit(1);
  }

  if(!iter && driver_iter_mode == DFT_DRIVER_IMAG_TIME && what < DFT_DRIVER_PROPAGATE_OTHER && dft_driver_init_wavefunction == 1) {
    if(dft_driver_verbose) fprintf(stderr, "libdft: first imag. time iteration - initializing the wavefunction.\n");
    grid3d_wf_constant(gwf, SQRT(dft_driver_otf->rho0));
  }

  /* droplet & column center release */
  if(driver_rels && iter > driver_rels && driver_norm_type > 0 && what < DFT_DRIVER_PROPAGATE_OTHER) {
    if(!center_release && dft_driver_verbose) fprintf(stderr, "libdft: center release activated.\n");
    center_release = 1;
  } else center_release = 0;

  dft_driver_propagate_kinetic_first(what, gwf, tstep);

  cgrid3d_zero(potential);
  switch(what) {
  case DFT_DRIVER_PROPAGATE_HELIUM:
    dft_driver_ot_potential(gwf, potential);
    if(viscosity != 0.0) {
      if(dft_driver_verbose) fprintf(stderr, "libdft: Including viscous potential.\n");
      dft_driver_viscous_potential(gwf, potential);
    }
    break;
  case DFT_DRIVER_PROPAGATE_OTHER:
  case DFT_DRIVER_PROPAGATE_OTHER_ONLYPOT:
    break;
  default:
    fprintf(stderr, "libdft: Unknown propagator flag.\n");
    exit(1);
  }
  if(ext_pot) grid3d_add_real_to_complex_re(potential, ext_pot);

  cgrid3d_copy(gwfp->grid, gwf->grid);

  dft_driver_propagate_potential(what, gwfp, potential, tstep);

  if(dft_driver_verbose) fprintf(stderr, "libdft: Predict step " FMT_R " wall clock seconds (iter = " FMT_I ").\n", grid_timer_wall_clock_time(&timer), iter);
  fflush(stderr);
}

/*
 * Correct step: propagate the given wf in time.
 *
 * what      = DFT_DRIVER_PROPAGATE_HELIUM  or DFT_DRIVER_PROPAGATE_OTHER (char; input).
 * ext_pot   = present external potential grid (rgrid3d *) (NULL = no ext. pot).
 * gwf       = liquid wavefunction to propagate (wf3d *).
 *             Note that gwf is NOT changed by this routine.
 * gwfp      = predicted wavefunction (wf3d *; output).
 * potential = storage space for the potential (cgrid3d *; output).
 * tstep     = time step in FS (REAL).
 * iter      = current iteration (INT).
 *
 * If what == DFT_DRIVER_PROPAGATE_HELIUM, the liquid potential is added automatically. Both kinetic and potential propagated.
 * If what == DFT_DIRVER_PROPAGATE_OTHER, propagate only with the external potential (i.e., impurity). Both kin + ext pot. propagated.
 * If what == DFT_DRIVER_PROPAGATE_OTHER_ONLYPOT, propagate only with the external potential - no kinetic energy (Lappacian).
 *
 * No return value.
 *
 */

EXPORT inline void dft_driver_propagate_correct(char what, rgrid3d *ext_pot, wf3d *gwf, wf3d *gwfp, cgrid3d *potential, REAL tstep, INT iter) {

  grid_timer_start(&timer);  

  switch(what) {
  case DFT_DRIVER_PROPAGATE_HELIUM:
    dft_driver_ot_potential(gwfp, potential);
    if(viscosity != 0.0) {
      if(dft_driver_verbose) fprintf(stderr, "libdft: Including viscous potential.\n");
      dft_driver_viscous_potential(gwfp, potential);
    }
    break;
  case DFT_DRIVER_PROPAGATE_OTHER:
  case DFT_DRIVER_PROPAGATE_OTHER_ONLYPOT:
    break;
  default:
    fprintf(stderr, "libdft: Unknown propagator flag.\n");
    exit(1);
  }
  if(ext_pot) grid3d_add_real_to_complex_re(potential, ext_pot);
  cgrid3d_multiply(potential, 0.5);
  dft_driver_propagate_potential(what, gwf, potential, tstep);
  dft_driver_propagate_kinetic_second(what, gwf, tstep);
  if(dft_driver_verbose) fprintf(stderr, "libdft: Correct step " FMT_R " wall clock seconds (iter = " FMT_I ").\n", grid_timer_wall_clock_time(&timer), iter);
  fflush(stderr);
}

/*
 * Prepare for convoluting potential and density.
 *
 * pot  = potential to be convoluted with (kernel) (input/output. rgrid3d *).
 * dens = denisity to be convoluted with (function) (input/output, rgrid3d *).
 *
 * This must be called before cgrid3d_driver_convolute_eval().
 * Both pot and dens are overwritten with their FFTs.
 * if either is specified as NULL, no transform is done for that grid.
 *
 * No return value.
 *
 */

EXPORT void dft_driver_convolution_prepare(rgrid3d *pot, rgrid3d *dens) {

  if(pot) rgrid3d_fft(pot);
  if(dens) rgrid3d_fft(dens);
}

/*
 * Convolute density and potential.
 *
 * out  = output from convolution (output, cgrid3d *).
 * pot  = potential grid that has been prepared with cgrid3d_driver_convolute_prepare() (input, rgrid3d *).
 * dens = density against which has been prepared with cgrid3d_driver_convolute_prepare() (input, rgrid3d *).
 *
 * No return value.
 *
 */

EXPORT void dft_driver_convolution_eval(rgrid3d *out, rgrid3d *pot, rgrid3d *dens) {

  rgrid3d_fft_convolute(out, pot, dens);
  rgrid3d_inverse_fft(out);
}

/*
 * Allocate a complex grid.
 *
 * Returns a pointer to new grid.
 *
 */

EXPORT cgrid3d *dft_driver_alloc_cgrid(char *id) {

  REAL complex (*grid_type)(cgrid3d *, INT, INT, INT);
  cgrid3d *tmp;

  if(driver_nx == 0) {
    fprintf(stderr, "libdft: dft_driver routines must be initialized first.\n");
    exit(1);
  }

  switch(driver_bc) {
  case DFT_DRIVER_BC_NORMAL: 
    grid_type = CGRID3D_PERIODIC_BOUNDARY;
    break;
  case DFT_DRIVER_BC_X:
    grid_type = CGRID3D_VORTEX_X_BOUNDARY;
    break;
  case DFT_DRIVER_BC_Y:
    grid_type = CGRID3D_VORTEX_Y_BOUNDARY;
    break;
  case DFT_DRIVER_BC_Z:
    grid_type = CGRID3D_VORTEX_Z_BOUNDARY;
    break;
  case DFT_DRIVER_BC_NEUMANN:
    grid_type = CGRID3D_NEUMANN_BOUNDARY;
    break;
  default:
    fprintf(stderr, "libdft: Illegal boundary type.\n");
    exit(1);
  }

  tmp = cgrid3d_alloc(driver_nx, driver_ny, driver_nz, driver_step, grid_type, 0, id);
  cgrid3d_set_origin(tmp, driver_x0, driver_y0, driver_z0);
  cgrid3d_set_momentum(tmp, driver_kx0, driver_ky0, driver_kz0);
  return tmp;
}

/*
 * Allocate a real grid.
 *
 * Returns a pointer to new allocated grid.
 *
 * Note: either if the condition is Neumann b.c. or vortex b.c. for the
 * wavefunction, the real grids such as density always have Neumann b.c.
 *
 */

EXPORT rgrid3d *dft_driver_alloc_rgrid(char *id) {

  REAL (*grid_type)(rgrid3d *, INT, INT, INT);
  rgrid3d *tmp;

  if(driver_nx == 0) {
    fprintf(stderr, "libdft: dft_driver routines must be initialized first.\n");
    exit(1);
  }

  switch(driver_bc) {
  case DFT_DRIVER_BC_NORMAL: 
  case DFT_DRIVER_BC_NEUMANN:
    grid_type = RGRID3D_PERIODIC_BOUNDARY;    // TODO: Neumann should belong to the case below
    break;
  case DFT_DRIVER_BC_X:
  case DFT_DRIVER_BC_Y:
  case DFT_DRIVER_BC_Z:
    /*  case DFT_DRIVER_BC_NEUMANN: */
    grid_type = RGRID3D_NEUMANN_BOUNDARY;
    break;
  default:
    fprintf(stderr, "libdft: Illegal boundary type.\n");
    exit(1);
  }

  tmp = rgrid3d_alloc(driver_nx, driver_ny, driver_nz, driver_step, grid_type, 0, id);
  rgrid3d_set_origin(tmp, driver_x0, driver_y0, driver_z0);
  rgrid3d_set_momentum(tmp, driver_kx0, driver_ky0, driver_kz0);
  return tmp;
}

/*
 * Allocate a wavefunction (initialized to SQRT(rho0)).
 *
 * mass = particle mass in a.u. (input, REAL).
 *
 * Returns pointer to the wavefunction.
 *
 */

EXPORT wf3d *dft_driver_alloc_wavefunction(REAL mass, char *id) {

  wf3d *tmp;
  char grid_type;
  
  if(driver_nx == 0) {
    fprintf(stderr, "libdft: dft_driver routines must be initialized first.\n");
    exit(1);
  }

  switch(driver_bc) {
  case DFT_DRIVER_BC_NORMAL: 
    grid_type = WF3D_PERIODIC_BOUNDARY;
    break;
  case DFT_DRIVER_BC_X:
    grid_type = WF3D_VORTEX_X_BOUNDARY;
    break;
  case DFT_DRIVER_BC_Y:
    grid_type = WF3D_VORTEX_Y_BOUNDARY;
    break;
  case DFT_DRIVER_BC_Z:
    grid_type = WF3D_VORTEX_Z_BOUNDARY;
    break;
  case DFT_DRIVER_BC_NEUMANN:
    grid_type = WF3D_NEUMANN_BOUNDARY;
    break;
  default:
    fprintf(stderr, "libdft: Illegal boundary type.\n");
    exit(1);
  }

  tmp = grid3d_wf_alloc(driver_nx, driver_ny, driver_nz, driver_step, mass, grid_type, WF3D_2ND_ORDER_PROPAGATOR, id);
  cgrid3d_set_origin(tmp->grid, driver_x0, driver_y0, driver_z0);
  cgrid3d_set_momentum(tmp->grid, driver_kx0, driver_ky0, driver_kz0);
  cgrid3d_constant(tmp->grid, SQRT(driver_rho0));
  return tmp;
}

/*
 * Initialize a wavefunction to SQRT of a gaussian function.
 * Useful function for generating an initial guess for impurities.
 *
 * dst   = Wavefunction to be initialized (cgrid3d *; output).
 * cx    = Gaussian center alogn x (REAL; input).
 * cy    = Gaussian center alogn y (REAL; input).
 * cz    = Gaussian center alogn z (REAL; input).
 * width = Gaussian width (REAL; input).
 *
 */

struct asd {
  REAL cx, cy, cz;
  REAL zp;
};
  
static REAL complex dft_gauss(void *ptr, REAL x, REAL y, REAL z) {

  struct asd *lp = (struct asd *) ptr;
  REAL zp = lp->zp, cx = x - lp->cx, cy = y - lp->cy, cz = z - lp->cz;

  return SQRT(POW(zp * zp * M_PI / M_LN2, -3.0/2.0) * EXP(-M_LN2 * (cx * cx + cy * cy + cz * cz) / (zp * zp)));
}

EXPORT void dft_driver_gaussian_wavefunction(wf3d *dst, REAL cx, REAL cy, REAL cz, REAL width) {

  struct asd lp;

  lp.cx = cx;
  lp.cy = cy;
  lp.cz = cz;
  lp.zp = width;
  grid3d_wf_map(dst, dft_gauss, &lp);
}

/*
 * Read in density from a binary file (.grd).
 *
 * grid = place to store the read density (output, rgrid3d *).
 * file = filename for the file (char *). Note: the .grd extension must NOT be given (input, char *).
 *
 * No return value.
 *
 */

EXPORT void dft_driver_read_density(rgrid3d *grid, char *file) {

  FILE *fp;
  char buf[512];

  strcpy(buf, file);
  strcat(buf, ".grd");
  if(!(fp = fopen(buf, "r"))) {
    fprintf(stderr, "libdft: Can't open density grid file %s.\n", file);
    exit(1);
  }
  rgrid3d_read(grid, fp);
  fclose(fp);
  fprintf(stderr, "libdft: Density read from %s.\n", file);
}

/*
 * Write output to ascii & binary files.
 * .grd file is the full grid in binary format.
 * .x ASCII file cut along (x, 0.0, 0.0)
 * .y ASCII file cut along (0.0, y, 0.0)
 * .z ASCII file cut along (0.0, 0.0, z)
 *
 * grid = density grid (input, rgrid3d *).
 * base = Basename for the output file (input, char *).
 *
 * No return value.
 *
 */

EXPORT void dft_driver_write_density(rgrid3d *grid, char *base) {

  FILE *fp;
  char file[2048];
  INT i, j, k;
  REAL x, y, z;

  sprintf(file, "%s.grd", base);
  if(!(fp = fopen(file, "w"))) {
    fprintf(stderr, "libdft: Can't open %s for writing.\n", file);
    exit(1);
  }
  rgrid3d_write(grid, fp);
  fclose(fp);

  sprintf(file, "%s.x", base);
  if(!(fp = fopen(file, "w"))) {
    fprintf(stderr, "libdft: Can't open %s for writing.\n", file);
    exit(1);
  }
  j = grid->ny / 2;
  k = grid->nz / 2;
  for(i = 0; i < grid->nx; i++) { 
    x = ((REAL) (i - grid->nx/2)) * grid->step - grid->x0;
    fprintf(fp, FMT_R " " FMT_R "\n", x, rgrid3d_value_at_index(grid, i, j, k));
  }
  fclose(fp);

  sprintf(file, "%s.y", base);
  if(!(fp = fopen(file, "w"))) {
    fprintf(stderr, "libdft: Can't open %s for writing.\n", file);
    exit(1);
  }
  i = grid->nx / 2;
  k = grid->nz / 2;
  for(j = 0; j < grid->ny; j++) {
    y = ((REAL) (j - grid->ny/2)) * grid->step - grid->y0;
    fprintf(fp, FMT_R " " FMT_R "\n", y, rgrid3d_value_at_index(grid, i, j, k));
  }
  fclose(fp);

  sprintf(file, "%s.z", base);
  if(!(fp = fopen(file, "w"))) {
    fprintf(stderr, "libdft: Can't open %s for writing.\n", file);
    exit(1);
  }
  i = grid->nx / 2;
  j = grid->ny / 2;
  for(k = 0; k < grid->nz; k++) {
    z = ((REAL) (k - grid->nz/2)) * grid->step - grid->z0;
    fprintf(fp, FMT_R " " FMT_R "\n", z, rgrid3d_value_at_index(grid, i, j, k));
  }
  fclose(fp);
  if(dft_driver_verbose) fprintf(stderr, "libdft: Density written to %s.\n", file);
}

/*
 * Write output to ascii & binary files.
 * .grd file is the full grid in binary format.
 * .x ASCII file cut along (x, 0.0, 0.0)
 * .y ASCII file cut along (0.0, y, 0.0)
 * .z ASCII file cut along (0.0, 0.0, z)
 *
 * wf = wf with the pase (input, grid3d *).
 * base = Basename for the output file (input, char *).
 *
 * No return value.
 *
 */

EXPORT void dft_driver_write_phase(wf3d *wf, char *base) {

  FILE *fp;
  char file[2048];
  INT i, j, k;
  cgrid3d *grid = wf->grid;
  REAL complex tmp;
  rgrid3d *phase;
  REAL x, y, z;
  INT nx = grid->nx, ny = grid->ny, nz = grid->nz;

  phase = dft_driver_alloc_rgrid("phase");
  for(i = 0; i < nx; i++)
    for(j = 0; j < ny; j++)
      for(k = 0; k < nz; k++) {
	tmp = cgrid3d_value_at_index(grid, i, j, k);
	if(CABS(tmp) < 1E-6)
          rgrid3d_value_to_index(phase, i, j, k, 0.0);
	else
          rgrid3d_value_to_index(phase, i, j, k, CIMAG(CLOG(tmp / CABS(tmp))));
      }

  sprintf(file, "%s.grd", base);
  if(!(fp = fopen(file, "w"))) {
    fprintf(stderr, "libdft: Can't open %s for writing.\n", file);
    exit(1);
  }
  rgrid3d_write(phase, fp);
  fclose(fp);

  sprintf(file, "%s.x", base);
  if(!(fp = fopen(file, "w"))) {
    fprintf(stderr, "libdft: Can't open %s for writing.\n", file);
    exit(1);
  }
  j = ny / 2;
  k = nz / 2;
  for(i = 0; i < grid->nx; i++) { 
    x = ((REAL) (i - grid->nx/2)) * grid->step;
    fprintf(fp, FMT_R " " FMT_R "\n", x, rgrid3d_value_at_index(phase, i, j, k));
  }
  fclose(fp);

  sprintf(file, "%s.y", base);
  if(!(fp = fopen(file, "w"))) {
    fprintf(stderr, "libdft: Can't open %s for writing.\n", file);
    exit(1);
  }
  i = nx / 2;
  k = nz / 2;
  for(j = 0; j < grid->ny; j++) {
    y = ((REAL) (j - grid->ny/2)) * grid->step;
    fprintf(fp, FMT_R " " FMT_R "\n", y, rgrid3d_value_at_index(phase, i, j, k));
  }
  fclose(fp);

  sprintf(file, "%s.z", base);
  if(!(fp = fopen(file, "w"))) {
    fprintf(stderr, "libdft: Can't open %s for writing.\n", file);
    exit(1);
  }
  i = nx / 2;
  j = ny / 2;
  for(k = 0; k < grid->nz; k++) {
    z = ((REAL) (k - grid->nz/2)) * grid->step;
    fprintf(fp, FMT_R " " FMT_R "\n", z, rgrid3d_value_at_index(phase, i, j, k));
  }
  fclose(fp);
  if(dft_driver_verbose) fprintf(stderr, "libdft: Density written to %s.\n", file);
  rgrid3d_free(phase);
}


/*
 * Write two-dimensional vector slices of a vector grid (three grids)
 * .xy ASCII file cut along z=0.
 * .yz ASCII file cut along x=0.
 * .zx ASCII file cut along y=0.
 *
 * px = x-component of vector grid (input, rgrid3d *).
 * py = y-component of vector grid (input, rgrid3d *).
 * pz = z-component of vector grid (input, rgrid3d *).
 *
 * base = Basename for the output file (input, char *).
 *
 * No return value.
 *
 */
EXPORT void dft_driver_write_vectorfield(rgrid3d *px, rgrid3d *py, rgrid3d *pz, char *base) {

  FILE *fp;
  char file[2048];
  INT i, j, k;
  REAL x0 = px->x0, y0 = px->y0, z0 = px->z0, step = px->step;
  INT nx = px->nx, ny = px->ny, nz = px->nz;
  REAL x, y, z;
  
  /*----- X Y -----*/	
  sprintf(file, "%s.xy", base);
  if(!(fp = fopen(file, "w"))) {
    fprintf(stderr, "libdft: Can't open %s for writing.\n", file);
    exit(1);
  }
  k = nz / 2;
  for(i = 0; i < nx; i++) {
    x = ((REAL) (i - nx/2)) * step - x0;
    for(j = 0; j < ny; j++) {
      y = ((REAL) (j - ny/2)) * step - y0;
      fprintf(fp, FMT_R "\t" FMT_R "\t" FMT_R "\t" FMT_R "\n", x, y, rgrid3d_value_at_index(px, i, j, k), rgrid3d_value_at_index(py, i, j, k));	
    } fprintf(fp,"\n");
  }
  fclose(fp);
  
  /*----- Y Z -----*/
  sprintf(file, "%s.yz", base);
  if(!(fp = fopen(file, "w"))) {
    fprintf(stderr, "libdft: Can't open %s for writing.\n", file);
    exit(1);
  }
  i = nx / 2;
  for(j = 0; j < ny; j++) {
    y = ((REAL) (j - ny/2)) * step - y0;
    for(k = 0; k < nz; k++) {
      z = ((REAL) (k - nz/2)) * step - z0;
      fprintf(fp, FMT_R "\t" FMT_R "\t" FMT_R "\t" FMT_R "\n", y, z, rgrid3d_value_at_index(py, i, j, k), rgrid3d_value_at_index(pz, i, j, k));	
    } fprintf(fp,"\n");
  }
  fclose(fp);
  
  /*----- Z X -----*/
  sprintf(file, "%s.zx", base);
  if(!(fp = fopen(file, "w"))) {
    fprintf(stderr, "libdft: Can't open %s for writing.\n", file);
    exit(1);
  }
  j = ny / 2;
  for(k = 0; k < nz; k++) {
    z = ((REAL) (k - nz/2)) * step - z0;
    for(i = 0; i < nx; i++) {
      x = ((REAL) (i - nx/2)) * step - z0;
      fprintf(fp, FMT_R "\t" FMT_R "\t" FMT_R "\t" FMT_R "\n", z, x, rgrid3d_value_at_index(pz, i, j, k), rgrid3d_value_at_index(px, i, j, k));	
    } fprintf(fp,"\n"); 
  }
  fclose(fp);
  
  if(dft_driver_verbose) fprintf(stderr, "libdft: vector 2D slices of density written to %s.\n", file);
}

/*
 * Write two-dimensional vector slices of a probability current 
 * .xy ASCII file cut along z=0.
 * .yz ASCII file cut along x=0.
 * .zx ASCII file cut along y=0.
 *
 * wf = wavefunction (wf3d, input)
 * base = Basename for the output file (char *, input).
 *
 * No return value.
 *
 * Uses workspace 1-3
 */

EXPORT void dft_driver_write_current(wf3d *wf, char *base) {

  grid3d_wf_probability_flux(wf, workspace1, workspace2, workspace3);
  dft_driver_write_vectorfield( workspace1, workspace2, workspace3, base);
}

/*
 * Write two-dimensional vector slices of velocity 
 * .xy ASCII file cut along z=0.
 * .yz ASCII file cut along x=0.
 * .zx ASCII file cut along y=0.
 *
 * wf = wavefunction (wf3d, input)
 * base = Basename for the output file (char *, input).
 *
 * No return value.
 *
 * Uses workspace 1-3
 * 
 */

EXPORT void dft_driver_write_velocity(wf3d *wf, char *base) {

  dft_driver_veloc_field(wf, workspace1, workspace2, workspace3);
  dft_driver_write_vectorfield(workspace1, workspace2, workspace3, base);
}

/*
 * Read in a grid from a binary file (.grd).
 *
 * grid = grid where the data is placed (cgrid3d *, output).
 * file = filename for the file (char *, input). Note: the .grd extension must be given.
 *
 * No return value.
 *
 */

EXPORT void dft_driver_read_grid(cgrid3d *grid, char *file) {

  FILE *fp;

  if(!(fp = fopen(file, "r"))) {
    fprintf(stderr, "libdft: Can't open complex grid file %s.\n", file);
    exit(1);
  }
  cgrid3d_read(grid, fp);
  fclose(fp);
}

/*
 * Write a complex grid to ascii & binary files.
 * .grd file is the full grid in binary format.
 * .x ASCII file cut along (x, 0.0, 0.0)
 * .y ASCII file cut along (0.0, y, 0.0)
 * .z ASCII file cut along (0.0, 0.0, z)
 *
 * grid = grid to be written (cgrid3d *, input).
 * base = Basename for the output file (char *, input).
 *
 * No return value.
 *
 */

EXPORT void dft_driver_write_grid(cgrid3d *grid, char *base) {

  FILE *fp;
  char file[2048];
  INT i, j, k;
  REAL x, y, z;

  sprintf(file, "%s.grd", base);
  if(!(fp = fopen(file, "w"))) {
    fprintf(stderr, "libdft: Can't open %s for writing.\n", file);
    exit(1);
  }
  cgrid3d_write(grid, fp);
  fclose(fp);

  sprintf(file, "%s.x", base);
  if(!(fp = fopen(file, "w"))) {
    fprintf(stderr, "libdft: Can't open %s for writing.\n", file);
    exit(1);
  }
  j = driver_ny2;
  k = driver_nz2;
  for(i = 0; i < driver_nx; i++) { 
    x = ((REAL) (i - driver_nx2)) * driver_step;
    fprintf(fp, FMT_R " " FMT_R " " FMT_R "\n", x, CREAL(cgrid3d_value_at_index(grid, i, j, k)), CIMAG(cgrid3d_value_at_index(grid, i, j, k)));
  }
  fclose(fp);

  sprintf(file, "%s.y", base);
  if(!(fp = fopen(file, "w"))) {
    fprintf(stderr, "libdft: Can't open %s for writing.\n", file);
    exit(1);
  }
  i = driver_nx2;
  k = driver_nz2;
  for(j = 0; j < driver_ny; j++) {
    y = ((REAL) (j - driver_ny2)) * driver_step;
    fprintf(fp, FMT_R " " FMT_R " " FMT_R "\n", y, CREAL(cgrid3d_value_at_index(grid, i, j, k)), CIMAG(cgrid3d_value_at_index(grid, i, j, k)));
  }
  fclose(fp);

  sprintf(file, "%s.z", base);
  if(!(fp = fopen(file, "w"))) {
    fprintf(stderr, "libdft: Can't open %s for writing.\n", file);
    exit(1);
  }
  i = driver_nx2;
  j = driver_ny2;
  for(k = 0; k < driver_nz; k++) {
    z = ((REAL) (k - driver_nz2)) * driver_step;
    fprintf(fp, FMT_R " " FMT_R " " FMT_R "\n", z, CREAL(cgrid3d_value_at_index(grid, i, j, k)), CIMAG(cgrid3d_value_at_index(grid, i, j, k)));
  }
  fclose(fp);
}

/*
 * Return the self-consistent OT potential (no external potential).
 *
 * gwf       = wavefunction for the system (wf3d *; input).
 * potential = potential grid (rgrid3d *; output).
 *
 * Note: the backflow is not included in the energy density calculation.
 *
 */

EXPORT void dft_driver_potential(wf3d *gwf, rgrid3d *potential) {

  /* we may need more memory for this... */
  if(!workspace7) workspace7 = dft_driver_alloc_rgrid("workspace7");
  if(!workspace8) workspace8 = dft_driver_alloc_rgrid("workspace8");
  if(!workspace9) workspace9 = dft_driver_alloc_rgrid("workspace9");
  if(!cworkspace) cworkspace = dft_driver_alloc_cgrid("cworkspace");

  grid3d_wf_density(gwf, density);
  cgrid3d_zero(cworkspace);
  dft_ot3d_potential(dft_driver_otf, cworkspace, gwf, density, workspace1, workspace2, workspace3, workspace4, workspace5, workspace6, workspace7, workspace8, workspace9);
  grid3d_complex_re_to_real(potential, cworkspace);
}

/*
 * Calculate the total energy of the system.
 *
 * gwf     = wavefunction for the system (wf3d *; input).
 * ext_pot = external potential grid (rgrid3d *; input).
 *
 * Return value = total energy for the system (in a.u.).
 *
 * Note: the backflow is not included in the energy density calculation.
 *
 */

EXPORT REAL dft_driver_energy(wf3d *gwf, rgrid3d *ext_pot) {

  return dft_driver_potential_energy(gwf, ext_pot) + dft_driver_kinetic_energy(gwf);
}

/*
 * Calculate the potential energy of the system.
 *
 * gwf     = wavefunction for the system (wf3d *; input).
 * ext_pot = external potential grid (rgrid3d *; input).
 *
 * Return value = potential energy for the system (in a.u.).
 *
 * Note: the backflow is not included in the energy density calculation.
 *
 */

EXPORT REAL dft_driver_potential_energy(wf3d *gwf, rgrid3d *ext_pot) {

  /* we may need more memory for this... */
  if(!workspace7) workspace7 = dft_driver_alloc_rgrid("workspace7");
  if(!workspace8) workspace8 = dft_driver_alloc_rgrid("workspace8");
  if(!workspace9) workspace9 = dft_driver_alloc_rgrid("workspace9");

  grid3d_wf_density(gwf, density);

  dft_ot3d_energy_density(dft_driver_otf, workspace9, gwf, density, workspace1, workspace2, workspace3, workspace4, workspace5, workspace6, workspace7, workspace8);

  if(ext_pot) rgrid3d_add_scaled_product(workspace9, 1.0, density, ext_pot);
  
  return rgrid3d_integral(workspace9);
}

/*
 * Calculate the kinetic energy of the system.
 *
 * gwf     = wavefunction for the system (wf3d *; input).
 *
 * Return value = kinetic energy for the system (in a.u.).
 *
 */

EXPORT REAL dft_driver_kinetic_energy(wf3d *gwf) {
  
  if(!cworkspace) cworkspace = dft_driver_alloc_cgrid("workspace");

  /* Since CN_NBC and CN_NBC_ROT do not use FFT for kinetic propagation, evaluate the kinetic energy with finite difference as well */
  if((dft_driver_kinetic == DFT_DRIVER_KINETIC_CN_NBC_ROT || dft_driver_kinetic == DFT_DRIVER_KINETIC_CN_NBC) && driver_bc == DFT_DRIVER_BC_NEUMANN) {
    /* FIXME: Right now there is no way to make grid3d_wf_energy() do finite difference energy with Neumann. */
    REAL mass = gwf->mass, kx = gwf->grid->kx0 , ky = gwf->grid->ky0 , kz = gwf->grid->kz0;
    REAL ekin = -HBAR * HBAR * (kx * kx + ky * ky + kz * kz) / (2.0 * mass);

    if(ekin != 0.0)
      ekin *= CREAL(cgrid3d_integral_of_square(gwf->grid)); 
    return grid3d_wf_energy_cn(gwf, gwf, NULL, cworkspace) + ekin;
  }
  return grid3d_wf_energy(gwf, NULL, cworkspace);
}

/*
 * Calculate the energy from the rotation constrain,
 * ie -<omega*L>.
 *
 * gwf     = wavefunction for the system (wf3d *; input).
 * omega_x = angular frequency in a.u., x-axis (REAL, input)
 * omega_y = angular frequency in a.u., y-axis (REAL, input)
 * omega_z = angular frequency in a.u., z-axis (REAL, input)
 *
 */

EXPORT REAL dft_driver_rotation_energy(wf3d *wf, REAL omega_x, REAL omega_y, REAL omega_z) {

  REAL lx, ly, lz;

  dft_driver_L( wf, &lx, &ly, &lz);
  return -(omega_x * lx) - (omega_y * ly) - (omega_z * lz);
}

/*
 * Calculate the energy in a certain region (box).
 *
 * gwf     = wavefunction for the system (wf3d *; input).
 * ext_pot = external potential grid (rgrid3d *; input).
 * xl   = lower limit for x (REAL, input).
 * xu   = upper limit for x (REAL, input).
 * yl   = lower limit for y (REAL, input).
 * yu   = upper limit for y (REAL, input).
 * zl   = lower limit for z (REAL, input).
 * zu   = upper limit for z (REAL, input).
 *
 * Return value = energy for the box (in a.u.).
 *
 * Note: the backflow is not included in the energy density calculation.
 *
 */

EXPORT REAL dft_driver_energy_region(wf3d *gwf, rgrid3d *ext_pot, REAL xl, REAL xu, REAL yl, REAL yu, REAL zl, REAL zu) {

  REAL energy;

  /* we may need more memory for this... */

  if(!workspace7) workspace7 = dft_driver_alloc_rgrid("workspace7");
  if(!workspace8) workspace8 = dft_driver_alloc_rgrid("workspace8");
  if(!workspace9) workspace9 = dft_driver_alloc_rgrid("workspace9");
  grid3d_wf_density(gwf, density);
  dft_ot3d_energy_density(dft_driver_otf, workspace9, gwf, density, workspace1, workspace2, workspace3, workspace4, workspace5, workspace6, workspace7, workspace8);
  if(ext_pot) rgrid3d_add_scaled_product(workspace9, 1.0, density, ext_pot);
  energy = rgrid3d_integral_region(workspace9, xl, xu, yl, yu, zl, zu);
  if(!cworkspace)
    cworkspace = dft_driver_alloc_cgrid("cworkspace");
  energy += grid3d_wf_energy(gwf, NULL, cworkspace);
  return energy;
}

/*
 * Return number of helium atoms represented by a given wavefuntion.
 *
 * gwf = wavefunction (wf3d *; input).
 *
 * Returns the # of He atoms (note: can be fractional).
 *
 */

EXPORT REAL dft_driver_natoms(wf3d *gwf) {

  return CREAL(cgrid3d_integral_of_square(gwf->grid));
}

/*
 * Evaluate absorption/emission spectrum using the Andersson
 * expression. No zero-point correction for the impurity.
 *
 * density  = Current liquid density (rgrid3d *; input).
 * tstep    = Time step for constructing the time correlation function
 *            (REAL; input in fs). Typically around 1 fs.
 * endtime  = End time in constructing the time correlation function
 *            (REAL; input in fs). Typically less than 10,000 fs.
 * finalave = Averaging of the final state potential.
 *            0: no averaging, 1 = average XY, 2 = average YZ, 3 = average XZ,
 *            4 = average XYZ.
 * finalx   = Final state potential along the X axis (char *; input).
 * finaly   = Final state potential along the Y axis (char *; input).
 * finalz   = Final state potential along the Z axis (char *; input).
 * initialave = Averaging of the initial state potential.
 *            0: no averaging, 1 = average XY, 2 = average YZ, 3 = average XZ,
 *            4 = average XYZ.
 * initialx   = Initial state potential along the X axis (char *; input).
 * initialy   = Initial state potential along the Y axis (char *; input).
 * initialz   = Initial state potential along the Z axis (char *; input).
 *
 * NOTE: This needs complex grid input!! 
 *
 * Returns the spectrum (cgrid1d *; output). Note that this is statically
 * allocated and overwritten by a subsequent call to this routine.
 * The spectrum will start from smaller to larger frequencies.
 * The spacing between the points is included in cm-1.
 *
 */

static REAL complex dft_eval_exp(REAL complex a) { /* a contains t */

  return (1.0 - CEXP(-I * a));
}

static REAL complex dft_do_int(rgrid3d *dens, rgrid3d *dpot, REAL t, cgrid3d *wrk) {

  grid3d_real_to_complex_re(wrk, dpot);
  cgrid3d_multiply(wrk, t);
  cgrid3d_operate_one(wrk, wrk, dft_eval_exp);
  grid3d_product_complex_with_real(wrk, dens);
  return cgrid3d_integral(wrk);            // debug: This should have minus in front?! Sign error elsewhere? (does not appear in ZP?!)
}

EXPORT cgrid1d *dft_driver_spectrum(rgrid3d *density, REAL tstep, REAL endtime, char finalave, char *finalx, char *finaly, char *finalz, char initialave, char *initialx, char *initialy, char *initialz) {

  rgrid3d *dpot;
  cgrid3d *wrk[256];
  static cgrid1d *corr = NULL;
  REAL t;
  INT i, ntime;
  static INT prev_ntime = -1;

  endtime /= GRID_AUTOFS;
  tstep /= GRID_AUTOFS;
  ntime = (1 + (int) (endtime / tstep));
  dpot = rgrid3d_alloc(density->nx, density->ny, density->nz, density->step, RGRID3D_PERIODIC_BOUNDARY, 0, "DR spectrum dpot");
  for (i = 0; i < omp_get_max_threads(); i++)
    wrk[i] = cgrid3d_alloc(density->nx, density->ny, density->nz, density->step, CGRID3D_PERIODIC_BOUNDARY, 0, "DR spectrum wrk");
  if(ntime != prev_ntime) {
    if(corr) cgrid1d_free(corr);
    corr = cgrid1d_alloc(ntime, 0.1, CGRID1D_PERIODIC_BOUNDARY, 0);
    prev_ntime = ntime;
  }

  dft_common_potential_map(finalave, finalx, finaly, finalz, workspace1);
  dft_common_potential_map(initialave, initialx, initialy, initialz, workspace2);
  rgrid3d_difference(dpot, workspace1, workspace2); /* final - initial */
  
  rgrid3d_product(workspace1, dpot, density);
  fprintf(stderr, "libdft: Average shift = " FMT_R " cm-1.\n", rgrid3d_integral(workspace1) * GRID_AUTOCM1);

#pragma omp parallel for firstprivate(tstep,ntime,density,dpot,corr,wrk) private(i,t) default(none) schedule(runtime)
  for(i = 0; i < ntime; i++) {
    t = tstep * (REAL) i;
    corr->value[i] = CEXP(dft_do_int(density, dpot, t, wrk[omp_get_thread_num()])) * POW(-1.0, (REAL) i);
  }
  cgrid1d_fft(corr);
  for (i = 0; i < corr->nx; i++)
    corr->value[i] = CABS(corr->value[i]);
  
  corr->step = GRID_HZTOCM1 / (tstep * GRID_AUTOFS * 1E-15 * (REAL) ntime);

  rgrid3d_free(dpot);
  for(i = 0; i < omp_get_max_threads(); i++)
    cgrid3d_free(wrk[i]);

  return corr;
}

/*
 *
 * TODO: This still needs to be modified so that it takes rgrid3d
 * for density and imdensity.
 *
 * Evaluate absorption/emission spectrum using the Andersson
 * expression. Zero-point correction for the impurity included.
 *
 * density  = Current liquid density (rgrid3d *; input).
 * imdensity= Current impurity zero-point density (cgrid3d *; input).
 * tstep    = Time step for constructing the time correlation function
 *            (REAL; input in fs). Typically around 1 fs.
 * endtime  = End time in constructing the time correlation function
 *            (REAL; input in fs). Typically less than 10,000 fs.
 * upperave = Averaging of the upperial state potential.
 *            0: no averaging, 1 = average XY, 2 = average YZ, 3 = average XZ,
 *            4 = average XYZ.
 * upperx   = Upper state potential along the X axis (char *; input).
 * uppery   = Upper state potential along the Y axis (char *; input).
 * upperz   = Upper state potential along the Z axis (char *; input).
 * lowerave = Averaging of the lower state potential.
 *            0: no averaging, 1 = average XY, 2 = average YZ, 3 = average XZ,
 *            4 = average XYZ.
 * lowerx   = Lower state potential along the X axis (char *; input).
 * lowery   = Lower state potential along the Y axis (char *; input).
 * lowerz   = Lower state potential along the Z axis (char *; input).
 *
 * NOTE: This needs complex grid input!! 
 *
 * Returns the spectrum (cgrid1d *; output). Note that this is statically
 * allocated and overwritten by a subsequent call to this routine.
 * The spectrum will start from smaller to larger frequencies.
 * The spacing between the points is included in cm-1.
 *
 */

static void do_gexp(cgrid3d *gexp, rgrid3d *dpot, REAL t) {

  grid3d_real_to_complex_re(gexp, dpot);
  cgrid3d_multiply(gexp, t);
  cgrid3d_operate_one(gexp, gexp, dft_eval_exp);
  cgrid3d_fft(gexp);  
#if 0
  cgrid3d_zero(gexp);
  cgrid3d_add_scaled(gexp, t, dpot);
  cgrid3d_operate_one(gexp, gexp, dft_eval_exp);
  cgrid3d_fft(gexp);
#endif
}

static REAL complex dft_do_int2(cgrid3d *gexp, rgrid3d *imdens, cgrid3d *fft_dens, REAL t, cgrid3d *wrk) {

  cgrid3d_fft_convolute(wrk, fft_dens, gexp);
  cgrid3d_inverse_fft(wrk);
  grid3d_product_complex_with_real(wrk, imdens);

  return -cgrid3d_integral(wrk);
#if 0
  cgrid3d_zero(wrk);
  cgrid3d_fft_convolute(wrk, dens, gexp);
  cgrid3d_inverse_fft(wrk);
  cgrid3d_product(wrk, wrk, imdens);

  return -cgrid3d_integral(wrk);
#endif
}

EXPORT cgrid1d *dft_driver_spectrum_zp(rgrid3d *density, rgrid3d *imdensity, REAL tstep, REAL endtime, char upperave, char *upperx, char *uppery, char *upperz, char lowerave, char *lowerx, char *lowery, char *lowerz) {

  cgrid3d *wrk, *fft_density, *gexp;
  rgrid3d *dpot;
  static cgrid1d *corr = NULL;
  REAL t;
  INT i, ntime;
  static INT prev_ntime = -1;

  endtime /= GRID_AUTOFS;
  tstep /= GRID_AUTOFS;
  ntime = (1 + (int) (endtime / tstep));
  dpot = rgrid3d_alloc(density->nx, density->ny, density->nz, density->step, RGRID3D_PERIODIC_BOUNDARY, 0, "DR spectrum_zp dpot");
  fft_density = cgrid3d_alloc(density->nx, density->ny, density->nz, density->step, CGRID3D_PERIODIC_BOUNDARY, 0, "DR spectrum_zp fftd");
  wrk = cgrid3d_alloc(density->nx, density->ny, density->nz, density->step, CGRID3D_PERIODIC_BOUNDARY, 0, "DR spectrum_zp wrk");
  gexp = cgrid3d_alloc(density->nx, density->ny, density->nz, density->step, CGRID3D_PERIODIC_BOUNDARY, 0, "DR spectrum_zp gexp");
  if(ntime != prev_ntime) {
    if(corr) cgrid1d_free(corr);
    corr = cgrid1d_alloc(ntime, 0.1, CGRID1D_PERIODIC_BOUNDARY, 0);
    prev_ntime = ntime;
  }
  
  fprintf(stderr, "libdft: Upper level potential.\n");
  dft_common_potential_map(upperave, upperx, uppery, upperz, workspace1);
  fprintf(stderr, "libdft: Lower level potential.\n");
  dft_common_potential_map(lowerave, lowerx, lowery, lowerz, workspace2);
  rgrid3d_difference(dpot, workspace2, workspace1);
  
  grid3d_real_to_complex_re(fft_density, density);
  cgrid3d_fft(fft_density);
  
  // can't run in parallel - actually no much sense since the most time intensive
  // part is the fft (which runs in parallel)
  for(i = 0; i < ntime; i++) {
    t = tstep * (REAL) i;
    do_gexp(gexp, dpot, t); /* gexp grid + FFT */
    corr->value[i] = CEXP(dft_do_int2(gexp, imdensity, fft_density, t, wrk)) * POW(-1.0, (REAL) i);
    fprintf(stderr,"libdft: Corr(" FMT_R " fs) = " FMT_R " " FMT_R "\n", t * GRID_AUTOFS, CREAL(corr->value[i]), CIMAG(corr->value[i]));
  }
  cgrid1d_fft(corr);
  for (i = 0; i < corr->nx; i++)
    corr->value[i] = CABS(corr->value[i]);
  
  corr->step = GRID_HZTOCM1 / (tstep * GRID_AUTOFS * 1E-15 * (REAL) ntime);
  
  rgrid3d_free(dpot);
  cgrid3d_free(fft_density);
  cgrid3d_free(wrk);
  cgrid3d_free(gexp);
  
  return corr;
}

/*
 * Routines for evaluating the dynamic lineshape (similar to CPL 396, 155 (2004) but
 * see intro of JCP 141, 014107 (2014) + references there in). The dynamics should be run on average
 * potential of gnd and excited states (returned by the init routine).
 *
 * 1) Initialize the difference potential:
 *     dft_driver_spectrum_init().
 * 
 * 2) During the trajectory, call function:
 *     dft_driver_spectrum_collect() to record the time dependent difference
 *     energy (difference potential convoluted with the time dependent
 *     liquid density).
 *
 * 3) At the end, call the following function to evaluate the spectrum:
 *     dft_driver_spectrum_evaluate() to evaluate the lineshape.
 *
 */

/*
 * Collect the time dependent difference energy data.
 * 
 * idensity = NULL: no averaging of pair potentials, rgrid3d *: impurity density for convoluting with pair potential. (input)
 * nt       = Maximum number of time steps to be collected (INT, input).
 * zerofill = How many zeros to fill in before FFT (int, input).
 * upperave = Averaging on the upper state (see dft_driver_potential_map()) (int, input).
 * upperx   = Upper potential file name along-x (char *, input).
 * uppery   = Upper potential file name along-y (char *, input).
 * upperz   = Upper potential file name along-z (char *, input).
 * lowerave = Averaging on the lower state (see dft_driver_potential_map()) (int, input).
 * lowerx   = Lower potential file name along-x (char *, input).
 * lowery   = Lower potential file name along-y (char *, input).
 * lowerz   = Lower potential file name along-z (char *, input).
 *
 * Returns difference potential for dynamics.
 *
 */

static rgrid3d *xxdiff = NULL, *xxave = NULL;
static cgrid1d *tdpot = NULL;
static INT ntime, cur_time, zerofill;

EXPORT rgrid3d *dft_driver_spectrum_init(rgrid3d *idensity, INT nt, INT zf, char upperave, char *upperx, char *uppery, char *upperz, char lowerave, char *lowerx, char *lowery, char *lowerz) {

  cur_time = 0;
  ntime = nt;
  zerofill = zf;
  if(!tdpot)
    tdpot = cgrid1d_alloc(ntime + zf, 0.1, CGRID1D_PERIODIC_BOUNDARY, 0);
  if(upperx == NULL) return NULL;   /* potentials not given */
  if(!xxdiff)
    xxdiff = rgrid3d_alloc(driver_nx, driver_ny, driver_nz, driver_step, RGRID3D_PERIODIC_BOUNDARY, 0, "xxdiff");
  if(!xxave)
    xxave = rgrid3d_alloc(driver_nx, driver_ny, driver_nz, driver_step, RGRID3D_PERIODIC_BOUNDARY, 0, "xxave");
  fprintf(stderr, "libdft: Upper level potential.\n");
  dft_common_potential_map(upperave, upperx, uppery, upperz, workspace1);
  fprintf(stderr, "libdft: Lower level potential.\n");
  dft_common_potential_map(lowerave, lowerx, lowery, lowerz, workspace2);
  fprintf(stderr, "libdft: spectrum init complete.\n");
  if(idensity) {
    dft_driver_convolution_prepare(idensity, workspace1);
    dft_driver_convolution_eval(workspace3, workspace1, idensity);
    dft_driver_convolution_prepare(NULL, workspace2);
    dft_driver_convolution_eval(workspace4, workspace2, idensity);    
  } else {
    rgrid3d_copy(workspace3, workspace1);
    rgrid3d_copy(workspace4, workspace2);
  }

  rgrid3d_difference(xxdiff, workspace3, workspace4);
  rgrid3d_sum(xxave, workspace3, workspace4);
  rgrid3d_multiply(xxave, 0.5);
  return xxave;
}

/*
 * Collect the time dependent difference energy data. Same as above but with direct
 * grid input for potentials.
 * 
 * nt       = Maximum number of time steps to be collected (INT, input).
 * zerofill = How many zeros to fill in before FFT (int, input).
 * upper    = upper state potential grid (rgrid3d *, input).
 * lower    = lower state potential grdi (rgrid3d *, input).
 *
 * Returns difference potential for dynamics.
 */

EXPORT rgrid3d *dft_driver_spectrum_init2(INT nt, INT zf, rgrid3d *upper, rgrid3d *lower) {

  cur_time = 0;
  ntime = nt;
  zerofill = zf;
  if(!tdpot)
    tdpot = cgrid1d_alloc(ntime + zf, 0.1, CGRID1D_PERIODIC_BOUNDARY, 0);
  if(upper == NULL) return NULL; /* not given */
  if(!xxdiff)
    xxdiff = rgrid3d_alloc(driver_nx, driver_ny, driver_nz, driver_step, RGRID3D_PERIODIC_BOUNDARY, 0, "xxdiff");
  if(!xxave)
    xxave = rgrid3d_alloc(driver_nx, driver_ny, driver_nz, driver_step, RGRID3D_PERIODIC_BOUNDARY, 0, "xxave");
  rgrid3d_difference(xxdiff, upper, lower);
  rgrid3d_sum(xxave, workspace1, workspace2);
  rgrid3d_multiply(xxave, 0.5);
  return xxave;
}

/*
 * Collect the difference energy data (user calculated).
 *
 * val = difference energy value to be inserted (input, REAL).
 *
 */

EXPORT void dft_driver_spectrum_collect_user(REAL val) {

  if(cur_time > ntime) {
    fprintf(stderr, "libdft: initialized with too few points (spectrum collect).\n");
    exit(1);
  }
  tdpot->value[cur_time] = val;

  fprintf(stderr, "libdft: spectrum collect complete (point = " FMT_I ", value = " FMT_R " K).\n", cur_time, CREAL(tdpot->value[cur_time]) * GRID_AUTOK);
  cur_time++;
}

/*
 * Collect the difference energy data. 
 *
 * gwf     = the current wavefunction (used for calculating the liquid density) (wf3d *, input).
 *
 */

EXPORT void dft_driver_spectrum_collect(wf3d *gwf) {

  if(cur_time > ntime) {
    fprintf(stderr, "libdft: initialized with too few points (spectrum collect).\n");
    exit(1);
  }
  grid3d_wf_density(gwf, workspace1);
  rgrid3d_product(workspace1, workspace1, xxdiff);
  tdpot->value[cur_time] = rgrid3d_integral(workspace1);

  fprintf(stderr, "libdft: spectrum collect complete (point = " FMT_I ", value = " FMT_R " K).\n", cur_time, CREAL(tdpot->value[cur_time]) * GRID_AUTOK);
  cur_time++;
}

/*
 * Evaluate the spectrum.
 *
 * tstep       = Time step length at which the energy difference data was collected
 *               (fs; usually the simulation time step) (REAL, input).
 * tc          = Exponential decay time constant (fs; REAL, input).
 *
 * Returns a pointer to the calculated spectrum (grid1d *). X-axis in cm$^{-1}$.
 *
 */

EXPORT cgrid1d *dft_driver_spectrum_evaluate(REAL tstep, REAL tc) {

  INT t, npts;
  static cgrid1d *spectrum = NULL;

  if(cur_time > ntime) {
    printf(FMT_I " " FMT_I "\n", cur_time, ntime);
    fprintf(stderr, "libdft: cur_time >= ntime. Increase ntime.\n");
    exit(1);
  }

  tstep /= GRID_AUTOFS;
  tc /= GRID_AUTOFS;
  npts = 2 * (cur_time + zerofill - 1);
  if(!spectrum)
    spectrum = cgrid1d_alloc(npts, GRID_HZTOCM1 / (tstep * GRID_AUTOFS * 1E-15 * ((REAL) npts)), CGRID1D_PERIODIC_BOUNDARY, 0);

#define SEMICLASSICAL /* */
  
#ifndef SEMICLASSICAL
  /* P(t) - full expression - see the Eloranta/Apkarian CPL paper on lineshapes */
  /* Instead of propagating the liquid on the excited state, it is run on the average (V_e + V_g)/2 potential */
  spectrum->value[0] = 0.0;
  fprintf(stderr, "libdft: Polarization at time 0 fs = 0.\n");
  for (t = 1; t < cur_time; t++) {
    REAL tmp;
    REAL complex tmp2;
    INT tp, tpp;
    tmp2 = 0.0;
    for(tp = 0; tp < t; tp++) {
      tmp = 0.0;
      for(tpp = tp; tpp < t; tpp++)
	tmp += tdpot->value[tpp] * tstep;
      tmp2 += CEXP(-(I / HBAR) * tmp) * tstep;
    }
    spectrum->value[t] = -2.0 * CIMAG(tmp2) * EXP(-t * tstep / tc);
    fprintf(stderr, "libdft: Polarization at time " FMT_R " fs = %le.\n", t * tstep * GRID_AUTOFS, CREAL(spectrum->value[t]));
    spectrum->value[npts - t] = -spectrum->value[t];
  }
#else
  { REAL complex last, tmp;
    REAL ct;
    /* This seems to perform poorly - not in use */
    /* Construct semiclassical dipole autocorrelation function */
    last = (REAL) tdpot->value[0];
    tdpot->value[0] = 0.0;
    for(t = 1; t < cur_time; t++) {
      tmp = tdpot->value[t];
      tdpot->value[t] = tdpot->value[t-1] + tstep * (last + tmp)/2.0;
      last = tmp;
    }
    
    if(tc < 0.0) tc = -tc;
    spectrum->value[0] = 1.0;
    for (t = 1; t < cur_time; t++) {
      ct = tstep * (REAL) t;
      spectrum->value[t] = CEXP(-I * tdpot->value[t] / HBAR - ct / tc); /* transition dipole moment = 1 */
      spectrum->value[npts - t] = CEXP(I * tdpot->value[t] / HBAR - ct / tc); /* last point rolled over */
    }
  }
#endif
  
  /* zero fill */
  for (t = cur_time; t < cur_time + zerofill; t++)
    spectrum->value[t] = spectrum->value[npts - t] = 0.0;

  /* flip zero frequency to the middle */
  for (t = 0; t < 2 * (cur_time + zerofill - 1); t++)
    spectrum->value[t] *= POW(-1.0, (REAL) t);
  
  cgrid1d_inverse_fft(spectrum);

  /* Make the spectrum appear in the real part rather than imaginary */
#ifndef SEMI
  for(t = 0; t < npts; t++)
    spectrum->value[t] *= -I;
#endif
  
  return spectrum;
}

/*
 * Evaluate the liquid velocity field for a given order paremeter (X component),
 * $v = \vec{J}/\rho$.
 *
 * gwf  = Order parameter for which the velocity field is evaluated (input; wf3d *).
 * vx   = Velocity field x component (output; rgrid3d *).
 * eps  = Epsilon to add to rho when dividing (input; REAL).
 *
 */

EXPORT void dft_driver_veloc_field_x_eps(wf3d *wf, rgrid3d *vx, REAL eps) {

  grid3d_wf_probability_flux_x(wf, vx);
  grid3d_wf_density(wf, workspace1);
  rgrid3d_division_eps(vx, vx, workspace1, eps);
}

/*
 * Evaluate the liquid velocity field for a given order paremeter (Y component),
 * $v = \vec{J}/\rho$.
 *
 * gwf  = Order parameter for which the velocity field is evaluated (input; wf3d *).
 * vy   = Velocity field y component (output; rgrid3d *).
 * eps  = Epsilon to add to rho when dividing (inputl REAL).
 *
 */

EXPORT void dft_driver_veloc_field_y_eps(wf3d *wf, rgrid3d *vy, REAL eps) {

  grid3d_wf_probability_flux_y(wf, vy);
  grid3d_wf_density(wf, workspace1);
  rgrid3d_division_eps(vy, vy, workspace1, eps);
}

/*
 * Evaluate the liquid velocity field for a given order paremeter (Z compinent),
 * $v = \vec{J}/\rho$.
 *
 * gwf  = Order parameter for which the velocity field is evaluated (input; wf3d *).
 * vz   = Velocity field z component (output; rgrid3d *).
 * eps  = Epsilon to add to rho when dividing (input; REAL).
 *
 */

EXPORT void dft_driver_veloc_field_z_eps(wf3d *wf, rgrid3d *vz, REAL eps) {

  grid3d_wf_probability_flux_z(wf, vz);
  grid3d_wf_density(wf, workspace1);
  rgrid3d_division_eps(vz, vz, workspace1, eps);
}

/*
 * Evaluate the liquid velocity field for a given order paremeter,
 * $v = \vec{J}/\rho$.
 *
 * gwf  = Order parameter for which the velocity field is evaluated (input; wf3d *).
 * vx    = Velocity field x component (output; rgrid3d *).
 * vy    = Velocity field y component (output; rgrid3d *).
 * vz    = Velocity field z component (output; rgrid3d *).
 * eps   = Epsilon to add to rho when dividing (input; REAL).
 *
 */

EXPORT void dft_driver_veloc_field_eps(wf3d *wf, rgrid3d *vx, rgrid3d *vy, rgrid3d *vz, REAL eps) {

  grid3d_wf_probability_flux(wf, vx, vy, vz);
  grid3d_wf_density(wf, workspace1);
  rgrid3d_division_eps(vx, vx, workspace1, eps);
  rgrid3d_division_eps(vy, vy, workspace1, eps);
  rgrid3d_division_eps(vz, vz, workspace1, eps);
}

/*
 * Evaluate the liquid velocity field for a given order paremeter (X component),
 * $v = \vec{J}/\rho$.
 *
 * gwf  = Order parameter for which the velocity field is evaluated (input; wf3d *).
 * vx    = Velocity field x component (output; rgrid3d *).
 *
 * Note: This routine caps the maximum liquid velocity using
 *       DFT_VELOC_EPS.
 *
 */

EXPORT void dft_driver_veloc_field_x(wf3d *wf, rgrid3d *vx) {

  dft_driver_veloc_field_x_eps(wf, vx, DFT_VELOC_EPS);
}

/*
 * Evaluate the liquid velocity field for a given order paremeter (Y component),
 * $v = \vec{J}/\rho$.
 *
 * gwf  = Order parameter for which the velocity field is evaluated (input; wf3d *).
 * vy    = Velocity field y component (output; rgrid3d *).
 *
 * Note: This routine caps the maximum liquid velocity using
 *       DFT_VELOC_EPS.
 *
 */

EXPORT void dft_driver_veloc_field_y(wf3d *wf, rgrid3d *vy) {

  dft_driver_veloc_field_y_eps(wf, vy, DFT_VELOC_EPS);
}

/*
 * Evaluate the liquid velocity field for a given order paremeter (Z compinent),
 * $v = \vec{J}/\rho$.
 *
 * gwf  = Order parameter for which the velocity field is evaluated (input; wf3d *).
 * vz    = Velocity field z component (output; rgrid3d *).
 *
 * Note: This routine caps the maximum liquid velocity using
 *       DFT_VELOC_EPS.
 *
 */

EXPORT void dft_driver_veloc_field_z(wf3d *wf, rgrid3d *vz) {

  dft_driver_veloc_field_z_eps(wf, vz, DFT_VELOC_EPS);
}

/*
 * Evaluate the liquid velocity field for a given order paremeter,
 * $v = \vec{J}/\rho$.
 *
 * gwf  = Order parameter for which the velocity field is evaluated (input; wf3d *).
 * vx    = Velocity field x component (output; rgrid3d *).
 * vy    = Velocity field y component (output; rgrid3d *).
 * vz    = Velocity field z component (output; rgrid3d *).
 *
 * Note: This routine caps the maximum liquid velocity using
 *       DFT_VELOC_EPS.
 *
 */

EXPORT void dft_driver_veloc_field(wf3d *wf, rgrid3d *vx, rgrid3d *vy, rgrid3d *vz) {

  dft_driver_veloc_field_eps(wf, vx, vy, vz, DFT_VELOC_EPS);
}

/*
 * Evaluate liquid momentum according to:
 * $\int\rho\v_idr$ where $i = x,y,z$.
 *
 * wf = Order parameter for evaluation (wf3d *; input).
 * px = Liquid momentum along x (REAL *; output).
 * py = Liquid momentum along y (REAL *; output).
 * pz = Liquid momentum along z (REAL *; output).
 *
 */

EXPORT void dft_driver_P(wf3d *wf, REAL *px, REAL *py, REAL *pz) {

  grid3d_wf_probability_flux(wf, workspace1, workspace2, workspace3);
  rgrid3d_multiply(workspace1, wf->mass);
  rgrid3d_multiply(workspace2, wf->mass);
  rgrid3d_multiply(workspace3, wf->mass);

  *px = rgrid3d_integral(workspace1);
  *py = rgrid3d_integral(workspace2);
  *pz = rgrid3d_integral(workspace3);
}

/*
 * Evaluate liquid momentum according to:
 * $\int\rho\v_idr$ where $i = x$.
 *
 * wf = Order parameter for evaluation (wf3d *; input).
 *
 * Returns px (momentum along x).
 *
 */

EXPORT REAL dft_driver_Px(wf3d *wf) {

  grid3d_wf_probability_flux_x(wf, workspace1);

  return wf->mass * rgrid3d_integral(workspace1);
}

/*
 * Evaluate liquid momentum according to:
 * $\int\rho\v_idr$ where $i = y$.
 *
 * wf = Order parameter for evaluation (wf3d *; input).
 *
 * Returns py (momentum along y).
 *
 */

EXPORT REAL dft_driver_Py(wf3d *wf) {

  grid3d_wf_probability_flux_y(wf, workspace1);
  rgrid3d_multiply(workspace1, wf->mass);

  return rgrid3d_integral(workspace1);
}

/*
 * Evaluate liquid momentum according to:
 * $\int\rho\v_idr$ where $i = z$.
 *
 * wf = Order parameter for evaluation (wf3d *; input).
 *
 * Returns pz (momentum along z).
 *
 */

EXPORT REAL dft_driver_Pz(wf3d *wf) {

  grid3d_wf_probability_flux_z(wf, workspace1);
  rgrid3d_multiply(workspace1, wf->mass);

  return rgrid3d_integral(workspace1);
}

/*
 * Evaluate liquid kinetic energy according to:
 * $\frac{1}{2}m_{He}\int\rho v^2dr$
 *
 * wf = Order parameter for evaluation (wf3d *; input).
 *
 * Returns the kinetic energy.
 *
 */

EXPORT REAL dft_driver_KE(wf3d *wf) {

  dft_driver_veloc_field(wf, workspace1, workspace2, workspace3);
  rgrid3d_product(workspace1, workspace1, workspace1);
  rgrid3d_product(workspace2, workspace2, workspace2);
  rgrid3d_product(workspace3, workspace3, workspace3);
  rgrid3d_sum(workspace1, workspace1, workspace2);
  rgrid3d_sum(workspace1, workspace1, workspace3);
  grid3d_wf_density(wf, workspace2);
  rgrid3d_product(workspace1, workspace1, workspace2);
  rgrid3d_multiply(workspace1, wf->mass / 2.0);
  return rgrid3d_integral(workspace1);
}

/*
 * Evaluate angular momentum about the origin (center of the grid).
 *
 * wf = Order parameter for evaluation (wf3d *; input).
 * lx = Anuglar momentum x component (REAL *; output).
 * ly = Anuglar momentum y component (REAL *; output).
 * lz = Anuglar momentum z component (REAL *; output).
 *
 */

static REAL origin_x = 0.0, origin_y = 0.0, origin_z = 0.0;

static REAL mult_mx(void *xx, REAL x, REAL y, REAL z) {

  rgrid3d *grid = (rgrid3d *) xx;

  return -rgrid3d_value(grid, x, y, z) * (x - origin_x);
}

static REAL mult_my(void *xx, REAL x, REAL y, REAL z) {

  rgrid3d *grid = (rgrid3d *) xx;

  return -rgrid3d_value(grid, x, y, z) * (y - origin_y);
}

static REAL mult_mz(void *xx, REAL x, REAL y, REAL z) {

  rgrid3d *grid = (rgrid3d *) xx;

  return -rgrid3d_value(grid, x, y, z) * (z - origin_z);
}

static REAL mult_x(void *xx, REAL x, REAL y, REAL z) {

  rgrid3d *grid = (rgrid3d *) xx;

  return rgrid3d_value(grid, x, y, z) * (x - origin_x);
}

static REAL mult_y(void *xx, REAL x, REAL y, REAL z) {

  rgrid3d *grid = (rgrid3d *) xx;

  return rgrid3d_value(grid, x, y, z) * (y - origin_y);
}

static REAL mult_z(void *xx, REAL x, REAL y, REAL z) {

  rgrid3d *grid = (rgrid3d *) xx;

  return rgrid3d_value(grid, x, y, z) * (z - origin_z);
}

/*
 * Calculate angular momentum expectation values * particle mass.
 *
 * wf = Wavefunction (gwf *).
 * lx = X component of angular momentum * mass (REAL *).
 * ly = Y component of angular momentum * mass (REAL *).
 * lz = Z component of angular momentum * mass (REAL *).
 *
 */
 
EXPORT void dft_driver_L(wf3d *wf, REAL *lx, REAL *ly, REAL *lz) {

  rgrid3d *px = workspace4, *py = workspace5, *pz = workspace6;
  
  if(!workspace7) workspace7 = dft_driver_alloc_rgrid("DR workspace7");
  if(!workspace8) workspace8 = dft_driver_alloc_rgrid("DR workspace8");

  origin_x = wf->grid->x0;
  origin_y = wf->grid->y0;
  origin_z = wf->grid->z0;

  grid3d_wf_probability_flux(wf, px, py, pz);

  // Lx
  rgrid3d_map(workspace7, mult_mz, py);      // -z*p_y
  rgrid3d_map(workspace8, mult_y, pz);       // y*p_z
  rgrid3d_sum(workspace7, workspace7, workspace8);
  *lx = rgrid3d_integral(workspace7) * wf->mass;

  // Ly
  rgrid3d_map(workspace7, mult_mx, pz);      // -x*p_z
  rgrid3d_map(workspace8, mult_z, px);       // z*p_x
  rgrid3d_sum(workspace7, workspace7, workspace8);
  *ly = rgrid3d_integral(workspace7) * wf->mass;

  // Lz
  rgrid3d_map(workspace7, mult_my, px);      // -y*p_x
  rgrid3d_map(workspace8, mult_x, py);       // x*p_y
  rgrid3d_sum(workspace7, workspace7, workspace8);
  *lz = rgrid3d_integral(workspace7) * wf->mass;
}

/*
 * Produce radially averaged density from a 3-D grid.
 * 
 * radial = Radial density (rgrid1d *; output).
 * grid   = Source grid (rgrid3d *; input).
 * dtheta = Integration step size along theta in radians (REAL; input).
 * dphi   = Integration step size along phi in radians (REAL, input).
 * xc     = x coordinate for the center (REAL; input).
 * yc     = y coordinate for the center (REAL; input).
 * zc     = z coordinate for the center (REAL; input).
 *
 */

EXPORT void dft_driver_radial(rgrid1d *radial, rgrid3d *grid, REAL dtheta, REAL dphi, REAL xc, REAL yc, REAL zc) {
  
  REAL r, theta, phi;
  REAL x, y, z, step = radial->step, tmp;
  INT ri, thetai, phii, nx = radial->nx;

  r = 0.0;
  for (ri = 0; ri < nx; ri++) {
    tmp = 0.0;
#pragma omp parallel for firstprivate(xc,yc,zc,grid,r,dtheta,dphi,step) private(theta,phi,x,y,z,thetai,phii) reduction(+:tmp) default(none) schedule(runtime)
    for (thetai = 0; thetai <= (INT) (M_PI / dtheta); thetai++)
      for (phii = 0; phii <= (INT) (2.0 * M_PI / dtheta); phii++) {
	theta = ((REAL) thetai) * dtheta;
	phi = ((REAL) phii) * dphi;
	x = r * COS(phi) * SIN(theta) + xc;
	y = r * SIN(phi) * SIN(theta) + yc;
	z = r * COS(theta) + zc;
	tmp += rgrid3d_value(grid, x, y, z) * SIN(theta);
      }
    tmp *= dtheta * dphi / (4.0 * M_PI);
    radial->value[ri] = tmp;
    r += step;
  }
}

/*
 * Produce radially averaged complex grid from a 3-D grid.
 *
 * radial = Radial density (cgrid1d *; output).
 * grid   = Source grid (cgrid3d *; input).
 * dtheta = Integration step size along theta in radians (REAL; input).
 * dphi   = Integration step size along phi in radians (REAL, input).
 * xc     = x coordinate for the center (REAL; input).
 * yc     = y coordinate for the center (REAL; input).
 * zc     = z coordinate for the center (REAL; input).
 *
 */

EXPORT void dft_driver_radial_complex(cgrid1d *radial, cgrid3d *grid, REAL dtheta, REAL dphi, REAL xc, REAL yc, REAL zc) {
  
  REAL r, theta, phi;
  REAL x, y, z, step = radial->step;
  REAL complex tmp;
  INT ri, thetai, phii, nx = radial->nx;

  r = 0.0;
  for (ri = 0; ri < nx; ri++) {
    tmp = 0.0;
#pragma omp parallel for firstprivate(xc,yc,zc,grid,r,dtheta,dphi,step) private(theta,phi,x,y,z,thetai,phii) reduction(+:tmp) default(none) schedule(runtime)
    for (thetai = 0; thetai <= (INT) (M_PI / dtheta); thetai++)
      for (phii = 0; phii <= (INT) (2.0 * M_PI / dtheta); phii++) {
	theta = ((REAL) thetai) * dtheta;
	phi = ((REAL) phii) * dphi;
	x = r * COS(phi) * SIN(theta) + xc;
	y = r * SIN(phi) * SIN(theta) + yc;
	z = r * COS(theta) + zc;
	tmp += cgrid3d_value(grid, x, y, z) * SIN(theta);
      }
    tmp *= dtheta * dphi / (4.0 * M_PI);
    radial->value[ri] = tmp;
    r += step;
  }
}

/*
 * Calculate the spherical radius R_b and bubble center given the density.
 * Note: R_b = (3 N_{disp} / (4 \pi \rho_0))^{1/3} is equivalent to the
 * integral definition of R_b. 
 *
 * density = liquid density (rgrid3d *; input).
 *
 * Return value: R_b.
 *
 */

EXPORT REAL dft_driver_spherical_rb(rgrid3d *density) {

  REAL disp;

  rgrid3d_multiply(density, -1.0);
  rgrid3d_add(density, driver_rho0);
  disp = rgrid3d_integral(density);
  rgrid3d_add(density, -driver_rho0);
  rgrid3d_multiply(density, -1.0);

  return POW(disp * 3.0 / (4.0 * M_PI * driver_rho0), 1.0 / 3.0);
}

/*
 * Calculate convergence norm:
 * \int \rho(r,t) - \rho(r,t - \Delta t) d^3r.
 *
 * density = Current density (rgrid3d *, input).
 *
 * Return value: norm.
 *
 * Note: This must be called during every iteration. For the first iteration
 *       this always returns 1.0.
 *
 * It is often a good idea to aim at "zero" - especially if real time dynamics
 * will be run afterwards.
 *
 * workspace = density from previous iteration.
 *
 */

EXPORT REAL dft_driver_norm(rgrid3d *density, rgrid3d *workspace) {

  static char been_here = 0;
  REAL tmp;

  if(!been_here) {
    rgrid3d_copy(workspace, density);
    return 1.0;
  }

  tmp = rgrid3d_max(density);
  
  rgrid3d_copy(workspace, density);

  return tmp;
}

/*
 * Force spherical symmetry by spherical averaging.
 *
 * wf = wavefunction to be averaged (wf3d *).
 * xc = x coordinate for the center (REAL).
 * yc = y coordinate for the center (REAL).
 * zc = z coordinate for the center (REAL).
 *
 */

EXPORT void dft_driver_force_spherical(wf3d *wf, REAL xc, REAL yc, REAL zc) {

  INT i, j, l, k, len;
  INT nx = wf->grid->nx, ny = wf->grid->ny, nz = wf->grid->nz;
  REAL step = wf->grid->step;
  REAL x, y, z, x2, y2, z2, d;
  cgrid1d *average;
  REAL complex *avalue;

  if(nx > ny) len = nx; else len = ny;
  if(nz > len) len = nz;
  average = cgrid1d_alloc(len, step, CGRID1D_PERIODIC_BOUNDARY, 0);
  avalue = average->value;
  dft_driver_radial_complex(average, wf->grid, 0.01, 0.01, xc, yc, zc);

  /* Write spherical average back to wf */
  for(i = 0; i < nx; i++) {
    x = ((REAL) (i - nx/2)) * step;
    x2 = (x - xc) * (x - xc);
    for (j = 0; j < ny; j++) {
      y = ((REAL) (j - ny/2)) * step;
      y2 = (y - yc) * (y - yc);
      for (l = 0; l < nz; l++) {
	z = ((REAL) (l - nz/2)) * step;
	z2 = (z - zc) * (z - zc);
	d = SQRT(x2 + y2 + z2);
	k = (INT) (0.5 + d / step);
	if(k >= len) k = len - 1;
        cgrid3d_value_to_index(wf->grid, i, j, k, avalue[k]);
      }
    }
  }
  cgrid1d_free(average);
}

#define R_M 0.05

static REAL complex vortex_x_n1(void *na, REAL x, REAL y, REAL z) {

  REAL d = SQRT(y * y + z * z);

  if(d < R_M) return 0.0;
  return (y + I * z) / d;
}

static REAL complex vortex_y_n1(void *na, REAL x, REAL y, REAL z) {

  REAL d = SQRT(x * x + z * z);

  if(d < R_M) return 0.0;
  return (x + I * z) / d;
}

static REAL complex vortex_z_n1(void *na, REAL x, REAL y, REAL z) {

  REAL d = SQRT(x * x + y * y);

  if(d < R_M) return 0.0;
  return (x + I * y) / d;
}

static REAL complex vortex_x_n2(void *na, REAL x, REAL y, REAL z) {

  REAL y2 = y * y, z2 = z * z;
  REAL d = SQRT(x * x + y * y);
  
  if(d < R_M) return 0.0;
  return ((y2 - z2) + I * 2 * y * z) / (y2 + z2);
}

static REAL complex vortex_y_n2(void *na, REAL x, REAL y, REAL z) {

  REAL x2 = x * x, z2 = z * z;
  REAL d = SQRT(x * x + y * y);
  
  if(d < R_M) return 0.0;  
  return ((x2 - z2) + I * 2 * x * z) / (x2 + z2);
}

static REAL complex vortex_z_n2(void *na, REAL x, REAL y, REAL z) {

  REAL x2 = x * x, y2 = y * y;
  REAL d = SQRT(x * x + y * y);
  
  if(d < R_M) return 0.0;  
  return ((x2 - y2) + I * 2 * x * y) / (x2 + y2);
}

/*
 * Modify a given wavefunction to have vorticity around a specified axis.
 *
 * gwf    = Wavefunction for the operation (input/output, gwf3d *).
 * n      = Quantum number (1 or 2) (input, int).
 * 
 */

EXPORT void dft_driver_vortex_initial(wf3d *gwf, int n, int axis) {

  if(!cworkspace)
    cworkspace = dft_driver_alloc_cgrid("DR cworkspace");
  
  if(axis == DFT_DRIVER_VORTEX_X) {
    switch(n) {
    case 1:
      cgrid3d_map(cworkspace, vortex_x_n1, NULL);
      cgrid3d_product(gwf->grid, gwf->grid, cworkspace);      
      break;
    case 2:
      cgrid3d_map(cworkspace, vortex_x_n2, NULL);
      cgrid3d_product(gwf->grid, gwf->grid, cworkspace);      
      break;
    default:
      fprintf(stderr,"libdft: Illegal value for n (dft_driver_vortex_initial()).\n");
      break;
    }
    return;
  }

  if(axis == DFT_DRIVER_VORTEX_Y) {
    switch(n) {
    case 1:
      cgrid3d_map(cworkspace, vortex_y_n1, NULL);
      cgrid3d_product(gwf->grid, gwf->grid, cworkspace);      
      break;
    case 2:
      cgrid3d_map(cworkspace, vortex_y_n2, NULL);
      cgrid3d_product(gwf->grid, gwf->grid, cworkspace);      
      break;
    default:
      fprintf(stderr,"libdft: Illegal value for n (dft_driver_vortex_initial()).\n");
      break;
    }
    return;
  }

  if(axis == DFT_DRIVER_VORTEX_Z) {
    switch(n) {
    case 1:
      cgrid3d_map(cworkspace, vortex_z_n1, NULL);
      cgrid3d_product(gwf->grid, gwf->grid, cworkspace);      
      break;
    case 2:
      cgrid3d_map(cworkspace, vortex_z_n2, NULL);
      cgrid3d_product(gwf->grid, gwf->grid, cworkspace);      
      break;
    default:
      fprintf(stderr,"libdft: Illegal value for n (dft_driver_vortex_initial()).\n");
      break;
    }
    return;
  }
  fprintf(stderr, "libdft: Illegal axis for dft_driver_vortex_initial().\n");
  exit(1);
}

/*
 * Add vortex potential (Feynman-Onsager ansatz) along a specified axis.
 *
 * potential = Potential grid where the vortex potential is added (rgrid3d *, input/output).
 * direction = Along which axis the vortex potential is added (int, input):
 *             DFT_DRIVER_VORTEX_{X,Y,Z}.
 *
 */

static REAL vortex_x(void *na, REAL x, REAL y, REAL z) {

  REAL rp2 = y * y + z * z;

  if(rp2 < R_M * R_M) rp2 = R_M * R_M;
  return 1.0 / (2.0 * dft_driver_otf->mass * rp2);
}

static REAL vortex_y(void *na, REAL x, REAL y, REAL z) {

  REAL rp2 = x * x + z * z;

  if(rp2 < R_M * R_M) rp2 = R_M * R_M;
  return 1.0 / (2.0 * dft_driver_otf->mass * rp2);
}

static REAL vortex_z(void *na, REAL x, REAL y, REAL z) {

  REAL rp2 = x * x + y * y;

  if(rp2 < R_M * R_M) rp2 = R_M * R_M;
  return 1.0 / (2.0 * dft_driver_otf->mass * rp2);
}

EXPORT void dft_driver_vortex(rgrid3d *potential, int direction) {

  switch(direction) {
  case DFT_DRIVER_VORTEX_X:
    rgrid3d_map(workspace6, vortex_x, NULL);
    rgrid3d_sum(potential, potential, workspace6);
    break;
  case DFT_DRIVER_VORTEX_Y:
    rgrid3d_map(workspace6, vortex_y, NULL);
    rgrid3d_sum(potential, potential, workspace6);
    break;
  case DFT_DRIVER_VORTEX_Z:
    rgrid3d_map(workspace6, vortex_z, NULL);
    rgrid3d_sum(potential, potential, workspace6);
    break;
  default:
    fprintf(stderr, "libdft: Unknown axis direction for vortex potential (dft_driver_vortex()).\n");
    exit(1);
  }
}

/*
 * This routine will zero a given wavefunction at points where the given potential exceeds the specified limit.
 *
 * gwf       = Wavefunction to be operated on (wf3d *).
 * potential = Potential that determines the points to be zeroed (rgrid3d *).
 * ul        = Limit for the potential above which the wf will be zeroed (REAL).
 * 
 * Application: Sometimes there are regions where the potential is very high
 *              but numerical instability begins to increase the wf amplitude there
 *              leading to the calculation exploding.
 *
 */

EXPORT void dft_driver_clear(wf3d *gwf, rgrid3d *potential, REAL ul) {

  INT i, j, k;

  for(i = 0; i < potential->nx; i++)
    for(j = 0; j < potential->ny; j++)
      for(k = 0; k < potential->nz; k++)
        if(rgrid3d_value_at_index(potential, i, j, k) >= ul) cgrid3d_value_to_index(gwf->grid, i, j, k, 0.0);
}

/*
 * This routine will limit the given potential exceeds the specified max value.
 *
 * potential = Potential that determines the points to be zeroed (rgrid3d *).
 * ul        = Limit for the potential above which the wf will be zeroed (REAL).
 * ll        = Limit for the potential below which the wf will be zeroed (REAL).
 * 
 */

EXPORT void dft_driver_clear_pot(rgrid3d *potential, REAL ul, REAL ll) {

  INT i, j, k;
  REAL tmp;

  for(i = 0; i < potential->nx; i++)
    for(j = 0; j < potential->ny; j++)
      for(k = 0; k < potential->nz; k++) {
        tmp = rgrid3d_value_at_index(potential, i, j, k);
        if(tmp > ul) rgrid3d_value_to_index(potential, i, j, k, ul);
        if(tmp < ll) rgrid3d_value_to_index(potential, i, j, k, ll);
      }
}

/*
 * Zero part of a given grid based on a given density treshold.
 *
 */

EXPORT void dft_driver_clear_core(rgrid3d *grid, rgrid3d *density, REAL thr) {

  INT i, j, k;

  for(i = 0; i < grid->nx; i++)
    for(j = 0; j < grid->ny; j++)
      for(k = 0; k < grid->nz; j++)
        if(rgrid3d_value_at_index(density, i, j, k) < thr) rgrid3d_value_to_index(grid, i, j, k, 0.0);
}

/*
 * Calculate running average to smooth unwanted high freq. components.
 *
 * dest   = destination grid (rgrid3d *).
 * source = source grid (rgrid3d *).
 * npts   = number of points used in running average (int). This smooths over +-npts points (effectively 2 X npts).
 *
 * No return value.
 *
 * Note: dest and source cannot be the same array.
 * 
 */

EXPORT void dft_driver_npoint_smooth(rgrid3d *dest, rgrid3d *source, int npts) {

  INT i, ip, j, jp, k, kp, nx = source->nx, ny = source->ny, nz = source->nz, pts;
  INT li, ui, lj, uj, lk, uk;
  REAL ave;

  if(npts < 2) {
    rgrid3d_copy(dest, source);
    return; /* nothing to do */
  }
  if(dest == source) {
    fprintf(stderr, "libdft: dft_driver_npoint_smooth() - dest and source cannot be equal.\n");
    exit(1);
  }

  for (i = 0; i < nx; i++) 
    for (j = 0; j < ny; j++) 
      for (k = 0; k < nz; k++) {
        ave = 0.0;
        pts = 0;
        if(i - npts < 0) li = 0; else li = i - npts;
        if(j - npts < 0) lj = 0; else lj = j - npts;
        if(k - npts < 0) lk = 0; else lk = k - npts;
        if(i + npts > nx) ui = nx; else ui = i + npts;
        if(j + npts > ny) uj = ny; else uj = j + npts;
        if(k + npts > nz) uk = nz; else uk = k + npts;
        for(ip = li; ip < ui; ip++)
          for(jp = lj; jp < uj; jp++)
            for(kp = lk; kp < uk; kp++) {
              pts++;
              ave += rgrid3d_value_at_index(source, ip, jp, kp);
            }
        ave /= (REAL) pts;
        rgrid3d_value_to_index(dest, i, j, k, ave);
      }
}

/*
 * Allow outside access to workspaces.
 *
 * w     = workspace # requested (char; input).
 * alloc = 1: allocate workspace if not allocated, 0 = do not allocate if not already allocated (char; input).
 *
 * Return value: Pointer to the workspace (rgrid3d *) or NULL if invalid workspace number requested.
 *
 * workspace1 - workspace9: used during propagation (evaluation of OT potential). No need to (and will not be) preserve between predict/correct (rgrid3d).
 * cworkspace             : used during propagation (Crank-Nicolson KE propagation and OT potential evaluation). No need to (and will not be) preserve between predict/correct (cgrid3d).
 * workspace10            : density storage (rgrid3d). Same applies as for the above.
 * All space is safe to use everywhere (but will be overwritten by either predict/correct propagation calls or possibly other driver3d.c functions).
 * 
 * Returns NULL if the requrested work space has not been allocated.
 *
 */

EXPORT void *dft_driver_get_workspace(char w, char alloc) {

  if (w < 0 || w > 10) return NULL;
  switch(w) {
    case 0:
      if(!cworkspace && alloc) cworkspace = dft_driver_alloc_cgrid("DR cworkspace");
      return (void *) cworkspace;
    case 1:
      if(!workspace1 && alloc) workspace1 = dft_driver_alloc_rgrid("DR workspace1");
      return (void *) workspace1;
    case 2:
      if(!workspace2 && alloc) workspace2 = dft_driver_alloc_rgrid("DR workspace2");
      return (void *) workspace2;
    case 3:
      if(!workspace3 && alloc) workspace3 = dft_driver_alloc_rgrid("DR workspace3");
      return (void *) workspace3;
    case 4:
      if(!workspace4 && alloc) workspace4 = dft_driver_alloc_rgrid("DR workspace4");
      return (void *) workspace4;
    case 5:
      if(!workspace5 && alloc) workspace5 = dft_driver_alloc_rgrid("DR workspace5");
      return (void *) workspace5;
    case 6:
      if(!workspace6 && alloc) workspace6 = dft_driver_alloc_rgrid("DR workspace6");
      return (void *) workspace6;
    case 7:
      if(!workspace7 && alloc) workspace7 = dft_driver_alloc_rgrid("DR workspace7");
      return (void *) workspace7;
    case 8:
      if(!workspace8 && alloc) workspace8 = dft_driver_alloc_rgrid("DR workspace8");
      return (void *) workspace8;
    case 9:
      if(!workspace9 && alloc) workspace9 = dft_driver_alloc_rgrid("DR workspace9");
      return (void *) workspace9;
    case 10:
      if(!density && alloc) density = dft_driver_alloc_rgrid("DR density");
      return (void *) density;
   }
   return NULL;
}
