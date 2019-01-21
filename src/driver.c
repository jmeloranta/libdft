/*
 * Simple driver routines to propagate the liquid.
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
char dft_driver_init_ot = 1; /* Init OT ? */
int dft_driver_temp_disable_other_normalization = 0;

INT dft_driver_nx = 0, dft_driver_ny = 0, dft_driver_nz = 0, dft_driver_nx2 = 0, dft_driver_ny2 = 0, dft_driver_nz2 = 0;
static INT driver_threads = 0, driver_dft_model = 0, driver_iter_mode = 0, driver_boundary_type = 0;
static INT driver_norm_type = 0, driver_nhe = 0, center_release = 0, driver_rels = 0;
static INT driver_bc_lx = 0, driver_bc_hx = 0, driver_bc_ly = 0, driver_bc_hy = 0, driver_bc_lz = 0, driver_bc_hz = 0;
static char driver_bc = 0;
static REAL driver_frad = 0.0, driver_omega = 0.0, driver_bc_amp = 1.0;
static REAL viscosity = 0.0, viscosity_alpha = 1.0;
REAL dft_driver_step = 0.0, dft_driver_rho0 = 0.0;
static REAL driver_x0 = 0.0, driver_y0 = 0.0, driver_z0 = 0.0;
static REAL driver_kx0 = 0.0, driver_ky0 = 0.0, driver_kz0 = 0.0;
static grid_timer timer;
static REAL driver_rmin = -1.0, driver_radd = 0.0, driver_a0 = 0.0, driver_a1 = 0.0, driver_a2 = 0.0, driver_a3 = 0.0, driver_a4 = 0.0, driver_a5 = 0.0;
cgrid *potential = NULL;

/*
 * Wave function normalization (for imaginary time).
 *
 */

inline static void scale_wf(char what, wf *gwf) {

  INT i, j, k;
  REAL x, y, z;
  REAL complex norm;

  if(what >= DFT_DRIVER_PROPAGATE_OTHER) { /* impurity */
    if(!dft_driver_temp_disable_other_normalization) grid_wf_normalize(gwf);
    return;
  }
  
  /* liquid helium */
  switch(driver_norm_type) {
  case DFT_DRIVER_NORMALIZE_BULK: /*bulk normalization */
    norm = SQRT(dft_driver_rho0) / CABS(cgrid_value_at_index(gwf->grid, 0, 0, 0));
    cgrid_multiply(gwf->grid, norm);
    break;
  case DFT_DRIVER_NORMALIZE_ZEROB:
    i = dft_driver_nx / driver_nhe;
    j = dft_driver_ny / driver_nhe;
    k = dft_driver_nz / driver_nhe;
    norm = SQRT(dft_driver_rho0) / CABS(cgrid_value_at_index(gwf->grid, i, j, k));
    cgrid_multiply(gwf->grid, norm);
    break;
  case DFT_DRIVER_NORMALIZE_DROPLET: /* helium droplet */
    if(!center_release) {
      REAL sq;
      sq = SQRT(3.0*dft_driver_rho0/4.0);
      for (i = 0; i < dft_driver_nx; i++) {
	x = ((REAL) (i - dft_driver_nx2)) * dft_driver_step;
	for (j = 0; j < dft_driver_ny; j++) {
	  y = ((REAL) (j - dft_driver_ny2)) * dft_driver_step;
	  for (k = 0; k < dft_driver_nz; k++) {
	    z = ((REAL) (k - dft_driver_nz2)) * dft_driver_step;
	    if(SQRT(x*x + y*y + z*z) < driver_frad && CABS(cgrid_value_at_index(gwf->grid, i, j, k)) < sq)
              cgrid_value_to_index(gwf->grid, i, j, k, sq);
	  }
	}
      }
    }
    grid_wf_normalize(gwf);
    cgrid_multiply(gwf->grid, SQRT((REAL) driver_nhe));
    break;
  case DFT_DRIVER_NORMALIZE_COLUMN: /* column along y */
    if(!center_release) {
      REAL sq;
      sq = SQRT(3.0*dft_driver_rho0/4.0);
      for (i = 0; i < dft_driver_nx; i++) {
	x = ((REAL) (i - dft_driver_nx2)) * dft_driver_step;
	for (j = 0; j < dft_driver_ny; j++) {
	  y = ((REAL) (j - dft_driver_ny2)) * dft_driver_step;
	  for (k = 0; k < dft_driver_nz; k++) {
	    z = ((REAL) (k - dft_driver_nz2)) * dft_driver_step;
	    if(SQRT(x * x + z * z) < driver_frad && CABS(cgrid_value_at_index(gwf->grid, i, j, k)) < sq)
              cgrid_value_to_index(gwf->grid, i, j, k, sq);
	  }
	}
      }
    }
    grid_wf_normalize(gwf);
    cgrid_multiply(gwf->grid, SQRT((REAL) driver_nhe));
    break;
  case DFT_DRIVER_NORMALIZE_SURFACE:   /* in (x,y) plane starting at z = 0 */
    if(!center_release) {
      for (i = 0; i < dft_driver_nx; i++)
	for (j = 0; j < dft_driver_ny; j++)
	  for (k = 0; k < dft_driver_nz; k++) {
	    z = ((REAL) (k - dft_driver_nz2)) * dft_driver_step;
	    if(FABS(z) < driver_frad)
              cgrid_value_to_index(gwf->grid, i, j, k, 0.0);
	  }
    }
    grid_wf_normalize(gwf);
    cgrid_multiply(gwf->grid, SQRT((REAL) driver_nhe));
    break;
  case DFT_DRIVER_NORMALIZE_N:
    grid_wf_normalize(gwf);
    cgrid_multiply(gwf->grid, SQRT((REAL) driver_nhe));
    break;    
  case DFT_DRIVER_DONT_NORMALIZE:
    break;
  default:
    fprintf(stderr, "libdft: Unknown normalization method.\n");
    exit(1);
  }
}

/*
 * Initialize dft_driver routines. This must always be called after the
 * parameters have been set.
 *
 * No return value.
 *
 */

EXPORT void dft_driver_initialize(wf *gwf) {

  fprintf(stderr, "libdft: GIT version ID %s\n", VERSION);

  if(dft_driver_nx == 0) {
    fprintf(stderr, "libdft: dft_driver not properly initialized.\n");
    exit(1);
  }

  if(dft_driver_otf) { dft_ot_free(dft_driver_otf); dft_driver_otf = NULL; }

  grid_timer_start(&timer);
  grid_threads_init(driver_threads);
  grid_fft_read_wisdom(NULL);

  dft_driver_otf = dft_ot_alloc(driver_dft_model, gwf, MIN_SUBSTEPS, MAX_SUBSTEPS);
  if(dft_driver_rho0 == 0.0) {
    if(dft_driver_verbose) fprintf(stderr, "libdft: Setting dft_driver_rho0 to " FMT_R "\n", dft_driver_otf->rho0);
    dft_driver_rho0 = dft_driver_otf->rho0;
  } else {
    if(dft_driver_verbose) fprintf(stderr, "libdft: Overwritting dft_driver_otf->rho0 to " FMT_R ".\n", dft_driver_rho0);
    dft_driver_otf->rho0 = dft_driver_rho0;
  }
  potential = cgrid_alloc(dft_driver_nx, dft_driver_ny, dft_driver_nz, dft_driver_step, CGRID_PERIODIC_BOUNDARY, 0, "OT Potential");
  if(dft_driver_verbose) fprintf(stderr, "libdft: rho0 = " FMT_R " Angs^-3.\n", dft_driver_rho0 / (GRID_AUTOANG * GRID_AUTOANG * GRID_AUTOANG));
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
  
  dft_driver_nx = nx; dft_driver_nx2 = dft_driver_nx / 2;
  dft_driver_ny = ny; dft_driver_ny2 = dft_driver_ny / 2;
  dft_driver_nz = nz; dft_driver_nz2 = dft_driver_nz / 2;
  dft_driver_step = step;
  // Set the origin to its default value if it is not defined
  if(dft_driver_verbose) fprintf(stderr, "libdft: Grid size = (" FMT_I "," FMT_I "," FMT_I ") with step = " FMT_R ".\n", nx, ny, nz, step);
  driver_threads = threads;
}

/*
 * Set up grid origin.
 * Can be overwritten for a particular grid calling (r/c)grid_set_origin
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
 * Can be overwritten for a particular grid calling (r/c)grid_set_momentum
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
 * Set up potential parameters (for libgrid external function #7).
 *
 * If potential grid == NULL and a0 == -1: No external potential will be used.
 * If potential grid == NULL and a0 != -1: External potential is based on funcion #7.
 * If potential grid != NULL that grid will specify the external potential.
 *
 * The potential function #7 is:
 *
 *              V(R) = A0*exp(-A1*R) - A2 / R^3 - A3 / R^6 - A4 / R^8 - A5 / R^10
 * 
 * The potential parameters are:
 *
 * rmin = Minimum distance where the potential will be evaluated (REAL; input). To disable this function, set to -1.0.
 * radd = Additive constant to R (increasing radd shifts the potential to longer R) (REAL; input).
 * a0   = Parameter A0 above (REAL; input).
 * a1   = Parameter A1 above (REAL; input).
 * a2   = Parameter A2 above (REAL; input).
 * a3   = Parameter A3 above (REAL; input).
 * a4   = Parameter A4 above (REAL; input).
 * a5   = Parameter A5 above (REAL; input).
 *
 * No return value.
 *
 */

EXPORT void dft_driver_setup_potential(REAL rmin, REAL radd, REAL a0, REAL a1, REAL a2, REAL a3, REAL a4, REAL a5) {

  driver_rmin = rmin;
  driver_radd = radd;
  driver_a0 = a0;
  driver_a1 = a1;
  driver_a2 = a2;
  driver_a3 = a3;
  driver_a4 = a4;
  driver_a5 = a5;
}

/*
 * Set up the DFT calculation model.
 *
 * dft_model = specify the DFT Hamiltonian to use (see ot.h) (input, INT).
 * iter_mode = iteration mode: 2 = user specified time (complex), 1 = imaginary time, 0 = real time (input, INT).
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
  if(bh && dft_driver_verbose) fprintf(stderr,"libdft: WARNING -- Overwritting dft_driver_rho0 to " FMT_R "\n", rho0);
  dft_driver_rho0 = rho0;
  bh = 1;
}

/*
 * Define boundary type.
 *
 * type    = Boundary type: 0 = regular, 1 = absorbing (imag time) 
 *           (input, INT).
 * amp     = Max. imaginary amplitude (REAL; input).
 * width_x = Width of the absorbing region along x. Only when type = 1 (REAL; input).
 * width_y = Width of the absorbing region along y. Only when type = 1 (REAL; input).
 * width_z = Width of the absorbing region along z. Only when type = 1 (REAL; input).
 * 
 * NOTE: For the absorbing BC to work, one MUST use DFT_DRIVER_DONT_NORMALIZE and include the chemical potential!
 *       (in both imaginary & real time propagation). Otherwise, you will find issues at the boundary (due to the 
 *       constraint no longer present in RT simulations).
 *
 * No return value.
 *
 */

EXPORT void dft_driver_setup_boundary_type(INT boundary_type, REAL amp, REAL width_x, REAL width_y, REAL width_z) {

  driver_boundary_type = boundary_type;
  if(driver_boundary_type == DFT_DRIVER_BOUNDARY_ITIME) {
    if(dft_driver_verbose) fprintf(stderr, "libdft: ITIME absorbing boundary.\n");
    if(dft_driver_kinetic == DFT_DRIVER_KINETIC_CN_NBC_ROT) {
      fprintf(stderr, "libdft: Absorbing boundaries with rotating liquid (not implemented).\n");
      exit(1);
    }
  }
  driver_bc_lx = (INT) (width_x / dft_driver_step);
  driver_bc_hx = dft_driver_nx - driver_bc_lx - 1;
  driver_bc_ly = (INT) (width_y / dft_driver_step);
  driver_bc_hy = dft_driver_ny - driver_bc_ly - 1;
  driver_bc_lz = (INT) (width_z / dft_driver_step);
  driver_bc_hz = dft_driver_nz - driver_bc_lz - 1;
  fprintf(stderr, "libdft: Absorbing boundary indices: lx = " FMT_I ", hx = " FMT_I ", ly = " FMT_I ", hy = " FMT_I ", lz = " FMT_I ", hz = " FMT_I "\n",
    driver_bc_lx, driver_bc_hx, driver_bc_ly, driver_bc_hy, driver_bc_lz, driver_bc_hz);
  driver_bc_amp = amp;
  fprintf(stderr, "libdft: Max. absorbtion amplitude = " FMT_R ".\n", driver_bc_amp);
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
 * Propagate kinetic (1st half).
 *
 * what   = DFT_DRIVER_PROPAGATE_HELIUM  or DFT_DRIVER_PROPAGATE_OTHER (char; input).
 * gwf    = wavefunction (wf *; input).
 * ctstep = time step in au (REAL complex; input).
 *
 */

EXPORT void dft_driver_propagate_kinetic_first(char what, wf *gwf, REAL complex ctstep) {

  struct grid_abs ab;

  if(what == DFT_DRIVER_PROPAGATE_OTHER_ONLYPOT) return;   /* skip kinetic */

  /* 1/2 x kinetic */
  switch(dft_driver_kinetic) {
  case DFT_DRIVER_KINETIC_FFT:
    grid_wf_propagate_kinetic_fft(gwf, ctstep / 2.0);  // skip possible imag boundary but keep it for potential
    break;
  case DFT_DRIVER_KINETIC_CN_DBC:
  case DFT_DRIVER_KINETIC_CN_NBC:
  case DFT_DRIVER_KINETIC_CN_NBC_ROT:
  case DFT_DRIVER_KINETIC_CN_PBC:
    if(driver_boundary_type == DFT_DRIVER_BOUNDARY_ITIME && driver_iter_mode == DFT_DRIVER_REAL_TIME) {
      ab.amp = driver_bc_amp;
      ab.data[0] = driver_bc_lx;
      ab.data[1] = driver_bc_hx;
      ab.data[2] = driver_bc_ly;
      ab.data[3] = driver_bc_hy;
      ab.data[4] = driver_bc_lz;
      ab.data[5] = driver_bc_hz;
      grid_wf_propagate_cn(gwf, grid_wf_absorb, ctstep / 2.0, &ab, NULL);
    } else {
      grid_wf_propagate_cn(gwf, NULL, ctstep / 2.0, NULL, NULL);
    }
    break;
  default:
    fprintf(stderr, "libdft: Unknown BC for kinetic energy propagation.\n");
    exit(1);
  }

  if(driver_iter_mode == DFT_DRIVER_REAL_TIME) scale_wf(what, gwf);
}

/*
 * Propagate kinetic (2nd half).
 *
 * what   = DFT_DRIVER_PROPAGATE_HELIUM or DFT_DRIVER_PROPAGATE_OTHER (char; input).
 * gwf    = wavefunction (wf *; input/output).
 * ctstep = time step in au (REAL complex; input).
 *
 */

EXPORT void dft_driver_propagate_kinetic_second(char what, wf *gwf, REAL complex ctstep) {

  static char local_been_here = 0;
  
  if(what == DFT_DRIVER_PROPAGATE_OTHER_ONLYPOT) return;   /* skip kinetic */

  dft_driver_propagate_kinetic_first(what, gwf, ctstep);

  if(!local_been_here) {
    local_been_here = 1;
    grid_fft_write_wisdom(NULL); // we have done many FFTs at this point
  }
}

/*
 * Propagate potential.
 *
 * what   = DFT_DRIVER_PROPAGATE_HELIUM  or DFT_DRIVER_PROPAGATE_OTHER (char; input).
 * gwf    = wavefunction (wf *; input/output).
 * pot    = potential (cgrid *; input).
 * ctstep = time step in au (REAL complex, input).
 *
 */

EXPORT void dft_driver_propagate_potential(char what, wf *gwf, cgrid *pot, REAL complex ctstep) {

  struct grid_abs ab;

  if(driver_boundary_type == DFT_DRIVER_BOUNDARY_ITIME && driver_iter_mode != DFT_DRIVER_IMAG_TIME) {
    ab.amp = driver_bc_amp;
    ab.data[0] = driver_bc_lx;
    ab.data[1] = driver_bc_hx;
    ab.data[2] = driver_bc_ly;
    ab.data[3] = driver_bc_hy;
    ab.data[4] = driver_bc_lz;
    ab.data[5] = driver_bc_hz;
    grid_wf_propagate_potential(gwf, grid_wf_absorb, ctstep, &ab, pot);
  } else 
    grid_wf_propagate_potential(gwf, NULL, ctstep, NULL, pot);

  if(driver_iter_mode != DFT_DRIVER_REAL_TIME) scale_wf(what, gwf);
}

/*
 * Propagate step: propagate the given wf in time. No predict/correct. Must use smaller time step but uses less memory.
 *
 * what      = DFT_DRIVER_PROPAGATE_HELIUM  or DFT_DRIVER_PROPAGATE_OTHER (char; input).
 * ext_pot   = present external potential grid (rgrid *; input) (NULL = no ext. pot).
 * chempot   = chemical potential (REAL).
 * gwf       = liquid wavefunction to propagate (wf *; input).
 *             Note that gwf is NOT changed by this routine.
 * ctstep    = time step in FS (REAL complex; input).
 * iter      = current iteration (INT; input).
 *
 * If what == DFT_DRIVER_PROPAGATE_HELIUM, the liquid potential is added automatically. Both kinetic and potential propagated.
 * If what == DFT_DIRVER_PROPAGATE_OTHER, propagate only with the external potential (i.e., impurity). Both kin + ext pot. propagated.
 * If what == DFT_DRIVER_PROPAGATE_OTHER_ONLYPOT, propagate only with the external potential - no kinetic energy (Lappacian).
 *
 * No return value.
 *
 */

EXPORT inline void dft_driver_propagate(char what, rgrid *ext_pot, REAL chempot, wf *gwf, REAL complex ctstep, INT iter) {

  struct grid_abs ab;
  
  ctstep /= GRID_AUTOFS;
  switch(driver_iter_mode) {
    case DFT_DRIVER_REAL_TIME:
      ctstep = CREAL(ctstep);
      break;
    case DFT_DRIVER_IMAG_TIME:
      ctstep = -I * CREAL(ctstep);
    break;
    case DFT_DRIVER_USER_TIME:
      /* Use whatever we were given */
    break;
    default:
      fprintf(stderr, "libdft: Illegal value for driver_iter_mode.\n");
      exit(1);
  }

  grid_timer_start(&timer);  

  if(dft_driver_nx == 0) {
    fprintf(stderr, "libdft: dft_driver not setup.\n");
    exit(1);
  }

  if(!iter && driver_iter_mode == DFT_DRIVER_IMAG_TIME && what < DFT_DRIVER_PROPAGATE_OTHER && dft_driver_init_wavefunction == 1) {
    if(dft_driver_verbose) fprintf(stderr, "libdft: first imag. time iteration - initializing the wavefunction.\n");
    grid_wf_constant(gwf, SQRT(dft_driver_otf->rho0));
  }

  /* droplet & column center release */
  if(driver_rels && iter > driver_rels && driver_norm_type > 0 && what < DFT_DRIVER_PROPAGATE_OTHER) {
    if(!center_release && dft_driver_verbose) fprintf(stderr, "libdft: center release activated.\n");
    center_release = 1;
  } else center_release = 0;

  dft_driver_propagate_kinetic_first(what, gwf, ctstep);

  cgrid_zero(potential);
  switch(what) {
  case DFT_DRIVER_PROPAGATE_HELIUM:
    dft_ot_potential(dft_driver_otf, potential, gwf);
    if(viscosity != 0.0) {
      if(dft_driver_verbose) fprintf(stderr, "libdft: Including viscous potential.\n");
      dft_viscous_potential(gwf, potential, dft_driver_rho0, viscosity, viscosity_alpha);
    }
    break;
  case DFT_DRIVER_PROPAGATE_OTHER:
  case DFT_DRIVER_PROPAGATE_OTHER_ONLYPOT:
    break;
  default:
    fprintf(stderr, "libdft: Unknown propagator flag.\n");
    exit(1);
  }
  if(ext_pot && driver_rmin != -1.0) fprintf(stderr, "libdft(warning): Both external potential and potential function in use!\n");
  if(ext_pot) grid_add_real_to_complex_re(potential, ext_pot);
  else if(driver_rmin != -1.0) grid_func7a_operate_one(potential, driver_rmin, driver_radd, driver_a0, driver_a1, driver_a2, driver_a3, driver_a4, driver_a5);
  if(gwf->grid->kx0 != 0.0) chempot += HBAR * HBAR * gwf->grid->kx0 * gwf->grid->kx0 / (2.0 * gwf->mass);
  if(gwf->grid->ky0 != 0.0) chempot += HBAR * HBAR * gwf->grid->ky0 * gwf->grid->ky0 / (2.0 * gwf->mass);
  if(gwf->grid->kz0 != 0.0) chempot += HBAR * HBAR * gwf->grid->kz0 * gwf->grid->kz0 / (2.0 * gwf->mass);
  cgrid_add(potential, (REAL complex) -chempot);

  if((dft_driver_kinetic == DFT_DRIVER_KINETIC_CN_NBC || dft_driver_kinetic == DFT_DRIVER_KINETIC_CN_NBC_ROT
     || dft_driver_kinetic == DFT_DRIVER_KINETIC_CN_DBC || dft_driver_kinetic == DFT_DRIVER_KINETIC_CN_PBC)
     && what != DFT_DRIVER_PROPAGATE_OTHER_ONLYPOT) {
    fprintf(stderr, "lidft(INFO): Simultaneous propagation of kinetic & potential using CN.\n");
    /* Avoid operator splitting: propagate both kinetic and potential simultaneously with CN */
    if(driver_boundary_type == DFT_DRIVER_BOUNDARY_ITIME && driver_iter_mode == DFT_DRIVER_REAL_TIME) {
      ab.amp = driver_bc_amp;
      ab.data[0] = driver_bc_lx;
      ab.data[1] = driver_bc_hx;
      ab.data[2] = driver_bc_ly;
      ab.data[3] = driver_bc_hy;
      ab.data[4] = driver_bc_lz;
      ab.data[5] = driver_bc_hz;
      grid_wf_propagate_cn(gwf, grid_wf_absorb, ctstep / 2.0, &ab, potential);
    } else 
      grid_wf_propagate_cn(gwf, NULL, ctstep / 2.0, NULL, potential);
  } else {
    dft_driver_propagate_potential(what, gwf, potential, ctstep);
    dft_driver_propagate_kinetic_second(what, gwf, ctstep);
  }

  if(dft_driver_verbose) fprintf(stderr, "libdft: Propagate step " FMT_R " wall clock seconds (iter = " FMT_I ").\n", grid_timer_wall_clock_time(&timer), iter);
  fflush(stderr);
}

/*
 * Predict step: propagate the given wf in time.
 *
 * what      = DFT_DRIVER_PROPAGATE_HELIUM  or DFT_DRIVER_PROPAGATE_OTHER (char; input).
 * ext_pot   = present external potential grid (rgrid *; input) (NULL = no ext. pot).
 * chempot   = chemical potential (REAL).
 * gwf       = liquid wavefunction to propagate (wf *; input).
 *             Note that gwf is NOT changed by this routine.
 * gwfp      = predicted wavefunction (wf *; output).
 * potential = storage space for the potential (cgrid *; output).
 *             Do not overwrite this before calling the correct routine.
 * ctstep    = time step in FS (REAL complex; input).
 * iter      = current iteration (INT; input).
 *
 * If what == DFT_DRIVER_PROPAGATE_HELIUM, the liquid potential is added automatically. Both kinetic and potential propagated.
 * If what == DFT_DIRVER_PROPAGATE_OTHER, propagate only with the external potential (i.e., impurity). Both kin + ext pot. propagated.
 * If what == DFT_DRIVER_PROPAGATE_OTHER_ONLYPOT, propagate only with the external potential - no kinetic energy (Lappacian).
 *
 * No return value.
 *
 */

EXPORT inline void dft_driver_propagate_predict(char what, rgrid *ext_pot, REAL chempot, wf *gwf, wf *gwfp, cgrid *potential, REAL complex ctstep, INT iter) {

  ctstep /= GRID_AUTOFS;
  switch(driver_iter_mode) {
    case DFT_DRIVER_REAL_TIME:
      ctstep = CREAL(ctstep);
      break;
    case DFT_DRIVER_IMAG_TIME:
      ctstep = -I * CREAL(ctstep);
    break;
    case DFT_DRIVER_USER_TIME:
      /* Use whatever we were given */
    break;
    default:
      fprintf(stderr, "libdft: Illegal value for driver_iter_mode.\n");
      exit(1);
  }

  grid_timer_start(&timer);  

  if(dft_driver_nx == 0) {
    fprintf(stderr, "libdft: dft_driver not setup.\n");
    exit(1);
  }

  if(!iter && driver_iter_mode == DFT_DRIVER_IMAG_TIME && what < DFT_DRIVER_PROPAGATE_OTHER && dft_driver_init_wavefunction == 1) {
    if(dft_driver_verbose) fprintf(stderr, "libdft: first imag. time iteration - initializing the wavefunction.\n");
    grid_wf_constant(gwf, SQRT(dft_driver_otf->rho0));
  }

  /* droplet & column center release */
  if(driver_rels && iter > driver_rels && driver_norm_type > 0 && what < DFT_DRIVER_PROPAGATE_OTHER) {
    if(!center_release && dft_driver_verbose) fprintf(stderr, "libdft: center release activated.\n");
    center_release = 1;
  } else center_release = 0;

  dft_driver_propagate_kinetic_first(what, gwf, ctstep);

  cgrid_zero(potential);
  switch(what) {
  case DFT_DRIVER_PROPAGATE_HELIUM:
    dft_ot_potential(dft_driver_otf, potential, gwf);
    if(viscosity != 0.0) {
      if(dft_driver_verbose) fprintf(stderr, "libdft: Including viscous potential.\n");
      dft_viscous_potential(gwf, potential, dft_driver_rho0, viscosity, viscosity_alpha);
    }
    break;
  case DFT_DRIVER_PROPAGATE_OTHER:
  case DFT_DRIVER_PROPAGATE_OTHER_ONLYPOT:
    break;
  default:
    fprintf(stderr, "libdft: Unknown propagator flag.\n");
    exit(1);
  }
  if(ext_pot && driver_rmin != -1.0) fprintf(stderr, "libdft(warning): Both external potential and potential function in use!\n");
  if(ext_pot) grid_add_real_to_complex_re(potential, ext_pot);
  else if(driver_rmin != -1.0) grid_func7a_operate_one(potential, driver_rmin, driver_radd, driver_a0, driver_a1, driver_a2, driver_a3, driver_a4, driver_a5);
  if(gwf->grid->kx0 != 0.0) chempot += HBAR * HBAR * gwf->grid->kx0 * gwf->grid->kx0 / (2.0 * gwf->mass);
  if(gwf->grid->ky0 != 0.0) chempot += HBAR * HBAR * gwf->grid->ky0 * gwf->grid->ky0 / (2.0 * gwf->mass);
  if(gwf->grid->kz0 != 0.0) chempot += HBAR * HBAR * gwf->grid->kz0 * gwf->grid->kz0 / (2.0 * gwf->mass);
  cgrid_add(potential, (REAL complex) -chempot);

  cgrid_copy(gwfp->grid, gwf->grid);

  dft_driver_propagate_potential(what, gwfp, potential, ctstep);

  if(dft_driver_verbose) fprintf(stderr, "libdft: Predict step " FMT_R " wall clock seconds (iter = " FMT_I ").\n", grid_timer_wall_clock_time(&timer), iter);
  fflush(stderr);
}

/*
 * Correct step: propagate the given wf in time.
 *
 * what      = DFT_DRIVER_PROPAGATE_HELIUM  or DFT_DRIVER_PROPAGATE_OTHER (char; input).
 * ext_pot   = present external potential grid (rgrid *) (NULL = no ext. pot).
 * chempot   = chemical potential (REAL; input).
 * gwf       = liquid wavefunction to propagate (wf *).
 *             Note that gwf is NOT changed by this routine.
 * gwfp      = predicted wavefunction (wf *; output).
 * potential = storage space for the potential (cgrid *; output).
 * ctstep    = time step in FS (REAL complex; input).
 * iter      = current iteration (INT).
 *
 * If what == DFT_DRIVER_PROPAGATE_HELIUM, the liquid potential is added automatically. Both kinetic and potential propagated.
 * If what == DFT_DIRVER_PROPAGATE_OTHER, propagate only with the external potential (i.e., impurity). Both kin + ext pot. propagated.
 * If what == DFT_DRIVER_PROPAGATE_OTHER_ONLYPOT, propagate only with the external potential - no kinetic energy (Lappacian).
 *
 * No return value.
 *
 */

EXPORT inline void dft_driver_propagate_correct(char what, rgrid *ext_pot, REAL chempot, wf *gwf, wf *gwfp, cgrid *potential, REAL complex ctstep, INT iter) {

  ctstep /= GRID_AUTOFS;
  switch(driver_iter_mode) {
    case DFT_DRIVER_REAL_TIME:
      ctstep = CREAL(ctstep);
      break;
    case DFT_DRIVER_IMAG_TIME:
      ctstep = -I * CREAL(ctstep);
    break;
    case DFT_DRIVER_USER_TIME:
      /* Use whatever we were given */
    break;
    default:
      fprintf(stderr, "libdft: Illegal value for driver_iter_mode.\n");
      exit(1);
  }

  grid_timer_start(&timer);  

  switch(what) {
  case DFT_DRIVER_PROPAGATE_HELIUM:
    dft_ot_potential(dft_driver_otf, potential, gwfp);
    if(viscosity != 0.0) {
      if(dft_driver_verbose) fprintf(stderr, "libdft: Including viscous potential.\n");
      dft_viscous_potential(gwfp, potential, dft_driver_rho0, viscosity, viscosity_alpha);
    }
    break;
  case DFT_DRIVER_PROPAGATE_OTHER:
  case DFT_DRIVER_PROPAGATE_OTHER_ONLYPOT:
    break;
  default:
    fprintf(stderr, "libdft: Unknown propagator flag.\n");
    exit(1);
  }
  if(ext_pot && driver_rmin != -1.0) fprintf(stderr, "libdft(warning): Both external potential and potential function in use!\n");
  if(ext_pot) grid_add_real_to_complex_re(potential, ext_pot);
  else if(driver_rmin != -1.0) grid_func7a_operate_one(potential, driver_rmin, driver_radd, driver_a0, driver_a1, driver_a2, driver_a3, driver_a4, driver_a5);
  if(gwf->grid->kx0 != 0.0) chempot += HBAR * HBAR * gwf->grid->kx0 * gwf->grid->kx0 / (2.0 * gwf->mass);
  if(gwf->grid->ky0 != 0.0) chempot += HBAR * HBAR * gwf->grid->ky0 * gwf->grid->ky0 / (2.0 * gwf->mass);
  if(gwf->grid->kz0 != 0.0) chempot += HBAR * HBAR * gwf->grid->kz0 * gwf->grid->kz0 / (2.0 * gwf->mass);
  cgrid_add(potential, (REAL complex) -chempot);

  cgrid_multiply(potential, 0.5);
  dft_driver_propagate_potential(what, gwf, potential, ctstep);

  dft_driver_propagate_kinetic_second(what, gwf, ctstep);

  if(dft_driver_verbose) fprintf(stderr, "libdft: Correct step " FMT_R " wall clock seconds (iter = " FMT_I ").\n", grid_timer_wall_clock_time(&timer), iter);
  fflush(stderr);
}

/*
 * Allocate a complex grid.
 *
 * Returns a pointer to new grid.
 *
 */

EXPORT cgrid *dft_driver_alloc_cgrid(char *id) {

  REAL complex (*grid_type)(cgrid *, INT, INT, INT);
  cgrid *tmp;

  if(dft_driver_nx == 0) {
    fprintf(stderr, "libdft: dft_driver routines must be initialized first.\n");
    exit(1);
  }

  switch(driver_bc) {
  case DFT_DRIVER_BC_NORMAL: 
    grid_type = CGRID_PERIODIC_BOUNDARY;
    break;
  case DFT_DRIVER_BC_X:
    grid_type = CGRID_VORTEX_X_BOUNDARY;
    break;
  case DFT_DRIVER_BC_Y:
    grid_type = CGRID_VORTEX_Y_BOUNDARY;
    break;
  case DFT_DRIVER_BC_Z:
    grid_type = CGRID_VORTEX_Z_BOUNDARY;
    break;
  case DFT_DRIVER_BC_NEUMANN:
    grid_type = CGRID_NEUMANN_BOUNDARY;
    break;
  default:
    fprintf(stderr, "libdft: Illegal boundary type.\n");
    exit(1);
  }

  tmp = cgrid_alloc(dft_driver_nx, dft_driver_ny, dft_driver_nz, dft_driver_step, grid_type, 0, id);
  cgrid_set_origin(tmp, driver_x0, driver_y0, driver_z0);
  cgrid_set_momentum(tmp, driver_kx0, driver_ky0, driver_kz0);
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

EXPORT rgrid *dft_driver_alloc_rgrid(char *id) {

  REAL (*grid_type)(rgrid *, INT, INT, INT);
  rgrid *tmp;

  if(dft_driver_nx == 0) {
    fprintf(stderr, "libdft: dft_driver routines must be initialized first.\n");
    exit(1);
  }

  switch(driver_bc) {
  case DFT_DRIVER_BC_NORMAL: 
  case DFT_DRIVER_BC_NEUMANN:
    grid_type = RGRID_PERIODIC_BOUNDARY;    // TODO: Neumann should belong to the case below
    break;
  case DFT_DRIVER_BC_X:
  case DFT_DRIVER_BC_Y:
  case DFT_DRIVER_BC_Z:
    /*  case DFT_DRIVER_BC_NEUMANN: */
    grid_type = RGRID_NEUMANN_BOUNDARY;
    break;
  default:
    fprintf(stderr, "libdft: Illegal boundary type.\n");
    exit(1);
  }

  tmp = rgrid_alloc(dft_driver_nx, dft_driver_ny, dft_driver_nz, dft_driver_step, grid_type, 0, id);
  rgrid_set_origin(tmp, driver_x0, driver_y0, driver_z0);
  rgrid_set_momentum(tmp, driver_kx0, driver_ky0, driver_kz0);
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

EXPORT wf *dft_driver_alloc_wavefunction(REAL mass, char *id) {

  wf *tmp;
  char grid_type;
  
  if(dft_driver_nx == 0) {
    fprintf(stderr, "libdft: dft_driver routines must be initialized first.\n");
    exit(1);
  }

  switch(driver_bc) {
  case DFT_DRIVER_BC_NORMAL: 
    grid_type = WF_PERIODIC_BOUNDARY;
    break;
  case DFT_DRIVER_BC_X:
    grid_type = WF_VORTEX_X_BOUNDARY;
    break;
  case DFT_DRIVER_BC_Y:
    grid_type = WF_VORTEX_Y_BOUNDARY;
    break;
  case DFT_DRIVER_BC_Z:
    grid_type = WF_VORTEX_Z_BOUNDARY;
    break;
  case DFT_DRIVER_BC_NEUMANN:
    grid_type = WF_NEUMANN_BOUNDARY;
    break;
  default:
    fprintf(stderr, "libdft: Illegal boundary type.\n");
    exit(1);
  }

  tmp = grid_wf_alloc(dft_driver_nx, dft_driver_ny, dft_driver_nz, dft_driver_step, mass, grid_type, WF_2ND_ORDER_PROPAGATOR, id);
  cgrid_set_origin(tmp->grid, driver_x0, driver_y0, driver_z0);
  cgrid_set_momentum(tmp->grid, driver_kx0, driver_ky0, driver_kz0);
  cgrid_constant(tmp->grid, SQRT(dft_driver_rho0));
  return tmp;
}

/*
 * Produce radially averaged density from a 3-D grid.
 * 
 * radial = Radial density (rgrid *; output).
 * grid   = Source grid (rgrid *; input).
 * dtheta = Integration step size along theta in radians (REAL; input).
 * dphi   = Integration step size along phi in radians (REAL, input).
 * xc     = x coordinate for the center (REAL; input).
 * yc     = y coordinate for the center (REAL; input).
 * zc     = z coordinate for the center (REAL; input).
 *
 */

EXPORT void dft_driver_radial(rgrid *radial, rgrid *grid, REAL dtheta, REAL dphi, REAL xc, REAL yc, REAL zc) {
  
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
	tmp += rgrid_value(grid, x, y, z) * SIN(theta);
      }
    tmp *= dtheta * dphi / (4.0 * M_PI);
    radial->value[ri] = tmp;
    r += step;
  }
}

/*
 * Produce radially averaged complex grid from a 3-D grid.
 *
 * radial = Radial density (cgrid *; output).
 * grid   = Source grid (cgrid *; input).
 * dtheta = Integration step size along theta in radians (REAL; input).
 * dphi   = Integration step size along phi in radians (REAL, input).
 * xc     = x coordinate for the center (REAL; input).
 * yc     = y coordinate for the center (REAL; input).
 * zc     = z coordinate for the center (REAL; input).
 *
 */

EXPORT void dft_driver_radial_complex(cgrid *radial, cgrid *grid, REAL dtheta, REAL dphi, REAL xc, REAL yc, REAL zc) {
  
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
	tmp += cgrid_value(grid, x, y, z) * SIN(theta);
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
 * density = liquid density (rgrid *; input).
 *
 * Return value: R_b.
 *
 */

EXPORT REAL dft_driver_spherical_rb(rgrid *density) {

  REAL disp;

  rgrid_multiply(density, -1.0);
  rgrid_add(density, dft_driver_rho0);
  disp = rgrid_integral(density);
  rgrid_add(density, -dft_driver_rho0);
  rgrid_multiply(density, -1.0);

  return POW(disp * 3.0 / (4.0 * M_PI * dft_driver_rho0), 1.0 / 3.0);
}

/*
 * Force spherical symmetry by spherical averaging.
 *
 * wf = wavefunction to be averaged (wf *).
 * xc = x coordinate for the center (REAL).
 * yc = y coordinate for the center (REAL).
 * zc = z coordinate for the center (REAL).
 *
 */

EXPORT void dft_driver_force_spherical(wf *wf, REAL xc, REAL yc, REAL zc) {

  INT i, j, l, k, len;
  INT nx = wf->grid->nx, ny = wf->grid->ny, nz = wf->grid->nz;
  REAL step = wf->grid->step;
  REAL x, y, z, x2, y2, z2, d;
  cgrid *average;
  REAL complex *avalue;

  if(nx > ny) len = nx; else len = ny;
  if(nz > len) len = nz;
  average = cgrid_alloc(1, 1, len, step, CGRID_PERIODIC_BOUNDARY, 0, "average");
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
        cgrid_value_to_index(wf->grid, i, j, k, avalue[k]);
      }
    }
  }
  cgrid_free(average);
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
 * gwf    = Wavefunction for the operation (input/output, gwf *).
 * n      = Quantum number (1 or 2) (input, int).
 * 
 */

EXPORT void dft_driver_vortex_initial(wf *gwf, int n, int axis) {

  cgrid *cworkspace = gwf->cworkspace;

  if(axis == DFT_DRIVER_VORTEX_X) {
    switch(n) {
    case 1:
      cgrid_map(cworkspace, vortex_x_n1, NULL);
      cgrid_product(gwf->grid, gwf->grid, cworkspace);      
      break;
    case 2:
      cgrid_map(cworkspace, vortex_x_n2, NULL);
      cgrid_product(gwf->grid, gwf->grid, cworkspace);      
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
      cgrid_map(cworkspace, vortex_y_n1, NULL);
      cgrid_product(gwf->grid, gwf->grid, cworkspace);      
      break;
    case 2:
      cgrid_map(cworkspace, vortex_y_n2, NULL);
      cgrid_product(gwf->grid, gwf->grid, cworkspace);      
      break;
    default:
      fprintf(stderr,"libdft: Illegal value for n (dft_driver_vortex_initial()).\n");
      break;
    }
    cgrid_release(cworkspace);
    return;
  }

  if(axis == DFT_DRIVER_VORTEX_Z) {
    switch(n) {
    case 1:
      cgrid_map(cworkspace, vortex_z_n1, NULL);
      cgrid_product(gwf->grid, gwf->grid, cworkspace);      
      break;
    case 2:
      cgrid_map(cworkspace, vortex_z_n2, NULL);
      cgrid_product(gwf->grid, gwf->grid, cworkspace);      
      break;
    default:
      fprintf(stderr,"libdft: Illegal value for n (dft_driver_vortex_initial()).\n");
      break;
    }
    cgrid_release(cworkspace);
    return;
  }
  fprintf(stderr, "libdft: Illegal axis for dft_driver_vortex_initial().\n");
  exit(1);
}

/*
 * Add vortex potential (Feynman-Onsager ansatz) along a specified axis.
 *
 * potential = Potential grid where the vortex potential is added (rgrid *, input/output).
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

EXPORT void dft_driver_vortex(rgrid *potential, int direction) {

  rgrid *workspace6 = dft_driver_otf->workspace6;

  rgrid_claim(workspace6);
  switch(direction) {
  case DFT_DRIVER_VORTEX_X:
    rgrid_map(workspace6, vortex_x, NULL);
    rgrid_sum(potential, potential, workspace6);
    break;
  case DFT_DRIVER_VORTEX_Y:
    rgrid_map(workspace6, vortex_y, NULL);
    rgrid_sum(potential, potential, workspace6);
    break;
  case DFT_DRIVER_VORTEX_Z:
    rgrid_map(workspace6, vortex_z, NULL);
    rgrid_sum(potential, potential, workspace6);
    break;
  default:
    fprintf(stderr, "libdft: Unknown axis direction for vortex potential (dft_driver_vortex()).\n");
    exit(1);
  }
  rgrid_release(workspace6);
}

