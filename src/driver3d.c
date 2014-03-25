/*
 * Simple driver routines to propagate the liquid (3D).
 *
 * TODO: Add comments to show which internal workspaces are used by 
 * each function.
 *
 * TODO: DFT_DRIVER_BC_NEUMANN generates problems with DFT_DRIVER_KINETIC_CN_NBC_ROT .
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
#define MAX_SUBSTEPS 128

/* End of tunable parameters */

/* Global user accessible variables */
dft_ot_functional *dft_driver_otf = 0;
int dft_driver_init_wavefunction = 1;

static long driver_nx = 0, driver_ny = 0, driver_nz = 0, driver_threads = 0, driver_dft_model = 0, driver_iter_mode = 0, driver_boundary_type = 0;
static long driver_norm_type = 0, driver_nhe = 0, center_release = 0, driver_bc = 0;
static long driver_rels = 0;
static double driver_frad = 0.0, driver_omega = 0.0;
static double driver_step = 0.0, driver_abs = 0.0, driver_rho0 = 0.0, driver_halfbox_length = 0.0 ;
static double driver_x0 = 0.0 , driver_y0 = 0.0 , driver_z0 = 0.0 ;
static double driver_kx0 = 0. , driver_ky0 = 0. ,driver_kz0 = 0. ;
static rgrid3d *density = 0;
static rgrid3d *workspace1 = 0;
static rgrid3d *workspace2 = 0;
static rgrid3d *workspace3 = 0;
static rgrid3d *workspace4 = 0;
static rgrid3d *workspace5 = 0;
static rgrid3d *workspace6 = 0;
static rgrid3d *workspace7 = 0;
static rgrid3d *workspace8 = 0;
static rgrid3d *workspace9 = 0;
static cgrid3d *cworkspace = 0;
static grid_timer timer;
static double damp = 0.2;

int dft_internal_using_3d = 0;
extern int dft_internal_using_2d, dft_internal_using_cyl;
int dft_driver_kinetic = 0; /* default FFT propagation for kinetic */

static inline void check_mode() {

  if(dft_internal_using_2d || dft_internal_using_cyl) {
    fprintf(stderr, "libdft: 2D or Cylindrical 3D routine called in Cartesian 3D code.\n");
    exit(1);
  } else dft_internal_using_3d = 1;
}

static double region_func(void *gr, double x, double y, double z) {

  double ulx = (driver_nx/2.0) * driver_step - driver_abs, uly = (driver_ny/2.0) * driver_step - driver_abs, ulz = (driver_nz/2.0) * driver_step - driver_abs;
  double d = 0.0;
  
  x = fabs(x);
  y = fabs(y);
  z = fabs(z);

  if(x >= ulx) d += damp * (x - ulx) / driver_abs;
  if(y >= uly) d += damp * (y - uly) / driver_abs;
  if(z >= ulz) d += damp * (z - ulz) / driver_abs;
  return d / 3.0;
}

/*
 * Spherical region going from 0 to 1 radially, increasing as tanh(r).
 * It has a value of ~0 (6.e-4) when r = driver_abs, and goes up to 1 
 * when r = driver_halfbox_length (i.e. the smallest end of the box).
 *
 */
static double complex cregion_func(void *gr, double x, double y, double z) {
	double r = sqrt(x*x + y*y + z*z) ;
	return 1. + tanh( 4.0 * (r - driver_halfbox_length)/driver_abs ) ;
}


inline static void scale_wf(long what, dft_ot_functional *local_otf, wf3d *gwf) {

  long i, j, k;
  double x, y, z;
  double complex norm;

  if(what) { /* impurity */
    grid3d_wf_normalize(gwf);
    return;
  }
  
  /* liquid helium */
  switch(driver_norm_type) {
  case DFT_DRIVER_NORMALIZE_BULK: /* bulk normalization */
    norm = sqrt(local_otf->rho0) / cabs(gwf->grid->value[0]);
    cgrid3d_multiply(gwf->grid, norm);
    break;
  case DFT_DRIVER_NORMALIZE_ZEROB:
#if 0
    i = 0;
    for (j = 0; j < driver_ny; j++)
      for (k = 0; k < driver_nz; k++)
	gwf->grid->value[i * driver_ny * driver_nz + j * driver_nz + k] = 0.0;
    i = driver_nx - 1;
    for (j = 0; j < driver_ny; j++)
      for (k = 0; k < driver_nz; k++)
	gwf->grid->value[i * driver_ny * driver_nz + j * driver_nz + k] = 0.0;
    j = 0;
    for (i = 0; i < driver_nx; i++)
      for (k = 0; k < driver_nz; k++)
	gwf->grid->value[i * driver_ny * driver_nz + j * driver_nz + k] = 0.0;
    j = driver_ny - 1;
    for (i = 0; i < driver_nx; i++)
      for (k = 0; k < driver_nz; k++)
	gwf->grid->value[i * driver_ny * driver_nz + j * driver_nz + k] = 0.0;
    k = 0;
    for (i = 0; i < driver_nx; i++)
      for (j = 0; j < driver_ny; j++)
	gwf->grid->value[i * driver_ny * driver_nz + j * driver_nz + k] = 0.0;
    k = driver_nz - 1;
    for (i = 0; i < driver_nx; i++)
      for (j = 0; j < driver_ny; j++)
	gwf->grid->value[i * driver_ny * driver_nz + j * driver_nz + k] = 0.0;
 #endif
    i = driver_nx / driver_nhe;
    j = driver_ny / driver_nhe;
    k = driver_nz / driver_nhe;
    norm = sqrt(local_otf->rho0) / cabs(gwf->grid->value[i * driver_ny * driver_nz + j * driver_nz + k]);
    cgrid3d_multiply(gwf->grid, norm);
    break;
  case DFT_DRIVER_NORMALIZE_DROPLET: /* helium droplet */
    if(!center_release) {
      double sq = sqrt(3.0*driver_rho0/4.0);
      for (i = 0; i < driver_nx; i++) {
	x = (i - driver_nx/2.0) * driver_step;
	for (j = 0; j < driver_ny; j++) {
	  y = (j - driver_ny/2.0) * driver_step;
	  for (k = 0; k < driver_nz; k++) {
	    z = (k - driver_nz/2.0) * driver_step;
	    if(sqrt(x*x + y*y + z*z) < driver_frad && cabs(gwf->grid->value[i * driver_ny * driver_nz + j * driver_nz + k]) < sq)
	      gwf->grid->value[i * driver_ny * driver_nz + j * driver_nz + k] = sq;
	  }
	}
      }
    }
    grid3d_wf_normalize(gwf);
    cgrid3d_multiply(gwf->grid, sqrt((double) driver_nhe));
    break;
  case DFT_DRIVER_NORMALIZE_COLUMN: /* column along y */
    if(!center_release) {
      double sq = sqrt(3.0*driver_rho0/4.0);
      long nyz = driver_ny * driver_nz;
      for (i = 0; i < driver_nx; i++) {
	x = (i - driver_nx/2.0) * driver_step;
	for (j = 0; j < driver_ny; j++) {
	  y = (j - driver_ny/2.0) * driver_step;
	  for (k = 0; k < driver_nz; k++) {
	    z = (k - driver_nz/2.0) * driver_step;
	    if(sqrt(x * x + z * z) < driver_frad && cabs(gwf->grid->value[i * nyz + j * driver_nz + k]) < sq)
	      gwf->grid->value[i * nyz + j * driver_nz + k] = sq;
	  }
	}
      }
    }
    grid3d_wf_normalize(gwf);
    cgrid3d_multiply(gwf->grid, sqrt((double) driver_nhe));
    break;
  case DFT_DRIVER_NORMALIZE_SURFACE:   /* in (x,y) plane starting at z = 0 */
    if(!center_release) {
      for (i = 0; i < driver_nx; i++)
	for (j = 0; j < driver_ny; j++)
	  for (k = 0; k < driver_nz; k++) {
	    z = (k - driver_nz/2.0) * driver_step;
	    if(fabs(z) < driver_frad)
	      gwf->grid->value[i * driver_ny * driver_nz + j * driver_nz + k] = 0.0;
	  }
    }
    grid3d_wf_normalize(gwf);
    cgrid3d_multiply(gwf->grid, sqrt((double) driver_nhe));
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

static int been_here = 0;

EXPORT void dft_driver_initialize() {

  check_mode();

  if(!been_here) {
    if(driver_nx == 0) {
      fprintf(stderr, "libdft: dft_driver not properly initialized.\n");
      exit(1);
    }
    grid_timer_start(&timer);
    grid_threads_init(driver_threads);
    workspace1 = dft_driver_alloc_rgrid();
    workspace2 = dft_driver_alloc_rgrid();
    workspace3 = dft_driver_alloc_rgrid();
    workspace4 = dft_driver_alloc_rgrid();
    workspace5 = dft_driver_alloc_rgrid();
    workspace6 = dft_driver_alloc_rgrid();
    workspace7 = dft_driver_alloc_rgrid();
    if(driver_dft_model & DFT_OT_BACKFLOW) {
      workspace8 = dft_driver_alloc_rgrid();
      workspace9 = dft_driver_alloc_rgrid();
    }
    density = dft_driver_alloc_rgrid();
    dft_driver_otf = dft_ot3d_alloc(driver_dft_model, driver_nx, driver_ny, driver_nz, driver_step, driver_bc, MIN_SUBSTEPS, MAX_SUBSTEPS);
    if(driver_rho0 == 0.0){
	    fprintf(stderr, "libdft: Setting driver_rho0 to %le\n", dft_driver_otf->rho0 ) ;
	    driver_rho0 = dft_driver_otf->rho0;
    }
    else{
	    fprintf(stderr, "libdft: Overwritting dft_driver_otf->rho0 to %le\n", driver_rho0 ) ;
	    dft_driver_otf->rho0 = driver_rho0;
    }
    fprintf(stderr, "libdft: rho0 = %le Angs^-3.\n", driver_rho0 / (GRID_AUTOANG * GRID_AUTOANG * GRID_AUTOANG));
    been_here = 1;
    fprintf(stderr, "libdft: %lf wall clock seconds for initialization.\n", grid_timer_wall_clock_time(&timer));
  }
}

/*
 * Set up the DFT calculation grid.
 *
 * nx      = number of grid points along x (long).
 * ny      = number of grid points along y (long).
 * nz      = number of grid points along z (long).
 * threads = number of parallel execution threads (long).
 *
 * No return value.
 *
 */

EXPORT void dft_driver_setup_grid(long nx, long ny, long nz, double step, long threads) {
  
  check_mode();

  // TODO: fixme
  //  if((nx % 2) || (ny % 2) || (nz % 2)) {
  //    fprintf(stderr, "libdft: Currently works only with array sizes of multiples of two.\n");
  //    exit(1);
  //  }

  driver_nx = nx;
  driver_ny = ny;
  driver_nz = nz;
  driver_step = step;
  driver_halfbox_length = 0.5 * (nx<ny?(nx<nz?nx:nz):(ny<nz?ny:nz)) * step ;
  // Set the origin to its default value if it is not defined
  fprintf(stderr, "libdft: Grid size = (%ld,%ld,%ld) with step = %le.\n", nx, ny, nz, step);
  driver_threads = threads;
}

/*
 * Set up the origin of coordinates for the grids.
 * Can be overwritten for a particular grid calling (r/c)grid3d_set_origin
 *
 */
EXPORT void dft_driver_setup_origin(double x0, double y0, double z0){
	driver_x0 = x0 ;
	driver_y0 = y0 ;
	driver_z0 = z0 ;
	fprintf(stderr, "libdft: Origin of coordinates set at (%le,%le,%le)\n", x0, y0, z0 );
}

/*
 * Set up the momentum of the frame of reference, i.e. a background velocity for the grids.
 * Can be overwritten for a particular grid calling (r/c)grid3d_set_momentum
 *
 */
EXPORT void dft_driver_setup_momentum(double kx0, double ky0, double kz0){
	driver_kx0 = kx0 ;
	driver_ky0 = ky0 ;
	driver_kz0 = kz0 ;
	fprintf(stderr, "libdft: Frame of reference momentum = (%le,%le,%le)\n", kx0, ky0, kz0 );
}

/*
 * Set up the DFT calculation model.
 *
 * dft_model = specify the DFT Hamiltonian to use (see ot.h).
 * iter_mode = iteration mode: 1 = imaginary time, 0 = real time.
 * rho0      = equilibrium density for the liquid (in a.u.; double).
 *             if 0.0, the equilibrium density will be used
 *             when dft_driver_initialize is called.
 *
 * No return value.
 *
 */

EXPORT void dft_driver_setup_model(long dft_model, long iter_mode, double rho0) {

  check_mode();

  driver_dft_model = dft_model;
  driver_iter_mode = iter_mode;
  if(been_here)
	  fprintf(stderr,"libdft: WARNING -- Overwritting driver_rho0 to %le\n", rho0) ;
  driver_rho0 = rho0;
}

/*
 * Set up absorbing boundaries.
 *
 * type    = boundary type: 0 = regular, 1 = absorbing (long).
 * absb    = width of absorbing boundary (double; bohr).
 *           In this region the density tends towards driver_rho0.
 * 
 * No return value.
 *
 */

EXPORT void dft_driver_setup_boundaries(long boundary_type, double absb) {

  check_mode();

  driver_boundary_type = boundary_type;
  driver_abs = absb;
}

/*
 * Impose normal or vortex compatible boundaries.
 *
 * bc = Boundary type:
 *           Normal (0), Vortex along X (1), Vortex along Y (2), Vortex along Z (3), Neumann (4) (int).
 *
 */

EXPORT void dft_driver_setup_boundary_condition(int bc) {

  check_mode();

  driver_bc = bc;
}

/*
 * Modify the value of the damping constant for absorbing boundary.
 *
 * dmp = damping constant (default 0.03).
 *
 */

EXPORT void dft_driver_setup_boundaries_damp(double dmp) {

  check_mode();

  damp = fabs(dmp);
}

/*
 * Set up normalization method for imaginary time propagation.
 *
 * type = how to renormalize the wavefunction: 0 = bulk; 1 = droplet
 *        placed at the origin; 2 = column placed at x = 0.
 * nhe  = desired # of He atoms for types 1 & 2 above (long).
 * frad = fixed volume radius (double). Liquid within this radius
 *        willl be fixed to rho0 to converge to droplet or column.
 * rels = iteration after which the fixing condition will be released.
 *        This should be done for the last few iterations to avoid
 *        artifacts arising from the fixing constraint. Set to zero to disable.
 * 
 */

EXPORT void dft_driver_setup_normalization(long norm_type, long nhe, double frad, long rels) {

  check_mode();

  driver_norm_type = norm_type;
  driver_nhe = nhe;
  driver_rels = rels;
  driver_frad = frad;
}

/*
 * Modify the value of the angular velocity omega (rotating liquid).
 *
 * omega = angular velocity (double);
 *
 */

EXPORT void dft_driver_setup_rotation_omega(double omega) {

  check_mode();

  driver_omega = omega;
}

/*
 * Predict: propagate the given wf in time.
 *
 * what      = what is propagated: 0 = L-He, 1 = other.
 * ext_pot   = present external potential grid (rgrid3d *; input) (NULL = no ext. pot).
 * gwf       = liquid wavefunction to propagate (wf3d *; input).
 *             Note that gwf is NOT changed by this routine.
 * gwfp      = predicted wavefunction (wf3d *; output).
 * potential = storage space for the potential (cgrid3d *; output).
 *             Do not overwrite this before calling the correct routine.
 * tstep     = time step in FS (double; input).
 * iter      = current iteration (long; input).
 *
 * If what == 0, the liquid potential is added automatically.
 *               Also the absorbing boundaries are only active for this.
 * If what == 1, the propagation is carried out only with ext_pot.
 *
 * No return value.
 *
 */

EXPORT inline void dft_driver_propagate_predict(long what, rgrid3d *ext_pot, wf3d *gwf, wf3d *gwfp, cgrid3d *potential, double tstep, long iter) {

  double complex time, htime;
  static double last_tstep = -1.0;

  check_mode();

  if(driver_nx == 0) {
    fprintf(stderr, "libdft: dft_driver not setup.\n");
    exit(1);
  }

  if(last_tstep != tstep) {
    fprintf(stderr, "libdft: New propagation time step = %le fs.\n", tstep);
    last_tstep = tstep;
  }

  tstep /= GRID_AUTOFS;

  if(!iter && driver_iter_mode == 1 && what == 0 && dft_driver_init_wavefunction == 1) {
    fprintf(stderr, "libdft: first imag. time iteration - initializing the wavefunction.\n");
    grid3d_wf_constant(gwf, sqrt(dft_driver_otf->rho0));
  }

  if(driver_iter_mode == 0) {
    time = tstep;
    htime = tstep / 2.0;
  } else {
    time = -I * tstep;
    htime = -I * tstep / 2.0;
  }

  /* droplet & column center release */
  if(driver_rels && iter > driver_rels && driver_norm_type > 0 && what == 0) {
    if(!center_release) fprintf(stderr, "libdft: center release activated.\n");
    center_release = 1;
  } else center_release = 0;
  
  grid_timer_start(&timer);

  /* 1/2 x kinetic */
  switch(dft_driver_kinetic) {
  case DFT_DRIVER_KINETIC_FFT:
    grid3d_wf_propagate_kinetic_fft(gwf, htime);
    break;
  case DFT_DRIVER_KINETIC_CN_DBC:
    if(!cworkspace)
      cworkspace = dft_driver_alloc_cgrid();
    grid3d_wf_propagate_kinetic_cn_dbc(gwf, htime, cworkspace);
    break;
  case DFT_DRIVER_KINETIC_CN_NBC:
    if(!cworkspace)
      cworkspace = dft_driver_alloc_cgrid();
    grid3d_wf_propagate_kinetic_cn_nbc(gwf, htime, cworkspace);
    break;
  case DFT_DRIVER_KINETIC_CN_NBC_ROT:
    if(!cworkspace)
      cworkspace = dft_driver_alloc_cgrid();
    grid3d_wf_propagate_kinetic_cn_nbc_rot(gwf, htime, driver_omega, cworkspace);
    break;
  case DFT_DRIVER_KINETIC_CN_PBC:
    if(!cworkspace)
      cworkspace = dft_driver_alloc_cgrid();
    grid3d_wf_propagate_kinetic_cn_pbc(gwf, htime, cworkspace);
    break;
#if 0
  case DFT_DRIVER_KINETIC_CN_APBC:
    if(!cworkspace)
      cworkspace = dft_driver_alloc_cgrid();
    grid3d_wf_propagate_kinetic_cn_apbc(gwf, htime, cworkspace);
    break;
#endif
  default:
    fprintf(stderr, "libdft: Unknown BC for kinetic energy propagation.\n");
    exit(1);
  }
  if(driver_iter_mode) scale_wf(what, dft_driver_otf, gwf);
  cgrid3d_copy(gwfp->grid, gwf->grid);

  /* predict */
  cgrid3d_zero(potential);  // new
  if(!what) {
    grid3d_wf_density(gwfp, density);
    dft_ot3d_potential(dft_driver_otf, potential, gwfp, density, workspace1, workspace2, workspace3, workspace4, workspace5, workspace6, workspace7, workspace8, workspace9);
  }
  /* absorbing boundary - imaginary potential */
  if(driver_boundary_type == 1 && !what && !driver_iter_mode) {
    fprintf(stderr, "libdft: Predict - absorbing boundary for helium ; imaginary potential.\n");
    grid3d_wf_absorb(potential, density, driver_rho0, region_func, workspace1, (driver_iter_mode==1) ? I:1.0);
  }
 /* External potential for Helium */
  /* Im - He contribution */
  if(ext_pot) grid3d_add_real_to_complex_re(potential, ext_pot);

  /* potential */
  grid3d_wf_propagate_potential(gwfp, potential, time);
  if(driver_iter_mode) scale_wf(what, dft_driver_otf, gwfp);

  /* wavefunction damping  */
  if(driver_boundary_type == 2 && !what && !driver_iter_mode) {
    fprintf(stderr, "libdft: Predict - absorbing boundary for helium ; wavefunction damping.\n");
    if(!cworkspace)
      cworkspace = dft_driver_alloc_cgrid();
    grid3d_damp_wf(gwfp, driver_rho0, damp, cregion_func, cworkspace, NULL) ; 
  }
}

/*
 * Correct: propagate the given wf in time.
 *
 * what      = what is propagated: 0 = L-He, 1 = other.
 * ext_pot   = present external potential grid (rgrid3d *) (NULL = no ext. pot).
 * gwf       = liquid wavefunction to propagate (wf3d *).
 *             Note that gwf is NOT changed by this routine.
 * gwfp      = predicted wavefunction (wf3d *; output).
 * potential = storage space for the potential (cgrid3d *; output).
 * tstep     = time step in FS (double).
 * iter      = current iteration (long).
 *
 * If what == 0, the liquid potential is added automatically.
 * If what == 1, the propagation is carried out only with et_pot.
 *
 * No return value.
 *
 */

EXPORT inline void dft_driver_propagate_correct(long what, rgrid3d *ext_pot, wf3d *gwf, wf3d *gwfp, cgrid3d *potential, double tstep, long iter) {

  double complex time, htime;
  
  check_mode();

  tstep /= GRID_AUTOFS;
  
  if(driver_iter_mode == 0) {
    time = tstep;
    htime = tstep / 2.0;
  } else {
    time = -I * tstep;
    htime = -I * tstep / 2.0;
  }
  
  /* correct */
  if(!what) {
    grid3d_wf_density(gwfp, density);
    // no zeroing - add to the previous potential to get avg (new)
    dft_ot3d_potential(dft_driver_otf, potential, gwfp, density, workspace1, workspace2, workspace3, workspace4, workspace5, workspace6, workspace7, workspace8, workspace9);
  }
  /* absorbing boundary */
  if(driver_boundary_type == 1 && !what && !driver_iter_mode) {
    fprintf(stderr, "libdft: Correct - absorbing boundary for helium ; imaginary potential.\n");
    grid3d_wf_absorb(potential, density, driver_rho0, region_func, workspace1, (driver_iter_mode==1) ? I:1.0);
  }  
  /* External potential for Helium */
  /* Im - He contribution (new) */
  if(ext_pot) grid3d_add_real_to_complex_re(potential, ext_pot);
  /* average of future and current (new) */
  cgrid3d_multiply(potential, 0.5);
  
  /* potential */
  grid3d_wf_propagate_potential(gwf, potential, time);
  if(driver_iter_mode) scale_wf(what, dft_driver_otf, gwf);
  
  /* 1/2 x kinetic */
  switch(dft_driver_kinetic) {
  case DFT_DRIVER_KINETIC_FFT:
    grid3d_wf_propagate_kinetic_fft(gwf, htime);
    break;
  case DFT_DRIVER_KINETIC_CN_DBC:
    grid3d_wf_propagate_kinetic_cn_dbc(gwf, htime, cworkspace);
    break;
  case DFT_DRIVER_KINETIC_CN_NBC:
    grid3d_wf_propagate_kinetic_cn_nbc(gwf, htime, cworkspace);
    break;
  case DFT_DRIVER_KINETIC_CN_NBC_ROT:
    if(!cworkspace)
      cworkspace = dft_driver_alloc_cgrid();
    grid3d_wf_propagate_kinetic_cn_nbc_rot(gwf, htime, driver_omega, cworkspace);
    break;
  case DFT_DRIVER_KINETIC_CN_PBC:
    grid3d_wf_propagate_kinetic_cn_pbc(gwf, htime, cworkspace);
    break;
#if 0
  case DFT_DRIVER_KINETIC_CN_APBC:
    grid3d_wf_propagate_kinetic_cn_apbc(gwf, htime, cworkspace);
    break;
#endif
  default:
    fprintf(stderr, "libdft: Unknown BC for kinetic energy propagation.\n");
    exit(1);
  }
  if(driver_iter_mode) scale_wf(what, dft_driver_otf, gwf);
  
  /* wavefunction damping  */
  if(driver_boundary_type == 2 && !what && !driver_iter_mode) {
    fprintf(stderr, "libdft: Correct - absorbing boundary for helium ; wavefunction damping.\n");
    if(!cworkspace)
      cworkspace = dft_driver_alloc_cgrid();
    grid3d_damp_wf(gwf, driver_rho0, damp, cregion_func, cworkspace,  NULL) ; 
  }
  fprintf(stderr, "libdft: Iteration %ld took %lf wall clock seconds.\n", iter, grid_timer_wall_clock_time(&timer));
  fflush(stdout);
}

/*
 * Prepare for convoluting potential and density.
 *
 * pot  = potential to be convoluted with (rgrid3d *).
 * dens = denisity to be convoluted with (rgrid3d *).
 *
 * This must be called before cgrid3d_driver_convolute_eval().
 * Both pot and dens are overwritten with their FFTs.
 * if either is specified as NULL, no transform is done for that grid.
 *
 * No return value.
 *
 */

EXPORT void dft_driver_convolution_prepare(rgrid3d *pot, rgrid3d *dens) {

  check_mode();

  if(pot) rgrid3d_fft(pot);
  if(dens) rgrid3d_fft(dens);
}

/*
 * Convolute density and potential.
 *
 * out  = output from convolution (cgrid3d *).
 * pot  = potential grid that has been prepared with cgrid3d_driver_convolute_prepare().
 * dens = density against which has been prepared with cgrid3d_driver_convolute_prepare().
 *
 * No return value.
 *
 */

EXPORT void dft_driver_convolution_eval(rgrid3d *out, rgrid3d *pot, rgrid3d *dens) {

  check_mode();

  rgrid3d_fft_convolute(out, pot, dens);
  rgrid3d_inverse_fft(out);
}

/*
 * Allocate a complex grid.
 *
 * Returns a pointer to the allocated grid.
 *
 */

EXPORT cgrid3d *dft_driver_alloc_cgrid() {

  double complex (*grid_type)(const cgrid3d *, long, long, long);
  cgrid3d *tmp ;

  check_mode();
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
    grid_type = CGRID3D_NEUMANN_BOUNDARY ;
    break;
  default:
    fprintf(stderr, "libdft: Illegal boundary type.\n");
    exit(1);
  }

  tmp = cgrid3d_alloc(driver_nx, driver_ny, driver_nz, driver_step, grid_type, 0);
  cgrid3d_set_origin(tmp, driver_x0, driver_y0, driver_z0) ;
  cgrid3d_set_momentum(tmp, driver_kx0, driver_ky0, driver_kz0) ;
  return tmp ;
}

/*
 * Allocate a real grid.
 *
 * Returns a pointer to the allocated grid.
 *
 * Note: either if the condition is Neumann b.c. or vortex b.c. for the
 * wavefunction, the real grids such as density always have Neumann b.c.
 *
 */

EXPORT rgrid3d *dft_driver_alloc_rgrid() {

  double (*grid_type)(const rgrid3d *, long, long, long);
  rgrid3d *tmp ;

  check_mode();

  if(driver_nx == 0) {
    fprintf(stderr, "libdft: dft_driver routines must be initialized first.\n");
    exit(1);
  }

  switch(driver_bc) {
  case DFT_DRIVER_BC_NORMAL: 
    grid_type = RGRID3D_PERIODIC_BOUNDARY;
    break;
  case DFT_DRIVER_BC_X:
  case DFT_DRIVER_BC_Y:
  case DFT_DRIVER_BC_Z:
  case DFT_DRIVER_BC_NEUMANN:
    grid_type = RGRID3D_NEUMANN_BOUNDARY ;
    break;
  default:
    fprintf(stderr, "libdft: Illegal boundary type.\n");
    exit(1);
  }

  tmp = rgrid3d_alloc(driver_nx, driver_ny, driver_nz, driver_step, grid_type, 0);
  rgrid3d_set_origin(tmp, driver_x0, driver_y0, driver_z0) ;
  rgrid3d_set_momentum(tmp, driver_kx0, driver_ky0, driver_kz0) ;
  return tmp ;
}

/*
 * Allocate a wavefunction (initialized to sqrt(rho0)).
 *
 * mass = particle mass in a.u. (double).
 *
 * Returns pointer to the wavefunction.
 *
 */

EXPORT wf3d *dft_driver_alloc_wavefunction(double mass) {

  wf3d *tmp;
  int grid_type;
  
  check_mode();

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

  tmp = grid3d_wf_alloc(driver_nx, driver_ny, driver_nz, driver_step, mass, grid_type, WF3D_2ND_ORDER_PROPAGATOR);
  cgrid3d_set_origin(tmp->grid, driver_x0, driver_y0, driver_z0) ;
  cgrid3d_set_momentum(tmp->grid, driver_kx0, driver_ky0, driver_kz0) ;
  cgrid3d_constant(tmp->grid, sqrt(driver_rho0));
  return tmp;
}

/*
 * Initialize a wavefunction to sqrt of a gaussian function.
 * Useful function for generating an initial guess for impurities.
 *
 * dft   = Wavefunction to be initialized (cgrid3d *; input/output).
 * cx    = Gaussian center alogn x (double; input).
 * cy    = Gaussian center alogn y (double; input).
 * cz    = Gaussian center alogn z (double; input).
 * width = Gaussian width (double; input).
 *
 */

struct asd {
  double cx, cy, cz;
  double zp;
};
  

static double complex dft_gauss(void *ptr, double x, double y, double z) {

  struct asd *lp = (struct asd *) ptr;
  double zp = lp->zp, cx = x - lp->cx, cy = y - lp->cy, cz = z - lp->cz;

  return sqrt(pow(zp * zp * M_PI / M_LN2, -3.0/2.0) * exp(-M_LN2 * (cx * cx + cy * cy + cz * cz) / (zp * zp)));
}

EXPORT void dft_driver_gaussian_wavefunction(wf3d *dst, double cx, double cy, double cz, double width) {

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
 * grid = place to store the read density (rgrid3d *).
 * file = filename for the file (char *). Note: the .grd extension must NOT be given.
 *
 * No return value.
 *
 */

EXPORT void dft_driver_read_density(rgrid3d *grid, char *file) {

  FILE *fp;
  char buf[512];

  check_mode();

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
 * grid = density grid (rgrid3d *).
 * base = Basename for the output file (char *).
 *
 * No return value.
 *
 */

EXPORT void dft_driver_write_density(rgrid3d *grid, char *base) {

  FILE *fp;
  char file[2048];
  long i, j, k;
  double x, y, z;

  //debug
  //check_mode();

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
    x = (i - grid->nx/2.0) * grid->step;
    fprintf(fp, "%le %le\n", x, rgrid3d_value_at_index(grid, i, j, k));
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
    y = (j - grid->ny/2.0) * grid->step;
    fprintf(fp, "%le %le\n", y, rgrid3d_value_at_index(grid, i, j, k));
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
    z = (k - grid->nz/2.0) * grid->step;
    fprintf(fp, "%le %le\n", z, rgrid3d_value_at_index(grid, i, j, k));
  }
  fclose(fp);
  fprintf(stderr, "libdft: Density written to %s.\n", file);
}

/*
 * Write output to ascii & binary files.
 * .grd file is the full grid in binary format.
 * .x ASCII file cut along (x, 0.0, 0.0)
 * .y ASCII file cut along (0.0, y, 0.0)
 * .z ASCII file cut along (0.0, 0.0, z)
 *
 * wf = wf with the pase (rgrid3d *).
 * base = Basename for the output file (char *).
 *
 * No return value.
 *
 */

EXPORT void dft_driver_write_phase(wf3d *wf, char *base) {

  FILE *fp;
  char file[2048];
  long i, j, k;
  cgrid3d *grid = wf->grid;
  double complex tmp;
  rgrid3d *phase;
  double x, y, z;
  long nx = grid->nx, ny = grid->ny, nz = grid->nz;

  check_mode();

  phase = dft_driver_alloc_rgrid();
  for(i = 0; i < nx; i++)
    for(j = 0; j < ny; j++)
      for(k = 0; k < nz; k++) {
	tmp = cgrid3d_value_at_index(grid, i, j, k);
	if(cabs(tmp) < 1E-6)
	  phase->value[i * ny * nz + j * nz + k] = 0.0;
	else
	  phase->value[i * ny * nz + j * nz + k] =  cimag(clog(tmp / cabs(tmp)));
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
    x = (i - grid->nx/2.0) * grid->step;
    fprintf(fp, "%le %le\n", x, rgrid3d_value_at_index(phase, i, j, k));
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
    y = (j - grid->ny/2.0) * grid->step;
    fprintf(fp, "%le %le\n", y, rgrid3d_value_at_index(phase, i, j, k));
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
    z = (k - grid->nz/2.0) * grid->step;
    fprintf(fp, "%le %le\n", z, rgrid3d_value_at_index(phase, i, j, k));
  }
  fclose(fp);
  fprintf(stderr, "libdft: Density written to %s.\n", file);
  rgrid3d_free(phase);
}



/*
 * Write two-dimensional slices of a grid
 * .xy ASCII file cut along z=0.
 * .yz ASCII file cut along x=0.
 * .zx ASCII file cut along y=0.
 *
 * grid = density grid (rgrid3d *).
 * base = Basename for the output file (char *).
 *
 * No return value.
 *
 */

EXPORT void dft_driver_write_2d_density(rgrid3d *grid, char *base) {
  FILE *fp;
  char file[2048];
  long i, j, k;
  double x0 = grid->x0 , y0 = grid->y0 , z0 = grid->z0 ;
  long nx = grid->nx , ny = grid->ny , nz = grid->nz ;
  double step = grid->step ;
  double x, y, z;

/*----- X Y -----*/
  sprintf(file, "%s.xy", base);
  if(!(fp = fopen(file, "w"))) {
    fprintf(stderr, "libdft: Can't open %s for writing.\n", file);
    exit(1);
  }
  for(i = 0; i < nx; i++) {
    x = (i - nx/2) * step - x0;
    for(j = 0; j < ny; j++) {
	y = (j - ny/2) * step - y0;
        fprintf(fp, "%le\t%le\t%le\n", x, y, rgrid3d_value_at_index(grid, i, j, nz/2));	
    } fprintf(fp,"\n") ;
  }
  fclose(fp);

/*----- Y Z -----*/
  sprintf(file, "%s.yz", base);
  if(!(fp = fopen(file, "w"))) {
    fprintf(stderr, "libdft: Can't open %s for writing.\n", file);
    exit(1);
  }
  for(j = 0; j < ny; j++) {
    y = (j - ny/2) * step - y0;
    for(k = 0; k < nz; k++) {
	z = (k - nz/2) * step - z0;
        fprintf(fp, "%le\t%le\t%le\n", y, z, rgrid3d_value_at_index(grid, nx/2, j, k));	
    } fprintf(fp,"\n") ;
  }
  fclose(fp);

/*----- Z X -----*/
  sprintf(file, "%s.zx", base);
  if(!(fp = fopen(file, "w"))) {
    fprintf(stderr, "libdft: Can't open %s for writing.\n", file);
    exit(1);
  }
  for(k = 0; k < nz; k++) {
    z = (k - nz/2) * step - z0;
    for(i = 0; i < nx; i++) {
	x = (i - nx/2) * step - z0;
        fprintf(fp, "%le\t%le\t%le\n", z, x, rgrid3d_value_at_index(grid, i, ny/2, k));	
    } fprintf(fp,"\n") ; 
  }
  fclose(fp);

  fprintf(stderr, "libdft: 2D slices of density written to %s.\n", file);
}

/*
 * Write two-dimensional vector slices of a vector grid (three grids)
 * .xy ASCII file cut along z=0.
 * .yz ASCII file cut along x=0.
 * .zx ASCII file cut along y=0.
 *
 * px = x-component of vector grid (rgrid3d *).
 * py = y-component of vector grid (rgrid3d *).
 * pz = z-component of vector grid (rgrid3d *).
 *
 * base = Basename for the output file (char *).
 *
 * No return value.
 *
 */
EXPORT void dft_driver_write_vectorfield(rgrid3d *px, rgrid3d *py, rgrid3d *pz, char *base){
    	FILE *fp;
    	char file[2048];
    	long i, j, k;
  	double x0 = px->x0 , y0 = px->y0 , z0 = px->z0 , step = px->step ;
  	long nx = px->nx , ny = px->ny , nz = px->nz ;
    	double x, y, z;

	check_mode();
/*----- X Y -----*/	
  sprintf(file, "%s.xy", base);
  if(!(fp = fopen(file, "w"))) {
    fprintf(stderr, "libdft: Can't open %s for writing.\n", file);
    exit(1);
  }
  k = nz / 2 ;
  for(i = 0; i < nx; i++) {
    x = (i - nx/2) * step - x0;
    for(j = 0; j < ny; j++) {
	y = (j - ny/2) * step - y0;
        fprintf(fp, "%le\t%le\t%le\t%le\n", x, y, rgrid3d_value_at_index(px, i, j, k), rgrid3d_value_at_index(py, i, j, k));	
    } fprintf(fp,"\n") ;
  }
  fclose(fp);

/*----- Y Z -----*/
  sprintf(file, "%s.yz", base);
  if(!(fp = fopen(file, "w"))) {
    fprintf(stderr, "libdft: Can't open %s for writing.\n", file);
    exit(1);
  }
  i = nx / 2 ;
  for(j = 0; j < ny; j++) {
    y = (j - ny/2) * step - y0;
    for(k = 0; k < nz; k++) {
	z = (k - nz/2) * step - z0;
        fprintf(fp, "%le\t%le\t%le\t%le\n", y, z, rgrid3d_value_at_index(py, i, j, k), rgrid3d_value_at_index(pz, i, j, k));	
    } fprintf(fp,"\n") ;
  }
  fclose(fp);

/*----- Z X -----*/
  sprintf(file, "%s.zx", base);
  if(!(fp = fopen(file, "w"))) {
    fprintf(stderr, "libdft: Can't open %s for writing.\n", file);
    exit(1);
  }
  j = ny / 2 ;
  for(k = 0; k < nz; k++) {
    z = (k - nz/2) * step - z0;
    for(i = 0; i < nx; i++) {
	x = (i - nx/2) * step - z0;
        fprintf(fp, "%le\t%le\t%le\t%le\n", z, x, rgrid3d_value_at_index(pz, i, j, k), rgrid3d_value_at_index(px, i, j, k));	
    } fprintf(fp,"\n") ; 
  }
  fclose(fp);

  fprintf(stderr, "libdft: vector 2D slices of density written to %s.\n", file);

}


/*
 * Write two-dimensional vector slices of a probability current 
 * .xy ASCII file cut along z=0.
 * .yz ASCII file cut along x=0.
 * .zx ASCII file cut along y=0.
 *
 * wf = wavefunction (wf3d, input)
 * base = Basename for the output file (char *).
 *
 * No return value.
 *
 * Uses workspace 1-3
 */
EXPORT void dft_driver_write_current(wf3d *wf, char *base){
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
 * base = Basename for the output file (char *).
 *
 * No return value.
 *
 * Uses workspace 1-3
 */
EXPORT void dft_driver_write_velocity(wf3d *wf, char *base){
	dft_driver_veloc_field(wf, workspace1, workspace2, workspace3);
	dft_driver_write_vectorfield( workspace1, workspace2, workspace3, base);
}


/*
 * Read in a grid from a binary file (.grd).
 *
 * grid = grid where the data is placed (cgrid3d *).
 * file = filename for the file (char *). Note: the .grd extension must be given.
 *
 * No return value.
 *
 */

EXPORT void dft_driver_read_grid(cgrid3d *grid, char *file) {

  FILE *fp;

  check_mode();

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
 * grid = grid to be written (cgrid3d *).
 * base = Basename for the output file (char *).
 *
 * No return value.
 *
 */

EXPORT void dft_driver_write_grid(cgrid3d *grid, char *base) {

  FILE *fp;
  char file[2048];
  long i, j, k;
  double x, y, z;

  check_mode();

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
  j = driver_ny / 2;
  k = driver_nz / 2;
  for(i = 0; i < driver_nx; i++) { 
    x = (i - driver_nx/2.0) * driver_step;
    fprintf(fp, "%le %le %le\n", x, creal(cgrid3d_value_at_index(grid, i, j, k)), cimag(cgrid3d_value_at_index(grid, i, j, k)));
  }
  fclose(fp);

  sprintf(file, "%s.y", base);
  if(!(fp = fopen(file, "w"))) {
    fprintf(stderr, "libdft: Can't open %s for writing.\n", file);
    exit(1);
  }
  i = driver_nx / 2;
  k = driver_nz / 2;
  for(j = 0; j < driver_ny; j++) {
    y = (j - driver_ny/2.0) * driver_step;
    fprintf(fp, "%le %le %le\n", y, creal(cgrid3d_value_at_index(grid, i, j, k)), cimag(cgrid3d_value_at_index(grid, i, j, k)));
  }
  fclose(fp);

  sprintf(file, "%s.z", base);
  if(!(fp = fopen(file, "w"))) {
    fprintf(stderr, "libdft: Can't open %s for writing.\n", file);
    exit(1);
  }
  i = driver_nx / 2;
  j = driver_ny / 2;
  for(k = 0; k < driver_nz; k++) {
    z = (k - driver_nz/2.0) * driver_step;
    fprintf(fp, "%le %le %le\n", z, creal(cgrid3d_value_at_index(grid, i, j, k)), cimag(cgrid3d_value_at_index(grid, i, j, k)));
  }
  fclose(fp);
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

EXPORT double dft_driver_energy(wf3d *gwf, rgrid3d *ext_pot) {
  return dft_driver_potential_energy(gwf, ext_pot) + dft_driver_kinetic_energy(gwf) ;
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

EXPORT double dft_driver_potential_energy(wf3d *gwf, rgrid3d *ext_pot) {

  /* we may need more memory for this... */

  check_mode();
  if(!workspace7) workspace7 = dft_driver_alloc_rgrid();
  if(!workspace8) workspace8 = dft_driver_alloc_rgrid();
  if(!workspace9) workspace9 = dft_driver_alloc_rgrid();
  grid3d_wf_density(gwf, density);
  dft_ot3d_energy_density(dft_driver_otf, workspace9, density, workspace1, workspace2, workspace3, workspace4, workspace5, workspace6, workspace7, workspace8);
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

EXPORT double dft_driver_kinetic_energy(wf3d *gwf) {
  
  check_mode();

  if(!cworkspace)
    cworkspace = dft_driver_alloc_cgrid();

  return grid3d_wf_energy(gwf, NULL, cworkspace);

}

/*
 * Calculate the energy from the rotation constrain,
 * ie -<omega*L>.
 *
 * gwf     = wavefunction for the system (wf3d *; input).
 * omega_x = angular frequency in a.u., x-axis (double)
 * omega_y = angular frequency in a.u., y-axis (double)
 * omega_z = angular frequency in a.u., z-axis (double)
 */
EXPORT double dft_driver_rotation_energy(wf3d *wf, double omega_x, double omega_y, double omega_z){
	double lx, ly, lz;
	dft_driver_L( wf, &lx, &ly, &lz) ;
	return - (omega_x * lx) - (omega_y * ly) - (omega_z * lz) ;
}


/*
 * Calculate the energy in a certain region (box).
 *
 * gwf     = wavefunction for the system (wf3d *; input).
 * ext_pot = external potential grid (rgrid3d *; input).
 * xl   = lower limit for x (double).
 * xu   = upper limit for x (double).
 * yl   = lower limit for y (double).
 * yu   = upper limit for y (double).
 * zl   = lower limit for z (double).
 * zu   = upper limit for z (double).
 *
 * Return value = energy for the box (in a.u.).
 *
 * Note: the backflow is not included in the energy density calculation.
 *
 */
EXPORT double dft_driver_energy_region(wf3d *gwf, rgrid3d *ext_pot, double xl, double xu, double yl, double yu, double zl, double zu) {

  double energy;

  check_mode();

  /* we may need more memory for this... */

  check_mode();
  if(!workspace7) workspace7 = dft_driver_alloc_rgrid();
  if(!workspace8) workspace8 = dft_driver_alloc_rgrid();
  if(!workspace9) workspace9 = dft_driver_alloc_rgrid();
  grid3d_wf_density(gwf, density);
  dft_ot3d_energy_density(dft_driver_otf, workspace9, density, workspace1, workspace2, workspace3, workspace4, workspace5, workspace6, workspace7, workspace8);
  if(ext_pot) rgrid3d_add_scaled_product(workspace9, 1.0, density, ext_pot);
  energy = rgrid3d_integral_region(workspace9, xl, xu, yl, yu, zl, zu);
  if(!cworkspace)
    cworkspace = dft_driver_alloc_cgrid();
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

EXPORT double dft_driver_natoms(wf3d *gwf) {

  check_mode();

  return creal(cgrid3d_integral_of_square(gwf->grid));
}

/*
 * Evaluate absorption/emission spectrum using the Andersson
 * expression. No zero-point correction for the impurity.
 *
 * density  = Current liquid density (rgrid3d *; input).
 * tstep    = Time step for constructing the time correlation function
 *            (double; input in fs). Typically around 1 fs.
 * endtime  = End time in constructing the time correlation function
 *            (double; input in fs). Typically less than 10,000 fs.
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

static double complex dft_eval_exp(double complex a) { /* a contains t */

  return (1.0 - cexp(-I * a));
}

static double complex dft_do_int(rgrid3d *dens, rgrid3d *dpot, double t, cgrid3d *wrk) {

  grid3d_real_to_complex_re(wrk, dpot);
  cgrid3d_multiply(wrk, t);
  cgrid3d_operate_one(wrk, wrk, dft_eval_exp);
  grid3d_product_complex_with_real(wrk, dens);
  return cgrid3d_integral(wrk);            // debug: This should have minus in front?! Sign error elsewhere? (does not appear in ZP?!)
}

EXPORT cgrid1d *dft_driver_spectrum(rgrid3d *density, double tstep, double endtime, int finalave, char *finalx, char *finaly, char *finalz, int initialave, char *initialx, char *initialy, char *initialz) {

  rgrid3d *dpot;
  cgrid3d *wrk[256];
  static cgrid1d *corr = NULL;
  double t;
  long i, ntime;
  static long prev_ntime = -1;

  check_mode();

  endtime /= GRID_AUTOFS;
  tstep /= GRID_AUTOFS;
  ntime = (1 + (int) (endtime / tstep));
  dpot = rgrid3d_alloc(density->nx, density->ny, density->nz, density->step, RGRID3D_PERIODIC_BOUNDARY, 0);
  for (i = 0; i < omp_get_max_threads(); i++)
    wrk[i] = cgrid3d_alloc(density->nx, density->ny, density->nz, density->step, CGRID3D_PERIODIC_BOUNDARY, 0);
  if(ntime != prev_ntime) {
    if(corr) cgrid1d_free(corr);
    corr = cgrid1d_alloc(ntime, 0.1, CGRID1D_PERIODIC_BOUNDARY, 0);
    prev_ntime = ntime;
  }

  dft_common_potential_map(finalave, finalx, finaly, finalz, workspace1);
  dft_common_potential_map(initialave, initialx, initialy, initialz, workspace2);
  rgrid3d_difference(dpot, workspace1, workspace2); /* final - initial */
  
  rgrid3d_product(workspace1, dpot, density);
  fprintf(stderr, "libdft: Average shift = %le.\n", rgrid3d_integral(workspace1) * GRID_AUTOCM1);

#pragma omp parallel for firstprivate(stderr,tstep,ntime,density,dpot,corr,wrk) private(i,t) default(none) schedule(runtime)
  for(i = 0; i < ntime; i++) {
    t = tstep * (double) i;
    corr->value[i] = cexp(dft_do_int(density, dpot, t, wrk[omp_get_thread_num()])) * pow(-1.0, (double) i);
    //    fprintf(stderr,"libdft: Corr(%le fs) = %le %le\n", t * GRID_AUTOFS, creal(corr->value[i]), cimag(corr->value[i]));
  }
  cgrid1d_fft(corr);
  for (i = 0; i < corr->nx; i++)
    corr->value[i] = cabs(corr->value[i]);
  
  corr->step = GRID_HZTOCM1 / (tstep * GRID_AUTOFS * 1E-15 * (double) ntime);

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
 *            (double; input in fs). Typically around 1 fs.
 * endtime  = End time in constructing the time correlation function
 *            (double; input in fs). Typically less than 10,000 fs.
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

static void do_gexp(cgrid3d *gexp, rgrid3d *dpot, double t) {

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

static double complex dft_do_int2(cgrid3d *gexp, rgrid3d *imdens, cgrid3d *fft_dens, double t, cgrid3d *wrk) {

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

EXPORT cgrid1d *dft_driver_spectrum_zp(rgrid3d *density, rgrid3d *imdensity, double tstep, double endtime, int upperave, char *upperx, char *uppery, char *upperz, int lowerave, char *lowerx, char *lowery, char *lowerz) {

  cgrid3d *wrk, *fft_density, *gexp;
  rgrid3d *dpot;
  static cgrid1d *corr = NULL;
  double t;
  long i, ntime;
  static long prev_ntime = -1;

  check_mode();

  endtime /= GRID_AUTOFS;
  tstep /= GRID_AUTOFS;
  ntime = (1 + (int) (endtime / tstep));
  dpot = rgrid3d_alloc(density->nx, density->ny, density->nz, density->step, RGRID3D_PERIODIC_BOUNDARY, 0);
  fft_density = cgrid3d_alloc(density->nx, density->ny, density->nz, density->step, CGRID3D_PERIODIC_BOUNDARY, 0);
  wrk = cgrid3d_alloc(density->nx, density->ny, density->nz, density->step, CGRID3D_PERIODIC_BOUNDARY, 0);
  gexp = cgrid3d_alloc(density->nx, density->ny, density->nz, density->step, CGRID3D_PERIODIC_BOUNDARY, 0);
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
    t = tstep * (double) i;
    do_gexp(gexp, dpot, t); /* gexp grid + FFT */
    corr->value[i] = cexp(dft_do_int2(gexp, imdensity, fft_density, t, wrk)) * pow(-1.0, (double) i);
    fprintf(stderr,"libdft: Corr(%le fs) = %le %le\n", t * GRID_AUTOFS, creal(corr->value[i]), cimag(corr->value[i]));
  }
  cgrid1d_fft(corr);
  for (i = 0; i < corr->nx; i++)
    corr->value[i] = cabs(corr->value[i]);
  
  corr->step = GRID_HZTOCM1 / (tstep * GRID_AUTOFS * 1E-15 * (double) ntime);
  
  rgrid3d_free(dpot);
  cgrid3d_free(fft_density);
  cgrid3d_free(wrk);
  cgrid3d_free(gexp);
  
  return corr;
}

/*
 * Routines for evaluating the dynamic lineshape (CPL 396, 155 (2004)).
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
 * gwf      = Initial order parameter (used to get the initial density) (wf3d *).
 * nt       = Maximum number of time steps to be collected (long).
 * zerofill = How many zeros to fill in before FFT (int).
 * upperave = Averaging on the upper state (see dft_driver_potential_map()) (int).
 * upperx   = Upper potential file name along-x (char *).
 * uppery   = Upper potential file name along-y (char *).
 * upperz   = Upper potential file name along-z (char *).
 * lowerave = Averaging on the lower state (see dft_driver_potential_map()) (int).
 * lowerx   = Lower potential file name along-x (char *).
 * lowery   = Lower potential file name along-y (char *).
 * lowerz   = Lower potential file name along-z (char *).
 *
 */

static rgrid3d *xxdiff = NULL;
static cgrid1d *tdpot = NULL;
static long ntime, cur_time, zerofill;

EXPORT void dft_driver_spectrum_init(wf3d *gwf, long nt, long zf, int upperave, char *upperx, char *uppery, char *upperz, int lowerave, char *lowerx, char *lowery, char *lowerz) {

  check_mode();

  cur_time = 0;
  ntime = nt;
  zerofill = zf;
  if(!xxdiff)
    xxdiff = rgrid3d_alloc(driver_nx, driver_ny, driver_nz, driver_step, RGRID3D_PERIODIC_BOUNDARY, 0);
  if(!tdpot)
    tdpot = cgrid1d_alloc(ntime + zf, 0.1, CGRID1D_PERIODIC_BOUNDARY, 0);
  fprintf(stderr, "libdft: Upper level potential.\n");
  dft_common_potential_map(upperave, upperx, uppery, upperz, workspace1);
  fprintf(stderr, "libdft: Lower level potential.\n");
  dft_common_potential_map(lowerave, lowerx, lowery, lowerz, workspace2);
  fprintf(stderr, "libdft: spectrum init complete.\n");
  rgrid3d_difference(xxdiff, workspace1, workspace2);
}

/*
 * Collect the time dependent difference energy data. Same as above but with direct
 * grid input for potentials.
 * 
 * gwf      = Initial order parameter (used to get the initial density) (wf3d *).
 * nt       = Maximum number of time steps to be collected (long).
 * zerofill = How many zeros to fill in before FFT (int).
 * upper    = upper state potential grid (rgrid3d *).
 * lower    = lower state potential grdi (rgrid3d *).
 *
 */

EXPORT void dft_driver_spectrum_init2(wf3d *gwf, long nt, long zf, rgrid3d *upper, rgrid3d *lower) {

  check_mode();

  cur_time = 0;
  ntime = nt;
  zerofill = zf;
  if(!xxdiff)
    xxdiff = rgrid3d_alloc(driver_nx, driver_ny, driver_nz, driver_step, RGRID3D_PERIODIC_BOUNDARY, 0);
  if(!tdpot)
    tdpot = cgrid1d_alloc(ntime + zf, 0.1, CGRID1D_PERIODIC_BOUNDARY, 0);
  rgrid3d_difference(xxdiff, upper, lower);
}

/*
 * Collect the difference energy data. 
 *
 * gwf     = the current wavefunction (used for calculating the liquid density) (wf3d *).
 *
 */

EXPORT void dft_driver_spectrum_collect(wf3d *gwf) {

  check_mode();

  if(cur_time > ntime) {
    fprintf(stderr, "libdft: initialized with too few points (spectrum collect).\n");
    exit(1);
  }
  grid3d_wf_density(gwf, workspace1);
  rgrid3d_product(workspace1, workspace1, xxdiff);
  tdpot->value[cur_time] = rgrid3d_integral(workspace1);

  fprintf(stderr, "libdft: spectrum collect complete (point = %ld, value = %le K).\n", cur_time, creal(tdpot->value[cur_time]) * GRID_AUTOK);
  cur_time++;
}

/*
 * Collect the difference energy data (with impurity zero-point). 
 *
 * gwf     = the current wavefunction (used for calculating the liquid density) (wf3d *).
 * igwf    = the current wavefunction (used for calculating the impurity density) (wf3d *).
 *
 */

EXPORT void dft_driver_spectrum_collect_zp(wf3d *gwf, wf3d *igwf) {

  check_mode();

  if(cur_time > ntime) {
    fprintf(stderr, "libdft: initialized with too few points (spectrum collect).\n");
    exit(1);
  }
  grid3d_wf_density(gwf, workspace1);
  rgrid3d_copy(workspace2, xxdiff);
  dft_driver_convolution_prepare(workspace2, workspace1);
  dft_driver_convolution_eval(workspace3, workspace1, workspace2);
  grid3d_wf_density(igwf, workspace1);
  rgrid3d_product(workspace1, workspace3, workspace1);
  tdpot->value[cur_time] = rgrid3d_integral(workspace1);

  fprintf(stderr, "libdft: spectrum collect complete (point = %ld, value = %le K).\n", cur_time, creal(tdpot->value[cur_time]) * GRID_AUTOK);
  cur_time++;
}

/*
 * Evaluate the spectrum.
 *
 * tstep       = Time step length at which the energy difference data was collected
 *               (fs; usually the simulation time step) (double).
 * zero_offset = Frequency offset (to account for 1/omega dependency); usually zero.
 * tc          = Exponential decay time constant (fs; double).
 *
 * Returns a pointer to the calculated spectrum (grid1d *). X-axis in cm$^{-1}$.
 *
 */

EXPORT cgrid1d *dft_driver_spectrum_evaluate(double tstep, double zero_offset, double tc) {

  long i;
  double omega, de, ct;
  static cgrid1d *spectrum = NULL;

  check_mode();

  if(cur_time > ntime) {
    printf("%ld %ld\n", cur_time, ntime);
    fprintf(stderr, "libdft: cur_time >= ntime. Increase ntime.\n");
    exit(1);
  }

  tstep /= GRID_AUTOFS;
  tc /= GRID_AUTOFS;
  if(!spectrum)
    spectrum = cgrid1d_alloc(cur_time + zerofill, GRID_HZTOCM1 / (tstep * GRID_AUTOFS * 1E-15 * (double) (cur_time + zerofill)), CGRID1D_PERIODIC_BOUNDARY, 0);
  tdpot->value[0] = 0.0;
  /* This is approximate form of the first order polarization (see the CPL paper) */
  /* g(t) */
  for (i = 1; i < cur_time; i++)
    tdpot->value[i] = tdpot->value[i-1] + tstep * tdpot->value[i];
  for (i = 0; i < cur_time; i++)
    tdpot->value[i] = cexp(I * tdpot->value[i]);
  /* exponential decay */
  if(tc > 0.0) {
    for (i = 0; i < cur_time; i++) {
      ct = i * (double) tstep;
      de = exp(-ct / tc);
      tdpot->value[i] *= de;
    }
  }
  /* zero fill */
  for (i = cur_time; i < cur_time + zerofill; i++)
    tdpot->value[i] = 0.0;
  for (i = 0; i < cur_time + zerofill; i++)
    printf("libdft: Correlation function at %le fs = (%le, %le)\n", i * tstep * GRID_AUTOFS, creal(tdpot->value[i]), cimag(tdpot->value[i]));

  /* flip zero frequency to the middle */
  for (i = 0; i < cur_time + zerofill; i++)
    tdpot->value[i] *= pow(-1.0, (double) i);

  cgrid1d_fft(tdpot);
  for (i = 0, omega = -0.5 * spectrum->step * (spectrum->nx - 1); i < spectrum->nx; i++, omega += spectrum->step) {
    spectrum->value[i] = pow(creal(tdpot->value[i]), 2.0) + pow(cimag(tdpot->value[i]), 2.0);
    if(zero_offset != 0.0) spectrum->value[i] *= zero_offset + omega;
  }

  return spectrum;
}

/*
 * Evaluate the liquid velocity field for a given order paremeter,
 * $v = m_{He} \vec{J}/\rho$.
 *
 * gwf  = Order parameter for which the velocity field is evaluated (input; wf3d *).
 * vx    = Velocity field x component (output; rgrid3d *).
 * vy    = Velocity field y component (output; rgrid3d *).
 * vz    = Velocity field z component (output; rgrid3d *).
 *
 */

EXPORT void dft_driver_veloc_field(wf3d *wf, rgrid3d *vx, rgrid3d *vy, rgrid3d *vz) {

  check_mode();

  grid3d_wf_probability_flux(wf, vx, vy, vz);
  grid3d_wf_density(wf, workspace1);
  rgrid3d_division(vx, vx, workspace1);
  rgrid3d_division(vy, vy, workspace1);
  rgrid3d_division(vz, vz, workspace1);
}

/*
 * Evaluate liquid momentum according to:
 * $\int\rho\v_idr$ where $i = x,y,z$.
 *
 * wf = Order parameter for evaluation (wf3d *; input).
 * px = Liquid momentum along x (double *; output).
 * py = Liquid momentum along y (double *; output).
 * pz = Liquid momentum along z (double *; output).
 *
 */

EXPORT void dft_driver_P(wf3d *wf, double *px, double *py, double *pz) {

  check_mode();

  grid3d_wf_probability_flux(wf, workspace1, workspace2, workspace3);
  rgrid3d_multiply(workspace1, wf->mass);
  rgrid3d_multiply(workspace2, wf->mass);
  rgrid3d_multiply(workspace3, wf->mass);

  *px = rgrid3d_integral(workspace1);
  *py = rgrid3d_integral(workspace2);
  *pz = rgrid3d_integral(workspace3);
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

EXPORT double dft_driver_KE(wf3d *wf) {

  check_mode();

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
 * lx = Anuglar momentum x component (double *; output).
 * ly = Anuglar momentum y component (double *; output).
 * lz = Anuglar momentum z component (double *; output).
 *
 */

static double origin_x = 0.0, origin_y = 0.0, origin_z = 0.0;

static double mult_mx(void *xx, double x, double y, double z) {

  rgrid3d *grid = (rgrid3d *) xx;

  return -rgrid3d_value(grid, x, y, z) * (x - origin_x);
}

static double mult_my(void *xx, double x, double y, double z) {

  rgrid3d *grid = (rgrid3d *) xx;

  return -rgrid3d_value(grid, x, y, z) * (y - origin_y);
}

static double mult_mz(void *xx, double x, double y, double z) {

  rgrid3d *grid = (rgrid3d *) xx;

  return -rgrid3d_value(grid, x, y, z) * (z - origin_z);
}

static double mult_x(void *xx, double x, double y, double z) {

  rgrid3d *grid = (rgrid3d *) xx;

  return rgrid3d_value(grid, x, y, z) * (x - origin_x);
}

static double mult_y(void *xx, double x, double y, double z) {

  rgrid3d *grid = (rgrid3d *) xx;

  return rgrid3d_value(grid, x, y, z) * (y - origin_y);
}

static double mult_z(void *xx, double x, double y, double z) {

  rgrid3d *grid = (rgrid3d *) xx;

  return rgrid3d_value(grid, x, y, z) * (z - origin_z);
}

EXPORT void dft_driver_L(wf3d *wf, double *lx, double *ly, double *lz) {

  rgrid3d *px = workspace4, *py = workspace5, *pz = workspace6;
  
  check_mode();

  if(!workspace7) workspace7 = dft_driver_alloc_rgrid();
  if(!workspace8) workspace8 = dft_driver_alloc_rgrid();

  origin_x = wf->grid->x0 ;
  origin_y = wf->grid->y0 ;
  origin_z = wf->grid->z0 ;

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
 
 * radial = Radial density (rgrid1d *; output).
 * grid   = Source grid (rgrid3d *; input).
 * dtheta = Integration step size along theta in radians (double; input).
 * dphi   = Integration step size along phi in radians (double, input).
 * xc     = x coordinate for the center (double; input).
 * yc     = y coordinate for the center (double; input).
 * zc     = z coordinate for the center (double; input).
 *
 */

EXPORT void dft_driver_radial(rgrid1d *radial, rgrid3d *grid, double dtheta, double dphi, double xc, double yc, double zc) {
  
  double r, theta, phi;
  double x, y, z, step = radial->step, tmp;
  long ri, thetai, phii, nx = radial->nx;

  check_mode();

  r = 0.0;
  for (ri = 0; ri < nx; ri++) {
    tmp = 0.0;
#pragma omp parallel for firstprivate(xc,yc,zc,grid,r,dtheta,dphi,step) private(theta,phi,x,y,z,thetai,phii) reduction(+:tmp) default(none) schedule(runtime)
    for (thetai = 0; thetai <= (long) (M_PI / dtheta); thetai++)
      for (phii = 0; phii <= (long) (2.0 * M_PI / dtheta); phii++) {
	theta = thetai * dtheta;
	phi = phii * dphi;
	x = r * cos(phi) * sin(theta) + xc;
	y = r * sin(phi) * sin(theta) + yc;
	z = r * cos(theta) + zc;
	tmp += rgrid3d_value(grid, x, y, z) * sin(theta);
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
 * dtheta = Integration step size along theta in radians (double; input).
 * dphi   = Integration step size along phi in radians (double, input).
 * xc     = x coordinate for the center (double; input).
 * yc     = y coordinate for the center (double; input).
 * zc     = z coordinate for the center (double; input).
 *
 */

EXPORT void dft_driver_radial_complex(cgrid1d *radial, cgrid3d *grid, double dtheta, double dphi, double xc, double yc, double zc) {
  
  double r, theta, phi;
  double x, y, z, step = radial->step;
  double complex tmp;
  long ri, thetai, phii, nx = radial->nx;

  check_mode();

  r = 0.0;
  for (ri = 0; ri < nx; ri++) {
    tmp = 0.0;
#pragma omp parallel for firstprivate(xc,yc,zc,grid,r,dtheta,dphi,step) private(theta,phi,x,y,z,thetai,phii) reduction(+:tmp) default(none) schedule(runtime)
    for (thetai = 0; thetai <= (long) (M_PI / dtheta); thetai++)
      for (phii = 0; phii <= (long) (2.0 * M_PI / dtheta); phii++) {
	theta = thetai * dtheta;
	phi = phii * dphi;
	x = r * cos(phi) * sin(theta) + xc;
	y = r * sin(phi) * sin(theta) + yc;
	z = r * cos(theta) + zc;
	tmp += cgrid3d_value(grid, x, y, z) * sin(theta);
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

EXPORT double dft_driver_spherical_rb(rgrid3d *density) {

  double disp;

  check_mode();

  rgrid3d_multiply(density, -1.0);
  rgrid3d_add(density, driver_rho0);
  disp = rgrid3d_integral(density);
  rgrid3d_add(density, -driver_rho0);
  rgrid3d_multiply(density, -1.0);

  return pow(disp * 3.0 / (4.0 * M_PI * driver_rho0), 1.0 / 3.0);
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
 */

static rgrid3d *prev_dens = NULL;

EXPORT double dft_driver_norm(rgrid3d *density) {

  long i;
  double mx = -1.0, tmp;

  check_mode();

  if(!prev_dens) {
    prev_dens = dft_driver_alloc_rgrid();
    rgrid3d_copy(prev_dens, density);
    return 1.0;
  }

  for (i = 0; i < density->nx * density->ny * density->nz; i++)
    if((tmp = fabs(density->value[i] - prev_dens->value[i])) > mx) mx = tmp;
  
  rgrid3d_copy(prev_dens, density);

  return mx;
}


/*
 * Force spherical symmetry by spherical averaging.
 *
 * wf = wavefunction to be averaged (wf3d *).
 * xc = x coordinate for the center (double).
 * yc = y coordinate for the center (double).
 * zc = z coordinate for the center (double).
 *
 */

EXPORT void dft_driver_force_spherical(wf3d *wf, double xc, double yc, double zc) {

  long i, j, l, k, len;
  long nx = wf->grid->nx, ny = wf->grid->ny, nz = wf->grid->nz;
  long nyz = ny * nz;
  double step = wf->grid->step;
  double x, y, z, x2, y2, z2, d;
  cgrid1d *average;
  double complex *avalue, *value = wf->grid->value;

  check_mode();

  if(nx > ny) len = nx; else len = ny;
  if(nz > len) len = nz;
  average = cgrid1d_alloc(len, step, CGRID1D_PERIODIC_BOUNDARY, 0);
  avalue = average->value;
  dft_driver_radial_complex(average, wf->grid, 0.01, 0.01, xc, yc, zc);

  /* Write spherical average back to wf */
  for(i = 0; i < nx; i++) {
    x = (i - nx/2) * step;
    x2 = (x - xc) * (x - xc);
    for (j = 0; j < ny; j++) {
      y = (j - ny/2) * step;
      y2 = (y - yc) * (y - yc);
      for (l = 0; l < nz; l++) {
	z = (l - nz/2) * step;
	z2 = (z - zc) * (z - zc);
	d = sqrt(x2 + y2 + z2);
	k = (long) (0.5 + d / step);
	if(k >= len) k = len - 1;
	value[i * nyz + j * nz + l] = avalue[k];
      }
    }
  }
  cgrid1d_free(average);
}

/*
 * Read in a 2-D cylindrical grid as a 3-D grid.
 * Useful for getting the initial guess from a 2-D calculation.
 *
 * in  = File descriptor for reading the 2-D grid (FILE *).
 * out = Grid for output (rgrid3d *).
 *
 */

EXPORT void dft_driver_read_2d_to_3d(rgrid3d *grid, char *filename) {

  long nx = grid->nx, ny = grid->ny, nz = grid->nz, i, j, k, nyz = ny * nz;
  long nx2, ny2;
  double step2, x, y, z, r, step = grid->step;
  rgrid2d *tmp;
  FILE *in;
  char fname[512];

  check_mode();

  sprintf(fname, "%s.grd", filename);
  if(!(in = fopen(fname, "r"))) {
    fprintf(stderr, "libdft: Can't open file %s.\n", filename);
    abort();
  }
  
  fread(&nx2, sizeof(long), 1, in);
  fread(&ny2, sizeof(long), 1, in);
  fread(&step2, sizeof(double), 1, in);
  
  if(!(tmp = rgrid2d_alloc(nx2, ny2, step2, RGRID2D_NEUMANN_BOUNDARY, NULL))) {
    fprintf(stderr, "libdft: Error allocating grid in dft_driver_read_2d_to_3d().\n");
    abort();
  }
  fread(tmp->value, sizeof(double), nx2 * ny2, in);
  for(i = 0; i < grid->nx; i++) {
    x = (i - nx/2.0) * step;
    for (j = 0; j < grid->ny; j++) {
      y = (j - ny/2.0) * step;
      r = sqrt(x * x + y * y);
      for (k = 0; k < grid->nz; k++) {
	z = (k - nz/2.0) * step;
	grid->value[i * nyz + j * nz + k] = rgrid2d_value_cyl(tmp, z, r);
      }
    }
  }
  rgrid2d_free(tmp);
  fclose(in);
}

#define R_M 0.05

static double complex vortex_x_n1(void *na, double x, double y, double z) {

  double d = sqrt(y * y + z * z);

  if(d < R_M) return 0.0;
  return (y + I * z) / d;
}

static double complex vortex_y_n1(void *na, double x, double y, double z) {

  double d = sqrt(x * x + z * z);

  if(d < R_M) return 0.0;
  return (x + I * z) / d;
}

static double complex vortex_z_n1(void *na, double x, double y, double z) {

  double d = sqrt(x * x + y * y);

  if(d < R_M) return 0.0;
  return (x + I * y) / d;
}

static double complex vortex_x_n2(void *na, double x, double y, double z) {

  double y2 = y * y, z2 = z * z;
  double d = sqrt(x * x + y * y);
  
  if(d < R_M) return 0.0;
  return ((y2 - z2) + I * 2 * y * z) / (y2 + z2);
}

static double complex vortex_y_n2(void *na, double x, double y, double z) {

  double x2 = x * x, z2 = z * z;
  double d = sqrt(x * x + y * y);
  
  if(d < R_M) return 0.0;  
  return ((x2 - z2) + I * 2 * x * z) / (x2 + z2);
}

static double complex vortex_z_n2(void *na, double x, double y, double z) {

  double x2 = x * x, y2 = y * y;
  double d = sqrt(x * x + y * y);
  
  if(d < R_M) return 0.0;  
  return ((x2 - y2) + I * 2 * x * y) / (x2 + y2);
}

/*
 * Modify a given wavefunction to have vorticity around a specified axis.
 *
 * gwf    = Wavefunction for the operation (gwf3d *).
 * n      = Quantum number (1 or 2) (int).
 * 
 */

EXPORT void dft_driver_vortex_initial(wf3d *gwf, int n, int axis) {

  check_mode();

  if(!cworkspace)
    cworkspace = dft_driver_alloc_cgrid();
  
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
 * potential = Potential grid where the vortex potential is added (rgrid3d *).
 * direction = Along which axis the vortex potential is added:
 *             DFT_DRIVER_VORTEX_{X,Y,Z}.
 */

static double vortex_x(void *na, double x, double y, double z) {

  double rp2 = y * y + z * z;

  if(rp2 < R_M * R_M) rp2 = R_M * R_M;
  return 1.0 / (2.0 * dft_driver_otf->mass * rp2);
}

static double vortex_y(void *na, double x, double y, double z) {

  double rp2 = x * x + z * z;

  if(rp2 < R_M * R_M) rp2 = R_M * R_M;
  return 1.0 / (2.0 * dft_driver_otf->mass * rp2);
}

static double vortex_z(void *na, double x, double y, double z) {

  double rp2 = x * x + y * y;

  if(rp2 < R_M * R_M) rp2 = R_M * R_M;
  return 1.0 / (2.0 * dft_driver_otf->mass * rp2);
}

EXPORT void dft_driver_vortex(rgrid3d *potential, int direction) {

  check_mode();

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
