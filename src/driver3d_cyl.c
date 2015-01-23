/*
 * Simple driver routines to propagate the liquid (3D cylindrical coordinates).
 *
 * TOOD: Add comments to show which internal workspaces are used by 
 * each function.
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

#define MIN_SUBSTEPS 2
#define MAX_SUBSTEPS 16

/* End of tunable parameters */

/* Global user accessible variables */
dft_ot_functional *dft_driver_otf_cyl = 0;
int dft_driver_init_wavefunction_cyl = 1;

static long driver_nz = 0, driver_nr = 0, driver_nphi = 0, driver_threads = 0, driver_dft_model = 0, driver_iter_mode = 0, driver_boundary_type = 0;
static long driver_norm_type = 0, driver_nhe = 0, center_release = 0;
static long driver_rels = 0;
static double driver_frad = 0.0;
static double driver_step = 0.0, driver_abs = 0.0, driver_rho0 = 0.0;
static rgrid3d *cart_density = 0, *cyl_density = 0;
static rgrid3d *workspace1 = 0;
static rgrid3d *workspace2 = 0;
static rgrid3d *workspace3 = 0;
static rgrid3d *workspace4 = 0;
static rgrid3d *workspace5 = 0;
static rgrid3d *workspace6 = 0;
static rgrid3d *workspace7 = 0;
static rgrid3d *workspace8 = 0;
static rgrid3d *workspace9 = 0;
static cgrid3d *cworkspace = 0, *cart_potential = 0;
static grid_timer timer;
static double damp = 0.2;

int dft_internal_using_cyl = 0;   // TODO add sanity checks to 2d and 3d 
extern int dft_internal_using_3d;
extern int dft_internal_using_2d;
extern int dft_driver_kinetic; /* default FFT propagation for kinetic */

static inline void check_mode() {

  if(dft_internal_using_2d || dft_internal_using_3d) {
    fprintf(stderr, "libdft: Cylindrical 3D routine called in 2D or Cartesian 3D code.\n");
    exit(1);
  } else dft_internal_using_cyl = 1;
}

static double region_func(void *gr, double z, double r, double phi) {

  double ulz = (driver_nz/2.0) * driver_step - driver_abs, ulr =  driver_nr * driver_step - driver_abs;
  double d = 0.0;
  
  z = fabs(z);
  r = fabs(r);

  if(z >= ulz) d += damp * (z - ulz) / driver_abs;
  if(r >= ulr) d += damp * (r - ulr) / driver_abs;
  return d / 2.0;
}

static inline void scale_wf_cyl(long what, dft_ot_functional *local_otf, wf3d *gwf) {

  long i, j, k, driver_nrphi = driver_nr * driver_nphi;
  double z, r;
  double complex norm;

  if(what) { /* impurity */
    grid3d_wf_normalize_cyl(gwf);
    return;
  }
  
  /* liquid helium */
  switch(driver_norm_type) {
  case DFT_DRIVER_NORMALIZE_BULK: /* bulk normalization */
    norm = sqrt(local_otf->rho0) / cabs(cgrid3d_value_at_index_cyl(gwf->grid, driver_nr-1, 0, 0));
    cgrid3d_multiply(gwf->grid, norm);
    break;
  case DFT_DRIVER_NORMALIZE_DROPLET: /* helium droplet */
    if(!center_release) {
      double sq = sqrt(3.0*driver_rho0/4.0);
      for (i = 0; i < driver_nz; i++) {
	z = (i - driver_nz/2.0) * driver_step;
	for (j = 0; j < driver_nr; j++) {
	  r = j * driver_step;
	  for (k = 0; k < driver_nphi; k++) {
	    if(sqrt(z*z + r*r) < driver_frad && cabs(gwf->grid->value[i * driver_nrphi + j * driver_nphi + k]) < sq)
	      gwf->grid->value[i * driver_nrphi + j * driver_nphi + k] = sq;
	  }
	}
      }
    }
    grid3d_wf_normalize_cyl(gwf);
    cgrid3d_multiply(gwf->grid, sqrt((double) driver_nhe));
    break;
  case DFT_DRIVER_NORMALIZE_COLUMN: /* column along z */
    if(!center_release) {
      double sq = sqrt(3.0*driver_rho0/4.0);
      for (i = 0; i < driver_nz; i++) {
	z = (i - driver_nz/2.0) * driver_step;
	for (j = 0; j < driver_nr; j++) {
	  r = j * driver_step;
	  for (k = 0; k < driver_nphi; k++) {
	    if(fabs(r) < driver_frad && cabs(gwf->grid->value[i * driver_nrphi + j * driver_nphi + k]) < sq)
	      gwf->grid->value[i * driver_nrphi + j * driver_nphi + k] = sq;
	  }
	}
      }
    }
    grid3d_wf_normalize_cyl(gwf);
    cgrid3d_multiply(gwf->grid, sqrt((double) driver_nhe));
    break;
  case DFT_DRIVER_NORMALIZE_SURFACE:   /* in (r,phi) plane starting at z = 0 */
    if(!center_release) {
      for (i = 0; i < driver_nz; i++)
	for (j = 0; j < driver_nr; j++)
	  for (k = 0; k < driver_nphi; k++) {
	    z = (k - driver_nz/2.0) * driver_step;
	    if(fabs(z) < driver_frad)
	      gwf->grid->value[i * driver_nrphi * driver_nphi + j * driver_nz + k] = 0.0;
	  }
    }
    grid3d_wf_normalize_cyl(gwf);
    cgrid3d_multiply(gwf->grid, sqrt((double) driver_nhe));
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

EXPORT void dft_driver_initialize_cyl() {

  static int been_here = 0;

  check_mode();

  if(!been_here) {
    if(driver_nz == 0) {
      fprintf(stderr, "libdft: dft_driver not properly initialized.\n");
      exit(1);
    }

    grid_timer_start(&timer);
    grid_threads_init(driver_threads);
    // Cartesian workspaces
    workspace1 = rgrid3d_alloc(2*driver_nr, 2*driver_nr, driver_nz, driver_step, RGRID3D_PERIODIC_BOUNDARY, 0);
    workspace2 = rgrid3d_alloc(2*driver_nr, 2*driver_nr, driver_nz, driver_step, RGRID3D_PERIODIC_BOUNDARY, 0);
    workspace3 = rgrid3d_alloc(2*driver_nr, 2*driver_nr, driver_nz, driver_step, RGRID3D_PERIODIC_BOUNDARY, 0);
    workspace4 = rgrid3d_alloc(2*driver_nr, 2*driver_nr, driver_nz, driver_step, RGRID3D_PERIODIC_BOUNDARY, 0);
    workspace5 = rgrid3d_alloc(2*driver_nr, 2*driver_nr, driver_nz, driver_step, RGRID3D_PERIODIC_BOUNDARY, 0);
    workspace6 = rgrid3d_alloc(2*driver_nr, 2*driver_nr, driver_nz, driver_step, RGRID3D_PERIODIC_BOUNDARY, 0);
    workspace7 = rgrid3d_alloc(2*driver_nr, 2*driver_nr, driver_nz, driver_step, RGRID3D_PERIODIC_BOUNDARY, 0);
    if(driver_dft_model & DFT_OT_BACKFLOW) {
      workspace8 = rgrid3d_alloc(2*driver_nr, 2*driver_nr, driver_nz, driver_step, RGRID3D_PERIODIC_BOUNDARY, 0);
      workspace9 = rgrid3d_alloc(2*driver_nr, 2*driver_nr, driver_nz, driver_step, RGRID3D_PERIODIC_BOUNDARY, 0);
    }
    // Cartesian density
    cart_density = rgrid3d_alloc(2*driver_nr, 2*driver_nr, driver_nz, driver_step, RGRID3D_PERIODIC_BOUNDARY, 0);  
    // Cylindrical density
    cyl_density = rgrid3d_alloc(driver_nr, driver_nphi, driver_nz, driver_step, RGRID3D_PERIODIC_BOUNDARY, 0);  
    // Cartesian potential
    cart_potential = cgrid3d_alloc(2*driver_nr, 2*driver_nr, driver_nz, driver_step, CGRID3D_PERIODIC_BOUNDARY, 0);  
    // all OT 3D stuff is in Cartesian
    dft_driver_otf_cyl = dft_ot3d_alloc(driver_dft_model, 2*driver_nr, 2*driver_nr, driver_nz, driver_step, DFT_DRIVER_BC_NORMAL, MIN_SUBSTEPS, MAX_SUBSTEPS);
    if(driver_rho0 == 0.0) driver_rho0 = dft_driver_otf_cyl->rho0;
    else dft_driver_otf_cyl->rho0 = driver_rho0;
    fprintf(stderr, "libdft: rho0 = %le Angs^-3.\n", driver_rho0 / (GRID_AUTOANG * GRID_AUTOANG * GRID_AUTOANG));
    been_here = 1;
    fprintf(stderr, "libdft: %lf wall clock seconds for initialization.\n", grid_timer_wall_clock_time(&timer));
  }
}

/*
 * Set up the DFT calculation grid.
 *
 * nr      = number of grid points along r (long).
 * nphi    = number of grid points along phi (long).
 * nz      = number of grid points along z (long).
 * threads = number of parallel execution threads (long).
 *
 * No return value.
 *
 */

EXPORT void dft_driver_setup_grid_cyl(long nr, long nphi, long nz, double step, long threads) {
  
  check_mode();

  // TODO: fixme
  if((nr % 2) || (nz % 2)) {
    fprintf(stderr, "libdft: Currently works only with array sizes of multiples of two.\n");
    exit(1);
  }

  driver_nz = nz;
  driver_nr = nr;
  driver_nphi = nphi;
  driver_step = step;
  fprintf(stderr, "libdft: Cartesian cylindrical grid size = (%ld,%ld,%ld) with steps = (%le,%le,%le).\n", nr, nphi, nz, step, (2.0 * M_PI) / (double) driver_nphi, step);
  driver_threads = threads;
}

/*
 * Set up the DFT calculation model.
 *
 * dft_model = specify the DFT Hamiltonian to use (see ot.h).
 * iter_mode = iteration mode: 1 = imaginary time, 0 = real time.
 * rho0      = equilibrium density for the liquid (in a.u.; double).
 *             if 0.0, the equilibrium density will be used.
 *
 * No return value.
 *
 */

EXPORT void dft_driver_setup_model_cyl(long dft_model, long iter_mode, double rho0) {

  check_mode();

  driver_dft_model = dft_model;
  driver_iter_mode = iter_mode;
  driver_rho0 = rho0;
}

/*
 * Set up boundaries.
 *
 * type    = boundary type: 0 = regular, 1 = absorbing (long).
 * absb    = width of absorbing boundary (double; bohr).
 *           In this region the density tends towards driver_rho0.
 * 
 * No return value.
 *
 */

EXPORT void dft_driver_setup_boundaries_cyl(long boundary_type, double absb) {

  check_mode();

  driver_boundary_type = boundary_type;
  driver_abs = absb;
}

/*
 * Modify the value of the damping constant for absorbing boundary.
 *
 * dmp = damping constant (default 0.03).
 *
 */

EXPORT void dft_driver_setup_boundaries_damp_cyl(double dmp) {

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
 * rels = iteration after which the fixing condition will be release.
 *        This should be done for the last few iterations to avoid
 *        artifacts arising from the fixing constraint. Set to zero to disable.
 * 
 */

EXPORT void dft_driver_setup_normalization_cyl(long norm_type, long nhe, double frad, long rels) {

  check_mode();

  driver_norm_type = norm_type;
  driver_nhe = nhe;
  driver_rels = rels;
  driver_frad = frad;
}

/*
 * Predict: propagate the given wf in time. All input/output grids are cylindrical 3D grids.
 *
 * what          = what is propagated: 0 = L-He, 1 = other.
 * ext_pot       = present external potential grid (rgrid3d *; input) (NULL = no ext. pot).
 * gwf           = liquid wavefunction to propagate (wf3d *; input).
 *                 Note that gwf is NOT changed by this routine.
 * gwfp          = predicted wavefunction (wf3d *; output).
 * cyl_potential = storage space for the potential (cgrid3d *; output).
 *                 Do not overwrite this before calling the correct routine.
 * tstep         =  time step in FS (double; input).
 * iter          = current iteration (long; input).
 *
 * If what == 0, the liquid potential is added automatically.
 *               Also the absorbing boundaries are only active for this.
 * If what == 1, the propagation is carried out only with ext_pot.
 *
 * No return value.
 *
 */

EXPORT inline void dft_driver_propagate_predict_cyl(long what, rgrid3d *ext_pot, wf3d *gwf, wf3d *gwfp, cgrid3d *cyl_potential, double tstep, long iter) {

  double complex time, htime;
  static double last_tstep = -1.0;

  check_mode();

  if(driver_nz == 0) {
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
    grid3d_wf_constant(gwf, sqrt(dft_driver_otf_cyl->rho0));
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
  if(!cworkspace)
    cworkspace = cgrid3d_alloc(driver_nr, driver_nphi, driver_nz, driver_step, CGRID3D_PERIODIC_BOUNDARY, 0);  // CYL
  grid3d_wf_propagate_kinetic_cn_cyl(gwf, htime, cworkspace);
  if(driver_iter_mode) scale_wf_cyl(what, dft_driver_otf_cyl, gwfp);
  cgrid3d_copy(gwfp->grid, gwf->grid);

  /* predict */
  if(!what) {
    grid3d_wf_density(gwfp, cyl_density);
    rgrid3d_map_cyl_on_cart(cart_density, cyl_density);
    cgrid3d_zero(cart_potential);
    // TODO: backflow will dig stuff from gwfp - WILL NOT WORK; would be better to pass gwfp but velocity grids?
    dft_ot3d_potential(dft_driver_otf_cyl, cart_potential, gwfp, cart_density, workspace1, workspace2, workspace3, workspace4, workspace5, workspace6, workspace7, workspace8, workspace9);
    cgrid3d_map_cart_on_cyl(cyl_potential, cart_potential);
  } else cgrid3d_zero(cyl_potential);
  /* absorbing boundary */
  if(driver_boundary_type == 1 && !what) {
    fprintf(stderr, "libdft: Predict - absorbing boundary for helium.\n");
    grid3d_wf_absorb_cyl(cyl_potential, cyl_density, driver_rho0, region_func, (rgrid3d *) cworkspace, (driver_iter_mode==1) ? I:1.0);
  }
  /* External potential for Helium */
  /* Im - He contribution */
  if(ext_pot) grid3d_add_real_to_complex_re(cyl_potential, ext_pot);

  /* potential */
  grid3d_wf_propagate_potential(gwfp, cyl_potential, time);
  if(driver_iter_mode) scale_wf_cyl(what, dft_driver_otf_cyl, gwfp);
}

/*
 * Correct: propagate the given wf in time. 3D cyl. All input grids are in 3D cyl.
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

EXPORT inline void dft_driver_propagate_correct_cyl(long what, rgrid3d *ext_pot, wf3d *gwf, wf3d *gwfp, cgrid3d *cyl_potential, double tstep, long iter) {

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
    grid3d_wf_density(gwfp, cyl_density);
    rgrid3d_map_cyl_on_cart(cart_density, cyl_density);
    cgrid3d_zero(cart_potential);
    dft_ot3d_potential(dft_driver_otf_cyl, cart_potential, gwfp, cart_density, workspace1, workspace2, workspace3, workspace4, workspace5, workspace6, workspace7, workspace8, workspace9);
    cgrid3d_add_cart_on_cyl(cyl_potential, cart_potential);  // add and multiply by 1/2 to get average later
  }

  /* absorbing boundary */
  if(driver_boundary_type == 1 && !what) {
    fprintf(stderr, "libdft: Correct - absorbing boundary for helium.\n");
    grid3d_wf_absorb_cyl(cyl_potential, cyl_density, driver_rho0, region_func, (rgrid3d *) cworkspace, (driver_iter_mode==1) ? I:1.0);
  }  
  /* External potential for Helium */
  /* Im - He contribution (new) */
  if(ext_pot) grid3d_add_real_to_complex_re(cyl_potential, ext_pot);
  /* average of future and current (new) */
  cgrid3d_multiply(cyl_potential, 0.5);
  
  /* potential */
  grid3d_wf_propagate_potential(gwf, cyl_potential, time);
  if(driver_iter_mode) scale_wf_cyl(what, dft_driver_otf_cyl, gwf);

  /* 1/2 x kinetic */
  grid3d_wf_propagate_kinetic_cn_cyl(gwf, htime, cworkspace);
  if(driver_iter_mode) scale_wf_cyl(what, dft_driver_otf_cyl, gwf);
  
  fprintf(stderr, "libdft: Iteration %ld took %lf wall clock seconds.\n", iter, grid_timer_wall_clock_time(&timer));
  fflush(stdout);
}

/*
 * Allocate a complex grid.
 *
 * Returns a pointer to the allocated grid.
 *
 */

EXPORT cgrid3d *dft_driver_alloc_cgrid_cyl() {

  check_mode();

  if(driver_nz == 0) {
    fprintf(stderr, "libdft: dft_driver routines must be initialized first.\n");
    exit(1);
  }
  return cgrid3d_alloc(driver_nr, driver_nphi, driver_nz, driver_step, CGRID3D_PERIODIC_BOUNDARY, 0);
}

/*
 * Allocate a real grid.
 *
 * Returns a pointer to the allocated grid.
 *
 */

EXPORT rgrid3d *dft_driver_alloc_rgrid_cyl() {

  check_mode();

  if(driver_nz == 0) {
    fprintf(stderr, "libdft: dft_driver routines must be initialized first.\n");
    exit(1);
  }
  return rgrid3d_alloc(driver_nr, driver_nphi, driver_nz, driver_step, RGRID3D_PERIODIC_BOUNDARY, 0);
}

/*
 * Allocate a complex grid. Cartesian that can contain cylindrical.
 *
 * Returns a pointer to the allocated grid.
 *
 */

EXPORT cgrid3d *dft_driver_alloc_cgrid_cyl_cart() {

  check_mode();

  if(driver_nz == 0) {
    fprintf(stderr, "libdft: dft_driver routines must be initialized first.\n");
    exit(1);
  }
  return cgrid3d_alloc(2*driver_nr, 2*driver_nr, driver_nz, driver_step, CGRID3D_PERIODIC_BOUNDARY, 0);
}

/*
 * Allocate a real grid. Cartesian that can contain cylindrical.
 *
 * Returns a pointer to the allocated grid.
 *
 */

EXPORT rgrid3d *dft_driver_alloc_rgrid_cyl_cart() {

  check_mode();

  if(driver_nz == 0) {
    fprintf(stderr, "libdft: dft_driver routines must be initialized first.\n");
    exit(1);
  }
  return rgrid3d_alloc(2*driver_nr, 2*driver_nr, driver_nz, driver_step, RGRID3D_PERIODIC_BOUNDARY, 0);
}

/*
 * Allocate a wavefunction (initialized to sqrt(rho0)).
 *
 * mass = particle mass in a.u. (double).
 *
 * Returns pointer to the wavefunction.
 *
 */

EXPORT wf3d *dft_driver_alloc_wavefunction_cyl(double mass) {

  wf3d *tmp;

  check_mode();

  if(driver_nz == 0) {
    fprintf(stderr, "libdft: dft_driver routines must be initialized first.\n");
    exit(1);
  }
  tmp = grid3d_wf_alloc(driver_nr, driver_nphi, driver_nz, driver_step, mass, WF3D_PERIODIC_BOUNDARY, WF3D_2ND_ORDER_PROPAGATOR);
  cgrid3d_constant(tmp->grid, sqrt(driver_rho0));
  return tmp;
}

/*
 * Read in density from a binary file (.grd).
 *
 * grid = place to store the read density (rgrid3d *).
 * file = filename for the file (char *). Note: the .grd extension must be given.
 *
 * No return value.
 *
 */

EXPORT void dft_driver_read_density_cyl(rgrid3d *grid, char *file) {

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
 * .z   ASCII file cut along (z, 0.0, 0.0)
 * .r   ASCII file cut along (0.0, r, 0.0)
 * .phi ASCII file cut along (0.0, r_max/2, phi)
 *
 * grid = density grid (rgrid3d *).
 * base = Basename for the output file (char *).
 *
 * No return value.
 *
 */

EXPORT void dft_driver_write_density_cyl(rgrid3d *grid, char *base) {

  FILE *fp;
  char file[2048];
  long i, j, k;
  double z, r, phi;

  check_mode();

  sprintf(file, "%s.grd", base);
  if(!(fp = fopen(file, "w"))) {
    fprintf(stderr, "libdft: Can't open %s for writing.\n", file);
    exit(1);
  }
  rgrid3d_write(grid, fp);
  fclose(fp);

  sprintf(file, "%s.r", base);
  if(!(fp = fopen(file, "w"))) {
    fprintf(stderr, "libdft: Can't open %s for writing.\n", file);
    exit(1);
  }
  j = 0; /* phi = 0, z = 0 */
  k = grid->nz/2;
  for(i = 0; i < grid->nx; i++) { 
    r = i * grid->step;
    fprintf(fp, "%le %le\n", r, rgrid3d_value_at_index_cyl(grid, i, j, k));
  }
  fclose(fp);

  sprintf(file, "%s.phi", base);
  if(!(fp = fopen(file, "w"))) {
    fprintf(stderr, "libdft: Can't open %s for writing.\n", file);
    exit(1);
  }
  i = grid->nx / 2;     /* r = rmax/2, z = 0 */
  k = grid->nz / 2;
  for(j = 0; j < grid->ny; j++) {
    phi = j * (2.0 * M_PI) / (double) grid->ny;
    fprintf(fp, "%le %le\n", phi, rgrid3d_value_at_index_cyl(grid, i, j, k));
  }
  fclose(fp);

  sprintf(file, "%s.z", base);
  if(!(fp = fopen(file, "w"))) {
    fprintf(stderr, "libdft: Can't open %s for writing.\n", file);
    exit(1);
  }
  i = 0; /* r = 0, phi = 0 */
  j = 0;
  for(k = 0; k < grid->nz; k++) {
    z = (k - grid->nz/2.0) * grid->step;
    fprintf(fp, "%le %le\n", z, rgrid3d_value_at_index_cyl(grid, i, j, k));
  }
  fclose(fp);
  fprintf(stderr, "libdft: Density written to %s.\n", file);
}

/*
 * Write output to ascii & binary files.
 * .grd file is the full grid in binary format.
 * .z   ASCII file cut along (z, 0.0, 0.0)
 * .r   ASCII file cut along (0.0, r, 0.0)
 * .phi ASCII file cut along (0.0, r_max/2, phi)
 *
 * wf = wf with the pase (rgrid3d *).
 * base = Basename for the output file (char *).
 *
 * No return value.
 *
 */

EXPORT void dft_driver_write_phase_cyl(wf3d *wf, char *base) {

  FILE *fp;
  char file[2048];
  long i, j, k;
  cgrid3d *grid = wf->grid;
  double complex tmp;
  rgrid3d *phase;
  double z, r, phi;
  long nz = grid->nx, nr = grid->ny, nphi = grid->nz;

  check_mode();

  phase = rgrid3d_alloc(nz, nr, nphi, grid->step, RGRID3D_PERIODIC_BOUNDARY, 0);
  for(i = 0; i < nz; i++)
    for(j = 0; j < nr; j++)
      for(k = 0; k < nphi; k++) {
	tmp = cgrid3d_value_at_index_cyl(grid, i, j, k);
	if(cabs(tmp) < 1E-6)
	  phase->value[i * nr * nphi + j * nphi + k] = 0.0;
	else
	  phase->value[i * nr * nphi + j * nphi + k] =  cimag(clog(tmp / cabs(tmp)));
      }

  sprintf(file, "%s.grd", base);
  if(!(fp = fopen(file, "w"))) {
    fprintf(stderr, "libdft: Can't open %s for writing.\n", file);
    exit(1);
  }
  rgrid3d_write(phase, fp);
  fclose(fp);

  sprintf(file, "%s.r", base);
  if(!(fp = fopen(file, "w"))) {
    fprintf(stderr, "libdft: Can't open %s for writing.\n", file);
    exit(1);
  }
  j = 0; /* phi = 0, z = 0 */
  k = nz / 2;
  for(i = 0; i < nr; i++) { 
    r = i * grid->step;
    fprintf(fp, "%le %le\n", r, rgrid3d_value_at_index_cyl(phase, i, j, k));
  }
  fclose(fp);

  sprintf(file, "%s.phi", base);
  if(!(fp = fopen(file, "w"))) {
    fprintf(stderr, "libdft: Can't open %s for writing.\n", file);
    exit(1);
  }
  i = nr / 2; /* r = rmax/2, z = 0 */
  k = nz / 2;
  for(j = 0; j < nphi; j++) {
    phi = j * (2.0 * M_PI) / (double) driver_nphi;
    fprintf(fp, "%le %le\n", phi, rgrid3d_value_at_index_cyl(phase, i, j, k));
  }
  fclose(fp);

  sprintf(file, "%s.z", base);
  if(!(fp = fopen(file, "w"))) {
    fprintf(stderr, "libdft: Can't open %s for writing.\n", file);
    exit(1);
  }
  i = 0; /* r = 0, phi = 0 */
  j = 0;
  for(k = 0; k < nz; k++) {
    z = (k - nz/2.0) * grid->step;
    fprintf(fp, "%le %le\n", z, rgrid3d_value_at_index_cyl(phase, i, j, k));
  }
  fclose(fp);

  fprintf(stderr, "libdft: Phase written to %s.\n", file);
  rgrid3d_free(phase);
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

EXPORT void dft_driver_read_grid_cyl(cgrid3d *grid, char *file) {

  FILE *fp;

  check_mode();

  if(!(fp = fopen(file, "r"))) {
    fprintf(stderr, "libdft: Can't open density grid file %s.\n", file);
    exit(1);
  }
  cgrid3d_read(grid, fp);
  fclose(fp);
}

/*
 * Write a complex grid to ascii & binary files.
 * .grd file is the full grid in binary format.
 * .z   ASCII file cut along (z, 0.0, 0.0)
 * .r   ASCII file cut along (0.0, r, 0.0)
 * .phi ASCII file cut along (0.0, r_max/2, phi)
 *
 * grid = grid to be written (cgrid3d *).
 * base = Basename for the output file (char *).
 *
 * No return value.
 *
 */

EXPORT void dft_driver_write_grid_cyl(cgrid3d *grid, char *base) {

  FILE *fp;
  char file[2048];
  long i, j, k;
  double z, r, phi;

  check_mode();

  sprintf(file, "%s.grd", base);
  if(!(fp = fopen(file, "w"))) {
    fprintf(stderr, "libdft: Can't open %s for writing.\n", file);
    exit(1);
  }
  cgrid3d_write(grid, fp);
  fclose(fp);

  sprintf(file, "%s.r", base);
  if(!(fp = fopen(file, "w"))) {
    fprintf(stderr, "libdft: Can't open %s for writing.\n", file);
    exit(1);
  }
  j = 0;   /* phi = 0, z = 0 */
  k = grid->nz/2;
  for(i = 0; i < grid->nx; i++) { 
    r = i * grid->step;
    fprintf(fp, "%le %le %le\n", r, creal(cgrid3d_value_at_index_cyl(grid, i, j, k)), cimag(cgrid3d_value_at_index_cyl(grid, i, j, k)));
  }
  fclose(fp);

  sprintf(file, "%s.phi", base);
  if(!(fp = fopen(file, "w"))) {
    fprintf(stderr, "libdft: Can't open %s for writing.\n", file);
    exit(1);
  }
  i = grid->nx/2; /* r = rmax/2, z = 0 */
  k = grid->nz/2;
  for(j = 0; j < grid->ny; j++) {
    phi = j * (2.0 * M_PI) / (double) grid->ny;
    fprintf(fp, "%le %le %le\n", phi, creal(cgrid3d_value_at_index_cyl(grid, i, j, k)), cimag(cgrid3d_value_at_index_cyl(grid, i, j, k)));
  }
  fclose(fp);

  sprintf(file, "%s.z", base);
  if(!(fp = fopen(file, "w"))) {
    fprintf(stderr, "libdft: Can't open %s for writing.\n", file);
    exit(1);
  }
  i = 0; /* r = 0, phi = 0 */
  j = 0;
  for(k = 0; k < grid->nz; k++) {
    z = (k - grid->nz/2.0) * grid->step;
    fprintf(fp, "%le %le %le\n", z, creal(cgrid3d_value_at_index_cyl(grid, i, j, k)), cimag(cgrid3d_value_at_index_cyl(grid, i, j, k)));
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

EXPORT double dft_driver_energy_cyl(wf3d *gwf, rgrid3d *ext_pot) {

  double energy;

  check_mode();

  if(!cworkspace)
    cworkspace = cgrid3d_alloc(driver_nr, driver_nphi, driver_nz, driver_step, CGRID3D_PERIODIC_BOUNDARY, 0); // CYL

  /* we may need more memory for this... */
  if(!workspace7) workspace7 = rgrid3d_alloc(2*driver_nr, 2*driver_nr, driver_nz, driver_step, RGRID3D_PERIODIC_BOUNDARY, 0); // CART
  if(!workspace8) workspace8 = rgrid3d_alloc(2*driver_nr, 2*driver_nr, driver_nz, driver_step, RGRID3D_PERIODIC_BOUNDARY, 0);
  if(!workspace9) workspace9 = rgrid3d_alloc(2*driver_nr, 2*driver_nr, driver_nz, driver_step, RGRID3D_PERIODIC_BOUNDARY, 0);
  grid3d_wf_density(gwf, cyl_density);
  rgrid3d_map_cyl_on_cart(cart_density, cyl_density);
  /* WARNING: GWF not in cylindrical coords!!!!! (TODO) */
  fprintf(stderr, "libdft: CART does not work with backflow.\n");
  dft_ot3d_energy_density(dft_driver_otf_cyl, workspace9, gwf, cart_density, workspace1, workspace2, workspace3, workspace4, workspace5, workspace6, workspace7, workspace8);
  if(ext_pot) {
    rgrid3d_map_cyl_on_cart((rgrid3d *) cworkspace, ext_pot);    
    rgrid3d_add_scaled_product(workspace9, 1.0, cart_density, (rgrid3d *) cworkspace);
  }
  energy = rgrid3d_integral(workspace9); // eval in CART
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

EXPORT double dft_driver_natoms_cyl(wf3d *gwf) {

  check_mode();

  return creal(cgrid3d_integral_of_square_cyl(gwf->grid));
}
