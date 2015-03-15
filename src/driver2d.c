/*
 * Simple driver routines to propagate the liquid (2D).
 *
 * TOOD: Add comments to show which internal workspaces are used by 
 * each function.
 *
 * 
 * TODO: Latest change: NEUMANN -> PERIODIC. Seems to fix unsymmetric
 * solution behavior along z. Does it affect anything else?
 * Ideally one should have Neumann along r and periodic along z.
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
dft_ot_functional_2d *dft_driver_otf_2d = 0;
int dft_driver_init_wavefunction_2d = 1;

static long driver_nz = 0, driver_nr = 0, driver_threads = 0, driver_dft_model = 0, driver_iter_mode = 0, driver_boundary_type = 0;
static long driver_norm_type = 0, driver_nhe = 0, center_release = 0;
static long driver_rels = 0;
static double driver_frad = 0.0;
static double driver_step = 0.0, driver_abs = 0.0, driver_rho0 = 0.0;
static rgrid2d *density = 0;
static rgrid2d *workspace1 = 0;
static rgrid2d *workspace2 = 0;
static rgrid2d *workspace3 = 0;
static rgrid2d *workspace4 = 0;
static rgrid2d *workspace5 = 0;
static rgrid2d *workspace6 = 0;
static rgrid2d *workspace7 = 0;
static rgrid2d *workspace8 = 0;
static rgrid2d *workspace9 = 0;
static cgrid2d *cworkspace = 0;
static grid_timer timer;
static double damp = 0.2;

int dft_internal_using_2d = 0;
extern int dft_internal_using_3d, dft_internal_using_cyl;

static inline void check_mode() {

  if(dft_internal_using_3d || dft_internal_using_cyl) {
    fprintf(stderr, "libdft: Cartesian 3D or Cylindrical 3D routine called in 2D Cylindrical code.\n");
    exit(1);
  } else dft_internal_using_2d = 1;
}

static double region_func(void *gr, double z, double r) {

  double ulz = (driver_nz/2.0) * driver_step - driver_abs, ulr = driver_nr * driver_step - driver_abs;
  double d = 0.0;

  z = fabs(z);
  
  if(z >= ulz) d += damp * (z - ulz) / driver_abs;
  if(r >= ulr) d += damp * (r - ulr) / driver_abs;
  return d / 2.0;
}

// TODO: Out of date: merge from 3D.

static inline void scale_wf(long what, wf2d *gwf) {

  long i, j;
  double z, r;
  double complex norm;

  if(what) { /* impurity */
    grid2d_wf_normalize_cyl(gwf);
    return;
  }
  
  /* liquid helium */
  switch(driver_norm_type) {
  case DFT_DRIVER_NORMALIZE_BULK: /* bulk normalization */
    norm = cgrid2d_value_at_index(gwf->grid, gwf->grid->nx-1, 0); /* zmax, rmin */
    norm = sqrt(driver_rho0) / norm;
    cgrid2d_multiply(gwf->grid, norm);
    break;
  case DFT_DRIVER_NORMALIZE_DROPLET: /* helium droplet */
    if(!center_release) {
      double sq = sqrt(3.0*driver_rho0/4.0);
      for (i = 0; i < driver_nz; i++) {
	z = (i - driver_nz/2.0) * driver_step;
	for (j = 0; j < driver_nr; j++) {
	  r = j * driver_step;
	  if(sqrt(z * z + r * r) < driver_frad && cabs(gwf->grid->value[i * driver_nr + j]) < sq)
	    gwf->grid->value[i * driver_nr + j] = sq;
	}
      }
    }
    grid2d_wf_normalize_cyl(gwf);
    cgrid2d_multiply(gwf->grid, sqrt((double) driver_nhe));
    break;
  case DFT_DRIVER_NORMALIZE_COLUMN:
    fprintf(stderr, "libdft: Illegal normalization condition in 2-D.\n");
    exit(1);
  case DFT_DRIVER_NORMALIZE_SURFACE: /* 2-D surface */
    if(!center_release) {
      double sq = sqrt(0.9*driver_rho0);
      for (i = 0; i < driver_nz; i++) { /* force zero around z = 0 */
	z = (i - driver_nz/2.0) * driver_step;      
	for (j = 0; j < driver_nr; j++)
	  if(fabs(z) < driver_frad && cabs(gwf->grid->value[i * driver_nr + j]) < sq)
	    gwf->grid->value[i * driver_nr + j] = sq;
      }
    }
    grid2d_wf_normalize_cyl(gwf); /* normalize to given # of He */
    cgrid2d_multiply(gwf->grid, sqrt((double) driver_nhe));
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

EXPORT void dft_driver_initialize_2d() {

  static int been_here = 0;

  check_mode();

  if(!been_here) {
    if(driver_nz == 0) {
      fprintf(stderr, "libdft: dft_driver not properly initialized.\n");
      exit(1);
    }
    grid_timer_start(&timer);
    grid_threads_init(driver_threads);
    workspace1 = rgrid2d_alloc(driver_nz, driver_nr, driver_step, RGRID2D_NEUMANN_BOUNDARY, 0);
    workspace2 = rgrid2d_alloc(driver_nz, driver_nr, driver_step, RGRID2D_NEUMANN_BOUNDARY, 0);
    workspace3 = rgrid2d_alloc(driver_nz, driver_nr, driver_step, RGRID2D_NEUMANN_BOUNDARY, 0);
    workspace4 = rgrid2d_alloc(driver_nz, driver_nr, driver_step, RGRID2D_NEUMANN_BOUNDARY, 0);
    workspace5 = rgrid2d_alloc(driver_nz, driver_nr, driver_step, RGRID2D_NEUMANN_BOUNDARY, 0);
    workspace6 = rgrid2d_alloc(driver_nz, driver_nr, driver_step, RGRID2D_NEUMANN_BOUNDARY, 0);
    workspace7 = rgrid2d_alloc(driver_nz, driver_nr, driver_step, RGRID2D_NEUMANN_BOUNDARY, 0);
    cworkspace = cgrid2d_alloc(driver_nz, driver_nr, driver_step, CGRID2D_NEUMANN_BOUNDARY, 0);

    if(driver_dft_model & DFT_OT_BACKFLOW) {
      workspace8 = rgrid2d_alloc(driver_nz, driver_nr, driver_step, RGRID2D_NEUMANN_BOUNDARY, 0);
      workspace9 = rgrid2d_alloc(driver_nz, driver_nr, driver_step, RGRID2D_NEUMANN_BOUNDARY, 0);
    }
    density = rgrid2d_alloc(driver_nz, driver_nr, driver_step, RGRID2D_NEUMANN_BOUNDARY, 0);
    dft_driver_otf_2d = dft_ot2d_alloc(driver_dft_model, driver_nz, driver_nr, driver_step, MIN_SUBSTEPS, MAX_SUBSTEPS);
    if(driver_rho0 == 0.0) driver_rho0 = dft_driver_otf_2d->rho0;
    else dft_driver_otf_2d->rho0 = driver_rho0;
    fprintf(stderr, "libdft: rho0 = %le Angs^-3.\n", driver_rho0 / (GRID_AUTOANG * GRID_AUTOANG * GRID_AUTOANG));
    been_here = 1;
    fprintf(stderr, "libdft: %lf wall clock seconds for initialization.\n", grid_timer_wall_clock_time(&timer));
  }
}

/*
 * Set up the DFT calculation grid.
 *
 * nz      = number of grid points along z (long).
 * nr      = number of grid points along r (long).
 * threads = number of parallel execution threads (long).
 *
 * No return value.
 *
 */

EXPORT void dft_driver_setup_grid_2d(long nz, long nr, double step, long threads) {
  
  check_mode();

  driver_nr = nr;
  driver_nz = nz;
  driver_step = step;
  fprintf(stderr, "libgrid: Grid size = (%ld,%ld) with step = %le.\n", nz, nr, step);
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

EXPORT void dft_driver_setup_model_2d(long dft_model, long iter_mode, double rho0) {

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

EXPORT void dft_driver_setup_boundaries_2d(long boundary_type, double absb) {

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

EXPORT void dft_driver_setup_boundaries_damp_2d(double dmp) {

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

EXPORT void dft_driver_setup_normalization_2d(long norm_type, long nhe, double frad, long rels) {

  check_mode();

  driver_norm_type = norm_type;
  driver_nhe = nhe;
  driver_rels = rels;
  driver_frad = frad;
}

/*
 * Predict: propagate the given wf in time.
 *
 * what      = what is propagated: 0 = L-He, 1 = other.
 * ext_pot   = present external potential grid (rgrid2d *; input) (NULL = no ext. pot).
 * gwf       = liquid wavefunction to propagate (wf2d *; input).
 *             Note that gwf is NOT changed by this routine.
 * gwfp      = predicted wavefunction (wf2d *; output).
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
 * TODO: It is now impossible to propagate two different wavefunctions
 * simultaneously such that both will active predict/correct cycles.
 *
 */

EXPORT inline void dft_driver_propagate_predict_2d(long what, rgrid2d *ext_pot, wf2d *gwf, wf2d *gwfp, cgrid2d *potential, double tstep, long iter) {

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
    grid2d_wf_constant(gwf, sqrt(dft_driver_otf_2d->rho0));
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

  cgrid2d_copy(gwfp->grid, gwf->grid);

  /* 1/2 x kinetic */
  grid2d_wf_propagate_kinetic_cn_cyl(gwfp, htime, cworkspace); 
  if(driver_iter_mode) scale_wf(what, gwfp);
  cgrid2d_copy(gwf->grid, gwfp->grid);

  /* predict */
  cgrid2d_zero(potential);  // new
  if(!what) {
    grid2d_wf_density(gwfp, density);
    dft_ot2d_potential(dft_driver_otf_2d, potential, gwfp, density, workspace1, workspace2, workspace3, workspace4, workspace5, workspace6, workspace7, workspace8, workspace9);
  }
  /* absorbing boundary */
  if(!driver_iter_mode && driver_boundary_type == 1 && !what) {
    fprintf(stderr, "libdft: Predict - absorbing boundary for helium.\n");
    grid2d_wf_absorb_cyl(potential, density, driver_rho0, region_func, workspace1);
  }
  /* External potential for Helium */
  /* Im - He contribution */
  if(ext_pot) grid2d_add_real_to_complex_re(potential, ext_pot);

  /* potential */
  grid2d_wf_propagate_potential(gwfp, potential, time);
  if(driver_iter_mode) scale_wf(what, gwfp);
}

/*
 * Correct: propagate the given wf in time.
 *
 * what      = what is propagated: 0 = L-He, 1 = other.
 * ext_pot   = present external potential grid (rgrid2d *) (NULL = no ext. pot).
 * gwf       = liquid wavefunction to propagate (wf2d *).
 * gwfp      = predicted wavefunction (wf3d *; output).
 *             Note that gwfp is NOT changed by this routine.
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

EXPORT inline void dft_driver_propagate_correct_2d(long what, rgrid2d *ext_pot, wf2d *gwf, wf2d *gwfp, cgrid2d *potential, double tstep, long iter) {

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
    grid2d_wf_density(gwfp, density);
    /* no zeroing - add to the previous potential to get avg (new) */
    dft_ot2d_potential(dft_driver_otf_2d, potential, gwfp, density, workspace1, workspace2, workspace3, workspace4, workspace5, workspace6, workspace7, workspace8, workspace9);
  }
  /* absorbing boundary */
  if(!driver_iter_mode && driver_boundary_type == 1 && !what) {
    fprintf(stderr, "libdft: Correct - absorbing boundary for helium.\n");
    grid2d_wf_absorb_cyl(potential, density, driver_rho0, region_func, workspace1);
  }  
  /* External potential for Helium */
  /* Im - He contribution (new) */
  if(ext_pot) grid2d_add_real_to_complex_re(potential, ext_pot);
  /* average of future and current (new) */
  cgrid2d_multiply(potential, 0.5);

  /* potential */
  grid2d_wf_propagate_potential(gwf, potential, time);
  if(driver_iter_mode) scale_wf(what, gwf);
  
  /* 1/2 x kinetic */
  grid2d_wf_propagate_kinetic_cn_cyl(gwf, htime, cworkspace); 
  if(driver_iter_mode) scale_wf(what, gwf);
  
  fprintf(stderr, "libdft: Iteration %ld took %lf wall clock seconds.\n", iter, grid_timer_wall_clock_time(&timer));
  fflush(stdout);
}

/*
 * Prepare for convoluting potential and density.
 *
 * pot  = potential to be convoluted with (rgrid2d *).
 * dens = denisity to be convoluted with (rgrid2d *).
 *
 * This must be called before cgrid3d_driver_convolute_eval().
 * Both pot and dens are overwritten with their FFTs.
 * if either is specified as NULL, no transform is done for that grid.
 *
 * No return value.
 *
 */

EXPORT void dft_driver_convolution_prepare_2d(rgrid2d *pot, rgrid2d *dens) {

  check_mode();

  if (pot) rgrid2d_fft_cylindrical(pot);
  if (dens) rgrid2d_fft_cylindrical(dens);
}

/*
 * Convolute density and potential.
 *
 * out  = output from convolution (cgrid2d *).
 * pot  = potential grid that has been prepared with cgrid3d_driver_convolute_prepare().
 * dens = density against which has been prepared with cgrid3d_driver_convolute_prepare().
 *
 * No return value.
 *
 */

EXPORT void dft_driver_convolution_eval_2d(rgrid2d *out, rgrid2d *pot, rgrid2d *dens) {

  check_mode();

  rgrid2d_fft_cylindrical_convolute(out, pot, dens);
  rgrid2d_inverse_fft_cylindrical(out);
  rgrid2d_fft_cylindrical_cleanup(out, dft_ot2d_hankel_pad);
}

/*
 * Allocate a complex grid.
 *
 * Returns a pointer to the allocated grid.
 *
 */

EXPORT cgrid2d *dft_driver_alloc_cgrid_2d() {

  check_mode();

  if(driver_nz == 0) {
    fprintf(stderr, "libdft: dft_driver routines must be initialized first.\n");
    exit(1);
  }
  return cgrid2d_alloc(driver_nz, driver_nr, driver_step, CGRID2D_NEUMANN_BOUNDARY, 0);
}

/*
 * Allocate a real grid.
 *
 * Returns a pointer to the allocated grid.
 *
 */

EXPORT rgrid2d *dft_driver_alloc_rgrid_2d() {

  check_mode();

  if(driver_nz == 0) {
    fprintf(stderr, "libdft: dft_driver routines must be initialized first.\n");
    exit(1);
  }
  return rgrid2d_alloc(driver_nz, driver_nr, driver_step, RGRID2D_NEUMANN_BOUNDARY, 0);
}

/*
 * Allocate a wavefunction (initialized to sqrt(rho0)).
 *
 * mass = particle mass in a.u. (double).
 *
 * Returns pointer to the wavefunction.
 *
 */

EXPORT wf2d *dft_driver_alloc_wavefunction_2d(double mass) {

  wf2d *tmp;

  check_mode();

  if(driver_nz == 0) {
    fprintf(stderr, "libdft: dft_driver routines must be initialized first.\n");
    exit(1);
  }
  tmp = grid2d_wf_alloc(driver_nz, driver_nr, driver_step, mass, WF2D_NEUMANN_BOUNDARY, WF2D_2ND_ORDER_PROPAGATOR);
  cgrid2d_constant(tmp->grid, sqrt(driver_rho0));
  return tmp;
}

/*
 * Initialize a wavefunction to sqrt of a gaussian function.
 * Useful function for generating an initial guess for impurities.
 *
 * dft   = Wavefunction to be initialized (cgrid2d *; input/output).
 * cz    = Gaussian center alogn r (double; input).
 * cr    = Gaussian center alogn z (double; input).
 * width = Gaussian width (double; input).
 *
 */

struct asd {
  double cz, cr;
  double zp;
};
  

static double complex dft_gauss_2d(void *ptr, double z, double r) {

  struct asd *lp = (struct asd *) ptr;
  double zp = lp->zp, cz = z - lp->cz, cr = r - lp->cr;

  return sqrt(pow(zp * zp * M_PI / M_LN2, -3.0/2.0) * exp(-M_LN2 * (cz * cz + cr * cr) / (zp * zp)));
}

EXPORT void dft_driver_gaussian_wavefunction_2d(wf2d *dst, double cz, double cr, double width) {

  struct asd lp;

  lp.cz = cz;
  lp.cr = cr;
  lp.zp = width;
  grid2d_wf_map_cyl(dst, dft_gauss_2d, &lp);
}

/*
 * Read in density from a binary file (.grd).
 *
 * grid = place to store the read density (rgrid2d *).
 * file = filename for the file (char *). Note: the .grd extension must be given.
 *
 * No return value.
 *
 */

EXPORT void dft_driver_read_density_2d(rgrid2d *grid, char *file) {

  FILE *fp;
  char buf[512];

  check_mode();

  strcpy(buf, file);
  strcat(buf, ".grd");
  if(!(fp = fopen(buf, "r"))) {
    fprintf(stderr, "libdft: Can't open density grid file %s.\n", file);
    exit(1);
  }
  rgrid2d_read(grid, fp);
  fclose(fp);
  fprintf(stderr, "libdft: Density read from %s.\n", file);
}

/*
 * Write output to ascii & binary files.
 * .grd file is the full grid in binary format.
 * .z ASCII file cut along (z, 0.0)
 * .r ASCII file cut along (0.0, r)
 *
 * grid = density grid (rgrid2d *).
 * base = Basename for the output file (char *).
 *
 * No return value.
 *
 */

EXPORT void dft_driver_write_density_2d(rgrid2d *grid, char *base) {

  FILE *fp;
  char file[2048];
  long i, j;
  double z, r;

  check_mode();

  sprintf(file, "%s.grd", base);
  if(!(fp = fopen(file, "w"))) {
    fprintf(stderr, "libdft: Can't open %s for writing.\n", file);
    exit(1);
  }
  rgrid2d_write(grid, fp);
  fclose(fp);

  sprintf(file, "%s.z", base);
  if(!(fp = fopen(file, "w"))) {
    fprintf(stderr, "libdft: Can't open %s for writing.\n", file);
    exit(1);
  }
  j = 0;
  for(i = 0; i < grid->nx; i++) { 
    z = (i - grid->nx/2.0) * grid->step;
    fprintf(fp, "%.20le %.20le\n", z, rgrid2d_value_at_index(grid, i, j));
  }
  fclose(fp);

  sprintf(file, "%s.r", base);
  if(!(fp = fopen(file, "w"))) {
    fprintf(stderr, "libdft: Can't open %s for writing.\n", file);
    exit(1);
  }
  i = grid->nx / 2;
  for(j = 0; j < grid->ny; j++) {
    r = j * grid->step;
    fprintf(fp, "%.20le %.20le\n", r, rgrid2d_value_at_index(grid, i, j));
  }
  fclose(fp);

  fprintf(stderr, "libdft: Density written to %s.\n", file);
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

EXPORT void dft_driver_read_grid_2d(cgrid2d *grid, char *file) {

  FILE *fp;

  check_mode();

  if(!(fp = fopen(file, "r"))) {
    fprintf(stderr, "libdft: Can't open density grid file %s.\n", file);
    exit(1);
  }
  cgrid2d_read(grid, fp);
  fclose(fp);
}

/*
 * Write a complex grid to ascii & binary files.
 * .grd file is the full grid in binary format.
 * .z ASCII file cut along (z, 0.0)
 * .r ASCII file cut along (0.0, r)
 *
 * grid = grid to be written (cgrid2d *).
 * base = Basename for the output file (char *).
 *
 * No return value.
 *
 */

EXPORT void dft_driver_write_grid_2d(cgrid2d *grid, char *base) {

  FILE *fp;
  char file[2048];
  long i, j;
  double z, r;

  check_mode();

  sprintf(file, "%s.grd", base);
  if(!(fp = fopen(file, "w"))) {
    fprintf(stderr, "libdft: Can't open %s for writing.\n", file);
    exit(1);
  }
  cgrid2d_write(grid, fp);
  fclose(fp);

  sprintf(file, "%s.z", base);
  if(!(fp = fopen(file, "w"))) {
    fprintf(stderr, "libdft: Can't open %s for writing.\n", file);
    exit(1);
  }
  j = 0;
  for(i = 0; i < driver_nz; i++) { 
    z = (i - driver_nz/2.0) * driver_step;
    fprintf(fp, "%.20le %.20le %.20le\n", z, creal(cgrid2d_value_at_index(grid, i, j)), cimag(cgrid2d_value_at_index(grid, i, j)));
  }
  fclose(fp);

  sprintf(file, "%s.r", base);
  if(!(fp = fopen(file, "w"))) {
    fprintf(stderr, "libdft: Can't open %s for writing.\n", file);
    exit(1);
  }
  i = driver_nz / 2;
  for(j = 0; j < driver_nr; j++) {
    r = j * driver_step;
    fprintf(fp, "%.20le %.20le %.20le\n", r, creal(cgrid2d_value_at_index(grid, i, j)), cimag(cgrid2d_value_at_index(grid, i, j)));
  }
  fclose(fp);
}

/*
 * Calculate the total energy of the system.
 *
 * gwf     = wavefunction for the system (wf2d *; input).
 * ext_pot = external potential grid (rgrid2d *; input).
 *
 * Return value = total energy for the system (in a.u.).
 *
 * Note: the backflow is not included in the energy density calculation.
 *
 */

EXPORT double dft_driver_energy_2d(wf2d *gwf, rgrid2d *ext_pot) {

  return dft_driver_potential_energy_2d(gwf, ext_pot) + dft_driver_kinetic_energy_2d(gwf);
}

/*
 * Calculate the potential energy of the system.
 *
 * gwf     = wavefunction for the system (wf2d *; input).
 * ext_pot = external potential grid (rgrid2d *; input).
 *
 * Return value = potential energy for the system (in a.u.).
 *
 * Note: the backflow is not included in the energy density calculation.
 *
 */

EXPORT double dft_driver_potential_energy_2d(wf2d *gwf, rgrid2d *ext_pot) {

  check_mode();

  /* we may need more memory for this... */
  if(!workspace7) workspace7 = dft_driver_alloc_rgrid_2d();
  if(!workspace8) workspace8 = dft_driver_alloc_rgrid_2d();
  if(!workspace9) workspace9 = dft_driver_alloc_rgrid_2d();
  grid2d_wf_density(gwf, density);
  dft_ot2d_energy_density(dft_driver_otf_2d, workspace9, gwf, density, workspace1, workspace2, workspace3, workspace4, workspace5, workspace6, workspace7, workspace8);
  if(ext_pot) rgrid2d_add_scaled_product(workspace9, 1.0, density, ext_pot);

  return rgrid2d_integral_cyl(workspace9);
}

/*
 * Calculate the kinetic energy of the system.
 *
 * gwf     = wavefunction for the system (wf2d *; input).
 *
 * Return value = kinetic energy for the system (in a.u.).
 *
 */

EXPORT double dft_driver_kinetic_energy_2d(wf2d *gwf) {
  
  check_mode();

  if(!cworkspace) cworkspace = dft_driver_alloc_cgrid_2d();

  return grid2d_wf_energy_cyl(gwf, NULL, cworkspace);
}

/*
 * Return number of helium atoms represented by a given wavefuntion.
 *
 * gwf = wavefunction (wf2d *; input).
 *
 * Returns the # of He atoms (note: can be fractional).
 *
 */

/* TODO: fixme */

EXPORT double dft_driver_natoms_2d(wf2d *gwf) {

  check_mode();

  return creal(cgrid2d_integral_of_square_cyl(gwf->grid));
}

/*
 * Evaluate absorption/emission spectrum using the Andersson
 * expression. No zero-point correction for the impurity.
 *
 * density  = Current liquid density (rgrid2d *; input).
 * tstep    = Time step for constructing the time correlation function
 *            (double; input in fs). Typically around 1 fs.
 * endtime  = End time in constructing the time correlation function
 *            (double; input in fs). Typically less than 10,000 fs.
 * upperave = Averaging of the upper state potential.
 *            0: no averaging, 1 = average ZR.
 * upperz   = Upper state potential along the Z axis (char *; input).
 * upperr   = Upper state potential along the R axis (char *; input).
 * lowerave = Averaging of the lower state potential.
 *            0: no averaging, 1 = average ZR.
 * lowerz   = Lower state potential along the Z axis (char *; input).
 * lowerr   = Lower state potential along the R axis (char *; input).
 *
 * NOTE: This needs complex grid input!! 
 *
 * Returns the spectrum (cgrid1d *; output). Note that this is statically
 * allocated and overwritten by a subsequent call to this routine.
 * The spectrum will start from smaller to larger frequencies.
 * The spacing between the points is included in cm-1.
 *
 * TODO: Implement changes that were made in 3D (see driver3d.c).
 *
 */

static double complex dft_eval_exp(double complex a) { /* a contains t */

  return (1.0 - cexp(-I * a));
}

static double complex dft_do_int(rgrid2d *dens, rgrid2d *dpot, double t, cgrid2d *wrk) {

  grid2d_real_to_complex_re(wrk, dens);
  cgrid2d_multiply(wrk, t);
  cgrid2d_operate_one(wrk, wrk, dft_eval_exp);
  grid2d_product_complex_with_real(wrk, dens);
  return -cgrid2d_integral_cyl(wrk);
}

EXPORT cgrid1d *dft_driver_spectrum_2d(rgrid2d *density, double tstep, double endtime, int upperave, char *upperz, char *upperr, int lowerave, char *lowerz, char *lowerr) {

  rgrid2d *dpot;
  cgrid2d *wrk[256];
  static cgrid1d *corr = NULL;
  double t;
  long i, ntime;
  static long prev_ntime = -1;

  check_mode();

  endtime /= GRID_AUTOFS;
  tstep /= GRID_AUTOFS;
  ntime = (1 + (int) (endtime / tstep));
  dpot = rgrid2d_alloc(density->nx, density->ny, density->step, RGRID2D_NEUMANN_BOUNDARY, 0);
  for (i = 0; i < omp_get_max_threads(); i++)
    wrk[i] = cgrid2d_alloc(density->nx, density->ny, density->step, CGRID2D_NEUMANN_BOUNDARY, 0);
  if(ntime != prev_ntime) {
    if(corr) cgrid1d_free(corr);
    corr = cgrid1d_alloc(ntime, 0.1, CGRID1D_PERIODIC_BOUNDARY, 0);
    prev_ntime = ntime;
  }

  fprintf(stderr, "libdft: Upper level potential.\n");
  dft_common_potential_map_2d(upperave, upperz, upperr, workspace1);
  fprintf(stderr, "libdft: Lower level potential.\n");
  dft_common_potential_map_2d(lowerave, lowerz, lowerr, workspace2);
  rgrid2d_difference(dpot, workspace2, workspace1);

#pragma omp parallel for firstprivate(stderr,tstep,ntime,density,dpot,corr,wrk) private(i,t) default(none) schedule(runtime)
  for(i = 0; i < ntime; i++) {
    t = tstep * (double) i;
    corr->value[i] = cexp(dft_do_int(density, dpot, t, wrk[omp_get_thread_num()])) * pow(-1.0, (double) i);
    fprintf(stderr,"libdft: Corr(%le fs) = %le %le\n", t * GRID_AUTOFS, creal(corr->value[i]), cimag(corr->value[i]));
  }
  cgrid1d_fft(corr);
  for (i = 0; i < corr->nx; i++)
    corr->value[i] = cabs(corr->value[i]);
  
  corr->step = GRID_HZTOCM1 / (tstep * GRID_AUTOFS * 1E-15 * (double) ntime);

  rgrid2d_free(dpot);
  for(i = 0; i < omp_get_max_threads(); i++)
    cgrid2d_free(wrk[i]);

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
 * density  = Current liquid density (rgrid2d *; input).
 * imdensity= Current impurity zero-point density (cgrid2d *; input).
 * tstep    = Time step for constructing the time correlation function
 *            (double; input in fs). Typically around 1 fs.
 * endtime  = End time in constructing the time correlation function
 *            (double; input in fs). Typically less than 10,000 fs.
 * upperave = Averaging of the upperial state potential.
 *            0: no averaging, 1 = average ZR.
 * upperz   = Upper state potential along the Z axis (char *; input).
 * upperr   = Upper state potential along the R axis (char *; input).
 * lowerave = Averaging of the lower state potential.
 *            0: no averaging, 1 = average ZR.
 * lowerz   = Lower state potential along the Z axis (char *; input).
 * lowerr   = Lower state potential along the R axis (char *; input).
 *
 * NOTE: This needs complex grid input!! 
 *
 * Returns the spectrum (cgrid1d *; output). Note that this is statically
 * allocated and overwritten by a subsequent call to this routine.
 * The spectrum will start from smaller to larger frequencies.
 * The spacing between the points is included in cm-1.
 *
 */

static void do_gexp(cgrid2d *gexp, cgrid2d *dpot, double t) {

  cgrid2d_zero(gexp);
  cgrid2d_add_scaled(gexp, t, dpot);
  cgrid2d_operate_one(gexp, gexp, dft_eval_exp);
  // FIXME: a) calculate manually or b) map to 3D
  // Hankel does not support complex...
  //  cgrid2d_fft_cylindrical(gexp);
}

static double complex dft_do_int2(cgrid2d *gexp, cgrid2d *imdens, cgrid2d *dens, cgrid2d *dpot, double t, cgrid2d *wrk) {

  cgrid2d_zero(wrk);
  // FIXME, see above.
  //  cgrid2d_fft_convolute_cylindrical(wrk, dens, gexp);
  // cgrid2d_inverse_fft_cylindrical(wrk);
  cgrid2d_product(wrk, wrk, imdens);
  
  return -cgrid2d_integral_cyl(wrk);
}

EXPORT cgrid1d *dft_driver_spectrum_zp_2d(cgrid2d *density, cgrid2d *imdensity, double tstep, double endtime, int upperave, char *upperz, char *upperr, int lowerave, char *lowerz, char *lowerr) {

  cgrid2d *dpot, *wrk, *fft_density, *gexp;
  static cgrid1d *corr = NULL;
  double t;
  long i, ntime;
  static long prev_ntime = -1;

  check_mode();

  endtime /= GRID_AUTOFS;
  tstep /= GRID_AUTOFS;
  ntime = (1 + (int) (endtime / tstep));
  dpot = cgrid2d_alloc(density->nx, density->ny, density->step, CGRID2D_PERIODIC_BOUNDARY, 0);
  fft_density = cgrid2d_alloc(density->nx, density->ny, density->step, CGRID2D_PERIODIC_BOUNDARY, 0);
  wrk = cgrid2d_alloc(density->nx, density->ny, density->step, CGRID2D_PERIODIC_BOUNDARY, 0);
  gexp = cgrid2d_alloc(density->nx, density->ny, density->step, CGRID2D_PERIODIC_BOUNDARY, 0);
  if(ntime != prev_ntime) {
    if(corr) cgrid1d_free(corr);
    corr = cgrid1d_alloc(ntime, 0.1, CGRID1D_PERIODIC_BOUNDARY, 0);
    prev_ntime = ntime;
  }
  
  fprintf(stderr, "libdft: Upper level potential.\n");
  dft_common_potential_map_2d(upperave, upperz, upperr, workspace1);
  fprintf(stderr, "libdft: Lower level potential.\n");
  dft_common_potential_map_2d(lowerave, lowerz, lowerr, workspace2);
  rgrid2d_difference(workspace2, workspace2, workspace1);
  grid2d_real_to_complex_re(dpot, workspace2);
  
  cgrid2d_copy(fft_density, density);
  // TODO: no complex hankel...
  //  cgrid2d_fft_cylindrical(fft_density);
  
  // can't run in parallel - actually no much sense since the most time intensive
  // part is the fft (which runs in parallel)
  for(i = 0; i < ntime; i++) {
    t = tstep * (double) i;
    do_gexp(gexp, dpot, t); /* gexp grid + FFT */
    corr->value[i] = cexp(dft_do_int2(gexp, imdensity, fft_density, dpot, t, wrk)) * pow(-1.0, (double) i);
    fprintf(stderr,"libdft: Corr(%le fs) = %le %le\n", t * GRID_AUTOFS, creal(corr->value[i]), cimag(corr->value[i]));
  }
  cgrid1d_fft(corr);
  for (i = 0; i < corr->nx; i++)
    corr->value[i] = cabs(corr->value[i]);
  
  corr->step = GRID_HZTOCM1 / (tstep * GRID_AUTOFS * 1E-15 * (double) ntime);
  
  cgrid2d_free(dpot);
  cgrid2d_free(fft_density);
  cgrid2d_free(wrk);
  cgrid2d_free(gexp);
  
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
 * gwf      = Initial order parameter (used to get the initial density) (wf2d *).
 * nt       = Maximum number of time steps to be collected (long).
 * upperave = Averaging on the upper state (see dft_driver_potential_map()) (int).
 * upperz   = Upper potential file name along-z (char *).
 * upperr   = Upper potential file name along-r (char *).
 * lowerave = Averaging on the lower state (see dft_driver_potential_map()) (int).
 * lowerz   = Lower potential file name along-z (char *).
 * lowerr   = Lower potential file name along-r (char *).
 *
 */

static rgrid2d *xxupper = NULL;
static cgrid1d *tdpot = NULL;
static double eg;
static long ntime, cur_time;

EXPORT void dft_driver_spectrum_init_2d(wf2d *gwf, long nt, int upperave, char *upperz, char *upperr, int lowerave, char *lowerz, char *lowerr) {

  check_mode();

  cur_time = 0;
  ntime = nt;
  if(!xxupper)
    xxupper = rgrid2d_alloc(driver_nz, driver_nr, driver_step, RGRID2D_PERIODIC_BOUNDARY, 0);
  if(!tdpot)
    tdpot = cgrid1d_alloc(ntime, 0.1, CGRID1D_PERIODIC_BOUNDARY, 0);
  fprintf(stderr, "libdft: Upper level potential.\n");
  dft_common_potential_map_2d(upperave, upperz, upperr, xxupper);
  fprintf(stderr, "libdft: Lower level potential.\n");
  dft_common_potential_map_2d(lowerave, lowerz, lowerr, workspace1);
  grid2d_wf_density(gwf, workspace2);
  rgrid2d_product(workspace2, workspace2, workspace1);
  eg = rgrid2d_integral(workspace2);
  fprintf(stderr, "libdft: spectrum init complete.\n");
}

/*
 * Collect the difference energy data. 
 *
 * gwf     = the current wavefunction (used for calculating the liquid density) (wf2d *).
 *
 * TODO: _collect2  to allow for zero-point.
 *
 */

EXPORT void dft_driver_spectrum_collect_2d(wf2d *gwf) {

  check_mode();

  if(cur_time >= ntime) {
    fprintf(stderr, "libdft: initialized with too few points (spectrum collect).\n");
    exit(1);
  }
  grid2d_wf_density(gwf, workspace1);
  rgrid2d_product(workspace1, workspace1, xxupper);
  tdpot->value[cur_time] = rgrid2d_integral_cyl(workspace1) - eg;

  fprintf(stderr, "libdft: spectrum collect complete (point = %ld, value = %le K).\n", cur_time, cabs(tdpot->value[cur_time]) * GRID_AUTOK);
  cur_time++;
}

/*
 * Evaluate the spectrum.
 *
 * tstep       = Time step length at which the energy difference data was collected
 *               (fs; usually the simulation time step) (double).
 * zero_offset = Frequency offset (to account for 1/omega dependency); usually zero.
 * tc          = Exponential decay time constant (fs; double).
 * zerofill    = How many zeros to fill in before FFT (int). The total number of
 *               points cannot exceed ntime.
 *
 * Returns a pointer to the calculated spectrum (grid1d *). X-axis in cm$^{-1}$.
 *
 */

EXPORT cgrid1d *dft_driver_spectrum_evaluate_2d(double tstep, double zero_offset, double tc, int zerofill) {

  long i, tmp;
  double omega, de, ct;
  static cgrid1d *spectrum = NULL;
  static cgrid1d *tdpot2 = NULL;

  check_mode();

  if(cur_time + zerofill >= ntime) {
    fprintf(stderr, "libdft: cur_time + zerofill >= ntime. Increase ntime.\n");
    exit(1);
  }

  tstep /= GRID_AUTOFS;
  tc /= GRID_AUTOFS;
  if(!spectrum)
    spectrum = cgrid1d_alloc(cur_time + zerofill, GRID_HZTOCM1 / (tstep * GRID_AUTOFS * 1E-15 * (double) cur_time), CGRID1D_PERIODIC_BOUNDARY, 0);
  if(!tdpot2)
    tdpot2 = cgrid1d_alloc(cur_time + zerofill, 0.1, CGRID1D_PERIODIC_BOUNDARY, 0);
  for (i = 0; i < cur_time; i++) {
    if(tc > 0.0) {
      ct = i * (double) tstep;
      de = exp(-ct / tc);
    } else de = 1.0;
    tdpot2->value[i] = de * sin(creal(tdpot->value[i])) * pow(-1.0, (double) i);
    tdpot->value[i] = de * cos(creal(tdpot->value[i])) * pow(-1.0, (double) i);
  }
  for (i = cur_time; i < cur_time + zerofill; i++) {
    tdpot2->value[i] = 0.0;
    tdpot->value[i] = 0.0;
  }
  tmp = tdpot->nx;
  tdpot->nx = cur_time + zerofill;  
  cgrid1d_fft(tdpot);
  cgrid1d_fft(tdpot2);
  for (i = 0, omega = -0.5 * spectrum->step * (spectrum->nx - 1); i < cur_time; i++, omega += spectrum->step) {
    spectrum->value[i] = pow(creal(tdpot->value[i]) + cimag(tdpot2->value[i]), 2.0) + cpow(creal(tdpot2->value[i]) + cimag(tdpot->value[i]), 2.0);
    if(zero_offset != 0.0) spectrum->value[i] *= zero_offset + omega;
  }

  tdpot->nx = tmp;
  return spectrum;
}

/*
 * Evaluate the liquid velocity field for a given order paremeter,
 * $v = m_{He} \vec{J}/\rho$.
 *
 * gwf  = Order parameter for which the velocity field is evaluated (input; wf2d *).
 * vz    = Velocity field x component (output; rgrid2d *).
 * vr    = Velocity field y component (output; rgrid2d *).
 *
 */

EXPORT void dft_driver_veloc_field_2d(wf2d *wf, rgrid2d *vz, rgrid2d *vr) {

  check_mode();

  grid2d_wf_probability_flux(wf, vz, vr);  /* TODO: correct? */
  grid2d_wf_density(wf, workspace1);
  rgrid2d_division(vz, vz, workspace1);
  rgrid2d_division(vr, vr, workspace1);
}

/*
 * Evaluate liquid momentum according to:
 * $\int\rho\v_idr$ where $i = z,r$.
 *
 * wf = Order parameter for evaluation (wf2d *; input).
 * pz = Liquid momentum along z (double *; output).
 * pr = Liquid momentum along r (double *; output).
 *
 */

EXPORT void dft_driver_P_2d(wf2d *wf, double *pz, double *pr) {

  check_mode();

  /* TODO: correct? */
  grid2d_wf_probability_flux(wf, workspace1, workspace2);
  rgrid2d_multiply(workspace1, wf->mass);
  rgrid2d_multiply(workspace2, wf->mass);

  *pz = rgrid2d_integral_cyl(workspace1);
  *pr = rgrid2d_integral_cyl(workspace2);
}

/*
 * Evaluate liquid kinetic energy according to:
 * $\frac{1}{2}m_{He}\int\rho v^2dr$
 *
 * wf = Order parameter for evaluation (wf2d *; input).
 *
 * Returns the kinetic energy.
 *
 */

EXPORT double dft_driver_KE_2d(wf2d *wf) {

  check_mode();

  // TODO: correct?
  dft_driver_veloc_field_2d(wf, workspace1, workspace2);
  rgrid2d_product(workspace1, workspace1, workspace1);
  rgrid2d_product(workspace2, workspace2, workspace2);
  rgrid2d_sum(workspace1, workspace1, workspace2);
  grid2d_wf_density(wf, workspace2);
  rgrid2d_product(workspace1, workspace1, workspace2);
  rgrid2d_multiply(workspace1, wf->mass / 2.0);
  return rgrid2d_integral_cyl(workspace1);
}

/*
 * Produce radially averaged density from a 2-D grid (about origin).
 *
 * radial = Radial density (rgrid1d *; output).
 * grid   = Source grid (rgrid2d *; input).
 * dtheta = Integration step size along theta in radians (double; input).
 * dphi   = Integration step size along phi in radians (double, input).
 * zc     = z coorinate for the center (double; input).
 *
 */

EXPORT void dft_driver_radial_2d(rgrid1d *radial, rgrid2d *grid, double dtheta, double dphi, double zc) {
  
  double rr, theta;
  double z, r, step = radial->step, tmp, *value = radial->value;
  long ri, thetai, phii, nx = radial->nx;

  check_mode();

  r = 0.0;
  for (ri = 0; ri < nx; ri++) {
    tmp = 0.0;
#pragma omp parallel for firstprivate(grid,r,dtheta,dphi,zc,step,value) private(theta,z,rr,thetai,phii) reduction(+:tmp) default(none) schedule(runtime)
    for (thetai = 0; thetai <= (long) (M_PI / dtheta); thetai++)
      for (phii = 0; phii <= (long) (2.0 * M_PI / dtheta); phii++) {
	theta = thetai * dtheta;
	//	phi = phii * dphi;
	z = r * cos(theta) + zc;
	rr = r * fabs(sin(theta));
	tmp += rgrid2d_value_cyl(grid, z, rr) * sin(theta);
      }
    tmp *= dtheta * dphi / (4.0 * M_PI);
    value[ri] = tmp;
    r += step;
  }
}

/*
 * Produce radially averaged grid from a 2-D complex grid (about origin).
 *
 * radial = Radial grid (cgrid1d *; output).
 * grid   = Source grid (cgrid2d *; input).
 * dtheta = Integration step size along theta in radians (double; input).
 * dphi   = Integration step size along phi in radians (double, input).
 * zc     = z coorinate for the center (double; input).
 *
 */

EXPORT void dft_driver_radial_complex_2d(cgrid1d *radial, cgrid2d *grid, double dtheta, double dphi, double zc) {
  
  double rr, theta;
  double z, r, step = radial->step;
  double complex tmp, *value = radial->value;
  long ri, thetai, phii, nx = radial->nx;

  check_mode();

  r = 0.0;
  for (ri = 0; ri < nx; ri++) {
    tmp = 0.0;
#pragma omp parallel for firstprivate(grid,r,dtheta,dphi,zc,step,value) private(theta,z,rr,thetai,phii) reduction(+:tmp) default(none) schedule(runtime)
    for (thetai = 0; thetai <= (long) (M_PI / dtheta); thetai++)
      for (phii = 0; phii <= (long) (2.0 * M_PI / dtheta); phii++) {
	theta = thetai * dtheta;
	//	phi = phii * dphi;
	z = r * cos(theta) + zc;
	rr = r * fabs(sin(theta));
	tmp += cgrid2d_value_cyl(grid, z, rr) * sin(theta);
      }
    tmp *= dtheta * dphi / (4.0 * M_PI);
    value[ri] = tmp;
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

EXPORT double dft_driver_spherical_rb_2d(rgrid2d *density) {

  double disp, bulk;

  check_mode();

  bulk = density->value[0];
  // bulk = driver_rho0;
  rgrid2d_multiply(density, -1.0);
  rgrid2d_add(density, bulk);
  disp = rgrid2d_integral_cyl(density);
  rgrid2d_add(density, -bulk);
  rgrid2d_multiply(density, -1.0);

  return pow(disp * 3.0 / (4.0 * M_PI * bulk), 1.0 / 3.0);
}

/*
 * Calculate convergence norm:
 * max |\rho(r,t) - \rho(r,t - \Delta t)|
 *
 * density = Current density (rgrid3d *, input).
 *
 * Return value: norm.
 *
 * Notes: 
 *   - This must be called during every iteration. For the first iteration
 *     this always returns 1.0.
 *   - This allocates an additional grid for the previous density (static).
 *
 * It is often a good idea to aim at "zero" - especially if real time dynamics
 * will be run afterwards.
 *
 */

static rgrid2d *prev_dens = NULL;

EXPORT double dft_driver_norm_2d(rgrid2d *density) {

  long i, nx = density->nx, ny = density->ny;
  double mx = -1.0, tmp;

  check_mode();

  if(!prev_dens) {
    prev_dens = dft_driver_alloc_rgrid_2d();
    rgrid2d_copy(prev_dens, density);
    return 1.0;
  }

  for (i = 0; i < nx * ny; i++) {
    if((tmp = fabs(density->value[i] - prev_dens->value[i])) > mx) mx = tmp;
  }
  
  rgrid2d_copy(prev_dens, density);
  
  return mx;
}

/*
 * Force spherical symmetry by spherical averaging.
 *
 * wf = wavefunction to be averaged (wf2d *).
 * zc = z coordinate for the center (double).
 *
 */

EXPORT void dft_driver_force_spherical_2d(wf2d *wf, double zc) {

  long i, j, k, len;
  long nz = wf->grid->nx, nr = wf->grid->ny;
  double step = wf->grid->step;
  double z, r, z2, r2, d;
  cgrid1d *average;
  double complex *value = wf->grid->value, *avalue;

  check_mode();

  len = (nr > nz/2) ? nr : nz/2;
  
  average = cgrid1d_alloc(len, step, CGRID1D_PERIODIC_BOUNDARY, 0);
  avalue = average->value;
  dft_driver_radial_complex_2d(average, wf->grid, 0.01, 0.01, zc);

  /* Write spherical average back to wf */
  for(i = 0; i < nz; i++) {
    z = (i - nz/2) * step;
    z2 = (z - zc) * (z - zc);
    for (j = 0; j < nr; j++) {
      r = j * step;
      r2 = r * r;
      d = sqrt(z2 + r2);
      k = (long) (0.5 + d / step);
      if(k >= len) k = len-1;
      value[i * nr + j] = avalue[k];
    }
  }
  cgrid1d_free(average);
}
