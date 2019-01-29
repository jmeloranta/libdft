/*
 * Impurity atom in superfluid helium (no zero-point).
 *
 * All input in a.u. except the time step, which is fs.
 *
 * NOTE: The sample potential files (OCS and HCN) were generated
 *       with old version of libgrid and they must be read in
 *       using the _compat versions of the grid read-in functions.
 *
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <grid/grid.h>
#include <grid/au.h>
#include <dft/dft.h>
#include <dft/ot.h>

#define TIME_STEP 10.0  /* fs */
#define MAXITER 10000
#define NX 128
#define NY 128
#define NZ 128
#define STEP 0.5
#define THREADS 0

#define PRESSURE 0.0

/* Molecule */
#define OCS 1
/* #define HCN 1 */

#ifdef OCS
#define ID "OCS molecule"
#define NN 3
/* Molecule along x axis */
/*                                     O                       C                    S       */
static REAL masses[NN] = {15.9994 / GRID_AUTOAMU, 12.01115 / GRID_AUTOAMU, 32.064 / GRID_AUTOAMU};
static REAL x[NN] = {-3.1824324, -9.982324e-01, 1.9619176}; /* Bohr */
//static REAL masses[NN] = {32.064 / GRID_AUTOAMU, 12.01115 / GRID_AUTOAMU, 15.9994 / GRID_AUTOAMU};   /* SCO */
//static REAL x[NN] = {-2.96015, 0.0, 2.18420};
static REAL y[NN] = {0.0, 0.0, 0.0};
static REAL z[NN] = {0.0, 0.0, 0.0};
// #define POTENTIAL "newocs_pairpot_128_0.5.grd"
#define POTENTIAL "ocs_pairpot_128_0.5.grd"
#define SWITCH_AXIS 1                  /* Potential was along z - switch to x */
#define OMEGA 1E-9
#endif

#ifdef HCN
#define ID "HCN molecule"
#define NN 3
/* Molecule along x axis */
/*                                     H                       C                    N       */
static REAL masses[NN] = {1.00794 / GRID_AUTOAMU, 12.0107 / GRID_AUTOAMU, 14.0067 / GRID_AUTOAMU};
static REAL x[NN] = {-1.064 / GRID_AUTOANG - 1.057205e+00, 0.0 / GRID_AUTOANG - 1.057205e+00, 1.156 / GRID_AUTOANG - 1.057205e+00}; /* Bohr */
static REAL y[NN] = {0.0, 0.0, 0.0};
static REAL z[NN] = {0.0, 0.0, 0.0};
#define POTENTIAL "hcn_pairpot_128_0.5"
#define SWITCH_AXIS 1                  /* Potential was along z - switch to x */
#define OMEGA 1E-9
#endif

REAL switch_axis(void *xx, REAL x, REAL y, REAL z) {

  rgrid *grid = (rgrid *) xx;

  return rgrid_value(grid, z, y, x);  // swap x and z -> molecule along x axis
}

int main(int argc, char **argv) {

  dft_ot_functional *otf;
  cgrid *potential_store;
  rgrid *ext_pot, *density, *px, *py, *pz;
  wf *gwf, *gwfp;
  INT iter, N, i;
  REAL energy, natoms, beff, i_add, lz, i_free, b_free, mass, cmx, cmy, cmz, mu0, rho0;
  grid_timer timer;

  /* Normalization condition */
  if(argc != 2) {
    fprintf(stderr, "Usage: imag_time N\n");
    exit(1);
  }
  N = (INT) atoi(argv[1]);

#ifdef USE_CUDA
//  cuda_enable(1);
#endif

  /* Initialize threads & use wisdom */
  grid_set_fftw_flags(1);    // FFTW_MEASURE
  grid_threads_init(THREADS);
  grid_fft_read_wisdom(NULL);

  /* Allocate wave functions (CN needed for rotation) */
  if(!(gwf = grid_wf_alloc(NX, NY, NZ, STEP, DFT_HELIUM_MASS, WF_NEUMANN_BOUNDARY, WF_2ND_ORDER_CN, "gwf"))) {
    fprintf(stderr, "Cannot allocate gwf.\n");
    exit(1);
  }
  gwfp = grid_wf_clone(gwf, "gwfp");
  if(N) {
    gwf->norm = gwfp->norm = (REAL) N;
    printf("Helium dropet, N = " FMT_I ".\n", N);
  } else printf("Bulk helium liquid.\n");

  /* Allocate OT functional */
  if(!(otf = dft_ot_alloc(DFT_OT_PLAIN, gwf, DFT_MIN_SUBSTEPS, DFT_MAX_SUBSTEPS))) {
    fprintf(stderr, "Cannot allocate otf.\n");
    exit(1);
  }
  rho0 = dft_ot_bulk_density_pressurized(otf, PRESSURE);
  mu0 = dft_ot_bulk_chempot_pressurized(otf, PRESSURE);
  printf("mu0 = " FMT_R " K/atom, rho0 = " FMT_R " Angs^-3.\n", mu0 * GRID_AUTOK, rho0 / (GRID_AUTOANG * GRID_AUTOANG * GRID_AUTOANG));

  printf("ID: %s (N = " FMT_I ")\n", ID, N);

  /* Allocate space for external potential */
  ext_pot = rgrid_clone(otf->density, "ext_pot");
  potential_store = cgrid_clone(gwf->grid, "potential_store"); /* temporary storage */
  density = rgrid_clone(otf->density, "density");
  px = rgrid_clone(otf->density, "px");
  py = rgrid_clone(otf->density, "py");
  pz = rgrid_clone(otf->density, "pz");

  /* Read external potential from file */
  density->value_outside = RGRID_DIRICHLET_BOUNDARY;  // for extrapolation to work
#ifdef SWITCH_AXIS
  rgrid_read_grid_compat(density, POTENTIAL);
  rgrid_map(ext_pot, switch_axis, density);
#else
  rgrid_read_grid_compat(ext_pot, POTENTIAL);
#endif
  density->value_outside = RGRID_PERIODIC_BOUNDARY;   // done, back to original

  /* Rotation about z-axis */
  printf("Omega = " FMT_R "\n", OMEGA);
  cgrid_set_rotation(gwf->grid, OMEGA);
  cgrid_set_rotation(gwfp->grid, OMEGA);

  cgrid_constant(gwf->grid, SQRT(rho0));

  for (iter = 0; iter < MAXITER; iter++) {
    
    // Center of mass of the rotating system
    // 1. The molecule
    mass = cmx = cmy = cmz = 0.0;
    for (i = 0; i < NN; i++) {
      cmx += x[i] * masses[i];
      cmy += y[i] * masses[i];
      cmz += z[i] * masses[i];
      mass += masses[i];
    }
    cmx /= mass; cmy /= mass; cmz /= mass;
    // 2. Liquid
    grid_wf_probability_flux_y(gwf, density);
    cmx += rgrid_integral(density) * gwf->mass / (2.0 * OMEGA * mass);
    grid_wf_probability_flux_x(gwf, density);
    cmy -= rgrid_integral(density) * gwf->mass / (2.0 * OMEGA * mass);
    printf("Current center of inertia: " FMT_R " " FMT_R " " FMT_R "\n", cmx, cmy, cmz);

    /* Moment of inertia about the center of mass for the molecule */
    i_free = 0.0;
    for (i = 0; i < NN; i++) {
      REAL x2, y2, z2;
      x2 = x[i] - cmx; x2 *= x2;
      y2 = y[i] - cmy; y2 *= y2;
      z2 = z[i] - cmz; z2 *= z2;
      i_free += masses[i] * (x2 + y2 + z2);
    }
    b_free = HBAR * HBAR / (2.0 * i_free);
    printf("I_molecule = " FMT_R " AMU Angs^2\n", i_free * GRID_AUTOAMU * GRID_AUTOANG * GRID_AUTOANG);
    printf("B_molecule = " FMT_R " cm-1.\n", b_free * GRID_AUTOCM1);
    /* Liquid contribution to the moment of inertia */
    cgrid_set_origin(gwf->grid, cmx, cmy, cmz); // Evaluate L about center of mass in grid_wf_l() and -wL_z in the Hamiltonian
    cgrid_set_origin(gwfp->grid, cmx, cmy, cmz);// the point x=0 is shift by cmX 
    lz = grid_wf_lz(gwf, otf->workspace1, otf->workspace2);
    i_add = gwf->mass * lz / OMEGA;  // grid_wf_lz() does not multiply by mass as did the dft
    printf("I_eff = " FMT_R " AMU Angs^2.\n", (i_free + i_add) * GRID_AUTOAMU * GRID_AUTOANG * GRID_AUTOANG);
    beff =  HBAR * HBAR / (2.0 * (i_free + i_add));
    printf("B_eff = " FMT_R " cm-1.\n", beff * GRID_AUTOCM1);

    if(iter == 5) grid_fft_write_wisdom(NULL);

    grid_timer_start(&timer);

    /* Predict-Correct */
    grid_real_to_complex_re(potential_store, ext_pot);
    dft_ot_potential(otf, potential_store, gwf);
    cgrid_add(potential_store, -mu0);
    grid_wf_propagate_predict(gwf, gwfp, potential_store, -I * TIME_STEP / GRID_AUTOFS);
    grid_add_real_to_complex_re(potential_store, ext_pot);
    dft_ot_potential(otf, potential_store, gwfp);
    cgrid_add(potential_store, -mu0);
    cgrid_multiply(potential_store, 0.5);  // Use (current + future) / 2
    grid_wf_propagate_correct(gwf, potential_store, -I * TIME_STEP / GRID_AUTOFS);
    if(N) grid_wf_normalize(gwf); // droplet
    // If N = 0, Chemical potential included - no need to normalize
    printf("Norm = " FMT_R "\n", grid_wf_norm(gwf));

    printf("Iteration " FMT_I " - Wall clock time = " FMT_R " seconds.\n", iter, grid_timer_wall_clock_time(&timer));

    if(!(iter % 100)) {
      char buf[512];
      grid_wf_density(gwf, density);
      sprintf(buf, "output-" FMT_I, iter);
      rgrid_write_grid(buf, density);
      dft_ot_energy_density(otf, density, gwf);
      rgrid_add_scaled_product(density, 1.0, otf->density, ext_pot);
      energy = grid_wf_energy(gwf, NULL) + rgrid_integral(density);
      natoms = grid_wf_norm(gwf);
      printf("Total energy is " FMT_R " K\n", energy * GRID_AUTOK);
      printf("Number of He atoms is " FMT_R ".\n", natoms);
      printf("Energy / atom is " FMT_R " K\n", (energy/natoms) * GRID_AUTOK);
      grid_wf_probability_flux(gwf, px, py, pz);
      sprintf(buf, "flux_x-" FMT_I, iter);
      rgrid_write_grid(buf, px);
      sprintf(buf, "flux_y-" FMT_I, iter);
      rgrid_write_grid(buf, py);
      sprintf(buf, "flux_z-" FMT_I, iter);
      rgrid_write_grid(buf, pz);
    }
  }
  return 0;
}
