/*
 * Impurity atom in superfluid helium (no zero-point).
 *
 * All input in a.u. except the time step, which is fs.
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
#define NX 64
#define NY 64
#define NZ 64
#define STEP 1.0

/* #define OCS 1 /* OCS molecule */
#define HCN 1 /* HCN molecule */

#ifdef OCS
#define ID "OCS molecule"
#define NN 3
/* Molecule along x axis */
/*                                     O                       C                    S       */
static double masses[NN] = {15.9994 / GRID_AUTOAMU, 12.01115 / GRID_AUTOAMU, 32.064 / GRID_AUTOAMU};
static double x[NN] = {-3.1824324, -9.982324e-01, 1.9619176}; /* Bohr */
//static double masses[NN] = {32.064 / GRID_AUTOAMU, 12.01115 / GRID_AUTOAMU, 15.9994 / GRID_AUTOAMU};   /* SCO */
//static double x[NN] = {-2.96015, 0.0, 2.18420};
static double y[NN] = {0.0, 0.0, 0.0};
static double z[NN] = {0.0, 0.0, 0.0};
// #define POTENTIAL "newocs_pairpot_128_0.5"
#define POTENTIAL "ocs_pairpot_128_0.5"
#define SWITCH_AXIS 1                  /* Potential was along z - switch to x */
#define OMEGA 1E-9
#endif

#ifdef HCN
#define ID "HCN molecule"
#define NN 3
/* Molecule along x axis */
/*                                     H                       C                    N       */
static double masses[NN] = {1.00794 / GRID_AUTOAMU, 12.0107 / GRID_AUTOAMU, 14.0067 / GRID_AUTOAMU};
static double x[NN] = {-1.064 / GRID_AUTOANG - 1.057205e+00, 0.0 / GRID_AUTOANG - 1.057205e+00, 1.156 / GRID_AUTOANG - 1.057205e+00}; /* Bohr */
static double y[NN] = {0.0, 0.0, 0.0};
static double z[NN] = {0.0, 0.0, 0.0};
#define POTENTIAL "hcn_pairpot_128_0.5"
#define OMEGA 1E-9
#endif

#define HELIUM_MASS (4.002602 / GRID_AUTOAMU)
#define HBAR 1.0        /* au */

double switch_axis(void *xx, double x, double y, double z) {

  rgrid3d *grid = (rgrid3d *) xx;

  return rgrid3d_value(grid, z, y, x);  // swap x and z -> molecule along x axis
}

int main(int argc, char **argv) {

  cgrid3d *potential_store;
  rgrid3d *ext_pot, *density, *px, *py, *pz;
  wf3d *gwf, *gwfp;
  long iter, N, i;
  double energy, natoms, omega, rp, beff, i_add, lx, ly, lz, i_free, b_free, mass, cmx, cmy, cmz;

  /* Setup DFT driver parameters (256 x 256 x 256 grid) */
  dft_driver_setup_grid(NX, NY, NZ, STEP /* Bohr */, 8 /* threads */);
  /* Plain Orsay-Trento in imaginary time */
  dft_driver_setup_model(DFT_OT_PLAIN + DFT_OT_HD, DFT_DRIVER_IMAG_TIME, 0.0);
  /* No absorbing boundary */
  dft_driver_setup_boundaries(DFT_DRIVER_BOUNDARY_REGULAR, 2.0);
  /* Normalization condition */
  if(argc != 2) {
    fprintf(stderr, "Usage: imag_time N\n");
    exit(1);
  }
  N = atoi(argv[1]);
  if(N == 0) 
    dft_driver_setup_normalization(DFT_DRIVER_DONT_NORMALIZE, 0, 0.0, 1); // 1 = release center immediately
  else
    dft_driver_setup_normalization(DFT_DRIVER_NORMALIZE_DROPLET, N, 0.0, 1); // 1 = release center immediately

  printf("ID: %s (N = %ld)\n", ID, N);

  /* Set up rotating liquid */
  dft_driver_kinetic = DFT_DRIVER_KINETIC_CN_NBC_ROT;

  /* Initialize the DFT driver */
  dft_driver_initialize();

  /* Allocate space for external potential */
  ext_pot = dft_driver_alloc_rgrid();
  potential_store = dft_driver_alloc_cgrid(); /* temporary storage */
  density = dft_driver_alloc_rgrid();
  px = dft_driver_alloc_rgrid();
  py = dft_driver_alloc_rgrid();
  pz = dft_driver_alloc_rgrid();

  /* Allocate space for wavefunctions (initialized to sqrt(rho0)) */
  gwf = dft_driver_alloc_wavefunction(HELIUM_MASS); /* helium wavefunction */
  gwfp = dft_driver_alloc_wavefunction(HELIUM_MASS);/* temp. wavefunction */

  /* Read external potential from file */
  density->value_outside = RGRID3D_DIRICHLET_BOUNDARY;  // for extrapolation to work
#ifdef SWITCH_AXIS
  dft_driver_read_density(density, POTENTIAL);
  rgrid3d_map(ext_pot, switch_axis, density);
#else
  dft_driver_read_density(ext_pot, POTENTIAL);
#endif
  density->value_outside = RGRID3D_PERIODIC_BOUNDARY;   // done, back to original
  rgrid3d_add(ext_pot, 7.2 / GRID_AUTOK);

  omega = OMEGA;
  printf("Omega = %le\n", omega);
  dft_driver_setup_rotation_omega(omega);

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
    grid3d_wf_probability_flux_y(gwf, density);
    cmx += rgrid3d_integral(density) * gwf->mass / (2.0 * omega * mass);
    grid3d_wf_probability_flux_x(gwf, density);
    cmy -= rgrid3d_integral(density) * gwf->mass / (2.0 * omega * mass);
    printf("Current center of inertia: %le %le %le\n", cmx, cmy, cmz);

    /* Moment of inertia about the center of mass for the molecule */
    i_free = 0.0;
    for (i = 0; i < NN; i++) {
      double x2, y2, z2;
      x2 = x[i] - cmx; x2 *= x2;
      y2 = y[i] - cmy; y2 *= y2;
      z2 = z[i] - cmz; z2 *= z2;
      i_free += masses[i] * (x2 + y2 + z2);
    }
    b_free = HBAR * HBAR / (2.0 * i_free);
    printf("I_molecule = %le AMU Angs^2\n", i_free * GRID_AUTOAMU * GRID_AUTOANG * GRID_AUTOANG);
    printf("B_molecule = %le cm-1.\n", b_free * GRID_AUTOCM1);
    /* Liquid contribution to the moment of inertia */
    cgrid3d_set_origin(gwf->grid, cmx, cmy, cmz); // Evaluate L about center of mass in dft_driver_L() and -wL_z in the Hamiltonian
    cgrid3d_set_origin(gwfp->grid, cmx, cmy, cmz);// the point x=0 is shift by cmX 
    dft_driver_L(gwf, &lx, &ly, &lz);
    i_add = lz / omega;
    printf("I_eff = %le AMU Angs^2.\n", (i_free + i_add) * GRID_AUTOAMU * GRID_AUTOANG * GRID_AUTOANG);
    beff =  HBAR * HBAR / (2.0 * (i_free + i_add));
    printf("B_eff = %le cm-1.\n", beff * GRID_AUTOCM1);

    dft_driver_propagate_predict(DFT_DRIVER_PROPAGATE_HELIUM, ext_pot, gwf, gwfp, potential_store, TIME_STEP, iter);
    dft_driver_propagate_correct(DFT_DRIVER_PROPAGATE_HELIUM, ext_pot, gwf, gwfp, potential_store, TIME_STEP, iter);

    if(!(iter % 500)) {
      char buf[512];
      sprintf(buf, "output-%ld", iter);
      grid3d_wf_density(gwf, density);
      dft_driver_write_density(density, buf);
      energy = dft_driver_energy(gwf, ext_pot);
      natoms = dft_driver_natoms(gwf);
      printf("Total energy is %le K\n", energy * GRID_AUTOK);
      printf("Number of He atoms is %le.\n", natoms);
      printf("Energy / atom is %le K\n", (energy/natoms) * GRID_AUTOK);
      grid3d_wf_probability_flux(gwf, px, py, pz);
      sprintf(buf, "flux_x-%ld", iter);
      dft_driver_write_density(px, buf);
      sprintf(buf, "flux_y-%ld", iter);
      dft_driver_write_density(py, buf);
      sprintf(buf, "flux_z-%ld", iter);
      dft_driver_write_density(pz, buf);
    }
  }
  return 0;
}
