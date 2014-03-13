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

#define HELIUM_MASS (4.002602 / GRID_AUTOAMU)
#define HBAR 1.0        /* au */
#define C_SI 2.187691E6 /* au */

#define TIME_STEP 10.0  /* fs */
#define MAXITER 10000
#define NX 64
#define NY 64
#define NZ 64
#define STEP 1.0

#define OCS 1 /* OCS molecule */

#ifdef OCS
/* #define I_FREE (83.1 * I_CONV)   // Free molecule moment of inertia in amu Angs^2 */
#define ID "OCS molecule"
#define NN 3
/* Molecule along x axis */
/*                                     O                       C                    S       */
static double masses[NN] = {15.9994 / GRID_AUTOAMU, 12.01115 / GRID_AUTOAMU, 32.064 / GRID_AUTOAMU};
//static double x[NN] = {-2.18420, 0.0, 2.96015}; /* Bohr */
static double x[NN] = {-3.1824324, -9.982324e-01, 1.9619176}; /* Bohr */
static double y[NN] = {0.0, 0.0, 0.0};
static double z[NN] = {0.0, 0.0, 0.0};
#define POTENTIAL "ocs_pairpot_128_0.5"
#define SWITCH_AXIS 1                  /* Potential was along z - switch to x */
#define OMEGA 1E-9
#endif

double switch_axis(void *xx, double x, double y, double z) {

  rgrid3d *grid = (rgrid3d *) xx;

  return rgrid3d_value(grid, z, y, x);  // swap x and z -> molecule along x axis
}

double func_r(void *xx, double val, double x, double y, double z) {

  unsigned int what = (unsigned int) xx;

  switch(what) {
  case 0:
    return x;
  case 1:
    return y;
  case 2:
    return z;
  default:
    fprintf(stderr, "Illegal value for xx.\n");
    exit(1);
  }
  return 0.0;
}

int main(int argc, char **argv) {

  cgrid3d *potential_store, *workspace;
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

  printf("ID: %s\n", ID);

  /* Set up rotating liquid */
  dft_driver_kinetic = DFT_DRIVER_KINETIC_CN_NBC_ROT;

  /* Initialize the DFT driver */
  dft_driver_initialize();

  /* Allocate space for external potential */
  ext_pot = dft_driver_alloc_rgrid();
  potential_store = dft_driver_alloc_cgrid(); /* temporary storage */
  workspace = dft_driver_alloc_cgrid(); /* temporary storage */
  density = dft_driver_alloc_rgrid();
  px = dft_driver_alloc_rgrid();
  py = dft_driver_alloc_rgrid();
  pz = dft_driver_alloc_rgrid();

  /* Allocate space for wavefunctions (initialized to sqrt(rho0)) */
  gwf = dft_driver_alloc_wavefunction(HELIUM_MASS); /* helium wavefunction */
  gwfp = dft_driver_alloc_wavefunction(HELIUM_MASS);/* temp. wavefunction */

  /* Read external potential from file */
  density->value_outside = RGRID3D_DIRICHLET_BOUNDARY;  // for extrapolation to work
  dft_driver_read_density(density, POTENTIAL);
#ifdef SWITCH_AXIS
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
    grid3d_wf_momentum_y(gwf, potential_store, workspace);
    cgrid3d_conjugate_product(potential_store, gwf->grid, potential_store);
    cmx += cgrid3d_integral(potential_store) / (2.0 * omega * mass);
    grid3d_wf_momentum_x(gwf, potential_store, workspace);
    cgrid3d_conjugate_product(potential_store, gwf->grid, potential_store);
    cmy -= cgrid3d_integral(potential_store) / (2.0 * omega * mass);
    printf("Center of mass: %le %le %le\n", cmx, cmy, cmz);

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

    /* Liquid moment of inertia */
    cgrid3d_set_origin(gwf->grid, NX/2.0 + cmx/STEP, NY/2.0 + cmy/STEP, NZ/2.0 + cmz/STEP); // Evaluate L about center of mass (in dft_driver_L() and -wL_z in the Hamiltonian
    cgrid3d_set_origin(gwfp->grid, NX/2.0 + cmx/STEP, NY/2.0 + cmy/STEP, NZ/2.0 + cmz/STEP);// Grid origin is at (NX/2,NY/2,NZ/2)
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
#if 0
      grid3d_wf_probability_flux(gwf, px, py, pz);
      sprintf(buf, "flux_x-%ld", iter);
      dft_driver_write_density(px, buf);
      sprintf(buf, "flux_y-%ld", iter);
      dft_driver_write_density(py, buf);
      sprintf(buf, "flux_z-%ld", iter);
      dft_driver_write_density(pz, buf);
#endif
    }
  }
  /* At this point gwf contains the converged wavefunction */
  grid3d_wf_density(gwf, density);
  dft_driver_write_density(density, "output");
}
