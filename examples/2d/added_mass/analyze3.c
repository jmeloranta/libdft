/*
 * Convert given WFs for electron, super and normal
 * to corresponding densities and velocity fields.
 *
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <complex.h>
#include <grid/grid.h>
#include <grid/au.h>
#include <dft/dft.h>
#include <dft/ot.h>

#include "added_mass3.h"

int main(int argc, char **argv) {

  rgrid2d *density, *vz, *vr;
  wf2d *gwf, *nwf, *impwf;

  if(argc != 4) {
    printf("Usage: analyze3 superfluid_wf normalfluid_wf electron_wf\n");
    exit(1);
  }
   
  /* Setup grid driver parameters */
  dft_driver_setup_grid_2d(NZ, NR, STEP /* Bohr */, THREADS /* threads */);
  dft_driver_initialize_2d();

  density = dft_driver_alloc_rgrid_2d();
  vz = dft_driver_alloc_rgrid_2d();
  vr = dft_driver_alloc_rgrid_2d();
  gwf = dft_driver_alloc_wavefunction_2d(HELIUM_MASS);
  nwf = dft_driver_alloc_wavefunction_2d(HELIUM_MASS);
  impwf = dft_driver_alloc_wavefunction_2d(IMP_MASS);
  
  printf("Compiled with NZ = %d, NR = %d, STEP = %le, VZ = %le\n", NZ, NR, STEP, VZ * GRID_AUTOMPS);
  if(argc == 4) {
    dft_driver_read_grid_2d(gwf->grid, argv[1]);      
    dft_driver_read_grid_2d(nwf->grid, argv[2]);      
    dft_driver_read_grid_2d(impwf->grid, argv[3]);      
  } else {
    printf("Usage: analyze3 superfluid_wf normalfluid_wf electron_wf\n");
    exit(1);
  }
  /* super */
  grid2d_wf_density(gwf, density);
  dft_driver_write_density_2d(density, "super");
  dft_driver_veloc_field_2d(gwf, vz, vr);
  rgrid2d_add(vz, -VZ);
  dft_driver_write_density_2d(vz, "super-vz");
  dft_driver_write_density_2d(vr, "super-vr");

  /* normal */
  grid2d_wf_density(nwf, density);
  dft_driver_write_density_2d(density, "normal");
  dft_driver_veloc_field_2d(nwf, vz, vr);
  rgrid2d_add(vz, -VZ);
  dft_driver_write_density_2d(vz, "normal-vz");
  dft_driver_write_density_2d(vr, "normal-vr");

  /* electron */
  grid2d_wf_density(impwf, density);
  dft_driver_write_density_2d(density, "electron");
  dft_driver_veloc_field_2d(impwf, vz, vr);
  dft_driver_write_density_2d(vz, "electron-vz");
  dft_driver_write_density_2d(vr, "electron-vr");

  return 0;
}
