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

  rgrid3d *density, *vx, *vy, *vz;
  wf3d *gwf, *nwf, *impwf;
  
  /* Setup grid driver parameters */
  dft_driver_setup_grid(NX, NY, NZ, STEP /* Bohr */, THREADS /* threads */);
  dft_driver_initialize();

  density = dft_driver_alloc_rgrid();
  vx = dft_driver_alloc_rgrid();
  vy = dft_driver_alloc_rgrid();
  vz = dft_driver_alloc_rgrid();
  gwf = dft_driver_alloc_wavefunction(HELIUM_MASS);
  nwf = dft_driver_alloc_wavefunction(HELIUM_MASS);
  impwf = dft_driver_alloc_wavefunction(IMP_MASS);
  
  printf("Compiled with NX = %d, NY = %d, NZ = %d, STEP = %le, VX = %le\n", NX, NY, NZ, STEP, VX * GRID_AUTOMPS);
  if(argc == 4) {
    dft_driver_read_grid(gwf->grid, argv[1]);      
    dft_driver_read_grid(nwf->grid, argv[2]);      
    dft_driver_read_grid(impwf->grid, argv[3]);      
  } else {
    printf("Usage: analyze3 superfluid_wf normalfluid_wf electron_wf\n");
    exit(1);
  }
  /* super */
  grid3d_wf_density(gwf, density);
  dft_driver_write_density(density, "super");
  dft_driver_veloc_field(gwf, vx, vy, vz);
  rgrid3d_add(vx, -VX);
  dft_driver_write_density(vx, "super-vx");
  dft_driver_write_density(vy, "super-vy");
  dft_driver_write_density(vz, "super-vz");

  /* normal */
  grid3d_wf_density(nwf, density);
  dft_driver_write_density(density, "normal");
  dft_driver_veloc_field(nwf, vx, vy, vz);
  rgrid3d_add(vx, -VX);
  dft_driver_write_density(vx, "normal-vx");
  dft_driver_write_density(vy, "normal-vy");
  dft_driver_write_density(vz, "normal-vz");

  /* electron */
  grid3d_wf_density(impwf, density);
  dft_driver_write_density(density, "electron");
  dft_driver_veloc_field(impwf, vx, vy, vz);
  dft_driver_write_density(vx, "electron-vx");
  dft_driver_write_density(vy, "electron-vy");
  dft_driver_write_density(vz, "electron-vz");

  return 0;
}
