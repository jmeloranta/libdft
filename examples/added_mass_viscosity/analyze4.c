/*
 * Convert given WFs for electron and helium
 * to the corresponding densities and velocity fields.
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

#include "added_mass4.h"

int main(int argc, char **argv) {

  rgrid *density, *vx, *vy, *vz;
  wf *gwf, *impwf;
  
  /* Setup grid driver parameters */
  dft_driver_setup_grid(NX, NY, NZ, STEP /* Bohr */, THREADS /* threads */);

  gwf = dft_driver_alloc_wavefunction(HELIUM_MASS, "gwf");
  impwf = dft_driver_alloc_wavefunction(IMP_MASS, "impwf");

  dft_driver_initialize(gwf);

  density = dft_driver_alloc_rgrid("density");
  vx = dft_driver_alloc_rgrid("vx");
  vy = dft_driver_alloc_rgrid("vy");
  vz = dft_driver_alloc_rgrid("vz");
  
  printf("Compiled with NX = " FMT_I ", NY = " FMT_I ", NZ = " FMT_I ", "
    "STEP = " FMT_R ", VX = " FMT_R "\n", (INT) NX, (INT) NY, (INT) NZ, STEP, VX * GRID_AUTOMPS);

  if(argc == 3) {
    cgrid_read_grid(gwf->grid, argv[1]);
    cgrid_read_grid(impwf->grid, argv[2]);
  } else {
    printf("Usage: analyze4 helium_wf electron_wf\n");
    exit(1);
  }

  /* super */
  grid_wf_density(gwf, density);
  rgrid_write_grid("helium", density);
  grid_wf_velocity(gwf, vx, vy, vz, 200.0 / GRID_AUTOMPS);
  rgrid_add(vx, -VX);
  rgrid_write_grid("helium-vx", vx);
  rgrid_write_grid("helium-vy", vy);
  rgrid_write_grid("helium-vz", vz);

  /* electron */
  grid_wf_density(impwf, density);
  rgrid_write_grid("electron", density);
  grid_wf_velocity(impwf, vx, vy, vz, 200.0 / GRID_AUTOMPS);
  rgrid_write_grid("electron-vx", vx);
  rgrid_write_grid("electron-vy", vy);
  rgrid_write_grid("electron-vz", vz);

  return 0;
}
