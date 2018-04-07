#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <complex.h>
#include <grid/grid.h>
#include <grid/au.h>
#include <dft/dft.h>
#include <dft/ot.h>

#define THREADS 1     /* # of parallel threads to use */
#define NX 8          /* # of grid points along x */
#define NY 8          /* # of grid points along y */
#define NZ 8         /* # of grid points along z */
#define STEP 1.0        /* spatial step length (Bohr) */
#define DENSITY (0.0218360 * 0.529 * 0.529 * 0.529)     /* bulk liquid density (0.0 = default at SVP) */

#define MIN_PRES	0.0
#define MAX_PRES	(24.0 / GRID_AUTOBAR )
#define DELTA_PRES	(0.1  / GRID_AUTOBAR )

int main() {

  REAL pressure, rho, mu;

  /*
   * This block of instructions is needed to initialize everything
   */
  
  /* Setup DFT driver parameters (256 x 256 x 256 grid) */
  dft_driver_setup_grid(NX, NY, NZ, STEP /* Bohr */, THREADS /* threads */);
  /* Plain Orsay-Trento in real or imaginary time */
  dft_driver_setup_model(DFT_OT_PLAIN, 1, DENSITY);   /* DFT_OT_PLAIN = Orsay-Trento without kinetic corr. or backflow, 1 = imag time */
  /* Regular boundaries */
  dft_driver_setup_boundary_type(DFT_DRIVER_BOUNDARY_REGULAR, 0.0, 0.0, 0.0);   /* regular periodic boundaries */
  dft_driver_setup_boundary_condition(DFT_DRIVER_BC_NEUMANN);
  /* Initialize */
  dft_driver_initialize();
  
  /*
   * DFT initialized. The functional is in dft_driver_otf
   */
  for(pressure = MIN_PRES; pressure <= MAX_PRES; pressure += DELTA_PRES) {
    rho = dft_ot_bulk_density_pressurized(dft_driver_otf, pressure);
    mu  = dft_ot_bulk_chempot_pressurized(dft_driver_otf, pressure);
    printf("Pressure:\t" FMT_R " (bar)\t density:\t" FMT_R " (A**-3)\tchempot:\t" FMT_R " (K)\n",
	   pressure * GRID_AUTOBAR,
	   rho / (GRID_AUTOANG * GRID_AUTOANG * GRID_AUTOANG),
	   mu * GRID_AUTOK);
  }
  return 0;
}
