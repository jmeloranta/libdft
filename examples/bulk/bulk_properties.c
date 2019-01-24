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
#define NX 8          /* # of grid points along x (not relevant how many) */
#define NY 8          /* # of grid points along y (not relevant how many) */
#define NZ 8          /* # of grid points along z (not relevant how many) */
#define STEP 1.0        /* spatial step length (Bohr) */
#define DENSITY (0.0218360 * GRID_AUTOANG * GRID_AUTOANG * GRID_AUTOANG) /* bulk liquid density (0.0 = default at SVP) */

#define MIN_PRES	0.0
#define MAX_PRES	(24.0 / GRID_AUTOBAR)
#define DELTA_PRES	(0.1 / GRID_AUTOBAR)

#define PRESSURE 0.0

int main() {

  dft_ot_functional *otf;
  wf *gwf;
  REAL pressure, rho0, mu0;

  /*
   * This block of instructions is needed to initialize everything
   */
  
  /* Initialize threads & use wisdom */
  grid_set_fftw_flags(1);    // FFTW_MEASURE
  grid_threads_init(THREADS);
  grid_fft_read_wisdom(NULL);

  /* Allocate wave functions */
  if(!(gwf = grid_wf_alloc(NX, NY, NZ, STEP, DFT_HELIUM_MASS, WF_PERIODIC_BOUNDARY, WF_2ND_ORDER_FFT, "gwf"))) {
    fprintf(stderr, "Cannot allocate gwf.\n");
    exit(1);
  }

  /* Allocate OT functional */
  if(!(otf = dft_ot_alloc(DFT_OT_PLAIN | DFT_OT_BACKFLOW | DFT_OT_KC | DFT_OT_HD, gwf, DFT_MIN_SUBSTEPS, DFT_MAX_SUBSTEPS))) {
    fprintf(stderr, "Cannot allocate otf.\n");
    exit(1);
  }
  rho0 = dft_ot_bulk_density_pressurized(otf, PRESSURE);
  mu0 = dft_ot_bulk_chempot_pressurized(otf, PRESSURE);
  printf("mu0 = " FMT_R " K/atom, rho0 = " FMT_R " Angs^-3.\n", mu0 * GRID_AUTOK, rho0 / (GRID_AUTOANG * GRID_AUTOANG * GRID_AUTOANG));

  for(pressure = MIN_PRES; pressure <= MAX_PRES; pressure += DELTA_PRES) {
    rho0 = dft_ot_bulk_density_pressurized(otf, pressure);
    mu0 = dft_ot_bulk_chempot_pressurized(otf, pressure);
    printf("Pressure: " FMT_R " (bar)\t density: " FMT_R " (A**-3)\tchempot: " FMT_R " (K)\tSound (m/s): " FMT_R "\n",
	   pressure * GRID_AUTOBAR,
	   rho0 / (GRID_AUTOANG * GRID_AUTOANG * GRID_AUTOANG),
	   mu0 * GRID_AUTOK, dft_ot_bulk_sound_speed(otf, rho0) * GRID_AUTOMPS);
  }
  return 0;
}
