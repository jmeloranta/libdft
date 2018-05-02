/*
 * Analyze wavefunction files from grid files on disk.
 *
 */

#include "bubble.h"

void analyze(wf *wf, INT iter, REAL vx) {

  static REAL cur_mom = 0.0, prev_mom = 0.0;
  rgrid *cur_x, *cur_y, *cur_z, *circ;
  extern REAL dpot_func(void *, REAL, REAL, REAL);

  printf("Current time = " FMT_R " fs.\n", ((REAL) iter) * TIME_STEP * GRID_AUTOFS);

  cur_x = dft_driver_get_workspace(1, 1);
  cur_y = dft_driver_get_workspace(2, 1);
  cur_z = dft_driver_get_workspace(3, 1);
  circ = dft_driver_get_workspace(4, 1);

  grid_wf_probability_flux(wf, cur_x, cur_y, cur_z);
  cur_mom = rgrid_integral(cur_x) * wf->mass;
    
  if(vx > 0.0) printf("Added mass = " FMT_R " He atoms.\n", cur_mom / (wf->mass * vx));

  grid_wf_density(wf, circ);
  printf("Number of He atoms = " FMT_R "\n", rgrid_integral(circ));
  if(iter > 0) {
    printf("Drag force1 = " FMT_R " (au).\n", -rgrid_weighted_integral(circ, dpot_func, NULL)); // circ = density here
    printf("Drag force2 = " FMT_R " (au).\n",  (cur_mom - prev_mom) / (TIME_STEP * ((REAL) OUTPUT_ITER) / GRID_AUTOFS));
  }
  prev_mom = cur_mom;

  rgrid_abs_rot(circ, cur_x, cur_y, cur_z);
  rgrid_power(circ, circ, NN);
  printf("Total circulation = " FMT_R " (au; NN = " FMT_R ").\n", rgrid_integral(circ), NN);
#ifdef USE_CUDA
  cuda_statistics(0);
#endif
  fflush(stdout);    
}  
