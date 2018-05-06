/*
 * Analyze wavefunction files from grid files on disk.
 *
 */

#include "bubble.h"

void analyze(wf *wf, INT iter, REAL vx) {

  static REAL cur_mom_x = 0.0, cur_mom_y = 0.0, cur_mom_z = 0.0;
  static REAL prev_mom_x = 0.0, prev_mom_y = 0.0, prev_mom_z = 0.0;
  rgrid *cur_x, *cur_y, *cur_z, *circ;
  extern REAL dpot_func_x(void *, REAL, REAL, REAL);
  extern REAL dpot_func_y(void *, REAL, REAL, REAL);
  extern REAL dpot_func_z(void *, REAL, REAL, REAL);

  printf("Current time = " FMT_R " fs.\n", ((REAL) iter) * TIME_STEP * GRID_AUTOFS);

  cur_x = dft_driver_get_workspace(1, 1);
  cur_y = dft_driver_get_workspace(2, 1);
  cur_z = dft_driver_get_workspace(3, 1);
  circ = dft_driver_get_workspace(4, 1);

  grid_wf_probability_flux(wf, cur_x, cur_y, cur_z);
  cur_mom_x = rgrid_integral(cur_x) * wf->mass;
  cur_mom_y = rgrid_integral(cur_y) * wf->mass;
  cur_mom_z = rgrid_integral(cur_z) * wf->mass;
    
  if(vx > 0.0) printf("Added mass = " FMT_R " He atoms.\n", cur_mom_x / (wf->mass * vx));

  grid_wf_density(wf, circ);
  printf("Number of He atoms = " FMT_R "\n", rgrid_integral(circ));
  if(iter > 0) {
    printf("Drag force1_x = " FMT_R " (au).\n", -rgrid_weighted_integral(circ, dpot_func_x, NULL)); // circ = density here
    printf("Drag force1_y = " FMT_R " (au).\n", -rgrid_weighted_integral(circ, dpot_func_y, NULL)); // circ = density here
    printf("Drag force1_z = " FMT_R " (au).\n", -rgrid_weighted_integral(circ, dpot_func_z, NULL)); // circ = density here
    printf("Drag force2_x = " FMT_R " (au).\n",  (cur_mom_x - prev_mom_x) / (TIME_STEP * ((REAL) OUTPUT_ITER) / GRID_AUTOFS));
    printf("Drag force2_y = " FMT_R " (au).\n",  (cur_mom_y - prev_mom_y) / (TIME_STEP * ((REAL) OUTPUT_ITER) / GRID_AUTOFS));
    printf("Drag force2_z = " FMT_R " (au).\n",  (cur_mom_z - prev_mom_z) / (TIME_STEP * ((REAL) OUTPUT_ITER) / GRID_AUTOFS));
  }
  prev_mom_x = cur_mom_x;
  prev_mom_y = cur_mom_y;
  prev_mom_z = cur_mom_z;

  rgrid_abs_rot(circ, cur_x, cur_y, cur_z);
  rgrid_power(circ, circ, NN);
  printf("Total circulation = " FMT_R " (au; NN = " FMT_R ").\n", rgrid_integral(circ), NN);
#ifdef USE_CUDA
  cuda_statistics(0);
#endif
  fflush(stdout);    
}  
