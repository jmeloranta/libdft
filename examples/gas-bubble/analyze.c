/*
 * Analyze wavefunction files from grid files on disk.
 *
 * TODO: Use libgrid ext #7 functions for drag force1 (will run on GPU). Or maybe not needed....(?) This is not executed often...
 *
 */

#include "bubble.h"

extern grid_timer timer;

/* Output incompressible kinetic energy distribution in the k-space */
void do_ke(dft_ot_functional *otf, wf *gwf, INT iter) {

  static REAL *bins = NULL;
  FILE *fp;
  char file[256];
  INT i;

  return; // Not in use for now

#if 0
  if(!(otf->workspace1)) otf->workspace1 = rgrid_clone(otf->density, "OT Workspace 1");
  if(!(otf->workspace2)) otf->workspace2 = rgrid_clone(otf->density, "OT Workspace 2");
  if(!(otf->workspace3)) otf->workspace3 = rgrid_clone(otf->density, "OT Workspace 3");
  if(!(otf->workspace4)) otf->workspace4 = rgrid_clone(otf->density, "OT Workspace 4");
  if(!(otf->workspace5)) otf->workspace5 = rgrid_clone(otf->density, "OT Workspace 5");

  if(!bins) {
    if(!(bins = (REAL *) malloc(sizeof(REAL) * NBINS))) {
      fprintf(stderr, "Can't allocate memory for bins.\n");
      exit(1);
    }
  }
  /* Incompressible part */
  grid_wf_incomp_KE(gwf, bins, BINSTEP, NBINS, otf->workspace1, otf->workspace2, otf->workspace3, otf->workspace4, otf->workspace5);
  sprintf(file, "ke-incomp-" FMT_I ".dat", iter);
  if(!(fp = fopen(file, "w"))) {
    fprintf(stderr, "Can't open %s.\n", file);
    exit(1);
  }
  for(i = 1; i < NBINS; i++)  /* Leave out the DC component (= zero) */
    fprintf(fp, FMT_R " " FMT_R "\n", (BINSTEP * (REAL) i) / GRID_AUTOANG, bins[i] * GRID_AUTOK * GRID_AUTOANG); /* Angs^{-1} K*Angs */
  fclose(fp);
  /* Compressible part */
  grid_wf_comp_KE(gwf, bins, BINSTEP, NBINS, otf->workspace1, otf->workspace2, otf->workspace3, otf->workspace4);
  sprintf(file, "ke-comp-" FMT_I ".dat", iter);
  if(!(fp = fopen(file, "w"))) {
    fprintf(stderr, "Can't open %s.\n", file);
    exit(1);
  }
  for(i = 1; i < NBINS; i++)  /* Leave out the DC component (= zero) */
    fprintf(fp, FMT_R " " FMT_R "\n", (BINSTEP * (REAL) i) / GRID_AUTOANG, bins[i] * GRID_AUTOK * GRID_AUTOANG); /* Angs^{-1} K*Angs */
  fclose(fp);
#endif
}

void analyze(dft_ot_functional *otf, wf *wf, INT iter, REAL vz) {

  static REAL cur_mom_x = 0.0, cur_mom_y = 0.0, cur_mom_z = 0.0;
  static REAL prev_mom_x = 0.0, prev_mom_y = 0.0, prev_mom_z = 0.0;
  rgrid *cur_x, *cur_y, *cur_z, *circ;
  extern REAL dpot_func_x(void *, REAL, REAL, REAL);
  extern REAL dpot_func_y(void *, REAL, REAL, REAL);
  extern REAL dpot_func_z(void *, REAL, REAL, REAL);
  
  printf("Iteration " FMT_I " - Wall clock time = " FMT_R " seconds.\n", iter, grid_timer_wall_clock_time(&timer));
  printf("Iteration = " FMT_I ", Current time = " FMT_R " fs, velocity = " FMT_R ".\n", iter, ((REAL) iter) * TIME_STEP * GRID_AUTOFS, vz * GRID_AUTOMPS);
  fflush(stdout);

  if(!(otf->workspace1)) otf->workspace1 = rgrid_clone(otf->density, "OT Workspace 1");
  if(!(otf->workspace2)) otf->workspace2 = rgrid_clone(otf->density, "OT Workspace 2");
  if(!(otf->workspace3)) otf->workspace3 = rgrid_clone(otf->density, "OT Workspace 3");
  if(!(otf->workspace4)) otf->workspace4 = rgrid_clone(otf->density, "OT Workspace 4");
  cur_x = otf->workspace1;
  cur_y = otf->workspace2;
  cur_z = otf->workspace3;
  circ = otf->workspace4;

  grid_wf_probability_flux(wf, cur_x, cur_y, cur_z);
  cur_mom_x = rgrid_integral(cur_x) * wf->mass;
  cur_mom_y = rgrid_integral(cur_y) * wf->mass;
  cur_mom_z = rgrid_integral(cur_z) * wf->mass;
    
  if(vz > 0.0) printf("Added mass = " FMT_R " He atoms.\n", cur_mom_z / (wf->mass * vz));

  grid_wf_density(wf, circ);
  printf("Number of He atoms = " FMT_R "\n", rgrid_integral(circ));
  if(iter > 0) {
    printf("Drag force1_x = " FMT_R " (au).\n", -rgrid_weighted_integral(circ, dpot_func_x, NULL)); // circ = density here
    printf("Drag force1_y = " FMT_R " (au).\n", -rgrid_weighted_integral(circ, dpot_func_y, NULL)); // circ = density here
    printf("Drag force1_z = " FMT_R " (au).\n", -rgrid_weighted_integral(circ, dpot_func_z, NULL)); // circ = density here
    printf("Drag force2_x = " FMT_R " (au).\n",  (cur_mom_x - prev_mom_x) / (TIME_STEP * ((REAL) OUTPUT_ITER)));
    printf("Drag force2_y = " FMT_R " (au).\n",  (cur_mom_y - prev_mom_y) / (TIME_STEP * ((REAL) OUTPUT_ITER)));
    printf("Drag force2_z = " FMT_R " (au).\n",  (cur_mom_z - prev_mom_z) / (TIME_STEP * ((REAL) OUTPUT_ITER)));
  }
  prev_mom_x = cur_mom_x;
  prev_mom_y = cur_mom_y;
  prev_mom_z = cur_mom_z;

  rgrid_abs_rot(circ, cur_x, cur_y, cur_z);
  rgrid_abs_power(circ, circ, NN);

  printf("Total circulation = " FMT_R " (au; NN = " FMT_R ") at velocity " FMT_R ".\n", rgrid_integral(circ), NN, vz * GRID_AUTOMPS);
#ifdef USE_CUDA
  cuda_statistics(0);
#endif
  fflush(stdout);    

  grid_timer_start(&timer);
}  
