/*
 * Dynamics of a bubble (formed by external potential) travelling at
 * constant velocity in liquid helium (moving background). 
 * 
 */

#include "bubble.h"

extern void do_ke(dft_ot_functional *, wf *, INT);

grid_timer timer;

REAL round_veloc(REAL veloc) {   // Round to fit the simulation box

  INT n;
  REAL v;

  n = (INT) (0.5 + (NX * STEP * DFT_HELIUM_MASS * veloc) / (HBAR * 2.0 * M_PI));
  v = ((REAL) n) * 2.0 * HBAR * M_PI / (NX * STEP * DFT_HELIUM_MASS);
//  printf("Requested velocity = %le m/s.\n", veloc * GRID_AUTOMPS);
//  printf("Nearest velocity compatible with PBC = %le m/s.\n", v * GRID_AUTOMPS);
  return v;
}

REAL momentum(REAL vx) {

  return DFT_HELIUM_MASS * vx / HBAR;
}

int main(int argc, char *argv[]) {

  dft_ot_functional *otf;
  wf *gwf;
#ifdef PC
  wf *gwfp;
#endif
  cgrid *cworkspace;
  rgrid *ext_pot;
#ifdef OUTPUT_GRID
  char filename[2048];
#endif
  REAL vz = 0.0, mu0, kz, rho0;
  INT iter, sav_func;
  extern void analyze(dft_ot_functional *, wf *, INT, REAL);
  extern REAL pot_func(void *, REAL, REAL, REAL);
  
#ifdef USE_CUDA
  cuda_enable(1);
#endif

  /* Initialize threads & use wisdom */
  grid_set_fftw_flags(1);    // FFTW_MEASURE
  grid_threads_init(THREADS);
  grid_fft_read_wisdom(NULL);

  /* Allocate wave functions */
#if PROPAGATOR == WF_2ND_ORDER_CN
  if(!(gwf = grid_wf_alloc(NX, NY, NZ, STEP, DFT_HELIUM_MASS, WF_NEUMANN_BOUNDARY, PROPAGATOR, "gwf"))) { // works equally well, but is faster
//  if(!(gwf = grid_wf_alloc(NX, NY, NZ, STEP, DFT_HELIUM_MASS, WF_PERIODIC_BOUNDARY, PROPAGATOR, "gwf"))) { // slow
#else
  if(!(gwf = grid_wf_alloc(NX, NY, NZ, STEP, DFT_HELIUM_MASS, WF_PERIODIC_BOUNDARY, PROPAGATOR, "gwf"))) {
#endif
    fprintf(stderr, "Cannot allocate gwf.\n");
    exit(1);
  }
#ifdef PC
  gwfp = grid_wf_clone(gwf, "gwfp");
#endif
  cworkspace = cgrid_clone(gwf->grid, "cworkspace");

  /* Allocate OT functional */
  if(!(otf = dft_ot_alloc(FUNCTIONAL, gwf, DFT_MIN_SUBSTEPS, DFT_MAX_SUBSTEPS))) {
    fprintf(stderr, "Cannot allocate otf.\n");
    exit(1);
  }
  sav_func = otf->model;
  otf->model = DFT_OT_PLAIN;  // Obtain the initial structure using plain OT DFT

  rho0 = dft_ot_bulk_density_pressurized(otf, PRESSURE);
  mu0 = dft_ot_bulk_chempot_pressurized(otf, PRESSURE);
  printf("mu0 = " FMT_R " K/atom, rho0 = " FMT_R " Angs^-3.\n", mu0 * GRID_AUTOK, rho0 / (GRID_AUTOANG * GRID_AUTOANG * GRID_AUTOANG));
  // Velocity cutoff
  otf->veloc_cutoff = MAXVELOC;

  ext_pot = rgrid_clone(otf->density, "ext_pot");                /* allocate real external potential grid */

#ifdef RMIN
  printf("Potential: RMIN = " FMT_R ", RADD = " FMT_R ", A0 = " FMT_R ", A1 = " FMT_R ", A2 = " FMT_R ", A3 = " FMT_R ", A4 = " FMT_R ", A5 = " FMT_R "\n", RMIN, RADD, A0, A1, A2, A3, A4, A5);
#else
  printf("Potential: RMIN = NOT DEFINED, RADD = " FMT_R ", A0 = " FMT_R ", A1 = " FMT_R ", A2 = " FMT_R ", A3 = " FMT_R ", A4 = " FMT_R ", A5 = " FMT_R "\n", RADD, A0, A1, A2, A3, A4, A5);
#endif

  printf("Time step in fs   = " FMT_R "\n", TIME_STEP * GRID_AUTOFS);
  printf("Time step in a.u. = " FMT_R "\n", TIME_STEP);

//  rgrid_smooth_map(ext_pot, pot_func, NULL, 3); /* External potential */
  rgrid_map(ext_pot, pot_func, NULL); /* External potential */
//  rgrid_write_grid("pot", ext_pot); exit(0);
  rgrid_add(ext_pot, -mu0);

  vz = round_veloc(INIVZ);
  printf("Current velocity = " FMT_R " m/s.\n", vz * GRID_AUTOMPS);
  kz = momentum(vz);
  cgrid_set_momentum(gwf->grid, 0.0, 0.0, kz);
#ifdef PC
  cgrid_set_momentum(gwfp->grid, 0.0, 0.0, kz);
#endif

  if(argc == 1) {
    grid_wf_constant(gwf, SQRT(rho0));
    /* Mixed Imaginary & Real time iterations */
    printf("Warm up iterations.\n");

    for(iter = 0; iter < STARTING_ITER; iter++) {

      if(iter == 5) grid_fft_write_wisdom(NULL);

      grid_timer_start(&timer);

      /* Obtain the initial state by half real and half imag time propagation */
#ifdef PC
      /* Predict-Correct */
      grid_real_to_complex_re(cworkspace, ext_pot);
      dft_ot_potential(otf, cworkspace, gwf);
      grid_wf_propagate_predict(gwf, gwfp, cworkspace, - I * TIME_STEP);
      grid_add_real_to_complex_re(cworkspace, ext_pot);
      dft_ot_potential(otf, cworkspace, gwfp);
      cgrid_multiply(cworkspace, 0.5);  // Use (current + future) / 2
      grid_wf_propagate_correct(gwf, cworkspace, - I * TIME_STEP);
      // Chemical potential included - no need to normalize
#else
      grid_real_to_complex_re(cworkspace, ext_pot);
      dft_ot_potential(otf, cworkspace, gwf);
      grid_wf_propagate(gwf, cworkspace, - I * TIME_STEP);
#endif
      printf("Iteration " FMT_I " - Wall clock time = " FMT_R " seconds.\n", iter, grid_timer_wall_clock_time(&timer));
      fflush(stdout);
    }
    iter = 0;
  } else { /* restart from a file (.grd) */
    sscanf(argv[1], "bubble-" FMT_I ".grd", &iter);
    printf("Continuing from checkpoint file %s at iteration " FMT_I ".\n", argv[1], iter);
    cgrid_read_grid(gwf->grid, argv[1]);
  }

#ifdef ABS_AMP

#ifdef PC
  grid_wf_boundary(gwf, gwfp, ABS_AMP, rho0, (INT) (ABS_WIDTH_X / STEP), NX - (INT) (ABS_WIDTH_X / STEP),
                   (INT) (ABS_WIDTH_Y / STEP), NY - (INT) (ABS_WIDTH_Y / STEP), (INT) (ABS_WIDTH_Z / STEP), 
                   NZ - (INT) (ABS_WIDTH_Z / STEP));
#else
  grid_wf_boundary(gwf, NULL, ABS_AMP, rho0, (INT) (ABS_WIDTH_X / STEP), NX - (INT) (ABS_WIDTH_X / STEP),
                   (INT) (ABS_WIDTH_Y / STEP), NY - (INT) (ABS_WIDTH_Y / STEP), (INT) (ABS_WIDTH_Z / STEP), 
                   NZ - (INT) (ABS_WIDTH_Z / STEP));
#endif

#endif

  /* Real time iterations */
  otf->model = sav_func;  // Restore the original requested functional for real time iterations

  printf("Real time propagation.\n");

  grid_timer_start(&timer);

  for( ; iter < MAXITER; iter++) {

#ifdef OUTPUT_GRID
    if(!(iter % OUTPUT_GRID)) {
      printf("Current velocity = " FMT_R " m/s.\n", vz * GRID_AUTOMPS);
      sprintf(filename, "liquid-" FMT_I, iter);
      cgrid_write_grid(filename, gwf->grid);
//      system("rm *.grd");
      do_ke(otf, gwf, iter);
      fflush(stdout);
    }
#endif
    if(!(iter % OUTPUT_ITER)) analyze(otf, gwf, iter, vz);
#ifdef PC
    /* Predict-Correct */
    grid_real_to_complex_re(cworkspace, ext_pot);
    dft_ot_potential(otf, cworkspace, gwf);
    grid_wf_propagate_predict(gwf, gwfp, cworkspace, TIME_STEP);

    grid_add_real_to_complex_re(cworkspace, ext_pot);
    dft_ot_potential(otf, cworkspace, gwfp);
    cgrid_multiply(cworkspace, 0.5);  // Use (current + future) / 2
    grid_wf_propagate_correct(gwf, cworkspace, TIME_STEP);
#else /* PC */
    grid_real_to_complex_re(cworkspace, ext_pot);
    dft_ot_potential(otf, cworkspace, gwf);
    grid_wf_propagate(gwf, cworkspace, TIME_STEP - I * TIME_STEP*FFT_STAB);
#endif /* PC */
  }

  return 0;
}
