/*
 * Stationary state of an ion travelling at
 * constant velocity in liquid helium. 
 * All input in a.u. except the time step, which is fs.
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

/* Only imaginary time */
#define TIME_STEP 10.0	/* Time step in fs (5 for real, 10 for imag) */
#define MAXITER 50000   /* Maximum number of iterations (was 300) */
#define OUTPUT     100	/* output every this iteration (was 1000) */
#define THREADS 0	/* # of parallel threads to use */
#define NX 1024      	/* # of grid points along x */
#define NY 512          /* # of grid points along y */
#define NZ 512      	/* # of grid points along z */
#define STEP 0.15       /* spatial step length (Bohr) */
#define PRESSURE 0.0    /* External pressure */

#define ALPHA 2.00 /**/
#define T1200MK

/* initial (file) guess from density or wf? */
/* #define INITIAL_GUESS_FROM_DENSITY */

/* velocity components */
#define KX	(1.0 * 2.0 * M_PI / (NX * STEP))
#define KY	(0.0 * 2.0 * M_PI / (NX * STEP))
#define KZ	(0.0 * 2.0 * M_PI / (NX * STEP))
#define VX	(KX * HBAR / DFT_HELIUM_MASS)
#define VY	(KY * HBAR / DFT_HELIUM_MASS)
#define VZ	(KZ * HBAR / DFT_HELIUM_MASS)
#define EKIN	(0.5 * DFT_HELIUM_MASS * (VX * VX + VY * VY + VZ * VZ))

#ifdef T2100MK
/* Exp mobility = 0.0492 cm^2/Vs - gives 0.096 (well conv. kc+bf 0.087) */
#define DENSITY (0.021954 * 0.529 * 0.529 * 0.529)     /* bulk liquid density */
#define VISCOSITY (1.803E-6) /* In Pa s */
#define RHON    0.741       /* normal fraction (0.752) */
#define FUNCTIONAL DFT_OT_T2100MK
#define TEMP 2.1
#endif

#ifdef T2000MK
/* Exp mobility = 0.06862 cm^2/Vs (Donnelly) */
#define DENSITY (0.021909 * 0.529 * 0.529 * 0.529)     /* bulk liquid density */
#define VISCOSITY (1.47E-6) /* In Pa s */
#define RHON    (0.553)      /* normal fraction */
#define FUNCTIONAL DFT_OT_T2000MK
#define TEMP 2.0
#endif

#ifdef T1800MK
/* Exp mobility = 0.097 cm^2/Vs - gives 0.287 */
#define DENSITY (0.021885 * 0.529 * 0.529 * 0.529)     /* bulk liquid density */
#define VISCOSITY (1.298E-6) /* In Pa s */
#define RHON    0.313       /* normal fraction */
#define FUNCTIONAL DFT_OT_T1800MK
#define TEMP 1.8
#endif
  
#ifdef T1600MK
/* Exp mobiolity = 0.183 cm^2/Vs - gives 0.565 */
#define DENSITY (0.021845 * 0.529 * 0.529 * 0.529)     /* bulk liquid density */
#define VISCOSITY (1.306E-6) /* In Pa s */
#define RHON    0.162       /* normal fraction */
#define FUNCTIONAL DFT_OT_T1600MK
#define TEMP 1.6
#endif

#ifdef T1400MK
/* Exp mobility = 0.3636 cm^2/Vs (Donnelly) */
#define DENSITY (0.021837 * 0.529 * 0.529 * 0.529)     /* bulk liquid density */
#define VISCOSITY (1.40E-6) /* In Pa s */
#define RHON    0.0745       /* normal fraction */
#define FUNCTIONAL DFT_OT_T1400MK
#define TEMP 1.4
#endif

#ifdef T1200MK
/* Exp mobility = 1.0 cm^2/Vs - gives 2.45 */
#define DENSITY (0.021846 * 0.529 * 0.529 * 0.529)     /* bulk liquid density */
#define VISCOSITY (1.736E-6) /* In Pa s */
#define RHON    0.026       /* normal fraction */
#define FUNCTIONAL DFT_OT_T1200MK
#define TEMP 1.2
#endif

#ifdef T800MK
/* Exp mobility = 20.86 cm^2/Vs - gives 8.17 (512/0.1 grid gives maybe slightly higher */
#define DENSITY (0.021876 * 0.529 * 0.529 * 0.529)     /* bulk liquid density */
#define VISCOSITY (2E-6) /* In Pa s (was 15.82E-6) */
#define RHON    9.27E-4       /* normal fraction (Donnelly 0.001, nist 0.0025) */
#define FUNCTIONAL DFT_OT_T800MK
#endif

#ifdef T400MK
/* Exp mobility = 438 cm^2/Vs - gives 395 */
#define DENSITY (0.021845 * 0.529 * 0.529 * 0.529)     /* bulk liquid density */
#define VISCOSITY (114.15E-6) /* In Pa s */
#define RHON    2.92E-6       /* normal fraction */
#define FUNCTIONAL DFT_OT_T400MK
#endif

#define SBC     4.0         /* boundary condition: 4 = electron, 6 = + ion (for Stokes) */
  
/* Ion */
// #define EXP_P
#define K_P

#ifdef ZERO_P
#define A0 0.0
#define A1 0.0
#define A2 0.0
#define A3 0.0
#define A4 0.0
#define A5 0.0
#define RMIN 0.0
#define RADD 0.0
#endif

/* exponential repulsion (approx. electron bubble) - RADD = -19.0 */
#ifdef EXP_P
#define A0 (3.8003E5 / GRID_AUTOK)
#define A1 (1.6245 * GRID_AUTOANG)
#define A2 0.0
#define A3 0.0
#define A4 0.0
#define A5 0.0
#define RMIN 2.0
#define RADD (-6.0)
#endif

/* Ca+ */
#ifdef CA_P
#define A0 4.83692
#define A1 1.23684
#define A2 0.273202
#define A3 59.5463
#define A4 1134.51
#define A5 0.0
#define RMIN 5.0
#define RADD 0.0
#endif

/* K+ */
#ifdef K_P
#define A0 140.757
#define A1 2.26202
#define A2 0.722065
#define A3 0.00144039
#define A4 356.303
#define A5 1358.98
#define RMIN 4.0
#define RADD 0.0
#endif

/* Be+ */
#ifdef BE_P
#define A0 4.73292
#define A1 1.53925
#define A2 0.557845
#define A3 26.7013
#define A4 0.0
#define A5 0.0
#define RMIN 3.4
#define RADD 0.0
#endif

/* Sr+ */
#ifdef SR_P
#define A0 3.64975
#define A1 1.13451
#define A2 0.293483
#define A3 99.0206
#define A4 693.904
#define A5 0.0
#define RMIN 5.0
#define RADD 0.0
#endif

/* Cl- */
#ifdef CL_M
#define A0 11.1909
#define A1 1.50971
#define A2 0.72186
#define A3 17.2434
#define A4 0.0
#define A5 0.0
#define RMIN 4.2
#define RADD 0.0
#endif

/* F- */
#ifdef F_M
#define A0 5.16101
#define A1 1.62798
#define A2 0.773982
#define A3 1.09722
#define A4 0.0
#define A5 0.0
#define RMIN 4.1
#define RADD 0.0
#endif

/* I- */
#ifdef I_M
#define A0 13.6874
#define A1 1.38037
#define A2 0.696409
#define A3 37.3331 
#define A4 0.0
#define A5 0.0
#define RMIN 4.1
#define RADD 0.0
#endif

/* Br- */
#ifdef BR_M
#define A0 12.5686
#define A1 1.45686
#define A2 0.714525
#define A3 24.114
#define A4 0.0
#define A5 0.0
#define RMIN 5.0
#define RADD 0.0
#endif

/* Impurity must always be at the origin (dU/dx) */
REAL dpot_func(void *NA, REAL x, REAL y, REAL z) {

  REAL rp, r2, r3, r5, r7, r9, r11, r;
  rp = SQRT(x * x + y * y + z * z);
  if(rp < RMIN) return 0.0;
  r = rp + RADD;
  r2 = r * r;
  r3 = r2 * r;
  r5 = r2 * r3;
  r7 = r5 * r2;
  r9 = r7 * r2;
  r11 = r9 * r2;
  
  return (x / rp) * (-A0 * A1 * EXP(-A1 * r) + 4.0 * A2 / r5 + 6.0 * A3 / r7 + 8.0 * A4 / r9 + 10.0 * A5 / r11);
}

REAL pot_func(void *asd, REAL x, REAL y, REAL z) {

  REAL r, r2, r4, r6, r8, r10, tmp, *asdf;

  if(asd) {
    asdf = asd;
    x -= *asdf;
  }
  r = SQRT(x * x + y * y + z * z);
  if(r < RMIN) r = RMIN;
  r += RADD;

  r2 = r * r;
  r4 = r2 * r2;
  r6 = r4 * r2;
  r8 = r6 * r2;
  r10 = r8 * r2;
  tmp = A0 * EXP(-A1 * r) - A2 / r4 - A3 / r6 - A4 / r8 - A5 / r10;
  return tmp;
}

REAL x_func(void *asd, REAL x, REAL y, REAL z) {

  if(FABS(x) < 0.01) return 0.0;
  return x / FABS(x);
}

REAL global_time;

int main(int argc, char *argv[]) {

  dft_ot_functional *otf;
  wf *gwf, *gwfp;
  cgrid *cworkspace;
  rgrid *ext_pot, *rworkspace;
  INT iter;
  char filename[2048];
  REAL kin, pot;
  REAL rho0, mu0, n;
  grid_timer timer;

#ifdef USE_CUDA
  cuda_enable(1);
#endif

  if(argc != 1 && argc != 2) {
    printf("Usage: added_mass2 <helium_wf>\n");
    exit(1);
  }
  
  printf("RADD = " FMT_R "\n", RADD);

  /* Initialize threads & use wisdom */
  grid_set_fftw_flags(1);    // FFTW_MEASURE
  grid_threads_init(THREADS);
  grid_fft_read_wisdom(NULL);

  /* Allocate wave functions */
  if(!(gwf = grid_wf_alloc(NX, NY, NZ, STEP, DFT_HELIUM_MASS, WF_PERIODIC_BOUNDARY, WF_2ND_ORDER_FFT, "gwf"))) {
    fprintf(stderr, "Cannot allocate gwf.\n");
    exit(1);
  }
  gwfp = grid_wf_clone(gwf, "gwfp");

  /* Reference frame for gwf & gwfp */
  cgrid_set_momentum(gwf->grid, KX, KY, KZ);
  cgrid_set_momentum(gwfp->grid, KX, KY, KZ);

  /* Allocate OT functional */
  if(!(otf = dft_ot_alloc(DFT_OT_PLAIN | DFT_OT_BACKFLOW | DFT_OT_KC | DFT_OT_HD, gwf, DFT_MIN_SUBSTEPS, DFT_MAX_SUBSTEPS))) {
    fprintf(stderr, "Cannot allocate otf.\n");
    exit(1);
  }
  rho0 = dft_ot_bulk_density_pressurized(otf, PRESSURE);
  mu0 = dft_ot_bulk_chempot_pressurized(otf, PRESSURE);
  printf("mu0 = " FMT_R " K/atom, rho0 = " FMT_R " Angs^-3.\n", mu0 * GRID_AUTOK, rho0 / (GRID_AUTOANG * GRID_AUTOANG * GRID_AUTOANG));

#ifdef ALPHA  
  printf("Using preset alpha.\n");
#define EFF_VISCOSITY (VISCOSITY * RHON)
#define EFF_ALPHA ALPHA
#else
  printf("Using precomputed alpha. with T = " FMT_R "\n", TEMP);
#define EFF_VISCOSITY (VISCOSITY * RHON)
#define EFF_ALPHA (1.73 + 2.32E-10*EXP(11.15*TEMP))
#endif
  
  /* Allocate grids */
  cworkspace = cgrid_clone(gwf->grid, "cworkspace");            /* allocate complex workspace */
  ext_pot = rgrid_clone(otf->density, "ext pot"); /* allocate real external potential grid */
  rworkspace = rgrid_clone(otf->density, "density"); /* allocate real density grid */
  
  fprintf(stderr, "Time step in a.u. = " FMT_R "\n", TIME_STEP / GRID_AUTOFS);
  fprintf(stderr, "Relative velocity = (" FMT_R ", " FMT_R ", " FMT_R ") (m/s)\n", 
	  VX * GRID_AUTOMPS, VY * GRID_AUTOMPS, VZ * GRID_AUTOMPS);
  
  if(argc < 2) {
    printf("Standard initial guess.\n");
    /* Initial wavefunctions. Read from file or set to initial guess */
    /* Constant density (initial guess) */
    if(rho0 == 0.0) rho0 = DENSITY;  /* for GP testing */    
    cgrid_constant(gwf->grid, SQRT(rho0));
  } else if (argc == 2) {   /* restarting */
    printf("Initial guess read from a file.\n");
#ifndef INITIAL_GUESS_FROM_DENSITY
    printf("Helium WF from %s.\n", argv[1]);
    cgrid_read_grid(gwf->grid, argv[1]);      
    cgrid_multiply(gwf->grid, SQRT(rho0) / gwf->grid->value[0]);
#else
    printf("Helium DENSITY from %s.\n", argv[1]);
    rgrid_read_grid(otf->density, argv[1]);
    rgrid_power(otf->density, otf->density, 0.5);
    grid_real_to_complex_re(gwf->grid, otf->density);
#endif
  }
  
  /* Read pair potential from file and do FFT */
#if 1
  rgrid_map(ext_pot, pot_func, NULL);
#else
  dft_common_potential_map(DFT_DRIVER_AVERAGE_XYZ, "pot.dat", "pot.dat", "pot.dat", ext_pot);
#endif
  
  for(iter = 1; iter < MAXITER; iter++) { /* start from 1 to avoid automatic wf initialization to a constant value */

    if(iter == 5) grid_fft_write_wisdom(NULL);

    grid_timer_start(&timer);

    /* Predict-Correct */
    grid_real_to_complex_re(cworkspace, ext_pot);
    dft_ot_potential(otf, cworkspace, gwf);
    dft_viscous_potential(gwf, otf, cworkspace, EFF_VISCOSITY, EFF_ALPHA);
    cgrid_add(cworkspace, -mu0);
    grid_wf_propagate_predict(gwf, gwfp, cworkspace, -I * TIME_STEP / GRID_AUTOFS);
    grid_add_real_to_complex_re(cworkspace, ext_pot);
    dft_ot_potential(otf, cworkspace, gwfp);
    dft_viscous_potential(gwfp, otf, cworkspace, EFF_VISCOSITY, EFF_ALPHA);
    cgrid_add(cworkspace, -mu0);
    cgrid_multiply(cworkspace, 0.5);  // Use (current + future) / 2
    grid_wf_propagate_correct(gwf, cworkspace, -I * TIME_STEP / GRID_AUTOFS);
    // Chemical potential included - no need to normalize

    printf("Iteration " FMT_I " - Wall clock time = " FMT_R " seconds.\n", iter, grid_timer_wall_clock_time(&timer));
    
    if(iter && !(iter % OUTPUT)) {   /* every OUTPUT iterations, write output */
      REAL force, mobility;
      /* Helium energy */
      kin = grid_wf_energy(gwf, NULL);            /* Kinetic energy for gwf */
      dft_ot_energy_density(otf, rworkspace, gwf);
      rgrid_add_scaled_product(rworkspace, 1.0, otf->density, ext_pot);
      pot = rgrid_integral(rworkspace);           /* Potential energy for gwf */
      //ene = kin + pot;           /* Total energy for gwf */
      n = grid_wf_norm(gwf);
      printf("Iteration " FMT_I ":\n", iter);
      printf("Background kinetic = " FMT_R "\n", n * EKIN * GRID_AUTOK);
      printf("Helium natoms    = " FMT_R " particles.\n", n);   /* Energy / particle in K */
      printf("Helium kinetic   = " FMT_R "\n", kin * GRID_AUTOK);  /* Print result in K */
      printf("Helium potential = " FMT_R "\n", pot * GRID_AUTOK);  /* Print result in K */
      printf("Helium energy    = " FMT_R "\n", (kin + pot) * GRID_AUTOK);  /* Print result in K */

      grid_wf_probability_flux_x(gwf, rworkspace);
      printf("Added mass = " FMT_R "\n", rgrid_integral(rworkspace) / VX); 

      grid_wf_density(gwf, otf->density);                     /* Density from gwf */
      force = rgrid_weighted_integral(otf->density, dpot_func, NULL);   /* includes the minus already somehow (cmp FD below) */

      printf("Drag force on ion = " FMT_R " a.u.\n", force);

      printf("E-field = " FMT_R " V/m\n", -force * GRID_AUTOVPM);
      mobility = VX * GRID_AUTOMPS / (-force * GRID_AUTOVPM);
      printf("Mobility = " FMT_R " [cm^2/(Vs)]\n", 1.0E4 * mobility); /* 1E4 = m^2 to cm^2 */
      printf("Hydrodynamic radius (Stokes) = " FMT_R " Angs.\n", 1E10 * 1.602176565E-19 / (SBC * M_PI * mobility * RHON * VISCOSITY));

      sprintf(filename, "wf-" FMT_I, iter);      
      cgrid_write_grid(filename, gwf->grid);
      
      fflush(stdout);
    }
  }
  return 0;
}

