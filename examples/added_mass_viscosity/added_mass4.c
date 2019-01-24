/*
 * Stationary state of an electron bubble travelling at
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

#include "added_mass4.h"

#define PRESSURE 0.0 /* External pressure */

#ifdef T2100MK
/* Exp mobility = 0.0492 cm^2/Vs (Donnelly 0.05052) */
#define T 2.1
#define DENSITY (0.021954 * 0.529 * 0.529 * 0.529)     /* bulk liquid density */
#ifdef DONNELLY
#define IDENT "T = 2.1 K (Donnelly)"
#define VISCOSITY (1.803E-6) /* In Pa s */
#define RHON    (0.741)      /* normal fraction */
#define EXP_MOBILITY 0.05052
#endif
#ifdef FRED
#define IDENT "T = 2.1 K (Fred)"
#define VISCOSITY (1.719E-6) /* In Pa s */
#define RHON    (0.752)      /* normal fraction */
#define EXP_MOBILITY 0.0492
#endif
#define FUNCTIONAL DFT_OT_T2100MK
#endif

#ifdef T2000MK
/* Exp mobility = 0.06862 cm^2/Vs (Donnelly) */
#define T 2.0
#define DENSITY (0.021909 * 0.529 * 0.529 * 0.529)     /* bulk liquid density */
#ifdef DONNELLY
#define IDENT "T = 2.0 K (Donnelly)"
#define VISCOSITY (1.47E-6) /* In Pa s */
#define RHON    (0.553)      /* normal fraction */
#define EXP_MOBILITY 0.06862
#endif
#ifdef FRED
#define IDENT "T = 2.0 K (Fred)"
#define VISCOSITY (1.406E-6) /* In Pa s */
#define RHON    (0.566)      /* normal fraction */
#define EXP_MOBILITY TODO
#endif
#define FUNCTIONAL DFT_OT_T2000MK
#endif

#ifdef T1800MK
/* Exp mobility = 0.097 cm^2/Vs (Donnelly 0.1088) */
#define T 1.8
#define DENSITY (0.021885 * 0.529 * 0.529 * 0.529)     /* bulk liquid density */
#ifdef DONNELLY
#define IDENT "T = 1.8 K (Donnelly)"
#define VISCOSITY (1.298E-6)  /* In Pa s */
#define RHON    (0.313)       /* normal fraction (Fred 0.35)) */
#define EXP_MOBILITY 0.1088
#endif
#ifdef FRED
#define IDENT "T = 1.8 K (Fred)"
#define VISCOSITY (1.25E-6)  /* In Pa s */
#define RHON    (0.35)       /* normal fraction */
#define EXP_MOBILITY 0.097
#endif
#define FUNCTIONAL DFT_OT_T1800MK
#endif
  
#ifdef T1600MK
/* Exp mobility = 0.183 cm^2/Vs (Donnelly 0.1772) */
#define T 1.6
#define DENSITY (0.021845 * 0.529 * 0.529 * 0.529)     /* bulk liquid density */
#ifdef DONNELLY
#define IDENT "T = 1.6 K (Donnelly)"
#define VISCOSITY (1.306E-6) /* In Pa s */
#define RHON    0.162       /* normal fraction */
#define EXP_MOBILITY 0.1772
#endif
#ifdef FRED
#define IDENT "T = 1.6 K (Fred)"
#define VISCOSITY (1.310E-6) /* In Pa s */
#define RHON    0.171       /* normal fraction */
#define EXP_MOBILITY 0.183
#endif
#define FUNCTIONAL DFT_OT_T1600MK
#endif

#ifdef T1400MK
/* Exp mobility = 0.3636 cm^2/Vs (Donnelly) */
#define T 1.4
#define DENSITY (0.021837 * 0.529 * 0.529 * 0.529)     /* bulk liquid density */
#ifdef DONNELLY
#define IDENT "T = 1.4 K (Donnelly)"
#define VISCOSITY (1.40E-6) /* In Pa s */
#define RHON    0.0745       /* normal fraction */
#define EXP_MOBILITY 0.3636
#endif
#ifdef FRED
#define IDENT "T = 1.4 K (Fred)"
#define VISCOSITY (1.406E-6) /* In Pa s */
#define RHON    0.0787       /* normal fraction */
#define EXP_MOBILITY TODO
#endif
#define FUNCTIONAL DFT_OT_T1400MK
#endif

#ifdef T1200MK
/* Exp mobility = 1.0 cm^2/Vs (Donnelly 0.9880) */
#define T 1.2
#define DENSITY (0.021846 * 0.529 * 0.529 * 0.529)     /* bulk liquid density */
#ifdef DONNELLY
#define IDENT "T = 1.2 K (Donnelly)"
#define VISCOSITY (1.736E-6) /* Pa s */
#define RHON    0.026       /* normal fraction */
#define EXP_MOBILITY 0.9880
#endif
#ifdef FRED
#define IDENT "T = 1.2 K (Fred)"
#define VISCOSITY (1.809E-6) /* Pa s */
#define RHON    0.0289      /* normal fraction */
#define EXP_MOBILITY 1.0
#endif
#define FUNCTIONAL DFT_OT_T1200MK
#endif

#ifdef T800MK
/* Exp mobility = 20.86 cm^2/Vs (Donnelly 21.38) */
#define T 0.8
#define DENSITY (0.021876 * 0.529 * 0.529 * 0.529)     /* bulk liquid density */
#ifdef DONNELLY
#define IDENT "T = 0.8 K (Donnelly)"
#define VISCOSITY (15.82E-6)  /* In Pa s */
#define RHON    9.27E-4       /* normal fraction */
#define EXP_MOBILITY 21.38
#endif
#ifdef FRED
#define IDENT "T = 0.8 K (Fred)"
#define VISCOSITY (15.823E-6) /* Pa s */
#define RHON    0.0025       /* normal fraction */
#define EXP_MOBILITY 20.86
#endif
#define FUNCTIONAL DFT_OT_T800MK
#endif

#define SBC     4.0         /* boundary condition: 4 = electron, 6 = + ion (for Stokes) */

REAL global_time;

REAL complex center_func(void *NA, REAL complex val, REAL x, REAL y, REAL z) {

  return x;   /* (x - x_0) but x_0 = 0 */
}

REAL center_func2(void *NA, REAL x, REAL y, REAL z) {

  return x;   /* (x - x_0) but x_0 = 0 */
}

REAL stddev_x(void *NA, REAL x, REAL y, REAL z) {

  return x * x;
}

REAL stddev_y(void *NA, REAL x, REAL y, REAL z) {

  return y * y;
}

/* Sign: - to + */
REAL eval_force(wf *gwf, wf *impwf, rgrid *pair_pot, rgrid *dpair_pot, rgrid *workspace1, rgrid *workspace2) {

  REAL tmp;

#if 1
  grid_wf_density(impwf, workspace1);
  rgrid_fft(workspace1);
  rgrid_fft_convolute(workspace2, pair_pot, workspace1);
  rgrid_inverse_fft(workspace2);
  rgrid_fd_gradient_x(workspace2, workspace1);
  grid_wf_density(gwf, workspace2);
  rgrid_product(workspace1, workspace1, workspace2);
  tmp = rgrid_integral(workspace1);   /* minus -> plus */
#else
  grid_wf_density(gwf, workspace1);
  rgrid_fft(workspace1);
  rgrid_fft_convolute(workspace2, dpair_pot, workspace1)
  rgrid_inverse_fft(workspace2);
  grid_wf_density(impwf, workspace1);
  rgrid_product(workspace1, workspace1, workspace2);
  tmp = -rgrid_integral(workspace1);
#endif

  return tmp;
}

int main(int argc, char *argv[]) {

  dft_ot_functional *otf;
  wf *gwf, *gwfp;
  wf *impwf, *impwfp; /* impurity wavefunction */
  cgrid *cpot_el, *cpot;
  rgrid *pair_pot, *dpair_pot, *ext_pot, *rworkspace;
  rgrid *vx;
  INT iter;
  char filename[2048];
  REAL kin, pot;
  REAL rho0, mu0, n, tmp;
  REAL force, mobility, last_mobility = 0.0;
  REAL inv_width = 0.05;
  grid_timer timer;

  if(argc != 1 && argc != 3) {
    printf("Usage: added_mass4 <helium_wf> <electron_wf>\n");
    exit(1);
  }
  
#ifdef USE_CUDA
  cuda_enable(1);
#endif

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
  impwf = grid_wf_clone(gwf, "impwf");
  impwf->mass = IMP_MASS;
  impwfp = grid_wf_clone(gwf, "impwfp");
  impwfp->mass = IMP_MASS;

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

  printf("Run ID: %s\n", IDENT);
  
  /* Allocate grids */
  cpot_el = cgrid_clone(gwf->grid, "cpot_el");
  cpot = cgrid_clone(gwf->grid, "cpot");
  pair_pot = rgrid_clone(otf->density, "pair_pot");              /* allocate real external potential grid */
  dpair_pot = rgrid_clone(otf->density, "dpair_pot");            /* allocate real external potential grid */
  ext_pot = rgrid_clone(otf->density, "ext_pot");                /* allocate real external potential grid */
  rworkspace = rgrid_clone(otf->density, "density");               /* allocate real density grid */
  vx = rgrid_clone(otf->density, "vx");                            /* allocate real density grid */
  
  fprintf(stderr, "Time step in a.u. = " FMT_R "\n", TIME_STEP / GRID_AUTOFS);
  fprintf(stderr, "Relative velocity = (" FMT_R ", " FMT_R ", " FMT_R ") (a.u.)\n", VX, VY, VZ);
  fprintf(stderr, "Relative velocity = (" FMT_R ", " FMT_R ", " FMT_R ") (m/s)\n", VX * GRID_AUTOMPS, VY * GRID_AUTOMPS, VZ * GRID_AUTOMPS);
  fprintf(stderr, "VISCOSITY = " FMT_R " Pa s; RHON/RHO0 = " FMT_R " .\n", VISCOSITY, RHON);
  
  if(argc < 2) {
    printf("Standard initial guess.\n");
    /* Initial wavefunctions. Read from file or set to initial guess */
    /* Constant density (initial guess) */
    if(rho0 == 0.0) rho0 = DENSITY;  /* for GP testing */    
    cgrid_constant(gwf->grid, SQRT(rho0));
    /* Gaussian for impurity (initial guess) */
    cgrid_map(impwf->grid, dft_common_cgaussian, &inv_width);
    cgrid_multiply(impwf->grid, 1.0 / SQRT(grid_wf_norm(impwf)));
  } else if (argc == 3) {   /* restarting */
    printf("Initial guess read from a file.\n");
    printf("Helium WF from %s.\n", argv[1]);
    cgrid_read_grid(gwf->grid, argv[1]);      
    cgrid_multiply(gwf->grid, SQRT(rho0) / gwf->grid->value[0]);
    printf("Electron WF from %s.\n", argv[2]);
    cgrid_read_grid(impwf->grid, argv[2]);      
    cgrid_multiply(impwf->grid, 1.0 / SQRT(grid_wf_norm(impwf)));
  } else {
    printf("Usage: added_mass4 <helium_wf electron_wf>\n");
    exit(1);
  }

#ifdef ALPHA
    printf("Using preset alpha.\n");
#define EFF_VISCOSITY (RHON * VISCOSITY)
#define EFF_ALPHA ALPHA
#else
    printf("Using precomputed alpha. with T = " FMT_R "\n", T);
#define EFF_VISCOSITY (RHON * VISCOSITY)
#define EFF_ALPHA (1.72 + 2.32E-10*EXP(11.15*T))
#endif
    
  /* Read pair potential from file and do FFT */
  dft_common_potential_map(4, PSPOT, PSPOT, PSPOT, pair_pot); // 4 = XYZ average
  rgrid_fd_gradient_x(pair_pot, dpair_pot);
  rgrid_fft(pair_pot);
  rgrid_fft(dpair_pot);
  
  for(iter = 1; iter < MAXITER; iter++) {

    if(iter == 5) grid_fft_write_wisdom(NULL);

    grid_timer_start(&timer);

    /* Predict */

    /* Electron */
    grid_wf_density(gwf, otf->density);
    rgrid_fft(otf->density);
    rgrid_fft_convolute(ext_pot, otf->density, pair_pot);
    rgrid_inverse_fft(ext_pot);
    cgrid_copy(impwfp->grid, impwf->grid);
    grid_real_to_complex_re(cpot_el, ext_pot);
    grid_wf_propagate_predict(impwfp, cpot_el, -I * IMP_STEP / GRID_AUTOFS);

    /* helium */
    cgrid_zero(cpot);
    dft_ot_potential(otf, cpot, gwf);
    cgrid_zero(cpot);
    grid_wf_density(impwf, otf->density);
    rgrid_fft(otf->density);
    rgrid_fft_convolute(ext_pot, otf->density, pair_pot);
    rgrid_inverse_fft(ext_pot);
    rgrid_add(ext_pot, -mu0); // chemical potential (same for super & normal)
    grid_add_real_to_complex_re(cpot, ext_pot);
    dft_ot_potential(otf, cpot, gwf);
    dft_viscous_potential(gwfp, otf, cpot, EFF_VISCOSITY, EFF_ALPHA);
    cgrid_copy(gwfp->grid, gwf->grid);
    grid_wf_propagate_predict(gwfp, cpot, -I * TIME_STEP / GRID_AUTOFS);

    /* Correct */

    /* Electron */
    grid_wf_density(gwfp, otf->density);
    rgrid_fft(otf->density);
    rgrid_fft_convolute(ext_pot, otf->density, pair_pot);
    rgrid_inverse_fft(ext_pot);
    grid_add_real_to_complex_re(cpot_el, ext_pot);
    cgrid_multiply(cpot_el, 0.5);
    grid_wf_propagate_correct(impwf, cpot_el, -I * IMP_STEP / GRID_AUTOFS);

    /* helium */
    dft_ot_potential(otf, cpot, gwfp);
    grid_wf_density(impwfp, otf->density);
    rgrid_fft(otf->density);
    rgrid_fft_convolute(ext_pot, otf->density, pair_pot);
    rgrid_inverse_fft(ext_pot);
    rgrid_add(ext_pot, -mu0); // chemical potential (same for super & normal)
    grid_add_real_to_complex_re(cpot, ext_pot);
    dft_ot_potential(otf, cpot, gwf);
    dft_viscous_potential(gwfp, otf, cpot, EFF_VISCOSITY, EFF_ALPHA);
    cgrid_multiply(cpot, 0.5);
    grid_wf_propagate_correct(gwf, cpot, -I * TIME_STEP / GRID_AUTOFS);

    printf("Iteration " FMT_I " - Wall clock time = " FMT_R " seconds.\n", iter, grid_timer_wall_clock_time(&timer));

    force = CREAL(cgrid_grid_expectation_value_func(NULL, center_func, impwf->grid)); // force is temp
    printf("Expectation value of position (electron): " FMT_R "\n", force * GRID_AUTOANG);

    /* keep electron at origin */
    cgrid_shift(cpot, impwf->grid, -force, 0.0, 0.0);  // cpot_super is temp
    cgrid_copy(impwf->grid, cpot);
    
    if(iter && !(iter % OUTPUT)){	
      printf("Iteration " FMT_I ":\n", iter);
      grid_wf_probability_flux_x(gwf, vx);
      tmp =  rgrid_integral(vx) / VX;
      printf("Added mass = " FMT_R "\n", tmp);

      force = eval_force(gwf, impwf, pair_pot, dpair_pot, ext_pot, otf->density);  /* ext_pot & density are temps */

      printf("Drag force on ion = " FMT_R " a.u.\n", force);
      printf("E-field = " FMT_R " V/m\n", -force * GRID_AUTOVPM);
      mobility = VX * GRID_AUTOMPS / (-force * GRID_AUTOVPM);
      printf("Mobility = " FMT_R " [cm^2/(Vs)]\n", 1.0E4 * mobility); /* 1E4 = m^2 to cm^2 */
      printf("Hydrodynamic radius (Stokes) = " FMT_R " Angs.\n", 1E10 * 1.602176565E-19 / (SBC * M_PI * mobility * RHON * VISCOSITY));
      printf("Mobility convergence = " FMT_R " %%.\n", 100.0 * FABS(mobility - last_mobility) / mobility);

      last_mobility = mobility;

      if(!(iter % (OUTPUT2 * OUTPUT))) {   /* 10XOUTPUT for writing files */
	/* Impurity density */
	grid_wf_density(impwf, otf->density);
	
	/* Helium energy */
        rgrid_fft(otf->density);
        rgrid_fft_convolute(ext_pot, otf->density, pair_pot);
        rgrid_inverse_fft(ext_pot);
	kin = grid_wf_energy(gwf, NULL);                 /* Kinetic energy for gwf */
        dft_ot_energy_density(otf, rworkspace, gwf);
        rgrid_add_scaled_product(rworkspace, 1.0, otf->density, ext_pot);
        pot = rgrid_integral(rworkspace);
	n = grid_wf_norm(gwf);

	printf("Background kinetic = " FMT_R "\n", n * EKIN * GRID_AUTOK);
	printf("Helium natoms    = " FMT_R " particles.\n", n);
	printf("Helium kinetic   = " FMT_R "\n", kin * GRID_AUTOK);
	printf("Helium potential = " FMT_R "\n", pot * GRID_AUTOK);
	printf("Helium energy    = " FMT_R "\n", (kin + pot) * GRID_AUTOK);

	grid_wf_density(impwf, otf->density);
	printf("Electron asymmetry (stddev x/y) = %le\n", rgrid_weighted_integral(otf->density, stddev_x, NULL) / rgrid_weighted_integral(otf->density, stddev_y, NULL));

	/* write out superfluid WF */
	sprintf(filename, "wf_helium-" FMT_I, iter);
	cgrid_write_grid(filename, gwf->grid);
	
	/* write out impurity WF */
	sprintf(filename, "wf_electron-" FMT_I, iter);
	cgrid_write_grid(filename, impwf->grid);
      }
    }
  }
  return 0;
}
