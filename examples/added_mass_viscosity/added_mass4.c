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

  wf *gwf, *gwfp;
  wf *impwf, *impwfp; /* impurity wavefunction */
  cgrid *cpot_el, *cpot;
  rgrid *pair_pot, *dpair_pot, *ext_pot, *density;
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
  
  /* Set fftw planning */
  grid_set_fftw_flags(PLANNING);
  
  /* Setup grid driver parameters */
  dft_driver_setup_grid(NX, NY, NZ, STEP /* Bohr */, THREADS /* threads */);
  /* Setup frame of reference momentum */
  dft_driver_setup_momentum(KX, KY, KZ);

  printf("Run ID: %s\n", IDENT);
  
  /* Plain Orsay-Trento in real or imaginary time */
  dft_driver_setup_model(FUNCTIONAL, 1, DENSITY);
#ifdef ALPHA
  printf("Using preset alpha.\n");
  dft_driver_setup_viscosity(RHON * VISCOSITY, ALPHA);
#else
  printf("Using precomputed alpha. with T = " FMT_R "\n", T);
  dft_driver_setup_viscosity(RHON * VISCOSITY, 1.72 + 2.32E-10*EXP(11.15*T));  
#endif
  
  /* Regular boundaries */
  dft_driver_setup_boundary_type(DFT_DRIVER_BOUNDARY_REGULAR, 0.0, 0.0, 0.0, 0.0);
  dft_driver_setup_boundary_condition(DFT_DRIVER_BC_NORMAL);
  
  /* Initialize */
  dft_driver_initialize();

  /* normalization */
  dft_driver_setup_normalization(DFT_DRIVER_DONT_NORMALIZE, 4, 0.0, 0);
  
  /* get bulk density and chemical potential */
  rho0 = dft_ot_bulk_density(dft_driver_otf);
  mu0  = dft_ot_bulk_chempot(dft_driver_otf);

  printf("Bulk: rho0 = " FMT_R " Angs^-3, mu0 = " FMT_R " K.\n", rho0 / (0.529 * 0.529 * 0.529), mu0 * GRID_AUTOK);
  printf("Exp: rho0 = " FMT_R " Angs^-3.\n", DENSITY / (0.529 * 0.529 * 0.529));
  
  /* Allocate wavefunctions and grids */
  cpot_el = dft_driver_alloc_cgrid("cpot_el");
  cpot = dft_driver_alloc_cgrid("cpot");
  pair_pot = dft_driver_alloc_rgrid("pair_pot");               /* allocate real external potential grid */
  dpair_pot = dft_driver_alloc_rgrid("dpair_pot");               /* allocate real external potential grid */
  ext_pot = dft_driver_alloc_rgrid("ext_pot");                /* allocate real external potential grid */
  density = dft_driver_alloc_rgrid("density");                /* allocate real density grid */
  vx = dft_driver_alloc_rgrid("vx");                /* allocate real density grid */
  impwf = dft_driver_alloc_wavefunction(IMP_MASS, "impwf");   /* impurity - order parameter for current time */
  impwf->norm  = 1.0;
  impwfp = dft_driver_alloc_wavefunction(IMP_MASS, "impwfp");  /* impurity - order parameter for future (predict) */
  impwfp->norm = 1.0;
  cgrid_set_momentum(impwf->grid, 0.0, 0.0, 0.0); /* Electron at rest */
  cgrid_set_momentum(impwfp->grid, 0.0, 0.0, 0.0);
  gwf = dft_driver_alloc_wavefunction(HELIUM_MASS, "gwf");  /* order parameter for current time */
  gwfp = dft_driver_alloc_wavefunction(HELIUM_MASS, "gwfp"); /* order parameter for future (predict) */
  
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
    dft_driver_read_grid(gwf->grid, argv[1]);      
    cgrid_multiply(gwf->grid, SQRT(rho0) / gwf->grid->value[0]);
    printf("Electron WF from %s.\n", argv[2]);
    dft_driver_read_grid(impwf->grid, argv[2]);      
    cgrid_multiply(impwf->grid, 1.0 / SQRT(grid_wf_norm(impwf)));
  } else {
    printf("Usage: added_mass4 <helium_wf electron_wf>\n");
    exit(1);
  }
    
  /* Read pair potential from file and do FFT */
  dft_common_potential_map(DFT_DRIVER_AVERAGE_XYZ, PSPOT, PSPOT, PSPOT, pair_pot);
  rgrid_fd_gradient_x(pair_pot, dpair_pot);
  rgrid_fft(pair_pot);
  rgrid_fft(dpair_pot);
  
  for(iter = 1; iter < MAXITER; iter++) {
    grid_timer_start(&timer);
    printf("Iteration " FMT_I " took ", iter);

    /* FIRST HALF OF KINETIC ENERGY */
    dft_driver_propagate_kinetic_first(DFT_DRIVER_PROPAGATE_OTHER, impwf, IMP_STEP);
    dft_driver_propagate_kinetic_first(DFT_DRIVER_PROPAGATE_HELIUM, gwf, TIME_STEP);

    /* PREDICT */

    /* electron external potential */
    grid_wf_density(gwf, density);
    rgrid_fft(density);
    rgrid_fft_convolute(ext_pot, density, pair_pot);
    rgrid_inverse_fft(ext_pot);
    cgrid_copy(impwfp->grid, impwf->grid);
    grid_real_to_complex_re(cpot_el, ext_pot);
    dft_driver_propagate_potential(DFT_DRIVER_PROPAGATE_OTHER, impwfp, cpot_el, IMP_STEP);

    /* helium external potential */
    cgrid_zero(cpot);
    dft_driver_ot_potential(gwf, cpot);
    dft_driver_viscous_potential(gwf, cpot);
    grid_wf_density(impwf, density);
    rgrid_fft(density);
    rgrid_fft_convolute(ext_pot, density, pair_pot);
    rgrid_inverse_fft(ext_pot);
    rgrid_add(ext_pot, -mu0); // chemical potential (same for super & normal)
    grid_add_real_to_complex_re(cpot, ext_pot);
    cgrid_copy(gwfp->grid, gwf->grid);
    dft_driver_propagate_potential(DFT_DRIVER_PROPAGATE_HELIUM, gwfp, cpot, TIME_STEP);

    /* CORRECT */

    /* electron external potential */
    grid_wf_density(gwfp, density);
    rgrid_fft(density);
    rgrid_fft_convolute(ext_pot, density, pair_pot);
    rgrid_inverse_fft(ext_pot);
    grid_add_real_to_complex_re(cpot_el, ext_pot);
    cgrid_multiply(cpot_el, 0.5);
    dft_driver_propagate_potential(DFT_DRIVER_PROPAGATE_OTHER, impwf, cpot_el, IMP_STEP);

    /* helium external potential */
    dft_driver_ot_potential(gwfp, cpot);
    dft_driver_viscous_potential(gwfp, cpot);
    grid_wf_density(impwfp, density);
    rgrid_fft(density);
    rgrid_fft_convolute(ext_pot, density, pair_pot);
    rgrid_inverse_fft(ext_pot);
    rgrid_add(ext_pot, -mu0);
    grid_add_real_to_complex_re(cpot, ext_pot);
    cgrid_multiply(cpot, 0.5);
    dft_driver_propagate_potential(DFT_DRIVER_PROPAGATE_HELIUM, gwf, cpot, TIME_STEP);
    
    /* SECOND HALF OF KINETIC */
    dft_driver_propagate_kinetic_second(DFT_DRIVER_PROPAGATE_OTHER, impwf, IMP_STEP);
    dft_driver_propagate_kinetic_second(DFT_DRIVER_PROPAGATE_HELIUM, gwf, TIME_STEP);

    printf(FMT_R " wall clock seconds.\n", grid_timer_wall_clock_time(&timer));
    fflush(stdout);

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

      force = eval_force(gwf, impwf, pair_pot, dpair_pot, ext_pot, density);  /* ext_pot & density are temps */

      printf("Drag force on ion = " FMT_R " a.u.\n", force);
      printf("E-field = " FMT_R " V/m\n", -force * GRID_AUTOVPM);
      mobility = VX * GRID_AUTOMPS / (-force * GRID_AUTOVPM);
      printf("Mobility = " FMT_R " [cm^2/(Vs)]\n", 1.0E4 * mobility); /* 1E4 = m^2 to cm^2 */
      printf("Hydrodynamic radius (Stokes) = " FMT_R " Angs.\n", 1E10 * 1.602176565E-19 / (SBC * M_PI * mobility * RHON * VISCOSITY));
      printf("Mobility convergence = " FMT_R " %%.\n", 100.0 * FABS(mobility - last_mobility) / mobility);

      last_mobility = mobility;

      if(!(iter % (OUTPUT2*OUTPUT))) {   /* 10XOUTPUT for writing files */
	/* Impurity density */
	grid_wf_density(impwf, density);
	
	/* Helium energy */
        rgrid_fft(density);
        rgrid_fft_convolute(ext_pot, density, pair_pot);
        rgrid_inverse_fft(ext_pot);
	kin = dft_driver_kinetic_energy(gwf);            /* Kinetic energy for gwf */
	pot = dft_driver_potential_energy(gwf, ext_pot); /* Potential energy for gwf */
	n = dft_driver_natoms(gwf);

	printf("Background kinetic = " FMT_R "\n", n * EKIN * GRID_AUTOK);
	printf("Helium natoms    = " FMT_R " particles.\n", n);
	printf("Helium kinetic   = " FMT_R "\n", kin * GRID_AUTOK);
	printf("Helium potential = " FMT_R "\n", pot * GRID_AUTOK);
	printf("Helium energy    = " FMT_R "\n", (kin + pot) * GRID_AUTOK);

	grid_wf_density(impwf, density);
	printf("Electron asymmetry (stddev x/y) = %le\n", rgrid_weighted_integral(density, stddev_x, NULL) / rgrid_weighted_integral(density, stddev_y, NULL));

	/* write out superfluid WF */
	sprintf(filename, "wf_helium-" FMT_I, iter);
	dft_driver_write_grid(gwf->grid, filename);
	
	/* write out impurity WF */
	sprintf(filename, "wf_electron-" FMT_I, iter);
	dft_driver_write_grid(impwf->grid, filename);
      }
    }
  }
  return 0;
}
