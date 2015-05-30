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
#define TIME_STEP 50.0	/* Time step in fs (5 for real, 10 for imag) */
#define MAXITER 50000   /* Maximum number of iterations (was 300) */
#define OUTPUT     100	/* output every this iteration */
#define THREADS 32	/* # of parallel threads to use */
#define NX 256      	/* # of grid points along x */
#define NY 256          /* # of grid points along y */
#define NZ 256        	/* # of grid points along z */
#define STEP 1.0        /* spatial step length (Bohr) */
/* Note: density at 1.8 K */
#define DENSITY (0.0218735 * 0.529 * 0.529 * 0.529)     /* bulk liquid density (0.0 = default at SVP) */
#define HELIUM_MASS (4.002602 / GRID_AUTOAMU) /* helium mass */

/* velocity components */
#define KX	(1.0 * 2.0 * M_PI / (NX * STEP))
//#define KX	(0.0 * 2.0 * M_PI / (NX * STEP))
#define KY	(0.0 * 2.0 * M_PI / (NX * STEP))
#define KZ	(0.0 * 2.0 * M_PI / (NX * STEP))
#define VX	(KX * HBAR / HELIUM_MASS)
#define VY	(KY * HBAR / HELIUM_MASS)
#define VZ	(KZ * HBAR / HELIUM_MASS)
#define EKIN	(0.5 * HELIUM_MASS * (VX * VX + VY * VY + VZ * VZ))

#define VISCOSITY (1.26E-6) /* In Pa s */
#define RHON    0.286       /* normal fraction (0.286) */
#define SBC     4.0         /* boundary condition: 4 = electron, 6 = + ion */

/* Ion */
#define EXP_P

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

/* exponential repulsion (approx. electron bubble) */
#ifdef EXP_P
#define A0 (3.8003E5 / GRID_AUTOK)
#define A1 (1.6245 * GRID_AUTOANG)
#define A2 0.0
#define A3 0.0
#define A4 0.0
#define A5 0.0
#define RMIN 0.1
#define RADD (-19.0)
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

/* Impurity must always be at the origin */
double dpot_func(void *NA, double x, double y, double z) {

  double r = sqrt(x * x + y * y + z * z) + RADD;
  double r2 = r * r;
  double r3 = r2 * r;
  double r5 = r2 * r3;
  double r7 = r5 * r2;
  double r9 = r7 * r2;
  double r11 = r9 * r2;
  
  if(r < RMIN) return 0.0;   /* hopefully no liquid density in the core region */
  return (x / r) * (-A0 * A1 * exp(-A1 * r) + 4.0 * A2 / r5 + 6.0 * A3 / r7 + 8.0 * A4 / r9 + 10.0 * A5 / r11);
}

double pot_func(void *asd, double x, double y, double z) {

  double r = sqrt(x * x + y * y + z * z) + RADD;
  double r2, r4, r6, r8, r10, tmp;

  if(r < RMIN) r = RMIN;
  r2 = r * r;
  r4 = r2 * r2;
  r6 = r4 * r2;
  r8 = r6 * r2;
  r10 = r8 * r2;
  tmp = A0 * exp(-A1 * r) - A2 / r4 - A3 / r6 - A4 / r8 - A5 / r10;
  return tmp;
}

double global_time;

int main(int argc, char *argv[]) {

  wf3d *gwf, *gwfp;
  cgrid3d *cworkspace;
  rgrid3d *ext_pot, *ext_pot2, *density, *current, *temp, *vx, *vy, *vz;
  long iter;
  char filename[2048];
  double kin, pot;
  double rho0, mu0, n;

  /* Setup DFT driver parameters (256 x 256 x 256 grid) */
  dft_driver_setup_grid(NX, NY, NZ, STEP /* Bohr */, THREADS /* threads */);
  /* Setup frame of reference momentum */
  dft_driver_setup_momentum(KX, KY, KZ);

  /* Plain Orsay-Trento in real or imaginary time */
  dft_driver_setup_model(DFT_OT_T1800MK, 1, DENSITY);   /* DFT_OT_HD = Orsay-Trento with high-densiy corr. , 1 = imag time */

  /* Regular boundaries */
  dft_driver_setup_boundaries(DFT_DRIVER_BOUNDARY_REGULAR, 0.0);   /* regular periodic boundaries */
  dft_driver_setup_boundaries_damp(0.00);                          /* damping coeff., only needed for absorbing boundaries */
  dft_driver_setup_boundary_condition(DFT_DRIVER_BC_NORMAL);
  
  /* Initialize */
  dft_driver_initialize();

  /* bulk normalization */
  dft_driver_setup_normalization(DFT_DRIVER_DONT_NORMALIZE, 4, 0.0, 0);   /* Normalization: ZEROB = adjust grid point NX/4, NY/4, NZ/4 to bulk density after each imag. time iteration */
  
  /* get bulk density and chemical potential */
  rho0 = dft_ot_bulk_density(dft_driver_otf);
  mu0  = dft_ot_bulk_chempot(dft_driver_otf);
  printf("rho0 = %le Angs^-3, mu0 = %le K.\n", rho0 / (0.529 * 0.529 * 0.529), mu0 * GRID_AUTOK);

  /* Allocate wavefunctions and grids */
  gwf = dft_driver_alloc_wavefunction(HELIUM_MASS);  /* order parameter for current time */
  gwfp = dft_driver_alloc_wavefunction(HELIUM_MASS); /* order parameter for future (predict) */
  cworkspace = dft_driver_alloc_cgrid();             /* allocate complex workspace */
  ext_pot = dft_driver_alloc_rgrid();                /* allocate real external potential grid */
  ext_pot2 = dft_driver_alloc_rgrid();
  density = dft_driver_alloc_rgrid();                /* allocate real density grid */
  current = dft_driver_alloc_rgrid();                /* allocate real density grid */
  temp = dft_driver_alloc_rgrid();
  vx = dft_driver_alloc_rgrid();
  vy = dft_driver_alloc_rgrid();
  vz = dft_driver_alloc_rgrid();
  
  fprintf(stderr, "Time step in a.u. = %le\n", TIME_STEP / GRID_AUTOFS);
  fprintf(stderr, "Relative velocity = ( %le , %le ,%le ) (A/ps)\n", 
		  VX * 1000.0 * GRID_AUTOANG / GRID_AUTOFS,
		  VY * 1000.0 * GRID_AUTOANG / GRID_AUTOFS,
		  VZ * 1000.0 * GRID_AUTOANG / GRID_AUTOFS);

  /* Read pair potential from file and do FFT */
#if 1
  rgrid3d_map(ext_pot, pot_func, NULL);
#else
  dft_common_potential_map(DFT_DRIVER_AVERAGE_XYZ, "pot.dat", "pot.dat", "pot.dat", ext_pot);
#endif
  rgrid3d_add(ext_pot, -mu0) ; /* Add the chemical potential */

  for(iter = 0; iter < MAXITER; iter++) { /* start from 1 to avoid automatic wf initialization to a constant value */
    /*2. Predict + correct */
    /* Viscous response */
    rgrid3d_copy(ext_pot2, ext_pot);
    if(RHON != 0.0) {
      dft_driver_veloc_field(gwf, vx, vy, vz);
      rgrid3d_fd_gradient_x(vx, temp);
      rgrid3d_multiply(temp, -(VISCOSITY/GRID_AUTOPAS) * RHON / rho0);
      rgrid3d_sum(ext_pot2, ext_pot2, temp);
      rgrid3d_fd_gradient_y(vy, temp);
      rgrid3d_multiply(temp, -(VISCOSITY/GRID_AUTOPAS) * RHON / rho0);
      rgrid3d_sum(ext_pot2, ext_pot2, temp);
      rgrid3d_fd_gradient_z(vz, temp);
      rgrid3d_multiply(temp, -(VISCOSITY/GRID_AUTOPAS) * RHON / rho0);
      rgrid3d_sum(ext_pot2, ext_pot2, temp);
    }
    /* End viscous response */
    (void) dft_driver_propagate_predict(DFT_DRIVER_PROPAGATE_HELIUM, ext_pot2, gwf, gwfp, cworkspace, TIME_STEP, iter); /* PREDICT */ 
    (void) dft_driver_propagate_correct(DFT_DRIVER_PROPAGATE_HELIUM, ext_pot2, gwf, gwfp, cworkspace, TIME_STEP, iter); /* CORRECT */ 
    
    if(iter && !(iter % OUTPUT)) {   /* every OUTPUT iterations, write output */
      double force, mobility;
      /* Helium energy */
      kin = dft_driver_kinetic_energy(gwf);            /* Kinetic energy for gwf */
      pot = dft_driver_potential_energy(gwf, ext_pot); /* Potential energy for gwf */
      //ene = kin + pot;           /* Total energy for gwf */
      n = dft_driver_natoms(gwf) ;
      printf("Iteration %ld background kinetic = %.30lf\n", iter, n * EKIN * GRID_AUTOK);
      printf("Iteration %ld helium natoms    = %le particles.\n", iter, n);   /* Energy / particle in K */
      printf("Iteration %ld helium kinetic   = %.30lf\n", iter, kin * GRID_AUTOK);  /* Print result in K */
      printf("Iteration %ld helium potential = %.30lf\n", iter, pot * GRID_AUTOK);  /* Print result in K */
      printf("Iteration %ld helium energy    = %.30lf\n", iter, (kin + pot) * GRID_AUTOK);  /* Print result in K */
      
      /* Helium density */
      if(VX != 0.0)
	grid3d_wf_probability_flux_x(gwf, current);
      else if(VY != 0.0)
	grid3d_wf_probability_flux_y(gwf, current);
      else
	grid3d_wf_probability_flux_z(gwf, current);

      if(VX != 0.0)
	printf("Iteration %ld added mass = %.30lf\n", iter, rgrid3d_integral(current) / VX); 
      else
	printf("VX = 0, no added mass.\n");
      grid3d_wf_density(gwf, density);                     /* Density from gwf */
      force = -rgrid3d_weighted_integral(density, dpot_func, NULL);
      printf("Force = %le a.u.\n", force);
      printf("E-field = %le V/m\n", force * GRID_AUTOVPM);
      mobility = VX * GRID_AUTOMPS / (force * GRID_AUTOVPM);
      printf("Mobility = %le [cm^2/(Vs)]\n", 1.0E4 * mobility); /* 1E4 = m^2 to cm^2 */
      printf("Hydrodynamic radius (Stokes) = %le Angs.\n", 1E10 * 1.602176565E-19 / (SBC * M_PI * mobility * RHON * VISCOSITY));

      sprintf(filename, "liquid-%ld", iter);              
      dft_driver_write_density(density, filename);
    }
  }
  return 0;
}
