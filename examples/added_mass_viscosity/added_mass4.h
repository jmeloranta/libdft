#define TIME_STEP 100.0	/* Time step in fs (50-100) */
#define IMP_STEP 0.1	/* Time step in fs (0.01) */
#define MAXITER 500000 /* Maximum number of iterations (was 300) */
#define OUTPUT     5	/* output every this iteration */
#define OUTPUT2    10  /* write output files @ OUTPUT * OUTPUT2 */
#define THREADS 0	/* # of parallel threads to use (0 = all) */
#define PLANNING 1     /* 0 = estimate, 1 = measure, 2 = patient, 3 = exhaustive */
#define NX 1024       	/* # of grid points along x */
#define NY 512         /* # of grid points along y */
#define NZ 512      	/* # of grid points along z */
#define STEP 0.8        /* spatial step length (Bohr) */

#define HELIUM_MASS (4.002602 / GRID_AUTOAMU) /* helium mass */
#define IMP_MASS 1.0 /* electron mass */

#define PSPOT "/home/git/libdft/examples/electron/jortner.dat"

/* velocity components */
#define KX	(1.0 * 2.0 * M_PI / (NX * STEP))
#define KY	(0.0 * 2.0 * M_PI / (NY * STEP))
#define KZ	(0.0 * 2.0 * M_PI / (NZ * STEP))
#define VX	(KX * HBAR / HELIUM_MASS)
#define VY	(KY * HBAR / HELIUM_MASS)
#define VZ	(KZ * HBAR / HELIUM_MASS)
#define EKIN	(0.5 * HELIUM_MASS * (VX * VX + VY * VY + VZ * VZ))

#define T1200MK
#define DONNELLY /* Use rho_n and eta from Donnelly */
/* Use rho_n and eta from Fred or rho_n from NIST? */
/* #define FRED */

 /* If not def, Use 0 to get 1.73 + 2.32E-10 * exp(11.15 * T) fit */
/* #define ALPHA 150.0 */

