#define TIME_STEP 100.0	/* Time step in fs (50-100) */
#define IMP_STEP 0.1	/* Time step in fs (0.01) */
#define MAXITER 500000 /* Maximum number of iterations (was 300) */
#define OUTPUT     500	/* output every this iteration */
#define THREADS 0	/* # of parallel threads to use (0 = all) */
#define PLANNING 1     /* 0 = estimate, 1 = measure, 2 = patient, 3 = exhaustive */
#define NZ 1024       	/* # of grid points along x */
#define NR 512         /* # of grid points along y */
#define STEP 0.5        /* spatial step length (Bohr) */

#define HELIUM_MASS (4.002602 / GRID_AUTOAMU) /* helium mass */
#define IMP_MASS 1.0 /* electron mass */

#define PSPOT "/home/eloranta/c/libdft-code/examples/3d/electron/jortner.dat"

/* velocity components */
#define KZ	(2.0 * 2.0 * M_PI / (NZ * STEP))
#define KR      0.0
#define VZ	(KZ * HBAR / HELIUM_MASS)
#define EKIN	(0.5 * HELIUM_MASS * VZ * VZ)

#define T1800MK
#define DONNELLY /* Use rho_n and eta from Donnelly */
/* #define FRED     /* Use rho_n and eta from Fred; rho_n from NIST */
#define EPSILON 5E-7
