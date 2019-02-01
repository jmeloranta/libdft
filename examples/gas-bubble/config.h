/*
 * All input in a.u. except the time step, which is in fs.
 *
 */

/* Time step in imag/real iterations (fs) */
#define TIME_STEP (15.0 / GRID_AUTOFS)

/* Functional to be used (could add DFT_OT_KC and/or DFT_OT_BACKFLOW) */
//#define FUNCTIONAL (DFT_OT_PLAIN | DFT_OT_KC | DFT_OT_BACKFLOW)
#define FUNCTIONAL (DFT_OT_PLAIN)

/* Start real time simulation at this time (fs) - (400,000) */
#define STARTING_TIME (10000.0 / GRID_AUTOFS)
#define STARTING_ITER ((INT) (STARTING_TIME / TIME_STEP))

/* Maximum number of real time iterations */
#define MAXITER 80000000 

/* Output interval time (fs) (2500) */
#define OUTPUT_TIME (10000.0 / GRID_AUTOFS)
#define OUTPUT_ITER ((INT) (OUTPUT_TIME / TIME_STEP))

/* Output grid at given iterations (10,000) (leave undefined if not needed) */
#define OUTPUT_GRID (10*OUTPUT_ITER)

/* Use CUDA ? (auto detect) */
#ifdef USE_CUDA
#define CUDA
#endif

/* Predict-Correct (accurate but uses more memory) (at ts = 15 fs, no PC needed) */
/* #define PC */

/* Flow acceleration (m/s^2); 100 m/s in 10 ns -> 10^10 m/s^2 */
#define AZ (1.0E10 * GRID_AUTOS / GRID_AUTOMPS)
/* Maximum final velocity (m/s) */
#define MAXVZ (100.0 / GRID_AUTOMPS)

/* Maximum velocity allowed for evaluating backflow */
#define MAXVELOC (250.0 / GRID_AUTOMPS);

/* External pressure in bar (normal = 0) */
#define PRESSURE (0.0 / GRID_AUTOBAR)

#define THREADS 0	/* # of parallel threads to use (0 = all) */
#define NX 256    	/* # of grid points along x */ /* Largest: 729x384x384 */
#define NY 256         /* # of grid points along y */
#define NZ 512        	/* # of grid points along z */
#define STEP 2.0        /* spatial step length (Bohr) */
#define ABS_AMP 2.0     /* Absorption strength */
#define ABS_WIDTH_X 25.0  /* Width of the absorbing boundary */
#define ABS_WIDTH_Y 25.0  /* Width of the absorbing boundary */
#define ABS_WIDTH_Z 60.0  /* Width of the absorbing boundary */

#define NBINS 32                          /* Number of bins for kinetic energy */
#define BINSTEP (0.1 * GRID_AUTOANG)      /* Bin step */

/* Kinetic energy propagator */
/* #define PROPAGATOR WF_2ND_ORDER_FFT */
#define PROPAGATOR WF_2ND_ORDER_CN

#define FFTW_PLANNER 1 /* 0: FFTW_ESTIMATE, 1: FFTW_MEASURE (default), 2: FFTW_PATIENT, 3: FFTW_EXHAUSTIVE */

#define NN 2.0    /* Exponent for circulation (1, 2, 3...) */

/* Bubble parameters using exponential repulsion (approx. electron bubble) - RADD = 19.0 */
#define A0 (3.8003E5 / GRID_AUTOK)
#define A1 (1.6245 * GRID_AUTOANG)
#define A2 0.0
#define A3 0.0
#define A4 0.0
#define A5 0.0
#define RMIN 2.0
#define RADD 6.0

