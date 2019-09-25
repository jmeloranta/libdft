/*
 * All input in a.u. except the time step, which is in fs.
 *
 *
 */

/* Number of GPUs to use */
#define NGPUS 6
#define GPUS {0, 1, 2, 3, 4, 5}

/* Time step in imag/real iterations (fs) */
#define TIME_STEP (5.0 / GRID_AUTOFS)

/* Functional to be used (could add DFT_OT_KC and/or DFT_OT_BACKFLOW) */
//#define FUNCTIONAL (DFT_OT_PLAIN | DFT_OT_KC)
//#define FUNCTIONAL (DFT_OT_PLAIN | DFT_OT_KC | DFT_OT_BACKFLOW | DFT_OT_HD)
//#define FUNCTIONAL (DFT_OT_PLAIN | DFT_OT_BACKFLOW)
#define FUNCTIONAL (DFT_OT_PLAIN)
//#define FUNCTIONAL (DFT_GP)

/* Start real time simulation at this time (fs) - (100,000) */
#define STARTING_TIME (100000.0 / GRID_AUTOFS)
#define STARTING_ITER ((INT) (STARTING_TIME / TIME_STEP))

/* Maximum number of real time iterations */
#define MAXITER 80000000

/* Initial velocity */
#define INIVZ (50.0 / GRID_AUTOMPS)

/* Final velocity for simulation */
#define FINVZ (85.0 / GRID_AUTOMPS)

/* Acceleration to go from INIVZ to FINVZ */
#define ACCVZ (3E11 * GRID_AUTOS / GRID_AUTOMPS)

/* Output interval time (fs) (5,000) */
#define OUTPUT_TIME (5000.0 / GRID_AUTOFS)
#define OUTPUT_ITER ((INT) (OUTPUT_TIME / TIME_STEP))

/* Output grid at given iterations (10,000) (leave undefined if not needed) */
#define OUTPUT_GRID (2*OUTPUT_ITER)

/* Use CUDA ? (auto detect) */
#ifdef USE_CUDA
#define CUDA
#endif

/* Predict-Correct (accurate but uses more memory) (at ts = 15 fs, no PC needed) */
#define PC

/* Max velocity for evaluating backflow */
#define MAXVELOC (200.0 / GRID_AUTOMPS)

/* External pressure in bar (normal = 0) */
#define PRESSURE (0.0 / GRID_AUTOBAR)

#define THREADS 0	/* # of parallel threads to use (0 = all) */
#define NX 384    	/* # of grid points along x */ /* Largest: 729x384x384 */
#define NY 384         /* # of grid points along y */
#define NZ 2048        	/* # of grid points along z */
#define STEP 2.0        /* spatial step length (Bohr) */

/* Kinetic energy propagator */
#define PROPAGATOR WF_2ND_ORDER_FFT
//#define PROPAGATOR WF_2ND_ORDER_CFFT
//#define PROPAGATOR WF_4TH_ORDER_FFT
//#define PROPAGATOR WF_4TH_ORDER_CFFT
//#define PROPAGATOR WF_2ND_ORDER_CN

#define FFT_STAB 0.005  /* Fraction of imaginary time to use during real-time propagation (to stabilize the solution) */

#define ABS_WIDTH_X 25.0  /* Width of the absorbing boundary */
#define ABS_WIDTH_Y 25.0  /* Width of the absorbing boundary */
#define ABS_WIDTH_Z 50.0  /* Width of the absorbing boundary */
//#define ABS_AMP 1.0       /* Use the baseline value (comment out to remove the abs boundary) */

#define NBINS 32                          /* Number of bins for kinetic energy */
#define BINSTEP (0.1 * GRID_AUTOANG)      /* Bin step */

#define FFTW_PLANNER 1 /* 0: FFTW_ESTIMATE, 1: FFTW_MEASURE (default), 2: FFTW_PATIENT, 3: FFTW_EXHAUSTIVE */

#define NN 1.0    /* Exponent for circulation (1, 2, 3...) -- 1 appears standard */

/* Bubble parameters using exponential repulsion (approx. electron bubble) - RADD = 19.0 */
#define A0 (3.8003E5 / GRID_AUTOK)
#define A1 (1.6245 * GRID_AUTOANG)
#define A2 0.0
#define A3 0.0
#define A4 0.0
#define A5 0.0
// #define RMIN 12.0
#define RADD 6.0  // was 6.0
#define EPS 1E-6
