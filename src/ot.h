/*
 * Orsay-Trento functional for superfluid helium.
 *
 */

#ifndef __DFT_OT__
#define __DFT_OT__

/*
 * Models:
 *
 * DFT_OT_KC       Include the non-local kinetic energy correlation.
 * DFT_OT_HD       Include Barranco's high density correction (short-range and backflow).
 * DFT_OT_HD2      Include Barranco's high density correction (short-range and backflow). Different parametrization.
 * DFT_OT_BACKFLOW Inlcude the backflow potential (dynamics). To get the new BF form, use DFT_OT_HD or DFT_OT_HD2.
 * DFT_OT_T0MK     Thermal model 0.0 K (i.e. just new parametrization)
 * DFT_OT_T400MK   Thermal model 0.4 K
 * DFT_OT_T600MK   Thermal model 0.6 K
 * DFT_OT_T800MK   Thermal model 0.8 K
 * DFT_OT_T1200MK  Thermal model 1.2 K
 * DFT_OT_T1400MK  Thermal model 1.4 K
 * DFT_OT_T1600MK  Thermal model 1.6 K
 * DFT_OT_T1800MK  Thermal model 1.8 K
 * DFT_OT_T2000MK  Thermal model 2.0 K
 * DFT_OT_T2100MK  Thermal model 2.1 K
 * DFT_OT_T2200MK  Thermal model 2.2 K
 * DFT_OT_T2400MK  Thermal model 2.4 K
 * DFT_OT_T2600MK  Thermal model 2.6 K
 * DFT_OT_T2800MK  Thermal model 2.8 K
 * DFT_OT_T3000MK  Thermal model 3.0 K
 * DFT_GP          Gross-Pitaevskii potential (gives good solvation structures)
 * DFT_GP2         Gross-Pitaevskii potential (gives the correct speed of sound)
 * DFT_DR          Dupont-Roc functional   
 *
 */

#define DFT_OT_PLAIN    0
#define DFT_OT_KC       1
#define DFT_OT_HD       2
#define DFT_OT_HD2      4
#define DFT_OT_BACKFLOW 8
#define DFT_OT_T0MK    16
#define DFT_OT_T400MK  32
#define DFT_OT_T600MK  64
#define DFT_OT_T800MK  128
#define DFT_OT_T1200MK 256
#define DFT_OT_T1400MK 512
#define DFT_OT_T1600MK 1024
#define DFT_OT_T1800MK 2048
#define DFT_OT_T2000MK 4096
#define DFT_OT_T2100MK 8192
#define DFT_OT_T2200MK 16384
#define DFT_OT_T2400MK 32768
#define DFT_OT_T2600MK 65536
#define DFT_OT_T2800MK 131072
#define DFT_OT_T3000MK 262144
#define DFT_GP         524288
#define DFT_DR         1048576
#define DFT_ZERO       2097152
#define DFT_GP2        4194304

/*
 * Structures.
 *
 */

typedef struct bf_struct {
  REAL g11, g12, g21, g22, a1, a2;
} dft_ot_bf;

/*
 *
 * Original Orsay-Trento functional: Phys. Rev. B 52, 1192 (1995).
 * Extension of OT to non-zero temperatures (thermal OT): Phys. Rev. B 62, 17035 (2000).
 * High-density correction to Orsay-Trento functional: Phys. Rev. B 72, 214522 (2005).
 * Implementation of Orsay-Trento: J. Comput. Phys. 194, 78 (2004) and J. Comput. Phys. 221, 148 (2007).
 * Review of Orsay-Trento and applications: Int. Rev. Phys. Chem. 36, 621 (2017).
 *
 * H = (1/2) int rho(r) V_lj(|r-r'|) rho(r') dr' 
 *
 *     + (c2/2) rho(r) rho_sa(r)^2 + (c3/3) rho(r) rho_sa(r)^3
 *
 *       hbar^2
 *     - ------ alpha_s int F(|r-r'|) ( 1 - rho_g(r) / rho_0s ) grad rho(r) . grad rho(r') ( 1 - rho_g(r') / rho_0s ) dr'
 *        4m
 *
 *     + BACKFLOW
 */

typedef struct dft_ot_functional_struct {   /* All values in atomic units */
  INT model;                /* Functional DFT_OT_* (Orsay-Trento), DFT_DR (Dupont-Roc), DFT_GP (Gross-Pitaevskii) */
  REAL b;                   /* Lennard-Jones integral value for bulk (not used in functional) */
  REAL c2;                  /* Orsay-Trento short-range correlation parameter c_2 (2nd power) */
  REAL c2_exp;              /* Exponent for the above (= 2 for OT) */
  REAL c3;                  /* Orsay-Trento short-range correlation parameter c_3 (3rd power) */
  REAL c3_exp;              /* Exponent for the above (= 3 for OT) */
  REAL c4;                  /* Thermal Orsay-Trento parameter c_4 (Ancilotto et al.) */
  REAL rho_0s;              /* Orsay-Trento kinetic energy correlation parameter \rho_{0s} */
  REAL alpha_s;             /* Orsay-Trento kinetic energy correlation parameter \alpha_s */
  REAL l_g;                 /* Width of gaussian F used in kinetic correlation */
  REAL mass;                /* ^4He mass */
  REAL rho0;                /* Uniform bulk liquid density at saturated vapor pressure */
  REAL temp;                /* Temperature (Kelvin) */
  dft_common_lj lj_params;  /* Lennard-Jones parameters */
  dft_ot_bf bf_params;      /* Backflow functional parameters */
  rgrid *lennard_jones;     /* Grid holding Fourier transformed effective Lennard-Jones function */
  rgrid *spherical_avg;     /* Grid holding Fourier transformed spherical average function */
  rgrid *gaussian_tf;       /* Grid holding Fourier transformed gaussian F (kinetic correlation) */ 
  rgrid *gaussian_x_tf;     /* Grid holding Fourier transformed derivative of gaussian F (dF/dx; kinetic correlation) */ 
  rgrid *gaussian_y_tf;     /* Grid holding Fourier transformed derivative of gaussian F (dF/dy; kinetic correlation) */ 
  rgrid *gaussian_z_tf;     /* Grid holding Fourier transformed derivative of gaussian F (dF/dz; kinetic correlation) */ 
  rgrid *backflow_pot;      /* Grid holding Fourier transformed bacflow function (V_j) */
  REAL beta;                /* High density correction parameter \beta */
  REAL rhom;                /* High density correction parameter \rho_m */
  REAL C;                   /* High density correction parameter C */
  REAL mu0;                 /* Determines Gross-Pitaevskii contact strength: \mu_0 / \rho_0 */
  REAL xi;                  /* High density correction parameter for backflow \xi */
  REAL rhobf;               /* High density correction parameter for backflow \rho_{bf} */
  REAL div_epsilon;         /* Epsilon to use when dividing by density (affects, e.g., backflow) */
  REAL max_veloc;           /* Maximum velocity value allowed for BF */
  REAL max_bfpot;           /* Maximum allowed potential value for BF */
  REAL c_bfpot;             /* Backflow potential weight (1 = normal OT) */
  rgrid *workspace1;        /* Workspace 1 (these may be NULL if not allocated) */
  rgrid *workspace2;        /* Workspace 2 */
  rgrid *workspace3;        /* Workspace 3 */
  rgrid *workspace4;        /* Workspace 4 */
  rgrid *workspace5;        /* Workspace 5 */
  rgrid *workspace6;        /* Workspace 6 */
  rgrid *workspace7;        /* Workspace 7 */
  rgrid *workspace8;        /* Workspace 8 */
  rgrid *workspace9;        /* Workspace 9 */
  rgrid *density;           /* Liquid density */
} dft_ot_functional;

/* Prototypes (automatically generated) */
#include "proto.h"

/* Constants */
#define DFT_HELIUM_MASS (4.002602 / GRID_AUTOAMU) /* Helium mass in atomic units */

/* Default number of substeps for Xgrid_smooth_map() */
#define DFT_MIN_SUBSTEPS 4
#define DFT_MAX_SUBSTEPS 32

/* Use special 1D OT-DFT code? */
#define DFT_OT_1D

/* Smallest density for evaluating velocity */
#define DFT_EPS 1E-4

#endif /* __DFT_OT__ */
