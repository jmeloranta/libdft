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
 * DFT_OT_HD2       Include Barranco's high density correction (short-range and backflow). Different parametrization.
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
 * DFT_GP          Gross-Pitaevskii potential
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

/*
 * Driver defines.
 *
 */

#define DFT_DRIVER_REAL_TIME 0
#define DFT_DRIVER_IMAG_TIME 1

#define DFT_DRIVER_BOUNDARY_REGULAR  0
#define DFT_DRIVER_BOUNDARY_ITIME    1

#define DFT_DRIVER_NORMALIZE_BULK    0
#define DFT_DRIVER_NORMALIZE_DROPLET 1
#define DFT_DRIVER_NORMALIZE_COLUMN  2
#define DFT_DRIVER_NORMALIZE_SURFACE 3
#define DFT_DRIVER_NORMALIZE_ZEROB   4
#define DFT_DRIVER_DONT_NORMALIZE    5
#define DFT_DRIVER_NORMALIZE_N       6

#define DFT_DRIVER_PROPAGATE_HELIUM        0
#define DFT_DRIVER_PROPAGATE_OTHER         1
#define DFT_DRIVER_PROPAGATE_OTHER_ONLYPOT 2

#define DFT_DRIVER_AVERAGE_NONE 0
#define DFT_DRIVER_AVERAGE_XY   1
#define DFT_DRIVER_AVERAGE_YZ   2
#define DFT_DRIVER_AVERAGE_XZ   3
#define DFT_DRIVER_AVERAGE_XYZ  4

#define DFT_DRIVER_VORTEX_X 0
#define DFT_DRIVER_VORTEX_Y 1
#define DFT_DRIVER_VORTEX_Z 2

#define DFT_DRIVER_BC_NORMAL 0
#define DFT_DRIVER_BC_X      1
#define DFT_DRIVER_BC_Y      2
#define DFT_DRIVER_BC_Z      3
#define DFT_DRIVER_BC_NEUMANN 4

#define DFT_DRIVER_KINETIC_FFT           0
#define DFT_DRIVER_KINETIC_CN_DBC        1
#define DFT_DRIVER_KINETIC_CN_NBC        2
#define DFT_DRIVER_KINETIC_CN_PBC        3
#define DFT_DRIVER_KINETIC_CN_APBC       4
#define DFT_DRIVER_KINETIC_CN_NBC_ROT    5

/*
 * Structures.
 *
 */

typedef struct bf_struct {
  REAL g11, g12, g21, g22, a1, a2;
} dft_ot_bf;

/*
 *
 * H = (1/2) int rho(r) V_lj(|r-r'|) rho(r') dr' 
 *
 *     + (c2/2) rho(r) rho_sa(r)^2 +(c3/3) rho(r) rho_sa(r)^3
 *
 *       hbar^2
 *     - ------ alpha_s int F(|r-r'|) ( 1 - rho_g(r) / rho_0s ) grad rho(r) . grad rho(r') ( 1 - rho_g(r') / rho_0s ) dr'
 *        4m
 */

typedef struct dft_ot_functional_struct {
  REAL b, c2, c3, c4, rho_0s, alpha_s, l_g, mass, rho_eps, rho0, temp;
  REAL c2_exp, c3_exp;
  dft_common_lj lj_params; /* Lennard-Jones parameters */
  dft_ot_bf bf_params; /* Backflow parameters */
  INT model;
  rgrid3d *lennard_jones;
  rgrid3d *spherical_avg;
  rgrid3d *gaussian_tf;
  rgrid3d *gaussian_x_tf;
  rgrid3d *gaussian_y_tf;
  rgrid3d *gaussian_z_tf;
  rgrid3d *backflow_pot;
  REAL beta; /* Barranco */
  REAL rhom; /* Barranco */
  REAL C;    /* Barranco */
  REAL mu0;  /* Gross-Pitaevskii single particle energy */
} dft_ot_functional;

/* Global user accessible variables */
extern dft_ot_functional *dft_driver_otf;
extern char dft_driver_init_wavefunction;
extern char dft_driver_kinetic;
extern REAL complex (*dft_driver_bc_function)(void *, REAL complex, INT, INT, INT);

/* Prototypes (in wrong place; TODO separate common and OT specific prototypes) */
#include "proto.h"

/* Constants */

/* Boltzmann constant in au */
#define DFT_KB 3.1668773658e-06

/* Helium mass */
#define DFT_HELIUM_MASS (4.002602 / GRID_AUTOAMU)

/* Density cutoff for backflow evaluation (was rho0 / 100 = 3E-5) */
#define DFT_BF_EPS 1E-5

/* Density cutoff for velocity calculation in driver3d.c */
#define DFT_VELOC_EPS (1E-8)

#endif /* __DFT_OT__ */
