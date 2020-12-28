/*
 * Orsay-Trento functional for superfluid helium. Energy density.
 *
 * NOTE: This code uses FFT for evaluating all the integrals, which
 *       implies periodic boundary conditions!
 *
 * TODO: For 1-D, do not compute X, Y components of vector fields.
 * TODO: add analytic calculation of pressure using P = dU/dV.
 *
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <grid/grid.h>
#include <grid/au.h>
#include "dft.h"
#include "ot.h"

/*
 * @FUNC{dft_ot_energy_density, "Evaluate Orsay-Trento (helium) energy density"}
 * @DESC{"Evaluate the potential part to the energy density. Integrate to get the total energy.
          Note: the single particle kinetic portion is NOT included.
          (use grid_wf_kinetic_energy() to calculate this separately)"}
 * @ARG1{dft_ot_functional *otf, "OT functional structure"}
 * @ARG2{rgrid *energy_density, "Energy density grid (output)"}
 * @ARG3{wf *wf, "Wave function (input)"}
 * @RVAL{void, "No return value"}
 *
 * Workspace usage (uses density as well):
 * GP: none
 * Plain OT: workspace1 - workspace2
 * KC: workspace1 - workspace8
 * BF: workspace1 - workspace7
 *
 */

EXPORT void dft_ot_energy_density(dft_ot_functional *otf, rgrid *energy_density, wf *wf) {

  rgrid *workspace1, *workspace2;
  rgrid *density;

  density = otf->density;  
  grid_wf_density(wf, density);

  rgrid_zero(energy_density);

  if(otf->model & DFT_ZERO) {
    fprintf(stderr, "libdft: Warning - zero potential used.\n");
    return;
  }

  if((otf->model & DFT_GP) || (otf->model & DFT_GP2)) {
    /* the energy functional is: (\lambda/2)\int \left|\psi\right|^4 d\tau */
    rgrid_add_scaled_product(energy_density, 0.5 * otf->mu0 / otf->rho0, density, density);
    return;
  }

  if(!otf->workspace1) otf->workspace1 = rgrid_clone(density, "OTF workspace 1");
  if(!otf->workspace2) otf->workspace2 = rgrid_clone(density, "OTF workspace 2");
  workspace1 = otf->workspace1;
  workspace2 = otf->workspace2;

  /* transform rho (wrk1) */
  rgrid_copy(workspace1, density);
  rgrid_fft(workspace1);

  /* Lennard-Jones */  
  /* (1/2) rho(r) int V_lj(|r-r'|) rho(r') dr' */
  rgrid_fft_convolute(workspace2, workspace1, otf->lennard_jones);
  rgrid_inverse_fft_norm2(workspace2);
  rgrid_add_scaled_product(energy_density, 0.5, density, workspace2);

  /* non-local correlation */
  /* wrk1 = \bar{\rho} */
  rgrid_fft_convolute(workspace1, workspace1, otf->spherical_avg);
  rgrid_inverse_fft_norm2(workspace1);

  /* C2 term */
  if(otf->model & DFT_DR) 
    rgrid_power(workspace2, workspace1, otf->c2_exp);
  else
    rgrid_ipower(workspace2, workspace1, (INT) otf->c2_exp);
  rgrid_product(workspace2, workspace2, density);
  rgrid_add_scaled(energy_density, otf->c2 / 2.0, workspace2);

  /* C3 term */
  if(otf->model & DFT_DR) 
    rgrid_power(workspace2, workspace1, otf->c3_exp);
  else
    rgrid_ipower(workspace2, workspace1, (INT) otf->c3_exp);
  rgrid_product(workspace2, workspace2, density);
  rgrid_add_scaled(energy_density, otf->c3 / 3.0, workspace2);

  /* Barranco's contribution (high density) */
  if((otf->model & DFT_OT_HD) || (otf->model & DFT_OT_HD2)) {
    grid_func5_operate_one(workspace1, density, otf->beta, otf->rhom, otf->C);
    rgrid_sum(energy_density, energy_density, workspace1);
  }

  /* Ideal gas contribution (thermal) */
  if(otf->model >= DFT_OT_T400MK && otf->model < DFT_GP) { /* do not add this for DR */
    grid_func6b_operate_one(workspace1, density, otf->mass, otf->temp, otf->c4);
    rgrid_sum(energy_density, energy_density, workspace1);
  }

  if(otf->model & DFT_OT_KC) dft_ot_energy_density_kc(otf, energy_density, wf, density);

  if(otf->model & DFT_OT_BACKFLOW) dft_ot_energy_density_bf(otf, energy_density, wf, density);
}

/*
 * @FUNC{dft_ot_energy_density_kc, "Orsay-Trento (helium): Evaluate non-local (KC) energy density"}
 * @DESC{"Evaluate non-local (KC) energy density. This is usually not called directly"}
 * @ARG1{dft_ot_functional *otf, "OT functional structure"}
 * @ARG2{rgrid *energy_density, "Energy density grid"}
 * @ARG3{wf *wf, "Wave function"}
 * @ARG4{rgrid *density, "Density (input). This can be otf$->$density"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT void dft_ot_energy_density_kc(dft_ot_functional *otf, rgrid *energy_density, wf *wf, rgrid *density) {

  rgrid *workspace1, *workspace2, *workspace3, *workspace4, *workspace5, *workspace6, *workspace7, *workspace8;

  if(!otf->workspace1) otf->workspace1 = rgrid_clone(density, "OTF workspace 1");
  if(!otf->workspace2) otf->workspace2 = rgrid_clone(density, "OTF workspace 2");
  if(!otf->workspace3) otf->workspace3 = rgrid_clone(density, "OTF workspace 3");
  if(!otf->workspace4) otf->workspace4 = rgrid_clone(density, "OTF workspace 4");
  if(!otf->workspace5) otf->workspace5 = rgrid_clone(density, "OTF workspace 5");
  if(!otf->workspace6) otf->workspace6 = rgrid_clone(density, "OTF workspace 6");
  if(!otf->workspace7) otf->workspace7 = rgrid_clone(density, "OTF workspace 7");
  if(!otf->workspace8) otf->workspace8 = rgrid_clone(density, "OTF workspace 8");
  workspace1 = otf->workspace1;
  workspace2 = otf->workspace2;
  workspace3 = otf->workspace3;
  workspace4 = otf->workspace4;
  workspace5 = otf->workspace5;
  workspace6 = otf->workspace6;
  workspace7 = otf->workspace7;
  workspace8 = otf->workspace8;

  /* 1. convolute density with F to get \tilde{\rho} (wrk1) */
  rgrid_copy(workspace2, density);
  rgrid_fft(workspace2);
  rgrid_fft_convolute(workspace1, workspace2, otf->gaussian_tf);   /* otf->gaussian_tf is already in Fourier space */    
  rgrid_inverse_fft_norm2(workspace1);

  /* 2. modify wrk1 from \tilde{\rho} to (1 - \tilde{\rho}/\rho_{0s} */
  rgrid_multiply_and_add(workspace1, -1.0/otf->rho_0s, 1.0);

  /* 3. gradient \rho to wrk3 (x), wrk4 (y), wrk5 (z) */
  rgrid_gradient(density, workspace3, workspace4, workspace5);
    
  /* 4. X component: wrk6 = wrk3 * wrk1 = ((d/dx)\rho_x) * (1 - \tilde{\rho}/\rho_{0s}) */
  /*    Y component: wrk7 = wrk4 * wrk1 = ((d/dy)\rho_y) * (1 - \tilde{\rho}/\rho_{0s}) */
  /*    Z component: wrk8 = wrk5 * wrk1 = ((d/dz)\rho_z) * (1 - \tilde{\rho}/\rho_{0s}) */
  rgrid_product(workspace6, workspace3, workspace1);
  rgrid_product(workspace7, workspace4, workspace1);
  rgrid_product(workspace8, workspace5, workspace1);

  /* 5. convolute (X): wrk6 = convolution(otf->gaussian * wrk6). */
  /*    convolute (Y): wrk7 = convolution(otf->gaussian * wrk7). */
  /*    convolute (Z): wrk8 = convolution(otf->gaussian * wrk8). */
  rgrid_fft(workspace6);
  rgrid_fft(workspace7);
  rgrid_fft(workspace8);
  rgrid_fft_convolute(workspace6, workspace6, otf->gaussian_tf);
  rgrid_fft_convolute(workspace7, workspace7, otf->gaussian_tf);
  rgrid_fft_convolute(workspace8, workspace8, otf->gaussian_tf);
  rgrid_inverse_fft_norm2(workspace6);
  rgrid_inverse_fft_norm2(workspace7);
  rgrid_inverse_fft_norm2(workspace8);

  /* 6. X: wrk6 = wrk6 * wrk3 * wrk1 */
  /*    Y: wrk7 = wrk7 * wrk4 * wrk1 */
  /*    Z: wrk8 = wrk8 * wrk5 * wrk1 */
  rgrid_product(workspace6, workspace6, workspace3);
  rgrid_product(workspace6, workspace6, workspace1);
  rgrid_product(workspace7, workspace7, workspace4);
  rgrid_product(workspace7, workspace7, workspace1);
  rgrid_product(workspace8, workspace8, workspace5);
  rgrid_product(workspace8, workspace8, workspace1);
    
  /* 7. add wrk6 + wrk7 + wrk8 (components from the dot product) */
  /* 8. multiply by -\hbar^2\alpha_s/(4M_{He}) */    
  rgrid_add_scaled(energy_density, -otf->alpha_s / (4.0 * otf->mass), workspace6);
  rgrid_add_scaled(energy_density, -otf->alpha_s / (4.0 * otf->mass), workspace7);
  rgrid_add_scaled(energy_density, -otf->alpha_s / (4.0 * otf->mass), workspace8);
}

/*
 * @FUNC{dft_ot_energy_density_bf, "Orsay-Trento (helium): Backflow energy density (BF)"}
 * @DESC{"Evaluate backflow (BF) energy density. This is usually not called directly"}
 * @ARG1{dft_ot_functional *otf, "OT functional structure"}
 * @ARG2{rgrid *energy_density, "Energy density grid (output). This can be otf->density"}
 * @RVAL{void, "No return value"}
 *
 */

EXPORT void dft_ot_energy_density_bf(dft_ot_functional *otf, rgrid *energy_density, wf *wf, rgrid *density) {

  rgrid *workspace1, *workspace2, *workspace3, *workspace4, *workspace5, *workspace6, *workspace7;

  if(!otf->workspace1) otf->workspace1 = rgrid_clone(density, "OTF workspace 1");
  if(!otf->workspace2) otf->workspace2 = rgrid_clone(density, "OTF workspace 2");
  if(!otf->workspace3) otf->workspace3 = rgrid_clone(density, "OTF workspace 3");
  if(!otf->workspace4) otf->workspace4 = rgrid_clone(density, "OTF workspcae 4");
  if(!otf->workspace5) otf->workspace5 = rgrid_clone(density, "OTF workspace 5");
  if(!otf->workspace6) otf->workspace6 = rgrid_clone(density, "OTF workspace 6");
  if(!otf->workspace7) otf->workspace7 = rgrid_clone(density, "OTF workspace 7");
  workspace1 = otf->workspace1;
  workspace2 = otf->workspace2;
  workspace3 = otf->workspace3;
  workspace4 = otf->workspace4;
  workspace5 = otf->workspace5;
  workspace6 = otf->workspace6;
  workspace7 = otf->workspace7;

  if((otf->model & DFT_OT_HD) || (otf->model & DFT_OT_HD2)) {
    /* M & M high density cutoff */
    grid_func2_operate_one(workspace7, density, otf->xi, otf->rhobf);
  } else {
    /* Original BF */
    rgrid_copy(workspace7, density);
  }
  // workspace7 = density from this on

  /* Velocity components to wrk1(X), wrk2(Y), wrk3(Z) */    
  grid_wf_velocity(wf, workspace1, workspace2, workspace3, DFT_EPS);
#ifdef DFT_MAX_VELOC
  rgrid_threshold_clear(workspace1, workspace1, DFT_MAX_VELOC, -DFT_MAX_VELOC, DFT_MAX_VELOC, -DFT_MAX_VELOC);
  rgrid_threshold_clear(workspace2, workspace2, DFT_MAX_VELOC, -DFT_MAX_VELOC, DFT_MAX_VELOC, -DFT_MAX_VELOC);
  rgrid_threshold_clear(workspace3, workspace3, DFT_MAX_VELOC, -DFT_MAX_VELOC, DFT_MAX_VELOC, -DFT_MAX_VELOC);
#endif

  rgrid_product(workspace4, workspace1, workspace1);   /* v_x^2 */
  rgrid_product(workspace5, workspace2, workspace2);   /* v_y^2 */
  rgrid_sum(workspace4, workspace4, workspace5);
  rgrid_product(workspace5, workspace3, workspace3);   /* v_z^2 */
  rgrid_sum(workspace4, workspace4, workspace5);       /* wrk4 = v_x^2 + v_y^2 + v_z^2 */

  /* Term 1: -(M/4) * rho(r) * v(r)^2 \int U_j(|r - r'|) * rho(r') d3r' */
  rgrid_copy(workspace5, workspace7);   /* wrk7 = density */
  rgrid_fft(workspace5);                        /* This was done before - TODO: save previous rho FFT and reuse here */
  rgrid_fft_convolute(workspace6, otf->backflow_pot, workspace5);
  rgrid_inverse_fft_norm2(workspace6);
  rgrid_product(workspace6, workspace6, workspace4); /* x v(r)^2 */
  rgrid_product(workspace6, workspace6, workspace7); /* x rho(r) */
  rgrid_add_scaled(energy_density, -otf->mass / 4.0, workspace6);

  /* Term 2 (cross term, 2x): +(M/2) * rho(r) v(r) . \int U_j(|r - r'|) * rho(r') v(r') d3r' */
  /* x contribution */
  rgrid_product(workspace5, workspace7, workspace1);   /* wrk5 = rho(r') * v_x(r') */
  rgrid_fft(workspace5);
  rgrid_fft_convolute(workspace6, otf->backflow_pot, workspace5);
  rgrid_inverse_fft_norm2(workspace6);
  rgrid_product(workspace6, workspace6, workspace7); /* x rho(wrk7) */
  rgrid_product(workspace6, workspace6, workspace1); /* x v_x(wrk1) */
  rgrid_add_scaled(energy_density, otf->mass / 2.0, workspace6);

  /* y contribution */
  rgrid_product(workspace5, workspace7, workspace2);   /* rho(r') * v_y(r') */
  rgrid_fft(workspace5);
  rgrid_fft_convolute(workspace6, otf->backflow_pot, workspace5);
  rgrid_inverse_fft_norm2(workspace6);
  rgrid_product(workspace6, workspace6, workspace7); /* x rho(wrk7) */
  rgrid_product(workspace6, workspace6, workspace2); /* x v_y(wrk2) */
  rgrid_add_scaled(energy_density, otf->mass / 2.0, workspace6);

  /* z contribution */
  rgrid_product(workspace5, workspace7, workspace3);   /* rho(r') * v_z(r') */
  rgrid_fft(workspace5);
  rgrid_fft_convolute(workspace6, otf->backflow_pot, workspace5);
  rgrid_inverse_fft_norm2(workspace6);
  rgrid_product(workspace6, workspace6, workspace7); /* x rho(wrk7) */
  rgrid_product(workspace6, workspace6, workspace3); /* x v_z(wrk3) */
  rgrid_add_scaled(energy_density, otf->mass / 2.0, workspace6);

  /* Term 3: -(M/4) rho(r) \int U_j(|r - r'|) rho(r') v^2(r') d3r' */
  rgrid_product(workspace5, workspace7, workspace4); /* wrk5 = rho x |v|^2 */
  rgrid_fft(workspace5);
  rgrid_fft_convolute(workspace6, otf->backflow_pot, workspace5);
  rgrid_inverse_fft_norm2(workspace6);
  rgrid_product(workspace6, workspace6, workspace7);
  rgrid_add_scaled(energy_density, -otf->mass / 4.0, workspace6);
}
