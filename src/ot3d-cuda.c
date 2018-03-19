/*
 * Orsay-Trento functions implemented in CUDA.
 *
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <grid/grid.h>
#include <grid/au.h>
#include "dft.h"
#include "ot.h"

#ifdef USE_CUDA

/*
 * Calculate Lennard-Jones term using cuda. 
 *
 * On entry: all GPU memory areas unused.
 *
 * On exit: Area 2 = density, 3 = FFT of density.
 *
 * workspace1: intermediate host memory buffer for the LJ potential.
 *
 * TODO: The last gpu2mem & add_real_to_complex can be eliminate (& modify local correlation function accordingly)).
 *
 */

EXPORT inline void dft_ot3d_cuda_add_lennard_jones_potential(dft_ot_functional *otf, cgrid3d *potential, rgrid3d *density, rgrid3d *workspace1, rgrid3d *workspace2) {

  INT len = density->grid_len, nx = density->nx, ny = density->ny, nz = density->nz, nz2 = density->cint->nz;

  grid_cuda_mem2gpu(0, (void *) otf->lennard_jones->value, len);   // to GPU area 0
  grid_cuda_mem2gpu(2, (void *) density->value, len);
  rgrid3d_cufft_fft(2, 3, nx, ny, nz);

  if(((otf->model & DFT_OT_KC) || (otf->model & DFT_OT_BACKFLOW))) {
    // The following non-cuda routines in ot3d.c expect to have FFT of density in workspace1
    // Remove this when cuda versions of KC & backflow have been implemented
    grid_cuda_gpu2mem(3, workspace1->value, workspace1->grid_len);  // area 3 has fft of rho
  }

  rgrid3d_cuda_fft_convolute(0, 3, nx, ny, nz2, workspace1->fft_norm2); // convolute on GPU
  rgrid3d_cufft_fft_inv(0, 1, nx, ny, nz);
  grid_cuda_gpu2mem(1, (void *) workspace2->value, len);
  // Areas: 2 = density, 3 = FFT of density
  grid3d_add_real_to_complex_re(potential, workspace2);
}
  
/*
 * Calculate local (hard sphere) potential using cuda.
 *
 * On entry: Area 2 = density, 3 = FFT of density.
 *
 * On exit: Area 2 = density, 3 = \bar{\rho}
 *
 */

EXPORT inline void dft_ot3d_cuda_add_local_correlation_potential(dft_ot_functional *otf, cgrid3d *potential, rgrid3d *workspace2) {

  INT nx = workspace2->nx, ny = workspace2->ny, nz = workspace2->nz, nz2 = workspace2->cint->nz, len = workspace2->grid_len;
  REAL norm = workspace2->fft_norm2;

  /* Areas: 2 = density, 3 = FFT of density (form LJ) */
  grid_cuda_mem2gpu(0, (void *) otf->spherical_avg->value, len);
  rgrid3d_cuda_fft_convolute(0, 3, nx, ny, nz2, norm);
  rgrid3d_cufft_fft_inv(0, 3, nx, ny, nz);
  /* Area 3 = \bar{\rho} */

  /* C2.1 & C3.1 */
  rgrid3d_cuda_power(3, 0, otf->c2_exp, nx, ny, nz); // 0 = ipow(3, c2)
  rgrid3d_cuda_power(3, 1, otf->c3_exp, nx, ny, nz); // 0 = ipow(3, c3)
  rgrid3d_cuda_multiply(0, otf->c2 / 2.0, nx, ny, nz);
  rgrid3d_cuda_multiply(1, otf->c3 / 3.0, nx, ny, nz);
  rgrid3d_cuda_sum(0, 1, nx, ny, nz);
  grid_cuda_gpu2mem(0, workspace2->value, len);   // when more GPU memory: add GPU workspaces and eliminate this intermediate sum
  grid3d_add_real_to_complex_re(potential, workspace2);

  /* C2.2 & C3.2 */
  rgrid3d_cuda_power(3, 0, otf->c2_exp - 1, nx, ny, nz);  // 0 = ipow(3, c2-1)
  rgrid3d_cuda_power(3, 1, otf->c3_exp - 1, nx, ny, nz);  // 1 = ipow(3, c3-1)
  rgrid3d_cuda_multiply(0, otf->c2 * otf->c2_exp / 2.0, nx, ny, nz); // For Orsay-Trento c2_exp / 2 = 1
  rgrid3d_cuda_multiply(1, otf->c3 * otf->c3_exp / 3.0, nx, ny, nz); // For Orsay-Trento c3_exp / 3 = 1
  rgrid3d_cuda_sum(0, 1, nx, ny, nz); // Area 0 = Area 0 + Area 1
  rgrid3d_cuda_product(0, 2, nx, ny, nz); // Area 0 = Area 0 * Area 2 (density)
  rgrid3d_cufft_fft(0, 1, nx, ny, nz);
  grid_cuda_mem2gpu(0, otf->spherical_avg->value, len);
  rgrid3d_cuda_fft_convolute(0, 1, nx, ny, nz2, norm);
  rgrid3d_cufft_fft_inv(0, 1, nx, ny, nz);
  grid_cuda_gpu2mem(1, workspace2->value, len);   // when more GPU memory: add GPU workspaces and eliminate this
  grid3d_add_real_to_complex_re(potential, workspace2);
}

#endif /* USE_CUDA */
