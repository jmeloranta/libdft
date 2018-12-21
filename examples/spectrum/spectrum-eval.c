/*
 * Impurity atom in superfluid helium (no zero-point).
 *
 * All input in a.u. except the time step, which is fs.
 *
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <grid/grid.h>
#include <grid/au.h>
#include <dft/dft.h>
#include <dft/ot.h>

#define NDIV 1
#define TS (5.0 / NDIV) /* fs */
#define TC 150.0
#define ZEROFILL 1024

int main(int argc, char **argv) {

  cgrid *spectrum;
  REAL val, en;
  INT iter = 0, i;
  FILE *fp;

  grid_threads_init(6);
  if(!(fp = fopen(argv[1], "r"))) exit(1);
  while(!feof(stdin)) {
    if(fscanf(fp, " " FMT_R, &val) != 1) break;
    iter++;
  }
  rewind(fp);
  dft_driver_spectrum_init(NULL, NDIV*iter, ZEROFILL, DFT_DRIVER_AVERAGE_NONE, NULL, NULL, NULL, DFT_DRIVER_AVERAGE_NONE, NULL, NULL, NULL);
  while(!feof(stdin)) {
    if(fscanf(fp, " " FMT_R, &val) != 1) break;
    for(i = 0; i < NDIV; i++)
      dft_driver_spectrum_collect_user(val / GRID_AUTOK);
  }
  spectrum = dft_driver_spectrum_evaluate(TS, TC);

  if(!(fp = fopen("spectrum.dat", "w"))) {
    fprintf(stderr, "Can't open spectrum.dat for writing.\n");
    exit(1);
  }
  for (iter = 0, en = -0.5 * spectrum->step * spectrum->nx; iter < spectrum->nx; iter++, en += spectrum->step)
    fprintf(fp, FMT_R " " FMT_R " " FMT_R "\n", en, CREAL(cgrid_value_at_index(spectrum, 1, 1, iter)), CIMAG(cgrid_value_at_index(spectrum, 1, 1, iter)));
  fclose(fp);
  printf("Spectrum written to spectrum.dat\n");
  exit(0);
}
