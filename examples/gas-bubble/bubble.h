#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <complex.h>
#include <grid/grid.h>
#include <grid/au.h>
#include <dft/dft.h>
#include <dft/ot.h>
#include "config.h"

#define HELIUM_MASS (4.002602 / GRID_AUTOAMU) /* helium mass in AMU */
