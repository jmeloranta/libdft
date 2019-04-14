/*
 * Potential functions for bubble.
 *
 */

#include "bubble.h"

REAL pot_func(void *NA, REAL x, REAL y, REAL z) {

  REAL r, r2, r4, r6, r8, r10;

  r = SQRT(x * x + y * y + z * z);
  r -= RADD;
#ifdef RMIN
  if(r < RMIN) r = RMIN;
#endif

  r2 = r * r;
  r4 = r2 * r2;
  r6 = r4 * r2;
  r8 = r6 * r2;
  r10 = r8 * r2;
  return A0 * EXP(-A1 * r) 
#ifdef RMIN
   - A2 / r4 - A3 / r6 - A4 / r8 - A5 / r10
#endif
  ;
}

/* Impurity must always be at the origin (dU/dx) */
REAL dpot_func_x(void *NA, REAL x, REAL y, REAL z) {

  REAL r, rp, r2, r3, r5, r7, r9, r11;

  rp = SQRT(x * x + y * y + z * z);
  r = rp - RADD;
#ifdef RMIN
  if(r < RMIN) return 0.0;
#endif

  r2 = r * r;
  r3 = r2 * r;
  r5 = r2 * r3;
  r7 = r5 * r2;
  r9 = r7 * r2;
  r11 = r9 * r2;
  
  return (x / (rp + EPS)) * (-A0 * A1 * EXP(-A1 * r)
#ifdef RMIN
   + 4.0 * A2 / r5 + 6.0 * A3 / r7 + 8.0 * A4 / r9 + 10.0 * A5 / r11
#endif
  );
}

/* Impurity must always be at the origin (dU/dy) */
REAL dpot_func_y(void *NA, REAL x, REAL y, REAL z) {

  REAL r, rp, r2, r3, r5, r7, r9, r11;

  rp = SQRT(x * x + y * y + z * z);
  r = rp - RADD;
#ifdef RMIN
  if(r < RMIN) return 0.0;
#endif

  r2 = r * r;
  r3 = r2 * r;
  r5 = r2 * r3;
  r7 = r5 * r2;
  r9 = r7 * r2;
  r11 = r9 * r2;
  
  return (y / (rp + EPS)) * (-A0 * A1 * EXP(-A1 * r) 
#ifdef RMIN
   + 4.0 * A2 / r5 + 6.0 * A3 / r7 + 8.0 * A4 / r9 + 10.0 * A5 / r11
#endif
  );
}

/* Impurity must always be at the origin (dU/dz) */
REAL dpot_func_z(void *NA, REAL x, REAL y, REAL z) {

  REAL r, rp, r2, r3, r5, r7, r9, r11;

  rp = SQRT(x * x + y * y + z * z);
  r = rp - RADD;
#ifdef RMIN
  if(r < RMIN) return 0.0;
#endif

  r2 = r * r;
  r3 = r2 * r;
  r5 = r2 * r3;
  r7 = r5 * r2;
  r9 = r7 * r2;
  r11 = r9 * r2;
  
  return (z / (rp + EPS)) * (-A0 * A1 * EXP(-A1 * r) 
#ifdef RMIN
    + 4.0 * A2 / r5 + 6.0 * A3 / r7 + 8.0 * A4 / r9 + 10.0 * A5 / r11
#endif
  );
}
