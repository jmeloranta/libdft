/*
 * Runge-Kutta for propagating the classical particle.
 *
 * (incuded from classical.c)
 *
 */

double ZI; /* global variables for ion coords */

#ifdef ZERO_P
#define A0 0.0
#define A1 0.0
#define A2 0.0
#define A3 0.0
#define A4 0.0
#define A5 0.0
#define RMIN 0.0
#define RADD 0.0
#endif

/* exponential repulsion */
#ifdef EXP_P
#define A0 (3.8003E5 / GRID_AUTOK)
#define A1 (1.6245 * GRID_AUTOANG)
#define A2 0.0
#define A3 0.0
#define A4 0.0
#define A5 0.0
#define RMIN 3.0
#define RADD (-8.0)
#endif

/* Ca+ */
#ifdef CA_P
#define A0 4.83692
#define A1 1.23684
#define A2 0.273202
#define A3 59.5463
#define A4 1134.51
#define A5 0.0
#define RMIN 5.0
#define RADD 0.0
#endif

/* K+ */
#ifdef K_P
#define A0 140.757
#define A1 2.26202
#define A2 0.722065
#define A3 0.00144039
#define A4 356.303
#define A5 1358.98
#define RMIN 4.0
#define RADD 0.0
#endif

/* Be+ */
#ifdef BE_P
#define A0 4.73292
#define A1 1.53925
#define A2 0.557845
#define A3 26.7013
#define A4 0.0
#define A5 0.0
#define RMIN 3.4
#define RADD 0.0
#endif

/* Sr+ */
#ifdef SR_P
#define A0 3.64975
#define A1 1.13451
#define A2 0.293483
#define A3 99.0206
#define A4 693.904
#define A5 0.0
#define RMIN 5.0
#define RADD 0.0
#endif

/* Ba+ */
#ifdef BA_P
#define A0 10.5807
#define A1 1.24428
#define A2 0.695007
#define A3 31.9518
#define A4 2087.89
#define A5 9.14233E-8
#define RMIN 5.0  /* was 7.0 !!! */
#define RADD 0.0
#endif

/* Cl- */
#ifdef CL_M
#define A0 11.1909
#define A1 1.50971
#define A2 0.72186
#define A3 17.2434
#define A4 0.0
#define A5 0.0
#define RMIN 4.2
#define RADD 0.0
#endif

/* F- */
#ifdef F_M
#define A0 5.16101
#define A1 1.62798
#define A2 0.773982
#define A3 1.09722
#define A4 0.0
#define A5 0.0
#define RMIN 4.1
#define RADD 0.0
#endif

/* I- */
#ifdef I_M
#define A0 13.6874
#define A1 1.38037
#define A2 0.696409
#define A3 37.3331 
#define A4 0.0
#define A5 0.0
#define RMIN 4.1
#define RADD 0.0
#endif

/* Br- */
#ifdef BR_M
#define A0 12.5686
#define A1 1.45686
#define A2 0.714525
#define A3 24.114
#define A4 0.0
#define A5 0.0
#define RMIN 5.0
#define RADD 0.0
#endif

double dpot_func(void *NA, double z, double rr) {

  double r = sqrt(rr * rr + (z - ZI) * (z - ZI)) + RADD;
  double r2 = r * r;
  double r3 = r2 * r;
  double r5 = r2 * r3;
  double r7 = r5 * r2;
  double r9 = r7 * r2;
  double r11 = r9 * r2;

  if(r < RMIN) return 0.0;   /* hopefully no liquid density in the core region */
  return ((ZI - z) / r) * (-A0 * A1 * exp(-A1 * r) + 4.0 * A2 / r5 + 6.0 * A3 / r7 + 8.0 * A4 / r9 + 10.0 * A5 / r11);
}


double pot_func(void *asd, double z, double rr) {

  double r = sqrt(rr * rr + (z - ZI) * (z - ZI)) + RADD;
  double r2, r4, r6, r8, r10, tmp;

  //  if(r < RMIN) r = RMIN;
  if(r < RMIN) {
    double rt = r;
    r = RMIN;
    r2 = r * r;
    r4 = r2 * r2;
    r6 = r4 * r2;
    r8 = r6 * r2;
    r10 = r8 * r2;
    tmp = A0 * exp(-A1 * r) - A2 / r4 - A3 / r6 - A4 / r8 - A5 / r10;
    return 0.5 * (RMIN - rt) / RMIN + tmp;
  }
  r2 = r * r;
  r4 = r2 * r2;
  r6 = r4 * r2;
  r8 = r6 * r2;
  r10 = r8 * r2;
  tmp = A0 * exp(-A1 * r) - A2 / r4 - A3 / r6 - A4 / r8 - A5 / r10;
  //  if(tmp > (250.0 / GRID_AUTOK)) tmp = 250.0 / GRID_AUTOK;
  return tmp;
}

double test_weighted_integral_cyl(const rgrid2d *grid, double (*weight)(double z, double r)) {

  long i, j, ij, nxy = grid->nx * grid->ny, ny = grid->ny, nx = grid->nx;
  double z, r, step = grid->step;
  double sum = 0.0;
  double w;
  double *value = grid->value;
  
#pragma omp parallel for firstprivate(nxy,nx,ny,step,value,weight) private(w,i,j,ij,z,r) reduction(+:sum) default(none) schedule(runtime)
  for(ij = 0; ij < nxy; ij++) {
    i = ij / ny;
    j = ij % ny;    
    z = (i - nx/2) * step;
    r = step * (double) j;
    w = weight(z, r);
    sum += w * value[ij] * r;
    if(!i) {
      w = weight(-z,r);
      sum += w * value[ij] * r;
    }
  }
  
  return sum * step * step * 2.0 * M_PI;
}

/* Force evaluation for the classical atom/molecule */
/* Checked with numerical gradient */
double force(rgrid2d *density, double z) {

  ZI = z;
  return -rgrid2d_weighted_integral_cyl(density, dpot_func, NULL) + EFIELDZ;
}

double propagate_impurity(double *z, double *vz, double *az, rgrid2d *density) {

  double time_step = TIME_STEP / GRID_AUTOFS;
  double fz, vhalf;

  vhalf = *vz + 0.5 * (*az) * time_step;
  *z += vhalf * time_step;
  fz = force(density, *z);
#ifdef UNCOUPLED
  *az = EFIELDZ / IMASS;
#else
  *az = fz / IMASS;
#endif
  *vz = vhalf + 0.5 * (*az) * time_step;

  return fz;
}

void zero_core(wf2d *wf) {

  long i, j;
  double z, r;

  for (i = 0; i < NZ; i++) {
    z = (i - NZ/2) * STEP;
    for (j = 0; j < NR; j++) { 
      r = j * STEP;
      if(sqrt(r * r + (z - ZI) * (z - ZI)) < RMIN) wf->grid->value[i * NR + j] = 0.0;
    }
  }
}
