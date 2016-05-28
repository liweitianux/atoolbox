// calc_distance.h
// calculate commonly used cosmology distances
// Ref: https://en.wikipedia.org/wiki/Distance_measures_(cosmology)
//
// Junhua Gu
//
// ChangeLogs:
// 2012/08/12, LIweitiaNux
//   fix a bug in `calc_angular_distance()'
//   add `calc_transcomv_distance()' and
//     account `omega_k >0, <0' cases
//

#ifndef CALC_DISTANCE_H
#define CALC_DISTANCE_H

double cm=1.0;
double s=1.0;
double km=1000*100*cm;
double Mpc=3.08568e+24*cm;
double kpc=3.08568e+21*cm;
double yr=365.*24.*3600.*s;
double Gyr=1e9*yr;
double arcsec2arc_ratio=1./60/60/180*3.1415926;
double c=299792458.*100.*cm;

// cosmology parameters
double H0=71.0;         // units: [km/s/Mpc]
double omega_m=0.27;
double omega_l=0.73;
double omega_k=1.0-omega_m-omega_l;

// precision to determine float number equality
double eps=1.0e-7;

// open functions
//
// dimensionless Hubble parameter
extern double E(double z);
// Hubble distance
extern double calc_hubble_distance(void);
// calc `comoving distance'
extern double calc_comoving_distance(double z);
// calc `transverse comoving distance'
extern double calc_transcomv_distance(double z);
// calc `angular diameter distance'
// d_A(z) = d_M(z)/(1+z)
extern double calc_angdia_distance(double z);
// calc `luminoisity distance'
// d_L(z) = (1+z)*d_M(z)
extern double calc_luminosity_distance(double z);
// calc `light-travel distance'
// d_T(z) = d_H \int_0^z 1/((1+z)*E(z)) dz
extern double calc_ligtrav_distance(double z);

#endif /* CALC_DISTANCE_H */

//EOF
/* vim: set ts=8 sw=4 tw=0 fenc=utf-8 ft=cpp: */
