// calc_distance.cc
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
//   add `calc_ligtrav_distance()'
//

#include <iostream>
#include <cmath>
#include <cstdlib>
#include <cstddef>
#include <cassert>
// integrator
#include "adapt_trapezoid.h"

// extern variables in `calc_distance.h'
extern double c;
extern double km;
extern double s;
extern double Mpc;
// cosmology parameters
extern double H0;       // units: [km/s/Mpc]
extern double omega_m;
extern double omega_l;
extern double omega_k;
// precision to determine float number equality
extern double eps;

using namespace std;

// auxiliary functions
// for `calc_comoving_distance'
double f_comvdist(double z);
// for `calc_ligtrav_distance'
double f_ligdist(double z);
double f_age(double z);

/////////////////////////////////////////////////
// main functions
// //////////////////////////////////////////////
// dimensionless Hubble parameter
double E(double z)
{
    return sqrt(omega_m*(1+z)*(1+z)*(1+z) + omega_k*(1+z)*(1+z) + omega_l);
}

// Hubble distance
double calc_hubble_distance(void)
{
    //
    // cerr << "*** H0 = " << H0 << " ***" << endl;
    //
    double _H0_ = H0 * km/s/Mpc;
    return c / _H0_;
}

// calc `comoving distance'
double calc_comoving_distance(double z)
{
    double d_H = calc_hubble_distance();
    return d_H * adapt_trapezoid(f_comvdist, 0.0, z, 1e-4);
}

// calc `transverse comoving distance'
double calc_transcomv_distance(double z)
{
    if (fabs(omega_k) <= eps) {
        // omega_k = 0
        // flat, d_M(z) = d_C(z)
        return calc_comoving_distance(z);
    }
    else if (omega_k > 0.0) {
        // omega_k > 0
        // d_M(z) = d_H/sqrt(omega_k) * sinh(sqrt(omega_k) * d_C(z) / d_H)
        //                   ^             ^
        double d_H = calc_hubble_distance();
        double d_C = calc_comoving_distance(z);
        return (d_H / sqrt(omega_k) * sinh(sqrt(omega_k) * d_C / d_H));
    }
    else {
        // omega_k < 0
        // d_M(z) = d_H/sqrt(-omega_k) * sin(sqrt(-omega_k) * d_C(z) / d_H)
        //                   ^             ^      ^
        double d_H = calc_hubble_distance();
        double d_C = calc_comoving_distance(z);
        return (d_H / sqrt(-omega_k) * sin(sqrt(-omega_k) * d_C / d_H));
    }
}

// calc `angular diameter distance'
// d_A(z) = d_M(z)/(1+z)
double calc_angdia_distance(double z)
{
  return (calc_transcomv_distance(z) / (1+z));
}

// calc `luminoisity distance'
// d_L(z) = (1+z)*d_M(z)
double calc_luminosity_distance(double z)
{
  return (calc_transcomv_distance(z) * (1+z));
}

// calc `light-travel distance'
// d_T(z) = d_H \int_0^z 1/((1+z)*E(z)) dz
double calc_ligtrav_distance(double z)
{
    double d_H = calc_hubble_distance();
    return d_H * adapt_trapezoid(f_ligdist, 0.0, z, 1e-4);
}


// auxiliary functions
// for `calc_comoving_distance'
double f_comvdist(double z)
{
    return 1.0/E(z);
}

// for `calc_ligtrav_distance'
double f_ligdist(double z)
{
    return 1.0 / ((1+z)*E(z));
}

double f_age(double z)
{
    return f_comvdist(1.0/z) / (z*z);
}


// EOF
/* vim: set ts=8 sw=4 tw=0 fenc=utf-8 ft=cpp: */
