//
// simple `cosmology calculator'
// can calc commonly used cosmology distances:
//      hubble distance
//      comoving distance
//      transverse comoving distance
//      angular diameter distance
//      luminoisity distance
//      light-travel distance
// in addition, this calculator also calc some other
// useful infomation related with Chandra
//
// Junhua Gu
//
// Modified by: LIweitiaNux
//
// ChangeLogs:
// v2.1, 2012/08/12, LIweitiaNux
//   improve cmdline parameters
// v2.2, 2013/02/09, LIweitiaNux
//   add 'hubble_parameter E(z)'
//   modify output format
//


#include "calc_distance.h"
#include "ddivid.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <iostream>

using namespace std;

// extern variables in `calc_distance.h'
extern double cm;
extern double s;
extern double km;
extern double Mpc;
extern double kpc;
extern double yr;
extern double Gyr;
extern double H0;           // units: [km/s/Mpc]
extern double c;
extern double omega_m;
extern double omega_l;
extern double omega_k;
extern double arcsec2arc_ratio;

// README, ABOUT
static char DESC[] = "simple cosmology calculator";
static char VER[]  = "v2.2, 2013-02-09";

// setting parameters
static double PI = 4*atan(1.0);
// chandra related
static double arcsec_per_pixel = 0.492;
// above cosmology paramters can also be changed
// through cmdline paramters

// functions
void usage(const char *name);

//////////////////////////////////////////////
// main part
//////////////////////////////////////////////
int main(int argc,char* argv[])
{
    double z;       // given redshift

    // cmdline parameters
    if (argc == 2) {
        // get input redshift
        z = atof(argv[1]);
    }
    else if (argc == 3) {
        z = atof(argv[1]);
        // use specified `H0'
        H0 = atof(argv[2]);
    }
    else if (argc == 4) {
        z = atof(argv[1]);
        H0 = atof(argv[2]);
        // get specified `Omega_M'
        omega_m = atof(argv[3]);
        omega_l = 1.0-omega_m;
    }
    else {
        usage(argv[0]);
        exit(-1);
    }

    // calc `Hubble parameter E(z)'
    double E_z = E(z);
    // calc `comoving distance'
    double d_C = calc_comoving_distance(z);
    // calc `angular diameter distance'
    double d_A = calc_angdia_distance(z);
    // calc `luminoisity distance'
    double d_L = calc_luminosity_distance(z);

    // output results
    // parameters
    printf("Parameters:\n");
    printf("  z= %lf, H0= %lf, Omega_M= %lf, Omega_L= %lf\n",
            z, H0, omega_m, omega_l);
    printf("Distances:\n");
    printf("  Comoving_distance: D_C(%lf)= %lg [cm], %lf [Mpc]\n",
            z, d_C, d_C/Mpc);
    printf("  Angular_diameter_distance: D_A(%lf)= %lg [cm], %lf [Mpc]\n",
            z, d_A, d_A/Mpc);
    printf("  Luminoisity_distance: D_L(%lf)= %lg [cm], %lf [Mpc]\n",
            z, d_L, d_L/Mpc);
    printf("Chandra_related:\n");
    printf("  kpc/pixel (D_A): %lf\n",
            (d_A / kpc * arcsec_per_pixel* arcsec2arc_ratio));
    printf("  cm/pixel (D_A): %lg\n",
            (d_A * arcsec_per_pixel* arcsec2arc_ratio));
    printf("Other_data:\n");
    printf("  Hubble_parameter: E(%lf)= %lf\n", z, E_z);
    printf("  kpc/arcsec (D_A): %lf\n", (d_A / kpc * arcsec2arc_ratio));
    printf("  norm (cooling_function): %lg\n",
            (1e-14 / (4.0 * PI * pow(d_A*(1+z), 2))));

    //cout<<ddivid(calc_distance,d,0,1,.0001)<<endl;

    return 0;
}

// other auxiliary functions
void usage(const char *name)
{
    cerr << "Usage: " << endl;
    cerr << "    " << name << " z [H0] [Omega_M]" << endl;
    // description
    cout << endl << "About:" << endl;
    cout << DESC << endl << VER << endl;
}

/* vim: set ts=8 sw=4 tw=0 fenc=utf-8 ft=cpp: */
