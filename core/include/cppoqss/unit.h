#ifndef UNIT_H
#define UNIT_H


#include <cmath>


namespace cppoqss {


/**
 * SI unit system.
 */
namespace u {


/**
* Base units.
*/

constexpr double m	= 1.0;
constexpr double cm	= 1e-2 * m;

constexpr double kg	= 1.0;

constexpr double s	= 1.0;
constexpr double us	= 1e-6 * s;
constexpr double ns	= 1e-9 * s;
constexpr double ps	= 1e-12 * s;

constexpr double A	= 1.0;

constexpr double Hz	= 1.0 / s;
constexpr double MHz	= 1e6 * Hz;
constexpr double GHz	= 1e9 * Hz;

constexpr double J	= kg * m * m / s / s;
constexpr double W	= J / s;

constexpr double V	= W / A;
constexpr double C	= A * s;
constexpr double F	= C / V;


/**
* Fundamental constants.
*/

constexpr double pi	    = M_PI;

constexpr double e	    = 1.602176634e-19 * C;	/**< Exact */
constexpr double c	    = 2.99792458e8 * m / s;	/**< Exact */
constexpr double hbar	    = 1.054571817e-34 * J * s;	/**< Exact */
constexpr double epsilon0   = 8.8541878128e-12 * F / m;

constexpr double eV	    = e * V;

constexpr double alpha	    = 7.2973525693e-3;
constexpr double m_e	    = 9.1093837015e-31 * kg;


} // namespace u


/**
 * Natural unit system.
 */
namespace u_nat {


constexpr double c	    = 1.0;
constexpr double hbar	    = 1.0;
constexpr double epsilon0   = 1.0;


/**
 * Energy unit base
 */

constexpr double eV	= 1;

constexpr double s	= 1.0 / (6.582119569e-16 * eV);
constexpr double us	= 1e-6 * s;
constexpr double ns	= 1e-9 * s;
constexpr double ps	= 1e-12 * s;

constexpr double m	= 1.0 / (197.3269804e-9 * eV);
constexpr double cm	= 1e-2 * m;

constexpr double Hz	= 1.0 / s;
constexpr double MHz	= 1e6 * Hz;
constexpr double GHz	= 1e9 * Hz;

constexpr double J	= 6.241509074e18 * eV;

constexpr double W	= 1. * J / s;


/**
* Fundamental constants.
*/

constexpr double pi	= M_PI;
constexpr double alpha	= 7.2973525693e-3;
constexpr double e	= 1.602176634e-19;
constexpr double m_e	= 0.51099895000e6 * eV;


} // namespace u_nat


} // namespace cppoqss


#endif
