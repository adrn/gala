#include "extra_compile_macros.h"

#if USE_EXP == 1

#include <cstdlib>
#include <memory>

// EXP headers
#include <Coefficients.H>
#include <BiorthBasis.H>
#include <FieldGenerator.H>

#include "exp_fields.h"

namespace gala_exp {

State exp_init(
  const char *config, const char *coeffile,
  int stride, double tmin, double tmax)
{
  YAML::Node yaml = YAML::LoadFile(std::string(config));

  auto basis = BasisClasses::Basis::factory(yaml);

  auto coefs = CoefClasses::Coefs::factory(std::string(coeffile),
				       stride, tmin, tmax);

  bool is_static = tmax == tmin;

  return { basis, coefs, is_static };
}

// Linear interpolator on coefficients.  Higher order interpolation
// could be implemented similarly.  This is the same implementation
// used in BiorthBasis and probably belongs in CoefClasses . . .
//
CoefClasses::CoefStrPtr interpolator(double t, CoefClasses::CoefsPtr coefs)
{
  // Interpolate coefficients
  //
  auto times = coefs->Times();

  if (t<times.front() or t>times.back()) {
    std::ostringstream sout;
    sout << "FieldWrapper::interpolator: time t=" << t << " is out of bounds: ["
	 << times.front() << ", " << times.back() << "]";
    throw std::runtime_error(sout.str());
  }
    
  auto it1 = std::lower_bound(times.begin(), times.end(), t);
  auto it2 = it1 + 1;

  if (it2 == times.end()) {
    it2--;
    it1 = it2 - 1;
  }

  double a = (*it2 - t)/(*it2 - *it1);
  double b = (t - *it1)/(*it2 - *it1);

  auto coefsA = coefs->getCoefStruct(*it1);
  auto coefsB = coefs->getCoefStruct(*it2);

  // Duplicate a coefficient instance.  Shared pointer for proper
  // garbage collection.
  //
  auto newcoef = coefsA->deepcopy();

  // Now interpolate the matrix
  //
  newcoef->time = t;
  
  auto & cN = newcoef->store;
  auto & cA = coefsA->store;
  auto & cB = coefsB->store;
  
  for (int i=0; i<newcoef->store.size(); i++)
    cN(i) = a * cA(i) + b * cB(i);
  
  // Interpolate the center data
  //
  if (coefsA->ctr.size() and coefsB->ctr.size()) {
    newcoef->ctr.resize(3);
    for (int k=0; k<3; k++)
      newcoef->ctr[k] = a * coefsA->ctr[k] + b * coefsB->ctr[k];
  }
  
  return newcoef;
}

}

/* ---------------------------------------------------------------------------
    EXP potential

    Calls the EXP code (https://github.com/exp-code/exp).
    Only available if EXP available at build time.
*/

extern "C" {

// TODO
double exp_value(double t, double *pars, double *q, int n_dim, void* state) {
  /*  pars:
          - G (Gravitational constant)
          - m (mass scale)
          - c (length scale)
  */

  gala_exp::State *exp_state = static_cast<gala_exp::State *>(state);
  
  if (!exp_state->is_static) {
    // TODO: how expensive is this, actually?
    exp_state->basis->set_coefs(gala_exp::interpolator(t, exp_state->coefs));
  }

  // Get the field quantities
  // TODO: this computes many quantities, not just the potential
  auto field = exp_state->basis->getFields(q[0], q[1], q[2]);
  
  return field[5];  // potl
}

// TODO
void exp_gradient(double t, double *pars, double *q, int n_dim, double *grad, void* state){
  /*  pars:
          - G (Gravitational constant)
          - m (mass scale)
          - c (length scale)
  */
  double R, fac;
  R = sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2]);
  fac = pars[0] * pars[1] / pars[2] * (pars[2] / (R * (pars[2] + R)));

  grad[0] = grad[0] + fac*q[0]/R;
  grad[1] = grad[1] + fac*q[1]/R;
  grad[2] = grad[2] + fac*q[2]/R;
}

// TODO
double exp_density(double t, double *pars, double *q, int n_dim, void* state) {
  /*  pars:
          - G (Gravitational constant)
          - m (mass scale)
          - c (length scale)
  */
  double r, rho0;
  r = sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2]);
  rho0 = pars[1] / (4*M_PI*pars[2]*pars[2]*pars[2]);
  return rho0 / (pow(r/pars[2],2) * pow(1+r/pars[2],2));
}

// TODO
void exp_hessian(double t, double *pars, double *q, int n_dim, double *hess, void* state) {
  /*  pars:
          - G (Gravitational constant)
          - m (mass scale)
          - c (length scale)
  */
  double G = pars[0];
  double m = pars[1];
  double c = pars[2];
  double x = q[0];
  double y = q[1];
  double z = q[2];

  double tmp_0 = pow(x, 2);
  double tmp_1 = pow(y, 2);
  double tmp_2 = pow(z, 2);
  double tmp_3 = tmp_0 + tmp_1 + tmp_2;
  double tmp_4 = 1.0/tmp_3;
  double tmp_5 = sqrt(tmp_3);
  double tmp_6 = c + tmp_5;
  double tmp_7 = pow(tmp_6, -2);
  double tmp_8 = tmp_7*x;
  double tmp_9 = 1.0/tmp_5;
  double tmp_10 = 1.0/tmp_6;
  double tmp_11 = tmp_10*tmp_9;
  double tmp_12 = G*m/c;
  double tmp_13 = tmp_12*(tmp_11*x - tmp_8);
  double tmp_14 = tmp_13*tmp_4;
  double tmp_15 = pow(tmp_3, -3.0/2.0);
  double tmp_16 = tmp_13*tmp_15*tmp_6;
  double tmp_17 = tmp_10*tmp_15;
  double tmp_18 = tmp_4*tmp_7;
  double tmp_19 = 2*tmp_9/pow(tmp_6, 3);
  double tmp_20 = tmp_11 - tmp_7;
  double tmp_21 = tmp_12*tmp_6;
  double tmp_22 = tmp_21*tmp_9;
  double tmp_23 = tmp_19*x;
  double tmp_24 = tmp_4*tmp_8;
  double tmp_25 = tmp_17*x;
  double tmp_26 = tmp_14*y - tmp_16*y + tmp_22*(tmp_23*y - tmp_24*y - tmp_25*y);
  double tmp_27 = tmp_4*z;
  double tmp_28 = tmp_13*tmp_27 - tmp_16*z + tmp_22*(tmp_23*z - tmp_24*z - tmp_25*z);
  double tmp_29 = tmp_7*y;
  double tmp_30 = tmp_11*y - tmp_29;
  double tmp_31 = tmp_12*tmp_30;
  double tmp_32 = tmp_15*tmp_21;
  double tmp_33 = tmp_30*tmp_32;
  double tmp_34 = y*z;
  double tmp_35 = tmp_22*(-tmp_17*tmp_34 + tmp_19*tmp_34 - tmp_27*tmp_29) + tmp_27*tmp_31 - tmp_33*z;
  double tmp_36 = tmp_11*z - tmp_7*z;

  hess[0] = hess[0] + tmp_14*x - tmp_16*x + tmp_22*(-tmp_0*tmp_17 - tmp_0*tmp_18 + tmp_0*tmp_19 + tmp_20);
  hess[1] = hess[1] + tmp_26;
  hess[2] = hess[2] + tmp_28;
  hess[3] = hess[3] + tmp_26;
  hess[4] = hess[4] + tmp_22*(-tmp_1*tmp_17 - tmp_1*tmp_18 + tmp_1*tmp_19 + tmp_20) + tmp_31*tmp_4*y - tmp_33*y;
  hess[5] = hess[5] + tmp_35;
  hess[6] = hess[6] + tmp_28;
  hess[7] = hess[7] + tmp_35;
  hess[8] = hess[8] + tmp_12*tmp_27*tmp_36 + tmp_22*(-tmp_17*tmp_2 - tmp_18*tmp_2 + tmp_19*tmp_2 + tmp_20) - tmp_32*tmp_36*z;
}

}

#endif  // USE_EXP
