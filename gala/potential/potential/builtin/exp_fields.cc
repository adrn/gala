#include "extra_compile_macros.h"

#if USE_EXP == 1

#include <cstdlib>
#include <memory>
#include <cmath>

// EXP headers
#include <Coefficients.H>
#include <BiorthBasis.H>
#include <FieldGenerator.H>

#include "exp_fields.h"

namespace gala_exp {

State exp_init(
  const std::string &config_fn, const std::string &coeffile,
  int stride, double tmin, double tmax)
{
  YAML::Node yaml = YAML::LoadFile(std::string(config_fn));

  auto basis = BasisClasses::Basis::factory(yaml);

  auto coefs = CoefClasses::Coefs::factory(coeffile,
				       stride, tmin, tmax);

  bool is_static = tmax == tmin;

  if (is_static) {
    basis->set_coefs(gala_exp::interpolator(tmin, coefs));
  }

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

double exp_value(double t, double *pars, double *q, int n_dim, void* state) {
  /*  pars:
          - G (Gravitational constant)
          - m (mass scale)
          - r_vir (scale length)
  */

  double G = pars[0];
  double M = pars[1];
  double r_vir = pars[2];

  gala_exp::State *exp_state = static_cast<gala_exp::State *>(state);
  
  if (!exp_state->is_static) {
    // TODO: what time units does Gala use? What units does EXP expect?
    // TODO: how expensive is this, actually?
    exp_state->basis->set_coefs(gala_exp::interpolator(t, exp_state->coefs));
  }

  // Get the field quantities
  // TODO: this computes many quantities, not just the potential
  // TODO: check what lengths Gala is passing. Do they need to be rescaled to EXP units (using r_vir)?
  auto field = exp_state->basis->getFields(q[0] / r_vir, q[1] / r_vir, q[2] / r_vir);
  
  double pot = G * M * field[5];

  return pot;
}

void exp_gradient(double t, double *pars, double *q, int n_dim, double *grad, void* state){
  gala_exp::State *exp_state = static_cast<gala_exp::State *>(state);

  double G = pars[0];
  // G = 1.;
  double M = pars[1];
  double r_vir = pars[2];

  if (!exp_state->is_static) {
    exp_state->basis->set_coefs(gala_exp::interpolator(t, exp_state->coefs));
  }

  printf("exp_gradient: G_EXP = %g\n", G);
  printf("exp_gradient: m = %g\n", M);
  printf("exp_gradient: r_vir = %g\n", r_vir);
  printf("exp_gradient: q = %g %g %g\n", q[0], q[1], q[2]);
  auto field = exp_state->basis->getFields(q[0] / r_vir, q[1] / r_vir, q[2] / r_vir);

  printf("exp field[6] = %g\n", field[6]);
  printf("exp field[7] = %g\n", field[7]);
  printf("exp field[8] = %g\n", field[8]);

  grad[0] += -G * field[6];
  grad[1] += -G * field[7];
  grad[2] += -G * field[8];
}

// TODO: density units
double exp_density(double t, double *pars, double *q, int n_dim, void* state) {
  gala_exp::State *exp_state = static_cast<gala_exp::State *>(state);

  if (!exp_state->is_static) {
    exp_state->basis->set_coefs(gala_exp::interpolator(t, exp_state->coefs));
  }

  auto field = exp_state->basis->getFields(q[0], q[1], q[2]);

  // return field[2];  // dens
  return NAN;
}

// TODO
void exp_hessian(double t, double *pars, double *q, int n_dim, double *hess, void* state) {
  gala_exp::State *exp_state = static_cast<gala_exp::State *>(state);

  if (!exp_state->is_static) {
    exp_state->basis->set_coefs(gala_exp::interpolator(t, exp_state->coefs));
  }

  auto field = exp_state->basis->getFields(q[0], q[1], q[2]);

  for(int i=0; i<9; i++) {
    hess[i] += NAN;  // TODO: get hessian from EXP
  }
}

}

#endif  // USE_EXP
