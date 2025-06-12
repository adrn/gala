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
  int stride, double tmin, double tmax, int snapshot_index)
{
  YAML::Node yaml = YAML::LoadFile(std::string(config_fn));

  auto basis = BasisClasses::Basis::factory(yaml);
  auto coefs = CoefClasses::Coefs::factory(coeffile, stride, tmin, tmax);

  if (!basis) {
    throw std::runtime_error("Failed to create basis from YAML configuration");
  }
  if (!coefs) {
    throw std::runtime_error("Failed to create coefficients from file");
  }

  // Handle snapshot indexing and load initial coefficients
  CoefClasses::CoefStrPtr initial_coef;
  if (snapshot_index >= 0) {
    const auto& times = coefs->Times();
    if (snapshot_index >= times.size()) {
      std::ostringstream error_msg;
      error_msg << "Invalid snapshot_index: " << snapshot_index
                << ". Valid indices are in [0," << (times.size() - 1) << "]"
                << " (times [" << times.front() << ", " << times.back() << "]";
      throw std::runtime_error(error_msg.str());
    }
    double test_time = times[snapshot_index];
    initial_coef = gala_exp::interpolator(test_time, coefs);
  } else {
    const auto& times = coefs->Times();
    double test_time = times.empty() ? 0.0 : times[0];
    initial_coef = gala_exp::interpolator(test_time, coefs);
  }

  if (!initial_coef) {
    throw std::runtime_error("Failed to interpolate initial coefficients");
  }

  // Set the initial coefficients in the basis
  basis->set_coefs(initial_coef);

  // Determine if this is a static potential (single time step)
  bool is_static = false;
  if (snapshot_index >= 0) {
    const auto& times = coefs->Times();
    tmin = times[snapshot_index];
    tmax = tmin;
    is_static = true;
  } else {
    is_static = (tmax == tmin);
  }

  return { basis, coefs, initial_coef, is_static };
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

  // Handle degenerate case where it1 == it2 (single time entry)
  if (it1 == it2 || *it1 == *it2) {
    return coefs->getCoefStruct(*it1);
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
  gala_exp::State *exp_state = static_cast<gala_exp::State *>(state);

  // Add safety checks
  if (!exp_state || !exp_state->basis) {
    return std::numeric_limits<double>::quiet_NaN();
  }

  if (!exp_state->is_static) {
    auto new_coef = gala_exp::interpolator(t, exp_state->coefs);
    exp_state->basis->set_coefs(new_coef);
    // Update the stored coefficient to keep it alive
    exp_state->current_coef = new_coef;
  }

  auto field = exp_state->basis->getFields(q[0], q[1], q[2]);
  return field[5];
}

void exp_gradient(double t, double *pars, double *q, int n_dim, double *grad, void* state){
  gala_exp::State *exp_state = static_cast<gala_exp::State *>(state);

  // Add safety checks
  if (!exp_state || !exp_state->basis) {
    grad[0] = grad[1] = grad[2] = std::numeric_limits<double>::quiet_NaN();
    return;
  }

  if (!exp_state->is_static) {
    auto new_coef = gala_exp::interpolator(t, exp_state->coefs);
    exp_state->basis->set_coefs(new_coef);
    exp_state->current_coef = new_coef;
  }

  // TODO: ask Martin/Mike for a way to compute only the force/acceleration - we're wasting
  // computation time here by computing all fields
  auto field = exp_state->basis->getFields(q[0], q[1], q[2]);

  grad[0] += -field[6];
  grad[1] += -field[7];
  grad[2] += -field[8];
}

double exp_density(double t, double *pars, double *q, int n_dim, void* state) {
  gala_exp::State *exp_state = static_cast<gala_exp::State *>(state);

  // Add safety checks
  if (!exp_state || !exp_state->basis) {
    return std::numeric_limits<double>::quiet_NaN();
  }

  if (!exp_state->is_static) {
    auto new_coef = gala_exp::interpolator(t, exp_state->coefs);
    exp_state->basis->set_coefs(new_coef);
    exp_state->current_coef = new_coef;
  }

  // TODO: ask Martin/Mike for a way to compute only the density - we're wasting
  // computation time here by computing all fields
  auto field = exp_state->basis->getFields(q[0], q[1], q[2]);

  return field[2];
}

// TODO: No hessian available in EXP yet
// void exp_hessian(double t, double *pars, double *q, int n_dim, double *hess, void* state) {
//   gala_exp::State *exp_state = static_cast<gala_exp::State *>(state);

//   if (!exp_state->is_static) {
//     exp_state->basis->set_coefs(gala_exp::interpolator(t, exp_state->coefs));
//   }

//   auto field = exp_state->basis->getFields(q[0], q[1], q[2]);

//   for(int i=0; i<9; i++) {
//     hess[i] += NAN;  // TODO: get hessian from EXP
//   }
// }

}

#endif  // USE_EXP
