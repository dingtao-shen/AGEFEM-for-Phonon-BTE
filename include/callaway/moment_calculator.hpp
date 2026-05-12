#pragma once

#include "callaway/angular_quadrature.hpp"
#include "callaway/distribution.hpp"
#include "callaway/integration_cache.hpp"

namespace callaway
{

class MomentCalculator
{
public:
   static MomentFields Compute(const Distribution &distribution,
                               const AngularQuadrature &quadrature,
                               const IntegrationCache &integration,
                               double specific_heat);
};

} // namespace callaway
