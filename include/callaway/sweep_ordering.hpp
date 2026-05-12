#pragma once

#include "callaway/angular_quadrature.hpp"
#include "callaway/integration_cache.hpp"
#include "callaway/mesh_adapter.hpp"

#include <vector>

namespace callaway
{

class SweepOrdering
{
public:
   SweepOrdering(const MeshAdapter &mesh,
                 const IntegrationCache &integration,
                 const AngularQuadrature &quadrature);

   int angles() const { return angles_; }
   int elements() const { return elements_; }

   const std::vector<int> &Order(int angle) const { return orders_.at(angle); }
   int Position(int angle, int element) const;

private:
   int angles_ = 0;
   int elements_ = 0;
   std::vector<std::vector<int>> orders_;
   std::vector<int> positions_;
};

} // namespace callaway
