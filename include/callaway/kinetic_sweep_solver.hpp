#pragma once

#include "callaway/angular_quadrature.hpp"
#include "callaway/boundary.hpp"
#include "callaway/config.hpp"
#include "callaway/distribution.hpp"
#include "callaway/integration_cache.hpp"
#include "callaway/mesh_adapter.hpp"
#include "callaway/sweep_ordering.hpp"

#include <unordered_map>
#include <vector>

namespace callaway
{

class KineticSweepSolver
{
public:
   KineticSweepSolver(const MeshAdapter &mesh,
                      const IntegrationCache &integration,
                      const AngularQuadrature &quadrature,
                      const SweepOrdering &ordering,
                      FlowSettings flow,
                      std::vector<BoundaryCondition> boundary_conditions,
                      bool cache_local_lu = true);

   void Sweep(const MomentFields &moments, Distribution &distribution) const;

private:
   const MeshAdapter &mesh_;
   const IntegrationCache &integration_;
   const AngularQuadrature &quadrature_;
   const SweepOrdering &ordering_;
   FlowSettings flow_;
   std::unordered_map<int, BoundaryCondition> boundary_by_attribute_;
   bool cache_local_lu_ = true;
   std::vector<double> local_lu_cache_;
   std::vector<int> local_pivot_cache_;

   const BoundaryCondition &BoundaryForAttribute(int attribute) const;
   double ThermalizingInflowValue(const BoundaryCondition &bc) const;
   void AssembleLocalMatrix(int angle, int element, std::vector<double> &matrix) const;
   void BuildLocalLuCache();
   std::size_t LocalSystemIndex(int angle, int element) const;
   const double *LocalLu(int angle, int element) const;
   const int *LocalPivots(int angle, int element) const;
   void AddBoundaryInflow(int angle,
                          int element,
                          int local_face,
                          double inflow_speed,
                          std::vector<double> &rhs) const;
};

} // namespace callaway
