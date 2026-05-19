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

   // Recompute the per-curved-face diffuse-reflection inflow contributions
   // from the current distribution. Must be called once per CIS / GSIS
   // iteration before Sweep when there are diffuse curved faces; it is a
   // no-op otherwise. Stored on the solver via mutable state so the
   // existing const Sweep API is preserved.
   void RefreshDiffuseWallInflow(const Distribution &distribution) const;

   // Recompute the per-straight-face specular-reflection inflow
   // contributions from the previous-iteration distribution. The partner
   // direction for cx-flip on a vertical wall is precomputed at
   // construction; horizontal / oblique specular walls are not yet
   // supported. Must be called once per CIS iteration before Sweep when
   // there are straight specular faces; a no-op otherwise.
   void RefreshSpecularInflow(const Distribution &distribution) const;

   // Recompute the per-straight-face periodic inflow contribution from
   // the previous-iteration distribution. The contribution combines a
   // partner-cell coupling (against the periodically-shifted neighbour
   // face mass tensor populated by IntegrationCache) plus a constant
   // delta-temperature thermalising-like piece driven by the boundary
   // condition's `temperature` field, interpreted as ΔT_bc = T_self −
   // T_partner. Must be called once per CIS iteration before Sweep when
   // periodic straight faces are present; a no-op otherwise.
   void RefreshPeriodicInflow(const Distribution &distribution) const;

   bool has_diffuse_curved_face() const { return has_diffuse_curved_face_; }
   bool has_specular_straight_face() const { return has_specular_straight_face_; }
   bool has_periodic_straight_face() const { return has_periodic_straight_face_; }

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
   // Curved boundary face inflow: direction-dependent, applies the BC
   // through the IntegrationCache's CurvedFaceInflowWeight (Thermalizing)
   // or the precomputed diffuse_wall_inflow_ buffer (NonThermalizing /
   // diffuse). Specular and periodic curved-face BCs are not yet
   // implemented.
   void AddCurvedFaceInflow(int angle,
                            int element,
                            int local_face,
                            std::vector<double> &rhs) const;

   // Diffuse reflection state.
   //
   // Curved-face side: curved_face_to_elem_lf_ shares its indexing with
   // IntegrationCache::CurvedFaceIndex; diffuse_wall_inflow_ is indexed as
   // [(angle * N_cf + cf) * dofs + row].
   //
   // Straight-face side: straight_diffuse_to_elem_lf_ enumerates the diffuse
   // straight boundary faces in solver-local order; straight_diffuse_index_
   // maps [elem * 3 + lf] -> that order (or -1 if not a diffuse straight
   // face); diffuse_wall_inflow_straight_ is indexed as
   // [(angle * N_sd + sd) * dofs + row].
   //
   // Both buffers are populated by RefreshDiffuseWallInflow each iteration.
   // mutable because the inflow refresh writes to them from a const driver
   // context.
   std::vector<std::pair<int, int>> curved_face_to_elem_lf_;
   bool has_diffuse_curved_face_ = false;
   mutable std::vector<double> diffuse_wall_inflow_;

   std::vector<std::pair<int, int>> straight_diffuse_to_elem_lf_;
   std::vector<int> straight_diffuse_index_;
   bool has_diffuse_straight_face_ = false;
   mutable std::vector<double> diffuse_wall_inflow_straight_;

   // Straight specular face state. Currently only vertical walls (n_y == 0)
   // are supported — the partner direction for the cx-reflection is
   // computed from the polar-azimuthal layout of AngularQuadrature. The
   // partner table caches angle -> partner-angle once at construction.
   std::vector<std::pair<int, int>> straight_specular_to_elem_lf_;
   std::vector<int> straight_specular_index_;
   bool has_specular_straight_face_ = false;
   std::vector<int> vertical_specular_partner_angle_;  // size = n_angles
   mutable std::vector<double> specular_inflow_straight_;

   // Straight periodic face state. straight_periodic_to_elem_lf_ enumerates
   // the periodic straight boundary faces in solver-local order;
   // straight_periodic_index_ maps [elem * 3 + lf] -> that order (or -1 if
   // not a periodic straight face). straight_periodic_partner_element_
   // caches the partner element per ordered periodic face. The temperature
   // delta stored on the matching BoundaryCondition is interpreted as
   // ΔT_bc = T_self - T_partner.
   std::vector<std::pair<int, int>> straight_periodic_to_elem_lf_;
   std::vector<int> straight_periodic_index_;
   std::vector<int> straight_periodic_partner_element_;
   bool has_periodic_straight_face_ = false;
   mutable std::vector<double> periodic_inflow_straight_;
};

} // namespace callaway
