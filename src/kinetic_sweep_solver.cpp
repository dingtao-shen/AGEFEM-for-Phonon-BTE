#include "callaway/kinetic_sweep_solver.hpp"

#include "callaway/dense_solver.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>

namespace callaway
{

KineticSweepSolver::KineticSweepSolver(const MeshAdapter &mesh,
                                       const IntegrationCache &integration,
                                       const AngularQuadrature &quadrature,
                                       const SweepOrdering &ordering,
                                       FlowSettings flow,
                                       std::vector<BoundaryCondition> boundary_conditions,
                                       bool cache_local_lu)
   : mesh_(mesh),
     integration_(integration),
     quadrature_(quadrature),
     ordering_(ordering),
     flow_(flow),
     cache_local_lu_(cache_local_lu)
{
   if (mesh_.mesh().GetNE() != integration_.element_count())
   {
      throw std::runtime_error("Mesh and integration cache element counts differ.");
   }
   if (ordering_.angles() != quadrature_.size() ||
       ordering_.elements() != integration_.element_count())
   {
      throw std::runtime_error("Sweep ordering shape does not match the kinetic solver.");
   }
   if (flow_.specific_heat <= 0.0 || flow_.group_velocity <= 0.0 ||
       flow_.tau_r <= 0.0 || flow_.tau_n <= 0.0)
   {
      throw std::runtime_error("Invalid flow parameters for kinetic sweep solver.");
   }

   mesh_.ValidateBoundaryAttributes(boundary_conditions);
   for (const BoundaryCondition &bc : boundary_conditions)
   {
      const auto inserted = boundary_by_attribute_.emplace(bc.physical_id, bc);
      if (!inserted.second)
      {
         std::ostringstream os;
         os << "Duplicate boundary condition for physical_id " << bc.physical_id << ".";
         throw std::runtime_error(os.str());
      }
   }

   // Build the curved-face -> (element, local_face) reverse map and detect
   // diffuse curved faces. Diffuse reflection requires a per-iteration
   // outflow precomputation against the previous distribution.
   const int n_cf = integration_.CurvedFaceCount();
   curved_face_to_elem_lf_.assign(static_cast<std::size_t>(n_cf), {-1, -1});
   for (int e = 0; e < integration_.element_count(); ++e)
   {
      for (int lf = 0; lf < 3; ++lf)
      {
         if (integration_.IsCurvedFace(e, lf))
         {
            const int cf = integration_.CurvedFaceIndex(e, lf);
            curved_face_to_elem_lf_[static_cast<std::size_t>(cf)] = {e, lf};
         }
      }
   }
   for (int cf = 0; cf < n_cf; ++cf)
   {
      const auto [e, lf] = curved_face_to_elem_lf_[static_cast<std::size_t>(cf)];
      if (e < 0) { continue; }
      const int face_id = mesh_.ElementFace(e, lf);
      const FaceData &face = mesh_.Face(face_id);
      const BoundaryCondition &bc = BoundaryForAttribute(face.boundary_attribute);
      if (bc.type == BoundaryType::NonThermalizing)
      {
         has_diffuse_curved_face_ = true;
         break;
      }
   }

   // Build the straight diffuse boundary face mapping. A "straight diffuse"
   // face is a boundary face that is NOT curved (no AGE curved binding) and
   // whose configured BC is NonThermalizing (diffuse reflection).
   straight_diffuse_index_.assign(
      static_cast<std::size_t>(integration_.element_count()) * 3, -1);
   for (int e = 0; e < integration_.element_count(); ++e)
   {
      for (int lf = 0; lf < 3; ++lf)
      {
         if (integration_.IsCurvedFace(e, lf)) { continue; }
         if (mesh_.ElementNeighbor(e, lf) >= 0) { continue; } // interior face
         const int face_id = mesh_.ElementFace(e, lf);
         const FaceData &face = mesh_.Face(face_id);
         if (face.boundary_attribute <= 0) { continue; }
         const BoundaryCondition &bc = BoundaryForAttribute(face.boundary_attribute);
         if (bc.type != BoundaryType::NonThermalizing) { continue; }
         straight_diffuse_index_[static_cast<std::size_t>(e * 3 + lf)] =
            static_cast<int>(straight_diffuse_to_elem_lf_.size());
         straight_diffuse_to_elem_lf_.push_back({e, lf});
      }
   }
   has_diffuse_straight_face_ = !straight_diffuse_to_elem_lf_.empty();

   // Straight specular boundary face mapping (Symmetry BC).
   straight_specular_index_.assign(
      static_cast<std::size_t>(integration_.element_count()) * 3, -1);
   for (int e = 0; e < integration_.element_count(); ++e)
   {
      for (int lf = 0; lf < 3; ++lf)
      {
         if (integration_.IsCurvedFace(e, lf)) { continue; }
         if (mesh_.ElementNeighbor(e, lf) >= 0) { continue; }
         const int face_id = mesh_.ElementFace(e, lf);
         const FaceData &face = mesh_.Face(face_id);
         if (face.boundary_attribute <= 0) { continue; }
         const BoundaryCondition &bc = BoundaryForAttribute(face.boundary_attribute);
         if (bc.type != BoundaryType::Symmetry) { continue; }
         straight_specular_index_[static_cast<std::size_t>(e * 3 + lf)] =
            static_cast<int>(straight_specular_to_elem_lf_.size());
         straight_specular_to_elem_lf_.push_back({e, lf});
      }
   }
   has_specular_straight_face_ = !straight_specular_to_elem_lf_.empty();

   // Straight periodic boundary face mapping (Periodic BC). Each periodic
   // face must have an active $Periodic partner in the mesh adapter; the
   // partner element is cached for sweep-time inflow buffer assembly.
   straight_periodic_index_.assign(
      static_cast<std::size_t>(integration_.element_count()) * 3, -1);
   for (int e = 0; e < integration_.element_count(); ++e)
   {
      for (int lf = 0; lf < 3; ++lf)
      {
         if (integration_.IsCurvedFace(e, lf)) { continue; }
         if (mesh_.ElementNeighbor(e, lf) >= 0) { continue; }
         const int face_id = mesh_.ElementFace(e, lf);
         const FaceData &face = mesh_.Face(face_id);
         if (face.boundary_attribute <= 0) { continue; }
         const BoundaryCondition &bc = BoundaryForAttribute(face.boundary_attribute);
         if (bc.type != BoundaryType::Periodic) { continue; }
         const PeriodicFacePair *pair = mesh_.PeriodicPartner(face_id);
         if (pair == nullptr || pair->partner_element < 0)
         {
            std::ostringstream os;
            os << "Periodic boundary tag " << face.boundary_attribute
               << " is missing a paired partner face. Check the Gmsh $Periodic "
               << "section: the slave/master node lists must cover this face.";
            throw std::runtime_error(os.str());
         }
         straight_periodic_index_[static_cast<std::size_t>(e * 3 + lf)] =
            static_cast<int>(straight_periodic_to_elem_lf_.size());
         straight_periodic_to_elem_lf_.push_back({e, lf});
         straight_periodic_partner_element_.push_back(pair->partner_element);
      }
   }
   has_periodic_straight_face_ = !straight_periodic_to_elem_lf_.empty();

   // For each angle, precompute the cx-flip partner index used by vertical-
   // wall specular reflection. The mapping depends on the angular mode:
   //   3D: partner is the polar-reflected index in the same azimuth.
   //   2D: partner is the phi-reflected index inside its half-circle.
   // AngularQuadrature::CxFlipPartner encapsulates both.
   if (has_specular_straight_face_)
   {
      const int n_total = quadrature_.size();
      vertical_specular_partner_angle_.assign(static_cast<std::size_t>(n_total), -1);
      for (int a = 0; a < n_total; ++a)
      {
         vertical_specular_partner_angle_[static_cast<std::size_t>(a)] =
            quadrature_.CxFlipPartner(a);
      }
   }

   if (cache_local_lu_)
   {
      BuildLocalLuCache();
   }
}

void KineticSweepSolver::RefreshDiffuseWallInflow(const Distribution &distribution) const
{
   if (!has_diffuse_curved_face_ && !has_diffuse_straight_face_) { return; }
   const int n_angles = quadrature_.size();
   const int n_cf = integration_.CurvedFaceCount();
   const int n_dofs = integration_.dofs();

   // Compute the discrete inflow integral I(n) = sum_{a: s_a . n < 0} w_a |s_a . n|
   // for a given outward normal. In the continuous 3D limit I(n) -> pi, but for
   // coarse angular quadratures the analytic pi causes the diffuse BC to inject
   // more energy than was reflected, breaking the bound T <= max(T_BC). Using
   // the discrete I(n) preserves energy conservation at the wall to machine
   // precision regardless of angular resolution.
   auto discrete_inflow_integral = [&](double nx, double ny) -> double {
      double sum = 0.0;
      for (int a = 0; a < n_angles; ++a)
      {
         const Direction &d = quadrature_[a];
         const double sn = d.cx * nx + d.cy * ny;
         if (sn < 0.0) { sum += d.weight * (-sn); }
      }
      return sum;
   };

   diffuse_wall_inflow_.assign(
      static_cast<std::size_t>(n_angles) * static_cast<std::size_t>(n_cf) *
      static_cast<std::size_t>(n_dofs), 0.0);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
   for (int cf = 0; cf < n_cf; ++cf)
   {
      const auto [elem, lf] = curved_face_to_elem_lf_[static_cast<std::size_t>(cf)];
      if (elem < 0) { continue; }
      const int face_id = mesh_.ElementFace(elem, lf);
      const FaceData &face = mesh_.Face(face_id);
      const BoundaryCondition &bc = BoundaryForAttribute(face.boundary_attribute);
      if (bc.type != BoundaryType::NonThermalizing) { continue; }

      const auto &rec = integration_.CurvedFaceData(elem, lf);
      const std::size_t n_qpts = rec.points.size();

      for (std::size_t q = 0; q < n_qpts; ++q)
      {
         const double nx = rec.normals[q][0];
         const double ny = rec.normals[q][1];
         const std::vector<double> &basis_q = rec.basis[q];
         const double w_q = rec.weights[q];

         // F_out(x_q) = sum over directions with s . n > 0 of
         //             weight * (s . n) * e(s, x_q).
         double F_out = 0.0;
         for (int a = 0; a < n_angles; ++a)
         {
            const Direction &d = quadrature_[a];
            const double sn = d.cx * nx + d.cy * ny;
            if (sn <= 0.0) { continue; }
            double e_at_q = 0.0;
            for (int dof = 0; dof < n_dofs; ++dof)
            {
               e_at_q += basis_q[static_cast<std::size_t>(dof)] *
                         distribution(a, elem, dof);
            }
            F_out += d.weight * sn * e_at_q;
         }
         const double I_n = discrete_inflow_integral(nx, ny);
         const double e_star = (I_n > 0.0) ? (F_out / I_n) : 0.0;

         // Per inflow angle a (s_a . n_q < 0): contribution to rhs[row] is
         // (-s_a . n_q) * basis[row] * e_star * w_q.
         for (int a = 0; a < n_angles; ++a)
         {
            const Direction &d = quadrature_[a];
            const double sn_a = d.cx * nx + d.cy * ny;
            if (sn_a >= 0.0) { continue; }
            const double prefactor = -sn_a * e_star * w_q;
            const std::size_t base =
               (static_cast<std::size_t>(a) * static_cast<std::size_t>(n_cf) +
                static_cast<std::size_t>(cf)) *
               static_cast<std::size_t>(n_dofs);
            for (int row = 0; row < n_dofs; ++row)
            {
               diffuse_wall_inflow_[base + static_cast<std::size_t>(row)] +=
                  prefactor * basis_q[static_cast<std::size_t>(row)];
            }
         }
      }
   }

   // Straight diffuse faces. Constant outward normal lets us factor the
   // F_out integral as a product of ElementFaceMass and the angular sum:
   //   F_total[row] = (1/pi) sum_{a':s_a'.n>0} weight_a' (s_a'.n)
   //                  sum_dof ElementFaceMass(elem,lf,row,dof) dist(a',elem,dof)
   // and the per-inflow contribution for angle a is -(s_a.n) * F_total[row].
   if (has_diffuse_straight_face_)
   {
      const int n_sd = static_cast<int>(straight_diffuse_to_elem_lf_.size());
      diffuse_wall_inflow_straight_.assign(
         static_cast<std::size_t>(n_angles) * static_cast<std::size_t>(n_sd) *
         static_cast<std::size_t>(n_dofs), 0.0);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
      for (int sd = 0; sd < n_sd; ++sd)
      {
         const auto [elem, lf] = straight_diffuse_to_elem_lf_[static_cast<std::size_t>(sd)];
         const auto normal = integration_.OutwardNormal(elem, lf);
         const double nx = normal[0];
         const double ny = normal[1];
         const double I_n = discrete_inflow_integral(nx, ny);
         const double inv_I_n = (I_n > 0.0) ? (1.0 / I_n) : 0.0;

         // F_total[row] aggregated from outflow directions.
         std::vector<double> F_total(static_cast<std::size_t>(n_dofs), 0.0);
         for (int a2 = 0; a2 < n_angles; ++a2)
         {
            const Direction &d2 = quadrature_[a2];
            const double sn2 = d2.cx * nx + d2.cy * ny;
            if (sn2 <= 0.0) { continue; }
            const double angular_factor = inv_I_n * d2.weight * sn2;
            for (int row = 0; row < n_dofs; ++row)
            {
               double proj = 0.0;
               for (int dof = 0; dof < n_dofs; ++dof)
               {
                  proj += integration_.ElementFaceMass(elem, lf, row, dof) *
                          distribution(a2, elem, dof);
               }
               F_total[static_cast<std::size_t>(row)] += angular_factor * proj;
            }
         }

         // Spread to each inflow angle.
         for (int a = 0; a < n_angles; ++a)
         {
            const Direction &d = quadrature_[a];
            const double sn_a = d.cx * nx + d.cy * ny;
            if (sn_a >= 0.0) { continue; }
            const std::size_t base =
               (static_cast<std::size_t>(a) * static_cast<std::size_t>(n_sd) +
                static_cast<std::size_t>(sd)) *
               static_cast<std::size_t>(n_dofs);
            for (int row = 0; row < n_dofs; ++row)
            {
               diffuse_wall_inflow_straight_[base + static_cast<std::size_t>(row)] =
                  -sn_a * F_total[static_cast<std::size_t>(row)];
            }
         }
      }
   }
}

void KineticSweepSolver::RefreshSpecularInflow(const Distribution &distribution) const
{
   if (!has_specular_straight_face_) { return; }
   const int n_angles = quadrature_.size();
   const int n_sp = static_cast<int>(straight_specular_to_elem_lf_.size());
   const int n_dofs = integration_.dofs();
   specular_inflow_straight_.assign(
      static_cast<std::size_t>(n_angles) * static_cast<std::size_t>(n_sp) *
      static_cast<std::size_t>(n_dofs), 0.0);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
   for (int sp = 0; sp < n_sp; ++sp)
   {
      const auto [elem, lf] = straight_specular_to_elem_lf_[static_cast<std::size_t>(sp)];
      const auto normal = integration_.OutwardNormal(elem, lf);
      const double nx = normal[0];
      const double ny = normal[1];
      // Only vertical walls (n_y ~= 0) are supported; horizontal / oblique
      // walls would need their own partner-direction tables. Leave the buffer
      // zero for unsupported orientations — the sweep call below will throw
      // a clear error on first encounter.
      if (std::abs(ny) > 1.0e-10) { continue; }

      for (int a = 0; a < n_angles; ++a)
      {
         const Direction &d = quadrature_[a];
         const double sn = d.cx * nx + d.cy * ny;
         if (sn >= 0.0) { continue; }
         const int partner =
            vertical_specular_partner_angle_[static_cast<std::size_t>(a)];
         const double inflow_speed = -sn;
         const std::size_t base =
            (static_cast<std::size_t>(a) * static_cast<std::size_t>(n_sp) +
             static_cast<std::size_t>(sp)) *
            static_cast<std::size_t>(n_dofs);
         for (int row = 0; row < n_dofs; ++row)
         {
            double proj = 0.0;
            for (int dof = 0; dof < n_dofs; ++dof)
            {
               proj += integration_.ElementFaceMass(elem, lf, row, dof) *
                       distribution(partner, elem, dof);
            }
            specular_inflow_straight_[base + static_cast<std::size_t>(row)] =
               inflow_speed * proj;
         }
      }
   }
}

void KineticSweepSolver::RefreshPeriodicInflow(const Distribution &distribution) const
{
   if (!has_periodic_straight_face_) { return; }
   const int n_angles = quadrature_.size();
   const int n_pp = static_cast<int>(straight_periodic_to_elem_lf_.size());
   const int n_dofs = integration_.dofs();
   periodic_inflow_straight_.assign(
      static_cast<std::size_t>(n_angles) * static_cast<std::size_t>(n_pp) *
      static_cast<std::size_t>(n_dofs), 0.0);

   const double inv_four_pi_cv =
      flow_.specific_heat / quadrature_.equilibrium_normalization();

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
   for (int pp = 0; pp < n_pp; ++pp)
   {
      const auto [elem, lf] = straight_periodic_to_elem_lf_[static_cast<std::size_t>(pp)];
      const int partner =
         straight_periodic_partner_element_[static_cast<std::size_t>(pp)];
      const auto normal = integration_.OutwardNormal(elem, lf);
      const double nx = normal[0];
      const double ny = normal[1];
      const int face_id = mesh_.ElementFace(elem, lf);
      const FaceData &face = mesh_.Face(face_id);
      const BoundaryCondition &bc = BoundaryForAttribute(face.boundary_attribute);
      // ΔT_bc = T_self - T_partner: inflow energy density is shifted by
      // Cv * ΔT_bc / (4π) relative to the partner cell value when crossing
      // the periodic interface.
      const double delta_offset = inv_four_pi_cv * bc.temperature;

      for (int a = 0; a < n_angles; ++a)
      {
         const Direction &d = quadrature_[a];
         const double sn = d.cx * nx + d.cy * ny;
         if (sn >= 0.0) { continue; }
         const double inflow_speed = -sn;
         const std::size_t base =
            (static_cast<std::size_t>(a) * static_cast<std::size_t>(n_pp) +
             static_cast<std::size_t>(pp)) *
            static_cast<std::size_t>(n_dofs);
         for (int row = 0; row < n_dofs; ++row)
         {
            double partner_proj = 0.0;
            for (int col = 0; col < n_dofs; ++col)
            {
               partner_proj += integration_.NeighborFaceMass(elem, lf, row, col) *
                               distribution(a, partner, col);
            }
            periodic_inflow_straight_[base + static_cast<std::size_t>(row)] =
               inflow_speed * partner_proj +
               inflow_speed * delta_offset *
                  integration_.ElementFaceIntegral(elem, lf, row);
         }
      }
   }
}

void KineticSweepSolver::Sweep(const MomentFields &moments,
                               Distribution &distribution) const
{
   if (moments.elements() != integration_.element_count() ||
       moments.dofs() != integration_.dofs())
   {
      throw std::runtime_error("Moment fields do not match the kinetic solver shape.");
   }
   if (distribution.angles() != quadrature_.size() ||
       distribution.elements() != integration_.element_count() ||
       distribution.dofs() != integration_.dofs())
   {
      throw std::runtime_error("Distribution does not match the kinetic solver shape.");
   }

   const int dofs = integration_.dofs();
   const double inv_tau_c = 1.0 / flow_.tau_combined();
   const double inv_tau_r = 1.0 / flow_.tau_r;
   const double inv_tau_n = 1.0 / flow_.tau_n;
   const double four_pi = quadrature_.equilibrium_normalization();
   const double moment_factor = quadrature_.moment_factor();
   const double vg2 = flow_.group_velocity * flow_.group_velocity;

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
   for (int angle = 0; angle < quadrature_.size(); ++angle)
   {
      const Direction &direction = quadrature_[angle];
      std::vector<double> matrix(cache_local_lu_ ? 0 : static_cast<std::size_t>(dofs * dofs), 0.0);
      std::vector<double> rhs(static_cast<std::size_t>(dofs), 0.0);

      for (const int element : ordering_.Order(angle))
      {
         std::fill(rhs.begin(), rhs.end(), 0.0);

         if (!cache_local_lu_)
         {
            AssembleLocalMatrix(angle, element, matrix);
         }

         for (int row = 0; row < dofs; ++row)
         {
            for (int col = 0; col < dofs; ++col)
            {
               const double t = moments.TemperatureDof(element, col);
               const double q_dot_c =
                  moments.HeatFluxXDof(element, col) * direction.cx +
                  moments.HeatFluxYDof(element, col) * direction.cy;
               const double resistive_source =
                  flow_.specific_heat * t / four_pi * inv_tau_r;
               const double normal_source =
                  (flow_.specific_heat * t / four_pi +
                   moment_factor * q_dot_c / (four_pi * vg2)) * inv_tau_n;
               rhs[static_cast<std::size_t>(row)] +=
                  (resistive_source + normal_source) *
                  integration_.Mass(element, row, col);
            }
         }

         for (int local_face = 0; local_face < 3; ++local_face)
         {
            if (integration_.IsCurvedFace(element, local_face))
            {
               // Curved boundary face — apply BC inflow contribution. The
               // outflow part already entered the local matrix via
               // CurvedFaceMatrix in AssembleLocalMatrix.
               AddCurvedFaceInflow(angle, element, local_face, rhs);
               continue;
            }
            const auto normal = integration_.OutwardNormal(element, local_face);
            const double speed = direction.cx * normal[0] + direction.cy * normal[1];
            if (speed >= 0.0) { continue; }

            const double inflow_speed = -speed;
            const int neighbor = mesh_.ElementNeighbor(element, local_face);
            if (neighbor >= 0)
            {
               for (int row = 0; row < dofs; ++row)
               {
                  for (int col = 0; col < dofs; ++col)
                  {
                     rhs[static_cast<std::size_t>(row)] +=
                        inflow_speed *
                        integration_.NeighborFaceMass(element, local_face, row, col) *
                        distribution(angle, neighbor, col);
                  }
               }
            }
            else
            {
               AddBoundaryInflow(angle, element, local_face, inflow_speed, rhs);
            }
         }

         if (cache_local_lu_)
         {
            SolveDenseFactoredSystem(LocalLu(angle, element), LocalPivots(angle, element), dofs, rhs);
         }
         else
         {
            SolveDenseLinearSystem(matrix, rhs, dofs);
         }

         for (int dof = 0; dof < dofs; ++dof)
         {
            distribution(angle, element, dof) = rhs[static_cast<std::size_t>(dof)];
         }
      }
   }
}

const BoundaryCondition &KineticSweepSolver::BoundaryForAttribute(int attribute) const
{
   const auto it = boundary_by_attribute_.find(attribute);
   if (it == boundary_by_attribute_.end())
   {
      std::ostringstream os;
      os << "No boundary condition is configured for mesh boundary attribute " << attribute << ".";
      throw std::runtime_error(os.str());
   }
   return it->second;
}

double KineticSweepSolver::ThermalizingInflowValue(const BoundaryCondition &bc) const
{
   return flow_.specific_heat * bc.temperature / quadrature_.equilibrium_normalization();
}

void KineticSweepSolver::AssembleLocalMatrix(int angle,
                                             int element,
                                             std::vector<double> &matrix) const
{
   const int dofs = integration_.dofs();
   if (static_cast<int>(matrix.size()) != dofs * dofs)
   {
      matrix.assign(static_cast<std::size_t>(dofs * dofs), 0.0);
   }
   else
   {
      std::fill(matrix.begin(), matrix.end(), 0.0);
   }

   const Direction &direction = quadrature_[angle];
   const double inv_tau_c = 1.0 / flow_.tau_combined();

   for (int row = 0; row < dofs; ++row)
   {
      for (int col = 0; col < dofs; ++col)
      {
         double value =
            inv_tau_c * integration_.Mass(element, row, col) -
            direction.cx * integration_.GradX(element, row, col) -
            direction.cy * integration_.GradY(element, row, col);

         for (int local_face = 0; local_face < 3; ++local_face)
         {
            if (integration_.IsCurvedFace(element, local_face))
            {
               // Curved face: integral of max(s . n, 0) * phi_row * phi_col
               // is direction-dependent and already folded in by the cache.
               value += integration_.CurvedFaceMatrix(angle, element, local_face, row, col);
               continue;
            }
            const auto normal = integration_.OutwardNormal(element, local_face);
            const double speed = direction.cx * normal[0] + direction.cy * normal[1];
            if (speed > 0.0)
            {
               value += speed * integration_.ElementFaceMass(element, local_face, row, col);
            }
         }

         matrix[static_cast<std::size_t>(row * dofs + col)] = value;
      }
   }
}

void KineticSweepSolver::BuildLocalLuCache()
{
   const int dofs = integration_.dofs();
   const std::size_t system_count =
      static_cast<std::size_t>(quadrature_.size() * integration_.element_count());
   local_lu_cache_.assign(system_count * static_cast<std::size_t>(dofs * dofs), 0.0);
   local_pivot_cache_.assign(system_count * static_cast<std::size_t>(dofs), 0);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
   for (int angle = 0; angle < quadrature_.size(); ++angle)
   {
      std::vector<double> matrix(static_cast<std::size_t>(dofs * dofs), 0.0);
      for (int element = 0; element < integration_.element_count(); ++element)
      {
         AssembleLocalMatrix(angle, element, matrix);
         const std::size_t index = LocalSystemIndex(angle, element);
         const std::size_t matrix_offset = index * static_cast<std::size_t>(dofs * dofs);
         const std::size_t pivot_offset = index * static_cast<std::size_t>(dofs);
         std::copy(matrix.begin(), matrix.end(), local_lu_cache_.begin() + matrix_offset);
         FactorDenseMatrixInPlace(local_lu_cache_.data() + matrix_offset,
                                  local_pivot_cache_.data() + pivot_offset,
                                  dofs);
      }
   }
}

std::size_t KineticSweepSolver::LocalSystemIndex(int angle, int element) const
{
   return static_cast<std::size_t>(angle * integration_.element_count() + element);
}

const double *KineticSweepSolver::LocalLu(int angle, int element) const
{
   const int dofs = integration_.dofs();
   return local_lu_cache_.data() +
          LocalSystemIndex(angle, element) * static_cast<std::size_t>(dofs * dofs);
}

const int *KineticSweepSolver::LocalPivots(int angle, int element) const
{
   const int dofs = integration_.dofs();
   return local_pivot_cache_.data() +
          LocalSystemIndex(angle, element) * static_cast<std::size_t>(dofs);
}

void KineticSweepSolver::AddCurvedFaceInflow(int angle,
                                             int element,
                                             int local_face,
                                             std::vector<double> &rhs) const
{
   const int face_id = mesh_.ElementFace(element, local_face);
   const FaceData &face = mesh_.Face(face_id);
   const BoundaryCondition &bc = BoundaryForAttribute(face.boundary_attribute);
   switch (bc.type)
   {
      case BoundaryType::Thermalizing:
      {
         const double value = ThermalizingInflowValue(bc);
         for (int row = 0; row < integration_.dofs(); ++row)
         {
            // CurvedFaceInflowWeight = integral(min(s . n, 0) * phi_row),
            // which is <= 0; subtract to add the BC inflow to the RHS.
            rhs[static_cast<std::size_t>(row)] -=
               value * integration_.CurvedFaceInflowWeight(angle, element, local_face, row);
         }
         break;
      }
      case BoundaryType::NonThermalizing:
      {
         // Diffuse reflection: use the precomputed wall-inflow contribution
         // from RefreshDiffuseWallInflow (called by the iteration driver).
         const int cf_idx = integration_.CurvedFaceIndex(element, local_face);
         const int n_cf = integration_.CurvedFaceCount();
         const int n_dofs = integration_.dofs();
         const std::size_t base =
            (static_cast<std::size_t>(angle) * static_cast<std::size_t>(n_cf) +
             static_cast<std::size_t>(cf_idx)) *
            static_cast<std::size_t>(n_dofs);
         for (int row = 0; row < n_dofs; ++row)
         {
            rhs[static_cast<std::size_t>(row)] +=
               diffuse_wall_inflow_[base + static_cast<std::size_t>(row)];
         }
         break;
      }
      case BoundaryType::Periodic:
      case BoundaryType::Symmetry:
      {
         std::ostringstream os;
         os << "KineticSweepSolver: curved-face boundary type '" << ToString(bc.type)
            << "' is not implemented in this milestone.";
         throw std::runtime_error(os.str());
      }
   }
}

void KineticSweepSolver::AddBoundaryInflow(int angle,
                                           int element,
                                           int local_face,
                                           double inflow_speed,
                                           std::vector<double> &rhs) const
{
   const int face_id = mesh_.ElementFace(element, local_face);
   const FaceData &face = mesh_.Face(face_id);
   const BoundaryCondition &bc = BoundaryForAttribute(face.boundary_attribute);

   switch (bc.type)
   {
      case BoundaryType::Thermalizing:
      {
         const double value = ThermalizingInflowValue(bc);
         for (int row = 0; row < integration_.dofs(); ++row)
         {
            rhs[static_cast<std::size_t>(row)] +=
               inflow_speed * value *
               integration_.ElementFaceIntegral(element, local_face, row);
         }
         break;
      }
      case BoundaryType::NonThermalizing:
      {
         // Diffuse reflection on a straight boundary face: read the
         // precomputed inflow contribution from RefreshDiffuseWallInflow.
         const int sd_idx =
            straight_diffuse_index_[static_cast<std::size_t>(element * 3 + local_face)];
         if (sd_idx < 0)
         {
            throw std::runtime_error(
               "KineticSweepSolver: NonThermalizing straight boundary face "
               "was not registered for diffuse reflection during construction.");
         }
         const int n_sd = static_cast<int>(straight_diffuse_to_elem_lf_.size());
         const int n_dofs = integration_.dofs();
         const std::size_t base =
            (static_cast<std::size_t>(angle) * static_cast<std::size_t>(n_sd) +
             static_cast<std::size_t>(sd_idx)) *
            static_cast<std::size_t>(n_dofs);
         for (int row = 0; row < n_dofs; ++row)
         {
            rhs[static_cast<std::size_t>(row)] +=
               diffuse_wall_inflow_straight_[base + static_cast<std::size_t>(row)];
         }
         break;
      }
      case BoundaryType::Symmetry:
      {
         // Specular reflection. The partner angle (cx-flip for vertical
         // walls) was identified at construction; the precomputed buffer
         // already carries inflow_speed * <face mass projection of the
         // partner-direction distribution>.
         const int sp =
            straight_specular_index_[static_cast<std::size_t>(element * 3 + local_face)];
         if (sp < 0)
         {
            throw std::runtime_error(
               "KineticSweepSolver: Symmetry straight boundary face was not "
               "registered for specular reflection during construction.");
         }
         const auto normal = integration_.OutwardNormal(element, local_face);
         if (std::abs(normal[1]) > 1.0e-10)
         {
            throw std::runtime_error(
               "KineticSweepSolver: specular reflection on non-vertical walls "
               "is not yet implemented.");
         }
         const int n_sp = static_cast<int>(straight_specular_to_elem_lf_.size());
         const int n_dofs = integration_.dofs();
         const std::size_t base =
            (static_cast<std::size_t>(angle) * static_cast<std::size_t>(n_sp) +
             static_cast<std::size_t>(sp)) *
            static_cast<std::size_t>(n_dofs);
         for (int row = 0; row < n_dofs; ++row)
         {
            rhs[static_cast<std::size_t>(row)] +=
               specular_inflow_straight_[base + static_cast<std::size_t>(row)];
         }
         break;
      }
      case BoundaryType::Periodic:
      {
         // Periodic BC. RefreshPeriodicInflow precomputed
         //   inflow_speed * [ NFM × dist_partner + (Cv ΔT_bc / 4π) EFI ]
         // per (angle, periodic face, row); reading the buffer here folds
         // both the partner-cell coupling and the ΔT thermalising-like
         // delta into the local RHS.
         const int pp =
            straight_periodic_index_[static_cast<std::size_t>(element * 3 + local_face)];
         if (pp < 0)
         {
            throw std::runtime_error(
               "KineticSweepSolver: Periodic straight boundary face was not "
               "registered for periodic coupling during construction.");
         }
         const int n_pp = static_cast<int>(straight_periodic_to_elem_lf_.size());
         const int n_dofs = integration_.dofs();
         const std::size_t base =
            (static_cast<std::size_t>(angle) * static_cast<std::size_t>(n_pp) +
             static_cast<std::size_t>(pp)) *
            static_cast<std::size_t>(n_dofs);
         for (int row = 0; row < n_dofs; ++row)
         {
            rhs[static_cast<std::size_t>(row)] +=
               periodic_inflow_straight_[base + static_cast<std::size_t>(row)];
         }
         break;
      }
   }
   (void) inflow_speed; // already consumed in the thermalizing branch
}

} // namespace callaway
