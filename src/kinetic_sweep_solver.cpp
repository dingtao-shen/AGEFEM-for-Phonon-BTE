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

   if (cache_local_lu_)
   {
      BuildLocalLuCache();
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
   const double four_pi = 4.0 * Pi;
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
                   3.0 * q_dot_c / (four_pi * vg2)) * inv_tau_n;
               rhs[static_cast<std::size_t>(row)] +=
                  (resistive_source + normal_source) *
                  integration_.Mass(element, row, col);
            }
         }

         for (int local_face = 0; local_face < 3; ++local_face)
         {
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
   return flow_.specific_heat * bc.temperature / (4.0 * Pi);
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

   std::vector<double> matrix(static_cast<std::size_t>(dofs * dofs), 0.0);
   for (int angle = 0; angle < quadrature_.size(); ++angle)
   {
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

void KineticSweepSolver::AddBoundaryInflow(int,
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
      case BoundaryType::Periodic:
      case BoundaryType::Symmetry:
      {
         std::ostringstream os;
         os << "Boundary type '" << ToString(bc.type)
            << "' is defined in the interface but is not implemented in the first CIS sweep milestone.";
         throw std::runtime_error(os.str());
      }
   }
}

} // namespace callaway
