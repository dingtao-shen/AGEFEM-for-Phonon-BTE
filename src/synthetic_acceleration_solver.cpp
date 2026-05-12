#include "callaway/synthetic_acceleration_solver.hpp"

#include "callaway/dense_solver.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <unordered_map>

namespace callaway
{

MacroState::MacroState(int elements, int dofs)
   : elements_(elements),
     dofs_(dofs),
     values_(static_cast<std::size_t>(elements * MacroComponentCount * dofs), 0.0)
{
   if (elements_ <= 0 || dofs_ <= 0)
   {
      throw std::runtime_error("MacroState dimensions must be positive.");
   }
}

double &MacroState::operator()(MacroComponent component, int element, int dof)
{
   return values_.at(static_cast<std::size_t>(Index(component, element, dof)));
}

double MacroState::operator()(MacroComponent component, int element, int dof) const
{
   return values_.at(static_cast<std::size_t>(Index(component, element, dof)));
}

void MacroState::Fill(double value)
{
   std::fill(values_.begin(), values_.end(), value);
}

double MacroState::MaxAbs() const
{
   double value = 0.0;
   for (const double entry : values_)
   {
      value = std::max(value, std::abs(entry));
   }
   return value;
}

int MacroState::Index(MacroComponent component, int element, int dof) const
{
   return (element * MacroComponentCount + static_cast<int>(component)) * dofs_ + dof;
}

SyntheticAccelerationSolver::SyntheticAccelerationSolver(const IntegrationCache &integration,
                                                         const AngularQuadrature &quadrature,
                                                         FlowSettings flow)
   : integration_(integration),
     quadrature_(quadrature),
     flow_(flow)
{
   if (flow_.specific_heat <= 0.0 || flow_.group_velocity <= 0.0 ||
       flow_.tau_r <= 0.0 || flow_.tau_n <= 0.0)
   {
      throw std::runtime_error("Invalid flow parameters for synthetic acceleration.");
   }
   BuildLocalMacroLuCache();
}

int SyntheticAccelerationSolver::GlobalTraceDof(int face, int trace_component, int face_dof) const
{
   return face * trace_unknowns_per_face() +
          trace_component * integration_.face_dofs() + face_dof;
}

MacroState SyntheticAccelerationSolver::ComputeHighOrderSource(const Distribution &distribution) const
{
   if (distribution.angles() != quadrature_.size() ||
       distribution.elements() != integration_.element_count() ||
       distribution.dofs() != integration_.dofs())
   {
      throw std::runtime_error("Distribution does not match synthetic acceleration inputs.");
   }

   const int dofs = integration_.dofs();
   const int unknowns = local_unknowns();
   const double tau_c = flow_.tau_combined();
   MacroState source(integration_.element_count(), dofs);
   std::vector<double> local_source(static_cast<std::size_t>(unknowns), 0.0);

   for (int element = 0; element < integration_.element_count(); ++element)
   {
      std::fill(local_source.begin(), local_source.end(), 0.0);

      for (int test = 0; test < dofs; ++test)
      {
         double pixx = 0.0;
         double pixy = 0.0;
         double piyy = 0.0;

         for (int angle = 0; angle < quadrature_.size(); ++angle)
         {
            const Direction &direction = quadrature_[angle];
            double vdf_dx = 0.0;
            double vdf_dy = 0.0;
            for (int coeff = 0; coeff < dofs; ++coeff)
            {
               vdf_dx += distribution(angle, element, coeff) *
                         integration_.GradX(element, coeff, test);
               vdf_dy += distribution(angle, element, coeff) *
                         integration_.GradY(element, coeff, test);
            }

            pixx += direction.cx * (5.0 * direction.cx * direction.cx - 3.0) *
                    vdf_dx * direction.weight;
            pixy += direction.cy * (5.0 * direction.cx * direction.cx - 1.0) *
                    vdf_dx * direction.weight;
            piyy += direction.cx * (5.0 * direction.cy * direction.cy - 1.0) *
                    vdf_dx * direction.weight;

            pixx += direction.cy * (5.0 * direction.cx * direction.cx - 1.0) *
                    vdf_dy * direction.weight;
            pixy += direction.cx * (5.0 * direction.cy * direction.cy - 1.0) *
                    vdf_dy * direction.weight;
            piyy += direction.cy * (5.0 * direction.cy * direction.cy - 3.0) *
                    vdf_dy * direction.weight;
         }

         local_source[static_cast<std::size_t>(Offset(MacroComponent::Lxx, test))] =
            (pixx + 0.5 * piyy) * tau_c / 5.0;
         local_source[static_cast<std::size_t>(Offset(MacroComponent::Lxy, test))] =
            0.5 * pixy * tau_c / 5.0;
         local_source[static_cast<std::size_t>(Offset(MacroComponent::Lyx, test))] =
            0.5 * pixy * tau_c / 5.0;
         local_source[static_cast<std::size_t>(Offset(MacroComponent::Lyy, test))] =
            (0.5 * pixx + piyy) * tau_c / 5.0;
      }

      SolveDenseFactoredSystem(local_lu_cache_.data() + ElementMatrixOffset(element),
                               local_pivot_cache_.data() + ElementPivotOffset(element),
                               unknowns,
                               local_source);

      for (int component = 0; component < MacroComponentCount; ++component)
      {
         for (int dof = 0; dof < dofs; ++dof)
         {
            source(static_cast<MacroComponent>(component), element, dof) =
               local_source[static_cast<std::size_t>(component * dofs + dof)];
         }
      }
   }

   return source;
}

void SyntheticAccelerationSolver::BuildTraceCoupling(
   const MeshAdapter &mesh,
   const std::vector<BoundaryCondition> &boundary_conditions)
{
   if (mesh.mesh().GetNE() != integration_.element_count())
   {
      throw std::runtime_error("Mesh and integration cache element counts differ for GSIS trace coupling.");
   }
   mesh.ValidateBoundaryAttributes(boundary_conditions);

   const int unknowns = local_unknowns();
   const int trace_unknowns = trace_unknowns_per_face();
   trace_response_.assign(static_cast<std::size_t>(integration_.element_count() * 3 *
                                                   unknowns * trace_unknowns),
                          0.0);
   trace_projection_.assign(static_cast<std::size_t>(integration_.element_count() * 3 *
                                                     trace_unknowns * unknowns),
                            0.0);

   std::vector<double> local_trace(static_cast<std::size_t>(unknowns * trace_unknowns), 0.0);
   std::vector<double> rhs(static_cast<std::size_t>(unknowns), 0.0);
   for (int element = 0; element < integration_.element_count(); ++element)
   {
      for (int local_face = 0; local_face < 3; ++local_face)
      {
         AssembleLocalTraceMatrix(element, local_face, local_trace);
         for (int trace_col = 0; trace_col < trace_unknowns; ++trace_col)
         {
            for (int row = 0; row < unknowns; ++row)
            {
               rhs[static_cast<std::size_t>(row)] =
                  local_trace[static_cast<std::size_t>(row * trace_unknowns + trace_col)];
            }
            SolveDenseFactoredSystem(local_lu_cache_.data() + ElementMatrixOffset(element),
                                     local_pivot_cache_.data() + ElementPivotOffset(element),
                                     unknowns,
                                     rhs);
            for (int row = 0; row < unknowns; ++row)
            {
               trace_response_[TraceResponseIndex(element, local_face, row, trace_col)] =
                  rhs[static_cast<std::size_t>(row)];
            }
         }

         AssembleTraceProjection(mesh, boundary_conditions, element, local_face);
      }
   }

   trace_coupling_ready_ = true;
}

TraceSystem SyntheticAccelerationSolver::BuildTraceSystem(const MeshAdapter &mesh,
                                                          const MacroState &source,
                                                          const Distribution *distribution) const
{
   if (!trace_coupling_ready_)
   {
      throw std::runtime_error("GSIS trace coupling must be built before assembling the trace system.");
   }
   if (source.elements() != integration_.element_count() ||
       source.dofs() != integration_.dofs())
   {
      throw std::runtime_error("Macro source shape does not match the GSIS trace system.");
   }

   const int trace_unknowns = trace_unknowns_per_face();
   const int system_size = static_cast<int>(mesh.faces().size()) * trace_unknowns;
   TraceSystem system;
   system.matrix = std::make_unique<mfem::SparseMatrix>(system_size);
   system.rhs.SetSize(system_size);
   system.rhs = 0.0;

   for (const FaceData &face : mesh.faces())
   {
      for (int component = 0; component < TraceComponentCount; ++component)
      {
         for (int row = 0; row < integration_.face_dofs(); ++row)
         {
            const int global_row = GlobalTraceDof(face.index, component, row);
            for (int col = 0; col < integration_.face_dofs(); ++col)
            {
               const double value = integration_.FaceMass(face.index, row, col);
               if (std::abs(value) > 1.0e-30)
               {
                  system.matrix->Add(global_row, GlobalTraceDof(face.index, component, col), value);
               }
            }
         }
      }

      if (face.element1 >= 0)
      {
         AddTraceElementContribution(mesh, *system.matrix, face.index, face.element1, face.local_face1);
         AddTraceSourceContribution(system.rhs, face.index, face.element1, face.local_face1, source);
      }
      if (face.element2 >= 0)
      {
         AddTraceElementContribution(mesh, *system.matrix, face.index, face.element2, face.local_face2);
         AddTraceSourceContribution(system.rhs, face.index, face.element2, face.local_face2, source);
      }
      if (distribution && face.is_boundary())
      {
         const int element = face.element1 >= 0 ? face.element1 : face.element2;
         const int local_face = face.element1 >= 0 ? face.local_face1 : face.local_face2;
         ApplyThermalBoundaryTraceRhs(system.rhs, face.index, element, local_face, *distribution);
      }
   }

   system.matrix->Finalize(1);
   return system;
}

TraceSolveResult SyntheticAccelerationSolver::SolveTraceSystem(const TraceSystem &system,
                                                               double relative_tolerance,
                                                               double absolute_tolerance,
                                                               int max_iterations,
                                                               int print_level,
                                                               TracePreconditionerType preconditioner) const
{
   if (!system.matrix)
   {
      throw std::runtime_error("Cannot solve an empty GSIS trace system.");
   }
   if (system.rhs.Size() != system.matrix->Height())
   {
      throw std::runtime_error("GSIS trace matrix and RHS dimensions differ.");
   }
   if (relative_tolerance <= 0.0 || absolute_tolerance < 0.0 || max_iterations <= 0)
   {
      throw std::runtime_error("Invalid GMRES settings for GSIS trace solve.");
   }

   TraceSolveResult result;
   result.trace.SetSize(system.rhs.Size());
   result.trace = 0.0;

   mfem::GMRESSolver gmres;
   std::unique_ptr<mfem::Solver> preconditioner_solver;
   if (preconditioner == TracePreconditionerType::Jacobi)
   {
      preconditioner_solver = std::make_unique<mfem::DSmoother>(*system.matrix);
      gmres.SetPreconditioner(*preconditioner_solver);
   }
   gmres.SetOperator(*system.matrix);
   gmres.SetRelTol(relative_tolerance);
   gmres.SetAbsTol(absolute_tolerance);
   gmres.SetMaxIter(max_iterations);
   gmres.SetKDim(std::min(50, std::max(1, max_iterations)));
   gmres.SetPrintLevel(print_level);
   gmres.Mult(system.rhs, result.trace);

   result.iterations = gmres.GetNumIterations();
   result.converged = gmres.GetConverged();
   result.initial_norm = gmres.GetInitialNorm();
   result.final_norm = gmres.GetFinalNorm();
   return result;
}

MacroState SyntheticAccelerationSolver::ReconstructMacroState(const MeshAdapter &mesh,
                                                              const MacroState &source,
                                                              const mfem::Vector &trace) const
{
   if (!trace_coupling_ready_)
   {
      throw std::runtime_error("GSIS trace coupling must be built before local reconstruction.");
   }
   if (source.elements() != integration_.element_count() ||
       source.dofs() != integration_.dofs())
   {
      throw std::runtime_error("Macro source shape does not match local reconstruction.");
   }
   const int expected_trace_size =
      static_cast<int>(mesh.faces().size()) * trace_unknowns_per_face();
   if (trace.Size() != expected_trace_size)
   {
      throw std::runtime_error("Trace vector shape does not match local reconstruction.");
   }

   const int unknowns = local_unknowns();
   const int trace_unknowns = trace_unknowns_per_face();
   MacroState state(integration_.element_count(), integration_.dofs());

   for (int element = 0; element < integration_.element_count(); ++element)
   {
      std::vector<double> local_state(static_cast<std::size_t>(unknowns), 0.0);
      for (int component = 0; component < MacroComponentCount; ++component)
      {
         for (int dof = 0; dof < integration_.dofs(); ++dof)
         {
            local_state[static_cast<std::size_t>(component * integration_.dofs() + dof)] =
               source(static_cast<MacroComponent>(component), element, dof);
         }
      }

      for (int local_face = 0; local_face < 3; ++local_face)
      {
         const int global_face = mesh.ElementFace(element, local_face);
         for (int macro_unknown = 0; macro_unknown < unknowns; ++macro_unknown)
         {
            for (int trace_unknown = 0; trace_unknown < trace_unknowns; ++trace_unknown)
            {
               local_state[static_cast<std::size_t>(macro_unknown)] +=
                  trace_response_[TraceResponseIndex(element,
                                                    local_face,
                                                    macro_unknown,
                                                    trace_unknown)] *
                  trace(GlobalTraceDof(global_face,
                                       trace_unknown / integration_.face_dofs(),
                                       trace_unknown % integration_.face_dofs()));
            }
         }
      }

      for (int component = 0; component < MacroComponentCount; ++component)
      {
         for (int dof = 0; dof < integration_.dofs(); ++dof)
         {
            state(static_cast<MacroComponent>(component), element, dof) =
               local_state[static_cast<std::size_t>(component * integration_.dofs() + dof)];
         }
      }
   }

   return state;
}

MomentFields SyntheticAccelerationSolver::CorrectDistributionAndComputeMoments(
   const MacroState &state,
   Distribution &distribution) const
{
   if (state.elements() != integration_.element_count() ||
       state.dofs() != integration_.dofs())
   {
      throw std::runtime_error("Macro state shape does not match GSIS VDF correction.");
   }
   if (distribution.angles() != quadrature_.size() ||
       distribution.elements() != integration_.element_count() ||
       distribution.dofs() != integration_.dofs())
   {
      throw std::runtime_error("Distribution shape does not match GSIS VDF correction.");
   }

   const double tau_c = flow_.tau_combined();
   const double four_pi = 4.0 * Pi;
   const double vg2 = flow_.group_velocity * flow_.group_velocity;
   MomentFields fields(integration_.element_count(), integration_.dofs());

   for (int element = 0; element < integration_.element_count(); ++element)
   {
      const double tau_loc = flow_.tau_r / integration_.Geometry(element).h_min;
      const double beta = std::min(tau_loc, flow_.tau_threshold) / tau_loc;

      for (int dof = 0; dof < integration_.dofs(); ++dof)
      {
         double temperature_vdf = 0.0;
         double heat_flux_x_vdf = 0.0;
         double heat_flux_y_vdf = 0.0;
         for (int angle = 0; angle < quadrature_.size(); ++angle)
         {
            const Direction &direction = quadrature_[angle];
            const double value = distribution(angle, element, dof);
            temperature_vdf += value * direction.weight;
            heat_flux_x_vdf += direction.cx * value * direction.weight;
            heat_flux_y_vdf += direction.cy * value * direction.weight;
         }
         temperature_vdf /= flow_.specific_heat;

         const double d_temperature =
            (state(MacroComponent::Temperature, element, dof) - temperature_vdf) * beta;
         const double d_heat_flux_x =
            (state(MacroComponent::HeatFluxX, element, dof) - heat_flux_x_vdf) * beta;
         const double d_heat_flux_y =
            (state(MacroComponent::HeatFluxY, element, dof) - heat_flux_y_vdf) * beta;

         for (int angle = 0; angle < quadrature_.size(); ++angle)
         {
            const Direction &direction = quadrature_[angle];
            distribution(angle, element, dof) +=
               d_temperature * flow_.specific_heat / four_pi +
               (direction.cx * d_heat_flux_x + direction.cy * d_heat_flux_y) *
                  tau_c / flow_.tau_n * 3.0 / (four_pi * vg2);
         }

         fields.TemperatureDof(element, dof) = temperature_vdf + d_temperature;
         fields.HeatFluxXDof(element, dof) = heat_flux_x_vdf + d_heat_flux_x;
         fields.HeatFluxYDof(element, dof) = heat_flux_y_vdf + d_heat_flux_y;

         const double basis_integral = integration_.BasisIntegral(element, dof);
         fields.TemperatureCell(element) += fields.TemperatureDof(element, dof) * basis_integral;
         fields.HeatFluxXCell(element) += fields.HeatFluxXDof(element, dof) * basis_integral;
         fields.HeatFluxYCell(element) += fields.HeatFluxYDof(element, dof) * basis_integral;
      }
   }

   return fields;
}

double SyntheticAccelerationSolver::TraceResponse(int element,
                                                  int local_face,
                                                  MacroComponent macro_component,
                                                  int macro_dof,
                                                  int trace_component,
                                                  int face_dof) const
{
   if (!trace_coupling_ready_)
   {
      throw std::runtime_error("GSIS trace coupling has not been built.");
   }
   return trace_response_.at(TraceResponseIndex(element,
                                               local_face,
                                               Offset(macro_component, macro_dof),
                                               trace_component * integration_.face_dofs() + face_dof));
}

double SyntheticAccelerationSolver::TraceProjection(int element,
                                                    int local_face,
                                                    int trace_component,
                                                    int face_dof,
                                                    MacroComponent macro_component,
                                                    int macro_dof) const
{
   if (!trace_coupling_ready_)
   {
      throw std::runtime_error("GSIS trace coupling has not been built.");
   }
   return trace_projection_.at(TraceProjectionIndex(element,
                                                   local_face,
                                                   trace_component * integration_.face_dofs() + face_dof,
                                                   Offset(macro_component, macro_dof)));
}

double SyntheticAccelerationSolver::TraceResponseMaxAbs() const
{
   double value = 0.0;
   for (const double entry : trace_response_)
   {
      value = std::max(value, std::abs(entry));
   }
   return value;
}

double SyntheticAccelerationSolver::TraceProjectionMaxAbs() const
{
   double value = 0.0;
   for (const double entry : trace_projection_)
   {
      value = std::max(value, std::abs(entry));
   }
   return value;
}

void SyntheticAccelerationSolver::AssembleLocalMacroMatrix(int element,
                                                          std::vector<double> &matrix) const
{
   const int dofs = integration_.dofs();
   const int unknowns = local_unknowns();
   matrix.assign(static_cast<std::size_t>(unknowns * unknowns), 0.0);

   for (int row = 0; row < dofs; ++row)
   {
      for (int col = 0; col < dofs; ++col)
      {
         double face_mass_sum = 0.0;
         for (int local_face = 0; local_face < 3; ++local_face)
         {
            face_mass_sum += integration_.ElementFaceMass(element, local_face, row, col);
         }

         const double mass = integration_.Mass(element, row, col);
         const double grad_x = integration_.GradX(element, row, col);
         const double grad_y = integration_.GradY(element, row, col);
         const double grad_x_t = integration_.GradX(element, col, row);
         const double grad_y_t = integration_.GradY(element, col, row);

         matrix[static_cast<std::size_t>(Offset(MacroComponent::Temperature, row) * unknowns +
                                         Offset(MacroComponent::Temperature, col))] =
            stabilization_[0] * face_mass_sum;
         matrix[static_cast<std::size_t>(Offset(MacroComponent::Temperature, row) * unknowns +
                                         Offset(MacroComponent::HeatFluxX, col))] = grad_x_t;
         matrix[static_cast<std::size_t>(Offset(MacroComponent::Temperature, row) * unknowns +
                                         Offset(MacroComponent::HeatFluxY, col))] = grad_y_t;

         matrix[static_cast<std::size_t>(Offset(MacroComponent::HeatFluxX, row) * unknowns +
                                         Offset(MacroComponent::Temperature, col))] =
            -grad_x * flow_.specific_heat / 3.0;
         matrix[static_cast<std::size_t>(Offset(MacroComponent::HeatFluxX, row) * unknowns +
                                         Offset(MacroComponent::HeatFluxX, col))] =
            stabilization_[1] * face_mass_sum + mass / flow_.tau_r;
         matrix[static_cast<std::size_t>(Offset(MacroComponent::HeatFluxX, row) * unknowns +
                                         Offset(MacroComponent::Lxx, col))] =
            -4.0 * grad_x_t / 3.0;
         matrix[static_cast<std::size_t>(Offset(MacroComponent::HeatFluxX, row) * unknowns +
                                         Offset(MacroComponent::Lxy, col))] = -grad_y_t;
         matrix[static_cast<std::size_t>(Offset(MacroComponent::HeatFluxX, row) * unknowns +
                                         Offset(MacroComponent::Lyx, col))] = -grad_y_t;
         matrix[static_cast<std::size_t>(Offset(MacroComponent::HeatFluxX, row) * unknowns +
                                         Offset(MacroComponent::Lyy, col))] =
            2.0 * grad_x_t / 3.0;

         matrix[static_cast<std::size_t>(Offset(MacroComponent::HeatFluxY, row) * unknowns +
                                         Offset(MacroComponent::Temperature, col))] =
            -grad_y * flow_.specific_heat / 3.0;
         matrix[static_cast<std::size_t>(Offset(MacroComponent::HeatFluxY, row) * unknowns +
                                         Offset(MacroComponent::HeatFluxY, col))] =
            stabilization_[2] * face_mass_sum + mass / flow_.tau_r;
         matrix[static_cast<std::size_t>(Offset(MacroComponent::HeatFluxY, row) * unknowns +
                                         Offset(MacroComponent::Lxx, col))] =
            2.0 * grad_y_t / 3.0;
         matrix[static_cast<std::size_t>(Offset(MacroComponent::HeatFluxY, row) * unknowns +
                                         Offset(MacroComponent::Lxy, col))] = -grad_x_t;
         matrix[static_cast<std::size_t>(Offset(MacroComponent::HeatFluxY, row) * unknowns +
                                         Offset(MacroComponent::Lyx, col))] = -grad_x_t;
         matrix[static_cast<std::size_t>(Offset(MacroComponent::HeatFluxY, row) * unknowns +
                                         Offset(MacroComponent::Lyy, col))] =
            -4.0 * grad_y_t / 3.0;

         matrix[static_cast<std::size_t>(Offset(MacroComponent::Lxx, row) * unknowns +
                                         Offset(MacroComponent::HeatFluxX, col))] =
            grad_x * flow_.tau_combined() / 5.0;
         matrix[static_cast<std::size_t>(Offset(MacroComponent::Lxx, row) * unknowns +
                                         Offset(MacroComponent::Lxx, col))] = mass;

         matrix[static_cast<std::size_t>(Offset(MacroComponent::Lxy, row) * unknowns +
                                         Offset(MacroComponent::HeatFluxX, col))] =
            grad_y * flow_.tau_combined() / 5.0;
         matrix[static_cast<std::size_t>(Offset(MacroComponent::Lxy, row) * unknowns +
                                         Offset(MacroComponent::Lxy, col))] = mass;

         matrix[static_cast<std::size_t>(Offset(MacroComponent::Lyx, row) * unknowns +
                                         Offset(MacroComponent::HeatFluxY, col))] =
            grad_x * flow_.tau_combined() / 5.0;
         matrix[static_cast<std::size_t>(Offset(MacroComponent::Lyx, row) * unknowns +
                                         Offset(MacroComponent::Lyx, col))] = mass;

         matrix[static_cast<std::size_t>(Offset(MacroComponent::Lyy, row) * unknowns +
                                         Offset(MacroComponent::HeatFluxY, col))] =
            grad_y * flow_.tau_combined() / 5.0;
         matrix[static_cast<std::size_t>(Offset(MacroComponent::Lyy, row) * unknowns +
                                         Offset(MacroComponent::Lyy, col))] = mass;
      }
   }
}

void SyntheticAccelerationSolver::AssembleLocalTraceMatrix(int element,
                                                          int local_face,
                                                          std::vector<double> &matrix) const
{
   const int dofs = integration_.dofs();
   const int face_dofs = integration_.face_dofs();
   const int unknowns = local_unknowns();
   const int trace_unknowns = trace_unknowns_per_face();
   matrix.assign(static_cast<std::size_t>(unknowns * trace_unknowns), 0.0);

   const auto normal = integration_.OutwardNormal(element, local_face);
   const double nx = normal[0];
   const double ny = normal[1];
   const double tau_c = flow_.tau_combined();

   for (int tri = 0; tri < dofs; ++tri)
   {
      for (int face = 0; face < face_dofs; ++face)
      {
         const double ww = integration_.ElementFaceBasisMass(element, local_face, tri, face);
         matrix[static_cast<std::size_t>(Offset(MacroComponent::Temperature, tri) * trace_unknowns +
                                         face)] =
            stabilization_[0] * ww;

         matrix[static_cast<std::size_t>(Offset(MacroComponent::HeatFluxX, tri) * trace_unknowns +
                                         face)] =
            -nx * ww * flow_.specific_heat / 3.0;
         matrix[static_cast<std::size_t>(Offset(MacroComponent::HeatFluxX, tri) * trace_unknowns +
                                         face_dofs + face)] =
            stabilization_[1] * ww;

         matrix[static_cast<std::size_t>(Offset(MacroComponent::HeatFluxY, tri) * trace_unknowns +
                                         face)] =
            -ny * ww * flow_.specific_heat / 3.0;
         matrix[static_cast<std::size_t>(Offset(MacroComponent::HeatFluxY, tri) * trace_unknowns +
                                         2 * face_dofs + face)] =
            stabilization_[2] * ww;

         matrix[static_cast<std::size_t>(Offset(MacroComponent::Lxx, tri) * trace_unknowns +
                                         face_dofs + face)] =
            nx * ww * tau_c / 5.0;
         matrix[static_cast<std::size_t>(Offset(MacroComponent::Lxy, tri) * trace_unknowns +
                                         face_dofs + face)] =
            ny * ww * tau_c / 5.0;
         matrix[static_cast<std::size_t>(Offset(MacroComponent::Lyx, tri) * trace_unknowns +
                                         2 * face_dofs + face)] =
            nx * ww * tau_c / 5.0;
         matrix[static_cast<std::size_t>(Offset(MacroComponent::Lyy, tri) * trace_unknowns +
                                         2 * face_dofs + face)] =
            ny * ww * tau_c / 5.0;
      }
   }
}

void SyntheticAccelerationSolver::AssembleTraceProjection(
   const MeshAdapter &mesh,
   const std::vector<BoundaryCondition> &boundary_conditions,
   int element,
   int local_face)
{
   std::unordered_map<int, BoundaryCondition> boundary_by_attribute;
   for (const BoundaryCondition &bc : boundary_conditions)
   {
      boundary_by_attribute.emplace(bc.physical_id, bc);
   }

   const int face_dofs = integration_.face_dofs();
   const int trace_unknowns = trace_unknowns_per_face();
   const int unknowns = local_unknowns();
   const auto normal = integration_.OutwardNormal(element, local_face);
   const double nx = normal[0];
   const double ny = normal[1];
   const int neighbor = mesh.ElementNeighbor(element, local_face);

   BoundaryType boundary_type = BoundaryType::Thermalizing;
   const bool interior = neighbor >= 0;
   if (!interior)
   {
      const FaceData &face = mesh.Face(mesh.ElementFace(element, local_face));
      const auto it = boundary_by_attribute.find(face.boundary_attribute);
      if (it == boundary_by_attribute.end())
      {
         throw std::runtime_error("Missing boundary condition for GSIS trace projection.");
      }
      boundary_type = it->second.type;
      if (boundary_type != BoundaryType::Thermalizing)
      {
         throw std::runtime_error("Only thermalizing boundary trace projection is implemented for GSIS.");
      }
   }

   auto set_value = [&](int trace_component,
                        int face_row,
                        MacroComponent macro_component,
                        int macro_col,
                        double value)
   {
      trace_projection_[TraceProjectionIndex(element,
                                            local_face,
                                            trace_component * face_dofs + face_row,
                                            Offset(macro_component, macro_col))] = value;
   };

   for (int face_row = 0; face_row < face_dofs; ++face_row)
   {
      for (int tri_col = 0; tri_col < integration_.dofs(); ++tri_col)
      {
         const double bb = integration_.ElementFaceBasisMass(element, local_face, tri_col, face_row);

         if (interior)
         {
            set_value(0, face_row, MacroComponent::Temperature, tri_col, 0.5 * bb);
            set_value(0, face_row, MacroComponent::HeatFluxX, tri_col,
                      nx * bb / (2.0 * stabilization_[0]));
            set_value(0, face_row, MacroComponent::HeatFluxY, tri_col,
                      ny * bb / (2.0 * stabilization_[0]));

            set_value(1, face_row, MacroComponent::HeatFluxX, tri_col, 0.5 * bb);
            set_value(1, face_row, MacroComponent::Lxx, tri_col,
                      -2.0 * nx * bb / (3.0 * stabilization_[1]));
            set_value(1, face_row, MacroComponent::Lxy, tri_col,
                      -ny * bb / (2.0 * stabilization_[1]));
            set_value(1, face_row, MacroComponent::Lyx, tri_col,
                      -ny * bb / (2.0 * stabilization_[1]));
            set_value(1, face_row, MacroComponent::Lyy, tri_col,
                      nx * bb / (3.0 * stabilization_[1]));

            set_value(2, face_row, MacroComponent::HeatFluxY, tri_col, 0.5 * bb);
            set_value(2, face_row, MacroComponent::Lxx, tri_col,
                      ny * bb / (3.0 * stabilization_[2]));
            set_value(2, face_row, MacroComponent::Lxy, tri_col,
                      -nx * bb / (2.0 * stabilization_[2]));
            set_value(2, face_row, MacroComponent::Lyx, tri_col,
                      -nx * bb / (2.0 * stabilization_[2]));
            set_value(2, face_row, MacroComponent::Lyy, tri_col,
                      -2.0 * ny * bb / (3.0 * stabilization_[2]));
         }
         else if (boundary_type == BoundaryType::Thermalizing)
         {
            set_value(1, face_row, MacroComponent::HeatFluxX, tri_col, bb);
            set_value(1, face_row, MacroComponent::Lxx, tri_col,
                      -4.0 * nx * bb / (3.0 * stabilization_[1]));
            set_value(1, face_row, MacroComponent::Lxy, tri_col,
                      -ny * bb / stabilization_[1]);
            set_value(1, face_row, MacroComponent::Lyx, tri_col,
                      -ny * bb / stabilization_[1]);
            set_value(1, face_row, MacroComponent::Lyy, tri_col,
                      2.0 * nx * bb / (3.0 * stabilization_[1]));

            set_value(2, face_row, MacroComponent::HeatFluxY, tri_col, bb);
            set_value(2, face_row, MacroComponent::Lxx, tri_col,
                      2.0 * ny * bb / (3.0 * stabilization_[2]));
            set_value(2, face_row, MacroComponent::Lxy, tri_col,
                      -nx * bb / stabilization_[2]);
            set_value(2, face_row, MacroComponent::Lyx, tri_col,
                      -nx * bb / stabilization_[2]);
            set_value(2, face_row, MacroComponent::Lyy, tri_col,
                      -4.0 * ny * bb / (3.0 * stabilization_[2]));
         }
      }
   }
}

void SyntheticAccelerationSolver::AddTraceElementContribution(const MeshAdapter &mesh,
                                                              mfem::SparseMatrix &matrix,
                                                              int row_face,
                                                              int source_element,
                                                              int source_local_face) const
{
   const int unknowns = local_unknowns();
   const int trace_unknowns = trace_unknowns_per_face();

   for (int element_face = 0; element_face < 3; ++element_face)
   {
      const int column_face = mesh.ElementFace(source_element, element_face);
      for (int trace_row = 0; trace_row < trace_unknowns; ++trace_row)
      {
         const int global_row =
            GlobalTraceDof(row_face,
                           trace_row / integration_.face_dofs(),
                           trace_row % integration_.face_dofs());
         for (int trace_col = 0; trace_col < trace_unknowns; ++trace_col)
         {
            double value = 0.0;
            for (int macro_unknown = 0; macro_unknown < unknowns; ++macro_unknown)
            {
               value += trace_projection_[TraceProjectionIndex(source_element,
                                                              source_local_face,
                                                              trace_row,
                                                              macro_unknown)] *
                        trace_response_[TraceResponseIndex(source_element,
                                                          element_face,
                                                          macro_unknown,
                                                          trace_col)];
            }
            if (std::abs(value) > 1.0e-30)
            {
               matrix.Add(global_row,
                          GlobalTraceDof(column_face,
                                         trace_col / integration_.face_dofs(),
                                         trace_col % integration_.face_dofs()),
                          -value);
            }
         }
      }
   }
}

void SyntheticAccelerationSolver::AddTraceSourceContribution(mfem::Vector &rhs,
                                                             int row_face,
                                                             int source_element,
                                                             int source_local_face,
                                                             const MacroState &source) const
{
   const int unknowns = local_unknowns();
   const int trace_unknowns = trace_unknowns_per_face();

   for (int trace_row = 0; trace_row < trace_unknowns; ++trace_row)
   {
      double value = 0.0;
      for (int macro_unknown = 0; macro_unknown < unknowns; ++macro_unknown)
      {
         const int component = macro_unknown / integration_.dofs();
         const int dof = macro_unknown % integration_.dofs();
         value += trace_projection_[TraceProjectionIndex(source_element,
                                                        source_local_face,
                                                        trace_row,
                                                        macro_unknown)] *
                  source(static_cast<MacroComponent>(component), source_element, dof);
      }
      rhs(GlobalTraceDof(row_face,
                         trace_row / integration_.face_dofs(),
                         trace_row % integration_.face_dofs())) += value;
   }
}

void SyntheticAccelerationSolver::ApplyThermalBoundaryTraceRhs(
   mfem::Vector &rhs,
   int row_face,
   int source_element,
   int source_local_face,
   const Distribution &distribution) const
{
   if (distribution.angles() != quadrature_.size() ||
       distribution.elements() != integration_.element_count() ||
       distribution.dofs() != integration_.dofs())
   {
      throw std::runtime_error("Distribution shape does not match GSIS boundary trace RHS.");
   }

   for (int face_dof = 0; face_dof < integration_.face_dofs(); ++face_dof)
   {
      double temperature = 0.0;
      for (int angle = 0; angle < quadrature_.size(); ++angle)
      {
         for (int tri_dof = 0; tri_dof < integration_.dofs(); ++tri_dof)
         {
            temperature +=
               distribution(angle, source_element, tri_dof) *
               integration_.ElementFaceBasisMass(source_element, source_local_face, tri_dof, face_dof) *
               quadrature_[angle].weight;
         }
      }
      rhs(GlobalTraceDof(row_face, 0, face_dof)) = temperature / flow_.specific_heat;
   }
}

void SyntheticAccelerationSolver::BuildLocalMacroLuCache()
{
   const int unknowns = local_unknowns();
   local_lu_cache_.assign(static_cast<std::size_t>(integration_.element_count() * unknowns * unknowns),
                          0.0);
   local_pivot_cache_.assign(static_cast<std::size_t>(integration_.element_count() * unknowns),
                             0);

   std::vector<double> matrix;
   for (int element = 0; element < integration_.element_count(); ++element)
   {
      AssembleLocalMacroMatrix(element, matrix);
      std::copy(matrix.begin(), matrix.end(),
                local_lu_cache_.begin() + static_cast<std::ptrdiff_t>(ElementMatrixOffset(element)));
      FactorDenseMatrixInPlace(local_lu_cache_.data() + ElementMatrixOffset(element),
                               local_pivot_cache_.data() + ElementPivotOffset(element),
                               unknowns);
   }
}

std::size_t SyntheticAccelerationSolver::ElementMatrixOffset(int element) const
{
   const int unknowns = local_unknowns();
   return static_cast<std::size_t>(element * unknowns * unknowns);
}

std::size_t SyntheticAccelerationSolver::ElementPivotOffset(int element) const
{
   return static_cast<std::size_t>(element * local_unknowns());
}

int SyntheticAccelerationSolver::Offset(MacroComponent component, int dof) const
{
   return static_cast<int>(component) * integration_.dofs() + dof;
}

std::size_t SyntheticAccelerationSolver::TraceResponseIndex(int element,
                                                            int local_face,
                                                            int macro_unknown,
                                                            int trace_unknown) const
{
   const int unknowns = local_unknowns();
   const int trace_unknowns = trace_unknowns_per_face();
   return static_cast<std::size_t>(((element * 3 + local_face) * unknowns + macro_unknown) *
                                  trace_unknowns + trace_unknown);
}

std::size_t SyntheticAccelerationSolver::TraceProjectionIndex(int element,
                                                              int local_face,
                                                              int trace_unknown,
                                                              int macro_unknown) const
{
   const int unknowns = local_unknowns();
   const int trace_unknowns = trace_unknowns_per_face();
   return static_cast<std::size_t>(((element * 3 + local_face) * trace_unknowns + trace_unknown) *
                                  unknowns + macro_unknown);
}

} // namespace callaway
