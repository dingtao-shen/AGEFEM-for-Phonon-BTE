#include "callaway/angular_quadrature.hpp"
#include "callaway/config.hpp"
#include "callaway/distribution.hpp"
#include "callaway/integration_cache.hpp"
#include "callaway/mesh_adapter.hpp"
#include "callaway/nodal_basis.hpp"
#include "callaway/synthetic_acceleration_solver.hpp"

#include <cassert>
#include <cmath>
#include <filesystem>
#include <iostream>

int main(int argc, char **argv)
{
   if (argc < 3)
   {
      std::cerr << "Usage: test_synthetic_acceleration CONFIG MESH\n";
      return 2;
   }

   callaway::Config config = callaway::LoadConfig(argv[1]);
   config.files.mesh = std::filesystem::path(argv[2]);
   config.Validate();

   const callaway::MeshAdapter mesh(config.files.mesh);
   const callaway::NodalBasis basis(config.dg.order);
   const callaway::IntegrationCache integration(mesh, basis);
   const callaway::AngularQuadrature quadrature(config.velocity_mesh, config.flow.group_velocity);
   const callaway::SyntheticAccelerationSolver solver(integration, quadrature, config.flow);

   assert(solver.local_unknowns() == callaway::MacroComponentCount * integration.dofs());
   assert(solver.stabilization()[0] == 1.0);
   assert(solver.stabilization()[1] == 1.0);
   assert(solver.stabilization()[2] == 1.0);

   callaway::Distribution equilibrium(quadrature.size(), integration.element_count(), integration.dofs());
   equilibrium.SetThermalEquilibrium(config.flow.specific_heat, 1.0);
   const callaway::MacroState equilibrium_source = solver.ComputeHighOrderSource(equilibrium);
   assert(equilibrium_source.elements() == integration.element_count());
   assert(equilibrium_source.dofs() == integration.dofs());
   assert(equilibrium_source.MaxAbs() <= 1.0e-12);

   callaway::Distribution perturbed(quadrature.size(), integration.element_count(), integration.dofs());
   for (int angle = 0; angle < quadrature.size(); ++angle)
   {
      const double cx = quadrature[angle].cx;
      const double directional = 1.0 + 0.05 * cx * (5.0 * cx * cx - 3.0);
      for (int element = 0; element < integration.element_count(); ++element)
      {
         const auto &geometry = integration.Geometry(element);
         for (int dof = 0; dof < integration.dofs(); ++dof)
         {
            const double xi = basis.triangle_nodes()[static_cast<std::size_t>(dof)][0];
            const double eta = basis.triangle_nodes()[static_cast<std::size_t>(dof)][1];
            const double x = geometry.vertices[0][0] +
                             (geometry.vertices[1][0] - geometry.vertices[0][0]) * xi +
                             (geometry.vertices[2][0] - geometry.vertices[0][0]) * eta;
            perturbed(angle, element, dof) =
               config.flow.specific_heat / (4.0 * callaway::Pi) *
               directional * (1.0 + 0.01 * x);
         }
      }
   }
   const callaway::MacroState perturbed_source = solver.ComputeHighOrderSource(perturbed);
   assert(perturbed_source.MaxAbs() > 1.0e-12);

   callaway::SyntheticAccelerationSolver coupled_solver(integration, quadrature, config.flow);
   coupled_solver.BuildTraceCoupling(mesh, config.boundary_conditions);
   assert(coupled_solver.trace_coupling_ready());
   assert(coupled_solver.TraceResponseMaxAbs() > 0.0);
   assert(coupled_solver.TraceProjectionMaxAbs() > 0.0);

   const callaway::TraceSystem equilibrium_trace_system =
      coupled_solver.BuildTraceSystem(mesh, equilibrium_source);
   const int trace_size =
      static_cast<int>(mesh.faces().size()) * coupled_solver.trace_unknowns_per_face();
   assert(equilibrium_trace_system.matrix);
   assert(equilibrium_trace_system.matrix->Height() == trace_size);
   assert(equilibrium_trace_system.matrix->Width() == trace_size);
   assert(equilibrium_trace_system.matrix->NumNonZeroElems() > trace_size);
   assert(equilibrium_trace_system.rhs.Size() == trace_size);
   assert(equilibrium_trace_system.rhs.Norml2() <= 1.0e-12);
   const callaway::TraceSolveResult zero_trace =
      coupled_solver.SolveTraceSystem(equilibrium_trace_system, 1.0e-12, 1.0e-14, 50, -1);
   assert(zero_trace.trace.Size() == trace_size);
   assert(zero_trace.trace.Norml2() <= 1.0e-12);
   const callaway::MacroState zero_reconstruction =
      coupled_solver.ReconstructMacroState(mesh, equilibrium_source, zero_trace.trace);
   assert(zero_reconstruction.MaxAbs() <= 1.0e-12);

   const callaway::TraceSystem perturbed_trace_system =
      coupled_solver.BuildTraceSystem(mesh, perturbed_source);
   assert(perturbed_trace_system.matrix->NumNonZeroElems() ==
          equilibrium_trace_system.matrix->NumNonZeroElems());
   assert(perturbed_trace_system.rhs.Norml2() > 1.0e-12);
   mfem::Vector unit_trace(trace_size);
   unit_trace = 1.0;
   const callaway::MacroState unit_trace_reconstruction =
      coupled_solver.ReconstructMacroState(mesh, equilibrium_source, unit_trace);
   assert(unit_trace_reconstruction.MaxAbs() > 0.0);

   callaway::MacroState target_state(integration.element_count(), integration.dofs());
   for (int element = 0; element < integration.element_count(); ++element)
   {
      for (int dof = 0; dof < integration.dofs(); ++dof)
      {
         target_state(callaway::MacroComponent::Temperature, element, dof) = 2.0;
      }
   }
   callaway::Distribution corrected(quadrature.size(), integration.element_count(), integration.dofs());
   corrected.SetThermalEquilibrium(config.flow.specific_heat, 1.0);
   const callaway::MomentFields corrected_moments =
      coupled_solver.CorrectDistributionAndComputeMoments(target_state, corrected);
   for (int element = 0; element < integration.element_count(); ++element)
   {
      for (int dof = 0; dof < integration.dofs(); ++dof)
      {
         assert(std::abs(corrected_moments.TemperatureDof(element, dof) - 2.0) <= 1.0e-11);
         assert(std::abs(corrected_moments.HeatFluxXDof(element, dof)) <= 1.0e-11);
         assert(std::abs(corrected_moments.HeatFluxYDof(element, dof)) <= 1.0e-11);
      }
   }
   assert(std::abs(corrected_moments.Mass() - 2.0) <= 1.0e-10);

   bool checked_interior = false;
   bool checked_boundary = false;
   for (int element = 0; element < integration.element_count(); ++element)
   {
      for (int local_face = 0; local_face < 3; ++local_face)
      {
         if (!checked_interior && mesh.ElementNeighbor(element, local_face) >= 0)
         {
            double row_sum = 0.0;
            for (int dof = 0; dof < integration.dofs(); ++dof)
            {
               row_sum += std::abs(coupled_solver.TraceProjection(element,
                                                                   local_face,
                                                                   0,
                                                                   0,
                                                                   callaway::MacroComponent::Temperature,
                                                                   dof));
            }
            assert(row_sum > 0.0);
            checked_interior = true;
         }

         if (!checked_boundary && mesh.ElementNeighbor(element, local_face) < 0)
         {
            double temperature_row_sum = 0.0;
            double heat_flux_row_sum = 0.0;
            for (int dof = 0; dof < integration.dofs(); ++dof)
            {
               temperature_row_sum +=
                  std::abs(coupled_solver.TraceProjection(element,
                                                          local_face,
                                                          0,
                                                          0,
                                                          callaway::MacroComponent::Temperature,
                                                          dof));
               heat_flux_row_sum +=
                  std::abs(coupled_solver.TraceProjection(element,
                                                          local_face,
                                                          1,
                                                          0,
                                                          callaway::MacroComponent::HeatFluxX,
                                                          dof));
            }
            assert(temperature_row_sum == 0.0);
            assert(heat_flux_row_sum > 0.0);
            checked_boundary = true;
         }
      }
   }
   assert(checked_interior);
   assert(checked_boundary);

   return 0;
}
