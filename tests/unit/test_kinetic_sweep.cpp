#include "callaway/angular_quadrature.hpp"
#include "callaway/config.hpp"
#include "callaway/distribution.hpp"
#include "callaway/integration_cache.hpp"
#include "callaway/kinetic_sweep_solver.hpp"
#include "callaway/mesh_adapter.hpp"
#include "callaway/moment_calculator.hpp"
#include "callaway/nodal_basis.hpp"
#include "callaway/sweep_ordering.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <filesystem>
#include <iostream>

namespace
{

void CheckClose(double actual, double expected, double tolerance)
{
   assert(std::abs(actual - expected) <= tolerance);
}

} // namespace

int main(int argc, char **argv)
{
   if (argc < 3)
   {
      std::cerr << "Usage: test_kinetic_sweep CONFIG MESH\n";
      return 2;
   }

   callaway::Config config = callaway::LoadConfig(argv[1]);
   config.files.mesh = std::filesystem::path(argv[2]);
   for (auto &bc : config.boundary_conditions)
   {
      bc.temperature = 1.0;
   }
   config.Validate();

   const callaway::MeshAdapter mesh(config.files.mesh);
   const callaway::NodalBasis basis(config.dg.order);
   const callaway::IntegrationCache integration(mesh, basis);
   const callaway::AngularQuadrature quadrature(config.velocity_mesh, config.flow.group_velocity);
   const callaway::SweepOrdering ordering(mesh, integration, quadrature);

   callaway::Distribution distribution(quadrature.size(), integration.element_count(), integration.dofs());
   distribution.SetThermalEquilibrium(config.flow.specific_heat, 1.0);

   const callaway::MomentFields moments =
      callaway::MomentCalculator::Compute(distribution, quadrature, integration, config.flow.specific_heat);

   const callaway::KineticSweepSolver solver(mesh,
                                             integration,
                                             quadrature,
                                             ordering,
                                             config.flow,
                                             config.boundary_conditions);
   solver.Sweep(moments, distribution);

   const double expected = config.flow.specific_heat / (4.0 * callaway::Pi);
   double max_error = 0.0;
   for (int angle = 0; angle < distribution.angles(); ++angle)
   {
      for (int element = 0; element < distribution.elements(); ++element)
      {
         for (int dof = 0; dof < distribution.dofs(); ++dof)
         {
            max_error = std::max(max_error,
                                 std::abs(distribution(angle, element, dof) - expected));
         }
      }
   }
   CheckClose(max_error, 0.0, 1.0e-9);

   const callaway::MomentFields after =
      callaway::MomentCalculator::Compute(distribution, quadrature, integration, config.flow.specific_heat);
   CheckClose(after.Mass(), 1.0, 1.0e-10);

   return 0;
}
