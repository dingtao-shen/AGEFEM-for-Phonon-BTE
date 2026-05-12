#include "callaway/angular_quadrature.hpp"
#include "callaway/config.hpp"
#include "callaway/distribution.hpp"
#include "callaway/integration_cache.hpp"
#include "callaway/iteration_driver.hpp"
#include "callaway/kinetic_sweep_solver.hpp"
#include "callaway/mesh_adapter.hpp"
#include "callaway/nodal_basis.hpp"
#include "callaway/sweep_ordering.hpp"

#include <cassert>
#include <cmath>
#include <filesystem>
#include <iostream>

int main(int argc, char **argv)
{
   if (argc < 3)
   {
      std::cerr << "Usage: test_cis_iteration CONFIG MESH\n";
      return 2;
   }

   callaway::Config config = callaway::LoadConfig(argv[1]);
   config.files.mesh = std::filesystem::path(argv[2]);
   config.iteration.tolerance = 1.0e-12;
   config.iteration.max_steps = 3;
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
   const callaway::KineticSweepSolver sweep_solver(mesh,
                                                   integration,
                                                   quadrature,
                                                   ordering,
                                                   config.flow,
                                                   config.boundary_conditions);
   const callaway::CisIterationDriver driver(quadrature,
                                             integration,
                                             sweep_solver,
                                             config.flow.specific_heat,
                                             config.iteration);

   callaway::Distribution distribution(quadrature.size(), integration.element_count(), integration.dofs());
   distribution.SetThermalEquilibrium(config.flow.specific_heat, 1.0);

   const callaway::IterationResult result = driver.Run(distribution);

   assert(result.converged);
   assert(result.steps == 2);
   assert(result.residual_history.size() == 2);
   assert(std::abs(result.residual_history[0] - 1.0) <= 1.0e-12);
   assert(result.final_residual <= 1.0e-12);
   assert(std::abs(result.mass - 1.0) <= 1.0e-10);

   return 0;
}
