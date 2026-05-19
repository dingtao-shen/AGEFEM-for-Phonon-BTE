#include "callaway/age_basis.hpp"
#include "callaway/age_mesh.hpp"
#include "callaway/age_preprocessor.hpp"
#include "callaway/angular_quadrature.hpp"
#include "callaway/boundary.hpp"
#include "callaway/config.hpp"
#include "callaway/distribution.hpp"
#include "callaway/integration_cache.hpp"
#include "callaway/iteration_driver.hpp"
#include "callaway/kinetic_sweep_solver.hpp"
#include "callaway/mesh_adapter.hpp"
#include "callaway/moment_calculator.hpp"
#include "callaway/nodal_basis.hpp"
#include "callaway/sweep_ordering.hpp"

#include <cassert>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace
{

const char *kHalfDiskMesh =
   "MFEM mesh v1.0\n"
   "\ndimension\n2\n"
   "\nelements\n2\n15 2 0 1 3\n15 2 1 2 3\n"
   "\nboundary\n4\n9 1 0 1\n9 1 1 2\n10 1 2 3\n11 1 3 0\n"
   "\nvertices\n4\n2\n"
   "1.0 0.0\n0.0 1.0\n-1.0 0.0\n0.0 0.0\n";

const char *kHalfDiskSidecar =
   "version: 1\n"
   "curves:\n"
   "  - boundary_id: 9\n"
   "    type: circular_arc\n"
   "    center: [0.0, 0.0]\n"
   "    radius: 1.0\n"
   "    orientation: ccw\n";

void WriteFile(const std::filesystem::path &path, const std::string &content)
{
   std::ofstream out(path);
   if (!out) { std::cerr << "Failed to open " << path << "\n"; std::abort(); }
   out << content;
}

} // namespace

int main()
{
   const auto tmpdir = std::filesystem::temp_directory_path() / "callaway_age_sweep_test";
   std::filesystem::create_directories(tmpdir);
   const auto mesh_path = tmpdir / "halfdisk.mesh";
   const auto sidecar_path = tmpdir / "halfdisk.age.yaml";
   WriteFile(mesh_path, kHalfDiskMesh);
   WriteFile(sidecar_path, kHalfDiskSidecar);

   // BCs: hot arc, cold straight edges.
   std::vector<callaway::BoundaryCondition> bcs(3);
   bcs[0] = {"Arc",  9,  callaway::BoundaryType::Thermalizing, 1.0, 0.0, 0.0};
   bcs[1] = {"NegX", 10, callaway::BoundaryType::Thermalizing, 0.0, 0.0, 0.0};
   bcs[2] = {"PosX", 11, callaway::BoundaryType::Thermalizing, 0.0, 0.0, 0.0};

   callaway::MeshAdapter mesh(mesh_path);
   const callaway::AgePreprocessor pre;
   callaway::AgeMesh age_mesh = pre.Build(std::move(mesh), sidecar_path, bcs);

   const int order = 2;
   const callaway::NodalBasis basis(order);
   const auto age_bases = callaway::BuildAgeElementBases(age_mesh, order);

   callaway::VelocityMeshSettings vm;
   vm.polar_angles = 2;
   vm.azimuthal_angles = 16;
   const callaway::AngularQuadrature quadrature(vm, 1.0);

   callaway::AgeSettings age_settings;
   const callaway::IntegrationCache integration(age_mesh, basis, age_bases,
                                                quadrature, age_settings);

   callaway::FlowSettings flow;
   flow.specific_heat = 1.0;
   flow.group_velocity = 1.0;
   flow.tau_r = 1.0;
   flow.tau_n = 1.0e5;
   flow.tau_threshold = 1.0;

   const callaway::MeshAdapter &mesh_ref = age_mesh.mesh();
   const callaway::SweepOrdering ordering(mesh_ref, integration, quadrature);
   const callaway::KineticSweepSolver sweep_solver(mesh_ref, integration,
                                                   quadrature, ordering, flow, bcs);

   callaway::IterationSettings iter;
   iter.tolerance = 1.0e-6;
   iter.max_steps = 500;

   callaway::Distribution dist(quadrature.size(), integration.element_count(), integration.dofs());
   dist.Fill(0.0);

   const callaway::CisIterationDriver driver(quadrature, integration, sweep_solver,
                                              flow.specific_heat, iter);
   const callaway::IterationResult result = driver.Run(dist);

   std::cout << "AGE half-disk CIS: steps=" << result.steps
             << ", converged=" << (result.converged ? "yes" : "no")
             << ", final_residual=" << result.final_residual
             << ", mass=" << result.mass << "\n";

   // Sanity checks.
   if (!result.converged)
   {
      std::cerr << "FAIL: CIS did not converge in " << iter.max_steps << " steps.\n";
      return 1;
   }
   if (result.residual_history.size() < 2 ||
       result.residual_history.back() >= result.residual_history.front())
   {
      std::cerr << "FAIL: residual did not decrease.\n";
      return 1;
   }
   if (!std::isfinite(result.mass) || result.mass <= 0.0)
   {
      std::cerr << "FAIL: mass is non-finite or non-positive (" << result.mass << ").\n";
      return 1;
   }

   // Element-average temperature must lie within the BC range [0, 1].
   const callaway::MomentFields moments =
      callaway::MomentCalculator::Compute(dist, quadrature, integration, flow.specific_heat);
   for (int e = 0; e < integration.element_count(); ++e)
   {
      const double cell_avg =
         moments.TemperatureCell(e) / integration.Geometry(e).area;
      if (!(cell_avg > -1.0e-9 && cell_avg < 1.0 + 1.0e-9))
      {
         std::cerr << "FAIL: element " << e << " average T = " << cell_avg
                   << " is outside [0, 1].\n";
         return 1;
      }
   }

   std::cout << "test_age_kinetic_sweep: passed.\n";
   return 0;
}
