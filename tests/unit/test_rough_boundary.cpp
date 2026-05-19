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

// Fan mesh: apex A = (0.5, 1) above a rough bottom polyline (P_0..P_4).
// Four CCW triangles (A, P_k, P_{k+1}) for k = 0..3.
// Boundary attributes: 1 = polyline (diffuse), 10 = left side (hot), 11 = right side (cold).
const char *kFanMesh =
   "MFEM mesh v1.0\n\n"
   "dimension\n2\n\n"
   "elements\n4\n"
   "15 2 0 1 2\n"
   "15 2 0 2 3\n"
   "15 2 0 3 4\n"
   "15 2 0 4 5\n"
   "\nboundary\n6\n"
   "10 1 0 1\n"   // A -> P_0 (left side)
   "1 1 1 2\n"    // polyline segment 0
   "1 1 2 3\n"    // polyline segment 1
   "1 1 3 4\n"    // polyline segment 2
   "1 1 4 5\n"    // polyline segment 3
   "11 1 5 0\n"   // P_4 -> A (right side)
   "\nvertices\n6\n2\n"
   "0.5  1.0\n"   // 0: A (apex above polyline)
   "0.0  0.0\n"   // 1: P_0
   "0.25 0.1\n"   // 2: P_1
   "0.5 -0.1\n"   // 3: P_2 (dips below the chord)
   "0.75 0.1\n"   // 4: P_3
   "1.0  0.0\n";  // 5: P_4

const char *kRoughSidecar =
   "version: 1\n"
   "curves:\n"
   "  - boundary_id: 1\n"
   "    type: polyline\n"
   "    closed: false\n"
   "    orientation: ccw\n"
   "    data_file: rough.txt\n";

const char *kRoughData =
   "5\n"
   "0.0   0.0\n"
   "0.25  0.1\n"
   "0.5  -0.1\n"
   "0.75  0.1\n"
   "1.0   0.0\n";

void WriteFile(const std::filesystem::path &path, const std::string &content)
{
   std::ofstream out(path);
   if (!out) { std::cerr << "Failed to open " << path << "\n"; std::abort(); }
   out << content;
}

} // namespace

int main()
{
   const auto tmpdir = std::filesystem::temp_directory_path() / "callaway_rough_boundary";
   std::filesystem::create_directories(tmpdir);
   const auto mesh_path = tmpdir / "fan.mesh";
   const auto sidecar_path = tmpdir / "fan.age.yaml";
   const auto data_path = tmpdir / "rough.txt";
   WriteFile(mesh_path, kFanMesh);
   WriteFile(sidecar_path, kRoughSidecar);
   WriteFile(data_path, kRoughData);

   // BCs: rough polyline diffuse, left side hot (T=1), right side cold (T=0).
   std::vector<callaway::BoundaryCondition> bcs(3);
   bcs[0] = {"Rough", 1,  callaway::BoundaryType::NonThermalizing, 0.0, 0.0, 0.0};
   bcs[1] = {"Left",  10, callaway::BoundaryType::Thermalizing,    1.0, 0.0, 0.0};
   bcs[2] = {"Right", 11, callaway::BoundaryType::Thermalizing,    0.0, 0.0, 0.0};

   callaway::MeshAdapter mesh(mesh_path);
   const callaway::AgePreprocessor pre;
   callaway::AgePreprocessReport report;
   callaway::AgeMesh age_mesh = pre.Build(std::move(mesh), sidecar_path, bcs, &report);

   if (age_mesh.age_element_count() != 4)
   {
      std::cerr << "FAIL: expected 4 AGE elements, got " << age_mesh.age_element_count() << "\n";
      return 1;
   }
   if (age_mesh.curved_face_count() != 4)
   {
      std::cerr << "FAIL: expected 4 curved faces, got " << age_mesh.curved_face_count() << "\n";
      return 1;
   }
   if (report.bound_curves != 1)
   {
      std::cerr << "FAIL: expected 1 bound curve, got " << report.bound_curves << "\n";
      return 1;
   }

   const int order = 2;
   const callaway::NodalBasis basis(order);
   const auto age_bases = callaway::BuildAgeElementBases(age_mesh, order);

   callaway::VelocityMeshSettings vm;
   vm.polar_angles = 4;
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

   callaway::MeshAdapter &mesh_ref = age_mesh.mesh();
   const callaway::SweepOrdering ordering(mesh_ref, integration, quadrature);
   const callaway::KineticSweepSolver sweep_solver(mesh_ref, integration,
                                                   quadrature, ordering, flow, bcs);

   callaway::IterationSettings iter;
   iter.tolerance = 1.0e-6;
   iter.max_steps = 2000;

   callaway::Distribution dist(quadrature.size(),
                                integration.element_count(),
                                integration.dofs());
   dist.Fill(0.0);

   const callaway::CisIterationDriver driver(quadrature, integration, sweep_solver,
                                              flow.specific_heat, iter);
   const callaway::IterationResult result = driver.Run(dist);

   std::cout << "Rough-boundary CIS: steps=" << result.steps
             << ", converged=" << (result.converged ? "yes" : "no")
             << ", final_residual=" << result.final_residual << "\n";

   if (!result.converged)
   {
      std::cerr << "FAIL: CIS did not converge.\n";
      return 1;
   }

   // Element-average temperatures must remain bounded by the thermalizing BCs
   // (the rough diffuse boundary neither sources nor sinks energy).
   const callaway::MomentFields moments =
      callaway::MomentCalculator::Compute(dist, quadrature, integration, flow.specific_heat);
   for (int e = 0; e < integration.element_count(); ++e)
   {
      const double t = moments.TemperatureCell(e) / integration.Geometry(e).area;
      if (!(t > -1.0e-9 && t < 1.0 + 1.0e-9))
      {
         std::cerr << "FAIL: element " << e << " average T = " << t
                   << " is outside [0, 1].\n";
         return 1;
      }
   }

   // Left-right temperature gradient sanity check: the leftmost AGE element
   // (triangle 0 = (A, P_0, P_1)) is adjacent to the hot left side; the
   // rightmost (triangle 3 = (A, P_3, P_4)) is adjacent to the cold right side.
   const double t_left  = moments.TemperatureCell(0) / integration.Geometry(0).area;
   const double t_right = moments.TemperatureCell(3) / integration.Geometry(3).area;
   std::cout << "  leftmost AGE T = " << t_left
             << ", rightmost AGE T = " << t_right << "\n";
   if (!(t_left > t_right))
   {
      std::cerr << "FAIL: expected T_left > T_right; got " << t_left << " vs " << t_right << "\n";
      return 1;
   }

   std::cout << "test_rough_boundary: passed.\n";
   return 0;
}
