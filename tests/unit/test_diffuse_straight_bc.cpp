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

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace
{

// Unit-square mesh, 2 triangles, all-straight boundaries.
// Vertices: v0=(0,0), v1=(1,0), v2=(1,1), v3=(0,1).
// Boundaries: 10 = bottom, 11 = right, 12 = top, 13 = left.
const char *kSquareMesh =
   "MFEM mesh v1.0\n\n"
   "dimension\n2\n\n"
   "elements\n2\n15 2 0 1 2\n15 2 0 2 3\n\n"
   "boundary\n4\n"
   "10 1 0 1\n"
   "11 1 1 2\n"
   "12 1 2 3\n"
   "13 1 3 0\n"
   "\nvertices\n4\n2\n"
   "0.0 0.0\n1.0 0.0\n1.0 1.0\n0.0 1.0\n";

void WriteFile(const std::filesystem::path &path, const std::string &content)
{
   std::ofstream out(path);
   if (!out) { std::cerr << "Failed to open " << path << "\n"; std::abort(); }
   out << content;
}

callaway::IterationResult RunCis(const std::filesystem::path &mesh_path,
                                  const std::vector<callaway::BoundaryCondition> &bcs,
                                  bool *has_straight_diffuse_out,
                                  callaway::MomentFields *moments_out)
{
   callaway::MeshAdapter mesh(mesh_path);
   const callaway::AgePreprocessor pre;
   callaway::AgeMesh age_mesh = pre.BuildStraight(std::move(mesh));

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

   callaway::MeshAdapter &mesh_ref = age_mesh.mesh();
   const callaway::SweepOrdering ordering(mesh_ref, integration, quadrature);
   const callaway::KineticSweepSolver sweep_solver(mesh_ref, integration,
                                                   quadrature, ordering, flow, bcs);
   if (has_straight_diffuse_out)
   {
      *has_straight_diffuse_out = sweep_solver.has_diffuse_curved_face(); // placeholder
   }

   callaway::IterationSettings iter;
   iter.tolerance = 1.0e-6;
   iter.max_steps = 2000;

   callaway::Distribution dist(quadrature.size(),
                                integration.element_count(),
                                integration.dofs());
   dist.Fill(0.0);
   const callaway::CisIterationDriver driver(quadrature, integration, sweep_solver,
                                              flow.specific_heat, iter);
   callaway::IterationResult result = driver.Run(dist);

   if (moments_out)
   {
      *moments_out = callaway::MomentCalculator::Compute(dist, quadrature, integration,
                                                          flow.specific_heat);
      // Convert per-element integrated mass to per-element average T in place.
      for (int e = 0; e < integration.element_count(); ++e)
      {
         moments_out->TemperatureCell(e) /= integration.Geometry(e).area;
      }
   }
   return result;
}

} // namespace

int main()
{
   const auto tmpdir = std::filesystem::temp_directory_path() / "callaway_diffuse_straight_bc";
   std::filesystem::create_directories(tmpdir);
   const auto mesh_path = tmpdir / "square.mesh";
   WriteFile(mesh_path, kSquareMesh);

   // Diffuse left/right; thermalizing bottom hot, top cold.
   std::vector<callaway::BoundaryCondition> bcs_diffuse(4);
   bcs_diffuse[0] = {"Bottom", 10, callaway::BoundaryType::Thermalizing,    1.0, 0.0, 0.0};
   bcs_diffuse[1] = {"Right",  11, callaway::BoundaryType::NonThermalizing, 0.0, 0.0, 0.0};
   bcs_diffuse[2] = {"Top",    12, callaway::BoundaryType::Thermalizing,    0.0, 0.0, 0.0};
   bcs_diffuse[3] = {"Left",   13, callaway::BoundaryType::NonThermalizing, 0.0, 0.0, 0.0};

   // Reference: all-thermalizing with left/right at the average BC value 0.5.
   std::vector<callaway::BoundaryCondition> bcs_ref(4);
   bcs_ref[0] = {"Bottom", 10, callaway::BoundaryType::Thermalizing, 1.0, 0.0, 0.0};
   bcs_ref[1] = {"Right",  11, callaway::BoundaryType::Thermalizing, 0.5, 0.0, 0.0};
   bcs_ref[2] = {"Top",    12, callaway::BoundaryType::Thermalizing, 0.0, 0.0, 0.0};
   bcs_ref[3] = {"Left",   13, callaway::BoundaryType::Thermalizing, 0.5, 0.0, 0.0};

   callaway::MomentFields moments_diffuse;
   callaway::MomentFields moments_ref;
   const auto r_diff = RunCis(mesh_path, bcs_diffuse, nullptr, &moments_diffuse);
   const auto r_ref  = RunCis(mesh_path, bcs_ref,     nullptr, &moments_ref);

   std::cout << "Straight diffuse CIS: steps=" << r_diff.steps
             << ", converged=" << (r_diff.converged ? "yes" : "no") << "\n";
   std::cout << "Reference        CIS: steps=" << r_ref.steps
             << ", converged=" << (r_ref.converged ? "yes" : "no") << "\n";

   if (!r_diff.converged || !r_ref.converged)
   {
      std::cerr << "FAIL: CIS did not converge.\n";
      return 1;
   }

   // Bounds: T must lie within [T_top, T_bottom] = [0, 1].
   for (int e = 0; e < moments_diffuse.elements(); ++e)
   {
      const double t = moments_diffuse.TemperatureCell(e);
      if (!(t > -1.0e-9 && t < 1.0 + 1.0e-9))
      {
         std::cerr << "FAIL: diffuse-straight T = " << t
                   << " outside [0, 1] at element " << e << "\n";
         return 1;
      }
   }

   // Cross-check: the diffuse and thermalizing-at-0.5 solutions differ.
   // Diffuse walls don't pin T = 0.5; they reflect, so the radial profile
   // through the cross-section is different.
   double max_delta = 0.0;
   for (int e = 0; e < moments_diffuse.elements(); ++e)
   {
      max_delta = std::max(max_delta,
                            std::abs(moments_diffuse.TemperatureCell(e) -
                                     moments_ref.TemperatureCell(e)));
   }
   std::cout << "  max |T_diffuse - T_ref(T_side=0.5)| = " << max_delta << "\n";
   if (max_delta < 1.0e-3)
   {
      std::cerr << "FAIL: diffuse-straight solution is essentially identical to "
                << "the thermalizing-at-0.5 reference; the dispatch may not be wired.\n";
      return 1;
   }

   std::cout << "test_diffuse_straight_bc: passed.\n";
   return 0;
}
