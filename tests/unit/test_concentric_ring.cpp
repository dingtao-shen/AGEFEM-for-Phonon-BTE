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
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace
{

constexpr double kPi = 3.141592653589793238462643383279502884;

// Build an annular straight-sided ring mesh: N wedges around the origin, each
// split into two triangles. All triangles are adjacent to either the inner or
// the outer circle, and become AGE elements.
//   Triangle A (inner-touching, CCW vertex order): [Iv_k, Ov_k, Iv_{k+1}]
//   Triangle B (outer-touching, CCW vertex order): [Iv_{k+1}, Ov_k, Ov_{k+1}]
// Boundary attributes: 1 = inner arc, 2 = outer arc.
std::string GenerateAnnularMesh(int n_wedges, double r_inner, double r_outer)
{
   std::ostringstream out;
   out << "MFEM mesh v1.0\n\ndimension\n2\n\n";
   out << "elements\n" << (2 * n_wedges) << "\n";
   for (int k = 0; k < n_wedges; ++k)
   {
      const int kp1 = (k + 1) % n_wedges;
      // Triangle A: inner-touching.
      out << "15 2 " << k << " " << (n_wedges + k) << " " << kp1 << "\n";
      // Triangle B: outer-touching.
      out << "15 2 " << kp1 << " " << (n_wedges + k) << " " << (n_wedges + kp1) << "\n";
   }
   out << "\nboundary\n" << (2 * n_wedges) << "\n";
   for (int k = 0; k < n_wedges; ++k)
   {
      const int kp1 = (k + 1) % n_wedges;
      // Inner arc edge of Triangle A: traversed Iv_{k+1} -> Iv_k (local face 2),
      // which is CW around the inner circle. Attribute 1.
      out << "1 1 " << kp1 << " " << k << "\n";
      // Outer arc edge of Triangle B: traversed Ov_k -> Ov_{k+1} (local face 1),
      // which is CCW around the outer circle. Attribute 2.
      out << "2 1 " << (n_wedges + k) << " " << (n_wedges + kp1) << "\n";
   }
   out << "\nvertices\n" << (2 * n_wedges) << "\n2\n";
   out << std::setprecision(17);
   for (int k = 0; k < n_wedges; ++k)
   {
      const double angle = 2.0 * kPi * k / n_wedges;
      out << (r_inner * std::cos(angle)) << " " << (r_inner * std::sin(angle)) << "\n";
   }
   for (int k = 0; k < n_wedges; ++k)
   {
      const double angle = 2.0 * kPi * k / n_wedges;
      out << (r_outer * std::cos(angle)) << " " << (r_outer * std::sin(angle)) << "\n";
   }
   return out.str();
}

std::string MakeAnnularSidecar(double r_inner, double r_outer)
{
   std::ostringstream out;
   out << std::setprecision(17);
   out << "version: 1\n";
   out << "curves:\n";
   out << "  - boundary_id: 1\n";
   out << "    type: circular_arc\n";
   out << "    center: [0.0, 0.0]\n";
   out << "    radius: " << r_inner << "\n";
   out << "    orientation: cw\n";
   out << "  - boundary_id: 2\n";
   out << "    type: circular_arc\n";
   out << "    center: [0.0, 0.0]\n";
   out << "    radius: " << r_outer << "\n";
   out << "    orientation: ccw\n";
   return out.str();
}

void WriteFile(const std::filesystem::path &path, const std::string &content)
{
   std::ofstream out(path);
   if (!out) { std::cerr << "Failed to open " << path << "\n"; std::abort(); }
   out << content;
}

// Compute the radial centroid distance of an AGE element via its three
// underlying triangle vertices (a rough proxy for the element's mean radius;
// good enough for diagnostics).
double TriangleCentroidRadius(const callaway::ElementGeometry &g)
{
   const double cx = (g.vertices[0][0] + g.vertices[1][0] + g.vertices[2][0]) / 3.0;
   const double cy = (g.vertices[0][1] + g.vertices[1][1] + g.vertices[2][1]) / 3.0;
   return std::hypot(cx, cy);
}

} // namespace

int main()
{
   const auto tmpdir = std::filesystem::temp_directory_path() / "callaway_concentric_ring";
   std::filesystem::create_directories(tmpdir);

   constexpr int N = 16;          // wedges (16 -> 32 elements)
   constexpr double R1 = 0.1;
   constexpr double RL = 0.5;
   constexpr double T_hot = 0.5;
   constexpr double T_cold = -0.5;

   const auto mesh_path = tmpdir / "ring.mesh";
   const auto sidecar_path = tmpdir / "ring.age.yaml";
   WriteFile(mesh_path, GenerateAnnularMesh(N, R1, RL));
   WriteFile(sidecar_path, MakeAnnularSidecar(R1, RL));

   std::vector<callaway::BoundaryCondition> bcs(2);
   bcs[0] = {"Inner", 1, callaway::BoundaryType::Thermalizing, T_hot,  0.0, 0.0};
   bcs[1] = {"Outer", 2, callaway::BoundaryType::Thermalizing, T_cold, 0.0, 0.0};

   callaway::MeshAdapter mesh(mesh_path);
   const callaway::AgePreprocessor pre;
   callaway::AgeMesh age_mesh = pre.Build(std::move(mesh), sidecar_path, bcs);

   // Every element of the annulus should be AGE.
   if (age_mesh.age_element_count() != 2 * N)
   {
      std::cerr << "FAIL: expected " << (2 * N) << " AGE elements, got "
                << age_mesh.age_element_count() << "\n";
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

   // Mid-Knudsen regime: CIS converges in tens of iterations.
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
   iter.tolerance = 1.0e-5;
   iter.max_steps = 2000;

   callaway::Distribution dist(quadrature.size(),
                                integration.element_count(),
                                integration.dofs());
   dist.Fill(0.0);

   const callaway::CisIterationDriver driver(quadrature, integration, sweep_solver,
                                              flow.specific_heat, iter);
   const callaway::IterationResult result = driver.Run(dist);

   std::cout << "Concentric ring CIS: N=" << N << " wedges, " << (2 * N)
             << " AGE elements, order " << order
             << ", steps=" << result.steps
             << ", converged=" << (result.converged ? "yes" : "no")
             << ", final_residual=" << result.final_residual << "\n";

   if (!result.converged)
   {
      std::cerr << "FAIL: CIS did not converge in " << iter.max_steps << " steps.\n";
      return 1;
   }

   // Boundedness: every element-average temperature lies in [T_cold, T_hot].
   const callaway::MomentFields moments =
      callaway::MomentCalculator::Compute(dist, quadrature, integration, flow.specific_heat);
   double t_min =  1.0e9;
   double t_max = -1.0e9;
   double r_min =  1.0e9;
   double r_max = -1.0e9;
   for (int e = 0; e < integration.element_count(); ++e)
   {
      const auto &g = integration.Geometry(e);
      const double t = moments.TemperatureCell(e) / g.area;
      t_min = std::min(t_min, t);
      t_max = std::max(t_max, t);
      const double r = TriangleCentroidRadius(g);
      r_min = std::min(r_min, r);
      r_max = std::max(r_max, r);
   }
   std::cout << "  T range: [" << t_min << ", " << t_max << "]"
             << " (BC: [" << T_cold << ", " << T_hot << "])\n";
   std::cout << "  r range: [" << r_min << ", " << r_max << "]\n";

   if (!(t_min > T_cold - 1.0e-2 && t_max < T_hot + 1.0e-2))
   {
      std::cerr << "FAIL: element-average T outside the BC range.\n";
      return 1;
   }

   // Radial monotonicity: average T over elements whose centroid is near the
   // inner / outer half of the annulus should respect T_inner > T_outer.
   const double r_mid = 0.5 * (R1 + RL);
   double t_inner_sum = 0.0;
   double t_outer_sum = 0.0;
   int    n_inner = 0;
   int    n_outer = 0;
   for (int e = 0; e < integration.element_count(); ++e)
   {
      const auto &g = integration.Geometry(e);
      const double t = moments.TemperatureCell(e) / g.area;
      const double r = TriangleCentroidRadius(g);
      if (r < r_mid) { t_inner_sum += t; ++n_inner; }
      else           { t_outer_sum += t; ++n_outer; }
   }
   const double t_inner_avg = t_inner_sum / std::max(n_inner, 1);
   const double t_outer_avg = t_outer_sum / std::max(n_outer, 1);
   std::cout << "  inner-half avg T = " << t_inner_avg
             << ", outer-half avg T = " << t_outer_avg << "\n";
   if (!(t_inner_avg > t_outer_avg))
   {
      std::cerr << "FAIL: inner-half average T not greater than outer-half (no radial gradient).\n";
      return 1;
   }
   // Both halves should sit strictly between the two BC values.
   if (!(t_inner_avg > T_cold && t_inner_avg < T_hot &&
         t_outer_avg > T_cold && t_outer_avg < T_hot))
   {
      std::cerr << "FAIL: half-averages are not in (T_cold, T_hot).\n";
      return 1;
   }

   // Mass conservation (no source/sink in steady state for the gray model):
   // mass should be finite. Total mass is the integral of e = Cv*T over volume.
   if (!std::isfinite(result.mass))
   {
      std::cerr << "FAIL: mass is not finite.\n";
      return 1;
   }

   std::cout << "test_concentric_ring: passed.\n";
   return 0;
}
