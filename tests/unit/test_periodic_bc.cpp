// Periodic boundary condition smoke test: build a tiny Gmsh-2.2 mesh of
// the unit square [0, 1]^2 with two triangles, apply periodic BC on the
// top/bottom pair and thermalizing BC on the left/right pair, and verify
// that:
//   (i)   MeshAdapter detects and pairs the periodic faces (mesh-side
//         plumbing through the new Gmsh $Periodic parser).
//   (ii)  KineticSweepSolver registers the periodic faces.
//   (iii) The CIS solver converges with the periodic BC active.
//   (iv)  A non-zero ΔT_bc produces a vertical temperature gradient
//         consistent with that ΔT, while reversing the ΔT sign reverses
//         the gradient direction.

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

// Gmsh 2.2 mesh of the unit square split into two triangles, with four
// boundary physical groups (10=bottom, 11=right, 12=top, 13=left) and a
// $Periodic section pairing the bottom (slave) to the top (master).
// Affine maps slave = master + (0, -1), so master_pt = slave_pt + (0, 1).
const char *kPeriodicMesh = R"GMSH($MeshFormat
2.2 0 8
$EndMeshFormat
$PhysicalNames
5
1 10 "bottom"
1 11 "right"
1 12 "top"
1 13 "left"
2 99 "domain"
$EndPhysicalNames
$Nodes
4
1 0 0 0
2 1 0 0
3 1 1 0
4 0 1 0
$EndNodes
$Elements
6
1 1 2 10 10 1 2
2 1 2 11 11 2 3
3 1 2 12 12 3 4
4 1 2 13 13 4 1
5 2 2 99 99 1 2 3
6 2 2 99 99 1 3 4
$EndElements
$Periodic
1
1 10 12
Affine 1 0 0 0 0 1 0 -1 0 0 1 0 0 0 0 1
2
1 4
2 3
$EndPeriodic
)GMSH";

void WriteFile(const std::filesystem::path &path, const std::string &content)
{
   std::ofstream out(path);
   if (!out) { std::cerr << "Failed to open " << path << "\n"; std::abort(); }
   out << content;
}

callaway::IterationResult RunCis(
   const std::filesystem::path &mesh_path,
   const std::vector<callaway::BoundaryCondition> &bcs,
   bool *has_periodic_out,
   callaway::MomentFields *moments_out)
{
   callaway::MeshAdapter mesh(mesh_path);
   if (!mesh.has_periodic_faces())
   {
      std::cerr << "FAIL: MeshAdapter reported no periodic faces.\n";
      std::abort();
   }
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
   if (has_periodic_out)
   {
      *has_periodic_out = sweep_solver.has_periodic_straight_face();
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
   const auto tmpdir = std::filesystem::temp_directory_path() / "callaway_periodic_bc";
   std::filesystem::create_directories(tmpdir);
   const auto mesh_path = tmpdir / "periodic_square.msh";
   WriteFile(mesh_path, kPeriodicMesh);

   // First case: ΔT_bc = -1 on bottom (slave), +1 on top (master). Left/
   // right are thermalizing at T = 0 to remove side-wall heating. The
   // periodic pair imposes T_self - T_partner = ±1 across y so that the
   // domain develops a vertical gradient (top warmer than bottom).
   std::vector<callaway::BoundaryCondition> bcs_pos(4);
   bcs_pos[0] = {"Bottom", 10, callaway::BoundaryType::Periodic,      -1.0, 0.0, 0.0};
   bcs_pos[1] = {"Right",  11, callaway::BoundaryType::Thermalizing,   0.0, 0.0, 0.0};
   bcs_pos[2] = {"Top",    12, callaway::BoundaryType::Periodic,      +1.0, 0.0, 0.0};
   bcs_pos[3] = {"Left",   13, callaway::BoundaryType::Thermalizing,   0.0, 0.0, 0.0};

   // Reverse polarity: top cooler than bottom.
   std::vector<callaway::BoundaryCondition> bcs_neg = bcs_pos;
   bcs_neg[0].temperature = +1.0;
   bcs_neg[2].temperature = -1.0;

   bool has_periodic_pos = false;
   bool has_periodic_neg = false;
   callaway::MomentFields moments_pos;
   callaway::MomentFields moments_neg;
   const auto r_pos = RunCis(mesh_path, bcs_pos, &has_periodic_pos, &moments_pos);
   const auto r_neg = RunCis(mesh_path, bcs_neg, &has_periodic_neg, &moments_neg);

   std::cout << "Periodic positive ΔT_bc: steps=" << r_pos.steps
             << ", converged=" << (r_pos.converged ? "yes" : "no") << "\n";
   std::cout << "Periodic negative ΔT_bc: steps=" << r_neg.steps
             << ", converged=" << (r_neg.converged ? "yes" : "no") << "\n";

   if (!has_periodic_pos || !has_periodic_neg)
   {
      std::cerr << "FAIL: KineticSweepSolver did not register periodic faces.\n";
      return 1;
   }
   if (!r_pos.converged || !r_neg.converged)
   {
      std::cerr << "FAIL: CIS did not converge for periodic BC test.\n";
      return 1;
   }

   // Two triangles, share the diagonal. Element 0 is {1,2,3} (lower-right),
   // element 1 is {1,3,4} (upper-left). Their centroid y-coordinates are
   // 1/3 and 2/3 respectively, so the "upper" element should be warmer
   // in the positive-ΔT case.
   const double T_lower_pos = moments_pos.TemperatureCell(0);
   const double T_upper_pos = moments_pos.TemperatureCell(1);
   const double T_lower_neg = moments_neg.TemperatureCell(0);
   const double T_upper_neg = moments_neg.TemperatureCell(1);

   std::cout << "  T_lower (pos ΔT) = " << T_lower_pos
             << ", T_upper (pos ΔT) = " << T_upper_pos << "\n";
   std::cout << "  T_lower (neg ΔT) = " << T_lower_neg
             << ", T_upper (neg ΔT) = " << T_upper_neg << "\n";

   if (!(T_upper_pos > T_lower_pos))
   {
      std::cerr << "FAIL: positive ΔT_bc did not produce upper > lower.\n";
      return 1;
   }
   if (!(T_upper_neg < T_lower_neg))
   {
      std::cerr << "FAIL: negative ΔT_bc did not produce upper < lower.\n";
      return 1;
   }
   // Mirror symmetry: flipping the sign of ΔT_bc should flip the gradient.
   const double delta_pos = T_upper_pos - T_lower_pos;
   const double delta_neg = T_upper_neg - T_lower_neg;
   if (std::abs(delta_pos + delta_neg) > 1.0e-6)
   {
      std::cerr << "FAIL: gradient does not flip cleanly under ΔT_bc sign reversal: "
                << "delta_pos=" << delta_pos << ", delta_neg=" << delta_neg << "\n";
      return 1;
   }

   std::cout << "test_periodic_bc: passed.\n";
   return 0;
}
