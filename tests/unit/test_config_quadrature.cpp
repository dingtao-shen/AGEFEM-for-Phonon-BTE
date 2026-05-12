#include "callaway/angular_quadrature.hpp"
#include "callaway/config.hpp"
#include "callaway/mesh_adapter.hpp"

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
      std::cerr << "Usage: test_config_quadrature CONFIG MESH\n";
      return 2;
   }

   const std::filesystem::path config_path = argv[1];
   const std::filesystem::path mesh_path = argv[2];

   const callaway::Config config = callaway::LoadConfig(config_path);
   assert(config.iteration.max_steps == 8000000);
   CheckClose(config.iteration.tolerance, 1.0e-8, 1.0e-20);
   assert(!config.gsis.enabled);
   CheckClose(config.gsis.trace_relative_tolerance, 1.0e-10, 1.0e-22);
   CheckClose(config.gsis.trace_absolute_tolerance, 1.0e-14, 1.0e-26);
   assert(config.gsis.trace_max_iterations == 500);
   assert(config.gsis.trace_print_level == -1);
   assert(config.gsis.trace_preconditioner == callaway::TracePreconditionerType::None);
   assert(config.velocity_mesh.polar_angles == 20);
   assert(config.velocity_mesh.azimuthal_angles == 40);
   assert(config.dg.order == 3);
   assert(config.dg.triangle_dofs() == 10);
   assert(config.dg.face_dofs() == 4);
   assert(config.dg.triangle_quadrature_points() == 12);
   assert(config.files.output_samples == 109);
   CheckClose(config.flow.tau_combined(), 9.9999999e-4, 1.0e-12);
   assert(config.boundary_conditions.size() == 4);
   assert(config.boundary_conditions[1].physical_id == 12);
   CheckClose(config.boundary_conditions[1].temperature, 1.0, 1.0e-15);

   const callaway::AngularQuadrature quadrature(config.velocity_mesh, config.flow.group_velocity);
   assert(quadrature.size() == 800);
   CheckClose(quadrature.SumWeights(), 4.0 * callaway::Pi, 1.0e-12);
   CheckClose(quadrature.MomentCx(), 0.0, 1.0e-12);
   CheckClose(quadrature.MomentCy(), 0.0, 1.0e-12);
   CheckClose(quadrature.MomentCxCx(), 4.0 * callaway::Pi / 3.0, 5.0e-12);
   CheckClose(quadrature.MomentCyCy(), 4.0 * callaway::Pi / 3.0, 5.0e-12);

   const callaway::MeshAdapter mesh(mesh_path);
   mesh.ValidateBoundaryAttributes(config.boundary_conditions);
   const callaway::MeshSummary summary = mesh.Summary();
   assert(summary.dimension == 2);
   assert(summary.vertices == 121);
   assert(summary.elements == 200);
   assert(summary.boundary_elements == 40);
   assert(summary.faces == 320);
   assert(mesh.BoundaryFaceCount() == 40);
   assert(mesh.InteriorFaceCount() == 280);
   assert(mesh.HasBoundaryAttribute(11));
   assert(mesh.HasBoundaryAttribute(12));
   assert(mesh.HasBoundaryAttribute(13));
   assert(mesh.HasBoundaryAttribute(14));

   return 0;
}
