#include "callaway/angular_quadrature.hpp"
#include "callaway/config.hpp"
#include "callaway/distribution.hpp"
#include "callaway/integration_cache.hpp"
#include "callaway/mesh_adapter.hpp"
#include "callaway/moment_calculator.hpp"
#include "callaway/nodal_basis.hpp"
#include "callaway/output_manager.hpp"

#include <cassert>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

int main(int argc, char **argv)
{
   if (argc < 3)
   {
      std::cerr << "Usage: test_output_manager CONFIG MESH\n";
      return 2;
   }

   callaway::Config config = callaway::LoadConfig(argv[1]);
   config.files.mesh = std::filesystem::path(argv[2]);
   config.Validate();

   callaway::MeshAdapter mesh(config.files.mesh);
   const callaway::NodalBasis basis(config.dg.order);
   const callaway::IntegrationCache integration(mesh, basis);
   const callaway::AngularQuadrature quadrature(config.velocity_mesh, config.flow.group_velocity);
   callaway::Distribution distribution(quadrature.size(), integration.element_count(), integration.dofs());
   distribution.SetThermalEquilibrium(config.flow.specific_heat, 1.0);
   const callaway::MomentFields moments =
      callaway::MomentCalculator::Compute(distribution, quadrature, integration, config.flow.specific_heat);

   const auto samples = callaway::OutputManager::SampleConductionField(integration,
                                                                       basis,
                                                                       quadrature,
                                                                       distribution,
                                                                       config.flow.specific_heat,
                                                                       11);
   assert(samples.size() == 121);
   for (const auto &sample : samples)
   {
      assert(sample.element >= 0);
      assert(std::abs(sample.temperature - 1.0) <= 1.0e-11);
      assert(std::abs(sample.heat_flux_x) <= 1.0e-11);
      assert(std::abs(sample.heat_flux_y) <= 1.0e-11);
   }

   const auto reference = callaway::OutputManager::SampleSquareFourierReference(config.flow.specific_heat,
                                                                                config.flow.tau_r,
                                                                                11,
                                                                                200);
   assert(reference.size() == 121);
   const auto at = [&reference](int i, int j) -> const callaway::FieldSample &
   {
      return reference[static_cast<std::size_t>(j * 11 + i)];
   };
   assert(std::abs(at(5, 0).temperature) <= 1.0e-13);
   assert(std::abs(at(5, 10).temperature - 1.0) <= 5.0e-3);

   const std::filesystem::path field_path =
      std::filesystem::temp_directory_path() / "callaway_mfem_field_test.dat";
   callaway::OutputManager::WriteTecplotConduction(field_path,
                                                   integration,
                                                   basis,
                                                   quadrature,
                                                   distribution,
                                                   config.flow.specific_heat,
                                                   11);
   std::ifstream in(field_path);
   assert(in.good());
   std::string line;
   std::getline(in, line);
   assert(line == "VARIABLES=\"x\",\"y\",\"T\",\"qx\",\"qy\",\"Nxx\",\"Nxy\",\"Nyy\"");
   std::getline(in, line);
   assert(line == "ZONE I = 11 J = 11");

   const std::filesystem::path reference_path =
      std::filesystem::temp_directory_path() / "callaway_mfem_reference_test.dat";
   callaway::OutputManager::WriteTecplotReference(reference_path,
                                                  config.flow.specific_heat,
                                                  config.flow.tau_r,
                                                  11,
                                                  200);
   std::ifstream reference_in(reference_path);
   assert(reference_in.good());
   std::getline(reference_in, line);
   assert(line == "VARIABLES=\"x\",\"y\",\"T\",\"qx\",\"qy\"");

   const std::filesystem::path paraview_path =
      std::filesystem::temp_directory_path() / "callaway_mfem_paraview_test";
   callaway::OutputManager::WriteCellAverageParaView(paraview_path,
                                                     mesh,
                                                     integration,
                                                     moments,
                                                     0,
                                                     0.0);
   assert(std::filesystem::exists(paraview_path.parent_path() / paraview_path.filename()));

   callaway::IterationResult result;
   result.residual_history = {1.0, 0.25};
   result.mass_history = {2.0, 3.0};
   const std::filesystem::path residual_path =
      std::filesystem::temp_directory_path() / "callaway_mfem_residual_test.csv";
   callaway::OutputManager::WriteResidualHistory(residual_path, result);
   std::ifstream residual_in(residual_path);
   assert(residual_in.good());
   std::getline(residual_in, line);
   assert(line == "step,residual,mass");

   callaway::IterationResult gsis_result;
   gsis_result.residual_history = {1.0};
   gsis_result.mass_history = {2.0};
   gsis_result.trace_iterations_history = {7};
   gsis_result.trace_converged_history = {1};
   gsis_result.trace_initial_norm_history = {2.0};
   gsis_result.trace_final_norm_history = {1.0e-12};
   const std::filesystem::path gsis_residual_path =
      std::filesystem::temp_directory_path() / "callaway_mfem_gsis_residual_test.csv";
   callaway::OutputManager::WriteResidualHistory(gsis_residual_path, gsis_result);
   std::ifstream gsis_residual_in(gsis_residual_path);
   assert(gsis_residual_in.good());
   std::getline(gsis_residual_in, line);
   assert(line == "step,residual,mass,trace_iterations,trace_converged,trace_initial_norm,trace_final_norm");

   return 0;
}
