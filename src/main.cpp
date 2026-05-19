#include "callaway/age_basis.hpp"
#include "callaway/age_mesh.hpp"
#include "callaway/age_preprocessor.hpp"
#include "callaway/angular_quadrature.hpp"
#include "callaway/config.hpp"
#include "callaway/distribution.hpp"
#include "callaway/integration_cache.hpp"
#include "callaway/iteration_driver.hpp"
#include "callaway/kinetic_sweep_solver.hpp"
#include "callaway/mesh_adapter.hpp"
#include "callaway/moment_calculator.hpp"
#include "callaway/nodal_basis.hpp"
#include "callaway/output_manager.hpp"
#include "callaway/sweep_ordering.hpp"

#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace
{

std::filesystem::path DefaultConfigPath()
{
   return std::filesystem::path(CALLAWAY_SOURCE_DIR) / "config" / "control.example.yaml";
}

void PrintUsage(const char *program)
{
   std::cout << "Usage: " << program
             << " [--config path] [--mesh path] [--solve] [--max-steps n]"
             << " [--write-output] [--output-prefix path] [--output-samples n]"
             << " [--no-cache-local-lu]\n";
}

std::filesystem::path WithSuffix(std::filesystem::path prefix, const std::string &suffix)
{
   prefix += suffix;
   return prefix;
}

} // namespace

int main(int argc, char **argv)
{
   try
   {
      std::filesystem::path config_path = DefaultConfigPath();
      std::filesystem::path mesh_override;
      bool solve = false;
      bool write_output = false;
      bool cache_local_lu = true;
      std::filesystem::path output_prefix_override;
      int max_steps_override = 0;
      int output_samples_override = 0;

      for (int i = 1; i < argc; ++i)
      {
         const std::string arg = argv[i];
         if (arg == "--help" || arg == "-h")
         {
            PrintUsage(argv[0]);
            return 0;
         }
         if (arg == "--config" && i + 1 < argc)
         {
            config_path = argv[++i];
            continue;
         }
         if (arg == "--mesh" && i + 1 < argc)
         {
            mesh_override = argv[++i];
            continue;
         }
         if (arg == "--solve")
         {
            solve = true;
            continue;
         }
         if (arg == "--max-steps" && i + 1 < argc)
         {
            max_steps_override = std::stoi(argv[++i]);
            continue;
         }
         if (arg == "--write-output")
         {
            write_output = true;
            continue;
         }
         if (arg == "--no-cache-local-lu")
         {
            cache_local_lu = false;
            continue;
         }
         if (arg == "--output-prefix" && i + 1 < argc)
         {
            output_prefix_override = argv[++i];
            continue;
         }
         if (arg == "--output-samples" && i + 1 < argc)
         {
            output_samples_override = std::stoi(argv[++i]);
            continue;
         }
         throw std::runtime_error("Unknown or incomplete argument: " + arg);
      }

      callaway::Config config = callaway::LoadConfig(config_path);
      if (!mesh_override.empty()) { config.files.mesh = mesh_override; }
      if (max_steps_override > 0) { config.iteration.max_steps = max_steps_override; }
      if (!output_prefix_override.empty()) { config.files.output_prefix = output_prefix_override; }
      if (output_samples_override > 0) { config.files.output_samples = output_samples_override; }
      config.Validate();
      callaway::AngularQuadrature quadrature(config.velocity_mesh, config.flow.group_velocity);
      callaway::MeshAdapter mesh_loader(config.files.mesh);
      mesh_loader.ValidateBoundaryAttributes(config.boundary_conditions);

      // Route everything through AgePreprocessor. With no sidecar the result
      // is a thin pass-through over the loaded mesh; with a sidecar the AGE
      // elements are identified and bound to their curves. The AGE-aware
      // IntegrationCache reproduces the straight-only ctor bit-for-bit when
      // no AGE elements are present.
      const callaway::AgePreprocessor age_preprocessor;
      callaway::AgePreprocessReport age_report;
      callaway::AgeMesh age_mesh = config.files.geometry.empty()
         ? age_preprocessor.BuildStraight(std::move(mesh_loader), &age_report)
         : age_preprocessor.Build(std::move(mesh_loader),
                                  config.files.geometry,
                                  config.boundary_conditions,
                                  &age_report);
      if (config.gsis.enabled && age_report.age_elements > 0)
      {
         throw std::runtime_error(
            "GSIS on AGE meshes is not yet supported. "
            "Run with gsis.enabled: false on AGE meshes for now.");
      }

      callaway::MeshAdapter &mesh = age_mesh.mesh();
      const callaway::MeshSummary summary = mesh.Summary();
      const callaway::NodalBasis basis(config.dg.order);
      const auto age_bases = callaway::BuildAgeElementBases(age_mesh, config.dg.order);
      const callaway::IntegrationCache integration(age_mesh, basis, age_bases,
                                                    quadrature, config.age);
      callaway::Distribution distribution(quadrature.size(),
                                          integration.element_count(),
                                          integration.dofs());
      const callaway::SweepOrdering ordering(mesh, integration, quadrature);

      std::cout << "Callaway MFEM refactor smoke run\n";
      std::cout << "  config: " << config_path << "\n";
      std::cout << "  mesh:   " << mesh.path() << "\n";
      std::cout << "  scheme: " << (config.gsis.enabled ? "GSIS" : "CIS") << "\n";
      std::cout << "  DG order: " << config.dg.order
                << ", element dofs: " << config.dg.triangle_dofs()
                << ", face dofs: " << config.dg.face_dofs() << "\n";
      std::cout << "  tau_c: " << std::setprecision(12) << config.flow.tau_combined() << "\n";
      std::cout << "  mesh summary: dim=" << summary.dimension
                << ", vertices=" << summary.vertices
                << ", elements=" << summary.elements
                << ", boundary_elements=" << summary.boundary_elements
                << ", faces=" << summary.faces
                << ", interior_faces=" << mesh.InteriorFaceCount()
                << ", boundary_faces=" << mesh.BoundaryFaceCount() << "\n";
      std::cout << "  AGE: " << age_report.age_elements << " AGE elements, "
                << age_report.curved_faces << " curved faces, "
                << age_report.bound_curves << " bound curves";
      if (age_report.bound_curves > 0)
      {
         std::cout << ", max endpoint projection error="
                   << std::setprecision(3) << age_report.max_endpoint_projection_error;
      }
      std::cout << "\n";
      std::cout << "  angular directions: " << quadrature.size()
                << ", sum(weight)=" << std::setprecision(16) << quadrature.SumWeights()
                << ", 4*pi=" << 4.0 * callaway::Pi << "\n";
      std::cout << "  integration total area: " << integration.TotalArea() << "\n";
      std::cout << "  sweep ordering: " << ordering.angles() << " directions x "
                << ordering.elements() << " elements\n";

      if (solve)
      {
         distribution.Fill(0.0);
         const callaway::KineticSweepSolver sweep_solver(mesh,
                                                         integration,
                                                         quadrature,
                                                         ordering,
                                                         config.flow,
                                                         config.boundary_conditions,
                                                         cache_local_lu);
         callaway::IterationResult result;
         if (config.gsis.enabled)
         {
            callaway::SyntheticAccelerationSolver acceleration_solver(
               integration,
               quadrature,
               config.flow,
               config.gsis.boundary_heat_flux_from_vdf);
            acceleration_solver.BuildTraceCoupling(mesh, config.boundary_conditions);
            callaway::TraceSolverSettings trace_solver;
            trace_solver.relative_tolerance = config.gsis.trace_relative_tolerance;
            trace_solver.absolute_tolerance = config.gsis.trace_absolute_tolerance;
            trace_solver.max_iterations = config.gsis.trace_max_iterations;
            trace_solver.print_level = config.gsis.trace_print_level;
            trace_solver.preconditioner = config.gsis.trace_preconditioner;
            const callaway::GsisIterationDriver driver(mesh,
                                                       quadrature,
                                                       integration,
                                                       sweep_solver,
                                                       acceleration_solver,
                                                       config.flow.specific_heat,
                                                       config.iteration,
                                                       trace_solver);
            result = driver.Run(distribution);
         }
         else
         {
            const callaway::CisIterationDriver driver(quadrature,
                                                      integration,
                                                      sweep_solver,
                                                      config.flow.specific_heat,
                                                      config.iteration);
            result = driver.Run(distribution);
         }
         std::cout << "  " << (config.gsis.enabled ? "GSIS" : "CIS")
                   << " solve: steps=" << result.steps
                   << ", converged=" << (result.converged ? "yes" : "no")
                   << ", final_residual=" << std::setprecision(12) << result.final_residual
                   << ", mass=" << result.mass << "\n";
         if (!result.trace_iterations_history.empty())
         {
            const std::size_t last = result.trace_iterations_history.size() - 1;
            std::cout << "  trace solve: iterations=" << result.trace_iterations_history[last]
                      << ", converged=" << (result.trace_converged_history[last] ? "yes" : "no")
                      << ", initial_norm=" << result.trace_initial_norm_history[last]
                      << ", final_norm=" << result.trace_final_norm_history[last] << "\n";
         }

         if (write_output)
         {
            const std::filesystem::path field_path =
               WithSuffix(config.files.output_prefix, "_field.dat");
            const std::filesystem::path residual_path =
               WithSuffix(config.files.output_prefix, "_residual.csv");
            const std::filesystem::path reference_path =
               WithSuffix(config.files.output_prefix, "_reference.dat");
            const std::filesystem::path paraview_path =
               WithSuffix(config.files.output_prefix, "_paraview");
            callaway::OutputManager::WriteTecplotConduction(field_path,
                                                            integration,
                                                            basis,
                                                            quadrature,
                                                            distribution,
                                                            config.flow.specific_heat,
                                                            config.files.output_samples,
                                                            result.final_macro_state.get());
            callaway::OutputManager::WriteResidualHistory(residual_path, result);
            callaway::OutputManager::WriteTecplotReference(reference_path,
                                                          config.flow.specific_heat,
                                                          config.flow.tau_r,
                                                          config.files.output_samples);
            const callaway::MomentFields output_moments =
               callaway::MomentCalculator::Compute(distribution, quadrature, integration, config.flow.specific_heat);
            callaway::OutputManager::WriteCellAverageParaView(paraview_path,
                                                              mesh,
                                                              integration,
                                                              output_moments,
                                                              result.steps,
                                                              static_cast<double>(result.steps));
            std::cout << "  output field: " << field_path << "\n";
            std::cout << "  output residual: " << residual_path << "\n";
            std::cout << "  output reference: " << reference_path << "\n";
            std::cout << "  output paraview: " << paraview_path << "\n";
         }
      }
      else
      {
         distribution.SetThermalEquilibrium(config.flow.specific_heat, 1.0);
         const callaway::MomentFields moments =
            callaway::MomentCalculator::Compute(distribution, quadrature, integration, config.flow.specific_heat);
         std::cout << "  equilibrium mass: " << moments.Mass() << "\n";
         std::cout << "  milestone status: CIS kinetic sweep and iteration driver are available; pass --solve to run CIS iterations.\n";

         if (write_output)
         {
            const std::filesystem::path field_path =
               WithSuffix(config.files.output_prefix, "_field.dat");
            const std::filesystem::path paraview_path =
               WithSuffix(config.files.output_prefix, "_paraview");
            callaway::OutputManager::WriteTecplotConduction(field_path,
                                                            integration,
                                                            basis,
                                                            quadrature,
                                                            distribution,
                                                            config.flow.specific_heat,
                                                            config.files.output_samples);
            callaway::OutputManager::WriteCellAverageParaView(paraview_path,
                                                              mesh,
                                                              integration,
                                                              moments,
                                                              0,
                                                              0.0);
            std::cout << "  output field: " << field_path << "\n";
            std::cout << "  output paraview: " << paraview_path << "\n";
         }
      }
      return 0;
   }
   catch (const std::exception &ex)
   {
      std::cerr << "Error: " << ex.what() << "\n";
      return 1;
   }
}
