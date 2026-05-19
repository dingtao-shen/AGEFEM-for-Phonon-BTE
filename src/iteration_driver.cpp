#include "callaway/iteration_driver.hpp"

#include "callaway/moment_calculator.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace callaway
{

CisIterationDriver::CisIterationDriver(const AngularQuadrature &quadrature,
                                       const IntegrationCache &integration,
                                       const KineticSweepSolver &sweep_solver,
                                       double specific_heat,
                                       IterationSettings iteration)
   : quadrature_(quadrature),
     integration_(integration),
     sweep_solver_(sweep_solver),
     specific_heat_(specific_heat),
     iteration_(iteration)
{
   if (specific_heat_ <= 0.0)
   {
      throw std::runtime_error("specific_heat must be positive for CIS iteration.");
   }
   if (iteration_.tolerance <= 0.0 || iteration_.max_steps <= 0)
   {
      throw std::runtime_error("Invalid iteration settings for CIS iteration.");
   }
}

IterationResult CisIterationDriver::Run(Distribution &distribution) const
{
   if (distribution.angles() != quadrature_.size() ||
       distribution.elements() != integration_.element_count() ||
       distribution.dofs() != integration_.dofs())
   {
      throw std::runtime_error("Distribution does not match the CIS iteration driver.");
   }

   IterationResult result;
   result.residual_history.reserve(static_cast<std::size_t>(iteration_.max_steps));
   result.mass_history.reserve(static_cast<std::size_t>(iteration_.max_steps));

   MomentFields previous(integration_.element_count(), integration_.dofs());
   MomentFields moments =
      MomentCalculator::Compute(distribution, quadrature_, integration_, specific_heat_);

   for (int step = 1; step <= iteration_.max_steps; ++step)
   {
      sweep_solver_.RefreshDiffuseWallInflow(distribution);
      sweep_solver_.Sweep(moments, distribution);
      MomentFields current =
         MomentCalculator::Compute(distribution, quadrature_, integration_, specific_heat_);

      const double residual = CisIterationDriver::TemperatureResidual(current, previous);
      result.residual_history.push_back(residual);
      result.steps = step;
      result.final_residual = residual;
      result.mass = current.Mass();
      result.mass_history.push_back(result.mass);

      if (residual < iteration_.tolerance)
      {
         result.converged = true;
         break;
      }

      previous = current;
      moments = std::move(current);
   }

   return result;
}

double CisIterationDriver::TemperatureResidual(const MomentFields &current,
                                               const MomentFields &previous)
{
   if (current.elements() != previous.elements())
   {
      throw std::runtime_error("Cannot compute residual for moment fields with different element counts.");
   }

   double numerator = 0.0;
   double denominator = 0.0;
   for (int element = 0; element < current.elements(); ++element)
   {
      const double delta = current.TemperatureCell(element) -
                           previous.TemperatureCell(element);
      numerator += delta * delta;
      denominator += current.TemperatureCell(element) *
                     current.TemperatureCell(element);
   }

   if (denominator <= 0.0)
   {
      return numerator <= 0.0 ? 0.0 : std::numeric_limits<double>::infinity();
   }
   return std::sqrt(numerator / denominator);
}

GsisIterationDriver::GsisIterationDriver(const MeshAdapter &mesh,
                                         const AngularQuadrature &quadrature,
                                         const IntegrationCache &integration,
                                         const KineticSweepSolver &sweep_solver,
                                         const SyntheticAccelerationSolver &acceleration_solver,
                                         double specific_heat,
                                         IterationSettings iteration,
                                         TraceSolverSettings trace_solver)
   : mesh_(mesh),
     quadrature_(quadrature),
     integration_(integration),
     sweep_solver_(sweep_solver),
     acceleration_solver_(acceleration_solver),
     specific_heat_(specific_heat),
     iteration_(iteration),
     trace_solver_(trace_solver)
{
   if (!acceleration_solver_.trace_coupling_ready())
   {
      throw std::runtime_error("GSIS iteration requires prebuilt trace coupling.");
   }
   if (specific_heat_ <= 0.0)
   {
      throw std::runtime_error("specific_heat must be positive for GSIS iteration.");
   }
   if (iteration_.tolerance <= 0.0 || iteration_.max_steps <= 0)
   {
      throw std::runtime_error("Invalid iteration settings for GSIS iteration.");
   }
}

IterationResult GsisIterationDriver::Run(Distribution &distribution) const
{
   if (distribution.angles() != quadrature_.size() ||
       distribution.elements() != integration_.element_count() ||
       distribution.dofs() != integration_.dofs())
   {
      throw std::runtime_error("Distribution does not match the GSIS iteration driver.");
   }

   IterationResult result;
   result.residual_history.reserve(static_cast<std::size_t>(iteration_.max_steps));
   result.mass_history.reserve(static_cast<std::size_t>(iteration_.max_steps));
   result.trace_iterations_history.reserve(static_cast<std::size_t>(iteration_.max_steps));
   result.trace_converged_history.reserve(static_cast<std::size_t>(iteration_.max_steps));
   result.trace_initial_norm_history.reserve(static_cast<std::size_t>(iteration_.max_steps));
   result.trace_final_norm_history.reserve(static_cast<std::size_t>(iteration_.max_steps));

   MomentFields previous(integration_.element_count(), integration_.dofs());
   MomentFields moments =
      MomentCalculator::Compute(distribution, quadrature_, integration_, specific_heat_);

   for (int step = 1; step <= iteration_.max_steps; ++step)
   {
      sweep_solver_.RefreshDiffuseWallInflow(distribution);
      sweep_solver_.Sweep(moments, distribution);

      const MacroState source = acceleration_solver_.ComputeHighOrderSource(distribution);
      const mfem::Vector trace_rhs =
         acceleration_solver_.BuildTraceRhs(mesh_, source, &distribution);
      const TraceSolveResult trace_result =
         acceleration_solver_.SolveTraceRhs(trace_rhs,
                                            trace_solver_.relative_tolerance,
                                            trace_solver_.absolute_tolerance,
                                            trace_solver_.max_iterations,
                                            trace_solver_.print_level,
                                            trace_solver_.preconditioner);
      const MacroState macro_state =
         acceleration_solver_.ReconstructMacroState(mesh_, source, trace_result.trace);
      MomentFields current =
         acceleration_solver_.CorrectDistributionAndComputeMoments(macro_state, distribution);

      const double residual = CisIterationDriver::TemperatureResidual(current, previous);
      result.residual_history.push_back(residual);
      result.steps = step;
      result.final_residual = residual;
      result.mass = current.Mass();
      result.mass_history.push_back(result.mass);
      result.trace_iterations_history.push_back(trace_result.iterations);
      result.trace_converged_history.push_back(trace_result.converged ? 1 : 0);
      result.trace_initial_norm_history.push_back(trace_result.initial_norm);
      result.trace_final_norm_history.push_back(trace_result.final_norm);
      result.final_macro_state = std::make_unique<MacroState>(macro_state);

      if (residual < iteration_.tolerance)
      {
         result.converged = true;
         break;
      }

      previous = current;
      moments = std::move(current);
   }

   return result;
}

} // namespace callaway
