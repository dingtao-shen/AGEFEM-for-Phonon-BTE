#pragma once

#include "callaway/angular_quadrature.hpp"
#include "callaway/config.hpp"
#include "callaway/distribution.hpp"
#include "callaway/integration_cache.hpp"
#include "callaway/kinetic_sweep_solver.hpp"
#include "callaway/mesh_adapter.hpp"
#include "callaway/synthetic_acceleration_solver.hpp"

#include <memory>
#include <vector>

namespace callaway
{

struct IterationResult
{
   int steps = 0;
   bool converged = false;
   double final_residual = 0.0;
   double mass = 0.0;
   std::vector<double> residual_history;
   std::vector<double> mass_history;
   std::vector<int> trace_iterations_history;
   std::vector<int> trace_converged_history;
   std::vector<double> trace_initial_norm_history;
   std::vector<double> trace_final_norm_history;
   std::unique_ptr<MacroState> final_macro_state;
};

class CisIterationDriver
{
public:
   CisIterationDriver(const AngularQuadrature &quadrature,
                      const IntegrationCache &integration,
                      const KineticSweepSolver &sweep_solver,
                      double specific_heat,
                      IterationSettings iteration);

   IterationResult Run(Distribution &distribution) const;

   static double TemperatureResidual(const MomentFields &current,
                                     const MomentFields &previous);

private:
   const AngularQuadrature &quadrature_;
   const IntegrationCache &integration_;
   const KineticSweepSolver &sweep_solver_;
   double specific_heat_ = 0.0;
   IterationSettings iteration_;
};

struct TraceSolverSettings
{
   double relative_tolerance = 1.0e-10;
   double absolute_tolerance = 1.0e-14;
   int max_iterations = 500;
   int print_level = -1;
   TracePreconditionerType preconditioner = TracePreconditionerType::None;
};

class GsisIterationDriver
{
public:
   GsisIterationDriver(const MeshAdapter &mesh,
                       const AngularQuadrature &quadrature,
                       const IntegrationCache &integration,
                       const KineticSweepSolver &sweep_solver,
                       const SyntheticAccelerationSolver &acceleration_solver,
                       double specific_heat,
                       IterationSettings iteration,
                       TraceSolverSettings trace_solver = {});

   IterationResult Run(Distribution &distribution) const;

private:
   const MeshAdapter &mesh_;
   const AngularQuadrature &quadrature_;
   const IntegrationCache &integration_;
   const KineticSweepSolver &sweep_solver_;
   const SyntheticAccelerationSolver &acceleration_solver_;
   double specific_heat_ = 0.0;
   IterationSettings iteration_;
   TraceSolverSettings trace_solver_;
};

} // namespace callaway
