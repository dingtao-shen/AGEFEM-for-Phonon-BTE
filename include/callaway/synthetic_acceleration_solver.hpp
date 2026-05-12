#pragma once

#include "callaway/angular_quadrature.hpp"
#include "callaway/boundary.hpp"
#include "callaway/config.hpp"
#include "callaway/distribution.hpp"
#include "callaway/integration_cache.hpp"
#include "callaway/mesh_adapter.hpp"

#include <mfem.hpp>

#include <array>
#include <memory>
#include <vector>

namespace callaway
{

enum class MacroComponent
{
   Temperature = 0,
   HeatFluxX = 1,
   HeatFluxY = 2,
   Lxx = 3,
   Lxy = 4,
   Lyx = 5,
   Lyy = 6
};

constexpr int MacroComponentCount = 7;
constexpr int TraceComponentCount = 3;

class MacroState
{
public:
   MacroState() = default;
   MacroState(int elements, int dofs);

   int elements() const { return elements_; }
   int dofs() const { return dofs_; }
   int components() const { return MacroComponentCount; }

   double &operator()(MacroComponent component, int element, int dof);
   double operator()(MacroComponent component, int element, int dof) const;

   void Fill(double value);
   double MaxAbs() const;

private:
   int elements_ = 0;
   int dofs_ = 0;
   std::vector<double> values_;

   int Index(MacroComponent component, int element, int dof) const;
};

struct TraceSystem
{
   std::unique_ptr<mfem::SparseMatrix> matrix;
   mfem::Vector rhs;
};

struct TraceSolveResult
{
   mfem::Vector trace;
   int iterations = 0;
   bool converged = false;
   double initial_norm = 0.0;
   double final_norm = 0.0;
};

class SyntheticAccelerationSolver
{
public:
   SyntheticAccelerationSolver(const IntegrationCache &integration,
                               const AngularQuadrature &quadrature,
                               FlowSettings flow);

   int local_unknowns() const { return MacroComponentCount * integration_.dofs(); }
   int trace_unknowns_per_face() const { return TraceComponentCount * integration_.face_dofs(); }
   int GlobalTraceDof(int face, int trace_component, int face_dof) const;
   const std::array<double, TraceComponentCount> &stabilization() const { return stabilization_; }

   MacroState ComputeHighOrderSource(const Distribution &distribution) const;
   void BuildTraceCoupling(const MeshAdapter &mesh,
                           const std::vector<BoundaryCondition> &boundary_conditions);
   TraceSystem BuildTraceSystem(const MeshAdapter &mesh,
                                const MacroState &source,
                                const Distribution *distribution = nullptr) const;
   TraceSolveResult SolveTraceSystem(const TraceSystem &system,
                                     double relative_tolerance = 1.0e-10,
                                     double absolute_tolerance = 1.0e-14,
                                     int max_iterations = 500,
                                     int print_level = -1,
                                     TracePreconditionerType preconditioner =
                                        TracePreconditionerType::None) const;
   MacroState ReconstructMacroState(const MeshAdapter &mesh,
                                    const MacroState &source,
                                    const mfem::Vector &trace) const;
   MomentFields CorrectDistributionAndComputeMoments(const MacroState &state,
                                                     Distribution &distribution) const;

   bool trace_coupling_ready() const { return trace_coupling_ready_; }
   double TraceResponse(int element,
                        int local_face,
                        MacroComponent macro_component,
                        int macro_dof,
                        int trace_component,
                        int face_dof) const;
   double TraceProjection(int element,
                          int local_face,
                          int trace_component,
                          int face_dof,
                          MacroComponent macro_component,
                          int macro_dof) const;
   double TraceResponseMaxAbs() const;
   double TraceProjectionMaxAbs() const;

private:
   const IntegrationCache &integration_;
   const AngularQuadrature &quadrature_;
   FlowSettings flow_;
   std::array<double, TraceComponentCount> stabilization_{{1.0, 1.0, 1.0}};
   std::vector<double> local_lu_cache_;
   std::vector<int> local_pivot_cache_;
   std::vector<double> trace_response_;
   std::vector<double> trace_projection_;
   bool trace_coupling_ready_ = false;

   void AssembleLocalMacroMatrix(int element, std::vector<double> &matrix) const;
   void BuildLocalMacroLuCache();
   void AssembleLocalTraceMatrix(int element, int local_face, std::vector<double> &matrix) const;
   void AssembleTraceProjection(const MeshAdapter &mesh,
                                const std::vector<BoundaryCondition> &boundary_conditions,
                                int element,
                                int local_face);
   void AddTraceElementContribution(const MeshAdapter &mesh,
                                    mfem::SparseMatrix &matrix,
                                    int row_face,
                                    int source_element,
                                    int source_local_face) const;
   void AddTraceSourceContribution(mfem::Vector &rhs,
                                   int row_face,
                                   int source_element,
                                   int source_local_face,
                                   const MacroState &source) const;
   void ApplyThermalBoundaryTraceRhs(mfem::Vector &rhs,
                                     int row_face,
                                     int source_element,
                                     int source_local_face,
                                     const Distribution &distribution) const;
   std::size_t ElementMatrixOffset(int element) const;
   std::size_t ElementPivotOffset(int element) const;
   std::size_t TraceResponseIndex(int element,
                                  int local_face,
                                  int macro_unknown,
                                  int trace_unknown) const;
   std::size_t TraceProjectionIndex(int element,
                                    int local_face,
                                    int trace_unknown,
                                    int macro_unknown) const;
   int Offset(MacroComponent component, int dof) const;
};

} // namespace callaway
