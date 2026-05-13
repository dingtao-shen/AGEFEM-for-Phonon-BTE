#pragma once

#include "callaway/boundary.hpp"

#include <filesystem>
#include <string>
#include <vector>

namespace callaway
{

struct IterationSettings
{
   double tolerance = 1.0e-5;
   int max_steps = 10000;
};

enum class TracePreconditionerType
{
   None,
   Jacobi,
   Direct
};

struct SyntheticAccelerationSettings
{
   bool enabled = false;
   double trace_relative_tolerance = 1.0e-10;
   double trace_absolute_tolerance = 1.0e-14;
   int trace_max_iterations = 500;
   int trace_print_level = -1;
   TracePreconditionerType trace_preconditioner = TracePreconditionerType::None;
   bool boundary_heat_flux_from_vdf = false;
};

struct VelocityMeshSettings
{
   int polar_angles = 20;
   int azimuthal_angles = 12;
};

struct DgSettings
{
   int order = 1;

   int triangle_dofs() const;
   int face_dofs() const;
   int triangle_quadrature_points() const;
};

struct FlowSettings
{
   double specific_heat = 0.0;
   double group_velocity = 1.0;
   double tau_r = 0.0;
   double tau_n = 0.0;
   double tau_threshold = 1.0;

   double tau_combined() const;
};

struct FileSettings
{
   std::filesystem::path mesh;
   std::filesystem::path output_prefix = "output";
   int output_samples = 109;
};

struct Config
{
   IterationSettings iteration;
   SyntheticAccelerationSettings gsis;
   VelocityMeshSettings velocity_mesh;
   DgSettings dg;
   FlowSettings flow;
   FileSettings files;
   std::vector<BoundaryCondition> boundary_conditions;

   void Validate() const;
};

Config LoadConfig(const std::filesystem::path &path);
const char *ToString(TracePreconditionerType type);
TracePreconditionerType TracePreconditionerTypeFromString(const std::string &value);

} // namespace callaway
