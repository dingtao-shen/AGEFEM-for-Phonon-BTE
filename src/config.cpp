#include "callaway/config.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace callaway
{
namespace
{

enum class Section
{
   None,
   Iteration,
   Gsis,
   VelocityMesh,
   Dg,
   Flow,
   Files,
   BoundaryConditions
};

std::string Trim(std::string value)
{
   auto not_space = [](unsigned char ch) { return !std::isspace(ch); };
   value.erase(value.begin(), std::find_if(value.begin(), value.end(), not_space));
   value.erase(std::find_if(value.rbegin(), value.rend(), not_space).base(), value.end());
   return value;
}

std::string StripComment(const std::string &line)
{
   bool in_single = false;
   bool in_double = false;
   for (std::size_t i = 0; i < line.size(); ++i)
   {
      const char ch = line[i];
      if (ch == '\'' && !in_double) { in_single = !in_single; }
      if (ch == '"' && !in_single) { in_double = !in_double; }
      if (ch == '#' && !in_single && !in_double)
      {
         return line.substr(0, i);
      }
   }
   return line;
}

std::string Unquote(std::string value)
{
   value = Trim(value);
   if (value.size() >= 2)
   {
      const char first = value.front();
      const char last = value.back();
      if ((first == '"' && last == '"') || (first == '\'' && last == '\''))
      {
         return value.substr(1, value.size() - 2);
      }
   }
   return value;
}

std::string ToLower(std::string value)
{
   std::transform(value.begin(), value.end(), value.begin(),
                  [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
   return value;
}

std::pair<std::string, std::string> ParseKeyValue(const std::string &line, int line_number)
{
   const auto pos = line.find(':');
   if (pos == std::string::npos)
   {
      std::ostringstream os;
      os << "Expected key/value pair at line " << line_number << ": " << line;
      throw std::runtime_error(os.str());
   }
   return {Trim(line.substr(0, pos)), Unquote(line.substr(pos + 1))};
}

int ToInt(const std::string &value, const std::string &key)
{
   try
   {
      std::size_t consumed = 0;
      const int result = std::stoi(value, &consumed);
      if (consumed != value.size()) { throw std::invalid_argument("trailing characters"); }
      return result;
   }
   catch (const std::exception &)
   {
      throw std::runtime_error("Invalid integer for key '" + key + "': " + value);
   }
}

double ToDouble(const std::string &value, const std::string &key)
{
   try
   {
      std::size_t consumed = 0;
      const double result = std::stod(value, &consumed);
      if (consumed != value.size()) { throw std::invalid_argument("trailing characters"); }
      return result;
   }
   catch (const std::exception &)
   {
      throw std::runtime_error("Invalid floating-point value for key '" + key + "': " + value);
   }
}

bool ToBool(const std::string &value, const std::string &key)
{
   const std::string lower = ToLower(value);
   if (lower == "true" || lower == "yes" || lower == "1" || lower == "on")
   {
      return true;
   }
   if (lower == "false" || lower == "no" || lower == "0" || lower == "off")
   {
      return false;
   }
   throw std::runtime_error("Invalid boolean for key '" + key + "': " + value);
}

Section SectionFromName(const std::string &name)
{
   const std::string lower = ToLower(name);
   if (lower == "iteration") { return Section::Iteration; }
   if (lower == "gsis") { return Section::Gsis; }
   if (lower == "velocity_mesh") { return Section::VelocityMesh; }
   if (lower == "dg") { return Section::Dg; }
   if (lower == "flow") { return Section::Flow; }
   if (lower == "files") { return Section::Files; }
   if (lower == "boundary_conditions") { return Section::BoundaryConditions; }
   throw std::runtime_error("Unknown configuration section: " + name);
}

void AssignBoundaryField(BoundaryCondition &bc, const std::string &key, const std::string &value)
{
   if (key == "name") { bc.name = value; }
   else if (key == "physical_id") { bc.physical_id = ToInt(value, key); }
   else if (key == "type") { bc.type = BoundaryTypeFromString(value); }
   else if (key == "temperature") { bc.temperature = ToDouble(value, key); }
   else if (key == "x_offset") { bc.x_offset = ToDouble(value, key); }
   else if (key == "y_offset") { bc.y_offset = ToDouble(value, key); }
   else { throw std::runtime_error("Unknown boundary condition key: " + key); }
}

std::filesystem::path ResolvePath(const std::filesystem::path &config_path,
                                  const std::filesystem::path &value)
{
   if (value.empty() || value.is_absolute()) { return value; }
   return std::filesystem::weakly_canonical(config_path.parent_path() / value);
}

} // namespace

const char *ToString(BoundaryType type)
{
   switch (type)
   {
      case BoundaryType::Thermalizing: return "thermalizing";
      case BoundaryType::NonThermalizing: return "non_thermalizing";
      case BoundaryType::Periodic: return "periodic";
      case BoundaryType::Symmetry: return "symmetry";
   }
   return "unknown";
}

BoundaryType BoundaryTypeFromString(const std::string &value)
{
   const std::string lower = ToLower(value);
   if (lower == "1" || lower == "thermalizing" || lower == "thermalisation")
   {
      return BoundaryType::Thermalizing;
   }
   if (lower == "2" || lower == "non_thermalizing" || lower == "nonthermalizing" ||
       lower == "non_thermalisation" || lower == "nonthermalisation")
   {
      return BoundaryType::NonThermalizing;
   }
   if (lower == "3" || lower == "periodic")
   {
      return BoundaryType::Periodic;
   }
   if (lower == "4" || lower == "symmetry")
   {
      return BoundaryType::Symmetry;
   }
   throw std::runtime_error("Unknown boundary type: " + value);
}

const char *ToString(TracePreconditionerType type)
{
   switch (type)
   {
      case TracePreconditionerType::None: return "none";
      case TracePreconditionerType::Jacobi: return "jacobi";
      case TracePreconditionerType::Direct: return "direct";
   }
   return "unknown";
}

TracePreconditionerType TracePreconditionerTypeFromString(const std::string &value)
{
   const std::string lower = ToLower(value);
   if (lower == "none" || lower == "off" || lower == "false" || lower == "0")
   {
      return TracePreconditionerType::None;
   }
   if (lower == "jacobi" || lower == "diagonal")
   {
      return TracePreconditionerType::Jacobi;
   }
   if (lower == "direct" || lower == "sparse_direct" || lower == "eigen")
   {
      return TracePreconditionerType::Direct;
   }
   throw std::runtime_error("Unknown GSIS trace preconditioner type: " + value);
}

int DgSettings::triangle_dofs() const
{
   return (order + 1) * (order + 2) / 2;
}

int DgSettings::face_dofs() const
{
   return order + 1;
}

int DgSettings::triangle_quadrature_points() const
{
   switch (order)
   {
      case 1: return 3;
      case 2: return 6;
      case 3: return 12;
      case 4: return 16;
      default:
         throw std::runtime_error("Supported DG orders are 1 through 4 for strict Fortran quadrature.");
   }
}

double FlowSettings::tau_combined() const
{
   return 1.0 / (1.0 / tau_r + 1.0 / tau_n);
}

void Config::Validate() const
{
   if (iteration.tolerance <= 0.0) { throw std::runtime_error("Iteration tolerance must be positive."); }
   if (iteration.max_steps <= 0) { throw std::runtime_error("Maximum iteration steps must be positive."); }
   if (velocity_mesh.polar_angles <= 0) { throw std::runtime_error("polar_angles must be positive."); }
   if (velocity_mesh.azimuthal_angles <= 0 || velocity_mesh.azimuthal_angles % 2 != 0)
   {
      throw std::runtime_error("azimuthal_angles must be a positive even integer.");
   }
   if (dg.order < 1 || dg.order > 4)
   {
      throw std::runtime_error("Only DG orders 1 through 4 are supported by the current strict kernels.");
   }
   if (flow.specific_heat <= 0.0) { throw std::runtime_error("specific_heat must be positive."); }
   if (flow.group_velocity <= 0.0) { throw std::runtime_error("group_velocity must be positive."); }
   if (flow.tau_r <= 0.0 || flow.tau_n <= 0.0)
   {
      throw std::runtime_error("tau_r and tau_n must be positive.");
   }
   if (flow.tau_threshold <= 0.0) { throw std::runtime_error("tau_threshold must be positive."); }
   if (gsis.trace_relative_tolerance <= 0.0)
   {
      throw std::runtime_error("GSIS trace_relative_tolerance must be positive.");
   }
   if (gsis.trace_absolute_tolerance <= 0.0)
   {
      throw std::runtime_error("GSIS trace_absolute_tolerance must be positive.");
   }
   if (gsis.trace_max_iterations <= 0)
   {
      throw std::runtime_error("GSIS trace_max_iterations must be positive.");
   }
   if (files.mesh.empty()) { throw std::runtime_error("Mesh path is required."); }
   if (files.output_samples < 11) { throw std::runtime_error("output_samples must be at least 11."); }
   if (boundary_conditions.empty()) { throw std::runtime_error("At least one boundary condition is required."); }

   for (const auto &bc : boundary_conditions)
   {
      if (bc.name.empty()) { throw std::runtime_error("Boundary condition name cannot be empty."); }
      if (bc.physical_id <= 0) { throw std::runtime_error("Boundary physical_id must be positive."); }
   }
}

Config LoadConfig(const std::filesystem::path &path)
{
   std::ifstream in(path);
   if (!in)
   {
      throw std::runtime_error("Failed to open configuration file: " + path.string());
   }

   Config config;
   Section section = Section::None;
   std::string raw_line;
   int line_number = 0;

   while (std::getline(in, raw_line))
   {
      ++line_number;
      const std::string line = Trim(StripComment(raw_line));
      if (line.empty()) { continue; }

      if (line.back() == ':' && line.find(':') == line.size() - 1)
      {
         section = SectionFromName(Trim(line.substr(0, line.size() - 1)));
         continue;
      }

      if (section == Section::BoundaryConditions)
      {
         if (line.rfind("- ", 0) == 0)
         {
            config.boundary_conditions.emplace_back();
            const std::string rest = Trim(line.substr(2));
            if (!rest.empty())
            {
               auto [key, value] = ParseKeyValue(rest, line_number);
               AssignBoundaryField(config.boundary_conditions.back(), key, value);
            }
            continue;
         }
         if (config.boundary_conditions.empty())
         {
            throw std::runtime_error("Boundary field before first list item at line " +
                                     std::to_string(line_number));
         }
         auto [key, value] = ParseKeyValue(line, line_number);
         AssignBoundaryField(config.boundary_conditions.back(), key, value);
         continue;
      }

      auto [key, value] = ParseKeyValue(line, line_number);
      switch (section)
      {
         case Section::Iteration:
            if (key == "tolerance") { config.iteration.tolerance = ToDouble(value, key); }
            else if (key == "max_steps") { config.iteration.max_steps = ToInt(value, key); }
            else { throw std::runtime_error("Unknown iteration key: " + key); }
            break;
         case Section::Gsis:
            if (key == "enabled") { config.gsis.enabled = ToBool(value, key); }
            else if (key == "trace_relative_tolerance") { config.gsis.trace_relative_tolerance = ToDouble(value, key); }
            else if (key == "trace_absolute_tolerance") { config.gsis.trace_absolute_tolerance = ToDouble(value, key); }
            else if (key == "trace_max_iterations") { config.gsis.trace_max_iterations = ToInt(value, key); }
            else if (key == "trace_print_level") { config.gsis.trace_print_level = ToInt(value, key); }
            else if (key == "trace_preconditioner") { config.gsis.trace_preconditioner = TracePreconditionerTypeFromString(value); }
            else if (key == "boundary_heat_flux_from_vdf") { config.gsis.boundary_heat_flux_from_vdf = ToBool(value, key); }
            else { throw std::runtime_error("Unknown gsis key: " + key); }
            break;
         case Section::VelocityMesh:
            if (key == "polar_angles") { config.velocity_mesh.polar_angles = ToInt(value, key); }
            else if (key == "azimuthal_angles") { config.velocity_mesh.azimuthal_angles = ToInt(value, key); }
            else { throw std::runtime_error("Unknown velocity_mesh key: " + key); }
            break;
         case Section::Dg:
            if (key == "order") { config.dg.order = ToInt(value, key); }
            else { throw std::runtime_error("Unknown dg key: " + key); }
            break;
         case Section::Flow:
            if (key == "specific_heat") { config.flow.specific_heat = ToDouble(value, key); }
            else if (key == "group_velocity") { config.flow.group_velocity = ToDouble(value, key); }
            else if (key == "tau_r") { config.flow.tau_r = ToDouble(value, key); }
            else if (key == "tau_n") { config.flow.tau_n = ToDouble(value, key); }
            else if (key == "tau_threshold") { config.flow.tau_threshold = ToDouble(value, key); }
            else { throw std::runtime_error("Unknown flow key: " + key); }
            break;
         case Section::Files:
            if (key == "mesh") { config.files.mesh = ResolvePath(path, value); }
            else if (key == "output_prefix") { config.files.output_prefix = value; }
            else if (key == "output_samples") { config.files.output_samples = ToInt(value, key); }
            else { throw std::runtime_error("Unknown files key: " + key); }
            break;
         case Section::None:
            throw std::runtime_error("Key/value pair found before a section at line " +
                                     std::to_string(line_number));
         case Section::BoundaryConditions:
            break;
      }
   }

   config.Validate();
   return config;
}

} // namespace callaway
