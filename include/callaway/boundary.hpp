#pragma once

#include <string>

namespace callaway
{

enum class BoundaryType
{
   Thermalizing = 1,
   NonThermalizing = 2,
   Periodic = 3,
   Symmetry = 4
};

struct BoundaryCondition
{
   std::string name;
   int physical_id = 0;
   BoundaryType type = BoundaryType::Thermalizing;
   double temperature = 0.0;
   double x_offset = 0.0;
   double y_offset = 0.0;
};

const char *ToString(BoundaryType type);
BoundaryType BoundaryTypeFromString(const std::string &value);

} // namespace callaway
