#include "callaway/age_geometry.hpp"

#include "callaway/geometry/circular_arc.hpp"
#include "callaway/geometry/nurbs_curve.hpp"
#include "callaway/geometry/polyline_curve.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace callaway
{
namespace
{

std::string Trim(std::string value)
{
   auto not_space = [](unsigned char ch) { return !std::isspace(ch); };
   value.erase(value.begin(), std::find_if(value.begin(), value.end(), not_space));
   value.erase(std::find_if(value.rbegin(), value.rend(), not_space).base(), value.end());
   return value;
}

std::string StripComment(const std::string &line)
{
   for (std::size_t i = 0; i < line.size(); ++i)
   {
      if (line[i] == '#') { return line.substr(0, i); }
   }
   return line;
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
      os << "Geometry sidecar: expected key/value pair at line " << line_number << ": " << line;
      throw std::runtime_error(os.str());
   }
   return {Trim(line.substr(0, pos)), Trim(line.substr(pos + 1))};
}

int ToInt(const std::string &value, const std::string &key, int line_number)
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
      std::ostringstream os;
      os << "Geometry sidecar (line " << line_number << "): invalid integer for key '"
         << key << "': " << value;
      throw std::runtime_error(os.str());
   }
}

double ToDouble(const std::string &value, const std::string &key, int line_number)
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
      std::ostringstream os;
      os << "Geometry sidecar (line " << line_number << "): invalid number for key '"
         << key << "': " << value;
      throw std::runtime_error(os.str());
   }
}

bool ToBool(const std::string &value, const std::string &key, int line_number)
{
   const std::string lower = ToLower(value);
   if (lower == "true" || lower == "yes" || lower == "on" || lower == "1")  { return true; }
   if (lower == "false" || lower == "no" || lower == "off" || lower == "0") { return false; }
   std::ostringstream os;
   os << "Geometry sidecar (line " << line_number << "): invalid boolean for key '"
      << key << "': " << value;
   throw std::runtime_error(os.str());
}

CurvePoint ParseVector2(const std::string &value, const std::string &key, int line_number)
{
   std::string s = Trim(value);
   if (s.size() < 2 || s.front() != '[' || s.back() != ']')
   {
      std::ostringstream os;
      os << "Geometry sidecar (line " << line_number << "): expected `[a, b]` for key '"
         << key << "': " << value;
      throw std::runtime_error(os.str());
   }
   s = s.substr(1, s.size() - 2);
   const auto comma = s.find(',');
   if (comma == std::string::npos)
   {
      std::ostringstream os;
      os << "Geometry sidecar (line " << line_number << "): expected `[a, b]` for key '"
         << key << "': " << value;
      throw std::runtime_error(os.str());
   }
   const double a = ToDouble(Trim(s.substr(0, comma)), key, line_number);
   const double b = ToDouble(Trim(s.substr(comma + 1)), key, line_number);
   return {a, b};
}

CurveSpecKind ParseCurveSpecKind(const std::string &value, int line_number)
{
   const std::string lower = ToLower(value);
   if (lower == "circular_arc") { return CurveSpecKind::CircularArc; }
   if (lower == "nurbs")        { return CurveSpecKind::Nurbs; }
   if (lower == "polyline")     { return CurveSpecKind::Polyline; }
   std::ostringstream os;
   os << "Geometry sidecar (line " << line_number << "): unknown curve type '" << value
      << "' (expected circular_arc, nurbs, or polyline)";
   throw std::runtime_error(os.str());
}

int OrientationFromString(const std::string &value, int line_number)
{
   const std::string lower = ToLower(value);
   if (lower == "ccw" || lower == "+1" || lower == "1")  { return  1; }
   if (lower == "cw"  || lower == "-1")                  { return -1; }
   std::ostringstream os;
   os << "Geometry sidecar (line " << line_number << "): orientation must be ccw or cw, got '"
      << value << "'";
   throw std::runtime_error(os.str());
}

void AssignCurveField(CurveSpec &spec, const std::string &key, const std::string &value, int line_number)
{
   if      (key == "boundary_id") { spec.boundary_id = ToInt(value, key, line_number); }
   else if (key == "type")        { spec.kind = ParseCurveSpecKind(value, line_number); }
   else if (key == "center")      { spec.center = ParseVector2(value, key, line_number); }
   else if (key == "radius")      { spec.radius = ToDouble(value, key, line_number); }
   else if (key == "orientation") { spec.orientation = OrientationFromString(value, line_number); }
   else if (key == "degree")      { spec.degree = ToInt(value, key, line_number); }
   else if (key == "closed")      { spec.closed = ToBool(value, key, line_number); }
   else if (key == "data_file")   { spec.data_file = value; }
   else
   {
      std::ostringstream os;
      os << "Geometry sidecar (line " << line_number << "): unknown curve field '" << key << "'";
      throw std::runtime_error(os.str());
   }
}

std::vector<double> ReadNumericTokens(const std::filesystem::path &path)
{
   std::ifstream in(path);
   if (!in)
   {
      throw std::runtime_error("Failed to open geometry data_file: " + path.string());
   }
   std::vector<double> tokens;
   std::string line;
   while (std::getline(in, line))
   {
      const std::string clean = StripComment(line);
      std::istringstream iss(clean);
      double v;
      while (iss >> v) { tokens.push_back(v); }
   }
   return tokens;
}

void ValidateSpec(const CurveSpec &spec)
{
   if (spec.boundary_id <= 0)
   {
      throw std::runtime_error("Geometry sidecar: curve entry is missing a positive boundary_id.");
   }
   switch (spec.kind)
   {
      case CurveSpecKind::CircularArc:
         if (spec.radius <= 0.0)
         {
            throw std::runtime_error("circular_arc for boundary_id " +
               std::to_string(spec.boundary_id) + " requires a positive radius.");
         }
         break;
      case CurveSpecKind::Nurbs:
         if (spec.degree <= 0)
         {
            throw std::runtime_error("nurbs for boundary_id " +
               std::to_string(spec.boundary_id) + " requires positive degree.");
         }
         if (!spec.data_file)
         {
            throw std::runtime_error("nurbs for boundary_id " +
               std::to_string(spec.boundary_id) + " requires a data_file.");
         }
         break;
      case CurveSpecKind::Polyline:
         if (!spec.data_file)
         {
            throw std::runtime_error("polyline for boundary_id " +
               std::to_string(spec.boundary_id) + " requires a data_file.");
         }
         break;
   }
}

} // namespace

GeometrySidecar LoadGeometrySidecar(const std::filesystem::path &path)
{
   std::ifstream in(path);
   if (!in)
   {
      throw std::runtime_error("Failed to open geometry sidecar: " + path.string());
   }

   GeometrySidecar sidecar;
   bool in_curves_section = false;
   bool version_set = false;
   std::string raw;
   int line_number = 0;

   while (std::getline(in, raw))
   {
      ++line_number;
      const std::string line = Trim(StripComment(raw));
      if (line.empty()) { continue; }

      // Section header  "key:"  (entire line is "name:")
      if (line.back() == ':' && line.find(':') == line.size() - 1)
      {
         const std::string name = ToLower(Trim(line.substr(0, line.size() - 1)));
         if (name == "curves") { in_curves_section = true; continue; }
         std::ostringstream os;
         os << "Geometry sidecar (line " << line_number << "): unknown section '" << name << "'";
         throw std::runtime_error(os.str());
      }

      if (!in_curves_section)
      {
         auto [key, value] = ParseKeyValue(line, line_number);
         if (key == "version")
         {
            sidecar.version = ToInt(value, key, line_number);
            version_set = true;
         }
         else
         {
            std::ostringstream os;
            os << "Geometry sidecar (line " << line_number << "): unknown top-level key '"
               << key << "'";
            throw std::runtime_error(os.str());
         }
         continue;
      }

      // Inside  curves:
      if (line.rfind("- ", 0) == 0)
      {
         sidecar.curves.emplace_back();
         const std::string rest = Trim(line.substr(2));
         if (!rest.empty())
         {
            auto [key, value] = ParseKeyValue(rest, line_number);
            AssignCurveField(sidecar.curves.back(), key, value, line_number);
         }
         continue;
      }
      if (sidecar.curves.empty())
      {
         std::ostringstream os;
         os << "Geometry sidecar (line " << line_number
            << "): curve field before first list item ('- ').";
         throw std::runtime_error(os.str());
      }
      auto [key, value] = ParseKeyValue(line, line_number);
      AssignCurveField(sidecar.curves.back(), key, value, line_number);
   }

   if (!version_set)
   {
      throw std::runtime_error("Geometry sidecar: missing top-level `version` field.");
   }
   if (sidecar.version != 1)
   {
      throw std::runtime_error("Geometry sidecar: unsupported version " +
         std::to_string(sidecar.version) + " (this build understands version 1).");
   }
   for (const CurveSpec &spec : sidecar.curves)
   {
      ValidateSpec(spec);
   }
   return sidecar;
}

std::unique_ptr<BoundaryCurve> MakeBoundaryCurve(const CurveSpec &spec,
                                                 const std::filesystem::path &sidecar_dir)
{
   switch (spec.kind)
   {
      case CurveSpecKind::CircularArc:
         return std::make_unique<CircularArc>(spec.center, spec.radius, spec.orientation);

      case CurveSpecKind::Nurbs:
      {
         if (!spec.data_file)
         {
            throw std::runtime_error("MakeBoundaryCurve: nurbs spec missing data_file.");
         }
         const std::filesystem::path data_path = sidecar_dir / *spec.data_file;
         const std::vector<double> tokens = ReadNumericTokens(data_path);
         std::size_t cursor = 0;
         if (cursor >= tokens.size())
         {
            throw std::runtime_error("nurbs data_file is empty: " + data_path.string());
         }
         const int n = static_cast<int>(tokens[cursor++]);
         if (n <= 0 || cursor + static_cast<std::size_t>(3 * n) > tokens.size())
         {
            throw std::runtime_error("nurbs data_file declares " + std::to_string(n) +
               " control points but does not provide enough numeric tokens.");
         }
         std::vector<CurvePoint> control_points(n);
         std::vector<double> weights(n);
         for (int i = 0; i < n; ++i)
         {
            control_points[i] = {tokens[cursor], tokens[cursor + 1]};
            weights[i] = tokens[cursor + 2];
            cursor += 3;
         }
         if (cursor >= tokens.size())
         {
            throw std::runtime_error("nurbs data_file is missing the knot count.");
         }
         const int m = static_cast<int>(tokens[cursor++]);
         if (m <= 0 || cursor + static_cast<std::size_t>(m) != tokens.size())
         {
            throw std::runtime_error("nurbs data_file declares " + std::to_string(m) +
               " knots but the trailing token count does not match.");
         }
         std::vector<double> knots(tokens.begin() + static_cast<std::ptrdiff_t>(cursor),
                                   tokens.end());
         return std::make_unique<NurbsCurve>(spec.degree,
                                             std::move(control_points),
                                             std::move(weights),
                                             std::move(knots),
                                             spec.closed);
      }

      case CurveSpecKind::Polyline:
      {
         if (!spec.data_file)
         {
            throw std::runtime_error("MakeBoundaryCurve: polyline spec missing data_file.");
         }
         const std::filesystem::path data_path = sidecar_dir / *spec.data_file;
         const std::vector<double> tokens = ReadNumericTokens(data_path);
         if (tokens.empty())
         {
            throw std::runtime_error("polyline data_file is empty: " + data_path.string());
         }
         std::size_t cursor = 0;
         const int n = static_cast<int>(tokens[cursor++]);
         if (n < 2 || cursor + static_cast<std::size_t>(2 * n) != tokens.size())
         {
            throw std::runtime_error("polyline data_file declares " + std::to_string(n) +
               " nodes but the trailing token count does not match (expected " +
               std::to_string(2 * n) + " coordinate values).");
         }
         std::vector<CurvePoint> nodes(n);
         for (int i = 0; i < n; ++i)
         {
            nodes[i] = {tokens[cursor], tokens[cursor + 1]};
            cursor += 2;
         }
         // For polylines the orientation default (+1) is fine if the user
         // listed nodes with the domain interior on the LEFT of the
         // traversal direction; otherwise they should set orientation: cw.
         const int orientation = (spec.orientation != 0) ? spec.orientation : 1;
         return std::make_unique<PolylineCurve>(std::move(nodes), spec.closed, orientation);
      }
   }
   throw std::runtime_error("MakeBoundaryCurve: unhandled CurveSpecKind.");
}

} // namespace callaway
