#pragma once

#include "callaway/geometry/boundary_curve.hpp"

#include <filesystem>
#include <memory>
#include <optional>
#include <vector>

namespace callaway
{

// Which curve type a sidecar entry declares.
enum class CurveSpecKind
{
   CircularArc,
   Nurbs,
   Polyline
};

// One parsed entry of the AGE geometry sidecar: a boundary physical-id bound
// to a curve definition. Analytic curves carry their (few) parameters
// inline; nurbs and polyline curves carry their bulk numeric data
// (control points, weights, knots, sampling nodes) in a referenced
// `data_file`, because the line-based sidecar grammar deliberately omits
// nested numeric arrays.
struct CurveSpec
{
   int boundary_id = 0;
   CurveSpecKind kind = CurveSpecKind::CircularArc;

   // circular_arc
   CurvePoint center{{0.0, 0.0}};
   double radius = 0.0;
   int orientation = 1; // +1 = ccw, -1 = cw

   // nurbs
   int degree = 0;

   // polyline
   bool closed = false;

   // nurbs / polyline bulk data (whitespace/newline-delimited numeric file
   // with `#` comments). Path is resolved relative to the sidecar directory.
   std::optional<std::filesystem::path> data_file;
};

// The fully parsed sidecar, in declaration order.
struct GeometrySidecar
{
   int version = 1;
   std::vector<CurveSpec> curves;
};

// Parse the AGE geometry sidecar (typically <mesh-stem>.age.yaml or the
// path given by files.geometry). Throws on malformed input or unknown
// curve types. Validates only the syntactic schema; semantic validation
// (e.g. each boundary_id matches a mesh attribute) happens in the
// pre-processor.
GeometrySidecar LoadGeometrySidecar(const std::filesystem::path &path);

// Instantiate the BoundaryCurve described by a CurveSpec. For nurbs and
// polyline kinds, the referenced `data_file` is loaded and parsed; the
// path is resolved relative to `sidecar_dir`. Throws on missing data or
// inconsistencies (e.g. nurbs with control_count != knot_count - degree - 1).
std::unique_ptr<BoundaryCurve> MakeBoundaryCurve(const CurveSpec &spec,
                                                 const std::filesystem::path &sidecar_dir);

} // namespace callaway
