#include "callaway/geometry/polyline_curve.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace callaway
{
namespace
{

double Distance(const CurvePoint &a, const CurvePoint &b)
{
   return std::hypot(b[0] - a[0], b[1] - a[1]);
}

// Locate the polyline segment containing parameter lambda.
// Right-continuous at breakpoints; lambda == 1 maps to the last segment
// with t = 1.
struct SegmentLocation
{
   int segment;
   double t;
   CurvePoint a;
   CurvePoint b;
   double lambda_a;
   double lambda_b;
};

SegmentLocation Locate(const std::vector<CurvePoint> &nodes,
                       const std::vector<double> &lambdas,
                       bool closed,
                       double lambda)
{
   const std::size_t n = nodes.size();
   const std::size_t seg_count = closed ? n : n - 1;

   auto endpoint_of_segment = [&](std::size_t k) -> CurvePoint {
      if (closed && k + 1 == n) { return nodes[0]; }
      return nodes[k + 1];
   };
   auto lambda_of_endpoint = [&](std::size_t k) -> double {
      if (closed && k + 1 == n) { return 1.0; }
      return lambdas[k + 1];
   };

   if (lambda >= 1.0)
   {
      const std::size_t k = seg_count - 1;
      return SegmentLocation{static_cast<int>(k), 1.0, nodes[k], endpoint_of_segment(k),
                              lambdas[k], lambda_of_endpoint(k)};
   }
   for (std::size_t k = 0; k < seg_count; ++k)
   {
      const double la = lambdas[k];
      const double lb = lambda_of_endpoint(k);
      if (lambda >= la && lambda < lb)
      {
         const double t = (lambda - la) / (lb - la);
         return SegmentLocation{static_cast<int>(k), t, nodes[k], endpoint_of_segment(k), la, lb};
      }
   }
   // Floating-point fall-through: clamp to the last segment.
   const std::size_t k = seg_count - 1;
   return SegmentLocation{static_cast<int>(k), 1.0, nodes[k], endpoint_of_segment(k),
                           lambdas[k], lambda_of_endpoint(k)};
}

} // namespace

PolylineCurve::PolylineCurve(std::vector<CurvePoint> nodes, bool closed, int orientation)
   : nodes_(std::move(nodes)), closed_(closed), orientation_(orientation)
{
   if (nodes_.size() < 2)
   {
      throw std::runtime_error("PolylineCurve requires at least two nodes.");
   }
   if (orientation_ != 1 && orientation_ != -1)
   {
      throw std::runtime_error("PolylineCurve orientation must be +1 or -1.");
   }

   std::vector<double> cumulative(nodes_.size(), 0.0);
   for (std::size_t i = 1; i < nodes_.size(); ++i)
   {
      const double seg = Distance(nodes_[i - 1], nodes_[i]);
      if (seg <= 0.0)
      {
         throw std::runtime_error("PolylineCurve has a zero-length segment between nodes.");
      }
      cumulative[i] = cumulative[i - 1] + seg;
   }

   double total = cumulative.back();
   if (closed_)
   {
      const double closing = Distance(nodes_.back(), nodes_.front());
      if (closing <= 0.0)
      {
         throw std::runtime_error("PolylineCurve closing segment has zero length.");
      }
      total += closing;
   }

   node_parameters_.assign(nodes_.size(), 0.0);
   for (std::size_t i = 0; i < nodes_.size(); ++i)
   {
      node_parameters_[i] = cumulative[i] / total;
   }
}

CurveInterval PolylineCurve::domain() const
{
   return CurveInterval{0.0, 1.0};
}

int PolylineCurve::node_count() const
{
   return static_cast<int>(nodes_.size());
}

int PolylineCurve::segment_count() const
{
   return static_cast<int>(closed_ ? nodes_.size() : nodes_.size() - 1);
}

CurvePoint PolylineCurve::Point(double lambda) const
{
   const auto loc = Locate(nodes_, node_parameters_, closed_, lambda);
   return {(1.0 - loc.t) * loc.a[0] + loc.t * loc.b[0],
           (1.0 - loc.t) * loc.a[1] + loc.t * loc.b[1]};
}

CurvePoint PolylineCurve::Tangent(double lambda) const
{
   const auto loc = Locate(nodes_, node_parameters_, closed_, lambda);
   const double inv_dl = 1.0 / (loc.lambda_b - loc.lambda_a);
   return {(loc.b[0] - loc.a[0]) * inv_dl, (loc.b[1] - loc.a[1]) * inv_dl};
}

CurvePoint PolylineCurve::Normal(double lambda) const
{
   const CurvePoint t = Tangent(lambda);
   const double inv_norm = 1.0 / std::hypot(t[0], t[1]);
   // orientation = +1 (nodes in order, domain on LEFT): outward = right of tangent.
   // orientation = -1 (nodes in order, domain on RIGHT): outward = left of tangent.
   const double sign = static_cast<double>(orientation_);
   return {sign * t[1] * inv_norm, -sign * t[0] * inv_norm};
}

double PolylineCurve::ParameterOf(double x, double y) const
{
   const std::size_t n = nodes_.size();
   const std::size_t seg_count = closed_ ? n : n - 1;
   double best_lambda = 0.0;
   double best_dist2 = std::numeric_limits<double>::infinity();
   for (std::size_t k = 0; k < seg_count; ++k)
   {
      const CurvePoint &a = nodes_[k];
      const CurvePoint b = (closed_ && k + 1 == n) ? nodes_[0] : nodes_[k + 1];
      const double dx = b[0] - a[0];
      const double dy = b[1] - a[1];
      const double len2 = dx * dx + dy * dy;
      const double raw = ((x - a[0]) * dx + (y - a[1]) * dy) / len2;
      const double t = std::max(0.0, std::min(1.0, raw));
      const double px = a[0] + t * dx;
      const double py = a[1] + t * dy;
      const double d2 = (px - x) * (px - x) + (py - y) * (py - y);
      if (d2 < best_dist2)
      {
         best_dist2 = d2;
         const double la = node_parameters_[k];
         const double lb = (closed_ && k + 1 == n) ? 1.0 : node_parameters_[k + 1];
         best_lambda = (1.0 - t) * la + t * lb;
      }
   }
   return best_lambda;
}

} // namespace callaway
