#pragma once

#include "callaway/geometry/boundary_curve.hpp"

#include <vector>

namespace callaway
{

// Boundary reconstructed as a polyline from a sequence of sampling nodes
// (the paper's C_P(lambda)). The parametrization is piecewise-linear in
// cumulative chord length, normalized to [0, 1]; tangents are piecewise
// constant. A node parameter resolves to the segment that follows it
// (right-continuous), so Tangent/Normal are well-defined at breakpoints.
//
// orientation = +1 traces the polyline in node order with the domain
// interior on the left, -1 reverses; together with `closed` it fixes the
// domain-outward normal sense.
//
// node_parameters() gives the parameter values of the sampling nodes —
// the segment breakpoints. These are needed by the AGE area quadrature,
// which sub-triangulates an AGE element on the polyline segments that
// fall within its parameter sub-interval.
class PolylineCurve final : public BoundaryCurve
{
public:
   PolylineCurve(std::vector<CurvePoint> nodes, bool closed, int orientation);

   CurveKind kind() const override { return CurveKind::Polyline; }
   CurveInterval domain() const override; // {0, 1}
   bool is_closed() const override { return closed_; }

   CurvePoint Point(double lambda) const override;
   CurvePoint Tangent(double lambda) const override;
   CurvePoint Normal(double lambda) const override;
   double ParameterOf(double x, double y) const override;

   int node_count() const;
   int segment_count() const;
   const std::vector<CurvePoint> &nodes() const { return nodes_; }
   const std::vector<double> &node_parameters() const { return node_parameters_; }
   int orientation() const { return orientation_; }

private:
   std::vector<CurvePoint> nodes_;
   std::vector<double> node_parameters_;
   bool closed_ = false;
   int orientation_ = 1;
};

} // namespace callaway
