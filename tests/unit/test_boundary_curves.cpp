#include "callaway/geometry/boundary_curve.hpp"
#include "callaway/geometry/circular_arc.hpp"
#include "callaway/geometry/nurbs_curve.hpp"
#include "callaway/geometry/polyline_curve.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

namespace
{

constexpr double kTwoPi = 6.283185307179586476925286766559;

void CheckClose(double actual, double expected, double tol, const char *what)
{
   if (!(std::abs(actual - expected) <= tol))
   {
      std::cerr << "FAIL " << what << ": got " << actual
                << ", expected " << expected
                << ", diff " << std::abs(actual - expected) << "\n";
      assert(false);
   }
}

void CheckPoint(callaway::CurvePoint actual, callaway::CurvePoint expected,
                double tol, const char *what)
{
   CheckClose(actual[0], expected[0], tol, what);
   CheckClose(actual[1], expected[1], tol, what);
}

void TestCircularArcCcw()
{
   const callaway::CircularArc arc({{0.0, 0.0}}, 1.0, +1);
   CheckClose(arc.domain().begin, 0.0, 0.0, "arc(ccw) domain.begin");
   CheckClose(arc.domain().end, 1.0, 0.0, "arc(ccw) domain.end");
   assert(arc.is_closed());
   assert(arc.kind() == callaway::CurveKind::Parametric);

   CheckPoint(arc.Point(0.00), {{1.0, 0.0}}, 1.0e-13, "arc(ccw) Point(0)");
   CheckPoint(arc.Point(0.25), {{0.0, 1.0}}, 1.0e-13, "arc(ccw) Point(1/4)");
   CheckPoint(arc.Point(0.50), {{-1.0, 0.0}}, 1.0e-13, "arc(ccw) Point(1/2)");
   CheckPoint(arc.Point(0.75), {{0.0, -1.0}}, 1.0e-13, "arc(ccw) Point(3/4)");

   // |Tangent| = 2*pi*r = 2*pi at every parameter (constant-speed arc-length up to 2*pi).
   for (int k = 0; k < 16; ++k)
   {
      const double lambda = static_cast<double>(k) / 16.0;
      const auto t = arc.Tangent(lambda);
      const double mag = std::hypot(t[0], t[1]);
      CheckClose(mag, kTwoPi, 1.0e-12, "arc(ccw) |Tangent|");
   }

   // Outward normal of a CCW disk-outer boundary is radially outward.
   CheckPoint(arc.Normal(0.00), {{1.0, 0.0}}, 1.0e-13, "arc(ccw) Normal(0)");
   CheckPoint(arc.Normal(0.25), {{0.0, 1.0}}, 1.0e-13, "arc(ccw) Normal(1/4)");
   CheckPoint(arc.Normal(0.50), {{-1.0, 0.0}}, 1.0e-13, "arc(ccw) Normal(1/2)");

   // ParameterOf round-trip.
   for (int k = 0; k < 11; ++k)
   {
      const double lambda = static_cast<double>(k) / 11.0;
      const auto p = arc.Point(lambda);
      const double rt = arc.ParameterOf(p[0], p[1]);
      CheckClose(rt, lambda, 1.0e-13, "arc(ccw) ParameterOf round-trip");
   }
}

void TestCircularArcCw()
{
   // Hole boundary: center (0.5, -0.25), radius 0.4, traversed CW so the
   // domain (material outside the hole) stays on the LEFT of the tangent.
   const callaway::CircularArc arc({{0.5, -0.25}}, 0.4, -1);

   // Point at lambda=0 is at angle 0 -> (cx + r, cy).
   CheckPoint(arc.Point(0.0), {{0.9, -0.25}}, 1.0e-13, "arc(cw) Point(0)");
   // Lambda=0.25 traces CW by pi/2 -> angle -pi/2 -> (cx, cy - r).
   CheckPoint(arc.Point(0.25), {{0.5, -0.65}}, 1.0e-13, "arc(cw) Point(1/4)");

   // For a CW-traced hole, outward normal points TOWARD center.
   // At lambda=0 (point (cx+r, cy)) outward = (-1, 0) (toward center).
   CheckPoint(arc.Normal(0.0), {{-1.0, 0.0}}, 1.0e-13, "arc(cw) Normal(0)");

   // |Tangent| = 2*pi*r constant.
   const auto t = arc.Tangent(0.13);
   CheckClose(std::hypot(t[0], t[1]), kTwoPi * 0.4, 1.0e-12, "arc(cw) |Tangent|");

   // Round-trip several points.
   for (int k = 0; k < 11; ++k)
   {
      const double lambda = static_cast<double>(k) / 11.0;
      const auto p = arc.Point(lambda);
      const double rt = arc.ParameterOf(p[0], p[1]);
      CheckClose(rt, lambda, 1.0e-13, "arc(cw) ParameterOf round-trip");
   }
}

void TestPolylineSquareClosed()
{
   // CCW unit square, closed. Domain inside the square is on the LEFT of
   // each edge tangent, so the outward normal is right-of-tangent.
   std::vector<callaway::CurvePoint> nodes = {
      {{0.0, 0.0}}, {{1.0, 0.0}}, {{1.0, 1.0}}, {{0.0, 1.0}}
   };
   const callaway::PolylineCurve poly(nodes, /*closed=*/true, /*orientation=*/+1);

   assert(poly.kind() == callaway::CurveKind::Polyline);
   assert(poly.is_closed());
   assert(poly.node_count() == 4);
   assert(poly.segment_count() == 4);

   // Total perimeter = 4, so each corner is at lambda = 0, 0.25, 0.5, 0.75.
   const auto &lambdas = poly.node_parameters();
   CheckClose(lambdas[0], 0.00, 1.0e-15, "square lambda[0]");
   CheckClose(lambdas[1], 0.25, 1.0e-15, "square lambda[1]");
   CheckClose(lambdas[2], 0.50, 1.0e-15, "square lambda[2]");
   CheckClose(lambdas[3], 0.75, 1.0e-15, "square lambda[3]");

   CheckPoint(poly.Point(0.000), {{0.0, 0.0}}, 1.0e-15, "square Point(0)");
   CheckPoint(poly.Point(0.125), {{0.5, 0.0}}, 1.0e-15, "square Point(0.125)");
   CheckPoint(poly.Point(0.375), {{1.0, 0.5}}, 1.0e-15, "square Point(0.375)");
   CheckPoint(poly.Point(0.875), {{0.0, 0.5}}, 1.0e-15, "square Point(0.875)");

   // Tangent magnitude = total_length = 4 everywhere.
   const auto t0 = poly.Tangent(0.125);
   CheckClose(std::hypot(t0[0], t0[1]), 4.0, 1.0e-13, "square |Tangent|");
   CheckPoint({{t0[0] / 4.0, t0[1] / 4.0}}, {{1.0, 0.0}}, 1.0e-13, "square Tangent dir on edge 0");

   const auto t1 = poly.Tangent(0.375);
   CheckPoint({{t1[0] / 4.0, t1[1] / 4.0}}, {{0.0, 1.0}}, 1.0e-13, "square Tangent dir on edge 1");

   // Outward normals: edge 0 (bottom) -> down (0,-1); edge 1 (right) -> right (1,0);
   // edge 2 (top) -> up (0,1); edge 3 (left) -> left (-1,0).
   CheckPoint(poly.Normal(0.125), {{0.0, -1.0}}, 1.0e-13, "square Normal edge 0");
   CheckPoint(poly.Normal(0.375), {{1.0, 0.0}}, 1.0e-13, "square Normal edge 1");
   CheckPoint(poly.Normal(0.625), {{0.0, 1.0}}, 1.0e-13, "square Normal edge 2");
   CheckPoint(poly.Normal(0.875), {{-1.0, 0.0}}, 1.0e-13, "square Normal edge 3");

   // ParameterOf round-trip on segment midpoints.
   for (int k = 0; k < 4; ++k)
   {
      const double lambda = 0.125 + 0.25 * k;
      const auto p = poly.Point(lambda);
      const double rt = poly.ParameterOf(p[0], p[1]);
      CheckClose(rt, lambda, 1.0e-13, "square ParameterOf round-trip");
   }
}

void TestPolylineOpenOrientationFlip()
{
   // Open L-shape, two segments of length 1.
   std::vector<callaway::CurvePoint> nodes = {
      {{0.0, 0.0}}, {{1.0, 0.0}}, {{1.0, 1.0}}
   };
   const callaway::PolylineCurve poly_plus(nodes, /*closed=*/false, /*orientation=*/+1);
   const callaway::PolylineCurve poly_minus(nodes, /*closed=*/false, /*orientation=*/-1);

   assert(poly_plus.segment_count() == 2);
   CheckPoint(poly_plus.Point(0.25), {{0.5, 0.0}}, 1.0e-15, "L+ Point(0.25)");
   CheckPoint(poly_plus.Point(0.75), {{1.0, 0.5}}, 1.0e-15, "L+ Point(0.75)");

   // Orientation flip negates the normal.
   const auto n_plus = poly_plus.Normal(0.25);
   const auto n_minus = poly_minus.Normal(0.25);
   CheckPoint({{n_plus[0] + n_minus[0], n_plus[1] + n_minus[1]}},
              {{0.0, 0.0}}, 1.0e-13, "orientation flip negates normal");
}

void TestNurbsUnitCircle()
{
   // Standard quadratic rational B-spline representation of the unit circle.
   // 9 control points, weights {1, sqrt(2)/2}, knot vector with double knots
   // at the quarter points.
   const double s = std::sqrt(2.0) / 2.0;
   std::vector<callaway::CurvePoint> cp = {
      {{ 1.0,  0.0}},
      {{ 1.0,  1.0}},
      {{ 0.0,  1.0}},
      {{-1.0,  1.0}},
      {{-1.0,  0.0}},
      {{-1.0, -1.0}},
      {{ 0.0, -1.0}},
      {{ 1.0, -1.0}},
      {{ 1.0,  0.0}}
   };
   std::vector<double> w = {1.0, s, 1.0, s, 1.0, s, 1.0, s, 1.0};
   std::vector<double> knots = {
      0.0,  0.0,  0.0,
      0.25, 0.25,
      0.5,  0.5,
      0.75, 0.75,
      1.0,  1.0,  1.0
   };

   const callaway::NurbsCurve nurbs(/*degree=*/2, cp, w, knots, /*closed=*/true);

   const auto d = nurbs.domain();
   CheckClose(d.begin, 0.0, 0.0, "nurbs domain.begin");
   CheckClose(d.end, 1.0, 0.0, "nurbs domain.end");

   // Cardinal points are exactly the CP at multiple knots.
   CheckPoint(nurbs.Point(0.00), {{ 1.0,  0.0}}, 1.0e-14, "nurbs cardinal Point(0)");
   CheckPoint(nurbs.Point(0.25), {{ 0.0,  1.0}}, 1.0e-14, "nurbs cardinal Point(1/4)");
   CheckPoint(nurbs.Point(0.50), {{-1.0,  0.0}}, 1.0e-14, "nurbs cardinal Point(1/2)");
   CheckPoint(nurbs.Point(0.75), {{ 0.0, -1.0}}, 1.0e-14, "nurbs cardinal Point(3/4)");
   CheckPoint(nurbs.Point(1.00), {{ 1.0,  0.0}}, 1.0e-14, "nurbs cardinal Point(1)");

   // Every sampled point lies exactly on the unit circle.
   constexpr int kSamples = 200;
   for (int k = 1; k < kSamples; ++k)
   {
      const double lambda = static_cast<double>(k) / kSamples;
      const auto p = nurbs.Point(lambda);
      const double r = std::hypot(p[0], p[1]);
      CheckClose(r, 1.0, 1.0e-12, "nurbs |Point| = 1");
   }

   // Tangent is perpendicular to the radial direction for a circle: T . P = 0.
   for (int k = 1; k < 20; ++k)
   {
      const double lambda = static_cast<double>(k) / 20.0;
      const auto p = nurbs.Point(lambda);
      const auto t = nurbs.Tangent(lambda);
      const double dot = p[0] * t[0] + p[1] * t[1];
      CheckClose(dot, 0.0, 1.0e-11, "nurbs T . P = 0");
   }

   // CCW unit-circle Normal at cardinal points is radially outward.
   CheckPoint(nurbs.Normal(0.00), {{ 1.0,  0.0}}, 1.0e-12, "nurbs Normal(0)");
   CheckPoint(nurbs.Normal(0.25), {{ 0.0,  1.0}}, 1.0e-12, "nurbs Normal(1/4)");
   CheckPoint(nurbs.Normal(0.50), {{-1.0,  0.0}}, 1.0e-12, "nurbs Normal(1/2)");

   // ParameterOf round-trip at non-cardinal lambdas (no knot-multiplicity ambiguity).
   const double samples[] = {0.05, 0.13, 0.30, 0.42, 0.58, 0.66, 0.81, 0.93};
   for (double lambda : samples)
   {
      const auto p = nurbs.Point(lambda);
      const double rt = nurbs.ParameterOf(p[0], p[1]);
      CheckClose(rt, lambda, 1.0e-10, "nurbs ParameterOf round-trip");
   }
}

void TestNurbsAgainstCircularArc()
{
   // The 9-CP NURBS circle and a CircularArc should produce the same physical
   // circle, even though their parametrizations differ. Cross-check the Normal
   // directions at the cardinal points (where both parametrizations land at the
   // same physical location).
   const double s = std::sqrt(2.0) / 2.0;
   std::vector<callaway::CurvePoint> cp = {
      {{ 1.0,  0.0}}, {{ 1.0,  1.0}}, {{ 0.0,  1.0}},
      {{-1.0,  1.0}}, {{-1.0,  0.0}}, {{-1.0, -1.0}},
      {{ 0.0, -1.0}}, {{ 1.0, -1.0}}, {{ 1.0,  0.0}}
   };
   std::vector<double> w = {1.0, s, 1.0, s, 1.0, s, 1.0, s, 1.0};
   std::vector<double> knots = {
      0.0, 0.0, 0.0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1.0, 1.0, 1.0
   };
   const callaway::NurbsCurve nurbs(2, cp, w, knots, true);
   const callaway::CircularArc arc({{0.0, 0.0}}, 1.0, +1);

   const double cardinal[] = {0.0, 0.25, 0.5, 0.75};
   for (double lambda : cardinal)
   {
      CheckPoint(nurbs.Point(lambda), arc.Point(lambda), 1.0e-12,
                 "nurbs vs arc Point at cardinal lambda");
      CheckPoint(nurbs.Normal(lambda), arc.Normal(lambda), 1.0e-12,
                 "nurbs vs arc Normal at cardinal lambda");
   }
}

} // namespace

int main()
{
   TestCircularArcCcw();
   TestCircularArcCw();
   TestPolylineSquareClosed();
   TestPolylineOpenOrientationFlip();
   TestNurbsUnitCircle();
   TestNurbsAgainstCircularArc();
   std::cout << "test_boundary_curves: all checks passed.\n";
   return 0;
}
