#include "callaway/geometry/nurbs_curve.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace callaway
{
namespace
{

// Find the knot span index k such that knots[k] <= lambda < knots[k+1],
// clamped to [degree, n] where n = control_count - 1.
int FindKnotSpan(int degree, int n, const std::vector<double> &knots, double lambda)
{
   if (lambda >= knots[n + 1]) { return n; }
   if (lambda <= knots[degree]) { return degree; }
   int lo = degree, hi = n + 1;
   int mid = (lo + hi) / 2;
   while (lambda < knots[mid] || lambda >= knots[mid + 1])
   {
      if (lambda < knots[mid]) { hi = mid; }
      else                     { lo = mid; }
      mid = (lo + hi) / 2;
   }
   return mid;
}

// Cox-de Boor recursion. basis[d][j] = N_{span - d + j, d}(lambda) for
// d = 0, 1, ..., max_degree and j = 0, ..., d. The 0/0 NURBS convention
// (repeated knot in denominator) is handled by skipping the term.
void CoxDeBoor(int span, int max_degree, const std::vector<double> &knots,
               double lambda, std::vector<std::vector<double>> &basis)
{
   basis.assign(max_degree + 1, {});
   basis[0] = {1.0};
   for (int d = 1; d <= max_degree; ++d)
   {
      basis[d].assign(d + 1, 0.0);
      for (int j = 0; j <= d; ++j)
      {
         const int i = span - d + j;
         double sum = 0.0;
         if (j > 0)
         {
            const double denom = knots[i + d] - knots[i];
            if (denom != 0.0)
            {
               sum += (lambda - knots[i]) / denom * basis[d - 1][j - 1];
            }
         }
         if (j < d)
         {
            const double denom = knots[i + d + 1] - knots[i + 1];
            if (denom != 0.0)
            {
               sum += (knots[i + d + 1] - lambda) / denom * basis[d - 1][j];
            }
         }
         basis[d][j] = sum;
      }
   }
}

// First derivatives of the p+1 nonzero degree-p basis functions, using
// N'_{i, p} = p * (N_{i, p-1} / (u_{i+p} - u_i) - N_{i+1, p-1} / (u_{i+p+1} - u_{i+1})).
// dN[j] corresponds to N'_{span - p + j, p}.
void BasisDerivatives(int span, int degree, const std::vector<double> &knots,
                      const std::vector<std::vector<double>> &basis,
                      std::vector<double> &dN)
{
   dN.assign(degree + 1, 0.0);
   for (int j = 0; j <= degree; ++j)
   {
      const int i = span - degree + j;
      double v = 0.0;
      if (j > 0)
      {
         const double denom = knots[i + degree] - knots[i];
         if (denom != 0.0)
         {
            v += basis[degree - 1][j - 1] / denom;
         }
      }
      if (j < degree)
      {
         const double denom = knots[i + degree + 1] - knots[i + 1];
         if (denom != 0.0)
         {
            v -= basis[degree - 1][j] / denom;
         }
      }
      dN[j] = degree * v;
   }
}

// Evaluate the NURBS Point and Tangent at a single parameter value.
void EvaluatePointAndTangent(int degree,
                             const std::vector<CurvePoint> &control_points,
                             const std::vector<double> &weights,
                             const std::vector<double> &knots,
                             double lambda,
                             CurvePoint &point,
                             CurvePoint &tangent)
{
   const int n = static_cast<int>(control_points.size()) - 1;
   const int span = FindKnotSpan(degree, n, knots, lambda);
   std::vector<std::vector<double>> basis;
   CoxDeBoor(span, degree, knots, lambda, basis);
   std::vector<double> dN;
   BasisDerivatives(span, degree, knots, basis, dN);

   double ax = 0.0, ay = 0.0, w = 0.0;
   double dax = 0.0, day = 0.0, dw = 0.0;
   for (int j = 0; j <= degree; ++j)
   {
      const int i = span - degree + j;
      const double N = basis[degree][j];
      const double Nprime = dN[j];
      const double wi = weights[i];
      const double xi = control_points[i][0];
      const double yi = control_points[i][1];

      const double wx = wi * xi;
      const double wy = wi * yi;
      ax += N * wx;
      ay += N * wy;
      w  += N * wi;

      dax += Nprime * wx;
      day += Nprime * wy;
      dw  += Nprime * wi;
   }

   if (w == 0.0)
   {
      throw std::runtime_error("NurbsCurve evaluation produced zero rational denominator.");
   }
   point = {ax / w, ay / w};
   tangent = {(dax - point[0] * dw) / w, (day - point[1] * dw) / w};
}

} // namespace

NurbsCurve::NurbsCurve(int degree,
                       std::vector<CurvePoint> control_points,
                       std::vector<double> weights,
                       std::vector<double> knots,
                       bool closed)
   : degree_(degree),
     control_points_(std::move(control_points)),
     weights_(std::move(weights)),
     knots_(std::move(knots)),
     closed_(closed)
{
   if (degree_ < 1)
   {
      throw std::runtime_error("NurbsCurve degree must be at least 1.");
   }
   if (control_points_.size() < static_cast<std::size_t>(degree_ + 1))
   {
      throw std::runtime_error("NurbsCurve requires at least degree+1 control points.");
   }
   if (weights_.size() != control_points_.size())
   {
      throw std::runtime_error("NurbsCurve weights count must match control points count.");
   }
   if (knots_.size() != control_points_.size() + static_cast<std::size_t>(degree_ + 1))
   {
      throw std::runtime_error("NurbsCurve knots count must equal control_points.size() + degree + 1.");
   }
   for (double w : weights_)
   {
      if (w <= 0.0)
      {
         throw std::runtime_error("NurbsCurve weights must be positive.");
      }
   }
   for (std::size_t i = 1; i < knots_.size(); ++i)
   {
      if (knots_[i] < knots_[i - 1])
      {
         throw std::runtime_error("NurbsCurve knots must be non-decreasing.");
      }
   }
}

CurveInterval NurbsCurve::domain() const
{
   const int n = static_cast<int>(control_points_.size()) - 1;
   return CurveInterval{knots_[degree_], knots_[n + 1]};
}

CurvePoint NurbsCurve::Point(double lambda) const
{
   CurvePoint p, t;
   EvaluatePointAndTangent(degree_, control_points_, weights_, knots_, lambda, p, t);
   return p;
}

CurvePoint NurbsCurve::Tangent(double lambda) const
{
   CurvePoint p, t;
   EvaluatePointAndTangent(degree_, control_points_, weights_, knots_, lambda, p, t);
   return t;
}

CurvePoint NurbsCurve::Normal(double lambda) const
{
   const CurvePoint t = Tangent(lambda);
   const double inv_norm = 1.0 / std::hypot(t[0], t[1]);
   return {t[1] * inv_norm, -t[0] * inv_norm};
}

double NurbsCurve::ParameterOf(double x, double y) const
{
   const CurveInterval d = domain();
   const double span_length = d.end - d.begin;

   // Initial guess: nearest sample on a uniform discretization.
   constexpr int kSamples = 256;
   double best_lambda = d.begin;
   double best_dist2 = std::numeric_limits<double>::infinity();
   for (int s = 0; s <= kSamples; ++s)
   {
      const double lambda = d.begin + span_length * s / kSamples;
      const CurvePoint p = Point(lambda);
      const double dx = p[0] - x;
      const double dy = p[1] - y;
      const double dist2 = dx * dx + dy * dy;
      if (dist2 < best_dist2)
      {
         best_dist2 = dist2;
         best_lambda = lambda;
      }
   }

   // Newton refinement on f(lambda) = (C(lambda) - x_target) . C'(lambda).
   // We use f'(lambda) ~= |C'(lambda)|^2 (ignoring the curvature term), which
   // is enough for points known to lie on the curve.
   double lambda = best_lambda;
   constexpr int kMaxIter = 50;
   constexpr double kTol = 1.0e-13;
   for (int iter = 0; iter < kMaxIter; ++iter)
   {
      CurvePoint p, t;
      EvaluatePointAndTangent(degree_, control_points_, weights_, knots_, lambda, p, t);
      const double fx = p[0] - x;
      const double fy = p[1] - y;
      const double f = fx * t[0] + fy * t[1];
      const double tt = t[0] * t[0] + t[1] * t[1];
      if (tt <= 0.0) { break; }
      const double step = f / tt;
      lambda -= step;
      if (lambda < d.begin) { lambda = d.begin; }
      if (lambda > d.end)   { lambda = d.end; }
      if (std::abs(step) < kTol * std::max(1.0, span_length)) { break; }
      if (std::hypot(fx, fy) < kTol) { break; }
   }
   return lambda;
}

} // namespace callaway
