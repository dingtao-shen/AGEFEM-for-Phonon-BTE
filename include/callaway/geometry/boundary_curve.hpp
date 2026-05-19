#pragma once

#include <array>

namespace callaway
{

// Which curved-geometry family a BoundaryCurve belongs to. Determines the
// 2D area-quadrature strategy for an AGE element: the parametric Upsilon
// transformation (tensor product of 1D Gauss-Legendre rules) for smooth
// curves, sub-triangulation on segment breakpoints for polylines.
enum class CurveKind
{
   Parametric, // smooth parametric curve: circular arc, NURBS, analytic
   Polyline    // piecewise-linear reconstruction from sampling nodes
};

using CurvePoint = std::array<double, 2>;

// A closed parameter sub-interval [begin, end] on a curve, oriented so that
// increasing the parameter traces the boundary with the domain interior on
// the left (CCW element-boundary convention). For an AGE element, this is
// the portion of the bound curve that forms the element's curved edge.
struct CurveInterval
{
   double begin = 0.0;
   double end = 0.0;
};

// Uniform interface to an exact boundary curve.
//
// Every curved-geometry input — circular-arc formula, NURBS from a sidecar,
// reconstructed sampling-node polyline — is normalized to this interface
// before any numerics run. All AGE element processing (node placement
// Psi_n, edge quadrature, area quadrature Upsilon, sub-triangulation) is
// written against this interface alone, so the rest of the pipeline does
// not depend on how the geometry was specified.
class BoundaryCurve
{
public:
   virtual ~BoundaryCurve() = default;

   virtual CurveKind kind() const = 0;

   // Parameter domain [a, b] of the curve, and whether it is closed
   // (parameter a wraps to b at the same point).
   virtual CurveInterval domain() const = 0;
   virtual bool is_closed() const = 0;

   // C(lambda): physical coordinates of the curve point at parameter lambda.
   virtual CurvePoint Point(double lambda) const = 0;

   // C'(lambda): parametric tangent vector (not normalized). Its norm is
   // the edge Jacobian |J_C| used by 1D edge quadrature; it also enters
   // the Upsilon area Jacobian for parametric AGE elements.
   virtual CurvePoint Tangent(double lambda) const = 0;

   // Outward unit normal at parameter lambda — pointing OUT of the
   // computational domain (into a hole / away from the material). The
   // sense is fixed by the curve's orientation, supplied at construction.
   virtual CurvePoint Normal(double lambda) const = 0;

   // The parameter lambda such that Point(lambda) == (x, y), for a point
   // assumed to lie on the curve (a mesh vertex on a curved boundary
   // edge). Used by the pre-processor to bind mesh vertices to curve
   // sub-intervals. Implementations may use closed-form inversion,
   // segment search, or Newton projection depending on the curve type.
   virtual double ParameterOf(double x, double y) const = 0;
};

} // namespace callaway
