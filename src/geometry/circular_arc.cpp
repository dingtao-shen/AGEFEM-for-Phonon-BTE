#include "callaway/geometry/circular_arc.hpp"

#include <cmath>
#include <stdexcept>

namespace callaway
{
namespace
{
constexpr double kTwoPi = 6.283185307179586476925286766559;
}

CircularArc::CircularArc(CurvePoint center, double radius, int orientation)
   : center_(center), radius_(radius), orientation_(orientation)
{
   if (radius_ <= 0.0)
   {
      throw std::runtime_error("CircularArc requires a positive radius.");
   }
   if (orientation_ != 1 && orientation_ != -1)
   {
      throw std::runtime_error("CircularArc orientation must be +1 (ccw) or -1 (cw).");
   }
}

CurveInterval CircularArc::domain() const
{
   return CurveInterval{0.0, 1.0};
}

CurvePoint CircularArc::Point(double lambda) const
{
   const double angle = orientation_ * kTwoPi * lambda;
   return {center_[0] + radius_ * std::cos(angle),
           center_[1] + radius_ * std::sin(angle)};
}

CurvePoint CircularArc::Tangent(double lambda) const
{
   const double angle = orientation_ * kTwoPi * lambda;
   const double scale = radius_ * orientation_ * kTwoPi;
   return {-scale * std::sin(angle), scale * std::cos(angle)};
}

CurvePoint CircularArc::Normal(double lambda) const
{
   // Right-of-tangent (Ty, -Tx), normalized. The sigma sign baked into
   // Tangent's direction already produces the correct domain-outward sense:
   //   sigma = +1 (ccw outer boundary) -> outward = radially outward
   //   sigma = -1 (cw hole boundary)   -> outward = toward center
   const CurvePoint t = Tangent(lambda);
   const double inv_norm = 1.0 / std::hypot(t[0], t[1]);
   return {t[1] * inv_norm, -t[0] * inv_norm};
}

double CircularArc::ParameterOf(double x, double y) const
{
   // atan2 returns the angle in [-pi, pi]; map back to lambda in [0, 1).
   const double angle = std::atan2(y - center_[1], x - center_[0]);
   double lambda = angle / (orientation_ * kTwoPi);
   lambda -= std::floor(lambda);
   return lambda;
}

} // namespace callaway
