#pragma once

#include "callaway/config.hpp"

#include <vector>

namespace callaway
{

constexpr double Pi = 3.141592653589793238462643383279502884;

struct Direction
{
   double theta = 0.0;
   double phi = 0.0;
   double weight = 0.0;
   double cx = 0.0;
   double cy = 0.0;
};

// Two-dimensional vs full three-dimensional angular quadrature. The 2D
// mode (polar_angles == 1) collapses the quadrature to the equatorial
// circle: cx = vg cos(phi), cy = vg sin(phi), weight = wphi, sum = 2π.
// Used to reproduce 2D-physics benchmarks (e.g. the AGEFEM porous
// reference figure). The 3D mode (polar_angles >= 2) integrates over
// the full sphere with weight sin(theta) wtheta wphi summing to 4π;
// in this convention cx = vg cos(theta) and cy = vg sin(theta) cos(phi).
enum class AngularMode
{
   ThreeD,
   TwoD
};

class AngularQuadrature
{
public:
   AngularQuadrature() = default;
   AngularQuadrature(const VelocityMeshSettings &settings, double group_velocity);

   AngularMode mode() const { return mode_; }
   bool is_2d() const { return mode_ == AngularMode::TwoD; }
   int polar_count() const { return polar_count_; }
   int azimuthal_count() const { return azimuthal_count_; }
   int size() const { return static_cast<int>(directions_.size()); }

   const Direction &operator[](int index) const { return directions_.at(index); }
   const std::vector<Direction> &directions() const { return directions_; }

   double SumWeights() const;
   double MomentCx() const;
   double MomentCy() const;
   double MomentCxCx() const;
   double MomentCyCy() const;

   // Normalisation conventions. equilibrium_normalization() is the
   // analytic value the quadrature integrates to (4π in 3D, 2π in 2D)
   // and is used in equilibrium distributions e_eq = Cv T /
   // equilibrium_normalization(). moment_factor() is the dimensional
   // factor in the Callaway normal-mode source term: 3 in 3D (since
   // ∫ s_i s_j dΩ / 4π = δ_ij / 3) and 2 in 2D (since ∫ s_i s_j dφ /
   // 2π = δ_ij / 2).
   double equilibrium_normalization() const;
   double moment_factor() const;

   // Index of the direction obtained by negating the cx component while
   // keeping cy unchanged — used for specular reflection at a vertical
   // wall. The mapping depends on the quadrature mode.
   int CxFlipPartner(int angle) const;

   static void GaussLegendre(int n, double a, double b,
                             std::vector<double> &points,
                             std::vector<double> &weights);

private:
   int polar_count_ = 0;
   int azimuthal_count_ = 0;
   AngularMode mode_ = AngularMode::ThreeD;
   std::vector<Direction> directions_;
};

} // namespace callaway
