#include "callaway/angular_quadrature.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace callaway
{

AngularQuadrature::AngularQuadrature(const VelocityMeshSettings &settings, double group_velocity)
   : polar_count_(settings.polar_angles),
     azimuthal_count_(settings.azimuthal_angles),
     mode_(settings.polar_angles == 1 ? AngularMode::TwoD : AngularMode::ThreeD)
{
   if (polar_count_ <= 0)
   {
      throw std::runtime_error("polar_angles must be positive.");
   }
   if (azimuthal_count_ <= 0 || azimuthal_count_ % 2 != 0)
   {
      throw std::runtime_error("azimuthal_angles must be a positive even integer.");
   }

   std::vector<double> theta;
   std::vector<double> wtheta;
   GaussLegendre(polar_count_, 0.0, Pi, theta, wtheta);
   std::reverse(theta.begin(), theta.end());
   std::reverse(wtheta.begin(), wtheta.end());

   std::vector<double> phi(azimuthal_count_);
   std::vector<double> wphi(azimuthal_count_);
   std::vector<double> half_phi;
   std::vector<double> half_wphi;

   const int half = azimuthal_count_ / 2;
   GaussLegendre(half, 0.0, Pi, half_phi, half_wphi);
   for (int i = 0; i < half; ++i)
   {
      phi[i] = half_phi[half - 1 - i];
      wphi[i] = half_wphi[half - 1 - i];
   }

   GaussLegendre(half, Pi, 2.0 * Pi, half_phi, half_wphi);
   for (int i = 0; i < half; ++i)
   {
      phi[half + i] = half_phi[half - 1 - i];
      wphi[half + i] = half_wphi[half - 1 - i];
   }

   directions_.reserve(static_cast<std::size_t>(polar_count_ * azimuthal_count_));
   if (mode_ == AngularMode::TwoD)
   {
      // 2D circular quadrature: a single polar slot at theta=π/2 with all
      // phonons in the (x, y) plane. cx = vg cos(phi), cy = vg sin(phi),
      // weight = wphi summing to 2π. Matches the AGEFEM prototype's
      // VelocityMesh layout for the porous benchmark.
      for (int ja = 0; ja < azimuthal_count_; ++ja)
      {
         Direction direction;
         direction.theta = 0.5 * Pi;
         direction.phi = phi[ja];
         direction.weight = wphi[ja];
         direction.cx = group_velocity * std::cos(phi[ja]);
         direction.cy = group_velocity * std::sin(phi[ja]);
         directions_.push_back(direction);
      }
   }
   else
   {
      // 3D spherical quadrature: theta is the polar angle from the +x
      // axis, phi is the azimuthal angle around the +x axis. Weight is
      // sin(theta) wtheta wphi and sums to 4π.
      for (int jp = 0; jp < polar_count_; ++jp)
      {
         for (int ja = 0; ja < azimuthal_count_; ++ja)
         {
            Direction direction;
            direction.theta = theta[jp];
            direction.phi = phi[ja];
            direction.weight = std::sin(theta[jp]) * wtheta[jp] * wphi[ja];
            direction.cx = group_velocity * std::cos(theta[jp]);
            direction.cy = group_velocity * std::sin(theta[jp]) * std::cos(phi[ja]);
            directions_.push_back(direction);
         }
      }
   }
}

double AngularQuadrature::equilibrium_normalization() const
{
   return (mode_ == AngularMode::TwoD) ? (2.0 * Pi) : (4.0 * Pi);
}

double AngularQuadrature::moment_factor() const
{
   return (mode_ == AngularMode::TwoD) ? 2.0 : 3.0;
}

int AngularQuadrature::CxFlipPartner(int angle) const
{
   if (mode_ == AngularMode::TwoD)
   {
      // 2D vertical-wall reflection: cos(phi) -> -cos(phi). The phi
      // quadrature is laid out as two halves: indices [0, half) carry
      // GL nodes in [0, π] (high-to-low), [half, 2*half) in [π, 2π]
      // (high-to-low). Symmetric GL nodes give the partner index by
      // reflection within each half.
      const int half = azimuthal_count_ / 2;
      if (angle < half)
      {
         return (half - 1) - angle;
      }
      return (3 * half - 1) - angle;
   }
   // 3D vertical-wall reflection: cx = vg cos(theta) flips when
   // theta -> π - theta. Polar GL nodes are symmetric about π/2 so the
   // partner is the polar index reflected within the polar bank.
   const int jp = angle / azimuthal_count_;
   const int ja = angle - jp * azimuthal_count_;
   return (polar_count_ - 1 - jp) * azimuthal_count_ + ja;
}

double AngularQuadrature::SumWeights() const
{
   return std::accumulate(directions_.begin(), directions_.end(), 0.0,
                          [](double sum, const Direction &d) { return sum + d.weight; });
}

double AngularQuadrature::MomentCx() const
{
   return std::accumulate(directions_.begin(), directions_.end(), 0.0,
                          [](double sum, const Direction &d) { return sum + d.cx * d.weight; });
}

double AngularQuadrature::MomentCy() const
{
   return std::accumulate(directions_.begin(), directions_.end(), 0.0,
                          [](double sum, const Direction &d) { return sum + d.cy * d.weight; });
}

double AngularQuadrature::MomentCxCx() const
{
   return std::accumulate(directions_.begin(), directions_.end(), 0.0,
                          [](double sum, const Direction &d) { return sum + d.cx * d.cx * d.weight; });
}

double AngularQuadrature::MomentCyCy() const
{
   return std::accumulate(directions_.begin(), directions_.end(), 0.0,
                          [](double sum, const Direction &d) { return sum + d.cy * d.cy * d.weight; });
}

void AngularQuadrature::GaussLegendre(int n, double a, double b,
                                      std::vector<double> &points,
                                      std::vector<double> &weights)
{
   if (n <= 0)
   {
      throw std::runtime_error("GaussLegendre requires a positive order.");
   }

   points.assign(static_cast<std::size_t>(n), 0.0);
   weights.assign(static_cast<std::size_t>(n), 0.0);

   if (n == 1)
   {
      points[0] = 0.5 * (a + b);
      weights[0] = b - a;
      return;
   }

   const int n1 = n;
   const int n2 = n + 1;
   std::vector<double> y(static_cast<std::size_t>(n1));
   std::vector<double> y0(static_cast<std::size_t>(n1), 2.0);
   std::vector<double> lp(static_cast<std::size_t>(n1), 0.0);
   std::vector<std::vector<double>> l(static_cast<std::size_t>(n1),
                                      std::vector<double>(static_cast<std::size_t>(n2), 0.0));

   for (int i = 0; i < n1; ++i)
   {
      const double idx = static_cast<double>(i + 1);
      y[i] = std::cos((2.0 * static_cast<double>(i) + 1.0) * Pi /
                      (2.0 * static_cast<double>(n - 1) + 2.0))
             + 0.27 / static_cast<double>(n1) *
                   std::sin(Pi * (-1.0 + idx * 2.0 / static_cast<double>(n1 - 1)) *
                            static_cast<double>(n - 1) / static_cast<double>(n2));
   }

   for (;;)
   {
      for (int i = 0; i < n1; ++i)
      {
         l[i][0] = 1.0;
         l[i][1] = y[i];
      }

      for (int k = 2; k <= n1; ++k)
      {
         for (int i = 0; i < n1; ++i)
         {
            l[i][k] = ((2.0 * static_cast<double>(k) - 1.0) * y[i] * l[i][k - 1] -
                       static_cast<double>(k - 1) * l[i][k - 2]) /
                      static_cast<double>(k);
         }
      }

      double max_delta = 0.0;
      for (int i = 0; i < n1; ++i)
      {
         lp[i] = static_cast<double>(n2) * (l[i][n1 - 1] - y[i] * l[i][n2 - 1]) /
                 (1.0 - y[i] * y[i]);
         y0[i] = y[i];
         y[i] = y0[i] - l[i][n2 - 1] / lp[i];
         max_delta = std::max(max_delta, std::abs(y[i] - y0[i]));
      }

      if (max_delta < 1.0e-13) { break; }
   }

   for (int i = 0; i < n1; ++i)
   {
      points[i] = (a * (1.0 - y[i]) + b * (1.0 + y[i])) / 2.0;
      weights[i] = static_cast<double>(n2 * n2) * (b - a) /
                   ((1.0 - y[i] * y[i]) * lp[i] * lp[i]) /
                   static_cast<double>(n1 * n1);
   }
}

} // namespace callaway
