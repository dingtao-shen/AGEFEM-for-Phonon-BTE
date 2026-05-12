#include "callaway/angular_quadrature.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace callaway
{

AngularQuadrature::AngularQuadrature(const VelocityMeshSettings &settings, double group_velocity)
   : polar_count_(settings.polar_angles),
     azimuthal_count_(settings.azimuthal_angles)
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
