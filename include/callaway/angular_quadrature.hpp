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

class AngularQuadrature
{
public:
   AngularQuadrature() = default;
   AngularQuadrature(const VelocityMeshSettings &settings, double group_velocity);

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

   static void GaussLegendre(int n, double a, double b,
                             std::vector<double> &points,
                             std::vector<double> &weights);

private:
   int polar_count_ = 0;
   int azimuthal_count_ = 0;
   std::vector<Direction> directions_;
};

} // namespace callaway
