#include "callaway/nodal_basis.hpp"

#include <cmath>
#include <stdexcept>

namespace callaway
{
namespace
{

double PowInt(double x, int exponent)
{
   if (exponent == 0) { return 1.0; }
   return std::pow(x, static_cast<double>(exponent));
}

} // namespace

NodalBasis::NodalBasis(int order)
   : order_(order),
     triangle_dofs_(TriangleDofs(order)),
     face_dofs_(FaceDofs(order)),
     triangle_coefficients_(static_cast<std::size_t>(triangle_dofs_ * triangle_dofs_), 0.0),
     face_coefficients_(static_cast<std::size_t>(face_dofs_ * face_dofs_), 0.0)
{
   if (order_ < 1 || order_ > 4)
   {
      throw std::runtime_error("NodalBasis currently supports orders 1 through 4.");
   }
   BuildFaceBasis();
   BuildTriangleBasis();
}

int NodalBasis::TriangleDofs(int order)
{
   return (order + 1) * (order + 2) / 2;
}

int NodalBasis::FaceDofs(int order)
{
   return order + 1;
}

int NodalBasis::MonomialIndex(int x_power, int y_power)
{
   const int total = x_power + y_power;
   return total * (total + 1) / 2 + y_power;
}

double NodalBasis::TriangleCoefficient(int basis, int monomial) const
{
   return triangle_coefficients_.at(static_cast<std::size_t>(basis * triangle_dofs_ + monomial));
}

double NodalBasis::FaceCoefficient(int basis, int power) const
{
   return face_coefficients_.at(static_cast<std::size_t>(basis * face_dofs_ + power));
}

double NodalBasis::EvaluateTriangle(int basis, double xi, double eta) const
{
   double value = 0.0;
   int monomial = 0;
   for (int total = 0; total <= order_; ++total)
   {
      for (int y_power = 0; y_power <= total; ++y_power)
      {
         const int x_power = total - y_power;
         value += TriangleCoefficient(basis, monomial) *
                  PowInt(xi, x_power) * PowInt(eta, y_power);
         ++monomial;
      }
   }
   return value;
}

double NodalBasis::EvaluateFace(int basis, double t) const
{
   double value = 0.0;
   for (int power = 0; power <= order_; ++power)
   {
      value += FaceCoefficient(basis, power) * PowInt(t, power);
   }
   return value;
}

std::vector<double> NodalBasis::EvaluateTriangleAll(double xi, double eta) const
{
   std::vector<double> values(static_cast<std::size_t>(triangle_dofs_));
   for (int i = 0; i < triangle_dofs_; ++i)
   {
      values[static_cast<std::size_t>(i)] = EvaluateTriangle(i, xi, eta);
   }
   return values;
}

std::vector<double> NodalBasis::EvaluateFaceAll(double t) const
{
   std::vector<double> values(static_cast<std::size_t>(face_dofs_));
   for (int i = 0; i < face_dofs_; ++i)
   {
      values[static_cast<std::size_t>(i)] = EvaluateFace(i, t);
   }
   return values;
}

void NodalBasis::BuildFaceBasis()
{
   face_nodes_.assign(static_cast<std::size_t>(face_dofs_), 0.0);
   for (int i = 0; i < face_dofs_; ++i)
   {
      face_nodes_[static_cast<std::size_t>(i)] =
         -1.0 + 2.0 * static_cast<double>(i) / static_cast<double>(face_dofs_ - 1);
   }

   std::vector<double> scratch(static_cast<std::size_t>(face_dofs_), 0.0);
   for (int i = 0; i < face_dofs_; ++i)
   {
      face_coefficients_[static_cast<std::size_t>(i * face_dofs_)] = 1.0;
      int degree = 0;
      for (int j = 0; j < face_dofs_; ++j)
      {
         if (j == i) { continue; }

         for (int k = 0; k <= degree; ++k)
         {
            scratch[static_cast<std::size_t>(k)] =
               face_coefficients_[static_cast<std::size_t>(i * face_dofs_ + k)];
         }

         const double b0 = static_cast<double>(2 * (j + 1) - face_dofs_ - 1) /
                           static_cast<double>(2 * ((j + 1) - (i + 1)));
         const double b1 = -static_cast<double>(face_dofs_ - 1) /
                           static_cast<double>(2 * ((j + 1) - (i + 1)));

         face_coefficients_[static_cast<std::size_t>(i * face_dofs_)] = scratch[0] * b0;
         face_coefficients_[static_cast<std::size_t>(i * face_dofs_ + degree + 1)] =
            scratch[static_cast<std::size_t>(degree)] * b1;

         for (int l = 1; l <= degree; ++l)
         {
            face_coefficients_[static_cast<std::size_t>(i * face_dofs_ + l)] =
               scratch[static_cast<std::size_t>(l)] * b0 +
               scratch[static_cast<std::size_t>(l - 1)] * b1;
         }
         ++degree;
      }
   }
}

void NodalBasis::BuildTriangleBasis()
{
   triangle_nodes_.clear();
   triangle_nodes_.reserve(static_cast<std::size_t>(triangle_dofs_));
   for (int j = 0; j <= order_; ++j)
   {
      for (int i = 0; i <= order_ - j; ++i)
      {
         triangle_nodes_.push_back({static_cast<double>(i) / static_cast<double>(order_),
                                    static_cast<double>(j) / static_cast<double>(order_)});
      }
   }

   std::vector<double> a(static_cast<std::size_t>(triangle_dofs_), 0.0);
   std::vector<double> b(static_cast<std::size_t>(triangle_dofs_), 0.0);
   std::vector<double> tmp(static_cast<std::size_t>(triangle_dofs_), 0.0);

   int basis_index = 0;
   for (int j = 0; j <= order_; ++j)
   {
      for (int i = 0; i <= order_ - j; ++i)
      {
         const int k = order_ - i - j;
         std::fill(a.begin(), a.end(), 0.0);
         std::fill(b.begin(), b.end(), 0.0);

         if (i != 0)
         {
            a[static_cast<std::size_t>(1)] = static_cast<double>(order_) / static_cast<double>(i);
            if (i > 1)
            {
               for (int l = 1; l <= i - 1; ++l)
               {
                  tmp = a;
                  const double d0 = static_cast<double>(l) / static_cast<double>(l - i);
                  const double d1 = -static_cast<double>(order_) / static_cast<double>(l - i);

                  a[0] = tmp[0] * d0;
                  a[static_cast<std::size_t>((l + 1) * (l + 2) / 2)] =
                     tmp[static_cast<std::size_t>(l * (l + 1) / 2)] * d1;

                  for (int m = 1; m <= l; ++m)
                  {
                     a[static_cast<std::size_t>(m * (m + 1) / 2)] =
                        tmp[static_cast<std::size_t>(m * (m + 1) / 2)] * d0 +
                        tmp[static_cast<std::size_t>((m - 1) * m / 2)] * d1;
                  }
               }
            }
         }
         else
         {
            a[0] = 1.0;
         }

         if (j != 0)
         {
            b[static_cast<std::size_t>(2)] = static_cast<double>(order_) / static_cast<double>(j);
            if (j > 1)
            {
               for (int m = 1; m <= j - 1; ++m)
               {
                  tmp = b;
                  const double d0 = static_cast<double>(m) / static_cast<double>(m - j);
                  const double d1 = -static_cast<double>(order_) / static_cast<double>(m - j);

                  b[0] = tmp[0] * d0;
                  b[static_cast<std::size_t>((m + 2) * (m + 3) / 2 - 1)] =
                     tmp[static_cast<std::size_t>((m + 1) * (m + 2) / 2 - 1)] * d1;

                  for (int l = 1; l <= m; ++l)
                  {
                     b[static_cast<std::size_t>((l + 1) * (l + 2) / 2 - 1)] =
                        tmp[static_cast<std::size_t>((l + 1) * (l + 2) / 2 - 1)] * d0 +
                        tmp[static_cast<std::size_t>(l * (l + 1) / 2 - 1)] * d1;
                  }
               }
            }
         }
         else
         {
            b[0] = 1.0;
         }

         for (int l = 0; l <= i; ++l)
         {
            for (int m = 0; m <= j; ++m)
            {
               const int pos = MonomialIndex(l, m);
               triangle_coefficients_[static_cast<std::size_t>(basis_index * triangle_dofs_ + pos)] +=
                  a[static_cast<std::size_t>(l * (l + 1) / 2)] *
                  b[static_cast<std::size_t>((m + 1) * (m + 2) / 2 - 1)];
            }
         }

         if (k != 0)
         {
            for (int n = 0; n <= k - 1; ++n)
            {
               const auto row_begin = triangle_coefficients_.begin() +
                                      static_cast<std::ptrdiff_t>(basis_index * triangle_dofs_);
               tmp.assign(row_begin, row_begin + triangle_dofs_);

               const double d0 = static_cast<double>(n - order_) / static_cast<double>(n - k);
               const double d1 = static_cast<double>(order_) / static_cast<double>(n - k);
               const double d2 = static_cast<double>(order_) / static_cast<double>(n - k);

               for (int total = 0; total <= order_; ++total)
               {
                  for (int m = 0; m <= total; ++m)
                  {
                     const int l = total - m;
                     const int pos = MonomialIndex(l, m);
                     double value = tmp[static_cast<std::size_t>(pos)] * d0;
                     if (l > 0)
                     {
                        value += tmp[static_cast<std::size_t>(MonomialIndex(l - 1, m))] * d1;
                     }
                     if (m > 0)
                     {
                        value += tmp[static_cast<std::size_t>(MonomialIndex(l, m - 1))] * d2;
                     }
                     triangle_coefficients_[static_cast<std::size_t>(basis_index * triangle_dofs_ + pos)] =
                        value;
                  }
               }
            }
         }

         ++basis_index;
      }
   }
}

} // namespace callaway
