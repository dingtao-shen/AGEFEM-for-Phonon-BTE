#pragma once

#include <array>
#include <vector>

namespace callaway
{

class NodalBasis
{
public:
   explicit NodalBasis(int order);

   int order() const { return order_; }
   int triangle_dofs() const { return triangle_dofs_; }
   int face_dofs() const { return face_dofs_; }

   const std::vector<std::array<double, 2>> &triangle_nodes() const { return triangle_nodes_; }
   const std::vector<double> &face_nodes() const { return face_nodes_; }

   double TriangleCoefficient(int basis, int monomial) const;
   double FaceCoefficient(int basis, int power) const;

   double EvaluateTriangle(int basis, double xi, double eta) const;
   double EvaluateFace(int basis, double t) const;

   std::vector<double> EvaluateTriangleAll(double xi, double eta) const;
   std::vector<double> EvaluateFaceAll(double t) const;

   static int TriangleDofs(int order);
   static int FaceDofs(int order);
   static int MonomialIndex(int x_power, int y_power);

private:
   int order_ = 0;
   int triangle_dofs_ = 0;
   int face_dofs_ = 0;
   std::vector<std::array<double, 2>> triangle_nodes_;
   std::vector<double> face_nodes_;
   std::vector<double> triangle_coefficients_;
   std::vector<double> face_coefficients_;

   void BuildFaceBasis();
   void BuildTriangleBasis();
};

} // namespace callaway
