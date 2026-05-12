#pragma once

#include "callaway/mesh_adapter.hpp"
#include "callaway/nodal_basis.hpp"

#include <array>
#include <vector>

namespace callaway
{

struct ElementGeometry
{
   std::array<std::array<double, 2>, 3> vertices{};
   std::array<double, 3> face_lengths{{0.0, 0.0, 0.0}};
   std::array<std::array<double, 2>, 3> outward_normals{};
   double area = 0.0;
   double h_min = 0.0;
};

class IntegrationCache
{
public:
   IntegrationCache(const MeshAdapter &mesh, const NodalBasis &basis);

   int element_count() const { return element_count_; }
   int dofs() const { return dofs_; }
   int face_dofs() const { return face_dofs_; }

   const ElementGeometry &Geometry(int element) const { return geometries_.at(element); }
   double TotalArea() const;

   double BasisIntegral(int element, int basis) const;
   double Mass(int element, int row, int col) const;

   // Matches the Fortran tensor orientation: integral(phi_col * d(phi_row)/dx).
   double GradX(int element, int row, int col) const;

   // Matches the Fortran tensor orientation: integral(phi_col * d(phi_row)/dy).
   double GradY(int element, int row, int col) const;

   double ElementFaceMass(int element, int local_face, int row, int col) const;
   double ElementFaceBasisMass(int element, int local_face, int triangle_basis, int face_basis) const;
   double NeighborFaceMass(int element, int local_face, int row, int neighbor_col) const;
   double ElementFaceIntegral(int element, int local_face, int basis) const;
   double FaceMass(int face, int row, int col) const;
   std::array<double, 2> OutwardNormal(int element, int local_face) const;

private:
   int element_count_ = 0;
   int dofs_ = 0;
   int face_dofs_ = 0;
   int face_count_ = 0;
   std::vector<ElementGeometry> geometries_;
   std::vector<double> basis_integrals_;
   std::vector<double> mass_;
   std::vector<double> grad_x_;
   std::vector<double> grad_y_;
   std::vector<double> element_face_mass_;
   std::vector<double> element_face_basis_mass_;
   std::vector<double> neighbor_face_mass_;
   std::vector<double> face_mass_;

   double &BasisIntegralRef(int element, int basis);
   double &MassRef(int element, int row, int col);
   double &GradXRef(int element, int row, int col);
   double &GradYRef(int element, int row, int col);
   double &ElementFaceMassRef(int element, int local_face, int row, int col);
   double &ElementFaceBasisMassRef(int element, int local_face, int triangle_basis, int face_basis);
   double &NeighborFaceMassRef(int element, int local_face, int row, int neighbor_col);
   double &FaceMassRef(int face, int row, int col);
};

double ReferenceMonomialIntegral(int x_power, int y_power);

} // namespace callaway
