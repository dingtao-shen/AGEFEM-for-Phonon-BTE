#include "callaway/integration_cache.hpp"
#include "callaway/mesh_adapter.hpp"
#include "callaway/nodal_basis.hpp"

#include <cassert>
#include <cmath>
#include <filesystem>
#include <iostream>

namespace
{

void CheckClose(double actual, double expected, double tolerance)
{
   assert(std::abs(actual - expected) <= tolerance);
}

void CheckBasis(int order)
{
   const callaway::NodalBasis basis(order);

   for (int node = 0; node < basis.face_dofs(); ++node)
   {
      const double t = basis.face_nodes()[static_cast<std::size_t>(node)];
      for (int i = 0; i < basis.face_dofs(); ++i)
      {
         CheckClose(basis.EvaluateFace(i, t), i == node ? 1.0 : 0.0, 1.0e-12);
      }
   }

   for (int node = 0; node < basis.triangle_dofs(); ++node)
   {
      const auto xy = basis.triangle_nodes()[static_cast<std::size_t>(node)];
      for (int i = 0; i < basis.triangle_dofs(); ++i)
      {
         CheckClose(basis.EvaluateTriangle(i, xy[0], xy[1]), i == node ? 1.0 : 0.0, 1.0e-11);
      }
   }

   const auto values = basis.EvaluateTriangleAll(0.23, 0.31);
   double sum = 0.0;
   for (const double value : values) { sum += value; }
   CheckClose(sum, 1.0, 1.0e-12);

   const auto line_values = basis.EvaluateFaceAll(0.37);
   double line_sum = 0.0;
   for (const double value : line_values) { line_sum += value; }
   CheckClose(line_sum, 1.0, 1.0e-12);
}

} // namespace

int main(int argc, char **argv)
{
   if (argc < 2)
   {
      std::cerr << "Usage: test_basis_integration MESH\n";
      return 2;
   }

   for (int order = 1; order <= 4; ++order)
   {
      CheckBasis(order);
   }

   CheckClose(callaway::ReferenceMonomialIntegral(0, 0), 0.5, 1.0e-15);
   CheckClose(callaway::ReferenceMonomialIntegral(1, 0), 1.0 / 6.0, 1.0e-15);
   CheckClose(callaway::ReferenceMonomialIntegral(0, 1), 1.0 / 6.0, 1.0e-15);
   CheckClose(callaway::ReferenceMonomialIntegral(1, 1), 1.0 / 24.0, 1.0e-15);

   const callaway::MeshAdapter mesh{std::filesystem::path(argv[1])};
   assert(mesh.faces().size() == 320);
   assert(mesh.BoundaryFaceCount() == 40);
   assert(mesh.InteriorFaceCount() == 280);
   for (const auto &face : mesh.faces())
   {
      assert(face.index >= 0);
      assert(face.vertices[0] >= 0);
      assert(face.vertices[1] >= 0);
      assert(face.length > 0.0);
      assert(face.element1 >= 0);
      assert(face.local_face1 >= 0);
      if (face.is_interior())
      {
         assert(face.element2 >= 0);
         assert(face.local_face2 >= 0);
      }
   }
   for (int elem = 0; elem < 200; ++elem)
   {
      const auto &element_faces = mesh.ElementFaces(elem);
      assert(element_faces[0] >= 0);
      assert(element_faces[1] >= 0);
      assert(element_faces[2] >= 0);
      for (int local = 0; local < 3; ++local)
      {
         const int face_id = mesh.ElementFace(elem, local);
         const auto &face = mesh.Face(face_id);
         assert(face.element1 == elem || face.element2 == elem);
      }
   }

   const callaway::NodalBasis basis(3);
   const callaway::IntegrationCache cache(mesh, basis);

   assert(cache.element_count() == 200);
   assert(cache.dofs() == 10);
   CheckClose(cache.TotalArea(), 1.0, 1.0e-12);

   for (int elem = 0; elem < cache.element_count(); ++elem)
   {
      const double area = cache.Geometry(elem).area;
      assert(cache.Geometry(elem).h_min > 0.0);

      double basis_sum = 0.0;
      for (int i = 0; i < cache.dofs(); ++i)
      {
         basis_sum += cache.BasisIntegral(elem, i);
      }
      CheckClose(basis_sum, area, 1.0e-13);

      for (int row = 0; row < cache.dofs(); ++row)
      {
         double mass_row_sum = 0.0;
         double derivative_x_sum = 0.0;
         double derivative_y_sum = 0.0;
         for (int col = 0; col < cache.dofs(); ++col)
         {
            CheckClose(cache.Mass(elem, row, col), cache.Mass(elem, col, row), 1.0e-13);
            mass_row_sum += cache.Mass(elem, row, col);
            derivative_x_sum += cache.GradX(elem, col, row);
            derivative_y_sum += cache.GradY(elem, col, row);
         }
         CheckClose(mass_row_sum, cache.BasisIntegral(elem, row), 1.0e-13);
         CheckClose(derivative_x_sum, 0.0, 1.0e-11);
         CheckClose(derivative_y_sum, 0.0, 1.0e-11);
      }

      for (int local_face = 0; local_face < 3; ++local_face)
      {
         const double length = cache.Geometry(elem).face_lengths[static_cast<std::size_t>(local_face)];
         const auto normal = cache.OutwardNormal(elem, local_face);
         CheckClose(std::hypot(normal[0], normal[1]), 1.0, 1.0e-13);
         double element_face_sum = 0.0;
         double element_face_basis_sum = 0.0;
         double element_face_integral_sum = 0.0;
         for (int row = 0; row < cache.dofs(); ++row)
         {
            element_face_integral_sum += cache.ElementFaceIntegral(elem, local_face, row);
            for (int col = 0; col < cache.dofs(); ++col)
            {
               CheckClose(cache.ElementFaceMass(elem, local_face, row, col),
                          cache.ElementFaceMass(elem, local_face, col, row), 1.0e-12);
               element_face_sum += cache.ElementFaceMass(elem, local_face, row, col);
            }
            for (int face_basis = 0; face_basis < basis.face_dofs(); ++face_basis)
            {
               element_face_basis_sum += cache.ElementFaceBasisMass(elem, local_face, row, face_basis);
            }
         }
         CheckClose(element_face_sum, length, 1.0e-12);
         CheckClose(element_face_basis_sum, length, 1.0e-12);
         CheckClose(element_face_integral_sum, length, 1.0e-12);

         if (mesh.ElementNeighbor(elem, local_face) >= 0)
         {
            double neighbor_face_sum = 0.0;
            for (int row = 0; row < cache.dofs(); ++row)
            {
               for (int col = 0; col < cache.dofs(); ++col)
               {
                  neighbor_face_sum += cache.NeighborFaceMass(elem, local_face, row, col);
               }
            }
            CheckClose(neighbor_face_sum, length, 1.0e-12);
         }
      }
   }

   for (const auto &face : mesh.faces())
   {
      double face_mass_sum = 0.0;
      for (int row = 0; row < basis.face_dofs(); ++row)
      {
         for (int col = 0; col < basis.face_dofs(); ++col)
         {
            CheckClose(cache.FaceMass(face.index, row, col),
                       cache.FaceMass(face.index, col, row), 1.0e-12);
            face_mass_sum += cache.FaceMass(face.index, row, col);
         }
      }
      CheckClose(face_mass_sum, face.length, 1.0e-12);
   }

   return 0;
}
