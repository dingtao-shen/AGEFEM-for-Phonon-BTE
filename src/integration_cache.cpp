#include "callaway/integration_cache.hpp"

#include "callaway/angular_quadrature.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace callaway
{
namespace
{

double FactorialRatio(int numerator_start, int numerator_end,
                      int denominator_start, int denominator_end)
{
   const int numerator_count = numerator_end - numerator_start + 1;
   const int denominator_count = denominator_end - denominator_start + 1;
   double value = 1.0;

   const int paired_count = std::min(numerator_count, denominator_count);
   for (int i = 0; i < paired_count; ++i)
   {
      value *= static_cast<double>(numerator_start + i) /
               static_cast<double>(denominator_start + i);
   }
   for (int i = paired_count; i < numerator_count; ++i)
   {
      value *= static_cast<double>(numerator_start + i);
   }
   for (int i = paired_count; i < denominator_count; ++i)
   {
      value /= static_cast<double>(denominator_start + i);
   }
   return value;
}

struct ReferenceMatrices
{
   std::vector<double> basis_integral;
   std::vector<double> mass;
   std::vector<double> dxi;
   std::vector<double> deta;
};

std::array<double, 2> MapToPhysical(const ElementGeometry &geometry, double xi, double eta)
{
   const double x1 = geometry.vertices[0][0];
   const double y1 = geometry.vertices[0][1];
   const double x2 = geometry.vertices[1][0];
   const double y2 = geometry.vertices[1][1];
   const double x3 = geometry.vertices[2][0];
   const double y3 = geometry.vertices[2][1];
   return {{
      x1 + (x2 - x1) * xi + (x3 - x1) * eta,
      y1 + (y2 - y1) * xi + (y3 - y1) * eta
   }};
}

std::array<double, 2> MapToReference(const ElementGeometry &geometry, double x, double y)
{
   const double x1 = geometry.vertices[0][0];
   const double y1 = geometry.vertices[0][1];
   const double x2 = geometry.vertices[1][0];
   const double y2 = geometry.vertices[1][1];
   const double x3 = geometry.vertices[2][0];
   const double y3 = geometry.vertices[2][1];
   const double jacobian = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1);
   return {{
      ((y3 - y1) * (x - x1) + (x1 - x3) * (y - y1)) / jacobian,
      ((y1 - y2) * (x - x1) + (x2 - x1) * (y - y1)) / jacobian
   }};
}

ReferenceMatrices BuildReferenceMatrices(const NodalBasis &basis)
{
   const int order = basis.order();
   const int dofs = basis.triangle_dofs();
   ReferenceMatrices result;
   result.basis_integral.assign(static_cast<std::size_t>(dofs), 0.0);
   result.mass.assign(static_cast<std::size_t>(dofs * dofs), 0.0);
   result.dxi.assign(static_cast<std::size_t>(dofs * dofs), 0.0);
   result.deta.assign(static_cast<std::size_t>(dofs * dofs), 0.0);

   for (int m = 0; m < dofs; ++m)
   {
      int monomial = 0;
      for (int total = 0; total <= order; ++total)
      {
         for (int y_power = 0; y_power <= total; ++y_power)
         {
            const int x_power = total - y_power;
            result.basis_integral[static_cast<std::size_t>(m)] +=
               basis.TriangleCoefficient(m, monomial) *
               ReferenceMonomialIntegral(x_power, y_power);
            ++monomial;
         }
      }
   }

   for (int row = 0; row < dofs; ++row)
   {
      for (int col = 0; col < dofs; ++col)
      {
         double mass_value = 0.0;
         double dxi_value = 0.0;
         double deta_value = 0.0;

         int row_monomial = 0;
         for (int row_total = 0; row_total <= order; ++row_total)
         {
            for (int row_y = 0; row_y <= row_total; ++row_y)
            {
               const int row_x = row_total - row_y;
               const double row_coeff = basis.TriangleCoefficient(row, row_monomial);

               int col_monomial = 0;
               for (int col_total = 0; col_total <= order; ++col_total)
               {
                  for (int col_y = 0; col_y <= col_total; ++col_y)
                  {
                     const int col_x = col_total - col_y;
                     const double col_coeff = basis.TriangleCoefficient(col, col_monomial);
                     const double coeff_product = row_coeff * col_coeff;

                     mass_value += coeff_product *
                                   ReferenceMonomialIntegral(row_x + col_x, row_y + col_y);

                     if (row_x > 0)
                     {
                        dxi_value += static_cast<double>(row_x) * coeff_product *
                                     ReferenceMonomialIntegral(row_x + col_x - 1, row_y + col_y);
                     }
                     if (row_y > 0)
                     {
                        deta_value += static_cast<double>(row_y) * coeff_product *
                                      ReferenceMonomialIntegral(row_x + col_x, row_y + col_y - 1);
                     }
                     ++col_monomial;
                  }
               }
               ++row_monomial;
            }
         }

         result.mass[static_cast<std::size_t>(row * dofs + col)] = mass_value;
         result.dxi[static_cast<std::size_t>(row * dofs + col)] = dxi_value;
         result.deta[static_cast<std::size_t>(row * dofs + col)] = deta_value;
      }
   }

   return result;
}

} // namespace

double ReferenceMonomialIntegral(int x_power, int y_power)
{
   if (x_power < 0 || y_power < 0)
   {
      throw std::runtime_error("ReferenceMonomialIntegral requires nonnegative powers.");
   }
   return FactorialRatio(1, y_power, x_power + 1, x_power + y_power + 2);
}

IntegrationCache::IntegrationCache(const MeshAdapter &mesh_adapter, const NodalBasis &basis)
   : element_count_(mesh_adapter.mesh().GetNE()),
     dofs_(basis.triangle_dofs()),
     face_dofs_(basis.face_dofs()),
     face_count_(static_cast<int>(mesh_adapter.faces().size())),
     geometries_(static_cast<std::size_t>(element_count_)),
     basis_integrals_(static_cast<std::size_t>(element_count_ * dofs_), 0.0),
     mass_(static_cast<std::size_t>(element_count_ * dofs_ * dofs_), 0.0),
     grad_x_(static_cast<std::size_t>(element_count_ * dofs_ * dofs_), 0.0),
     grad_y_(static_cast<std::size_t>(element_count_ * dofs_ * dofs_), 0.0),
     element_face_mass_(static_cast<std::size_t>(element_count_ * 3 * dofs_ * dofs_), 0.0),
     element_face_basis_mass_(static_cast<std::size_t>(element_count_ * 3 * dofs_ * face_dofs_), 0.0),
     neighbor_face_mass_(static_cast<std::size_t>(element_count_ * 3 * dofs_ * dofs_), 0.0),
     face_mass_(static_cast<std::size_t>(face_count_ * face_dofs_ * face_dofs_), 0.0)
{
   const mfem::Mesh &mesh = mesh_adapter.mesh();
   const ReferenceMatrices reference = BuildReferenceMatrices(basis);
   mfem::Array<int> vertices;

   for (int elem = 0; elem < element_count_; ++elem)
   {
      mesh.GetElementVertices(elem, vertices);
      if (vertices.Size() != 3)
      {
         std::ostringstream os;
         os << "Element " << elem << " is not triangular.";
         throw std::runtime_error(os.str());
      }

      auto &geometry = geometries_[static_cast<std::size_t>(elem)];
      for (int i = 0; i < 3; ++i)
      {
         const double *vertex = mesh.GetVertex(vertices[i]);
         geometry.vertices[static_cast<std::size_t>(i)] = {vertex[0], vertex[1]};
      }

      const double x1 = geometry.vertices[0][0];
      const double y1 = geometry.vertices[0][1];
      const double x2 = geometry.vertices[1][0];
      const double y2 = geometry.vertices[1][1];
      const double x3 = geometry.vertices[2][0];
      const double y3 = geometry.vertices[2][1];

      const double jacobian = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1);
      geometry.area = 0.5 * jacobian;
      if (geometry.area <= 0.0)
      {
         std::ostringstream os;
         os << "Element " << elem << " has non-positive orientation/area.";
         throw std::runtime_error(os.str());
      }

      const double dxi_dx = (y3 - y1) / jacobian;
      const double deta_dx = (y1 - y2) / jacobian;
      const double dxi_dy = (x1 - x3) / jacobian;
      const double deta_dy = (x2 - x1) / jacobian;

      const double physical_scale = jacobian;
      geometry.face_lengths[0] = std::hypot(x2 - x1, y2 - y1);
      geometry.face_lengths[1] = std::hypot(x3 - x2, y3 - y2);
      geometry.face_lengths[2] = std::hypot(x1 - x3, y1 - y3);
      geometry.h_min = std::min({2.0 * geometry.area / geometry.face_lengths[0],
                                 2.0 * geometry.area / geometry.face_lengths[1],
                                 2.0 * geometry.area / geometry.face_lengths[2]});
      geometry.outward_normals[0] = {{(y2 - y1) / geometry.face_lengths[0],
                                      (x1 - x2) / geometry.face_lengths[0]}};
      geometry.outward_normals[1] = {{(y3 - y2) / geometry.face_lengths[1],
                                      (x2 - x3) / geometry.face_lengths[1]}};
      geometry.outward_normals[2] = {{(y1 - y3) / geometry.face_lengths[2],
                                      (x3 - x1) / geometry.face_lengths[2]}};

      for (int row = 0; row < dofs_; ++row)
      {
         BasisIntegralRef(elem, row) =
            reference.basis_integral[static_cast<std::size_t>(row)] * physical_scale;

         for (int col = 0; col < dofs_; ++col)
         {
            const std::size_t index = static_cast<std::size_t>(row * dofs_ + col);
            MassRef(elem, row, col) = reference.mass[index] * physical_scale;
            GradXRef(elem, row, col) =
               (dxi_dx * reference.dxi[index] + deta_dx * reference.deta[index]) * physical_scale;
            GradYRef(elem, row, col) =
               (dxi_dy * reference.dxi[index] + deta_dy * reference.deta[index]) * physical_scale;
         }
      }
   }

   std::vector<double> line_points;
   std::vector<double> line_weights;
   AngularQuadrature::GaussLegendre(15, 0.0, 1.0, line_points, line_weights);

   for (int elem = 0; elem < element_count_; ++elem)
   {
      for (int local_face = 0; local_face < 3; ++local_face)
      {
         const FaceData &global_face = mesh_adapter.Face(mesh_adapter.ElementFace(elem, local_face));
         const double *global_v0 = mesh.GetVertex(global_face.vertices[0]);
         const double *global_v1 = mesh.GetVertex(global_face.vertices[1]);
         const double length = geometries_[static_cast<std::size_t>(elem)]
                                  .face_lengths[static_cast<std::size_t>(local_face)];
         for (std::size_t q = 0; q < line_points.size(); ++q)
         {
            const double s = line_points[q];
            const double weight = line_weights[q] * length;
            double xi = 0.0;
            double eta = 0.0;
            switch (local_face)
            {
               case 0:
                  xi = s;
                  eta = 0.0;
                  break;
               case 1:
                  xi = 1.0 - s;
                  eta = s;
                  break;
               case 2:
                  xi = 0.0;
                  eta = 1.0 - s;
                  break;
               default:
                  throw std::runtime_error("Invalid local face id.");
            }

            const double t = 2.0 * s - 1.0;
            const std::vector<double> tri_values = basis.EvaluateTriangleAll(xi, eta);
            const std::array<double, 2> physical =
               MapToPhysical(geometries_[static_cast<std::size_t>(elem)], xi, eta);

            const std::array<double, 2> global_physical{{
               (1.0 - s) * global_v0[0] + s * global_v1[0],
               (1.0 - s) * global_v0[1] + s * global_v1[1]
            }};
            const std::array<double, 2> global_ref =
               MapToReference(geometries_[static_cast<std::size_t>(elem)],
                              global_physical[0],
                              global_physical[1]);
            const std::vector<double> global_tri_values =
               basis.EvaluateTriangleAll(global_ref[0], global_ref[1]);
            const std::vector<double> face_values = basis.EvaluateFaceAll(t);

            std::vector<double> neighbor_values;
            const int neighbor = mesh_adapter.ElementNeighbor(elem, local_face);
            if (neighbor >= 0)
            {
               const std::array<double, 2> neighbor_ref =
                  MapToReference(geometries_[static_cast<std::size_t>(neighbor)],
                                 physical[0], physical[1]);
               neighbor_values = basis.EvaluateTriangleAll(neighbor_ref[0], neighbor_ref[1]);
            }

            for (int row = 0; row < dofs_; ++row)
            {
               for (int col = 0; col < dofs_; ++col)
               {
                  ElementFaceMassRef(elem, local_face, row, col) +=
                     tri_values[static_cast<std::size_t>(row)] *
                     tri_values[static_cast<std::size_t>(col)] * weight;
               }
               for (int face_basis = 0; face_basis < face_dofs_; ++face_basis)
               {
                  ElementFaceBasisMassRef(elem, local_face, row, face_basis) +=
                     global_tri_values[static_cast<std::size_t>(row)] *
                     face_values[static_cast<std::size_t>(face_basis)] * weight;
               }
               if (neighbor >= 0)
               {
                  for (int neighbor_col = 0; neighbor_col < dofs_; ++neighbor_col)
                  {
                     NeighborFaceMassRef(elem, local_face, row, neighbor_col) +=
                        tri_values[static_cast<std::size_t>(row)] *
                        neighbor_values[static_cast<std::size_t>(neighbor_col)] * weight;
                  }
               }
            }
         }
      }
   }

   std::vector<double> face_points;
   std::vector<double> face_weights;
   AngularQuadrature::GaussLegendre(15, -1.0, 1.0, face_points, face_weights);
   for (const FaceData &face : mesh_adapter.faces())
   {
      for (std::size_t q = 0; q < face_points.size(); ++q)
      {
         const double weight = 0.5 * face.length * face_weights[q];
         const std::vector<double> face_values = basis.EvaluateFaceAll(face_points[q]);
         for (int row = 0; row < face_dofs_; ++row)
         {
            for (int col = 0; col < face_dofs_; ++col)
            {
               FaceMassRef(face.index, row, col) +=
                  face_values[static_cast<std::size_t>(row)] *
                  face_values[static_cast<std::size_t>(col)] * weight;
            }
         }
      }
   }
}

double IntegrationCache::TotalArea() const
{
   return std::accumulate(geometries_.begin(), geometries_.end(), 0.0,
                          [](double sum, const ElementGeometry &geometry)
                          {
                             return sum + geometry.area;
                          });
}

double IntegrationCache::BasisIntegral(int element, int basis) const
{
   return basis_integrals_.at(static_cast<std::size_t>(element * dofs_ + basis));
}

double IntegrationCache::Mass(int element, int row, int col) const
{
   return mass_.at(static_cast<std::size_t>((element * dofs_ + row) * dofs_ + col));
}

double IntegrationCache::GradX(int element, int row, int col) const
{
   return grad_x_.at(static_cast<std::size_t>((element * dofs_ + row) * dofs_ + col));
}

double IntegrationCache::GradY(int element, int row, int col) const
{
   return grad_y_.at(static_cast<std::size_t>((element * dofs_ + row) * dofs_ + col));
}

double IntegrationCache::ElementFaceMass(int element, int local_face, int row, int col) const
{
   return element_face_mass_.at(static_cast<std::size_t>(((element * 3 + local_face) * dofs_ + row) * dofs_ + col));
}

double IntegrationCache::ElementFaceBasisMass(int element, int local_face, int triangle_basis, int face_basis) const
{
   return element_face_basis_mass_.at(static_cast<std::size_t>(((element * 3 + local_face) * dofs_ + triangle_basis) * face_dofs_ + face_basis));
}

double IntegrationCache::NeighborFaceMass(int element, int local_face, int row, int neighbor_col) const
{
   return neighbor_face_mass_.at(static_cast<std::size_t>(((element * 3 + local_face) * dofs_ + row) * dofs_ + neighbor_col));
}

double IntegrationCache::ElementFaceIntegral(int element, int local_face, int basis) const
{
   double value = 0.0;
   for (int col = 0; col < dofs_; ++col)
   {
      value += ElementFaceMass(element, local_face, basis, col);
   }
   return value;
}

double IntegrationCache::FaceMass(int face, int row, int col) const
{
   return face_mass_.at(static_cast<std::size_t>((face * face_dofs_ + row) * face_dofs_ + col));
}

std::array<double, 2> IntegrationCache::OutwardNormal(int element, int local_face) const
{
   return geometries_.at(element).outward_normals.at(local_face);
}

double &IntegrationCache::BasisIntegralRef(int element, int basis)
{
   return basis_integrals_[static_cast<std::size_t>(element * dofs_ + basis)];
}

double &IntegrationCache::MassRef(int element, int row, int col)
{
   return mass_[static_cast<std::size_t>((element * dofs_ + row) * dofs_ + col)];
}

double &IntegrationCache::GradXRef(int element, int row, int col)
{
   return grad_x_[static_cast<std::size_t>((element * dofs_ + row) * dofs_ + col)];
}

double &IntegrationCache::GradYRef(int element, int row, int col)
{
   return grad_y_[static_cast<std::size_t>((element * dofs_ + row) * dofs_ + col)];
}

double &IntegrationCache::ElementFaceMassRef(int element, int local_face, int row, int col)
{
   return element_face_mass_[static_cast<std::size_t>(((element * 3 + local_face) * dofs_ + row) * dofs_ + col)];
}

double &IntegrationCache::ElementFaceBasisMassRef(int element, int local_face, int triangle_basis, int face_basis)
{
   return element_face_basis_mass_[static_cast<std::size_t>(((element * 3 + local_face) * dofs_ + triangle_basis) * face_dofs_ + face_basis)];
}

double &IntegrationCache::NeighborFaceMassRef(int element, int local_face, int row, int neighbor_col)
{
   return neighbor_face_mass_[static_cast<std::size_t>(((element * 3 + local_face) * dofs_ + row) * dofs_ + neighbor_col)];
}

double &IntegrationCache::FaceMassRef(int face, int row, int col)
{
   return face_mass_[static_cast<std::size_t>((face * face_dofs_ + row) * face_dofs_ + col)];
}

} // namespace callaway
