#include "callaway/integration_cache.hpp"

#include "callaway/age_basis.hpp"
#include "callaway/age_mesh.hpp"
#include "callaway/angular_quadrature.hpp"
#include "callaway/config.hpp"
#include "callaway/geometry/polyline_curve.hpp"

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

// =============================================================================
// AGE extension implementation
// =============================================================================

namespace
{

// Evaluate any element's basis at a physical point: AGE elements use their
// own physical-coord basis, straight elements use the reference-mapped
// NodalBasis after the affine inverse map.
std::vector<double> EvaluateBasisAtPhysical(int element,
                                            double x, double y,
                                            const AgeMesh &age_mesh,
                                            const NodalBasis &basis,
                                            const std::vector<const AgeElementBasis *> &age_basis_by_elem,
                                            const std::vector<ElementGeometry> &geometries)
{
   if (age_mesh.IsAge(element))
   {
      return age_basis_by_elem[static_cast<std::size_t>(element)]->EvaluateAll(x, y);
   }
   const auto ref = MapToReference(geometries[static_cast<std::size_t>(element)], x, y);
   return basis.EvaluateTriangleAll(ref[0], ref[1]);
}

} // namespace

IntegrationCache::IntegrationCache(const AgeMesh &age_mesh,
                                   const NodalBasis &basis,
                                   const std::vector<AgeElementBasis> &age_bases,
                                   const AngularQuadrature &quadrature,
                                   const AgeSettings &age_settings)
   : element_count_(age_mesh.mesh().mesh().GetNE()),
     dofs_(basis.triangle_dofs()),
     face_dofs_(basis.face_dofs()),
     face_count_(static_cast<int>(age_mesh.mesh().faces().size())),
     geometries_(static_cast<std::size_t>(element_count_)),
     basis_integrals_(static_cast<std::size_t>(element_count_ * dofs_), 0.0),
     mass_(static_cast<std::size_t>(element_count_ * dofs_ * dofs_), 0.0),
     grad_x_(static_cast<std::size_t>(element_count_ * dofs_ * dofs_), 0.0),
     grad_y_(static_cast<std::size_t>(element_count_ * dofs_ * dofs_), 0.0),
     element_face_mass_(static_cast<std::size_t>(element_count_ * 3 * dofs_ * dofs_), 0.0),
     element_face_basis_mass_(static_cast<std::size_t>(element_count_ * 3 * dofs_ * face_dofs_), 0.0),
     neighbor_face_mass_(static_cast<std::size_t>(element_count_ * 3 * dofs_ * dofs_), 0.0),
     face_mass_(static_cast<std::size_t>(face_count_ * face_dofs_ * face_dofs_), 0.0),
     angular_quadrature_(&quadrature),
     angle_count_(quadrature.size()),
     curved_face_tensor_mode_(age_settings.curved_face_tensors)
{
   if (static_cast<int>(age_bases.size()) != age_mesh.age_element_count())
   {
      std::ostringstream os;
      os << "IntegrationCache: age_bases size " << age_bases.size()
         << " does not match AgeMesh::age_element_count() " << age_mesh.age_element_count();
      throw std::runtime_error(os.str());
   }
   for (const AgeElementBasis &b : age_bases)
   {
      if (b.dofs() != dofs_)
      {
         throw std::runtime_error("IntegrationCache: AgeElementBasis dof count mismatch with NodalBasis.");
      }
   }

   const MeshAdapter &mesh_adapter = age_mesh.mesh();
   const mfem::Mesh &mesh = mesh_adapter.mesh();

   // Build element-indexed lookups for AGE geometry and basis.
   std::vector<const AgeElementGeometry *> age_geom_by_elem(
      static_cast<std::size_t>(element_count_), nullptr);
   std::vector<const AgeElementBasis *> age_basis_by_elem(
      static_cast<std::size_t>(element_count_), nullptr);
   for (std::size_t i = 0; i < age_mesh.age_elements().size(); ++i)
   {
      const AgeElementGeometry &g = age_mesh.age_elements()[i];
      age_geom_by_elem[static_cast<std::size_t>(g.element)] = &g;
      age_basis_by_elem[static_cast<std::size_t>(g.element)] = &age_bases[i];
   }

   // Build the curved-face lookup table.
   curved_face_index_.assign(static_cast<std::size_t>(element_count_ * 3), -1);
   curved_face_records_.clear();
   for (std::size_t i = 0; i < age_mesh.age_elements().size(); ++i)
   {
      const AgeElementGeometry &g = age_mesh.age_elements()[i];
      curved_face_index_[static_cast<std::size_t>(g.element * 3 + g.curved_local_face)] =
         static_cast<int>(curved_face_records_.size());
      curved_face_records_.emplace_back();
   }
   curved_face_count_ = static_cast<int>(curved_face_records_.size());

   const ReferenceMatrices reference = BuildReferenceMatrices(basis);

   // ---- Element-level loop: geometry + volume tensors ----
   // Per-element work writes to disjoint storage slots, so the loop can run
   // in parallel; declare per-thread scratch variables inside the body.
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
   for (int elem = 0; elem < element_count_; ++elem)
   {
      mfem::Array<int> vertices;
      mesh.GetElementVertices(elem, vertices);
      if (vertices.Size() != 3)
      {
         std::ostringstream os;
         os << "IntegrationCache(AGE): element " << elem << " is not triangular.";
         throw std::runtime_error(os.str());
      }
      auto &geometry = geometries_[static_cast<std::size_t>(elem)];
      for (int i = 0; i < 3; ++i)
      {
         const double *vertex = mesh.GetVertex(vertices[i]);
         geometry.vertices[static_cast<std::size_t>(i)] = {vertex[0], vertex[1]};
      }

      if (!age_mesh.IsAge(elem))
      {
         // Straight element — identical to the straight-only ctor.
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
            os << "Element " << elem << " has non-positive area.";
            throw std::runtime_error(os.str());
         }
         const double dxi_dx  = (y3 - y1) / jacobian;
         const double deta_dx = (y1 - y2) / jacobian;
         const double dxi_dy  = (x1 - x3) / jacobian;
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
               const std::size_t idx = static_cast<std::size_t>(row * dofs_ + col);
               MassRef(elem, row, col) = reference.mass[idx] * physical_scale;
               GradXRef(elem, row, col) =
                  (dxi_dx * reference.dxi[idx] + deta_dx * reference.deta[idx]) * physical_scale;
               GradYRef(elem, row, col) =
                  (dxi_dy * reference.dxi[idx] + deta_dy * reference.deta[idx]) * physical_scale;
            }
         }
         continue;
      }

      // AGE element. Parametric curves (smooth arc, NURBS, analytic) use
      // the Upsilon tensor-product quadrature. Polyline curves (sampling-
      // node reconstructions) sub-triangulate the AGE element on the
      // segment breakpoints contained in [lam_a, lam_b] and apply a
      // Duffy-transformed tensor-product GL quadrature to each sub-triangle.
      const AgeElementGeometry *geom = age_geom_by_elem[static_cast<std::size_t>(elem)];
      const AgeElementBasis *age_basis = age_basis_by_elem[static_cast<std::size_t>(elem)];

      const int curved_lf = geom->curved_local_face;
      const double lam_a = geom->parameter_interval.begin;
      const double lam_b = geom->parameter_interval.end;

      // Face lengths and outward normals for the two straight edges.
      for (int lf = 0; lf < 3; ++lf)
      {
         const auto &va = geometry.vertices[static_cast<std::size_t>(lf)];
         const auto &vb = geometry.vertices[static_cast<std::size_t>((lf + 1) % 3)];
         if (lf == curved_lf)
         {
            geometry.outward_normals[static_cast<std::size_t>(lf)] = {{0.0, 0.0}};
         }
         else
         {
            const double len = std::hypot(vb[0] - va[0], vb[1] - va[1]);
            geometry.face_lengths[static_cast<std::size_t>(lf)] = len;
            geometry.outward_normals[static_cast<std::size_t>(lf)] = {{(vb[1] - va[1]) / len,
                                                                       (va[0] - vb[0]) / len}};
         }
      }

      double arc_len = 0.0;
      double area = 0.0;
      const int n_edge = age_settings.edge_quadrature_points;
      const int n_area = age_settings.area_quadrature_points;

      if (geom->curve->kind() == CurveKind::Parametric)
      {
         // Arc length = integral of |C'(lambda)| dlambda over the edge.
         std::vector<double> lam_pts;
         std::vector<double> lam_wts;
         AngularQuadrature::GaussLegendre(n_edge, lam_a, lam_b, lam_pts, lam_wts);
         for (std::size_t q = 0; q < lam_pts.size(); ++q)
         {
            const auto t = geom->curve->Tangent(lam_pts[q]);
            arc_len += std::hypot(t[0], t[1]) * lam_wts[q];
         }

         // Area + volume tensors via the Upsilon transformation.
         std::vector<double> lam_area_pts;
         std::vector<double> lam_area_wts;
         std::vector<double> theta_pts;
         std::vector<double> theta_wts;
         AngularQuadrature::GaussLegendre(n_area, lam_a, lam_b, lam_area_pts, lam_area_wts);
         AngularQuadrature::GaussLegendre(n_area, 0.0, 1.0, theta_pts, theta_wts);

         for (std::size_t i = 0; i < lam_area_pts.size(); ++i)
         {
            const double lam = lam_area_pts[i];
            const CurvePoint c_lam = geom->curve->Point(lam);
            const CurvePoint t_lam = geom->curve->Tangent(lam);
            for (std::size_t j = 0; j < theta_pts.size(); ++j)
            {
               const double theta = theta_pts[j];
               const double x = (1.0 - theta) * c_lam[0] + theta * geom->interior_vertex[0];
               const double y = (1.0 - theta) * c_lam[1] + theta * geom->interior_vertex[1];
               // |J_Upsilon| = (1 - theta) * | t_x (x0_y - c_y) - t_y (x0_x - c_x) |
               const double cross = t_lam[0] * (geom->interior_vertex[1] - c_lam[1]) -
                                    t_lam[1] * (geom->interior_vertex[0] - c_lam[0]);
               const double jac = (1.0 - theta) * std::abs(cross);
               const double w = lam_area_wts[i] * theta_wts[j] * jac;
               area += w;
               const auto phi = age_basis->EvaluateAll(x, y);
               const auto grad = age_basis->EvaluateGradientAll(x, y);
               for (int r = 0; r < dofs_; ++r)
               {
                  BasisIntegralRef(elem, r) += phi[static_cast<std::size_t>(r)] * w;
                  for (int c = 0; c < dofs_; ++c)
                  {
                     const double pr = phi[static_cast<std::size_t>(r)];
                     const double pc = phi[static_cast<std::size_t>(c)];
                     MassRef(elem, r, c) += pr * pc * w;
                     GradXRef(elem, r, c) += pc * grad[static_cast<std::size_t>(r)][0] * w;
                     GradYRef(elem, r, c) += pc * grad[static_cast<std::size_t>(r)][1] * w;
                  }
               }
            }
         }
      }
      else if (geom->curve->kind() == CurveKind::Polyline)
      {
         const PolylineCurve *poly = dynamic_cast<const PolylineCurve *>(geom->curve);
         if (poly == nullptr)
         {
            throw std::runtime_error(
               "IntegrationCache(AGE): polyline AGE element bound to a curve "
               "of CurveKind::Polyline that is not a PolylineCurve.");
         }
         const std::vector<double> &node_lambdas = poly->node_parameters();
         const std::size_t n_nodes = node_lambdas.size();
         const std::size_t n_segments = static_cast<std::size_t>(poly->segment_count());
         const bool closed = poly->is_closed();
         const double eps = 1.0e-12;

         // Active polyline segments are those fully contained in [lam_a, lam_b].
         // We treat lam_a and lam_b as polyline-node parameters per the paper
         // convention (sampling-node mesh vertices coincide with polyline nodes).
         std::vector<std::array<std::size_t, 2>> active_segments;
         for (std::size_t k = 0; k < n_segments; ++k)
         {
            const double sa = node_lambdas[k];
            const double sb = (closed && k + 1 == n_nodes) ? 1.0 : node_lambdas[k + 1];
            if (sa >= lam_a - eps && sb <= lam_b + eps)
            {
               active_segments.push_back({k, (k + 1) % n_nodes});
            }
         }
         if (active_segments.empty())
         {
            std::ostringstream os;
            os << "IntegrationCache(AGE): polyline AGE element " << elem
               << " parameter interval [" << lam_a << ", " << lam_b
               << "] does not contain any complete polyline segment.";
            throw std::runtime_error(os.str());
         }

         // Arc length = sum of physical lengths of active segments.
         for (const auto &seg : active_segments)
         {
            const auto &a = poly->nodes()[seg[0]];
            const auto &b = poly->nodes()[seg[1]];
            arc_len += std::hypot(b[0] - a[0], b[1] - a[1]);
         }

         // Area + volume tensors via Duffy quadrature on each sub-triangle
         // (interior_vertex, segment_start, segment_end).
         std::vector<double> gl_pts;
         std::vector<double> gl_wts;
         AngularQuadrature::GaussLegendre(n_area, 0.0, 1.0, gl_pts, gl_wts);
         const CurvePoint &x0 = geom->interior_vertex;
         for (const auto &seg : active_segments)
         {
            const auto &a = poly->nodes()[seg[0]];
            const auto &b = poly->nodes()[seg[1]];
            // 2 * sub-triangle area = |(a - x0) x (b - x0)|.
            const double cross_tri = (a[0] - x0[0]) * (b[1] - x0[1]) -
                                     (a[1] - x0[1]) * (b[0] - x0[0]);
            const double two_area = std::abs(cross_tri);
            for (std::size_t i = 0; i < gl_pts.size(); ++i)
            {
               for (std::size_t j = 0; j < gl_pts.size(); ++j)
               {
                  const double s = gl_pts[i];
                  const double t = gl_pts[j];
                  const double xi = (1.0 - t) * s;
                  const double eta = t * s;
                  const double x = x0[0] + (a[0] - x0[0]) * xi + (b[0] - x0[0]) * eta;
                  const double y = x0[1] + (a[1] - x0[1]) * xi + (b[1] - x0[1]) * eta;
                  // Duffy Jacobian = s; physical-triangle Jacobian = two_area.
                  const double w = two_area * s * gl_wts[i] * gl_wts[j];
                  area += w;
                  const auto phi = age_basis->EvaluateAll(x, y);
                  const auto grad = age_basis->EvaluateGradientAll(x, y);
                  for (int r = 0; r < dofs_; ++r)
                  {
                     BasisIntegralRef(elem, r) += phi[static_cast<std::size_t>(r)] * w;
                     for (int c = 0; c < dofs_; ++c)
                     {
                        const double pr = phi[static_cast<std::size_t>(r)];
                        const double pc = phi[static_cast<std::size_t>(c)];
                        MassRef(elem, r, c) += pr * pc * w;
                        GradXRef(elem, r, c) += pc * grad[static_cast<std::size_t>(r)][0] * w;
                        GradYRef(elem, r, c) += pc * grad[static_cast<std::size_t>(r)][1] * w;
                     }
                  }
               }
            }
         }
      }
      else
      {
         throw std::runtime_error("IntegrationCache(AGE): unknown CurveKind.");
      }

      geometry.face_lengths[static_cast<std::size_t>(curved_lf)] = arc_len;
      geometry.area = area;
      geometry.h_min = std::min({2.0 * area / geometry.face_lengths[0],
                                 2.0 * area / geometry.face_lengths[1],
                                 2.0 * area / geometry.face_lengths[2]});
   }

   // ---- Face-level loop (per (elem, local_face)): mass + face_basis mass + neighbor mass ----
   std::vector<double> line_points;
   std::vector<double> line_weights;
   AngularQuadrature::GaussLegendre(15, 0.0, 1.0, line_points, line_weights);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
   for (int elem = 0; elem < element_count_; ++elem)
   {
      for (int local_face = 0; local_face < 3; ++local_face)
      {
         const int cf_idx = CurvedFaceLookup(elem, local_face);
         if (cf_idx >= 0)
         {
            // Curved face: build the geometry-side quadrature record and skip
            // the standard tensors (kinetic sweep uses CurvedFace* accessors
            // for these; GSIS curved-trace coupling lands in a later phase).
            const AgeElementGeometry *geom = age_geom_by_elem[static_cast<std::size_t>(elem)];
            const AgeElementBasis *age_basis = age_basis_by_elem[static_cast<std::size_t>(elem)];
            const int n_edge = age_settings.edge_quadrature_points;
            CurvedFaceQuadrature &rec = curved_face_records_[static_cast<std::size_t>(cf_idx)];
            rec.points.clear();
            rec.weights.clear();
            rec.normals.clear();
            rec.basis.clear();

            // Helper: append one quadrature point to the record.
            auto append_qpt = [&](double lam, double weight_factor)
            {
               const CurvePoint p = geom->curve->Point(lam);
               const CurvePoint t = geom->curve->Tangent(lam);
               const CurvePoint n = geom->curve->Normal(lam);
               rec.points.push_back(p);
               rec.weights.push_back(std::hypot(t[0], t[1]) * weight_factor);
               rec.normals.push_back(n);
               rec.basis.push_back(age_basis->EvaluateAll(p[0], p[1]));
            };

            if (geom->curve->kind() == CurveKind::Parametric)
            {
               // Smooth curve: a single 1D GL on the full [lam_a, lam_b].
               std::vector<double> lam_pts;
               std::vector<double> lam_wts;
               AngularQuadrature::GaussLegendre(n_edge,
                                                geom->parameter_interval.begin,
                                                geom->parameter_interval.end,
                                                lam_pts, lam_wts);
               for (std::size_t q = 0; q < lam_pts.size(); ++q)
               {
                  append_qpt(lam_pts[q], lam_wts[q]);
               }
            }
            else if (geom->curve->kind() == CurveKind::Polyline)
            {
               // Polyline: piecewise constant tangent. Run a fresh 1D GL on
               // each active segment so the kink at every breakpoint is
               // respected, then concatenate.
               const PolylineCurve *poly =
                  dynamic_cast<const PolylineCurve *>(geom->curve);
               if (poly == nullptr)
               {
                  throw std::runtime_error(
                     "IntegrationCache(AGE): polyline curved face bound to a "
                     "curve of CurveKind::Polyline that is not a PolylineCurve.");
               }
               const std::vector<double> &node_lambdas = poly->node_parameters();
               const std::size_t n_nodes = node_lambdas.size();
               const std::size_t n_segments = static_cast<std::size_t>(poly->segment_count());
               const bool closed = poly->is_closed();
               const double lam_a = geom->parameter_interval.begin;
               const double lam_b = geom->parameter_interval.end;
               const double eps = 1.0e-12;
               for (std::size_t k = 0; k < n_segments; ++k)
               {
                  const double sa = node_lambdas[k];
                  const double sb = (closed && k + 1 == n_nodes) ? 1.0 : node_lambdas[k + 1];
                  if (!(sa >= lam_a - eps && sb <= lam_b + eps)) { continue; }
                  std::vector<double> lam_pts;
                  std::vector<double> lam_wts;
                  AngularQuadrature::GaussLegendre(n_edge, sa, sb, lam_pts, lam_wts);
                  for (std::size_t q = 0; q < lam_pts.size(); ++q)
                  {
                     append_qpt(lam_pts[q], lam_wts[q]);
                  }
               }
            }
            else
            {
               throw std::runtime_error(
                  "IntegrationCache(AGE) curved face: unknown CurveKind.");
            }
            continue;
         }

         // Straight face. The integration line is the segment between two
         // mesh vertices; weights and self-basis depend on the element kind.
         const ElementGeometry &g = geometries_[static_cast<std::size_t>(elem)];
         const auto &va = g.vertices[static_cast<std::size_t>(local_face)];
         const auto &vb = g.vertices[static_cast<std::size_t>((local_face + 1) % 3)];
         const double len = g.face_lengths[static_cast<std::size_t>(local_face)];
         const int neighbor = mesh_adapter.ElementNeighbor(elem, local_face);

         // For straight elements at straight faces, mirror the existing
         // ctor exactly (uses MapToReference at the global-face direction
         // for ElementFaceBasisMass). For AGE elements at straight faces,
         // use the AGE basis on the self side; ElementFaceBasisMass and
         // FaceMass for AGE-element faces are not used by the kinetic
         // sweep and are left at zero (GSIS coupling lands in Phase 5).
         const bool elem_is_age = age_mesh.IsAge(elem);

         if (!elem_is_age)
         {
            // Straight element / straight face — identical to the straight-only ctor.
            const FaceData &global_face =
               mesh_adapter.Face(mesh_adapter.ElementFace(elem, local_face));
            const double *global_v0 = mesh.GetVertex(global_face.vertices[0]);
            const double *global_v1 = mesh.GetVertex(global_face.vertices[1]);
            for (std::size_t q = 0; q < line_points.size(); ++q)
            {
               const double s = line_points[q];
               const double weight = line_weights[q] * len;
               double xi = 0.0, eta = 0.0;
               switch (local_face)
               {
                  case 0: xi = s;        eta = 0.0;       break;
                  case 1: xi = 1.0 - s;  eta = s;         break;
                  case 2: xi = 0.0;      eta = 1.0 - s;   break;
                  default: throw std::runtime_error("Invalid local face id.");
               }
               const double t = 2.0 * s - 1.0;
               const std::vector<double> tri_values = basis.EvaluateTriangleAll(xi, eta);
               const std::array<double, 2> physical = MapToPhysical(g, xi, eta);
               const std::array<double, 2> global_physical{{
                  (1.0 - s) * global_v0[0] + s * global_v1[0],
                  (1.0 - s) * global_v0[1] + s * global_v1[1]
               }};
               const auto global_ref = MapToReference(g, global_physical[0], global_physical[1]);
               const std::vector<double> global_tri_values =
                  basis.EvaluateTriangleAll(global_ref[0], global_ref[1]);
               const std::vector<double> face_values = basis.EvaluateFaceAll(t);
               std::vector<double> neighbor_values;
               if (neighbor >= 0)
               {
                  neighbor_values = EvaluateBasisAtPhysical(neighbor,
                                                            physical[0], physical[1],
                                                            age_mesh, basis,
                                                            age_basis_by_elem, geometries_);
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
         else
         {
            // Straight face of an AGE element: self-basis is the AGE basis;
            // neighbor-basis is the neighbor's basis (AGE or straight).
            const AgeElementBasis *age_basis =
               age_basis_by_elem[static_cast<std::size_t>(elem)];
            for (std::size_t q = 0; q < line_points.size(); ++q)
            {
               const double s = line_points[q];
               const double weight = line_weights[q] * len;
               const double x = (1.0 - s) * va[0] + s * vb[0];
               const double y = (1.0 - s) * va[1] + s * vb[1];
               const std::vector<double> phi_self = age_basis->EvaluateAll(x, y);
               std::vector<double> phi_neighbor;
               if (neighbor >= 0)
               {
                  phi_neighbor = EvaluateBasisAtPhysical(neighbor, x, y, age_mesh, basis,
                                                          age_basis_by_elem, geometries_);
               }
               for (int row = 0; row < dofs_; ++row)
               {
                  for (int col = 0; col < dofs_; ++col)
                  {
                     ElementFaceMassRef(elem, local_face, row, col) +=
                        phi_self[static_cast<std::size_t>(row)] *
                        phi_self[static_cast<std::size_t>(col)] * weight;
                  }
                  if (neighbor >= 0)
                  {
                     for (int neighbor_col = 0; neighbor_col < dofs_; ++neighbor_col)
                     {
                        NeighborFaceMassRef(elem, local_face, row, neighbor_col) +=
                           phi_self[static_cast<std::size_t>(row)] *
                           phi_neighbor[static_cast<std::size_t>(neighbor_col)] * weight;
                     }
                  }
               }
            }
         }
      }
   }

   // ---- Face mass (per global face): straight faces only (GSIS curved-trace -> Phase 5) ----
   std::vector<double> face_points;
   std::vector<double> face_weights;
   AngularQuadrature::GaussLegendre(15, -1.0, 1.0, face_points, face_weights);
   for (const FaceData &face : mesh_adapter.faces())
   {
      // Skip if this face is the curved edge of an AGE element.
      bool is_curved = false;
      if (face.element1 >= 0 && CurvedFaceLookup(face.element1, face.local_face1) >= 0)
      {
         is_curved = true;
      }
      else if (face.element2 >= 0 && CurvedFaceLookup(face.element2, face.local_face2) >= 0)
      {
         is_curved = true;
      }
      if (is_curved) { continue; }
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

   // ---- Precomputed per-angle curved-face tensors ----
   if (curved_face_count_ > 0 && curved_face_tensor_mode_ == CurvedFaceTensorMode::Precomputed)
   {
      curved_face_matrix_.assign(
         static_cast<std::size_t>(angle_count_) * static_cast<std::size_t>(curved_face_count_) *
         static_cast<std::size_t>(dofs_) * static_cast<std::size_t>(dofs_), 0.0);
      curved_face_inflow_.assign(
         static_cast<std::size_t>(angle_count_) * static_cast<std::size_t>(curved_face_count_) *
         static_cast<std::size_t>(dofs_), 0.0);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
      for (int a = 0; a < angle_count_; ++a)
      {
         const Direction &dir = quadrature[a];
         for (int cf = 0; cf < curved_face_count_; ++cf)
         {
            const CurvedFaceQuadrature &rec = curved_face_records_[static_cast<std::size_t>(cf)];
            const std::size_t matrix_base =
               (static_cast<std::size_t>(a) * static_cast<std::size_t>(curved_face_count_) +
                static_cast<std::size_t>(cf)) *
               static_cast<std::size_t>(dofs_) * static_cast<std::size_t>(dofs_);
            const std::size_t inflow_base =
               (static_cast<std::size_t>(a) * static_cast<std::size_t>(curved_face_count_) +
                static_cast<std::size_t>(cf)) *
               static_cast<std::size_t>(dofs_);
            for (std::size_t q = 0; q < rec.points.size(); ++q)
            {
               const double speed = dir.cx * rec.normals[q][0] + dir.cy * rec.normals[q][1];
               const double up   = 0.5 * (speed + std::abs(speed));
               const double down = 0.5 * (speed - std::abs(speed));
               const double w    = rec.weights[q];
               for (int r = 0; r < dofs_; ++r)
               {
                  curved_face_inflow_[inflow_base + static_cast<std::size_t>(r)] +=
                     down * rec.basis[q][static_cast<std::size_t>(r)] * w;
                  for (int c = 0; c < dofs_; ++c)
                  {
                     curved_face_matrix_[matrix_base +
                        static_cast<std::size_t>(r * dofs_ + c)] +=
                        up * rec.basis[q][static_cast<std::size_t>(r)] *
                             rec.basis[q][static_cast<std::size_t>(c)] * w;
                  }
               }
            }
         }
      }
   }
}

int IntegrationCache::CurvedFaceLookup(int element, int local_face) const
{
   const std::size_t idx = static_cast<std::size_t>(element * 3 + local_face);
   if (idx >= curved_face_index_.size()) { return -1; }
   return curved_face_index_[idx];
}

bool IntegrationCache::IsCurvedFace(int element, int local_face) const
{
   return CurvedFaceLookup(element, local_face) >= 0;
}

const IntegrationCache::CurvedFaceQuadrature &
IntegrationCache::CurvedFaceData(int element, int local_face) const
{
   const int cf = CurvedFaceLookup(element, local_face);
   if (cf < 0)
   {
      std::ostringstream os;
      os << "IntegrationCache::CurvedFaceData: element " << element
         << " local face " << local_face << " is not a curved face.";
      throw std::runtime_error(os.str());
   }
   return curved_face_records_.at(static_cast<std::size_t>(cf));
}

double IntegrationCache::CurvedFaceMatrix(int angle, int element, int local_face,
                                          int row, int col) const
{
   const int cf = CurvedFaceLookup(element, local_face);
   if (cf < 0)
   {
      throw std::runtime_error("IntegrationCache::CurvedFaceMatrix called on a non-curved face.");
   }
   if (curved_face_tensor_mode_ == CurvedFaceTensorMode::Precomputed)
   {
      const std::size_t idx =
         (static_cast<std::size_t>(angle) * static_cast<std::size_t>(curved_face_count_) +
          static_cast<std::size_t>(cf)) * static_cast<std::size_t>(dofs_ * dofs_) +
         static_cast<std::size_t>(row * dofs_ + col);
      return curved_face_matrix_.at(idx);
   }
   // OnTheFly mode: integrate max(s . n, 0) * phi_row * phi_col on demand.
   const CurvedFaceQuadrature &rec = curved_face_records_.at(static_cast<std::size_t>(cf));
   const Direction &dir = (*angular_quadrature_)[angle];
   double sum = 0.0;
   for (std::size_t q = 0; q < rec.points.size(); ++q)
   {
      const double speed = dir.cx * rec.normals[q][0] + dir.cy * rec.normals[q][1];
      const double up = 0.5 * (speed + std::abs(speed));
      sum += up * rec.basis[q][static_cast<std::size_t>(row)] *
                  rec.basis[q][static_cast<std::size_t>(col)] * rec.weights[q];
   }
   return sum;
}

double IntegrationCache::CurvedFaceInflowWeight(int angle, int element, int local_face,
                                                int row) const
{
   const int cf = CurvedFaceLookup(element, local_face);
   if (cf < 0)
   {
      throw std::runtime_error(
         "IntegrationCache::CurvedFaceInflowWeight called on a non-curved face.");
   }
   if (curved_face_tensor_mode_ == CurvedFaceTensorMode::Precomputed)
   {
      const std::size_t idx =
         (static_cast<std::size_t>(angle) * static_cast<std::size_t>(curved_face_count_) +
          static_cast<std::size_t>(cf)) * static_cast<std::size_t>(dofs_) +
         static_cast<std::size_t>(row);
      return curved_face_inflow_.at(idx);
   }
   const CurvedFaceQuadrature &rec = curved_face_records_.at(static_cast<std::size_t>(cf));
   const Direction &dir = (*angular_quadrature_)[angle];
   double sum = 0.0;
   for (std::size_t q = 0; q < rec.points.size(); ++q)
   {
      const double speed = dir.cx * rec.normals[q][0] + dir.cy * rec.normals[q][1];
      const double down = 0.5 * (speed - std::abs(speed));
      sum += down * rec.basis[q][static_cast<std::size_t>(row)] * rec.weights[q];
   }
   return sum;
}

} // namespace callaway
