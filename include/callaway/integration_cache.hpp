#pragma once

#include "callaway/geometry/boundary_curve.hpp"
#include "callaway/mesh_adapter.hpp"
#include "callaway/nodal_basis.hpp"

#include <array>
#include <vector>

namespace callaway
{

class AgeMesh;
class AgeElementBasis;
class AngularQuadrature;
struct AgeSettings;

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

   // === AGE extension ======================================================
   // Interface frozen in Phase 0; implementations land in Phase 3 alongside
   // the AGE-aware construction path. These declarations are additive: they
   // do not affect the straight-sided code path or any existing call site.

   // Geometry-side quadrature record for one curved boundary edge — physical
   // quadrature points along the curve, weights with |C'(lambda)| folded in,
   // outward unit normals at each point, and element-basis values at each
   // point. Direction-independent; the s . n upwind split is applied per
   // direction either ahead of time (Precomputed mode) or at solve time
   // (OnTheFly mode), behind the uniform CurvedFace* accessors below.
   struct CurvedFaceQuadrature
   {
      std::vector<CurvePoint> points;
      std::vector<double> weights;
      std::vector<CurvePoint> normals;
      std::vector<std::vector<double>> basis; // [q][dof] basis values
   };

   // AGE-aware constructor. For straight-only meshes
   // (age_mesh.has_age_elements() == false) the result is bit-for-bit
   // identical to the straight constructor above. For AGE meshes, the
   // volume tensors of AGE elements and the curved-face tensors are built
   // by curved-geometry quadrature. The curved-face tensor mode
   // (Precomputed | OnTheFly) is selected via age_settings.
   IntegrationCache(const AgeMesh &age_mesh,
                    const NodalBasis &basis,
                    const std::vector<AgeElementBasis> &age_bases,
                    const AngularQuadrature &quadrature,
                    const AgeSettings &age_settings);

   bool IsCurvedFace(int element, int local_face) const;

   // Curved-face indexing. CurvedFaceCount returns the total number of
   // curved faces in the AGE mesh. CurvedFaceIndex returns a stable index
   // in [0, CurvedFaceCount()) for any curved face, or -1 otherwise.
   // Solver-side code (e.g. diffuse-reflection precomputation) uses these
   // to build per-curved-face state.
   int CurvedFaceCount() const { return curved_face_count_; }
   int CurvedFaceIndex(int element, int local_face) const
   {
      return CurvedFaceLookup(element, local_face);
   }

   // Geometry-side curved-face quadrature record. Available for any curved
   // face in either tensor mode; used by curved-face BC application and by
   // OnTheFly direction-dependent accessors.
   const CurvedFaceQuadrature &CurvedFaceData(int element, int local_face) const;

   // Direction-dependent curved-face accessors. Encapsulate the
   // Precomputed vs. OnTheFly choice — kernels never branch on mode.
   //   CurvedFaceMatrix:        integral over the curved edge of
   //                              max(s . n, 0) * phi_row * phi_col,
   //                            i.e. the contribution to the local matrix.
   //   CurvedFaceInflowWeight:  integral over the curved edge of
   //                              min(s . n, 0) * phi_row,
   //                            multiplied by the BC value at solve time
   //                            (thermalizing inflow). Reflective-BC
   //                            curved coupling is added in Phase 5.
   double CurvedFaceMatrix(int angle, int element, int local_face,
                           int row, int col) const;
   double CurvedFaceInflowWeight(int angle, int element, int local_face,
                                 int row) const;
   // === end AGE extension ==================================================

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

   // === AGE extension state (populated only by the AGE-aware ctor) ===
   const AngularQuadrature *angular_quadrature_ = nullptr;
   int angle_count_ = 0;
   int curved_face_count_ = 0;
   CurvedFaceTensorMode curved_face_tensor_mode_ = CurvedFaceTensorMode::Precomputed;
   // Per (element * 3 + local_face) -> index into curved_face_records_ or -1.
   std::vector<int> curved_face_index_;
   std::vector<CurvedFaceQuadrature> curved_face_records_;
   // Precomputed direction-dependent tensors. Indices (angle, curved-face id):
   //   curved_face_matrix_  : [angle][cf][row][col]
   //   curved_face_inflow_  : [angle][cf][row]
   std::vector<double> curved_face_matrix_;
   std::vector<double> curved_face_inflow_;

   int CurvedFaceLookup(int element, int local_face) const;
};

double ReferenceMonomialIntegral(int x_power, int y_power);

} // namespace callaway
