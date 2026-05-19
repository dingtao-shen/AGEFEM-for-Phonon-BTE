#pragma once

#include "callaway/geometry/boundary_curve.hpp"
#include "callaway/mesh_adapter.hpp"

#include <memory>
#include <vector>

namespace callaway
{

// Per-element classification produced by the AGE pre-processor.
enum class ElementKind
{
   Straight, // standard straight-sided triangle (classic DG)
   Age       // accurate-geometry-embodied: one curved edge
};

// Geometry of an AGE element. Two straight interior edges meet at the
// interior vertex x0; the third edge is the portion C([lambda1, lambda2])
// of a boundary curve, with endpoints x1 = C(lambda1), x2 = C(lambda2).
// The parameter interval is oriented so that traversing the curved edge
// from x1 to x2 keeps the element interior on the left (consistent with
// the CCW element-boundary convention).
struct AgeElementGeometry
{
   int element = -1;
   int curved_local_face = -1; // 0, 1, or 2
   const BoundaryCurve *curve = nullptr;
   CurveInterval parameter_interval;
   CurvePoint interior_vertex{{0.0, 0.0}}; // x0
   CurvePoint curve_begin{{0.0, 0.0}};     // x1 = C(lambda1)
   CurvePoint curve_end{{0.0, 0.0}};       // x2 = C(lambda2)
};

// A boundary face that follows an exact curve. Carries the curve handle,
// the parameter sub-interval over which this face traces the curve, and
// the boundary attribute (for BC dispatch).
struct CurvedFace
{
   int face = -1;
   int boundary_attribute = 0;
   const BoundaryCurve *curve = nullptr;
   CurveInterval parameter_interval;
};

// The unified mesh + geometry intermediate representation produced by the
// AGE pre-processor and consumed by the solver.
//
// AgeMesh composes the topological MeshAdapter (by value) and adds the AGE
// classification and curved-geometry binding. It also owns the BoundaryCurve
// objects, so the AGE element / curved face references stay valid for the
// lifetime of the mesh.
//
// For an all-straight mesh AgeMesh is a thin pass-through over MeshAdapter:
// has_age_elements() returns false, and Kind(e) returns Straight for every
// element. The solver treats both cases uniformly through this single
// interface; only the integration cache and curved-face BC dispatch need
// to look at Kind / IsCurvedFace.
class AgeMesh
{
public:
   // Straight-only construction: every element is ElementKind::Straight.
   explicit AgeMesh(MeshAdapter mesh);

   // Full construction (used by the pre-processor once geometry is bound).
   // `element_kinds` has one entry per element of `mesh`. `age_elements`
   // and `curved_faces` carry the AGE-specific data; their order is not
   // meaningful (lookups go through element / face index).
   AgeMesh(MeshAdapter mesh,
           std::vector<std::unique_ptr<BoundaryCurve>> curves,
           std::vector<ElementKind> element_kinds,
           std::vector<AgeElementGeometry> age_elements,
           std::vector<CurvedFace> curved_faces);

   AgeMesh(AgeMesh &&) = default;
   AgeMesh &operator=(AgeMesh &&) = default;
   AgeMesh(const AgeMesh &) = delete;
   AgeMesh &operator=(const AgeMesh &) = delete;

   const MeshAdapter &mesh() const { return mesh_; }
   MeshAdapter &mesh() { return mesh_; }

   int element_count() const;
   bool has_age_elements() const { return !age_elements_.empty(); }
   int age_element_count() const { return static_cast<int>(age_elements_.size()); }
   int curved_face_count() const { return static_cast<int>(curved_faces_.size()); }

   ElementKind Kind(int element) const;
   bool IsAge(int element) const { return Kind(element) == ElementKind::Age; }

   // AGE geometry for an element; nullptr for straight elements.
   const AgeElementGeometry *AgeGeometry(int element) const;
   const std::vector<AgeElementGeometry> &age_elements() const { return age_elements_; }

   // Curved boundary faces.
   const std::vector<CurvedFace> &curved_faces() const { return curved_faces_; }
   const CurvedFace *CurvedFaceOf(int face) const;
   bool IsCurvedFace(int face) const { return CurvedFaceOf(face) != nullptr; }

   // Bound curves, owned by this AgeMesh.
   const std::vector<std::unique_ptr<BoundaryCurve>> &curves() const { return curves_; }

private:
   MeshAdapter mesh_;
   std::vector<std::unique_ptr<BoundaryCurve>> curves_;
   std::vector<ElementKind> element_kinds_;
   std::vector<AgeElementGeometry> age_elements_;
   std::vector<int> age_index_by_element_;     // -1 if straight
   std::vector<CurvedFace> curved_faces_;
   std::vector<int> curved_face_index_by_face_; // -1 if straight
};

} // namespace callaway
