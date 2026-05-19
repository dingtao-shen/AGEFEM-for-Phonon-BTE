#pragma once

#include "callaway/age_geometry.hpp"
#include "callaway/age_mesh.hpp"
#include "callaway/boundary.hpp"
#include "callaway/mesh_adapter.hpp"

#include <filesystem>
#include <vector>

namespace callaway
{

// Diagnostics gathered while building an AgeMesh. Surfaced for logging,
// the design-review summary, and validation-error reporting.
struct AgePreprocessReport
{
   int straight_elements = 0;
   int age_elements = 0;
   int curved_faces = 0;
   int bound_curves = 0;

   // Maximum measured distance, across all curved-edge mesh vertices,
   // between the vertex and its bound curve before snap-projection.
   // Stage D rejects the mesh if this exceeds the configured tolerance.
   double max_endpoint_projection_error = 0.0;
};

// The AGE pre-processing pipeline.
//
// Stages (numbered as in AGEFEM_PROPOSAL.md section 7):
//   A. Ingest the topological mesh (already done; we receive a MeshAdapter).
//   B. Bind geometry: parse the geometry sidecar; instantiate one
//      BoundaryCurve per declared boundary id; attach to boundary faces.
//   C. Identify and enrich AGE elements: mark adjacent elements as AGE,
//      locate the curved local face, derive the oriented [lambda1, lambda2]
//      from face-endpoint vertices via ParameterOf, snap endpoints onto
//      the curve, and fix orientation so the parameter increases with the
//      CCW boundary traversal.
//   D. Validate: exactly one curved face per AGE element, the two interior
//      edges straight, monotone parametrization, positive curved-element
//      area, endpoints within the configured tolerance of the bound curve.
//      Emit AgeMesh.
//
// The pipeline is input-agnostic: whatever the geometry sidecar declared
// (circular arc, NURBS, polyline) lands as a BoundaryCurve and produces
// the same AgeMesh shape.
class AgePreprocessor
{
public:
   // endpoint_tolerance: maximum allowed pre-snap distance between a
   // curved-edge mesh vertex and its bound curve. Stage D rejects the
   // mesh with a clear diagnostic if any vertex exceeds this.
   explicit AgePreprocessor(double endpoint_tolerance = 1.0e-9);

   // Build from an explicit geometry sidecar path. Boundary conditions are
   // taken from the control config so the pipeline can validate that every
   // bound curve targets a configured boundary attribute.
   AgeMesh Build(MeshAdapter mesh,
                 const std::filesystem::path &geometry_sidecar,
                 const std::vector<BoundaryCondition> &boundary_conditions,
                 AgePreprocessReport *report = nullptr) const;

   // Build with no geometry sidecar: every element is straight. Provided
   // so that the pipeline is the single mesh-side entry point even for
   // pure straight-sided runs.
   AgeMesh BuildStraight(MeshAdapter mesh,
                         AgePreprocessReport *report = nullptr) const;

   double endpoint_tolerance() const { return endpoint_tolerance_; }

private:
   double endpoint_tolerance_ = 1.0e-9;
};

} // namespace callaway
