#include "callaway/age_preprocessor.hpp"

#include "callaway/age_geometry.hpp"

#include <mfem.hpp>

#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace callaway
{

AgePreprocessor::AgePreprocessor(double endpoint_tolerance)
   : endpoint_tolerance_(endpoint_tolerance)
{
   if (endpoint_tolerance_ <= 0.0)
   {
      throw std::runtime_error("AgePreprocessor: endpoint_tolerance must be positive.");
   }
}

AgeMesh AgePreprocessor::BuildStraight(MeshAdapter mesh, AgePreprocessReport *report) const
{
   if (report != nullptr)
   {
      report->straight_elements = mesh.mesh().GetNE();
      report->age_elements = 0;
      report->curved_faces = 0;
      report->bound_curves = 0;
      report->max_endpoint_projection_error = 0.0;
   }
   return AgeMesh(std::move(mesh));
}

AgeMesh AgePreprocessor::Build(MeshAdapter mesh,
                               const std::filesystem::path &geometry_sidecar,
                               const std::vector<BoundaryCondition> &boundary_conditions,
                               AgePreprocessReport *report) const
{
   // Stage B: bind geometry.
   const GeometrySidecar sidecar = LoadGeometrySidecar(geometry_sidecar);
   const std::filesystem::path sidecar_dir = geometry_sidecar.parent_path();

   std::unordered_map<int, const CurveSpec *> spec_by_id;
   for (const CurveSpec &spec : sidecar.curves)
   {
      if (spec_by_id.count(spec.boundary_id) > 0)
      {
         throw std::runtime_error(
            "AGE preprocessor: sidecar declares two curves for boundary_id " +
            std::to_string(spec.boundary_id) + ".");
      }
      spec_by_id.emplace(spec.boundary_id, &spec);
      if (!mesh.HasBoundaryAttribute(spec.boundary_id))
      {
         throw std::runtime_error(
            "AGE preprocessor: sidecar binds boundary_id " +
            std::to_string(spec.boundary_id) +
            " but the mesh has no matching boundary attribute.");
      }
   }

   if (!boundary_conditions.empty())
   {
      std::unordered_set<int> bc_ids;
      for (const BoundaryCondition &bc : boundary_conditions)
      {
         bc_ids.insert(bc.physical_id);
      }
      for (const CurveSpec &spec : sidecar.curves)
      {
         if (bc_ids.find(spec.boundary_id) == bc_ids.end())
         {
            throw std::runtime_error(
               "AGE preprocessor: sidecar binds boundary_id " +
               std::to_string(spec.boundary_id) +
               " but no boundary_condition entry targets it.");
         }
      }
   }

   std::vector<std::unique_ptr<BoundaryCurve>> curves;
   curves.reserve(sidecar.curves.size());
   std::unordered_map<int, BoundaryCurve *> curve_by_id;
   for (const CurveSpec &spec : sidecar.curves)
   {
      curves.push_back(MakeBoundaryCurve(spec, sidecar_dir));
      curve_by_id.emplace(spec.boundary_id, curves.back().get());
   }

   // Stage C: identify and enrich AGE elements.
   const int ne = mesh.mesh().GetNE();
   std::vector<ElementKind> element_kinds(static_cast<std::size_t>(ne), ElementKind::Straight);
   std::vector<AgeElementGeometry> age_elements;
   std::vector<CurvedFace> curved_faces;
   double max_proj_err = 0.0;

   mfem::Array<int> ev;
   for (const FaceData &face : mesh.faces())
   {
      if (face.boundary_attribute <= 0) { continue; }
      const auto it = curve_by_id.find(face.boundary_attribute);
      if (it == curve_by_id.end()) { continue; }
      const BoundaryCurve *curve = it->second;

      const int elem = (face.element1 >= 0) ? face.element1 : face.element2;
      const int local_face = (face.element1 >= 0) ? face.local_face1 : face.local_face2;
      if (elem < 0 || local_face < 0)
      {
         std::ostringstream os;
         os << "AGE preprocessor: curved boundary face " << face.index
            << " has no owning element / local face id.";
         throw std::runtime_error(os.str());
      }

      if (element_kinds[static_cast<std::size_t>(elem)] == ElementKind::Age)
      {
         std::ostringstream os;
         os << "AGE preprocessor: element " << elem
            << " has more than one curved boundary face; multi-curve AGE elements "
               "are not supported in this milestone.";
         throw std::runtime_error(os.str());
      }

      mesh.mesh().GetElementVertices(elem, ev);
      if (ev.Size() != 3)
      {
         throw std::runtime_error("AGE preprocessor: only triangular elements are supported.");
      }
      const int start_v    = ev[local_face];
      const int end_v      = ev[(local_face + 1) % 3];
      const int interior_v = ev[(local_face + 2) % 3];
      const double *start_xy    = mesh.mesh().GetVertex(start_v);
      const double *end_xy      = mesh.mesh().GetVertex(end_v);
      const double *interior_xy = mesh.mesh().GetVertex(interior_v);

      const double lambda_begin   = curve->ParameterOf(start_xy[0], start_xy[1]);
      const double lambda_end_raw = curve->ParameterOf(end_xy[0],   end_xy[1]);

      // Orientation consistency: the curve's tangent at lambda_begin should
      // align with the mesh's CCW traversal direction (start -> end).
      const CurvePoint t_begin = curve->Tangent(lambda_begin);
      const double dx = end_xy[0] - start_xy[0];
      const double dy = end_xy[1] - start_xy[1];
      if (t_begin[0] * dx + t_begin[1] * dy < 0.0)
      {
         std::ostringstream os;
         os << "AGE preprocessor: bound curve direction does not match the mesh CCW "
               "traversal at boundary " << face.boundary_attribute << " face " << face.index
            << " (element " << elem << ", local face " << local_face
            << "). Flip the sidecar `orientation` for that curve.";
         throw std::runtime_error(os.str());
      }

      // Wrap handling for closed curves.
      const CurveInterval dom = curve->domain();
      const double period = dom.end - dom.begin;
      double lambda_end = lambda_end_raw;
      if (curve->is_closed() && lambda_end < lambda_begin)
      {
         lambda_end += period;
      }
      if (lambda_end <= lambda_begin)
      {
         std::ostringstream os;
         os << "AGE preprocessor: derived parameter interval is non-monotone at boundary "
            << face.boundary_attribute << " face " << face.index << ".";
         throw std::runtime_error(os.str());
      }

      // Project endpoints onto the curve and bound the snap error.
      const CurvePoint proj_begin = curve->Point(lambda_begin);
      const CurvePoint proj_end   = curve->Point(lambda_end_raw);
      const double err_begin = std::hypot(proj_begin[0] - start_xy[0],
                                          proj_begin[1] - start_xy[1]);
      const double err_end   = std::hypot(proj_end[0]   - end_xy[0],
                                          proj_end[1]   - end_xy[1]);
      max_proj_err = std::max({max_proj_err, err_begin, err_end});
      if (err_begin > endpoint_tolerance_ || err_end > endpoint_tolerance_)
      {
         std::ostringstream os;
         os << "AGE preprocessor: boundary face " << face.index << " on element " << elem
            << " (boundary_attribute " << face.boundary_attribute << ") has endpoints "
            << std::max(err_begin, err_end) << " away from the bound curve "
            << "(tolerance " << endpoint_tolerance_ << ").";
         throw std::runtime_error(os.str());
      }

      AgeElementGeometry geom;
      geom.element = elem;
      geom.curved_local_face = local_face;
      geom.curve = curve;
      geom.parameter_interval = {lambda_begin, lambda_end};
      geom.interior_vertex = {interior_xy[0], interior_xy[1]};
      geom.curve_begin = proj_begin;
      geom.curve_end   = proj_end;

      CurvedFace cf;
      cf.face = face.index;
      cf.boundary_attribute = face.boundary_attribute;
      cf.curve = curve;
      cf.parameter_interval = {lambda_begin, lambda_end};

      element_kinds[static_cast<std::size_t>(elem)] = ElementKind::Age;
      age_elements.push_back(geom);
      curved_faces.push_back(cf);
   }

   // Stage D: counts.
   const int n_age = static_cast<int>(age_elements.size());
   const int n_straight = ne - n_age;

   if (report != nullptr)
   {
      report->straight_elements = n_straight;
      report->age_elements = n_age;
      report->curved_faces = static_cast<int>(curved_faces.size());
      report->bound_curves = static_cast<int>(curves.size());
      report->max_endpoint_projection_error = max_proj_err;
   }

   return AgeMesh(std::move(mesh),
                  std::move(curves),
                  std::move(element_kinds),
                  std::move(age_elements),
                  std::move(curved_faces));
}

} // namespace callaway
