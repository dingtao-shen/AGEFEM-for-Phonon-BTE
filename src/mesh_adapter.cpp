#include "callaway/mesh_adapter.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace callaway
{
namespace
{

std::vector<int> ToSortedVector(const mfem::Array<int> &values)
{
   std::vector<int> result;
   result.reserve(static_cast<std::size_t>(values.Size()));
   for (int i = 0; i < values.Size(); ++i)
   {
      result.push_back(values[i]);
   }
   std::sort(result.begin(), result.end());
   result.erase(std::unique(result.begin(), result.end()), result.end());
   return result;
}

bool SameUndirectedEdge(const std::array<int, 2> &a, const std::array<int, 2> &b)
{
   return (a[0] == b[0] && a[1] == b[1]) || (a[0] == b[1] && a[1] == b[0]);
}

} // namespace

MeshAdapter::MeshAdapter(std::filesystem::path mesh_path)
   : mesh_path_(std::filesystem::weakly_canonical(mesh_path))
{
   mesh_ = std::make_unique<mfem::Mesh>(mesh_path_.string().c_str(), 1, 1);
   if (mesh_->Dimension() != 2)
   {
      throw std::runtime_error("Only 2D meshes are supported by the current solver.");
   }
   mesh_->EnsureNodes();
   BuildFaceData();
}

MeshSummary MeshAdapter::Summary() const
{
   MeshSummary summary;
   summary.dimension = mesh_->Dimension();
   summary.vertices = mesh_->GetNV();
   summary.elements = mesh_->GetNE();
   summary.boundary_elements = mesh_->GetNBE();
   summary.faces = mesh_->GetNumFaces();
   summary.element_attributes = ToSortedVector(mesh_->attributes);
   summary.boundary_attributes = ToSortedVector(mesh_->bdr_attributes);
   return summary;
}

bool MeshAdapter::HasBoundaryAttribute(int attribute) const
{
   for (int i = 0; i < mesh_->bdr_attributes.Size(); ++i)
   {
      if (mesh_->bdr_attributes[i] == attribute) { return true; }
   }
   return false;
}

void MeshAdapter::ValidateBoundaryAttributes(const std::vector<BoundaryCondition> &bcs) const
{
   for (const auto &bc : bcs)
   {
      if (!HasBoundaryAttribute(bc.physical_id))
      {
         std::ostringstream os;
         os << "Boundary condition '" << bc.name << "' references physical_id "
            << bc.physical_id << ", but the mesh has no matching boundary attribute.";
         throw std::runtime_error(os.str());
      }
   }
}

int MeshAdapter::BoundaryFaceCount() const
{
   return static_cast<int>(std::count_if(faces_.begin(), faces_.end(),
                                         [](const FaceData &face) { return face.is_boundary(); }));
}

int MeshAdapter::InteriorFaceCount() const
{
   return static_cast<int>(std::count_if(faces_.begin(), faces_.end(),
                                         [](const FaceData &face) { return face.is_interior(); }));
}

int MeshAdapter::ElementNeighbor(int element, int local_face) const
{
   const int face_id = ElementFace(element, local_face);
   const FaceData &face = Face(face_id);
   if (face.element1 == element) { return face.element2; }
   if (face.element2 == element) { return face.element1; }
   return -1;
}

void MeshAdapter::BuildFaceData()
{
   const int num_faces = mesh_->GetNumFaces();
   faces_.assign(static_cast<std::size_t>(num_faces), FaceData{});
   element_faces_.assign(static_cast<std::size_t>(mesh_->GetNE()), std::array<int, 3>{{-1, -1, -1}});

   std::map<int, int> boundary_attribute_by_face;
   mfem::Array<int> bdr_edges;
   mfem::Array<int> bdr_orientations;
   for (int b = 0; b < mesh_->GetNBE(); ++b)
   {
      mesh_->GetBdrElementEdges(b, bdr_edges, bdr_orientations);
      if (bdr_edges.Size() != 1)
      {
         throw std::runtime_error("Expected each 2D boundary element to map to one edge.");
      }
      boundary_attribute_by_face[bdr_edges[0]] = mesh_->GetBdrAttribute(b);
   }

   mfem::Array<int> vertices;
   for (int f = 0; f < num_faces; ++f)
   {
      FaceData face;
      face.index = f;
      mesh_->GetFaceVertices(f, vertices);
      if (vertices.Size() != 2)
      {
         std::ostringstream os;
         os << "Face " << f << " does not have two vertices.";
         throw std::runtime_error(os.str());
      }
      face.vertices = {vertices[0], vertices[1]};

      const double *v0 = mesh_->GetVertex(vertices[0]);
      const double *v1 = mesh_->GetVertex(vertices[1]);
      const double dx = v1[0] - v0[0];
      const double dy = v1[1] - v0[1];
      face.length = std::sqrt(dx * dx + dy * dy);

      const auto attr_it = boundary_attribute_by_face.find(f);
      if (attr_it != boundary_attribute_by_face.end())
      {
         face.boundary_attribute = attr_it->second;
      }

      faces_[static_cast<std::size_t>(f)] = face;
   }

   mfem::Array<int> element_vertices;
   mfem::Array<int> element_edges;
   mfem::Array<int> element_edge_orientations;
   for (int elem = 0; elem < mesh_->GetNE(); ++elem)
   {
      mesh_->GetElementVertices(elem, element_vertices);
      mesh_->GetElementEdges(elem, element_edges, element_edge_orientations);
      if (element_vertices.Size() != 3 || element_edges.Size() != 3)
      {
         std::ostringstream os;
         os << "Element " << elem << " is not a triangular element with three edges.";
         throw std::runtime_error(os.str());
      }

      const std::array<std::array<int, 2>, 3> local_edges{{
         {{element_vertices[0], element_vertices[1]}},
         {{element_vertices[1], element_vertices[2]}},
         {{element_vertices[2], element_vertices[0]}}
      }};

      for (int local = 0; local < 3; ++local)
      {
         int matched_face = -1;
         for (int candidate = 0; candidate < element_edges.Size(); ++candidate)
         {
            const int face_id = element_edges[candidate];
            if (SameUndirectedEdge(local_edges[static_cast<std::size_t>(local)],
                                   faces_[static_cast<std::size_t>(face_id)].vertices))
            {
               matched_face = face_id;
               break;
            }
         }

         if (matched_face < 0)
         {
            std::ostringstream os;
            os << "Failed to match element " << elem << " local face " << local
               << " to a global face.";
            throw std::runtime_error(os.str());
         }

         element_faces_[static_cast<std::size_t>(elem)][static_cast<std::size_t>(local)] = matched_face;
         FaceData &face = faces_[static_cast<std::size_t>(matched_face)];
         if (face.element1 < 0)
         {
            face.element1 = elem;
            face.local_face1 = local;
         }
         else if (face.element2 < 0)
         {
            face.element2 = elem;
            face.local_face2 = local;
         }
         else
         {
            std::ostringstream os;
            os << "Face " << matched_face << " is connected to more than two elements.";
            throw std::runtime_error(os.str());
         }
      }
   }
}

} // namespace callaway
