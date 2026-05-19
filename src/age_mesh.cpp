#include "callaway/age_mesh.hpp"

#include <sstream>
#include <stdexcept>

namespace callaway
{

AgeMesh::AgeMesh(MeshAdapter mesh)
   : mesh_(std::move(mesh))
{
   const int n = mesh_.mesh().GetNE();
   element_kinds_.assign(static_cast<std::size_t>(n), ElementKind::Straight);
   age_index_by_element_.assign(static_cast<std::size_t>(n), -1);
   curved_face_index_by_face_.assign(mesh_.faces().size(), -1);
}

AgeMesh::AgeMesh(MeshAdapter mesh,
                 std::vector<std::unique_ptr<BoundaryCurve>> curves,
                 std::vector<ElementKind> element_kinds,
                 std::vector<AgeElementGeometry> age_elements,
                 std::vector<CurvedFace> curved_faces)
   : mesh_(std::move(mesh)),
     curves_(std::move(curves)),
     element_kinds_(std::move(element_kinds)),
     age_elements_(std::move(age_elements)),
     curved_faces_(std::move(curved_faces))
{
   const int n = mesh_.mesh().GetNE();
   if (static_cast<int>(element_kinds_.size()) != n)
   {
      std::ostringstream os;
      os << "AgeMesh: element_kinds size " << element_kinds_.size()
         << " does not match mesh element count " << n << ".";
      throw std::runtime_error(os.str());
   }

   age_index_by_element_.assign(static_cast<std::size_t>(n), -1);
   for (std::size_t i = 0; i < age_elements_.size(); ++i)
   {
      const int elem = age_elements_[i].element;
      if (elem < 0 || elem >= n)
      {
         throw std::runtime_error("AgeMesh: AGE element index out of range.");
      }
      if (element_kinds_[static_cast<std::size_t>(elem)] != ElementKind::Age)
      {
         throw std::runtime_error("AgeMesh: AGE element record does not match element_kinds.");
      }
      if (age_index_by_element_[static_cast<std::size_t>(elem)] >= 0)
      {
         std::ostringstream os;
         os << "AgeMesh: element " << elem << " has more than one AGE record.";
         throw std::runtime_error(os.str());
      }
      age_index_by_element_[static_cast<std::size_t>(elem)] = static_cast<int>(i);
   }

   curved_face_index_by_face_.assign(mesh_.faces().size(), -1);
   for (std::size_t i = 0; i < curved_faces_.size(); ++i)
   {
      const int face = curved_faces_[i].face;
      if (face < 0 || static_cast<std::size_t>(face) >= curved_face_index_by_face_.size())
      {
         throw std::runtime_error("AgeMesh: curved face index out of range.");
      }
      curved_face_index_by_face_[static_cast<std::size_t>(face)] = static_cast<int>(i);
   }
}

int AgeMesh::element_count() const
{
   return mesh_.mesh().GetNE();
}

ElementKind AgeMesh::Kind(int element) const
{
   return element_kinds_.at(static_cast<std::size_t>(element));
}

const AgeElementGeometry *AgeMesh::AgeGeometry(int element) const
{
   const int idx = age_index_by_element_.at(static_cast<std::size_t>(element));
   if (idx < 0) { return nullptr; }
   return &age_elements_.at(static_cast<std::size_t>(idx));
}

const CurvedFace *AgeMesh::CurvedFaceOf(int face) const
{
   const int idx = curved_face_index_by_face_.at(static_cast<std::size_t>(face));
   if (idx < 0) { return nullptr; }
   return &curved_faces_.at(static_cast<std::size_t>(idx));
}

} // namespace callaway
