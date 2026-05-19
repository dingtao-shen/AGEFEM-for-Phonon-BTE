#pragma once

#include "callaway/config.hpp"

#include <mfem.hpp>

#include <filesystem>
#include <array>
#include <memory>
#include <unordered_map>
#include <vector>

namespace callaway
{

struct MeshSummary
{
   int dimension = 0;
   int vertices = 0;
   int elements = 0;
   int boundary_elements = 0;
   int faces = 0;
   std::vector<int> element_attributes;
   std::vector<int> boundary_attributes;
};

struct FaceData
{
   int index = -1;
   std::array<int, 2> vertices{{-1, -1}};
   double length = 0.0;
   int boundary_attribute = 0;
   int element1 = -1;
   int element2 = -1;
   int local_face1 = -1;
   int local_face2 = -1;

   bool is_boundary() const { return boundary_attribute > 0; }
   bool is_interior() const { return boundary_attribute == 0 && element1 >= 0 && element2 >= 0; }
};

// Periodic boundary face linkage. translation_x/y is defined so that
// partner_coordinates = self_coordinates - (translation_x, translation_y).
struct PeriodicFacePair
{
   int partner_face = -1;
   int partner_element = -1;
   int partner_local_face = -1;
   double translation_x = 0.0;
   double translation_y = 0.0;
};

class MeshAdapter
{
public:
   explicit MeshAdapter(std::filesystem::path mesh_path);
   ~MeshAdapter();

   MeshAdapter(const MeshAdapter &) = delete;
   MeshAdapter &operator=(const MeshAdapter &) = delete;
   MeshAdapter(MeshAdapter &&other) noexcept;
   MeshAdapter &operator=(MeshAdapter &&other) noexcept;

   const std::filesystem::path &path() const { return mesh_path_; }
   const mfem::Mesh &mesh() const { return *mesh_; }
   mfem::Mesh &mesh() { return *mesh_; }

   MeshSummary Summary() const;
   bool HasBoundaryAttribute(int attribute) const;
   void ValidateBoundaryAttributes(const std::vector<BoundaryCondition> &bcs) const;
   const std::vector<FaceData> &faces() const { return faces_; }
   const FaceData &Face(int face) const { return faces_.at(face); }
   const std::array<int, 3> &ElementFaces(int element) const { return element_faces_.at(element); }
   int ElementFace(int element, int local_face) const { return element_faces_.at(element).at(local_face); }
   int ElementNeighbor(int element, int local_face) const;
   int BoundaryFaceCount() const;
   int InteriorFaceCount() const;

   // Periodic boundary support. Faces matched via the Gmsh $Periodic section
   // (which MFEM's reader does not consume) are recorded here. has_periodic
   // is true iff at least one periodic face pair was discovered.
   bool has_periodic_faces() const { return has_periodic_faces_; }
   const PeriodicFacePair *PeriodicPartner(int face_id) const;
   const std::vector<PeriodicFacePair> &PeriodicFaces() const { return face_periodic_; }

private:
   struct PeriodicLinkRaw
   {
      int slave_tag = 0;
      int master_tag = 0;
      double translation_x = 0.0;  // affine translation: slave = master + (tx, ty)
      double translation_y = 0.0;
      std::unordered_map<int, int> slave_to_master_node;  // 0-based vertex IDs
   };

   std::filesystem::path mesh_path_;
   std::unique_ptr<mfem::Mesh> mesh_;
   std::vector<FaceData> faces_;
   std::vector<std::array<int, 3>> element_faces_;

   // Periodic state. face_periodic_ is sized to the number of faces; entries
   // with partner_face < 0 are not periodic.
   std::vector<PeriodicFacePair> face_periodic_;
   bool has_periodic_faces_ = false;
   // Path of the temporary stripped mesh handed to MFEM when the original
   // mesh contained a $Periodic section. Cleaned up in the destructor.
   std::filesystem::path stripped_mesh_path_;

   std::filesystem::path PrepareMfemMesh(const std::filesystem::path &input_path,
                                         std::vector<PeriodicLinkRaw> &links);
   void BuildFaceData();
   void BuildPeriodicFacePairs(const std::vector<PeriodicLinkRaw> &links);
};

} // namespace callaway
