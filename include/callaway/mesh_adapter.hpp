#pragma once

#include "callaway/config.hpp"

#include <mfem.hpp>

#include <filesystem>
#include <array>
#include <memory>
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

class MeshAdapter
{
public:
   explicit MeshAdapter(std::filesystem::path mesh_path);

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

private:
   std::filesystem::path mesh_path_;
   std::unique_ptr<mfem::Mesh> mesh_;
   std::vector<FaceData> faces_;
   std::vector<std::array<int, 3>> element_faces_;

   void BuildFaceData();
};

} // namespace callaway
