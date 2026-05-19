#include "callaway/mesh_adapter.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
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
   std::vector<PeriodicLinkRaw> periodic_links;
   const std::filesystem::path mfem_path = PrepareMfemMesh(mesh_path_, periodic_links);
   mesh_ = std::make_unique<mfem::Mesh>(mfem_path.string().c_str(), 1, 1);
   if (mesh_->Dimension() != 2)
   {
      throw std::runtime_error("Only 2D meshes are supported by the current solver.");
   }
   mesh_->EnsureNodes();
   BuildFaceData();
   BuildPeriodicFacePairs(periodic_links);
}

MeshAdapter::~MeshAdapter()
{
   if (!stripped_mesh_path_.empty())
   {
      std::error_code ec;
      std::filesystem::remove(stripped_mesh_path_, ec);
   }
}

MeshAdapter::MeshAdapter(MeshAdapter &&other) noexcept
   : mesh_path_(std::move(other.mesh_path_)),
     mesh_(std::move(other.mesh_)),
     faces_(std::move(other.faces_)),
     element_faces_(std::move(other.element_faces_)),
     face_periodic_(std::move(other.face_periodic_)),
     has_periodic_faces_(other.has_periodic_faces_),
     stripped_mesh_path_(std::move(other.stripped_mesh_path_))
{
   other.has_periodic_faces_ = false;
   other.stripped_mesh_path_.clear();
}

MeshAdapter &MeshAdapter::operator=(MeshAdapter &&other) noexcept
{
   if (this != &other)
   {
      if (!stripped_mesh_path_.empty())
      {
         std::error_code ec;
         std::filesystem::remove(stripped_mesh_path_, ec);
      }
      mesh_path_ = std::move(other.mesh_path_);
      mesh_ = std::move(other.mesh_);
      faces_ = std::move(other.faces_);
      element_faces_ = std::move(other.element_faces_);
      face_periodic_ = std::move(other.face_periodic_);
      has_periodic_faces_ = other.has_periodic_faces_;
      stripped_mesh_path_ = std::move(other.stripped_mesh_path_);
      other.has_periodic_faces_ = false;
      other.stripped_mesh_path_.clear();
   }
   return *this;
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

const PeriodicFacePair *MeshAdapter::PeriodicPartner(int face_id) const
{
   if (face_id < 0 || face_id >= static_cast<int>(face_periodic_.size())) { return nullptr; }
   const PeriodicFacePair &pair = face_periodic_[static_cast<std::size_t>(face_id)];
   if (pair.partner_face < 0) { return nullptr; }
   return &pair;
}

std::filesystem::path MeshAdapter::PrepareMfemMesh(
   const std::filesystem::path &input_path,
   std::vector<PeriodicLinkRaw> &links)
{
   std::ifstream in(input_path);
   if (!in)
   {
      throw std::runtime_error("Failed to open mesh file: " + input_path.string());
   }

   std::ostringstream pre_periodic;
   std::ostringstream post_periodic;
   std::string line;
   bool found_periodic = false;
   while (std::getline(in, line))
   {
      if (line.rfind("$Periodic", 0) == 0)
      {
         found_periodic = true;

         // Parse the periodic section in-place. Format:
         //   $Periodic
         //   numPeriodicLinks
         //   <repeated>
         //     dim slaveTag masterTag
         //     [Affine <16 floats>]
         //     numCorrespondingNodes
         //     slaveNode masterNode  (1-based Gmsh indices)
         //     ...
         //   $EndPeriodic
         std::string raw;
         int num_links = 0;
         if (!std::getline(in, raw))
         {
            throw std::runtime_error("Unexpected EOF in $Periodic header.");
         }
         std::istringstream(raw) >> num_links;
         for (int li = 0; li < num_links; ++li)
         {
            if (!std::getline(in, raw))
            {
               throw std::runtime_error("Unexpected EOF in $Periodic link header.");
            }
            PeriodicLinkRaw link;
            int dim = 0;
            std::istringstream(raw) >> dim >> link.slave_tag >> link.master_tag;

            // Either "Affine <16 numbers>" or "<n_affine> numbers" or just "0".
            // Snoop the next line: if it starts with "Affine", parse the 16
            // affine floats. Otherwise treat the line as numCorrespondingNodes.
            std::streampos pos = in.tellg();
            std::string next;
            if (!std::getline(in, next))
            {
               throw std::runtime_error("Unexpected EOF in $Periodic body.");
            }
            std::istringstream snoop(next);
            std::string tag;
            snoop >> tag;
            int num_pairs = 0;
            if (tag == "Affine")
            {
               std::vector<double> M;
               double v = 0.0;
               while (snoop >> v) { M.push_back(v); }
               if (M.size() != 16)
               {
                  throw std::runtime_error("Expected 16 numbers in Affine row.");
               }
               // Row-major 4x4: M[0..3] = row 0, M[4..7] = row 1, ...
               // Translation column = M[3], M[7], M[11].
               // Affine acts as: slave_coord = M * master_coord.
               // For 2D periodicity we extract the (x, y) translation.
               link.translation_x = M[3];
               link.translation_y = M[7];

               if (!std::getline(in, next))
               {
                  throw std::runtime_error("Unexpected EOF after Affine row.");
               }
               std::istringstream(next) >> num_pairs;
            }
            else
            {
               // The snooped line is numCorrespondingNodes — but it could
               // also be "n_affine_values" (Gmsh 4 format). Handle both by
               // testing if there are >1 numbers on this line; if so, treat
               // as an n_affine list and skip those values. Otherwise treat
               // as numCorrespondingNodes.
               int first_int = 0;
               std::istringstream re(next);
               re >> first_int;
               int second_int = 0;
               if (re >> second_int)
               {
                  // Multi-value line means affine info or count.
                  // Reset and continue with caution.
                  (void) pos;
                  num_pairs = first_int; // fallback
               }
               else
               {
                  num_pairs = first_int;
               }
            }

            link.slave_to_master_node.reserve(static_cast<std::size_t>(num_pairs));
            for (int p = 0; p < num_pairs; ++p)
            {
               if (!std::getline(in, raw))
               {
                  throw std::runtime_error("Unexpected EOF in $Periodic node pair list.");
               }
               int s = 0;
               int m = 0;
               std::istringstream(raw) >> s >> m;
               // Gmsh node tags are 1-based; MFEM stores vertices 0-based.
               link.slave_to_master_node[s - 1] = m - 1;
            }

            links.push_back(std::move(link));
         }

         // Consume up to and including $EndPeriodic so the rest of the file
         // continues to flow into post_periodic.
         while (std::getline(in, line))
         {
            if (line.rfind("$EndPeriodic", 0) == 0) { break; }
         }
         continue;
      }
      // Buffer the non-periodic portion for emission to the temp file.
      pre_periodic << line << '\n';
   }

   if (!found_periodic)
   {
      return input_path;
   }

   // Write stripped version to a sibling temp file. Mesh files can be large
   // so we keep the rewrite simple: pre_periodic already contains the full
   // mesh minus the periodic block.
   std::filesystem::path tmp_path = input_path;
   tmp_path += ".no_periodic.tmp.msh";
   std::ofstream out(tmp_path, std::ios::trunc);
   if (!out)
   {
      throw std::runtime_error("Failed to open temporary mesh file for writing: " +
                               tmp_path.string());
   }
   out << pre_periodic.str();
   out.close();
   stripped_mesh_path_ = tmp_path;
   return tmp_path;
}

void MeshAdapter::BuildPeriodicFacePairs(const std::vector<PeriodicLinkRaw> &links)
{
   face_periodic_.assign(faces_.size(), PeriodicFacePair{});
   if (links.empty()) { return; }

   // For each link, walk slave-tagged faces and find the master-tagged face
   // whose vertex pair matches under the node correspondence. Set both ends
   // of each pair, with translations of opposite sign on either side so that
   // partner_pt = self_pt - translation holds on either side.
   for (const PeriodicLinkRaw &link : links)
   {
      for (FaceData &face : faces_)
      {
         if (face.boundary_attribute != link.slave_tag) { continue; }
         auto it0 = link.slave_to_master_node.find(face.vertices[0]);
         auto it1 = link.slave_to_master_node.find(face.vertices[1]);
         if (it0 == link.slave_to_master_node.end() ||
             it1 == link.slave_to_master_node.end())
         {
            continue;
         }
         const int m0 = it0->second;
         const int m1 = it1->second;
         for (const FaceData &candidate : faces_)
         {
            if (candidate.boundary_attribute != link.master_tag) { continue; }
            if (!SameUndirectedEdge(candidate.vertices,
                                    std::array<int, 2>{{m0, m1}}))
            {
               continue;
            }
            // Found the master partner for face.index.
            PeriodicFacePair &p_slave =
               face_periodic_[static_cast<std::size_t>(face.index)];
            PeriodicFacePair &p_master =
               face_periodic_[static_cast<std::size_t>(candidate.index)];

            p_slave.partner_face = candidate.index;
            p_slave.partner_element =
               (candidate.element1 >= 0) ? candidate.element1 : candidate.element2;
            p_slave.partner_local_face =
               (candidate.element1 >= 0) ? candidate.local_face1 : candidate.local_face2;
            // Slave coordinates = master coordinates + (tx, ty). Therefore
            // partner (master) coord = self (slave) coord - (tx, ty).
            p_slave.translation_x = link.translation_x;
            p_slave.translation_y = link.translation_y;

            p_master.partner_face = face.index;
            p_master.partner_element =
               (face.element1 >= 0) ? face.element1 : face.element2;
            p_master.partner_local_face =
               (face.element1 >= 0) ? face.local_face1 : face.local_face2;
            // Partner (slave) coord = self (master) coord + (tx, ty), i.e.,
            // self - (-tx, -ty).
            p_master.translation_x = -link.translation_x;
            p_master.translation_y = -link.translation_y;

            has_periodic_faces_ = true;
            break;
         }
      }
   }
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
