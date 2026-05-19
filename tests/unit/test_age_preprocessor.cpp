#include "callaway/age_mesh.hpp"
#include "callaway/age_preprocessor.hpp"
#include "callaway/boundary.hpp"
#include "callaway/mesh_adapter.hpp"

#include <cassert>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace
{

void CheckClose(double actual, double expected, double tol, const char *what)
{
   if (!(std::abs(actual - expected) <= tol))
   {
      std::cerr << "FAIL " << what << ": got " << actual << ", expected " << expected
                << ", diff " << std::abs(actual - expected) << "\n";
      std::abort();
   }
}

void CheckEqual(int actual, int expected, const char *what)
{
   if (actual != expected)
   {
      std::cerr << "FAIL " << what << ": got " << actual << ", expected " << expected << "\n";
      std::abort();
   }
}

void WriteFile(const std::filesystem::path &path, const std::string &content)
{
   std::ofstream out(path);
   if (!out) { std::cerr << "Failed to open " << path << " for writing\n"; std::abort(); }
   out << content;
}

// Tiny MFEM v1.0 mesh: upper half of the unit disk, two triangles.
//
//          v1 = (0, 1)
//          /\.
//         /  \.
//        /    \.   <- curved boundary, attribute 9 (unit circle)
//       /      \.
//      /        \.
//   v2 ---v3----- v0       v0 = (1, 0), v2 = (-1, 0), v3 = (0, 0)
//   (-1,0)  (0,0)   (1,0)
//
// Triangles (CCW): T0 = (v0, v1, v3), T1 = (v1, v2, v3).
// Boundary edges:
//   v0->v1  : attr 9 (curved, on unit circle, arc [0, 1/4])
//   v1->v2  : attr 9 (curved, on unit circle, arc [1/4, 1/2])
//   v2->v3  : attr 10 (straight, negative x axis)
//   v3->v0  : attr 11 (straight, positive x axis)
const char *kHalfDiskMesh =
   "MFEM mesh v1.0\n"
   "\n"
   "dimension\n"
   "2\n"
   "\n"
   "elements\n"
   "2\n"
   "15 2 0 1 3\n"
   "15 2 1 2 3\n"
   "\n"
   "boundary\n"
   "4\n"
   "9 1 0 1\n"
   "9 1 1 2\n"
   "10 1 2 3\n"
   "11 1 3 0\n"
   "\n"
   "vertices\n"
   "4\n"
   "2\n"
   "1.0 0.0\n"
   "0.0 1.0\n"
   "-1.0 0.0\n"
   "0.0 0.0\n";

const char *kHalfDiskSidecar =
   "version: 1\n"
   "curves:\n"
   "  - boundary_id: 9\n"
   "    type: circular_arc\n"
   "    center: [0.0, 0.0]\n"
   "    radius: 1.0\n"
   "    orientation: ccw\n";

std::vector<callaway::BoundaryCondition> MakeHalfDiskBoundaryConditions()
{
   std::vector<callaway::BoundaryCondition> bcs(3);
   bcs[0].name = "Arc";
   bcs[0].physical_id = 9;
   bcs[0].type = callaway::BoundaryType::Thermalizing;
   bcs[0].temperature = 1.0;
   bcs[1].name = "NegX";
   bcs[1].physical_id = 10;
   bcs[1].type = callaway::BoundaryType::Thermalizing;
   bcs[1].temperature = 0.0;
   bcs[2].name = "PosX";
   bcs[2].physical_id = 11;
   bcs[2].type = callaway::BoundaryType::Thermalizing;
   bcs[2].temperature = 0.0;
   return bcs;
}

std::filesystem::path PrepareTempDir(const std::string &name)
{
   const auto dir = std::filesystem::temp_directory_path() / ("callaway_" + name);
   std::filesystem::create_directories(dir);
   return dir;
}

void TestStraightOnly(const std::filesystem::path &mesh_path)
{
   callaway::MeshAdapter mesh(mesh_path);
   const callaway::AgePreprocessor pre;
   callaway::AgePreprocessReport report;
   const callaway::AgeMesh age_mesh = pre.BuildStraight(std::move(mesh), &report);

   CheckEqual(age_mesh.element_count(), 2, "straight: element_count");
   CheckEqual(age_mesh.age_element_count(), 0, "straight: age_element_count");
   CheckEqual(age_mesh.curved_face_count(), 0, "straight: curved_face_count");
   if (age_mesh.has_age_elements())
   {
      std::cerr << "FAIL straight: has_age_elements true\n";
      std::abort();
   }
   for (int e = 0; e < age_mesh.element_count(); ++e)
   {
      if (age_mesh.Kind(e) != callaway::ElementKind::Straight)
      {
         std::cerr << "FAIL straight: element " << e << " not Straight\n";
         std::abort();
      }
      if (age_mesh.AgeGeometry(e) != nullptr)
      {
         std::cerr << "FAIL straight: element " << e << " has AGE geometry\n";
         std::abort();
      }
   }
   CheckEqual(report.straight_elements, 2, "straight: report.straight_elements");
   CheckEqual(report.age_elements, 0, "straight: report.age_elements");
}

void TestSemicircleAge(const std::filesystem::path &mesh_path,
                       const std::filesystem::path &sidecar_path)
{
   callaway::MeshAdapter mesh(mesh_path);
   const auto bcs = MakeHalfDiskBoundaryConditions();
   const callaway::AgePreprocessor pre;
   callaway::AgePreprocessReport report;
   const callaway::AgeMesh age_mesh = pre.Build(std::move(mesh), sidecar_path, bcs, &report);

   CheckEqual(age_mesh.element_count(), 2, "age: element_count");
   CheckEqual(age_mesh.age_element_count(), 2, "age: age_element_count");
   CheckEqual(age_mesh.curved_face_count(), 2, "age: curved_face_count");
   CheckEqual(static_cast<int>(age_mesh.curves().size()), 1, "age: bound curves");
   CheckEqual(report.bound_curves, 1, "age: report.bound_curves");
   CheckEqual(report.straight_elements, 0, "age: report.straight_elements");
   CheckEqual(report.age_elements, 2, "age: report.age_elements");
   CheckClose(report.max_endpoint_projection_error, 0.0, 1.0e-14,
              "age: report.max_endpoint_projection_error");

   for (int e = 0; e < age_mesh.element_count(); ++e)
   {
      if (age_mesh.Kind(e) != callaway::ElementKind::Age)
      {
         std::cerr << "FAIL age: element " << e << " not Age\n";
         std::abort();
      }
      const callaway::AgeElementGeometry *geom = age_mesh.AgeGeometry(e);
      if (geom == nullptr)
      {
         std::cerr << "FAIL age: element " << e << " has no AGE geometry\n";
         std::abort();
      }
      if (geom->curve == nullptr)
      {
         std::cerr << "FAIL age: element " << e << " AGE geometry has null curve\n";
         std::abort();
      }
      // The interior vertex of each AGE element is v3 = (0, 0).
      CheckClose(geom->interior_vertex[0], 0.0, 1.0e-15, "age: interior x");
      CheckClose(geom->interior_vertex[1], 0.0, 1.0e-15, "age: interior y");

      // Parameter interval length is 1/4 of the unit circle for each triangle.
      const double dl = geom->parameter_interval.end - geom->parameter_interval.begin;
      CheckClose(dl, 0.25, 1.0e-13, "age: parameter interval length");

      // curve_begin and curve_end lie on the unit circle.
      const double rb = std::hypot(geom->curve_begin[0], geom->curve_begin[1]);
      const double re = std::hypot(geom->curve_end[0], geom->curve_end[1]);
      CheckClose(rb, 1.0, 1.0e-13, "age: |curve_begin|");
      CheckClose(re, 1.0, 1.0e-13, "age: |curve_end|");
   }

   // Verify the two AGE elements cover the parameter intervals [0, 0.25] and
   // [0.25, 0.5] (order across elements is mesh-dependent, so check the union).
   bool seen_first = false;
   bool seen_second = false;
   for (const callaway::AgeElementGeometry &geom : age_mesh.age_elements())
   {
      const double a = geom.parameter_interval.begin;
      const double b = geom.parameter_interval.end;
      if (std::abs(a - 0.00) < 1.0e-13 && std::abs(b - 0.25) < 1.0e-13) { seen_first = true; }
      if (std::abs(a - 0.25) < 1.0e-13 && std::abs(b - 0.50) < 1.0e-13) { seen_second = true; }
   }
   if (!seen_first || !seen_second)
   {
      std::cerr << "FAIL age: did not see parameter intervals [0, 0.25] and [0.25, 0.5]\n";
      std::abort();
   }

   // CurvedFaceOf lookup: the two curved faces should be discoverable.
   int curved_seen = 0;
   for (const callaway::CurvedFace &cf : age_mesh.curved_faces())
   {
      if (cf.boundary_attribute != 9)
      {
         std::cerr << "FAIL age: curved face has unexpected attribute "
                   << cf.boundary_attribute << "\n";
         std::abort();
      }
      if (!age_mesh.IsCurvedFace(cf.face))
      {
         std::cerr << "FAIL age: IsCurvedFace returned false for known curved face "
                   << cf.face << "\n";
         std::abort();
      }
      ++curved_seen;
   }
   CheckEqual(curved_seen, 2, "age: iterated curved faces");
}

void TestOrientationMismatchRejected(const std::filesystem::path &mesh_path,
                                     const std::filesystem::path &tmpdir)
{
   // Same mesh, but the sidecar declares the curve with the WRONG orientation
   // (cw). The mesh CCW traversal of the curved boundary goes CCW around the
   // unit circle, so the pipeline should reject this.
   const std::string bad_sidecar =
      "version: 1\n"
      "curves:\n"
      "  - boundary_id: 9\n"
      "    type: circular_arc\n"
      "    center: [0.0, 0.0]\n"
      "    radius: 1.0\n"
      "    orientation: cw\n";
   const auto bad_path = tmpdir / "bad.age.yaml";
   WriteFile(bad_path, bad_sidecar);

   callaway::MeshAdapter mesh(mesh_path);
   const auto bcs = MakeHalfDiskBoundaryConditions();
   const callaway::AgePreprocessor pre;
   bool threw = false;
   try
   {
      (void) pre.Build(std::move(mesh), bad_path, bcs, nullptr);
   }
   catch (const std::exception &ex)
   {
      threw = true;
      const std::string msg = ex.what();
      if (msg.find("orientation") == std::string::npos &&
          msg.find("direction") == std::string::npos)
      {
         std::cerr << "FAIL orientation: rejection message did not mention orientation/direction: "
                   << msg << "\n";
         std::abort();
      }
   }
   if (!threw)
   {
      std::cerr << "FAIL orientation: pipeline accepted an inconsistent curve orientation\n";
      std::abort();
   }
}

} // namespace

int main()
{
   const auto tmpdir = PrepareTempDir("age_preprocessor_test");
   const auto mesh_path = tmpdir / "halfdisk.mesh";
   const auto sidecar_path = tmpdir / "halfdisk.age.yaml";

   WriteFile(mesh_path, kHalfDiskMesh);
   WriteFile(sidecar_path, kHalfDiskSidecar);

   TestStraightOnly(mesh_path);
   TestSemicircleAge(mesh_path, sidecar_path);
   TestOrientationMismatchRejected(mesh_path, tmpdir);

   std::cout << "test_age_preprocessor: all checks passed.\n";
   return 0;
}
