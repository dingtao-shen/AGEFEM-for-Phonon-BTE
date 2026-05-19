#include "callaway/age_basis.hpp"
#include "callaway/age_mesh.hpp"
#include "callaway/age_preprocessor.hpp"
#include "callaway/angular_quadrature.hpp"
#include "callaway/boundary.hpp"
#include "callaway/config.hpp"
#include "callaway/integration_cache.hpp"
#include "callaway/mesh_adapter.hpp"
#include "callaway/nodal_basis.hpp"

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

void WriteFile(const std::filesystem::path &path, const std::string &content)
{
   std::ofstream out(path);
   if (!out) { std::cerr << "Failed to open " << path << "\n"; std::abort(); }
   out << content;
}

// Single triangle (0,0), (1,0), (0,1) in CCW order. Local face 1 (v1 -> v2)
// is the curved polyline boundary; the other two faces are straight axes.
const char *kSingleTriMesh =
   "MFEM mesh v1.0\n\n"
   "dimension\n2\n\n"
   "elements\n1\n15 2 0 1 2\n\n"
   "boundary\n3\n"
   "10 1 0 1\n"   // bottom (y = 0)
   "9 1 1 2\n"    // curved polyline
   "11 1 2 0\n"   // left (x = 0)
   "\nvertices\n3\n2\n"
   "0.0 0.0\n1.0 0.0\n0.0 1.0\n";

// 5-node polyline bulging outward from the chord (1,0)-(0,1).
const char *kBulgeData =
   "5\n"
   "1.0  0.0\n"
   "0.95 0.2\n"
   "0.7  0.7\n"
   "0.2  0.95\n"
   "0.0  1.0\n";

const char *kSidecar =
   "version: 1\n"
   "curves:\n"
   "  - boundary_id: 9\n"
   "    type: polyline\n"
   "    closed: false\n"
   "    orientation: ccw\n"
   "    data_file: bulge.txt\n";

// Analytic geometric quantities for the bulged polyline element above.
constexpr double kExpectedArcLength = 0.2061552812808831  +
                                      0.5590169943749474  +
                                      0.5590169943749474  +
                                      0.2061552812808831;
constexpr double kExpectedArea = 0.725;

} // namespace

int main()
{
   const auto tmpdir = std::filesystem::temp_directory_path() / "callaway_polyline_age_test";
   std::filesystem::create_directories(tmpdir);
   const auto mesh_path = tmpdir / "single.mesh";
   const auto sidecar_path = tmpdir / "single.age.yaml";
   const auto data_path = tmpdir / "bulge.txt";
   WriteFile(mesh_path, kSingleTriMesh);
   WriteFile(sidecar_path, kSidecar);
   WriteFile(data_path, kBulgeData);

   std::vector<callaway::BoundaryCondition> bcs(3);
   bcs[0] = {"Curve",  9,  callaway::BoundaryType::Thermalizing, 0.0, 0.0, 0.0};
   bcs[1] = {"Bottom", 10, callaway::BoundaryType::Thermalizing, 0.0, 0.0, 0.0};
   bcs[2] = {"Left",   11, callaway::BoundaryType::Thermalizing, 0.0, 0.0, 0.0};

   callaway::MeshAdapter mesh(mesh_path);
   const callaway::AgePreprocessor pre;
   callaway::AgePreprocessReport report;
   callaway::AgeMesh age_mesh = pre.Build(std::move(mesh), sidecar_path, bcs, &report);

   if (age_mesh.age_element_count() != 1)
   {
      std::cerr << "FAIL: expected 1 AGE element, got " << age_mesh.age_element_count() << "\n";
      return 1;
   }
   if (age_mesh.curved_face_count() != 1)
   {
      std::cerr << "FAIL: expected 1 curved face, got " << age_mesh.curved_face_count() << "\n";
      return 1;
   }
   CheckClose(report.max_endpoint_projection_error, 0.0, 1.0e-14,
              "polyline endpoint projection error");

   const int order = 2;
   const callaway::NodalBasis basis(order);
   const auto age_bases = callaway::BuildAgeElementBases(age_mesh, order);

   callaway::VelocityMeshSettings vm;
   vm.polar_angles = 2;
   vm.azimuthal_angles = 8;
   const callaway::AngularQuadrature quadrature(vm, 1.0);

   callaway::AgeSettings age_settings;
   const callaway::IntegrationCache cache(age_mesh, basis, age_bases, quadrature, age_settings);

   // Total area = polygon area via shoelace.
   CheckClose(cache.TotalArea(), kExpectedArea, 1.0e-12, "polyline AGE total area");
   CheckClose(cache.Geometry(0).area, kExpectedArea, 1.0e-12, "polyline AGE element area");

   // Arc length = sum of polyline segment lengths.
   const auto &geom = age_mesh.age_elements()[0];
   const double arc = cache.Geometry(geom.element).face_lengths[geom.curved_local_face];
   CheckClose(arc, kExpectedArcLength, 1.0e-12, "polyline AGE arc length");

   // Straight-edge lengths: bottom = 1.0, left = 1.0.
   const auto &g = cache.Geometry(0);
   double straight_total = 0.0;
   for (int lf = 0; lf < 3; ++lf)
   {
      if (lf == geom.curved_local_face) { continue; }
      straight_total += g.face_lengths[lf];
   }
   CheckClose(straight_total, 2.0, 1.0e-13, "polyline AGE straight edges total");

   // Mass-matrix symmetry, diagonal positivity, partition-of-unity consistency.
   const int dofs = cache.dofs();
   for (int r = 0; r < dofs; ++r)
   {
      if (cache.Mass(0, r, r) <= 0.0)
      {
         std::cerr << "FAIL: mass diagonal not positive at r=" << r << "\n";
         return 1;
      }
      for (int c = 0; c < dofs; ++c)
      {
         CheckClose(cache.Mass(0, r, c), cache.Mass(0, c, r), 1.0e-12,
                    "polyline AGE mass symmetry");
      }
      double row_sum = 0.0;
      for (int c = 0; c < dofs; ++c) { row_sum += cache.Mass(0, r, c); }
      CheckClose(row_sum, cache.BasisIntegral(0, r), 1.0e-10,
                 "polyline AGE Sigma_c Mass[r][c] = BasisIntegral[r]");
   }
   double basis_total = 0.0;
   for (int r = 0; r < dofs; ++r) { basis_total += cache.BasisIntegral(0, r); }
   CheckClose(basis_total, kExpectedArea, 1.0e-12,
              "polyline AGE Sigma_r BasisIntegral[r] = area");

   // CurvedFaceQuadrature: per-segment GL points, weights sum to arc length,
   // every normal is a unit vector and matches the outward sense (positive
   // dot product with the displacement from the centroid to the curve point —
   // the AGE element interior is toward the origin).
   const auto &rec = cache.CurvedFaceData(geom.element, geom.curved_local_face);
   double sum_w = 0.0;
   for (std::size_t q = 0; q < rec.points.size(); ++q)
   {
      sum_w += rec.weights[q];
      const double nmag = std::hypot(rec.normals[q][0], rec.normals[q][1]);
      CheckClose(nmag, 1.0, 1.0e-13, "polyline AGE curved face normal unit");
      // The polyline bulges outward away from the origin; outward normal at
      // each quadrature point should point away from the origin too — i.e.
      // it should have a non-negative dot product with the position vector.
      const double dot = rec.normals[q][0] * rec.points[q][0] +
                         rec.normals[q][1] * rec.points[q][1];
      if (dot < -1.0e-12)
      {
         std::cerr << "FAIL: polyline AGE curved-face normal points inward at q="
                   << q << " (dot " << dot << ")\n";
         return 1;
      }
   }
   CheckClose(sum_w, kExpectedArcLength, 1.0e-12,
              "polyline AGE curved face weights sum to arc length");

   std::cout << "test_polyline_age: passed.\n";
   return 0;
}
