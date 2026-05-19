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

constexpr double kPi = 3.141592653589793238462643383279502884;

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

// Same half-disk mesh + sidecar as test_age_preprocessor.
const char *kHalfDiskMesh =
   "MFEM mesh v1.0\n"
   "\ndimension\n2\n"
   "\nelements\n2\n15 2 0 1 3\n15 2 1 2 3\n"
   "\nboundary\n4\n9 1 0 1\n9 1 1 2\n10 1 2 3\n11 1 3 0\n"
   "\nvertices\n4\n2\n"
   "1.0 0.0\n0.0 1.0\n-1.0 0.0\n0.0 0.0\n";

const char *kHalfDiskSidecar =
   "version: 1\n"
   "curves:\n"
   "  - boundary_id: 9\n"
   "    type: circular_arc\n"
   "    center: [0.0, 0.0]\n"
   "    radius: 1.0\n"
   "    orientation: ccw\n";

// Small straight-sided unit-square mesh for the bit-for-bit AGE vs. straight comparison.
const char *kSquareMesh =
   "MFEM mesh v1.0\n"
   "\ndimension\n2\n"
   "\nelements\n2\n15 2 0 1 2\n15 2 0 2 3\n"
   "\nboundary\n4\n10 1 0 1\n11 1 1 2\n12 1 2 3\n13 1 3 0\n"
   "\nvertices\n4\n2\n"
   "0.0 0.0\n1.0 0.0\n1.0 1.0\n0.0 1.0\n";

std::vector<callaway::BoundaryCondition> MakeHalfDiskBCs()
{
   std::vector<callaway::BoundaryCondition> bcs(3);
   bcs[0] = {"Arc",  9, callaway::BoundaryType::Thermalizing, 1.0, 0.0, 0.0};
   bcs[1] = {"NegX", 10, callaway::BoundaryType::Thermalizing, 0.0, 0.0, 0.0};
   bcs[2] = {"PosX", 11, callaway::BoundaryType::Thermalizing, 0.0, 0.0, 0.0};
   return bcs;
}

callaway::AngularQuadrature MakeQuadrature()
{
   callaway::VelocityMeshSettings vm;
   vm.polar_angles = 4;
   vm.azimuthal_angles = 8;
   return callaway::AngularQuadrature(vm, 1.0);
}

void TestAgeAreasAndArcLength(const std::filesystem::path &tmpdir)
{
   const auto mesh_path = tmpdir / "halfdisk.mesh";
   const auto sidecar_path = tmpdir / "halfdisk.age.yaml";
   WriteFile(mesh_path, kHalfDiskMesh);
   WriteFile(sidecar_path, kHalfDiskSidecar);

   callaway::MeshAdapter mesh(mesh_path);
   const callaway::AgePreprocessor pre;
   callaway::AgeMesh age_mesh = pre.Build(std::move(mesh), sidecar_path, MakeHalfDiskBCs());

   const int order = 2;
   const callaway::NodalBasis basis(order);
   const auto age_bases = callaway::BuildAgeElementBases(age_mesh, order);
   const callaway::AngularQuadrature quadrature = MakeQuadrature();
   callaway::AgeSettings settings;  // default: Precomputed, 15 / 15 quadrature points

   const callaway::IntegrationCache cache(age_mesh, basis, age_bases, quadrature, settings);

   // Total area of the half-disk = pi/2.
   CheckClose(cache.TotalArea(), 0.5 * kPi, 1.0e-10, "total area = pi/2");

   // Per-element area ~ pi/4 each (quarter-disks).
   for (int e = 0; e < cache.element_count(); ++e)
   {
      const auto &g = cache.Geometry(e);
      CheckClose(g.area, 0.25 * kPi, 1.0e-10, "AGE element area = pi/4");
   }

   // Curved face arc length per element = pi/2 (quarter-circle arc of unit circle).
   for (const callaway::AgeElementGeometry &geom : age_mesh.age_elements())
   {
      const auto &g = cache.Geometry(geom.element);
      const double arc = g.face_lengths[geom.curved_local_face];
      CheckClose(arc, 0.5 * kPi, 1.0e-10, "AGE arc length = pi/2");
   }

   // CurvedFaceQuadrature: weights sum to arc length; normals are unit
   // and radially outward (= the curve point itself for unit circle at origin).
   for (const callaway::AgeElementGeometry &geom : age_mesh.age_elements())
   {
      const auto &rec = cache.CurvedFaceData(geom.element, geom.curved_local_face);
      double sum_w = 0.0;
      for (std::size_t q = 0; q < rec.points.size(); ++q)
      {
         sum_w += rec.weights[q];
         const double n_mag = std::hypot(rec.normals[q][0], rec.normals[q][1]);
         CheckClose(n_mag, 1.0, 1.0e-12, "curved normal unit");
         // Radial outward for the unit circle at the origin: n = point.
         CheckClose(rec.normals[q][0], rec.points[q][0], 1.0e-12, "curved normal radial x");
         CheckClose(rec.normals[q][1], rec.points[q][1], 1.0e-12, "curved normal radial y");
      }
      CheckClose(sum_w, 0.5 * kPi, 1.0e-10, "curved face weights sum to arc length");
   }
}

void TestMassPositivityAndConsistency(const std::filesystem::path &tmpdir)
{
   const auto mesh_path = tmpdir / "halfdisk.mesh";
   const auto sidecar_path = tmpdir / "halfdisk.age.yaml";
   callaway::MeshAdapter mesh(mesh_path);
   const callaway::AgePreprocessor pre;
   callaway::AgeMesh age_mesh = pre.Build(std::move(mesh), sidecar_path, MakeHalfDiskBCs());

   const int order = 2;
   const callaway::NodalBasis basis(order);
   const auto age_bases = callaway::BuildAgeElementBases(age_mesh, order);
   const callaway::AngularQuadrature quadrature = MakeQuadrature();
   callaway::AgeSettings settings;

   const callaway::IntegrationCache cache(age_mesh, basis, age_bases, quadrature, settings);

   const int dofs = cache.dofs();
   for (int e = 0; e < cache.element_count(); ++e)
   {
      // Symmetry of Mass.
      for (int r = 0; r < dofs; ++r)
      {
         for (int c = 0; c < dofs; ++c)
         {
            CheckClose(cache.Mass(e, r, c), cache.Mass(e, c, r), 1.0e-12,
                       "mass symmetry");
         }
      }
      // Partition of unity: sum_c Mass(e, r, c) = integral(phi_r * 1) = BasisIntegral(e, r).
      for (int r = 0; r < dofs; ++r)
      {
         double sum = 0.0;
         for (int c = 0; c < dofs; ++c) { sum += cache.Mass(e, r, c); }
         CheckClose(sum, cache.BasisIntegral(e, r), 1.0e-9,
                    "Sigma_c Mass[r][c] = BasisIntegral[r]");
      }
      // Diagonal positivity.
      for (int r = 0; r < dofs; ++r)
      {
         if (cache.Mass(e, r, r) <= 0.0)
         {
            std::cerr << "FAIL mass diagonal not positive: e=" << e << " r=" << r
                      << " value=" << cache.Mass(e, r, r) << "\n";
            std::abort();
         }
      }
      // BasisIntegral sums to element area (sum_r BasisIntegral = integral(1) = area).
      double total = 0.0;
      for (int r = 0; r < dofs; ++r) { total += cache.BasisIntegral(e, r); }
      CheckClose(total, cache.Geometry(e).area, 1.0e-9,
                 "Sigma_r BasisIntegral[r] = element area");
   }
}

void TestPrecomputedMatchesOnTheFly(const std::filesystem::path &tmpdir)
{
   const auto mesh_path = tmpdir / "halfdisk.mesh";
   const auto sidecar_path = tmpdir / "halfdisk.age.yaml";

   callaway::MeshAdapter mesh1(mesh_path);
   callaway::MeshAdapter mesh2(mesh_path);
   const callaway::AgePreprocessor pre;
   callaway::AgeMesh am1 = pre.Build(std::move(mesh1), sidecar_path, MakeHalfDiskBCs());
   callaway::AgeMesh am2 = pre.Build(std::move(mesh2), sidecar_path, MakeHalfDiskBCs());

   const int order = 2;
   const callaway::NodalBasis basis(order);
   const auto bases1 = callaway::BuildAgeElementBases(am1, order);
   const auto bases2 = callaway::BuildAgeElementBases(am2, order);
   const callaway::AngularQuadrature quadrature = MakeQuadrature();

   callaway::AgeSettings pre_settings;
   pre_settings.curved_face_tensors = callaway::CurvedFaceTensorMode::Precomputed;
   callaway::AgeSettings otf_settings;
   otf_settings.curved_face_tensors = callaway::CurvedFaceTensorMode::OnTheFly;

   const callaway::IntegrationCache cache_pre(am1, basis, bases1, quadrature, pre_settings);
   const callaway::IntegrationCache cache_otf(am2, basis, bases2, quadrature, otf_settings);

   const int dofs = cache_pre.dofs();
   for (const callaway::AgeElementGeometry &geom : am1.age_elements())
   {
      for (int a = 0; a < quadrature.size(); ++a)
      {
         for (int r = 0; r < dofs; ++r)
         {
            CheckClose(
               cache_pre.CurvedFaceInflowWeight(a, geom.element, geom.curved_local_face, r),
               cache_otf.CurvedFaceInflowWeight(a, geom.element, geom.curved_local_face, r),
               1.0e-12, "precomputed vs on-the-fly inflow weight");
            for (int c = 0; c < dofs; ++c)
            {
               CheckClose(
                  cache_pre.CurvedFaceMatrix(a, geom.element, geom.curved_local_face, r, c),
                  cache_otf.CurvedFaceMatrix(a, geom.element, geom.curved_local_face, r, c),
                  1.0e-12, "precomputed vs on-the-fly matrix");
            }
         }
      }
   }
}

void TestStraightOnlyBitForBit(const std::filesystem::path &tmpdir)
{
   const auto mesh_path = tmpdir / "square.mesh";
   WriteFile(mesh_path, kSquareMesh);

   const int order = 3;
   const callaway::NodalBasis basis(order);

   callaway::MeshAdapter mesh_for_straight(mesh_path);
   const callaway::IntegrationCache cache_straight(mesh_for_straight, basis);

   callaway::MeshAdapter mesh_for_age(mesh_path);
   const callaway::AgePreprocessor pre;
   callaway::AgeMesh age_mesh = pre.BuildStraight(std::move(mesh_for_age));
   const callaway::AngularQuadrature quadrature = MakeQuadrature();
   callaway::AgeSettings settings;
   const callaway::IntegrationCache cache_age(age_mesh, basis,
                                              std::vector<callaway::AgeElementBasis>{},
                                              quadrature, settings);

   const double tol = 1.0e-14;
   const int dofs = cache_straight.dofs();
   const int fdofs = cache_straight.face_dofs();
   CheckClose(static_cast<double>(cache_age.dofs()), static_cast<double>(dofs), 0.0, "dofs match");
   CheckClose(static_cast<double>(cache_age.face_dofs()), static_cast<double>(fdofs), 0.0, "face dofs match");
   CheckClose(cache_age.TotalArea(), cache_straight.TotalArea(), tol, "total area match");

   for (int e = 0; e < cache_straight.element_count(); ++e)
   {
      const auto &gs = cache_straight.Geometry(e);
      const auto &ga = cache_age.Geometry(e);
      CheckClose(ga.area, gs.area, tol, "geom area");
      for (int k = 0; k < 3; ++k)
      {
         CheckClose(ga.face_lengths[k], gs.face_lengths[k], tol, "geom face_length");
         CheckClose(ga.outward_normals[k][0], gs.outward_normals[k][0], tol, "geom normal x");
         CheckClose(ga.outward_normals[k][1], gs.outward_normals[k][1], tol, "geom normal y");
      }
      for (int r = 0; r < dofs; ++r)
      {
         CheckClose(cache_age.BasisIntegral(e, r), cache_straight.BasisIntegral(e, r), tol,
                    "basis integral");
         for (int c = 0; c < dofs; ++c)
         {
            CheckClose(cache_age.Mass(e, r, c), cache_straight.Mass(e, r, c), tol, "mass");
            CheckClose(cache_age.GradX(e, r, c), cache_straight.GradX(e, r, c), tol, "gradx");
            CheckClose(cache_age.GradY(e, r, c), cache_straight.GradY(e, r, c), tol, "grady");
         }
      }
      for (int lf = 0; lf < 3; ++lf)
      {
         for (int r = 0; r < dofs; ++r)
         {
            for (int c = 0; c < dofs; ++c)
            {
               CheckClose(cache_age.ElementFaceMass(e, lf, r, c),
                          cache_straight.ElementFaceMass(e, lf, r, c), tol, "element face mass");
               CheckClose(cache_age.NeighborFaceMass(e, lf, r, c),
                          cache_straight.NeighborFaceMass(e, lf, r, c), tol, "neighbor face mass");
            }
            for (int fb = 0; fb < fdofs; ++fb)
            {
               CheckClose(cache_age.ElementFaceBasisMass(e, lf, r, fb),
                          cache_straight.ElementFaceBasisMass(e, lf, r, fb), tol,
                          "element face basis mass");
            }
         }
      }
   }
   // FaceMass over all global faces.
   const callaway::MeshAdapter mesh_for_face_count(mesh_path);
   const int face_count = static_cast<int>(mesh_for_face_count.faces().size());
   for (int f = 0; f < face_count; ++f)
   {
      for (int r = 0; r < fdofs; ++r)
      {
         for (int c = 0; c < fdofs; ++c)
         {
            CheckClose(cache_age.FaceMass(f, r, c), cache_straight.FaceMass(f, r, c),
                       tol, "face mass");
         }
      }
   }
}

} // namespace

int main()
{
   const auto tmpdir = std::filesystem::temp_directory_path() / "callaway_age_integration_test";
   std::filesystem::create_directories(tmpdir);

   TestAgeAreasAndArcLength(tmpdir);
   TestMassPositivityAndConsistency(tmpdir);
   TestPrecomputedMatchesOnTheFly(tmpdir);
   TestStraightOnlyBitForBit(tmpdir);

   std::cout << "test_age_integration_cache: all checks passed.\n";
   return 0;
}
