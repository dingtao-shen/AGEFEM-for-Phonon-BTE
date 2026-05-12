#include "callaway/angular_quadrature.hpp"
#include "callaway/config.hpp"
#include "callaway/distribution.hpp"
#include "callaway/integration_cache.hpp"
#include "callaway/mesh_adapter.hpp"
#include "callaway/moment_calculator.hpp"
#include "callaway/nodal_basis.hpp"
#include "callaway/sweep_ordering.hpp"

#include <cassert>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <vector>

namespace
{

void CheckClose(double actual, double expected, double tolerance)
{
   assert(std::abs(actual - expected) <= tolerance);
}

} // namespace

int main(int argc, char **argv)
{
   if (argc < 3)
   {
      std::cerr << "Usage: test_moments_sweep CONFIG MESH\n";
      return 2;
   }

   callaway::Config config = callaway::LoadConfig(argv[1]);
   config.files.mesh = std::filesystem::path(argv[2]);
   config.Validate();

   const callaway::MeshAdapter mesh(config.files.mesh);
   const callaway::NodalBasis basis(config.dg.order);
   const callaway::IntegrationCache integration(mesh, basis);
   const callaway::AngularQuadrature quadrature(config.velocity_mesh, config.flow.group_velocity);

   callaway::Distribution distribution(quadrature.size(), integration.element_count(), integration.dofs());
   distribution.SetThermalEquilibrium(config.flow.specific_heat, 1.0);

   const callaway::MomentFields moments =
      callaway::MomentCalculator::Compute(distribution, quadrature, integration, config.flow.specific_heat);

   for (int elem = 0; elem < integration.element_count(); ++elem)
   {
      CheckClose(moments.TemperatureCell(elem), integration.Geometry(elem).area, 1.0e-12);
      CheckClose(moments.HeatFluxXCell(elem), 0.0, 1.0e-12);
      CheckClose(moments.HeatFluxYCell(elem), 0.0, 1.0e-12);
      for (int dof = 0; dof < integration.dofs(); ++dof)
      {
         CheckClose(moments.TemperatureDof(elem, dof), 1.0, 1.0e-12);
         CheckClose(moments.HeatFluxXDof(elem, dof), 0.0, 1.0e-12);
         CheckClose(moments.HeatFluxYDof(elem, dof), 0.0, 1.0e-12);
      }
   }
   CheckClose(moments.Mass(), 1.0, 1.0e-12);

   const callaway::SweepOrdering ordering(mesh, integration, quadrature);
   assert(ordering.angles() == quadrature.size());
   assert(ordering.elements() == integration.element_count());

   for (int angle = 0; angle < ordering.angles(); ++angle)
   {
      const auto &order = ordering.Order(angle);
      assert(static_cast<int>(order.size()) == ordering.elements());

      std::vector<int> seen(static_cast<std::size_t>(ordering.elements()), 0);
      for (int i = 0; i < ordering.elements(); ++i)
      {
         const int elem = order[static_cast<std::size_t>(i)];
         assert(elem >= 0);
         assert(elem < ordering.elements());
         assert(ordering.Position(angle, elem) == i);
         seen[static_cast<std::size_t>(elem)] += 1;
      }
      for (const int count : seen) { assert(count == 1); }

      const callaway::Direction &direction = quadrature[angle];
      for (int elem = 0; elem < ordering.elements(); ++elem)
      {
         for (int local = 0; local < 3; ++local)
         {
            const int neighbor = mesh.ElementNeighbor(elem, local);
            if (neighbor < 0) { continue; }

            const auto normal = integration.OutwardNormal(elem, local);
            const double speed = direction.cx * normal[0] + direction.cy * normal[1];
            if (speed > 1.0e-14)
            {
               assert(ordering.Position(angle, neighbor) > ordering.Position(angle, elem));
            }
            if (speed < -1.0e-14)
            {
               assert(ordering.Position(angle, neighbor) < ordering.Position(angle, elem));
            }
         }
      }
   }

   return 0;
}
