#include "callaway/moment_calculator.hpp"

#include <stdexcept>

namespace callaway
{

MomentFields MomentCalculator::Compute(const Distribution &distribution,
                                       const AngularQuadrature &quadrature,
                                       const IntegrationCache &integration,
                                       double specific_heat)
{
   if (distribution.angles() != quadrature.size())
   {
      throw std::runtime_error("Distribution angle count does not match angular quadrature.");
   }
   if (distribution.elements() != integration.element_count() ||
       distribution.dofs() != integration.dofs())
   {
      throw std::runtime_error("Distribution spatial shape does not match integration cache.");
   }
   if (specific_heat <= 0.0)
   {
      throw std::runtime_error("specific_heat must be positive.");
   }

   MomentFields fields(distribution.elements(), distribution.dofs());

   for (int elem = 0; elem < distribution.elements(); ++elem)
   {
      for (int dof = 0; dof < distribution.dofs(); ++dof)
      {
         double energy = 0.0;
         double qx = 0.0;
         double qy = 0.0;
         for (int angle = 0; angle < distribution.angles(); ++angle)
         {
            const Direction &direction = quadrature[angle];
            const double value = distribution(angle, elem, dof);
            energy += value * direction.weight;
            qx += direction.cx * value * direction.weight;
            qy += direction.cy * value * direction.weight;
         }

         fields.TemperatureDof(elem, dof) = energy / specific_heat;
         fields.HeatFluxXDof(elem, dof) = qx;
         fields.HeatFluxYDof(elem, dof) = qy;

         const double basis_integral = integration.BasisIntegral(elem, dof);
         fields.TemperatureCell(elem) += fields.TemperatureDof(elem, dof) * basis_integral;
         fields.HeatFluxXCell(elem) += qx * basis_integral;
         fields.HeatFluxYCell(elem) += qy * basis_integral;
      }
   }

   return fields;
}

} // namespace callaway
