#include "callaway/distribution.hpp"

#include "callaway/angular_quadrature.hpp"

#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace callaway
{

Distribution::Distribution(int angles, int elements, int dofs)
   : angles_(angles),
     elements_(elements),
     dofs_(dofs),
     values_(static_cast<std::size_t>(angles * elements * dofs), 0.0)
{
   if (angles_ <= 0 || elements_ <= 0 || dofs_ <= 0)
   {
      throw std::runtime_error("Distribution dimensions must be positive.");
   }
}

double &Distribution::operator()(int angle, int element, int dof)
{
   return values_.at(static_cast<std::size_t>(Index(angle, element, dof)));
}

double Distribution::operator()(int angle, int element, int dof) const
{
   return values_.at(static_cast<std::size_t>(Index(angle, element, dof)));
}

void Distribution::Fill(double value)
{
   std::fill(values_.begin(), values_.end(), value);
}

void Distribution::SetThermalEquilibrium(double specific_heat, double temperature)
{
   Fill(specific_heat * temperature / (4.0 * Pi));
}

int Distribution::Index(int angle, int element, int dof) const
{
   return (angle * elements_ + element) * dofs_ + dof;
}

MomentFields::MomentFields(int elements, int dofs)
   : elements_(elements),
     dofs_(dofs),
     temperature_cell_(static_cast<std::size_t>(elements), 0.0),
     heat_flux_x_cell_(static_cast<std::size_t>(elements), 0.0),
     heat_flux_y_cell_(static_cast<std::size_t>(elements), 0.0),
     temperature_dof_(static_cast<std::size_t>(elements * dofs), 0.0),
     heat_flux_x_dof_(static_cast<std::size_t>(elements * dofs), 0.0),
     heat_flux_y_dof_(static_cast<std::size_t>(elements * dofs), 0.0)
{
   if (elements_ <= 0 || dofs_ <= 0)
   {
      throw std::runtime_error("MomentFields dimensions must be positive.");
   }
}

void MomentFields::Clear()
{
   std::fill(temperature_cell_.begin(), temperature_cell_.end(), 0.0);
   std::fill(heat_flux_x_cell_.begin(), heat_flux_x_cell_.end(), 0.0);
   std::fill(heat_flux_y_cell_.begin(), heat_flux_y_cell_.end(), 0.0);
   std::fill(temperature_dof_.begin(), temperature_dof_.end(), 0.0);
   std::fill(heat_flux_x_dof_.begin(), heat_flux_x_dof_.end(), 0.0);
   std::fill(heat_flux_y_dof_.begin(), heat_flux_y_dof_.end(), 0.0);
}

double &MomentFields::TemperatureDof(int element, int dof)
{
   return temperature_dof_.at(static_cast<std::size_t>(DofIndex(element, dof)));
}

double &MomentFields::HeatFluxXDof(int element, int dof)
{
   return heat_flux_x_dof_.at(static_cast<std::size_t>(DofIndex(element, dof)));
}

double &MomentFields::HeatFluxYDof(int element, int dof)
{
   return heat_flux_y_dof_.at(static_cast<std::size_t>(DofIndex(element, dof)));
}

double MomentFields::TemperatureDof(int element, int dof) const
{
   return temperature_dof_.at(static_cast<std::size_t>(DofIndex(element, dof)));
}

double MomentFields::HeatFluxXDof(int element, int dof) const
{
   return heat_flux_x_dof_.at(static_cast<std::size_t>(DofIndex(element, dof)));
}

double MomentFields::HeatFluxYDof(int element, int dof) const
{
   return heat_flux_y_dof_.at(static_cast<std::size_t>(DofIndex(element, dof)));
}

double MomentFields::Mass() const
{
   return std::accumulate(temperature_cell_.begin(), temperature_cell_.end(), 0.0);
}

int MomentFields::DofIndex(int element, int dof) const
{
   return element * dofs_ + dof;
}

} // namespace callaway
