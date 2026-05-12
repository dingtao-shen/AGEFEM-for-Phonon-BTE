#pragma once

#include <vector>

namespace callaway
{

class Distribution
{
public:
   Distribution() = default;
   Distribution(int angles, int elements, int dofs);

   int angles() const { return angles_; }
   int elements() const { return elements_; }
   int dofs() const { return dofs_; }

   double &operator()(int angle, int element, int dof);
   double operator()(int angle, int element, int dof) const;

   void Fill(double value);
   void SetThermalEquilibrium(double specific_heat, double temperature);

private:
   int angles_ = 0;
   int elements_ = 0;
   int dofs_ = 0;
   std::vector<double> values_;

   int Index(int angle, int element, int dof) const;
};

class MomentFields
{
public:
   MomentFields() = default;
   MomentFields(int elements, int dofs);

   int elements() const { return elements_; }
   int dofs() const { return dofs_; }

   void Clear();

   double &TemperatureCell(int element) { return temperature_cell_.at(element); }
   double &HeatFluxXCell(int element) { return heat_flux_x_cell_.at(element); }
   double &HeatFluxYCell(int element) { return heat_flux_y_cell_.at(element); }

   double TemperatureCell(int element) const { return temperature_cell_.at(element); }
   double HeatFluxXCell(int element) const { return heat_flux_x_cell_.at(element); }
   double HeatFluxYCell(int element) const { return heat_flux_y_cell_.at(element); }

   double &TemperatureDof(int element, int dof);
   double &HeatFluxXDof(int element, int dof);
   double &HeatFluxYDof(int element, int dof);

   double TemperatureDof(int element, int dof) const;
   double HeatFluxXDof(int element, int dof) const;
   double HeatFluxYDof(int element, int dof) const;

   double Mass() const;

private:
   int elements_ = 0;
   int dofs_ = 0;
   std::vector<double> temperature_cell_;
   std::vector<double> heat_flux_x_cell_;
   std::vector<double> heat_flux_y_cell_;
   std::vector<double> temperature_dof_;
   std::vector<double> heat_flux_x_dof_;
   std::vector<double> heat_flux_y_dof_;

   int DofIndex(int element, int dof) const;
};

} // namespace callaway
