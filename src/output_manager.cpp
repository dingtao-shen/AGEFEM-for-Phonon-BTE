#include "callaway/output_manager.hpp"

#include <mfem.hpp>

#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace callaway
{
namespace
{

void EnsureParentDirectory(const std::filesystem::path &path)
{
   const std::filesystem::path parent = path.parent_path();
   if (!parent.empty())
   {
      std::filesystem::create_directories(parent);
   }
}

std::array<double, 2> MapToReference(const ElementGeometry &geometry, double x, double y)
{
   const double x1 = geometry.vertices[0][0];
   const double y1 = geometry.vertices[0][1];
   const double x2 = geometry.vertices[1][0];
   const double y2 = geometry.vertices[1][1];
   const double x3 = geometry.vertices[2][0];
   const double y3 = geometry.vertices[2][1];
   const double jacobian = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1);
   return {{
      ((y3 - y1) * (x - x1) + (x1 - x3) * (y - y1)) / jacobian,
      ((y1 - y2) * (x - x1) + (x2 - x1) * (y - y1)) / jacobian
   }};
}

int FindContainingElement(const IntegrationCache &integration, double x, double y)
{
   constexpr double tolerance = 1.0e-12;
   for (int element = 0; element < integration.element_count(); ++element)
   {
      const auto ref = MapToReference(integration.Geometry(element), x, y);
      if (ref[0] >= -tolerance && ref[1] >= -tolerance &&
          ref[0] + ref[1] <= 1.0 + tolerance)
      {
         return element;
      }
   }
   return -1;
}

} // namespace

std::vector<double> OutputManager::BuildFortranSquareGrid(int sample_count)
{
   if (sample_count < 11)
   {
      throw std::runtime_error("The Fortran-style square output grid requires at least 11 samples.");
   }

   std::vector<double> points(static_cast<std::size_t>(sample_count), 0.0);
   for (int j = 0; j < 5; ++j)
   {
      const double a = static_cast<double>(j) / 8.0;
      const double boundary_layer = a * a * a * (10.0 - 15.0 * a + 6.0 * a * a) * 0.02;
      points[static_cast<std::size_t>(j)] = boundary_layer;
      points[static_cast<std::size_t>(sample_count - 1 - j)] = 1.0 - boundary_layer;
   }

   const int interior_count = sample_count - 10;
   for (int j = 5; j < sample_count - 5; ++j)
   {
      points[static_cast<std::size_t>(j)] =
         (static_cast<double>(j - 4) * (1.0 - 0.02)) /
         static_cast<double>(interior_count + 1) + 0.01;
   }

   return points;
}

std::vector<FieldSample> OutputManager::SampleConductionField(const IntegrationCache &integration,
                                                              const NodalBasis &basis,
                                                              const AngularQuadrature &quadrature,
                                                              const Distribution &distribution,
                                                              double specific_heat,
                                                              int sample_count,
                                                              const MacroState *macro_state)
{
   if (specific_heat <= 0.0)
   {
      throw std::runtime_error("specific_heat must be positive for field output.");
   }
   if (distribution.angles() != quadrature.size() ||
       distribution.elements() != integration.element_count() ||
       distribution.dofs() != integration.dofs())
   {
      throw std::runtime_error("Distribution shape does not match field output inputs.");
   }
   if (macro_state &&
       (macro_state->elements() != integration.element_count() ||
        macro_state->dofs() != integration.dofs()))
   {
      throw std::runtime_error("Macro state shape does not match field output inputs.");
   }

   const std::vector<double> points = BuildFortranSquareGrid(sample_count);
   std::vector<FieldSample> samples;
   samples.reserve(static_cast<std::size_t>(sample_count * sample_count));

   for (int j = 0; j < sample_count; ++j)
   {
      for (int i = 0; i < sample_count; ++i)
      {
         FieldSample sample;
         sample.x = points[static_cast<std::size_t>(i)];
         sample.y = points[static_cast<std::size_t>(j)];
         sample.element = FindContainingElement(integration, sample.x, sample.y);
         if (sample.element < 0)
         {
            std::ostringstream os;
            os << "Output sample point (" << sample.x << ", " << sample.y
               << ") is outside the mesh.";
            throw std::runtime_error(os.str());
         }

         const auto ref = MapToReference(integration.Geometry(sample.element), sample.x, sample.y);
         const std::vector<double> basis_values = basis.EvaluateTriangleAll(ref[0], ref[1]);

         double energy = 0.0;
         double qx = 0.0;
         double qy = 0.0;
         for (int angle = 0; angle < quadrature.size(); ++angle)
         {
            const Direction &direction = quadrature[angle];
            for (int dof = 0; dof < integration.dofs(); ++dof)
            {
               const double value =
                  distribution(angle, sample.element, dof) *
                  basis_values[static_cast<std::size_t>(dof)];
               energy += value * direction.weight;
               qx += direction.cx * value * direction.weight;
               qy += direction.cy * value * direction.weight;
            }
         }

         sample.temperature = energy / specific_heat;
         sample.heat_flux_x = qx;
         sample.heat_flux_y = qy;
         if (macro_state)
         {
            double lxx = 0.0;
            double lxy = 0.0;
            double lyx = 0.0;
            double lyy = 0.0;
            for (int dof = 0; dof < integration.dofs(); ++dof)
            {
               const double shape = basis_values[static_cast<std::size_t>(dof)];
               lxx += (*macro_state)(MacroComponent::Lxx, sample.element, dof) * shape;
               lxy += (*macro_state)(MacroComponent::Lxy, sample.element, dof) * shape;
               lyx += (*macro_state)(MacroComponent::Lyx, sample.element, dof) * shape;
               lyy += (*macro_state)(MacroComponent::Lyy, sample.element, dof) * shape;
            }
            sample.nxx = -4.0 / 3.0 * lxx + 2.0 / 3.0 * lyy;
            sample.nxy = -lxy - lyx;
            sample.nyy = 2.0 / 3.0 * lxx - 4.0 / 3.0 * lyy;
         }
         samples.push_back(sample);
      }
   }

   return samples;
}

void OutputManager::WriteTecplotConduction(const std::filesystem::path &path,
                                           const IntegrationCache &integration,
                                           const NodalBasis &basis,
                                           const AngularQuadrature &quadrature,
                                           const Distribution &distribution,
                                           double specific_heat,
                                           int sample_count,
                                           const MacroState *macro_state)
{
   const std::vector<FieldSample> samples =
      SampleConductionField(integration,
                            basis,
                            quadrature,
                            distribution,
                            specific_heat,
                            sample_count,
                            macro_state);

   EnsureParentDirectory(path);
   std::ofstream out(path);
   if (!out)
   {
      throw std::runtime_error("Failed to open Tecplot output file: " + path.string());
   }

   out << "VARIABLES=\"x\",\"y\",\"T\",\"qx\",\"qy\",\"Nxx\",\"Nxy\",\"Nyy\"\n";
   out << "ZONE I = " << sample_count << " J = " << sample_count << "\n";
   out << std::scientific << std::setprecision(6);
   for (const FieldSample &sample : samples)
   {
      out << ' ' << std::setw(16) << sample.x
          << std::setw(16) << sample.y
          << std::setw(16) << sample.temperature
          << std::setw(16) << sample.heat_flux_x
          << std::setw(16) << sample.heat_flux_y
          << std::setw(16) << sample.nxx
          << std::setw(16) << sample.nxy
          << std::setw(16) << sample.nyy << '\n';
   }
}

std::vector<FieldSample> OutputManager::SampleSquareFourierReference(double specific_heat,
                                                                     double tau_r,
                                                                     int sample_count,
                                                                     int terms)
{
   if (specific_heat <= 0.0 || tau_r <= 0.0)
   {
      throw std::runtime_error("Reference solution requires positive specific_heat and tau_r.");
   }
   if (terms <= 0)
   {
      throw std::runtime_error("Reference solution requires a positive term count.");
   }

   const std::vector<double> points = BuildFortranSquareGrid(sample_count);
   std::vector<FieldSample> samples;
   samples.reserve(static_cast<std::size_t>(sample_count * sample_count));

   for (int j = 0; j < sample_count; ++j)
   {
      for (int i = 0; i < sample_count; ++i)
      {
         FieldSample sample;
         sample.x = points[static_cast<std::size_t>(i)];
         sample.y = points[static_cast<std::size_t>(j)];

         double temperature = 0.0;
         double qx = 0.0;
         double qy = 0.0;
         for (int m = 1; m <= terms; ++m)
         {
            const double mode = static_cast<double>(m);
            const double parity_factor = (m % 2 == 1) ? 2.0 : 0.0;
            if (parity_factor == 0.0) { continue; }

            const double denominator = std::sinh(mode * Pi);
            const double sin_x = std::sin(mode * Pi * sample.x);
            const double cos_x = std::cos(mode * Pi * sample.x);
            const double sinh_y = std::sinh(mode * Pi * sample.y);
            const double cosh_y = std::cosh(mode * Pi * sample.y);

            temperature += parity_factor / mode * sin_x * sinh_y / denominator;
            qx += parity_factor * cos_x * sinh_y / denominator;
            qy += parity_factor * sin_x * cosh_y / denominator;
         }

         sample.temperature = temperature * 2.0 / Pi;
         sample.heat_flux_x = -qx * specific_heat * tau_r / 3.0;
         sample.heat_flux_y = -qy * specific_heat * tau_r / 3.0;
         samples.push_back(sample);
      }
   }

   return samples;
}

void OutputManager::WriteTecplotReference(const std::filesystem::path &path,
                                          double specific_heat,
                                          double tau_r,
                                          int sample_count,
                                          int terms)
{
   const std::vector<FieldSample> samples =
      SampleSquareFourierReference(specific_heat, tau_r, sample_count, terms);

   EnsureParentDirectory(path);
   std::ofstream out(path);
   if (!out)
   {
      throw std::runtime_error("Failed to open reference output file: " + path.string());
   }

   out << "VARIABLES=\"x\",\"y\",\"T\",\"qx\",\"qy\"\n";
   out << "ZONE I = " << sample_count << " J = " << sample_count << "\n";
   out << std::scientific << std::setprecision(6);
   for (const FieldSample &sample : samples)
   {
      out << ' ' << std::setw(16) << sample.x
          << std::setw(16) << sample.y
          << std::setw(16) << sample.temperature
          << std::setw(16) << sample.heat_flux_x
          << std::setw(16) << sample.heat_flux_y << '\n';
   }
}

void OutputManager::WriteCellAverageParaView(const std::filesystem::path &collection_path,
                                             MeshAdapter &mesh,
                                             const IntegrationCache &integration,
                                             const MomentFields &moments,
                                             int cycle,
                                             double time)
{
   if (moments.elements() != integration.element_count())
   {
      throw std::runtime_error("Moment fields do not match the integration cache for ParaView output.");
   }

   const std::filesystem::path parent =
      collection_path.has_parent_path() ? collection_path.parent_path() : std::filesystem::path(".");
   std::filesystem::create_directories(parent);
   const std::string collection_name =
      collection_path.filename().empty() ? "callaway_output" : collection_path.filename().string();

   mfem::L2_FECollection collection(0, mesh.mesh().Dimension());
   mfem::FiniteElementSpace space(&mesh.mesh(), &collection);
   mfem::GridFunction temperature(&space);
   mfem::GridFunction heat_flux_x(&space);
   mfem::GridFunction heat_flux_y(&space);

   mfem::Array<int> dofs;
   for (int element = 0; element < integration.element_count(); ++element)
   {
      space.GetElementDofs(element, dofs);
      if (dofs.Size() != 1)
      {
         throw std::runtime_error("Expected one DG0 dof per element for ParaView output.");
      }

      const double area = integration.Geometry(element).area;
      temperature(dofs[0]) = moments.TemperatureCell(element) / area;
      heat_flux_x(dofs[0]) = moments.HeatFluxXCell(element) / area;
      heat_flux_y(dofs[0]) = moments.HeatFluxYCell(element) / area;
   }

   mfem::ParaViewDataCollection paraview(collection_name, &mesh.mesh());
   paraview.SetPrefixPath(parent.string());
   paraview.SetLevelsOfDetail(1);
   paraview.SetDataFormat(mfem::VTKFormat::BINARY);
   paraview.SetHighOrderOutput(false);
   paraview.SetCycle(cycle);
   paraview.SetTime(time);
   paraview.RegisterField("temperature", &temperature);
   paraview.RegisterField("heat_flux_x", &heat_flux_x);
   paraview.RegisterField("heat_flux_y", &heat_flux_y);
   paraview.Save();
}

void OutputManager::WriteResidualHistory(const std::filesystem::path &path,
                                         const IterationResult &result)
{
   EnsureParentDirectory(path);
   std::ofstream out(path);
   if (!out)
   {
      throw std::runtime_error("Failed to open residual history file: " + path.string());
   }

   const bool has_trace_history = !result.trace_iterations_history.empty();
   const bool has_mass_history = !result.mass_history.empty();
   if (has_mass_history && result.mass_history.size() != result.residual_history.size())
   {
      throw std::runtime_error("Mass history length does not match residual history.");
   }
   if (has_trace_history &&
       (result.trace_iterations_history.size() != result.residual_history.size() ||
        result.trace_converged_history.size() != result.residual_history.size() ||
        result.trace_initial_norm_history.size() != result.residual_history.size() ||
        result.trace_final_norm_history.size() != result.residual_history.size()))
   {
      throw std::runtime_error("Trace solver history length does not match residual history.");
   }

   if (has_trace_history)
   {
      out << "step,residual,mass,trace_iterations,trace_converged,trace_initial_norm,trace_final_norm\n";
   }
   else
   {
      out << "step,residual,mass\n";
   }
   out << std::scientific << std::setprecision(16);
   for (std::size_t i = 0; i < result.residual_history.size(); ++i)
   {
      out << (i + 1) << ',' << result.residual_history[i];
      out << ',' << (has_mass_history ? result.mass_history[i] : 0.0);
      if (has_trace_history)
      {
         out << ',' << result.trace_iterations_history[i]
             << ',' << result.trace_converged_history[i]
             << ',' << result.trace_initial_norm_history[i]
             << ',' << result.trace_final_norm_history[i];
      }
      out << '\n';
   }
}

} // namespace callaway
