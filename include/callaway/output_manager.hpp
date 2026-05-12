#pragma once

#include "callaway/angular_quadrature.hpp"
#include "callaway/distribution.hpp"
#include "callaway/integration_cache.hpp"
#include "callaway/iteration_driver.hpp"
#include "callaway/mesh_adapter.hpp"
#include "callaway/nodal_basis.hpp"

#include <filesystem>
#include <vector>

namespace callaway
{

struct FieldSample
{
   double x = 0.0;
   double y = 0.0;
   double temperature = 0.0;
   double heat_flux_x = 0.0;
   double heat_flux_y = 0.0;
   double nxx = 0.0;
   double nxy = 0.0;
   double nyy = 0.0;
   int element = -1;
};

class OutputManager
{
public:
   static std::vector<double> BuildFortranSquareGrid(int sample_count);

   static std::vector<FieldSample> SampleConductionField(const IntegrationCache &integration,
                                                         const NodalBasis &basis,
                                                         const AngularQuadrature &quadrature,
                                                         const Distribution &distribution,
                                                         double specific_heat,
                                                         int sample_count,
                                                         const MacroState *macro_state = nullptr);

   static void WriteTecplotConduction(const std::filesystem::path &path,
                                      const IntegrationCache &integration,
                                      const NodalBasis &basis,
                                      const AngularQuadrature &quadrature,
                                      const Distribution &distribution,
                                      double specific_heat,
                                      int sample_count = 109,
                                      const MacroState *macro_state = nullptr);

   static std::vector<FieldSample> SampleSquareFourierReference(double specific_heat,
                                                                double tau_r,
                                                                int sample_count,
                                                                int terms = 200);

   static void WriteTecplotReference(const std::filesystem::path &path,
                                     double specific_heat,
                                     double tau_r,
                                     int sample_count = 109,
                                     int terms = 200);

   static void WriteCellAverageParaView(const std::filesystem::path &collection_path,
                                        MeshAdapter &mesh,
                                        const IntegrationCache &integration,
                                        const MomentFields &moments,
                                        int cycle = 0,
                                        double time = 0.0);

   static void WriteResidualHistory(const std::filesystem::path &path,
                                    const IterationResult &result);
};

} // namespace callaway
