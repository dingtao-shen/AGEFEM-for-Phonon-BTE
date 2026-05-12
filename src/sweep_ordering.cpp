#include "callaway/sweep_ordering.hpp"

#include <array>
#include <cmath>
#include <sstream>
#include <stdexcept>

namespace callaway
{

SweepOrdering::SweepOrdering(const MeshAdapter &mesh,
                             const IntegrationCache &integration,
                             const AngularQuadrature &quadrature)
   : angles_(quadrature.size()),
     elements_(integration.element_count()),
     orders_(static_cast<std::size_t>(quadrature.size())),
     positions_(static_cast<std::size_t>(quadrature.size() * integration.element_count()), -1)
{
   if (mesh.mesh().GetNE() != elements_)
   {
      throw std::runtime_error("Mesh and integration cache element counts differ.");
   }

   for (int angle = 0; angle < angles_; ++angle)
   {
      std::vector<std::array<int, 3>> io_flag(static_cast<std::size_t>(elements_));
      for (auto &flags : io_flag) { flags = {{0, 0, 0}}; }

      const Direction &direction = quadrature[angle];
      for (int elem = 0; elem < elements_; ++elem)
      {
         for (int local = 0; local < 3; ++local)
         {
            if (mesh.ElementNeighbor(elem, local) < 0) { continue; }
            const auto normal = integration.OutwardNormal(elem, local);
            const double speed = direction.cx * normal[0] + direction.cy * normal[1];
            if (speed > 0.0) { io_flag[static_cast<std::size_t>(elem)][static_cast<std::size_t>(local)] = 1; }
            if (speed < 0.0) { io_flag[static_cast<std::size_t>(elem)][static_cast<std::size_t>(local)] = -1; }
         }
      }

      std::vector<char> in_array(static_cast<std::size_t>(elements_), 1);
      std::vector<int> order;
      order.reserve(static_cast<std::size_t>(elements_));

      while (static_cast<int>(order.size()) < elements_)
      {
         bool progressed = false;
         for (int elem = 0; elem < elements_; ++elem)
         {
            if (!in_array[static_cast<std::size_t>(elem)]) { continue; }

            bool outgoing_only = true;
            for (int local = 0; local < 3; ++local)
            {
               if (io_flag[static_cast<std::size_t>(elem)][static_cast<std::size_t>(local)] == -1)
               {
                  outgoing_only = false;
                  break;
               }
            }
            if (!outgoing_only) { continue; }

            const int position = static_cast<int>(order.size());
            order.push_back(elem);
            positions_[static_cast<std::size_t>(angle * elements_ + elem)] = position;
            in_array[static_cast<std::size_t>(elem)] = 0;
            progressed = true;

            for (int local = 0; local < 3; ++local)
            {
               const int neighbor = mesh.ElementNeighbor(elem, local);
               if (neighbor < 0) { continue; }
               for (int neighbor_local = 0; neighbor_local < 3; ++neighbor_local)
               {
                  if (mesh.ElementNeighbor(neighbor, neighbor_local) == elem)
                  {
                     io_flag[static_cast<std::size_t>(neighbor)][static_cast<std::size_t>(neighbor_local)] = 0;
                     break;
                  }
               }
            }
         }

         if (!progressed)
         {
            std::ostringstream os;
            os << "Failed to build acyclic sweep ordering for angle " << angle
               << ". The mesh/direction graph contains a cycle or unresolved zero-flux dependency.";
            throw std::runtime_error(os.str());
         }
      }

      orders_[static_cast<std::size_t>(angle)] = std::move(order);
   }
}

int SweepOrdering::Position(int angle, int element) const
{
   return positions_.at(static_cast<std::size_t>(angle * elements_ + element));
}

} // namespace callaway
