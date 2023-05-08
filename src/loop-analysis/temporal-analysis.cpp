#include "loop-analysis/temporal-analysis.hpp"

namespace analysis
{

/******************************************************************************
 * Local declarations
 *****************************************************************************/

std::pair<Occupancy, Fill> FillFromOccupancy(Occupancy);

/******************************************************************************
 * Global function implementations
 *****************************************************************************/

std::pair<LogicalBufOccupancies, LogicalBufFills>
TemporalReuseAnalysis(const LogicalBufOccupancies& occupancies)
{
  LogicalBufFills fills;
  LogicalBufOccupancies effectual_occupancies;

  for (auto& [buf, occupancy] : occupancies)
  {
    auto [eff_occupancy, fill] = FillFromOccupancy(occupancy);
    fills.emplace(std::make_pair(buf, fill));
    effectual_occupancies.emplace(std::make_pair(buf, eff_occupancy));
  }

  return std::make_pair(effectual_occupancies, fills);
}

/******************************************************************************
 * Local function implementations
 *****************************************************************************/

std::pair<Occupancy, Fill> FillFromOccupancy(Occupancy occupancy)
{
  /**
   * Compute fill by iteratively going through temporal loops and
   * computing delta, only stopping when delta is non-zero at time 1.
   */

  bool try_again = true;
  while (try_again)
  {
    try_again = false;
    for (auto dim_it = occupancy.in_rbegin(); dim_it != occupancy.in_rend();
        ++dim_it)
    {
      auto [dim_idx, dim_type] = *dim_it;
      if (dim_type == spacetime::Dimension::Time)
      {
        auto time_shift_map = occupancy.tag_like_this(
          isl::map_to_shifted(occupancy.space().domain(), dim_idx, -1)
        );
        auto occ_before = time_shift_map.apply_range(occupancy.map);
        auto fill = occupancy.subtract(occ_before.map);
        auto first_fill = isl::fix_si(fill.map, isl_dim_in, dim_idx, 1);
        if (first_fill.range().is_empty())
        {
          occupancy.project_dim_in(dim_idx, 1);
          try_again = true;
          break;
        }
        else
        {
          return std::make_pair(occupancy, fill);
        }
      }
    }
  }

  return std::make_pair(occupancy, occupancy);
}


} // namespace analysis