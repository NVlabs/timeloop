#include "loop-analysis/temporal-analysis.hpp"

#include <boost/range/adaptor/indexed.hpp>
#include <boost/range/adaptor/reversed.hpp>

#include "isl-wrapper/isl-functions.hpp"

namespace analysis
{

/******************************************************************************
 * Local declarations
 *****************************************************************************/

std::pair<Occupancy, Fill> FillFromOccupancy(const Occupancy&);

/******************************************************************************
 * Global function implementations
 *****************************************************************************/

TemporalReuseAnalysisInput::TemporalReuseAnalysisInput(
  const Occupancy& occupancy,
  BufTemporalReuseOpts reuse_opts
) : occupancy(occupancy), reuse_opts(reuse_opts)
{
}

TemporalReuseAnalysisOutput
TemporalReuseAnalysis(TemporalReuseAnalysisInput input)
{
  auto exploit_temporal_reuse = input.reuse_opts.exploit_temporal_reuse;
  const auto& occupancy = input.occupancy;

  if (exploit_temporal_reuse){
    auto [eff_occ, fill] = FillFromOccupancy(occupancy);
    return TemporalReuseAnalysisOutput{
      .effective_occupancy = std::move(eff_occ),
      .fill = std::move(fill)
    };
  }
  else
  {
    return TemporalReuseAnalysisOutput{
      .effective_occupancy = occupancy,
      .fill = Fill(occupancy.dim_in_tags, occupancy.map)
    };
  }
}

/******************************************************************************
 * Local function implementations
 *****************************************************************************/

std::pair<Occupancy, Fill> FillFromOccupancy(const Occupancy& occupancy)
{
  using namespace boost::adaptors;

  auto p_occ = occupancy.map.copy();
  auto tags = occupancy.dim_in_tags;
  for (const auto& it: occupancy.dim_in_tags | indexed(0) | reversed)
  {
    auto dim_type = it.value();
    auto dim_idx = it.index();

    if (dim_type != spacetime::Dimension::Time)
    {
      continue;
    }

    auto p_proj_occ =
      isl_map_project_out(isl_map_copy(p_occ), isl_dim_in, dim_idx, 1);
    auto p_reinserted_occ = isl_map_intersect_domain(
      isl_map_insert_dims(isl_map_copy(p_proj_occ), isl_dim_in, dim_idx, 1),
      isl_map_domain(isl_map_copy(p_occ))
    );

    if (isl_map_plain_is_equal(p_occ, p_reinserted_occ)
        || isl_map_is_equal(p_occ, p_reinserted_occ))
    {
      isl_map_free(p_reinserted_occ);
      isl_map_free(p_occ);
      p_occ = p_proj_occ;
      tags.erase(tags.begin() + dim_idx);
      continue;
    }

    isl_map_free(p_proj_occ);
    isl_map_free(p_reinserted_occ);

    auto p_time_shift = isl::map_to_shifted(
      isl_space_domain(isl_map_get_space(p_occ)),
      dim_idx,
      -1
    );
    auto p_occ_before = isl_map_apply_range(p_time_shift, isl_map_copy(p_occ));
    auto p_fill = isl_map_subtract(isl_map_copy(p_occ), p_occ_before);

    return std::make_pair(
      Occupancy(tags, isl::manage(p_occ)),
      Fill(tags, isl::manage(p_fill))
    );
  }

  return std::make_pair(Occupancy(tags, isl::manage(isl_map_copy(p_occ))),
                        Fill(tags, isl::manage(p_occ)));
}


} // namespace analysis