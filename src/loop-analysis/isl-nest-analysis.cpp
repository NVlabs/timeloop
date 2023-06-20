#include "loop-analysis/isl-nest-analysis.hpp"

#include "loop-analysis/mapping-to-isl/mapping-to-isl.hpp"
#include "loop-analysis/spatial-analysis.hpp"
#include "loop-analysis/temporal-analysis.hpp"

namespace analysis
{

LogicalBufferStats::LogicalBufferStats(const LogicalBuffer& buf) : buf(buf)
{
}

ReuseAnalysisInput::ReuseAnalysisInput(
  const std::map<LogicalBuffer, Occupancy>& buf_to_occ
) : buf_to_occupancy(buf_to_occ)
{
}

ReuseAnalysisOutput ReuseAnalysis(ReuseAnalysisInput input)
{
  const auto& occupancies = input.buf_to_occupancy;

  auto spatial_reuse_models = SpatialReuseModels();
  spatial_reuse_models.EmplaceLinkTransferModel<SimpleLinkTransferModel>();
  spatial_reuse_models.EmplaceMulticastModel<SimpleMulticastModel>();

  auto output = ReuseAnalysisOutput();

  // Detect maximum arch level to find the compute level
  auto compute_level = 0;
  for (const auto& [buf, _] : occupancies)
  {
    compute_level = std::max(compute_level, buf.buffer_id);
  }

  for (const auto& [buf, occ] : occupancies)
  {
    auto stats = LogicalBufferStats(buf);

    bool can_exploit_temporal_reuse = buf.buffer_id != compute_level;
    auto temp_reuse_out = TemporalReuseAnalysis(
      TemporalReuseAnalysisInput(
        occ,
        BufTemporalReuseOpts{
          .exploit_temporal_reuse=can_exploit_temporal_reuse
        }
      )
    );

    auto spatial_reuse_out = SpatialReuseAnalysis(
      SpatialReuseAnalysisInput(buf,
                                temp_reuse_out.fill,
                                temp_reuse_out.effective_occupancy),
      spatial_reuse_models
    );

    stats.occupancy = occ;
    stats.effective_occupancy = temp_reuse_out.effective_occupancy;
    stats.fill = temp_reuse_out.fill;
    stats.link_transfer = spatial_reuse_out.link_transfer_info.link_transfer;
    stats.parent_reads = spatial_reuse_out.multicast_info.reads;
    stats.compat_access_stats =
      spatial_reuse_out.multicast_info.compat_access_stats;

    output.buf_to_stats.emplace(std::make_pair(buf, std::move(stats)));
  }

  return output;
}

}; // namespace analysis