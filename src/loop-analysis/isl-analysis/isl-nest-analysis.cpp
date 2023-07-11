#include "loop-analysis/isl-analysis/isl-nest-analysis.hpp"

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

    auto reuse_analysis_input =
      SpatialReuseAnalysisInput(buf, temp_reuse_out.fill);
    reuse_analysis_input.fill_providers.emplace_back(
      FillProvider::MakeFillProvider<SimpleLinkTransferModel>(
        buf,
        temp_reuse_out.effective_occupancy
      )
    );
    // TODO: this should be the occupancy of the actual parent but we have no
    // way of connecting this buffer to its parent yet
    reuse_analysis_input.fill_providers.emplace_back(
      FillProvider::MakeFillProvider<SimpleMulticastModel>(buf, occ)
    );

    auto spatial_reuse_out = SpatialReuseAnalysis(reuse_analysis_input);

    stats.occupancy = occ;
    stats.effective_occupancy = temp_reuse_out.effective_occupancy;
    stats.fill = temp_reuse_out.fill;

    for (const auto& transfer_info : spatial_reuse_out.transfer_infos)
    {
      if (transfer_info.is_link_transfer)
      {
        stats.link_transfer = transfer_info.fulfilled_fill;
      }
      else if (transfer_info.is_multicast)
      {
        stats.compat_access_stats = transfer_info.compat_access_stats;
        stats.parent_reads = transfer_info.parent_reads;
      }
    }

    output.buf_to_stats.emplace(std::make_pair(buf, std::move(stats)));
  }

  return output;
}

}; // namespace analysis