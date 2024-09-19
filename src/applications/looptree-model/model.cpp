#include <fstream>

#include "util/accelergy_interface.hpp"
#include "util/banner.hpp"

#include "applications/looptree-model/model.hpp"
#include "loop-analysis/nest-analysis.hpp"
#include "loop-analysis/isl-analysis/isl-nest-analysis.hpp"
#include "loop-analysis/mapping-to-isl/fused-mapping-to-isl.hpp"
#include "loop-analysis/isl-ir.hpp"
#include "isl-wrapper/ctx-manager.hpp"
#include "isl-wrapper/isl-functions.hpp"
#include "mapping/fused-mapping.hpp"
#include "workload/fused-workload.hpp"
#include <isl/constraint.h>
#include <barvinok/isl.h>
#include "loop-analysis/temporal-analysis.hpp"
#include "loop-analysis/spatial-analysis.hpp"
#include "loop-analysis/isl-analysis/isl-nest-analysis.hpp"

//--------------------------------------------//
//                Application                 //
//--------------------------------------------//

namespace application
{

LooptreeModel::LooptreeModel(config::CompoundConfig* config,
                             std::string output_dir,
                             std::string name) :
    name_(name)
{    
  auto rootNode = config->getRoot();

  // Model application configuration.
  auto_bypass_on_failure_ = false;
  std::string semi_qualified_prefix = name;

  if (rootNode.exists("model"))
  {
    auto model = rootNode.lookup("model");
    model.lookupValue("verbose", verbose_);
    model.lookupValue("auto_bypass_on_failure", auto_bypass_on_failure_);
    model.lookupValue("out_prefix", semi_qualified_prefix);
  }

  out_prefix_ = output_dir + "/" + semi_qualified_prefix;

  if (verbose_)
  {
    for (auto& line: banner)
      std::cout << line << std::endl;
    std::cout << std::endl;
  }

  workload_ = problem::ParseFusedWorkload(rootNode.lookup("problem"));

  mapping_ = mapping::ParseMapping(rootNode.lookup("mapping"),
                                   workload_);
}

LooptreeModel::~LooptreeModel()
{
}

// Run the evaluation.
LooptreeModel::Result LooptreeModel::Run()
{
  Result model_result;
  analysis::MappingAnalysisResult mapping_analysis_result;
  mapping_analysis_result = analysis::OccupanciesFromMapping(mapping_,
                                                             workload_);

  auto reuse_analysis_options = analysis::ReuseAnalysisOptions();
  reuse_analysis_options.count_hops = false;

  const auto reuse_analysis_output = analysis::ReuseAnalysis(
    mapping_analysis_result.lbuf_to_occupancy,
    reuse_analysis_options
  );

  for (const auto& [buf, stats] : reuse_analysis_output.buf_to_stats)
  {
    const auto einsum_id =
      std::get<mapping::Compute>(mapping_.NodeAt(buf.branch_leaf_id)).kernel;

    auto p_occ_count = isl_map_card(stats.effective_occupancy.map.copy());
    auto key = std::tie(buf.buffer_id, buf.dspace_id, einsum_id);
    model_result.occupancy[key] = std::make_pair(
      stats.effective_occupancy.dim_in_tags,
      isl_pw_qpolynomial_to_str(p_occ_count)
    );
    isl_pw_qpolynomial_free(p_occ_count);

    auto p_fill_count = isl_map_card(stats.fill.map.copy());
    model_result.fills_by_parent[key] = std::make_pair(
      stats.fill.dim_in_tags,
      isl_pw_qpolynomial_to_str(p_fill_count)
    );
    isl_pw_qpolynomial_free(p_fill_count);

    auto p_parent_reads_count = isl_map_card(stats.parent_reads.map.copy());
    model_result.fills_by_parent[key] = std::make_pair(
      stats.parent_reads.dim_in_tags,
      isl_pw_qpolynomial_to_str(p_parent_reads_count)
    );
    isl_pw_qpolynomial_free(p_parent_reads_count);

    auto p_peer_fills_count = isl_map_card(stats.link_transfer.map.copy());
    p_peer_fills_count = isl_pw_qpolynomial_intersect_domain(
      p_peer_fills_count,
      stats.fill.map.domain().release()
    );
    model_result.fills_by_peer[key] = std::make_pair(
      stats.link_transfer.dim_in_tags,
      isl_pw_qpolynomial_to_str(p_peer_fills_count)
    );
    isl_pw_qpolynomial_free(p_peer_fills_count);
  }

  for (const auto& [lcomp, occupancy] : mapping_analysis_result.lcomp_to_occupancy)
  {
    const auto& dim_tags = occupancy.dim_in_tags;
    const auto& node =
      std::get<mapping::Compute>(mapping_.NodeAt(lcomp.branch_leaf_id));
    auto p_ops = isl_map_card(occupancy.map.copy());
    model_result.ops[node.kernel] = std::make_pair(
      dim_tags,
      isl_pw_qpolynomial_to_str(p_ops)
    );
    isl_pw_qpolynomial_free(p_ops);

    auto is_spatial_mask = std::vector<bool>(dim_tags.size());
    auto new_dim_tags = std::vector<analysis::SpaceTime>();
    for (unsigned long i = 0; i < is_spatial_mask.size(); ++i)
    {
      const auto& dim_tag = dim_tags.at(i);
      auto is_spatial = std::holds_alternative<analysis::Spatial>(dim_tag);
      is_spatial_mask.at(i) = is_spatial;
      if (!is_spatial)
      {
        new_dim_tags.emplace_back(dim_tag);
      }
    }
    const auto map_domain = occupancy.map.space().domain();
    const auto projector = isl::dim_projector(map_domain.copy(),
                                              is_spatial_mask);
    const auto non_spatial_map = isl_map_apply_range(projector,
                                                     occupancy.map.copy());
    const auto unbounded_identity = isl_map_identity(
      isl_space_map_from_set(
        isl_set_get_space(
          isl_map_domain(
            isl_map_copy(non_spatial_map)
            )
        )
      )
    );
    const auto bounded_identity = isl_map_intersect_domain(
      unbounded_identity,
      isl_map_domain(non_spatial_map)
    );
    const auto temporal_steps = isl_map_card(bounded_identity);
    model_result.temporal_steps[node.kernel] = std::make_pair(
      new_dim_tags,
      isl_pw_qpolynomial_to_str(temporal_steps)
    );
    isl_pw_qpolynomial_free(temporal_steps);
  }

  return model_result;
}

}