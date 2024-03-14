#include <fstream>

#include "util/accelergy_interface.hpp"
#include "util/banner.hpp"

#include "applications/looptree-model/model.hpp"
#include "loop-analysis/nest-analysis.hpp"
#include "loop-analysis/isl-analysis/capacity-analysis.hpp"
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

//--------------------------------------------//
//                Application                 //
//--------------------------------------------//

std::vector<analysis::SpaceTime>
ParseSpaceTime(const config::CompoundConfigNode& node)
{
  std::vector<std::string> spacetime_strings;
  node.getArrayValue(spacetime_strings);

  std::vector<analysis::SpaceTime> spacetimes;
  for (const std::string& spacetime_string : spacetime_strings)
  {
    analysis::SpaceTime spacetime;
    if (spacetime_string == "temporal")
    {
      spacetime = analysis::Temporal();
    }
    else if (spacetime_string == "spatial")
    {
      spacetime = analysis::Spatial(0);
    }
    else if (spacetime_string == "sequential")
    {
      spacetime = analysis::Sequential();
    }
    else if (spacetime_string == "pipeline")
    {
      spacetime = analysis::PipelineSpatial();
    }
    spacetimes.push_back(spacetime);
  }

  return spacetimes;
}

void ParseIslOccupancy(const config::CompoundConfigNode& node,
                       const problem::FusedWorkload& workload)
{
  const auto& dspace_name_to_id = workload.DataSpaceNameToId();
  for (int i = 0; i < node.getLength(); ++i)
  {
    std::string dspace;
    std::string target;
    std::string occupancy_string;

    const config::CompoundConfigNode& cur_node = node[i];
    cur_node.lookupValue("dspace", dspace);
    cur_node.lookupValue("target", target);

    cur_node.lookupValue("occupancy", occupancy_string);
    occupancy_string = "{ " + occupancy_string + " }";

    auto occ = analysis::Occupancy(
      ParseSpaceTime(cur_node.lookup("spacetime")),
      isl::map(GetIslCtx(), occupancy_string).intersect_range(
        workload.DataSpaceBound(dspace_name_to_id.at(dspace))
      )
    );

    analysis::TemporalReuseAnalysisOutput result =
      analysis::TemporalReuseAnalysis(
        analysis::TemporalReuseAnalysisInput(
          occ,
          analysis::BufTemporalReuseOpts{
            .exploit_temporal_reuse=1
          }
        )
      );

    auto p_fill = result.fill.map.copy();
    auto p_fill_count = isl_pw_qpolynomial_sum(isl_map_card(p_fill));
    double fill_count = isl::val_to_double(isl::get_val_from_singular(
      isl_pw_qpolynomial_copy(p_fill_count)
    ));
    std::cout << "[Fill] (" << dspace << "," << target << ") : " << fill_count
              << std::endl;

    /* TODO: Fills contribute to two accesses:
     *  1. Reads from parent or peer buffer
     *  2. Writes to current buffer
     */

    auto p_occ = result.effective_occupancy.map.copy();
    isl_bool tight;
    auto p_occ_size = isl_map_card(p_occ);
    auto p_occ_count = isl_pw_qpolynomial_bound(
      isl_pw_qpolynomial_copy(p_occ_size),
      isl_fold_max,
      &tight
    );
    if (tight != isl_bool_true)
    {
      double max_sample = 0;
      auto p_domain =
        isl_pw_qpolynomial_domain(isl_pw_qpolynomial_copy(p_occ_size));
      for (int i = 0; i < 8; ++i)
      {
        auto sample = isl::val_to_double(
          isl_pw_qpolynomial_eval(
            isl_pw_qpolynomial_copy(p_occ_size),
            isl_set_sample_point(isl_set_copy(p_domain))
          )
        );
        max_sample = std::max(max_sample, sample);
      }
      std::cout << "[Occupancy] (" << dspace << "," << target << "," << tight
                << ") : " << max_sample << std::endl;
      isl_set_free(p_domain);
    }
    else
    {
      double occ_count = isl::val_to_double(isl::get_val_from_singular(
        isl_pw_qpolynomial_fold_copy(p_occ_count)
      ));
      std::cout << "[Occupancy] (" << dspace << "," << target << "," << tight
                << ") : " << occ_count << std::endl;
    }
    isl_pw_qpolynomial_fold_free(p_occ_count);
    isl_pw_qpolynomial_free(p_occ_size);
  }
}

template <class Archive>
void Application::serialize(Archive& ar, const unsigned int version)
{
  if (version == 0)
  {
    ar& BOOST_SERIALIZATION_NVP(workload_);
  }
}

Application::Application(config::CompoundConfig* config,
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

  auto workload = problem::ParseFusedWorkload(rootNode.lookup("problem"));

  // Architecture configuration.
  config::CompoundConfigNode arch;
  if (rootNode.exists("arch"))
  {
    arch = rootNode.lookup("arch");
  }
  else if (rootNode.exists("architecture"))
  {
    arch = rootNode.lookup("architecture");
  }
  
  bool is_sparse_topology = rootNode.exists("sparse_optimizations");
  arch_specs_ = model::Engine::ParseSpecs(arch, is_sparse_topology);

  if (rootNode.exists("ERT"))
  {
    auto ert = rootNode.lookup("ERT");
    if (verbose_)
      std::cout << "Found Accelergy ERT (energy reference table), replacing internal energy model." << std::endl;
    arch_specs_.topology.ParseAccelergyERT(ert);
    if (rootNode.exists("ART")){ // Nellie: well, if the users have the version of Accelergy that generates ART
      auto art = rootNode.lookup("ART");
      if (verbose_)
        std::cout << "Found Accelergy ART (area reference table), replacing internal area model." << std::endl;
      arch_specs_.topology.ParseAccelergyART(art);  
    }
  }
  else
  {
#ifdef USE_ACCELERGY
    // Call accelergy ERT with all input files
    if (arch.exists("subtree") || arch.exists("local"))
    {
      accelergy::invokeAccelergy(config->inFiles, semi_qualified_prefix, output_dir);
      std::string ertPath = out_prefix_ + ".ERT.yaml";
      auto ertConfig = new config::CompoundConfig(ertPath.c_str());
      auto ert = ertConfig->getRoot().lookup("ERT");
      if (verbose_)
        std::cout << "Generate Accelergy ERT (energy reference table) to replace internal energy model." << std::endl;
      arch_specs_.topology.ParseAccelergyERT(ert);
        
      std::string artPath = out_prefix_ + ".ART.yaml";
      auto artConfig = new config::CompoundConfig(artPath.c_str());
      auto art = artConfig->getRoot().lookup("ART");
      if (verbose_)
        std::cout << "Generate Accelergy ART (area reference table) to replace internal area model." << std::endl;
      arch_specs_.topology.ParseAccelergyART(art);
    }
#endif
  }

  // Sparse optimizations
  config::CompoundConfigNode sparse_optimizations;
  if (is_sparse_topology)
    sparse_optimizations = rootNode.lookup("sparse_optimizations");
      sparse_optimizations_ = new sparse::SparseOptimizationInfo(sparse::ParseAndConstruct(sparse_optimizations, arch_specs_));
  // characterize workload on whether it has metadata
  workload_.SetDefaultDenseTensorFlag(sparse_optimizations_->compression_info.all_ranks_default_dense);
  
  if (verbose_)
    std::cout << "Sparse optimization configuration complete." << std::endl;

  if (verbose_)
    std::cout << "Architecture configuration complete." << std::endl;

<<<<<<< Updated upstream
  mapping::FusedMapping mapping = mapping::ParseMapping(
    rootNode.lookup("mapping"),
    workload,
    arch_specs_.topology
  );
=======
  // TODO: this is an issue because later analysis (latency and capacity)
  // requires information from mapping that is not captured here.
  analysis::MappingAnalysisResult mapping_analysis_result;
  if (rootNode.exists("occupancy"))
  {
    ParseIslOccupancy(rootNode.lookup("occupancy"), workload);
  }
  else
  {
    mapping::FusedMapping mapping =
      mapping::ParseMapping(rootNode.lookup("mapping"), workload);
>>>>>>> Stashed changes

    mapping_analysis_result = analysis::OccupanciesFromMapping(mapping,
                                                               workload);
  }

  for (const auto& [buf, occ] : mapping_analysis_result.lbuf_to_occupancy)
  {
    auto result = analysis::TemporalReuseAnalysis(
      analysis::TemporalReuseAnalysisInput(
        occ,
        analysis::BufTemporalReuseOpts{
          .exploit_temporal_reuse=1
        }
      )
    );

    auto p_fill = result.fill.map.copy();
    auto p_fill_count = isl_pw_qpolynomial_sum(isl_map_card(p_fill));

    auto p_occ = result.effective_occupancy.map.copy();
    isl_bool tight;
    auto p_occ_count = isl_pw_qpolynomial_bound(
      isl_map_card(p_occ),
      isl_fold_max,
      &tight
    );
    assert(tight == isl_bool_true);

    std::cout << "[Occupancy]" << buf << ": "
      << isl::pw_qpolynomial_fold_to_str(p_occ_count) << std::endl;
    std::cout << "[Fill]" << buf << ": "
      << isl_pw_qpolynomial_to_str(p_fill_count) << std::endl;
    isl_pw_qpolynomial_free(p_fill_count);
  }

  for (const auto& [compute, tiling] : mapping_analysis_result.branch_tiling)
  {
    auto p_ops = isl_map_card(tiling.copy());
    std::cout << "[Operations]" << compute << ": "
      << isl_pw_qpolynomial_to_str(p_ops) << std::endl;
    isl_pw_qpolynomial_free(p_ops);
  }

  // auto latency =
  //   mapping_analysis_result.compute_latency_aggregator.CalculateLatency(
  //     mapping_analysis_result.lcomp_to_occupancy,
  //     mapping_analysis_result.compute_to_assumed_parallelism
  //   );
  // std::cout << "[Latency]: " << latency << std::endl;

  // auto capacities = ComputeCapacityFromMapping(
  //   mapping,
  //   mapping_analysis_result.lbuf_to_occupancy,
  //   workload
  // );

  // for (const auto& [buf_id, cap] : capacities)
  // {
  //   std::cout << "[Capacity]" << buf_id << ": "
  //             << isl_pw_qpolynomial_to_str(cap) << std::endl;
  // }

  // // for (const auto& [buf, fill] : fills)
  // // {
  // //   std::cout << buf << std::endl;
  // //   auto p_fill_count = isl_map_card(fill.map.copy());
  // //   std::cout << isl_pw_qpolynomial_to_str(p_fill_count) << std::endl;
  // //   isl_pw_qpolynomial_free(p_fill_count);
  // // }

  // for (const auto& [buf, occ] : occupancies)
  // {
  //   std::cout << buf << std::endl;
  //   auto p_occ_count = isl_map_card(occ.map.copy());
  //   std::cout << isl_pw_qpolynomial_to_str(p_occ_count) << std::endl;
  //   isl_pw_qpolynomial_free(p_occ_count);
  // }
}

Application::~Application()
{
  if (constraints_)
    delete constraints_;

  if (sparse_optimizations_)
    delete sparse_optimizations_;
}

// Run the evaluation.
Application::Stats Application::Run()
{
  // Output file names.
  model::Engine engine;
  engine.Spec(arch_specs_);

  auto level_names = arch_specs_.topology.LevelNames();

  // if (engine.IsEvaluated())
  // {
  //   if (!sparse_optimizations_->no_optimization_applied)
  //   {   
  //     std::cout << "Utilization = " << std::setw(4) << OUT_FLOAT_FORMAT << std::setprecision(2) << engine.Utilization()
  //             << " | pJ/Algorithmic-Compute = " << std::setw(8) << OUT_FLOAT_FORMAT << PRINTFLOAT_PRECISION << engine.Energy() /
  //     engine.GetTopology().AlgorithmicComputes()
  //             << " | pJ/Compute = " << std::setw(8) << OUT_FLOAT_FORMAT << PRINTFLOAT_PRECISION << engine.Energy() /
  //     engine.GetTopology().ActualComputes() << std::endl;
  //   }
  //   else
  //   {
  //     std::cout << "Utilization = " << std::setw(4) << OUT_FLOAT_FORMAT << std::setprecision(2) << engine.Utilization()
  //                << " | pJ/Compute = " << std::setw(8) << OUT_FLOAT_FORMAT << PRINTFLOAT_PRECISION << engine.Energy() /
  //     engine.GetTopology().ActualComputes() << std::endl;
  //   }
  //   std::ofstream map_txt_file(map_txt_file_name);
  //   mapping.PrettyPrint(map_txt_file, arch_specs_.topology.StorageLevelNames(), engine.GetTopology().UtilizedCapacities(), engine.GetTopology().TileSizes());
  //   map_txt_file.close();

  //   std::ofstream stats_file(stats_file_name);
  //   stats_file << engine << std::endl;
  //   stats_file.close();
  // }

  Stats stats;
  // stats.cycles = engine.Cycles();
  // stats.energy = engine.Energy();
  return stats;
}