#include <fstream>

#include "util/accelergy_interface.hpp"
#include "util/banner.hpp"

#include "applications/looptree/model.hpp"
#include "loop-analysis/nest-analysis.hpp"
#include "loop-analysis/isl-ir.hpp"
#include "isl-wrapper/ctx-manager.hpp"
#include "mapping/fused-mapping.hpp"
#include "workload/fused-workload.hpp"
#include <isl/constraint.h>

/**************
 * Prototype
 */

namespace analysis
{

isl::map ConstraintDimEquals(isl::map map, size_t n_dims)
{
  auto p_map = map.release();
  auto p_space = isl_map_get_space(p_map);
  auto p_ls = isl_local_space_from_space(p_space);

  isl_constraint* p_c;
  for (size_t i = 0; i < n_dims; ++i)
  {
    p_c = isl_constraint_alloc_equality(isl_local_space_copy(p_ls));
    p_c = isl_constraint_set_coefficient_si(p_c, isl_dim_in, i, 1);
    p_c = isl_constraint_set_coefficient_si(p_c, isl_dim_out, i, -1);
    p_map = isl_map_add_constraint(p_map, p_c);
  }
  isl_local_space_free(p_ls);

  return isl::manage(p_map);
}

isl::map MapToPriorData(size_t n_in_dims, size_t top)
{
  isl_space* p_space;
  isl_basic_map* p_tmp_map;
  isl_map* p_map;
  isl_local_space* p_ls;
  isl_constraint* p_c;

  // Goal: { [i0, ..., i{n_in_dims-1}] -> [i0, ..., i{top}-1, o{top+1}, ..., o{n_in_dims}] }
  p_space = isl_space_alloc(GetIslCtx().get(), 0, n_in_dims, n_in_dims);
  p_map = isl_map_universe(isl_space_copy(p_space));
  p_ls = isl_local_space_from_space(p_space);

  if (top > 0)
  {
    p_tmp_map = isl_basic_map_universe(isl_space_copy(p_space));
    for (size_t i = 0; i < top-1; ++i)
    {
      p_c = isl_constraint_alloc_equality(isl_local_space_copy(p_ls));
      p_c = isl_constraint_set_coefficient_si(p_c, isl_dim_out, i, 1);
      p_c = isl_constraint_set_coefficient_si(p_c, isl_dim_in, i, -1);
      p_tmp_map = isl_basic_map_add_constraint(p_tmp_map, p_c);
    }

    p_c = isl_constraint_alloc_equality(isl_local_space_copy(p_ls));
    p_c = isl_constraint_set_coefficient_si(p_c, isl_dim_out, top-1, 1);
    p_c = isl_constraint_set_coefficient_si(p_c, isl_dim_in, top-1, -1);
    p_c = isl_constraint_set_constant_si(p_c, 1);
    p_tmp_map = isl_basic_map_add_constraint(p_tmp_map, p_c);

    p_map = isl_map_intersect(p_map, isl_map_from_basic_map(p_tmp_map));
  }

  if (top > 0 && top < n_in_dims)
  {
    p_tmp_map = isl_basic_map_universe(isl_space_copy(p_space));
    for (size_t i = 0; i < top; ++i)
    {
      p_c = isl_constraint_alloc_equality(isl_local_space_copy(p_ls));
      p_c = isl_constraint_set_coefficient_si(p_c, isl_dim_out, i, 1);
      p_c = isl_constraint_set_coefficient_si(p_c, isl_dim_in, i, -1);
      p_tmp_map = isl_basic_map_add_constraint(p_tmp_map, p_c);
    }

    for (auto i = top; i < n_in_dims; ++i)
    {
      p_c = isl_constraint_alloc_inequality(isl_local_space_copy(p_ls));
      p_c = isl_constraint_set_coefficient_si(p_c, isl_dim_out, i, -1);
      p_c = isl_constraint_set_coefficient_si(p_c, isl_dim_in, i, 1);
      p_c = isl_constraint_set_constant_si(p_c, -1);
      p_tmp_map = isl_basic_map_add_constraint(p_tmp_map, p_c);
    }
    p_map = isl_map_union(p_map, isl_map_from_basic_map(p_tmp_map));
  }

  isl_local_space_free(p_ls);

  return isl::manage(p_map);
}

BranchTilings LoopBoundsInference(BranchTilings tilings,
                                  const problem::FusedWorkload& workload,
                                  size_t pipeline_tiling_idx,
                                  const std::map<DataSpaceID, size_t>& dspace_top_idx)
{
  BranchTilings inferred_tilings(tilings.begin(), tilings.end());

  for (size_t i = 0; i < inferred_tilings.size(); ++i)
  {
    for (auto& [einsum_id, tiling] : inferred_tilings)
    {
      bool complete = true;
      auto domain = tiling.domain();
      for (auto i = 0; i < isl_set_dim(domain.get(), isl_dim_set); ++i)
      {
        complete = complete &&
                   isl_set_dim_has_lower_bound(domain.get(), isl_dim_set, i);
      }
      if (!complete)
      {
        continue;
      }

      for (const auto& read_tensor : workload.TensorsReadByEinsum(einsum_id))
      {
        auto producer_einsum_opt = workload.WriterEinsum(read_tensor);
        if (!producer_einsum_opt)
        {
          // Not an intermediate tensor
          continue;
        }

        // Decide how much the consumer (einsum_id) needs
        auto pruned_tiling = project_dim_in_after(tiling, pipeline_tiling_idx);
        auto required_data = pruned_tiling.apply_range(
          workload.ReadAccesses(einsum_id, read_tensor));

        auto top_idx = dspace_top_idx.at(read_tensor);
        auto shifter = 
          MapToPriorData(pipeline_tiling_idx, top_idx);
        auto buffered_data = shifter.apply_range(required_data);

        auto computed_data = required_data.subtract(buffered_data).coalesce();

        auto producer_write_dep =
          workload.WriteAccesses(*producer_einsum_opt, read_tensor);
        auto required_ops =
          computed_data.apply_range(producer_write_dep.reverse());
        auto producer_tiling = inferred_tilings.at(*producer_einsum_opt);
        auto required_iters = ConstraintDimEquals(
          required_ops.apply_range(producer_tiling.reverse()),
          pipeline_tiling_idx
        );

        auto inferred_prod_tiling =
          producer_tiling.intersect_domain(required_iters.range()).coalesce();
        std::cout << inferred_prod_tiling << std::endl;
        inferred_tilings.at(*producer_einsum_opt) = inferred_prod_tiling;
      }
    }
  }
  return inferred_tilings;
}
};
//--------------------------------------------//
//                Application                 //
//--------------------------------------------//

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

  // Problem configuration.
  // auto problem = rootNode.lookup("problem");
  // problem::ParseWorkload(problem, workload_);
  // if (verbose_)
  //   std::cout << "Problem configuration complete." << std::endl;

  auto workload = problem::ParseFusedWorkload(rootNode.lookup("problem"));
  // auto pwise1_id = workload.NewEinsum();
  // auto dwise1_id = workload.NewEinsum();
  // auto pwise2_id = workload.NewEinsum();
  // auto O1_id = workload.NewDataSpace();
  // auto I_id = workload.NewDataSpace();
  // auto F1_id = workload.NewDataSpace();
  // auto O2_id = workload.NewDataSpace();
  // auto F2_id = workload.NewDataSpace();
  // auto O3_id = workload.NewDataSpace();
  // auto F3_id = workload.NewDataSpace();

  // workload.SetEinsumProjection(pwise1_id, O1_id, true,
  // "{ pwise1[m1,c1,p1,q1] -> O1[m1,p1,q1] : 0 <= m1 < 32 and 0 <= p1 < 112 and 0 <= q1 < 112 and 0 <= c1 < 32 and 0 <= p1 < 112 and 0 <= q1 < 112 and 0 <= m1 < 32 and 0 <= c1 < 32 }"
  // );
  // workload.SetEinsumProjection(pwise1_id, I_id, false,
  // "{ pwise1[m1,c1,p1,q1] -> I[c1,p1,q1] : 0 <= m1 < 32 and 0 <= p1 < 112 and 0 <= q1 < 112 and 0 <= c1 < 32 and 0 <= p1 < 112 and 0 <= q1 < 112 and 0 <= m1 < 32 and 0 <= c1 < 32 }"
  // );
  // workload.SetEinsumProjection(pwise1_id, F1_id, false,
  // "{ pwise1[m1,c1,p1,q1] -> F1[m1,c1] : 0 <= m1 < 32 and 0 <= p1 < 112 and 0 <= q1 < 112 and 0 <= c1 < 32 and 0 <= p1 < 112 and 0 <= q1 < 112 and 0 <= m1 < 32 and 0 <= c1 < 32 }"
  // );
  // workload.SetEinsumProjection(dwise1_id,O2_id, true,
  // "{ dwise1[m2,p2,q2,r2,s2] -> O2[m2,p2,q2] : 0 <= m2 < 32 and 0 <= p2 < 112 and 0 <= q2 < 112 and 0 <= m2 < 32 and 0 <= p2+r2 < 112 and 0 <= q2+s2 < 112 and 0 <= m2 < 32 and 0 <= r2 < 3 and 0 <= s2 < 3 }"
  // );
  // workload.SetEinsumProjection(dwise1_id, O1_id, false,
  // "{ dwise1[m2,p2,q2,r2,s2] -> O1[m2,p2+r2,q2+s2] : 0 <= m2 < 32 and 0 <= p2 < 112 and 0 <= q2 < 112 and 0 <= m2 < 32 and 0 <= p2+r2 < 112 and 0 <= q2+s2 < 112 and 0 <= m2 < 32 and 0 <= r2 < 3 and 0 <= s2 < 3 }"
  // );
  // workload.SetEinsumProjection(dwise1_id, F2_id, false,
  // "{ dwise1[m2,p2,q2,r2,s2] -> F2[m2,r2,s2] : 0 <= m2 < 32 and 0 <= p2 < 112 and 0 <= q2 < 112 and 0 <= m2 < 32 and 0 <= p2+r2 < 112 and 0 <= q2+s2 < 112 and 0 <= m2 < 32 and 0 <= r2 < 3 and 0 <= s2 < 3 }"
  // );
  // workload.SetEinsumProjection(pwise2_id,O3_id, true,
  // "{ pwise2[m3,c3,p3,q3] -> O3[m3,p3,q3] : 0 <= m3 < 16 and 0 <= p3 < 112 and 0 <= q3 < 112 and 0 <= c3 < 32 and 0 <= p3 < 112 and 0 <= q3 < 112 and 0 <= m3 < 16 and 0 <= c3 < 32 }"
  // );
  // workload.SetEinsumProjection(pwise2_id, O2_id, false,
  // "{ pwise2[m3,c3,p3,q3] -> O2[c3,p3,q3] : 0 <= m3 < 16 and 0 <= p3 < 112 and 0 <= q3 < 112 and 0 <= c3 < 32 and 0 <= p3 < 112 and 0 <= q3 < 112 and 0 <= m3 < 16 and 0 <= c3 < 32 }"
  // );
  // workload.SetEinsumProjection(pwise2_id, F3_id, false,
  // "{ pwise2[m3,c3,p3,q3] -> F3[m3,c3] : 0 <= m3 < 16 and 0 <= p3 < 112 and 0 <= q3 < 112 and 0 <= c3 < 32 and 0 <= p3 < 112 and 0 <= q3 < 112 and 0 <= m3 < 16 and 0 <= c3 < 32 }"
  // );

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

  arch_props_ = new ArchProperties(arch_specs_);
  // Architecture constraints.
  config::CompoundConfigNode arch_constraints;

  if (arch.exists("constraints"))
    arch_constraints = arch.lookup("constraints");
  else if (rootNode.exists("arch_constraints"))
    arch_constraints = rootNode.lookup("arch_constraints");
  else if (rootNode.exists("architecture_constraints"))
    arch_constraints = rootNode.lookup("architecture_constraints");

  constraints_ = new mapping::Constraints(*arch_props_, workload_);
  constraints_->Parse(arch_constraints);

  if (verbose_)
    std::cout << "Architecture configuration complete." << std::endl;

  mapping::FusedMapping mapping =
    mapping::ParseMapping(rootNode.lookup("mapping"), workload);

  auto branch_tilings = analysis::TilingFromMapping(mapping, workload);

  // const std::string pwise2_tiling =
  // "{ [q3_2, p3_1, q3_1, c3_1, m3_0, c3_0] ->"
  // "  pwise2[m3=m3_0, c3=c3_1*16+c3_0, p3=p3_1, q3=q3_2*56+q3_1] :"
  // "  0 <= q3_2 < 2 and 0 <= p3_1 < 112 and 0 <= q3_1 < 56 and"
  // "  0 <= c3_1 < 2 and 0 <= m3_0 < 16 and 0 <= c3_0 < 16 }";

  // const std::string dwise1_tiling =
  // "{ [q3_2, p3_1, q3_1, p2_0, q2_0, r2_0, s2_0, m2_1, m2_0] ->"
  // "  dwise1[m2=m2_1*16+m2_0, p2=p3_1+p2_0, q2=q3_2*56+q3_1+q2_0,"
  //         "r2=r2_0, s2=s2_0] :"
  // "  0 <= q3_2 < 2 and 0 <= p3_1 < 112 and 0 <= q3_1 < 56 and"
  // "  0 <= r2_0 < 3 and 0 <= s2_0 < 3 and 0 <= m2_0 < 16 }";

  // const std::string pwise1_tiling =
  // "{ [q3_2, p3_1, q3_1, p1_0, q1_0, m1_1, c1_1, m1_0, c1_0] ->"
  // "  pwise1[m1=m1_1*16+m1_0, c1=c1_1*16+c1_0,"
  //         "p1=p3_1+p1_0, q1=q3_2*56+q3_1+q1_0] :"
  // "  0 <= q3_2 < 2 and 0 <= p3_1 < 112 and 0 <= q3_1 < 56 and"
  // "  0 <= c1_1 < 2 and 0 <= m1_0 < 16 and 0 <= c1_0 < 16 }";

  // std::map<analysis::DataSpaceID, size_t> dspace_top_indices;
  // dspace_top_indices.emplace(std::make_pair(O1_id, 2));
  // dspace_top_indices.emplace(std::make_pair(O2_id, 3));

  // // analysis::BranchTilings branch_tilings;
  // // branch_tilings.emplace(std::make_pair(pwise1_id, isl::map(GetIslCtx(),
  // //                                                           pwise1_tiling)));
  // // branch_tilings.emplace(std::make_pair(dwise1_id, isl::map(GetIslCtx(),
  // //                                                           dwise1_tiling)));
  // // branch_tilings.emplace(std::make_pair(pwise2_id, isl::map(GetIslCtx(),
  // //                                                           pwise2_tiling)));

  // const size_t PIPELINE_TILING_IDX = 3;

  // branch_tilings = analysis::LoopBoundsInference(std::move(branch_tilings),
  //                                                workload,
  //                                                PIPELINE_TILING_IDX,
  //                                                dspace_top_indices);

  // for (const auto& [einsum_id, tiling] : branch_tilings)
  // {
  //   std::cout << einsum_id << std::endl;
  //   std::cout << tiling << std::endl;
  //   for (auto i = 0; i < isl_map_dim(tiling.get(), isl_dim_out); ++i)
  //   {
  //     auto pared_tiling = project_dim_in_after(tiling, PIPELINE_TILING_IDX);
  //     std::cout << isl_pw_aff_to_str(isl_map_dim_min(pared_tiling.copy(), i))
  //               << std::endl;
  //     std::cout << isl_pw_aff_to_str(isl_map_dim_max(pared_tiling.copy(), i))
  //               << std::endl;
  //   }
  // }

}

Application::~Application()
{
  if (arch_props_)
    delete arch_props_;

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
