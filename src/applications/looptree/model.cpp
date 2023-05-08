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
#include <barvinok/isl.h>

/**************
 * Prototype
 */

namespace analysis
{


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


  auto occupancies = analysis::OccupanciesFromMapping(mapping, workload);

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
