/* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include <fstream>

#include <boost/serialization/vector.hpp>
#include <boost/serialization/array.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/bitset.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>

#include "util/accelergy_interface.hpp"
#include "util/banner.hpp"
#include "mapping/parser.hpp"
#include "compound-config/compound-config.hpp"

//--------------------------------------------//
//                Application                 //
//--------------------------------------------//

class Application
{
 protected:
  problem::Workload workload_;
  model::Engine::Specs arch_specs_;

  // The mapping has to be a dynamic object because we cannot
  // instantiate it before the problem shape has been parsed. UGH.
  Mapping* mapping_;
  
 private:

  // Serialization
  friend class boost::serialization::access;
  template <class Archive>
  void serialize(Archive& ar, const unsigned int version=0)
  {
    if(version == 0)
    {
      ar& BOOST_SERIALIZATION_NVP(workload_);
    }
  }

 public:

  Application(config::CompoundConfig* config)
  {    
    auto rootNode = config->getRoot();

    // Model application configuration.
    bool verbose = false;
    if (rootNode.exists("model"))
    {
      auto model = rootNode.lookup("model");
      model.lookupValue("verbose", verbose);
    }
    if (verbose)
    {
      for (auto& line: banner)
        std::cout << line << std::endl;
      std::cout << std::endl;
    }

    // Problem configuration.
    auto problem = rootNode.lookup("problem");
    problem::ParseWorkload(problem, workload_);
    if (verbose)
      std::cout << "Problem configuration complete." << std::endl;

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
    arch_specs_ = model::Engine::ParseSpecs(arch);

    if (rootNode.exists("ERT"))
    {
      auto ert = rootNode.lookup("ERT");
      if (verbose)
        std::cout << "Found Accelergy ERT (energy reference table), replacing internal energy model." << std::endl;
      arch_specs_.topology.ParseAccelergyERT(ert);
    }
    else
    {
#ifdef USE_ACCELERGY
      // Call accelergy ERT with all input files
      if (arch.exists("subtree") || arch.exists("local"))
      {
        accelergy::invokeAccelergy(config->inFiles);
        auto ertConfig = new config::CompoundConfig("ERT.yaml");
        auto ert = ertConfig->getRoot().lookup("ERT");
        if (verbose)
          std::cout << "Generate Accelergy ERT (energy reference table) to replace internal energy model." << std::endl;
        arch_specs_.topology.ParseAccelergyERT(ert);
      }
#endif
    }

    if (verbose)
      std::cout << "Architecture configuration complete." << std::endl;

    // Mapping configuration: expressed as a mapspace or mapping.
    auto mapping = rootNode.lookup("mapping");
    mapping_ = new Mapping(mapping::ParseAndConstruct(mapping, arch_specs_, workload_));
    if (verbose)
      std::cout << "Mapping construction complete." << std::endl;
  }

  // This class does not support being copied
  Application(const Application&) = delete;
  Application& operator=(const Application&) = delete;

  ~Application()
  {
    if (mapping_)
      delete mapping_;
  }

  // Run the evaluation.
  void Run()
  {
    // Output file names.
    const std::string out_prefix = "timeloop-model.";
    const std::string stats_file_name = out_prefix + "stats.txt";
    const std::string xml_file_name = out_prefix + "map+stats.xml";
    const std::string map_txt_file_name = out_prefix + "map.txt";

    model::Engine engine;
    engine.Spec(arch_specs_);

    auto& mapping = *mapping_;
    
    auto eval_status = engine.Evaluate(mapping, workload_);
    auto level_names = arch_specs_.topology.LevelNames();
    
    for (unsigned level = 0; level < eval_status.size(); level++)
    {
      if (!eval_status[level].success)
      {
        std::cerr << "ERROR: couldn't map level " << level_names.at(level) << ": ";
        std::cerr << eval_status[level].fail_reason << std::endl;
        exit(1);
      }
    }
    // if (!std::accumulate(success.begin(), success.end(), true, std::logical_and<>{}))
    // {
    //   std::cout << "Illegal mapping, evaluation failed." << std::endl;
    //   return;
    // }

    if (engine.IsEvaluated())
    {
      std::cout << "Utilization = " << std::setw(4) << std::fixed << std::setprecision(2) << engine.Utilization() 
                << " | pJ/MACC = " << std::setw(8) << std::fixed << std::setprecision(3) << engine.Energy() /
          engine.GetTopology().MACCs() << std::endl;
    
      std::ofstream map_txt_file(map_txt_file_name);
      mapping.PrettyPrint(map_txt_file, arch_specs_.topology.StorageLevelNames(), engine.GetTopology().TileSizes());
      map_txt_file.close();

      std::ofstream stats_file(stats_file_name);
      stats_file << engine << std::endl;
      stats_file.close();
    }

    // Print the engine stats and mapping to an XML file
    std::ofstream ofs(xml_file_name);
    boost::archive::xml_oarchive ar(ofs);
    ar << BOOST_SERIALIZATION_NVP(engine);
    ar << BOOST_SERIALIZATION_NVP(mapping);
    const Application* a = this;
    ar << BOOST_SERIALIZATION_NVP(a);
  }
};

