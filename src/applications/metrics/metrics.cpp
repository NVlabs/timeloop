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

#include "util/accelergy_interface.hpp"

#include "applications/metrics/metrics.hpp"

//--------------------------------------------//
//                Application                 //
//--------------------------------------------//

Application::Application(config::CompoundConfig* config)
  {
    auto rootNode = config->getRoot();
    // Architecture configuration.
    config::CompoundConfigNode arch;
    if (rootNode.exists("arch")) {
      arch = rootNode.lookup("arch");
    } else if (rootNode.exists("architecture")) {
      arch = rootNode.lookup("architecture");
    }
    arch_specs_ = model::Engine::ParseSpecs(arch);

    if (rootNode.exists("ERT")) {
      auto ert = rootNode.lookup("ERT");
      std::cout << "Found Accelergy ERT (energy reference table), replacing internal energy model." << std::endl;
      arch_specs_.topology.ParseAccelergyERT(ert);
     
     if (rootNode.exists("ART")){ // Nellie: well, if the users have the version of Accelergy that generates ART
          auto art = rootNode.lookup("ART");
          std::cout << "Found Accelergy ART (area reference table), replacing internal area model." << std::endl;
          arch_specs_.topology.ParseAccelergyART(art);  
      }     
      
    } else {
#ifdef USE_ACCELERGY
      // Call accelergy ERT with all input files
      if (arch.exists("subtree") || arch.exists("local")) {
        accelergy::invokeAccelergy(config->inFiles, out_prefix_, ".");
        std::string ertPath = out_prefix_ + ".ERT.yaml";
        auto ertConfig = new config::CompoundConfig(ertPath.c_str());
        auto ert = ertConfig->getRoot().lookup("ERT");
        arch_specs_.topology.ParseAccelergyERT(ert);

        std::string artPath = out_prefix_ + ".ART.yaml";
        auto artConfig = new config::CompoundConfig(artPath.c_str());
        auto art = artConfig->getRoot().lookup("ART");
        std::cout << "Generate Accelergy ART (area reference table) to replace internal area model." << std::endl;
        arch_specs_.topology.ParseAccelergyART(art);
      }
#endif
    }

    engine_.Spec(arch_specs_);
    std::cout << "Architecture configuration complete." << std::endl;
  }

  Application::~Application()
  {
  }

  // Run the evaluation.
  void Application::Run()
  {
    std::cout << engine_ << std::endl;
  }

