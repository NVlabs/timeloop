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
#include <iomanip>

#include <boost/serialization/vector.hpp>
#include <boost/serialization/array.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/bitset.hpp>

#include "compound-config/compound-config.hpp"

#include "problem.hpp"
#include "arch.hpp"
//#include "simple-mapper.hpp"
#include "mapper.hpp"
#include <vector>

using namespace config;

//--------------------------------------------//
//                Application                 //
//--------------------------------------------//

class Application
{
 protected:

  //want a list of files for each workload
  std::string problemspec_filename_;
  std::string archspec_filename_;

  std::vector<MapResult> designs_;

 public:

  Application(std::string problemfile, std::string archfile)
  {
    problemspec_filename_ = problemfile;
    archspec_filename_ = archfile;
  }

  // ---------------
  // Run the design space exploration.
  // ---------------
  void Run()
  {

    //read in the problem spec space file as YAML, decide type of space we have:
    // 1: list of problems
    // 2: *later* single problem with swept parameters
    std::ifstream pspec_stream;
    pspec_stream.open(problemspec_filename_);
    YAML::Node pspec_yaml = YAML::Load(pspec_stream);

    std::cout << "****** INITIALIZING PROBLEM SPACE ******" << std::endl;    
    ProblemSpace pspec_space;    
    if (auto list = pspec_yaml["problem-space-files"])
    {
      pspec_space.InitializeFromFileList(list);
    }
    else {
      assert(0 && "Failed to find problem-space-files in YAML");
    }

    //read in the arch spec space file as YAML, decide type of space we have:
    // 1: list of arch to evaluate
    // 2: *later* single arch with swept parameters
    std::ifstream aspec_stream;
    aspec_stream.open(archspec_filename_);
    YAML::Node aspec_yaml = YAML::Load(aspec_stream);

    std::cout << "****** INITIALIZING ARCH SPACE ******" << std::endl;
    ArchSpace aspec_space;
    if (auto list = aspec_yaml["arch-space-files"])
    {
      aspec_space.InitializeFromFileList(list);
    }
    else if (auto sweep = aspec_yaml["arch-space-sweep"])
    {
      aspec_space.InitializeFromFileSweep(sweep);
    }
    else {
      assert(0 && "Failed to find problem-space-files in YAML");
    }

    std::cout << "****** SOLVING ******" << std::endl;        
    //main loop, do the full product of problems x arches
    for (int arch_id = 0; arch_id < 1; /*aspec_space.GetSize(); */ arch_id ++)
    {
      //retrieved via reference
      ArchSpaceNode curr_arch = aspec_space.GetNode(arch_id);
      for (int problem_id = 0; problem_id < 1; /*pspec_space.GetSize();*/ problem_id ++)
      {
        //retrieved via reference
        ProblemSpaceNode curr_problem = pspec_space.GetNode(problem_id);
        
        // use problem and arch to run a mapper
        std::string config_name = curr_arch.name_ + "--" + curr_problem.name_;
        CompoundConfigNode arch = CompoundConfigNode(nullptr, YAML::Clone(curr_arch.yaml_));
        CompoundConfigNode problem = CompoundConfigNode(nullptr, YAML::Clone(curr_problem.yaml_));
        std::cout << "arch yaml: \n" << curr_arch.yaml_ << std::endl;
        Mapper mapper = Mapper(config_name, arch, problem);
        mapper.Run();
        //SimpleMapper mapper = SimpleMapper(config_name, arch, problem);
        designs_.push_back(mapper.GetResults());
        
      }
    }

    std::cout << "instance size" << sizeof(MapResult) << std::endl;
    //print final results
    for (size_t i = 0; i < designs_.size(); i++)
    {
      designs_[i].PrintResults();
    }

  }
};
