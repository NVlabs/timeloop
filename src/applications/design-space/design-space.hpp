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
#include <algorithm>

#include <boost/serialization/vector.hpp>
#include <boost/serialization/array.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/bitset.hpp>

#include "compound-config/compound-config.hpp"

#include "problem.hpp"
#include "arch.hpp"
//#include "simple-mapper.hpp"
#include "../mapper/mapper.hpp"
#include <vector>

using namespace config;

struct PointResult
{
  std::string config_name_;
  //Mapping best_mapping_; can't be used due to bug
  EvaluationResult result_;
  
  PointResult(std::string name, EvaluationResult result) : config_name_(name), result_(result) {}
  
  void PrintEvaluationResultsHeader(std::ostream& out)
  {
      out << "Summary stats for best mapping found by mapper:" << std::endl; 
      out << "config_name, Total Computes, utilization, pJ/Compute" << std::endl;
  }

  void PrintEvaluationResult(std::ostream& out)
  {
      out << config_name_ ; 
      out << ", " << result_.stats.total_computes;
      out << ", " << std::setw(4) << std::fixed << std::setprecision(2) << result_.stats.utilization;
      out << ", " << std::setw(8) << std::fixed << std::setprecision(3) << result_.stats.energy / result_.stats.total_computes << std::endl;
  }

};

//--------------------------------------------//
//                Application                 //
//--------------------------------------------//

class DesignSpaceExplorer
{
 protected:

  //want a list of files for each workload
  std::string problemspec_filename_;
  std::string archspec_filename_;

  std::vector<PointResult> designs_;
  std::vector<Application*> mappers_;

 public:

  DesignSpaceExplorer(std::string problemfile, std::string archfile)
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
      pspec_space.InitializeFromFile(problemspec_filename_);      
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
      aspec_space.InitializeFromFile(archspec_filename_);      
    }

    
    std::cout << "*** total arch: " << aspec_space.GetSize() << "   total prob: " << pspec_space.GetSize() << std::endl;        

    std::cout << "****** SOLVING ******" << std::endl;        
    //main loop, do the full product of problems x arches
    for (int arch_id = 0; arch_id < aspec_space.GetSize(); arch_id ++)
    {
      //retrieved via reference
      ArchSpaceNode curr_arch = aspec_space.GetNode(arch_id);

      std::cout << "*** working on arch : " << curr_arch.name_ << "  " << arch_id << std::endl;        

      for (int problem_id = 0; problem_id < pspec_space.GetSize(); problem_id ++)
      {
        //retrieved via reference
        ProblemSpaceNode curr_problem = pspec_space.GetNode(problem_id);
        
        // use problem and arch to run a mapper
        std::string config_name = curr_arch.name_ + "--" + curr_problem.name_;
        std::cout << "*** working on config : " << config_name << std::endl;        
        replace(config_name.begin(),config_name.end(),'/', '.'); 

        std::string file_name = "results/" + config_name;

        //output the two yaml to a single file
        //CompoundConfigNode arch = CompoundConfigNode(nullptr, YAML::Clone(curr_arch.yaml_));
        //CompoundConfigNode problem = CompoundConfigNode(nullptr, YAML::Clone(curr_problem.yaml_));
        std::ofstream combined_yaml_file("temp_dse.yaml");
        combined_yaml_file << curr_arch.yaml_ << std::endl; // <------------ HERE.
        combined_yaml_file << curr_problem.yaml_ << std::endl; // <------------ HERE.
        combined_yaml_file.close();
        //pull tempfile into a compound config
        config::CompoundConfig config("temp_dse.yaml");

        //std::cout << "arch yaml: \n" << curr_arch.yaml_ << std::endl;
        Application* mapper = new Application(&config, file_name);
        //SimpleMapper mapper = SimpleMapper(config_name, arch, problem);
        mapper->Run();
        PointResult result(mapper->name_, mapper->GetGlobalBest());
        mappers_.push_back(mapper);
        designs_.push_back(result);
    std::cout << "*** total arch: " << aspec_space.GetSize() << "   total prob: " << pspec_space.GetSize() << std::endl;        
        
      }
    }

    

    std::string result_filename =  "overview_" + archspec_filename_ + problemspec_filename_ + ".txt";
    replace(result_filename.begin(),result_filename.end(),'/', '.'); 
    std::ofstream result_txt_file("results/" + result_filename);
    //print final results
    designs_[0].PrintEvaluationResultsHeader(result_txt_file);
    for (size_t i = 0; i < designs_.size(); i++)
    {
      designs_[i].PrintEvaluationResult(result_txt_file);
    }
    result_txt_file.close();

  }


};
