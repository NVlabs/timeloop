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
#include <vector>

#include <boost/serialization/vector.hpp>
#include <boost/serialization/array.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/bitset.hpp>

#include "compound-config/compound-config.hpp"
#include "applications/design-space/problem.hpp"
#include "applications/design-space/arch.hpp"
#include "applications/mapper/mapper.hpp"

using namespace config;

struct PointResult
{
  std::string config_name_;
  //Mapping best_mapping_; can't be used due to bug
  EvaluationResult result_;
  
  PointResult(std::string name, EvaluationResult result);
  
  void PrintEvaluationResultsHeader(std::ostream& out);
  void PrintEvaluationResult(std::ostream& out);
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
  std::vector<application::Mapper*> mappers_;

 public:

  DesignSpaceExplorer(std::string problemfile, std::string archfile);

  // ---------------------------------
  // Run the design space exploration.
  // ---------------------------------
  void Run();
};
