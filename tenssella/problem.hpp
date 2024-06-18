/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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

#include <vector>
#include <list>
#include <map>
#include <set>
#include <functional>

#include <isl/set.h>
#include <isl/map.h>
#include "utils.hpp"
#include "isl_utils.hpp"

class ProblemShape;
typedef shared_ptr<ProblemShape> ProblemPtr;

struct ComputeSpace
{
  std::string name;
  std::size_t num_ranks;
  std::list<std::string> subscripts;
  std::string transform_txt;
};


class ProblemShape
{
 protected:
  isl_ctx* context_;
  isl_set* iteration_space_;
  std::list<std::string> iteration_space_subscripts_; // FIXME: this is useless, remove.
  std::vector<std::string> iteration_space_dimensions_;
  ComputeSpace compute_space_;
  std::vector<std::string> data_spaces_;
  std::set<std::string> is_read;
  std::set<std::string> is_write;

 public:
  ProblemShape() {}

  ProblemShape(isl_ctx* context) :
      context_(context)
  { }

  isl_set* IterationSpace()
  {
    return cpy(iteration_space_);
  }

  std::list<std::string> IterationSpaceSubscripts()
  {
    return iteration_space_subscripts_;
  }

  std::vector<std::string> IterationSpaceDimensions()
  {
    return iteration_space_dimensions_;
  }

  std::size_t NumDataSpaces()
  {
    return data_spaces_.size();
  }

  std::vector<std::string> DataSpaceNames() {
    return data_spaces_;
  }

  bool ReadDataSpace(std::string ds_name) {
    return is_read.count(ds_name);
  }

  bool WriteDataSpace(std::string ds_name) {
    return is_write.count(ds_name);
  }

  std::string ComputeSpaceName()
  {
    return compute_space_.name;
  }

  std::list<std::string> ComputeSpaceSubscripts()
  {
    return compute_space_.subscripts;
  }

  std::size_t ComputeSpaceNumRanks()
  {
    return compute_space_.num_ranks;
  }

  ComputeSpace& GetComputeSpace()
  {
    return compute_space_;
  }

  std::set<std::string> GetReadDataSpace() {
    std::set<std::string> ret;
    for (auto ds: data_spaces_) {
      if (is_read.count(ds)) {
        ret.insert(ds);
      }
    }
    return ret;
  }

  std::set<std::string> GetWriteDataSpace() {
    std::set<std::string> ret;
    for (auto ds: data_spaces_) {
      if (is_write.count(ds)) {
        ret.insert(ds);
      }
    }
    return ret;
  }
};


std::set<std::string> get_producers(map<string, ProblemPtr>& problem_map, string cur);

//TODO: test this topological sort
vector<string> topological_sort_einsums(map<string, ProblemPtr> & problem_map);
