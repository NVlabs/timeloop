/* Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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
#include "isl_utils.hpp"
#include "utils.hpp"

struct DataSpace;
typedef shared_ptr<DataSpace> DataPtr;

struct DataSpace
{
    std::string name;
    std::size_t num_ranks;
    std::list<std::string> subscripts;
    bool isInputBoundary, isOutputBoundary;

    //Change write projection to map from einsum name to the access map
    std::map<string, isl_map*> read_projection;
    std::map<string, isl_map*> write_projection;

    //This seems redundant , we could remove it with parse from the map
    //std::list<std::string> read_projection_txt;
    //std::list<std::string> write_projection_txt;


    DataSpace(): isInputBoundary(false), isOutputBoundary(false) {}
    DataSpace(string name_, size_t num_ranks_):
        name(name_), num_ranks(num_ranks_),
        isInputBoundary(false), isOutputBoundary(false){}

    DataSpace(const DataSpace& ds) {
      name = ds.name;
      num_ranks = ds.num_ranks;
      for (auto s: ds.subscripts) {
        subscripts.push_back(s);
      }
      for (auto & it: ds.read_projection) {
        read_projection[it.first] = cpy(it.second);
      }
      for (auto & it: ds.write_projection) {
        write_projection[it.first] = cpy(it.second);
      }
      isInputBoundary = ds.isInputBoundary;
      isOutputBoundary = ds.isOutputBoundary;
    }

    std::string DataSpaceName() const
    {
      return name;
    }

    std::list<std::string> DataSpaceSubscripts() const
    {
      return subscripts;
    }

    std::size_t DataSpaceNumRanks() const
    {
      return num_ranks;
    }

    isl_map* ReadProjection(const string& cs_name) const
    {
      return cpy(read_projection.at(cs_name));
    }

    isl_map* WriteProjection(const string& cs_name) const
    {
      return cpy(write_projection.at(cs_name));
    }

    void setOutput() {
        isOutputBoundary = true;
    }

    void setInput() {
        isInputBoundary = true;
    }

    bool isOutput() const {
        return isOutputBoundary;
    }

    bool isInput() const {
        return isInputBoundary;
    }

    std::vector<string> GetReadComputeSpaceNames() const {
      vector<string> RDE;
      for (auto it: read_projection) {
        RDE.push_back(it.first);
      }
      return RDE;
    }

    std::vector<string> GetWriteComputeSpaceNames() const {
      vector<string> RDE;
      for (auto it: write_projection) {
        RDE.push_back(it.first);
      }
      return RDE;
    }

    std::vector<std::string> ReadProjectionTxt(const std::string& cs_name);

    std::vector<std::string> WriteProjectionTxt(const std::string& cs_name);
};
