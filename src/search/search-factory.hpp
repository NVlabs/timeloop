/* Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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

#include <libconfig.h++>

#include "search/search.hpp"
#include "search/random.hpp"
#include "search/exhaustive.hpp"
#include "search/linear-pruned.hpp"
#include "search/hybrid.hpp"
#include "search/random-pruned.hpp"

namespace search
{

//--------------------------------------------//
//             Parser and Factory             //
//--------------------------------------------//

SearchAlgorithm* ParseAndConstruct(libconfig::Setting& config,
                                   mapspace::MapSpace* mapspace,
                                   unsigned id)
{
  SearchAlgorithm* search = nullptr;
  
  std::string search_alg = "hybrid";
  config.lookupValue("algorithm", search_alg);
    
  if (search_alg == "random")
  {
    search = new RandomSearch(config, mapspace);
  }
  else if (search_alg == "exhaustive")
  {
    search = new ExhaustiveSearch(config, mapspace);
  }
  else if (search_alg == "linear-pruned")
  {
    search = new LinearPrunedSearch(config, mapspace, id);
  }
  else if (search_alg == "hybrid")
  {
    search = new HybridSearch(config, mapspace, id);
  }
  else if (search_alg == "random-pruned")
  {
    search = new RandomPrunedSearch(config, mapspace, id);
  }
  else
  {
    std::cerr << "ERROR: unsupported search algorithm: " << search_alg << std::endl;
    exit(-1);
  }

  return search;
}

} // namespace search
