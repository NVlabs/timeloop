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

#include "problem.hpp"

std::set<std::string> get_producers(map<string, ProblemPtr> & problem_map, string cur) {
  std::set<string> prods;
  auto einsum = problem_map.at(cur);
  std::set<string> data_spaces_consumed = einsum->GetReadDataSpace();
  for (auto it: problem_map) {
    auto data_space_produced = it.second->GetWriteDataSpace();
    auto overlap_data_spaces = intersection(data_space_produced, data_spaces_consumed);
    if (overlap_data_spaces.size()) {
      prods.insert(it.first);
    }
  }
  return prods;
}

//TODO: test this topological sort
vector<string> topological_sort_einsums(map<string, ProblemPtr> & problem_map) {
  std::set<string> not_yet_sorted;
  map<string, std::set<string> > producer_utils_map;
  for (auto it: problem_map) {
    not_yet_sorted.insert(it.first);
    auto prods = get_producers(problem_map, it.first);
    prods.erase(it.first);
    producer_utils_map.insert({it.first, prods});
  }

  vector<string> sorted_einsums;
  while(not_yet_sorted.size()) {
    for (auto cur: not_yet_sorted) {
      bool all_prods_sorted = true;
      for (string prod: producer_utils_map.at(cur)) {
        if (not_yet_sorted.count(prod)) {
          all_prods_sorted = false;
          break;
        }
      }
      if (all_prods_sorted) {
        sorted_einsums.push_back(cur);
        not_yet_sorted.erase(cur);
        break;
      }
    }
  }
  return sorted_einsums;
}
