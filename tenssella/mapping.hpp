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

#include "isl_utils.hpp"
#include "utils.hpp"
#include "printers.hpp"

class Mapping;
typedef shared_ptr<Mapping> MappingPtr;

class Mapping
{
 protected:
  isl_ctx* context_;
  //Tiling information
  std::map<int, isl_set*> tile_id_domains_;
  std::map<int, isl_map*> tile_id_to_set_;

  //How it map to space time tree
 std::map<int, isl_map*> skews_;


 public:
  Mapping() = delete;

  Mapping(isl_ctx* context) :
      context_(context)
  { }

  isl_map* TileIDToSet(int level)
  {
    return tile_id_to_set_.at(level);
  }

 void print_level() {
   TRACE(1) << "tile level: " << tile_id_to_set_.size() << std::endl;
   TRACE(1) << "skews level: " << skews_.size() << std::endl;
 }

  //int GetLeastLevelStorage() {
  //  return binding.size()-1;
  //}

  isl_set* TileIDDomain(int level)
  {
    return tile_id_domains_.at(level);
  }

  isl_map* Skew(int level)
  {
    return skews_.at(level);
  }

  //const std::map<unsigned, int>& TemporalProducts()
  //{
  //  return temporal_products_;
  //}

  std::string Var(unsigned level, unsigned rank)
  {
    return std::string("st" + std::to_string(level) + "r" + std::to_string(rank));
  }

  std::string Bound(std::vector<std::vector<int>>& bounds, unsigned level, unsigned rank)
  {
    return std::to_string(bounds.at(bounds.size()-level-1).at(bounds.front().size()-rank-1));
  }

  std::string Tuple(unsigned level, unsigned num_ranks)
  {
    std::string tuple = "";
    for (unsigned rank = num_ranks-1; rank < num_ranks; rank--) // another one.
    {
      tuple += Var(level, rank);
      if (rank != 0)
        tuple += ",";
    }
    return tuple;
  }

  void GenerateTileMaps(std::vector<std::vector<int>> bounds,
          std::vector<std::vector<int>> strides)
  {
    // Note: we use vectors textually-big-endian format, so the highest rank
    // (N-1) will be in vector entry 0.

    unsigned num_levels = bounds.size();
    unsigned num_ranks = bounds.front().size();

    // We have a limitation that the innermost (ST0) bounds must be == 1.
    // I.e., the architecture must have a set of operand and result latches
    // that can store exactly 1 timestamp worth of data.
    for (unsigned rank = 0; rank < num_ranks; rank++)
      if (bounds.at(num_levels-1).at(rank) != 1)
      {
        std::cerr << "ERROR: innermost spacetime level in mapping *must* have a tile size of 1." << std::endl;
        std::exit(1);
      }


    TRACE(1) << "Bounds:\n";
    for (unsigned level = 0; level < num_levels; level++)
    {
      auto& b = bounds.at(level);
      TRACE(1) << "  " << level << ": ";
      for (unsigned rank = 0; rank < num_ranks; rank++)
        TRACE(1) << b.at(rank) << " ";
      TRACE(1) << std::endl;
    }

    // Constraints are always the same across all levels.
    std::map<unsigned, std::string> constraints;
    for (unsigned rank = num_ranks-1; rank < num_ranks; rank--)
    {
      // Flattening.
      std::string expr = Var(num_levels-1, rank);
      for (unsigned level = num_levels-2; level < num_levels; level--)
      {
        expr = "(" + expr + ")";
        expr += "*" + Bound(strides, level, rank) + " + " + Var(level, rank);
      }
      std::string constraint = "r" + std::to_string(rank) + " = " + expr;

      // Domains.
      for (unsigned level = num_levels-1; level < num_levels; level--)
      {
        std::string domain = "0 <= " + Var(level, rank) + " < " + Bound(bounds, level, rank);
        constraint += " and " + domain;
      }

      if (rank != 0)
        constraint += " and ";

      constraints[rank] = constraint;

      TRACE(1) << "Constraints at rank " << rank << ": " << constraint << std::endl;
    }

    // RHS is always the same.
    std::string rhs = "[";
    for (unsigned rank = num_ranks-1; rank < num_ranks; rank--)
    {
      rhs += "r" + std::to_string(rank);
      if (rank != 0)
        rhs += ",";
    }
    rhs += "]";

    // LHS will be built iteratively as we walk through each level.
    std::string lhs;
    for (unsigned level = num_levels-1; level < num_levels; level--) // funky unsigned countdown.
    {
      TRACE(1) << "Generating Tile Maps for level: " << level << std::endl;

      std::string tuple = "[" + Tuple(level, num_ranks) + "]";

      if (level == num_levels-1)
      {
        lhs = "[[] -> " + tuple + "]";
      }
      else
      {
        lhs += " -> " + tuple;
        lhs = "[" + lhs + "]";
      }

      // Exists.
      std::string exists;
      if (level != 0)
      {
        for (unsigned child_level = level-1; child_level < num_levels; child_level--)
        {
          exists += Tuple(child_level, num_ranks);
          if (child_level != 0)
          {
            exists += ", ";
          }
        }
      }

      std::string final = "{ " + lhs + " -> " + rhs + " : ";
      if (!exists.empty())
      {
        final += "exists " + exists + " : ";
      }

      std::string indent(final.length(), ' ');

      for (unsigned rank = num_ranks-1; rank < num_ranks; rank--)
      {
        if (rank != num_ranks-1)
          final += indent;

        final += constraints.at(rank);

        if (rank == 0)
          final += " }";
        else
          final += "\n";
      }

      TRACE(1) << "  final string: " << final << std::endl;

      // TileID-to-set.
      tile_id_to_set_[level] = isl_map_read_from_str(context_, final.c_str());

      TRACE(1) << "  tile_id_to_set: " << tile_id_to_set_[level] << std::endl;

      // TileID domains.
      std::string tidd_final = "{ " + tuple + " : ";
      std::string tidd_indent(tidd_final.length(), ' ');

      for (unsigned rank = num_ranks-1; rank < num_ranks; rank--)
      {
        if (rank != num_ranks-1)
          tidd_final += tidd_indent;

        std::string bound = Bound(bounds, level, rank);// (level == 0) ? "1" : Bound(bounds, level, rank);
        tidd_final += "0 <= " + Var(level, rank) + " < " + bound;

        if (rank == 0)
          tidd_final += " }";
        else
          tidd_final += " and\n";
      }

      tile_id_domains_[level] = isl_set_read_from_str(context_, tidd_final.c_str());

      TRACE(1) << "  tile_id_domain: " << tile_id_domains_[level] << std::endl;
    }
  }
};


struct Partition
{
  std::string unit_name;
  std::size_t context_id;

  bool operator< (const Partition & p) const {
    if (unit_name < p.unit_name) {
      return true;
    } else if (unit_name == p.unit_name) {
      return context_id < p.context_id;
    } else {
      return false;
    }
  }

};

class Binding {
public:
  map<string, Partition> compute_binding_;
  map<string, map<int, map<string, Partition> > > memory_binding_;

  int LeastLevelStorage(string dataspace_name)
  {
    return memory_binding_.at(dataspace_name).rbegin()->first;
  }

  Partition LLSBindingPartition(string dataspace_name) {
    int lls = LeastLevelStorage(dataspace_name);
    auto comp2ins = memory_binding_.at(dataspace_name).at(lls);
    std::set<Partition> ins_set;
    for (auto it: comp2ins) {
      ins_set.insert(it.second);
    }
    if (ins_set.size() > 1) {
      std::cout << "ERROR: LLS binding can only have One instance. " << std::endl;
      std::cout << "    but we get: " << ins_set.size() << std::endl;
      for (auto it: ins_set) {
          std::cout << tab(4) << it.unit_name << std::endl;
      }
      assert(false);
    }
    return pick(ins_set);
  }

  string LLSBinding(string dataspace_name) {
    auto binding = LLSBindingPartition(dataspace_name);
    return binding.unit_name;
  }

  string LLSBindingFQ(string dataspace_name) {
    auto binding = LLSBindingPartition(dataspace_name);
    char buf[256];
    sprintf(buf, "%s/%lu", binding.unit_name.c_str(), binding.context_id);
    return std::string(buf);
  }

  std::string MemBinding(int hlevel, string dataspace_name, string computespace_name){
    return memory_binding_.at(dataspace_name).at(hlevel)
        .at(computespace_name).unit_name;
  }

  std::string ComputeBinding(string computespace_name){
    return compute_binding_.at(computespace_name).unit_name;
  }

  std::string MemBindingFQ(int hlevel, string dataspace_name, string computespace_name)
  {
    auto& binding = memory_binding_.at(dataspace_name).at(hlevel).at(computespace_name);
    char buf[256];
    sprintf(buf, "%s/%lu", binding.unit_name.c_str(), binding.context_id);
    return std::string(buf);
  }

  std::string ComputeBindingFQ(string computespace_name)
  {
    auto& binding = compute_binding_.at(computespace_name);
    char buf[256];
    sprintf(buf, "%s/%lu", binding.unit_name.c_str(), binding.context_id);
    return std::string(buf);
  }

};


//static inline
//int LeastLevelStorage(map<string, map<int, Partition> >& binding_,
//        string dataspace_name)
//{
//  return binding_.at(dataspace_name).rbegin()->first;
//}


//static inline
//std::string Binding(map<string, map<int, Partition> >& binding_,
//        int hlevel, string dataspace_name, string computespace_name)
//{
//  return binding_.at(dataspace_name).at(hlevel).at(computespace_name).unit_name;
//}

//static inline
//std::string BindingFQ(map<string, map<int, Partition> >& binding_,
//        int hlevel, string dataspace_name, string computespace_name)
//{
//  auto& binding = binding_.at(dataspace_name).at(hlevel).at(computespace_name);
//  char buf[256];
//  sprintf(buf, "%s/%lu", binding.unit_name.c_str(), binding.context_id);
//  return std::string(buf);
//}
