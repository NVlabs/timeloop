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

#include <iostream>
#include <sstream>

#include <cassert>
#include <cstring>

#include <isl/constraint.h>
#include <isl/set.h>
#include <isl/map.h>
#include <isl/union_map.h>
#include <isl/ast_build.h>
#include <isl/ilp.h>

#include <barvinok/isl.h>

#include "printers.hpp"
#include "prelude.hpp"

#include "problem.hpp"
#include "architecture.hpp"
#include "mapping.hpp"
#include "data.hpp"

using std::string;
using std::ofstream;
using std::endl;

//Helper function for generate high dimensional time vector
pair<int, int> getTimeDim(isl_map* ispace);
string time_vec_str(int num_dim, string suffix);
isl_map* linear_domain_map_with_index(isl_set* s, unordered_set<int> index);

void CodegenCompute(map<string, ProblemPtr> & problems_,
        map<string, DataPtr> & data_spaces_, shared_ptr<Binding> bindings_,
        string cs_name, string compute_name, size_t t_dim, Printer& p);

class PHST {

public:

  //when generate code the level seems not essential, we have a flatten structure
  isl_ctx* context_;
  string name, binding;
  size_t max_ranks, max_tin_dim;

  //This three space is not necessary to have
  //from data space name to ispace
  map<string, isl_map*> init_ispace_map;

  // Ideally we could use the current PHST name and Binding to be the src of the update
  // However, because we are further pack this information in op into compute
  // we need to explicitly record the source as well.
  map<string, isl_map*> update_ispace_map; //Read/Drain to upper level and shrink
  map<string, pair<string, string>> update_dst_map; //The src and dst of update programs map
  map<string, pair<string, string>> update_dst_binding_map; //The src and dst of update programs map

  map<string, isl_map*> shrink_ispace_map; //Need to shrink data after read to lower level
  map<string, string> shrink_src_map;
  map<string, string> shrink_src_binding_map;

  //From compute space to ispace
  map<string, isl_map*> compute_ispace; //a phst/COMPUTE unit can have multiple einsum executed

  //Compute space name to data space to ispace
  map<string, map<string, isl_map*>> read_ispace_map; // all read space with cs_name as key
  map<string, map<string, string>> read_dst_map;  //For codegen purpose save the read destination
  map<string, map<string, string>> read_dst_binding_map;  //For codegen purpose save the read destination
  set<pair<string, string> > update_cs_name;        //Denote those with intention of Update

  bool isLLS;
  PHST() {isLLS = false; max_tin_dim = 0;}
  PHST(isl_ctx* ctx, const string& n, const string& b, size_t mr):
    context_(ctx), name(n), binding(b), max_ranks(mr)
    {isLLS = false; max_tin_dim = 0;}

  void SetLLS() {isLLS = true;}

  size_t GetComputeTDim(const string & cs_name) {
    return num_in_dims(compute_ispace.at(cs_name)) - 1;
  }

  void AddInit(isl_map* init_ispace, string ds_name) {
    init_ispace_map[ds_name] = init_ispace;
  }

  void AddShrink(isl_map* shrink_ispace, string ds_name, string src, string src_binding) {
    shrink_ispace_map[ds_name] = shrink_ispace;
    shrink_src_map[ds_name] = src;
    shrink_src_binding_map[ds_name] = src_binding;
  }

  void AddUpdate(isl_map* update_ispace, string ds_name,
    pair<string, string> & src2dst,
    pair<string, string> & src2dst_binding) {
    update_ispace_map[ds_name] = update_ispace;
    update_dst_map[ds_name] = src2dst;
    update_dst_binding_map[ds_name] = src2dst_binding;
  }

  void AddRead(isl_map* read_ispace, string cs_name,
    string ds_name, string dst, string dst_binding, bool isIU) {
    read_ispace_map[cs_name][ds_name] = read_ispace;
    read_dst_map[cs_name][ds_name] = dst;
    read_dst_binding_map[cs_name][ds_name] = dst_binding;
    if (isIU)
      update_cs_name.insert(std::make_pair(cs_name, ds_name));
  }

  void AddCompute(isl_map* compute_, string cs_name) {
    compute_ispace[cs_name] = compute_;
  }


  void Preprocessing();
  isl_map* SchedGenInit(string& ds_name, DataPtr data_space);
  isl_map* SchedGenShrink(string& ds_name, DataPtr data_space, int id);
  isl_map* SchedGenUpdate(string& ds_name, DataPtr data_space);
  isl_map* SchedGenCompute(string& cs_name, ProblemPtr problem_);
  isl_map* SchedGenRead(string& cs_name, string& ds_name, map<string, DataPtr> & data_spaces_);

  //Code generation pass on top of PHST
  void ScheduleGen(umap* schedule, map<string, ProblemPtr> & problems_, map<string, DataPtr>& data_spaces_);

  //Driver function for generate code
  void CodegenMemory(umap* sched, isl_ctx* ctx, Printer& p, Printer& q);

  //Driver function to generate init, this is specificly for emulation
  void CodegenInit(isl_ctx* ctx, Printer& p, map<string, DataPtr> & ds_map,
        map<string, umap*> & val_save_prog, map<string, umap*> & val_check_prog);
};

std::ostream& operator<<(std::ostream& out, const PHST& phst);
std::string DumpAccessCount(PHST& phst);

class Tenssella
{
 private:

  const bool kSerializeDrains = true;

  // ISL context.
  isl_ctx* context_;

  // User inputs.
  string test_name;

  // Iteration Space and its mapping
  map<string, ProblemPtr> problems_;
  map<string, MappingPtr> mappings_;

  //Data Space and its binding
  map<string, DataPtr> data_spaces_;
  shared_ptr<Architecture> arch_;
  shared_ptr<Binding> bindings_;

  // Architecture meta-information.
  int num_hardware_levels_;
  int num_tiling_levels_;

  // For validation and latency
  std::map<string, isl_union_map*> val_save_programs_;
  std::map<string, isl_union_map*> val_check_programs_;
  isl_ast_build* build_;

 public:
  Tenssella(isl_ctx* c,
          string name_,
          shared_ptr<Architecture> a,
          map<string, ProblemPtr> & p,
          map<string, MappingPtr> & m,
          map<string, DataPtr> & d,
          shared_ptr<Binding> b) :
      context_(c),
      test_name(name_),
      arch_(a),
      bindings_(b)
  {
    for (auto & it: p) {
      problems_.insert(it);
    }
    for (auto & it: m) {
      mappings_.insert(it);
    }
    for (auto & it: d) {
      data_spaces_.insert(it);
    }
    num_hardware_levels_ = arch_->NumLevels();
    num_tiling_levels_ = num_hardware_levels_-1; // [1:num_tiling_levels_]
  }

  // -----------------------------------------------------------------------------------
  // Stage T: Generate T-functions.
  // -----------------------------------------------------------------------------------
  void GenerateTs(
    std::string cs_name,
    // Outputs.
    std::map<int, isl_map*>& T,
    std::map<int, std::map<string, isl_map*>>& T_read,
    std::map<int, std::map<string, isl_map*>>& T_update)
  {
    TRACE(1) << "------------------------------" << std::endl;
    TRACE(1) << "    Generating T-functions    " << std::endl;
    TRACE(1) << "------------------------------" << std::endl;

    std::map<int, isl_map*> tile_id_to_set;
    std::map<int, isl_set*> space_time_domains;
    std::map<int, isl_map*> inv_skews;

    auto mapping_ = mappings_.at(cs_name);
    auto problem_ = problems_.at(cs_name);

    //Defensive coding add the root level inv skews
    //Noticing that root level does not have tiling

    inv_skews[num_tiling_levels_ + 1] =
        inv(cpy(mapping_->Skew(num_tiling_levels_ + 1)));

    for (int tlevel = num_tiling_levels_; tlevel >= 0; tlevel--)
    {
      TRACE(1) << "Tiling Level: " << tlevel << std::endl;

      // Limit domains.
      // Interestingly, with the latest formulations it seems we don't have to
      // do this any more. We'll keep this line in place in case we need to
      // revert back at some point.
      // tile_id_to_set[tlevel] = isl_map_intersect_domain(mapping_->TileIDToSet(tlevel), isl_set_copy(mapping_->TileIDDomain(tlevel)));
      tile_id_to_set[tlevel] = mapping_->TileIDToSet(tlevel);
      TRACE(2) << "  tile_id_to_set: " << tile_id_to_set[tlevel] << std::endl;

      // Compute space-time domains at each tlevel.
      space_time_domains[tlevel] = isl_map_range(isl_map_intersect_domain(
                                                   isl_map_copy(mapping_->Skew(tlevel)),
                                                   isl_set_copy(mapping_->TileIDDomain(tlevel))));

      // Invert skews and limit by space-time domains.
      TRACE(2) << "  skew (local): " << mapping_->Skew(tlevel) << std::endl;
      inv_skews[tlevel] = isl_map_reverse(isl_map_copy(mapping_->Skew(tlevel)));
      inv_skews[tlevel] = isl_map_intersect_domain(inv_skews[tlevel], isl_set_copy(space_time_domains[tlevel]));
      TRACE(2) << "  inv skew (local): " << inv_skews[tlevel] << std::endl;

      // Transform skews from local to global coordinates in both space-time and iteration-space.

      // Get rid of this special case
      //if (tlevel == num_tiling_levels_)
      //{
      //  // Top level: create the root []-> projection.
      //  const char* zero_d_space = "{ [i] }";
      //  isl_set* root = isl_set_read_from_str(context_, zero_d_space);
      //  inv_skews[tlevel] = isl_map_uncurry(isl_map_from_domain_and_range(root, isl_map_wrap(inv_skews[tlevel])));
      //}
      //if (tlevel != num_tiling_levels_)
      {
        TRACE(2) << "  inv skew (parent): " << inv_skews[tlevel+1] << std::endl;
        inv_skews[tlevel] = isl_map_product(isl_map_copy(inv_skews[tlevel+1]), inv_skews[tlevel]);
      }
      TRACE(2) << "  inv skew (global): " << inv_skews[tlevel] << std::endl;

      // Compute T functions.
      T[tlevel] = isl_map_apply_range(isl_map_copy(inv_skews[tlevel]), tile_id_to_set[tlevel]);
      TRACE(2) << "  T: " << T[tlevel] << std::endl;

      // Project the T functions to dataspaces.
      for (string ds_name : problem_->DataSpaceNames())
      {

        TRACE(2) << "Ds name: " << ds_name << std::endl;
        if (problem_->ReadDataSpace(ds_name))
        {
          T_read[tlevel][ds_name] = isl_map_apply_range(isl_map_copy(T[tlevel]),
                  isl_map_copy(data_spaces_.at(ds_name)->ReadProjection(problem_->ComputeSpaceName())));
          TRACE(1) << "    T_read: " << T_read[tlevel][ds_name] << std::endl;
        }
        else
        {
          T_read[tlevel][ds_name] = nullptr;
        }

        if (problem_->WriteDataSpace(ds_name))
        {
          T_update[tlevel][ds_name] = isl_map_apply_range(isl_map_copy(T[tlevel]),
                  isl_map_copy(data_spaces_.at(ds_name)->WriteProjection(problem_->ComputeSpaceName())));
          TRACE(1) << "    T_update: " << T_update[tlevel][ds_name] << std::endl;
        }
        else
        {
          T_update[tlevel][ds_name] = nullptr;
        }
      }
    }

    TRACE(1) << "T-functions generated." << std::endl;
    TRACE(1) << std::endl;
  }

  // -----------------------------------------------------------------------------------
  // Stage DECOUPLE: Decoupling: generate initial relations.
  // -----------------------------------------------------------------------------------

  void Decouple(
    std::string cs_name,
    // Inputs.
    std::map<int, isl_map*>& T,
    std::map<int, std::map<string, isl_map*>>& T_read,
    std::map<int, std::map<string, isl_map*>>& T_update,
    // Outputs.
    std::map<int, std::map<string, isl_map*>>& read_ispace,
    std::map<int, std::map<string, isl_map*>>& shrink_ispace,
    isl_map*& compute_ispace,
    std::map<int, std::map<string, isl_map*>>& update_ispace)
  {
    TRACE(1) << "------------------" << std::endl;
    TRACE(1) << "    Decoupling    " << std::endl;
    TRACE(1) << "------------------" << std::endl;

    auto problem_ = problems_.at(cs_name);

    TRACE(1) << "Einsum: " << cs_name << std::endl;
    // For each hardware level...
    for (string ds_name : problem_->DataSpaceNames())
    {
      std::string dataspace_name = ds_name;
      // For each dataspace, given:
      //   T(L)  : [<abstract s,t from root to L+1> -> <abstract s_L,t_L>] -> { dataspace coords }
      //   H(L)  : [<abstract s,t from root to L+1> -> <abstract s_L,t_L>] -> <q: physical partition spacetime coord at L-1>
      //   H(L+1): <abstract s,t from root to L+1> -> <p: physical partition spacetime coord at L>
      // compute:
      //   R(L)  : <p: physical partition spacetime coord at L> -> [<q: physical partition spacetime coord at L-1> -> { dataspace coords }
      for (int hlevel = bindings_->LeastLevelStorage(ds_name)-1; hlevel >= 1; hlevel--)
      {
        TRACE(1) << "Hardware Level: " << hlevel << std::endl;

        int tlevel = hlevel;

        std::string instance_name_cur =
            bindings_->MemBinding(hlevel+1, dataspace_name, cs_name);
        std::string instance_name_child =
            bindings_->MemBinding(hlevel, dataspace_name, cs_name);

        TRACE(2) << "  Dataspace: " << dataspace_name << " cur: " << instance_name_cur
                 << " child: " << instance_name_child << std::endl;

        // Yeah, the off-by-1 in looking up the instance map is weird.
        isl_map* H_child = isl_map_copy(arch_->MemInstanceMap(hlevel, instance_name_child, cs_name));
        isl_map* H_cur = isl_map_copy(arch_->MemInstanceMap(hlevel+1, instance_name_cur, cs_name));

        TRACE(2) << "    H_child: " << H_child << std::endl;
        TRACE(2) << "    H_cur: " << H_cur << std::endl;

        // The following code looks complicated but is just ISL acrobatics to
        // transform the maps to get what we want.
        TRACE(2) << "    H_cur after curry: " << isl_map_curry(cpy(H_child)) << std::endl;

        isl_map* instance_connection = isl_map_apply_range(isl_map_reverse(H_cur), isl_map_curry(H_child));
        TRACE(2) << "    instance_connection (before factor): " << instance_connection << std::endl;

        instance_connection = isl_map_range_factor_range(instance_connection);
        TRACE(2) << "    instance_connection (after factor): " << instance_connection << std::endl;

        // Reads and Shrinks.
        if (T_read[tlevel][ds_name] != nullptr)
        {
          isl_map* T = isl_map_copy(T_read[tlevel][ds_name]);
          TRACE(2) << "    T_read: " << T << std::endl;

          isl_map* child_to_address = isl_map_apply_range(isl_map_reverse(isl_map_copy(H_child)), T);
          TRACE(2) << "    child_to_address: " << child_to_address << std::endl;

          isl_map* read = isl_map_range_product(child_to_address, isl_map_reverse(isl_map_copy(instance_connection)));
          TRACE(1) << "    Read_" << ds_name << " before trans: " << read << std::endl;

          //auto dom = domain(read);
          //isl_set_coalesce(dom);
          //TRACE(2) << "    set to be linearize: " << str(dom) << std::endl;

          //unordered_set<int> indices;
          //for (int i = 1; i < num_dims(dom); i++) {
          //   indices.insert(num_dims(dom) - 1 -i);
          //}

          //auto linear_map = linear_domain_map_with_index(dom, indices);

          //comment out the linearization
          //read = dot(inv(linear_map), read);

          //TRACE(2) << "    linear map: " << str(linear_map) << std::endl;
          read = isl_map_uncurry(isl_map_reverse(isl_map_uncurry(read)));
          TRACE(1) << "    Read_" << ds_name << ": " << read << std::endl;

          read_ispace[hlevel][ds_name] = read;

          // Derive shrinks from reads.
          if (T_update[tlevel][ds_name] != nullptr)
          {
            // The shrink will be performed by the drain-update operation.
            shrink_ispace[hlevel][ds_name] = nullptr;
          }
          else
          {
            // Shrinks are a copy of Reads at this stage.
            isl_map* shrink = isl_map_copy(read);
            TRACE(1) << "    Shrink_" << ds_name << ": " << shrink << std::endl;
            shrink_ispace[hlevel][ds_name] = shrink;
          }
        }
        else
        {
          read_ispace[hlevel][ds_name] = nullptr;
        }

        // Updates.
        if (T_update[tlevel][ds_name] != nullptr)
        {
          isl_map* T = isl_map_copy(T_update[tlevel][ds_name]);
          TRACE(2) << "    T_update: " << T << std::endl;

          isl_map* child_to_address = isl_map_apply_range(isl_map_reverse(isl_map_copy(H_child)), T);
          TRACE(2) << "    child_to_address: " << child_to_address << std::endl;

          isl_map* update = isl_map_range_product(child_to_address, isl_map_reverse(instance_connection));

          //auto dom = domain(update);
          //isl_set_coalesce(dom);
          //TRACE(2) << "    set to be linearize: " << str(dom) << std::endl;

          //unordered_set<int> indices;
          //for (int i = 1; i < num_dims(dom); i++) {
          //   indices.insert(num_dims(dom) - 1 -i);
          //}

          //auto linear_map = linear_domain_map_with_index(dom, indices);
          //TRACE(2) << "    linear map: " << str(linear_map) << std::endl;

          //update = dot(inv(linear_map), update);

          update = isl_map_uncurry(isl_map_reverse(isl_map_uncurry(update)));
          TRACE(1) << "    Update_" << ds_name << ": " << update << std::endl;

          update_ispace[hlevel][ds_name] = update;
        }
        else
        {
          update_ispace[hlevel][ds_name] = nullptr;
        }

      }
    }

    //
    // ---- Temporary: Special case compute ----
    //

    // What's weird about compute as a "data transfer" is that we have a confluence
    // of multiple dataspaces. For all our other transfers, each code block was
    // handling only a single dataspace. Yes, we were handling multicast and reduction
    // of values, but they were within the same dataspace.
    std::string instance_name_cur =
        bindings_->ComputeBinding(problem_->ComputeSpaceName());
    std::string instance_name_child = instance_name_cur;

    TRACE(1) << "Hardware Level: COMPUTE" << std::endl;
    TRACE(2) << "  Compute space: " << problem_->ComputeSpaceName() << " cur: " << instance_name_cur
             << " child: " << instance_name_child << std::endl;

    // Yeah, the off-by-1 in looking up the instance map is weird.
    isl_map* T_compute = isl_map_copy(T[0]);
    isl_map* H_compute = isl_map_copy(arch_->ComputeInstanceMap(0, cs_name));

    TRACE(2) << "    T_compute: " << T_compute << std::endl;
    TRACE(2) << "    H_compute: " << H_compute << std::endl;

    compute_ispace = isl_map_apply_range(isl_map_reverse(H_compute), T_compute);
    // auto dom = domain(compute_ispace);
    // isl_set_coalesce(dom);
    // TRACE(2) << "    set to be linearize: " << str(dom) << std::endl;

    // unordered_set<int> indices;
    // for (int i = 1; i < num_dims(dom); i++) {
    //    indices.insert(num_dims(dom) - 1 -i);
    // }

    // auto linear_map = linear_domain_map_with_index(dom, indices);
    // TRACE(2) << "    linear map: " << str(linear_map) << std::endl;

    // compute_ispace = dot(inv(linear_map), compute_ispace);

    TRACE(1) << "    Compute_" << problem_->ComputeSpaceName() << ": " << compute_ispace << std::endl;

    // ---- End compute ----


    TRACE(1) << "Decoupling complete." << std::endl;
    TRACE(1) << std::endl;
  }

  void Init_Tensor(std::map<string, isl_map*>& init_ispace ,
          std::map<string, isl_map*>& LLS_shrink_ispace ,
          map<string, map<int, map<string, isl_map*>>> & shrink_ispace,
          map<string, map<int, map<string, isl_map*>>> & T_read_map,
          map<string, map<int, map<string, isl_map*>>> & T_update_map) {

    //
    // Emulation only: Initialize contents of last-level storage.
    //
    TRACE(1) << "Initializing tensors for emulation." << std::endl;

    for (auto it : data_spaces_)
    {
      //Each data space need to initilizing , we need to find its LLS
      std::string dataspace_name = it.first;
      int hlevel = bindings_->LeastLevelStorage(dataspace_name) - 1;
      string cs_name = pick(it.second->GetReadComputeSpaceNames());
      auto problem_ = problems_.at(cs_name);

      TRACE(2) << "    Data Space: " << dataspace_name << ", lls: " << hlevel+1 << std::endl;

      std::string instance_name_cur = bindings_->LLSBinding(dataspace_name);
      //Case 1 init at DRAM
      if (hlevel == num_hardware_levels_ - 1) {
        isl_set* ispace = isl_set_copy(problem_->IterationSpace());
        TRACE(2) << "    instance name: " << instance_name_cur << ", ispace: " << ispace << std::endl;

        isl_map* H_cur = isl_map_copy(arch_->InitInstanceMap(instance_name_cur));
        TRACE(2) << "    Binding: " << H_cur << std::endl;

        isl_set* last_level_storage = isl_map_range(H_cur);
        TRACE(2) << "    last level storage: " << last_level_storage << std::endl;

        // FIXME: this code will only work if there is exactly 1 last-level storage
        // element (space) that exists over exactly 1 time coordinate.
        isl_map* init = isl_map_from_domain_and_range(last_level_storage, ispace);
        TRACE(2) << "    init map: " << init << std::endl;
        TRACE(2) << "    read proj: " << it.second->ReadProjection(problem_->ComputeSpaceName()) << std::endl;
        isl_map* dspace_init = isl_map_apply_range(init,it.second->ReadProjection(problem_->ComputeSpaceName()));
        TRACE(2) << "    init_ispace: " << dspace_init << std::endl;

        assert(int_upper_bound(isl_union_set_card(to_uset(domain(dspace_init)))) == 1);

        init_ispace[it.first] = dspace_init;
      } else {

      //Case 2: init at an inner level
      //TODO: also need to handle shrink just put this into another function
      //if (hlevel < num_hardware_levels_ - 1) {
        isl_map* all_reads = nullptr;

        //Go through all of its read einsum, find the union of range and
        //that's the space you need to initializing
        for (auto it: T_read_map) {
          //Init the update space
          string cs_name = it.first;
          if (T_update_map.at(cs_name).at(hlevel + 1). at(dataspace_name) == nullptr) {
            isl_map* H_cur = cpy(arch_->MemInstanceMap(hlevel+1, instance_name_cur, it.first));
            isl_map* t_rel = it.second.at(hlevel+1).at(dataspace_name);
            isl_map* dspace_shrink = isl_map_apply_range(inv(H_cur), t_rel);
            TRACE(2) << "Shrink ispace for LLS: " << dspace_shrink << std::endl;
            TRACE(2) << "Shrink ispace for Einsum: " << cs_name << std::endl;

            //Also save in a separate data structure for the new PHST codegen Pass
            shrink_ispace[cs_name][hlevel + 1][dataspace_name] = dspace_shrink;
            assert(LLS_shrink_ispace.count(dataspace_name) == 0);
            LLS_shrink_ispace[dataspace_name] = dspace_shrink;

          } else {
            isl_map* H_cur = cpy(arch_->MemInstanceMap(hlevel+1, instance_name_cur, it.first));
            TRACE(2) << "      HST: " << H_cur << std::endl;
            TRACE(2) << "      compute space: " << it.first << std::endl;
            isl_map* t_rel = it.second.at(hlevel+1).at(dataspace_name);
            TRACE(2) << "      T rel: " << t_rel << std::endl;
            isl_map* new_dspace_init = isl_map_apply_range(inv(H_cur), t_rel);
            TRACE(2) << "      new init_ispace: " << new_dspace_init << std::endl;
            assert(all_reads == nullptr);
            all_reads = cpy(new_dspace_init);
            TRACE(2) << " final new init_ispace: " << all_reads << std::endl;
            init_ispace[dataspace_name] = all_reads;
          }
        }
      }
    }
    // ---- End emulation ----
  }

  // -----------------------------------------------------------------------------------
  // Stage ADDRMAP: Address mapping.
  // -----------------------------------------------------------------------------------

  // void MapAddresses(
  //   // Inputs.
  //   std::map<int, std::map<std::size_t, isl_map*>>& read_ispace,
  //   std::map<int, std::map<std::size_t, isl_map*>>& shrink_ispace,
  //   std::map<int, std::map<std::size_t, isl_map*>>& update_ispace)
  // {
  //   TRACE(1) << "-----------------------" << std::endl;
  //   TRACE(1) << "    Address mapping    " << std::endl;
  //   TRACE(1) << "-----------------------" << std::endl;

  //   // For each hardware level...
  //   for (int hlevel = num_hardware_levels_-1; hlevel >= 1; hlevel--)
  //   {
  //     TRACE(1) << "Address mapping for Hardware Level: " << hlevel << std::endl;

  //     // For each data space...
  //     for (std::size_t dsi = 0; dsi < problem_->NumDataSpaces(); dsi++)
  //     {
  //       if (read_ispace.at(hlevel).at(dsi) == nullptr)
  //       {
  //         // read_programs[hlevel][dsi] = nullptr;
  //       }
  //       else
  //       {
  //         TRACE(2) << "  Data space: " << problem_->DataSpaceName(dsi) << std::endl;

  //         bool iu = (update_ispace.at(hlevel).at(dsi) != nullptr);

  //         // Set up the iteration spaces.
  //         isl_set* read_iteration_space = isl_map_wrap(read_ispace.at(hlevel).at(dsi));
  //         read_iteration_space = isl_set_set_tuple_name(read_iteration_space, "__READ");
  //         TRACE(2) << "    read iteration space: " << read_iteration_space << std::endl;

  //         // Create the schedule.

  //         // Determine a schedule to transfer the internal contents of a tile.
  //         // For now we'll simply use the *same* order as the iteration space.
  //         // So we'll use [i0,i1,...] for both RHS and LHS of the map we're
  //         // producing. If we wish to change this, we'll have to generate a new
  //         // different indexing for the RHS.
  //         std::stringstream data_space_index;
  //         std::size_t rank;
  //         for (rank = 0; rank < problem_->DataSpaceNumRanks(dsi); rank++)
  //         {
  //           if (rank != 0)
  //             data_space_index << ",";
  //           data_space_index << "i" << rank;
  //         }

  //         // Pad ranks in schedule.
  //         std::stringstream padded_indices;
  //         for (; rank < max_ranks; rank++)
  //         {
  //           if (rank != 0)
  //             padded_indices << ",";
  //           padded_indices << "0";
  //         }

  //         char sched_str[256];
  //         sprintf(sched_str,
  //                 // Note:
  //                 // (1) that the 0 is inserted to serialize read-IUs with updates.
  //                 //     We'll keep them in plain reads too -- they are harmless.
  //                 "{ %s[[%s[s_out,t_out] -> %s[s_in,t_in]] -> %s[%s]] -> [t_out,0,t_in,%s%s,s_out,0,s_in] }",
  //                 "__READ",
  //                 mapping_->Binding(hlevel+1, problem_->DataSpaceName(dsi)).c_str(),
  //                 mapping_->Binding(hlevel, problem_->DataSpaceName(dsi)).c_str(),
  //                 problem_->DataSpaceName(dsi).c_str(),
  //                 data_space_index.str().c_str(),
  //                 data_space_index.str().c_str(),
  //                 padded_indices.str().c_str());

  //         isl_map* schedule_projection = isl_map_read_from_str(context_, sched_str);
  //         TRACE(2) << "    read schedule projection: " << schedule_projection << std::endl;

  //         isl_map* read_schedule = isl_map_intersect_domain(schedule_projection, isl_set_copy(read_iteration_space));
  //         TRACE(2) << "    read schedule: " << read_schedule << std::endl;

  //         isl_map* read_program = isl_map_intersect_domain(read_schedule, isl_set_copy(read_iteration_space));

  //         // Create a "name" for the program, which contains enough information about
  //         // the target engine, src/dst buffets and tensors to trigger a statement
  //         // decoder macro on the target platform.
  //         char read_xfer_name[256];

  //         // READs will be programmed onto engines at the source (outer) level.
  //         // However, to avoid over-serialization we also add the destination
  //         // name. Perhaps we should be adding the dataspace ID instead.
  //         std::string read_engine_name =
  //           mapping_->BindingFQ(hlevel+1, problem_->DataSpaceName(dsi));// + "->" +
  //         //mapping_->Binding(hlevel, problem_->DataSpaceName(dsi));

  //         sprintf(read_xfer_name, "ACTION_%s[@%s@, @%s@, @%s@, @%s@, %lu]",
  //                 iu ? "READ_IU" : "READ",
  //                 read_engine_name.c_str(),
  //                 mapping_->BindingFQ(hlevel+1, problem_->DataSpaceName(dsi)).c_str(),
  //                 mapping_->BindingFQ(hlevel, problem_->DataSpaceName(dsi)).c_str(),
  //                 problem_->DataSpaceName(dsi).c_str(),
  //                 problem_->DataSpaceNumRanks(dsi));

  //         read_program = isl_map_set_tuple_name(read_program, isl_dim_in, read_xfer_name);
  //         TRACE(1) << "    read program: " << read_program << std::endl;

  //         read_programs[hlevel][dsi] = isl_union_map_from_map(read_program);
  //       }
  //     }
  //   }
  // }

  // -----------------------------------------------------------------------------------
  // Stage REUSE: Reuse analysis.
  // -----------------------------------------------------------------------------------

  void AnalyzeReuse(
    // Inputs and Outputs.
    map<string, map<int, map<string, isl_map*>>>& read_ispace,
    map<string, map<int, map<string, isl_map*>>>& shrink_ispace,
    map<string, map<int, map<string, isl_map*>>>& update_ispace)
  {
    TRACE(1) << "----------------------" << std::endl;
    TRACE(1) << "    Reuse analysis    " << std::endl;
    TRACE(1) << "----------------------" << std::endl;

    //
    // Stage REUSE.1: Local temporal reuse.
    //

    //
    // Stage REUSE.1.1: READ blocks.
    //

    for (auto it: problems_) {

    string cs_name = it.first;
    auto problem_ = it.second;
    // For each data space...
    for (string ds_name: problem_->DataSpaceNames())
    {
      // For each hardware level...
      TRACE(1) << "Data space: " << ds_name << std::endl;
      for (int hlevel = bindings_->LeastLevelStorage(ds_name) - 1; hlevel >= 1; hlevel--)
      {
        TRACE(1) << "  Read programs for Hardware Level: " << hlevel << std::endl;

        if (read_ispace.at(cs_name).at(hlevel).at(ds_name) != nullptr)
        {
          TRACE(2) << "    xfer: " << read_ispace.at(cs_name).at(hlevel).at(ds_name) << std::endl;

          //Adative to multi-dimensional time in SHST
          pair<int, int> t_dim = getTimeDim(read_ispace.at(cs_name).at(hlevel).at(ds_name));
          string t_out_str_domain = time_vec_str(t_dim.first, "out");
          string t_out_str_range = time_vec_str(t_dim.first, "_out");
          string t_in_str = time_vec_str(t_dim.second, "in");

          std::string parent_instance_name =
              bindings_->MemBinding(hlevel+1, ds_name, cs_name);
          std::string child_instance_name =
              bindings_->MemBinding(hlevel, ds_name, cs_name);

          char time_shift_str[256];
          sprintf(time_shift_str, "{ [%s[s_out,%s] -> %s[s_in,%s]] -> [%s[s_out,%s] -> %s[s_in,%s+1]] }",
                  parent_instance_name.c_str(), t_out_str_domain.c_str(),
                  child_instance_name.c_str(), t_in_str.c_str(),
                  parent_instance_name.c_str(), t_out_str_range.c_str(),
                  child_instance_name.c_str(), t_in_str.c_str());

          isl_map* time_shift = isl_map_read_from_str(context_, time_shift_str);
          TRACE(2) << "    time shift: " << time_shift << std::endl;

          isl_map* inverse_xfer = isl_map_reverse(isl_map_copy(read_ispace.at(cs_name).at(hlevel).at(ds_name)));
          isl_map* shifted_xfer = isl_map_apply_range(inverse_xfer, time_shift);
          shifted_xfer = isl_map_reverse(shifted_xfer);
          TRACE(2) << "    shifted xfer: " << shifted_xfer << std::endl;

          read_ispace.at(cs_name).at(hlevel).at(ds_name) =
            isl_map_subtract(read_ispace.at(cs_name).at(hlevel).at(ds_name), shifted_xfer);
          TRACE(1) << "    delta xfer: " << read_ispace.at(cs_name).at(hlevel).at(ds_name) << std::endl;
        }
      }
    }

    //
    // Stage REUSE.1.2: SHRINK blocks.
    //

    // For each data space...
    for (string ds_name: problem_->DataSpaceNames())
    {
      // For each hardware level...
      TRACE(1) << "Data space: " << ds_name << std::endl;
      // For each hardware level...
      for (int hlevel = bindings_->LeastLevelStorage(ds_name)-1; hlevel >= 1; hlevel--)
      {
        TRACE(1) << "Shrink programs for Hardware Level: " << hlevel << std::endl;

        if (shrink_ispace.at(cs_name).at(hlevel).at(ds_name) != nullptr)
        {
          TRACE(2) << "  Data space: " << ds_name << std::endl;
          TRACE(2) << "    xfer: " << shrink_ispace.at(cs_name).at(hlevel).at(ds_name) << std::endl;

          pair<int, int> t_dim = getTimeDim(shrink_ispace.at(cs_name).at(hlevel).at(ds_name));
          string t_out_str_domain = time_vec_str(t_dim.first, "out");
          string t_out_str_range = time_vec_str(t_dim.first, "_out");
          string t_in_str = time_vec_str(t_dim.second, "in");


          std::string parent_instance_name =
              bindings_->MemBinding(hlevel+1, ds_name, cs_name);
          std::string child_instance_name =
              bindings_->MemBinding(hlevel, ds_name, cs_name);

          char time_shift_str[256];
          sprintf(time_shift_str, "{ [%s[s_out,%s] -> %s[s_in,%s]] -> [%s[s_out,%s] -> %s[s_in,%s-1]] }",
                  parent_instance_name.c_str(), t_out_str_domain.c_str(),
                  child_instance_name.c_str(), t_in_str.c_str(),
                  parent_instance_name.c_str(), t_out_str_range.c_str(),
                  child_instance_name.c_str(), t_in_str.c_str());

          isl_map* time_shift = isl_map_read_from_str(context_, time_shift_str);
          TRACE(2) << "    time shift: " << time_shift << std::endl;

          isl_map* inverse_xfer = isl_map_reverse(isl_map_copy(shrink_ispace.at(cs_name).at(hlevel).at(ds_name)));
          isl_map* shifted_xfer = isl_map_apply_range(inverse_xfer, time_shift);
          shifted_xfer = isl_map_reverse(shifted_xfer);
          TRACE(2) << "    shifted xfer: " << shifted_xfer << std::endl;

          shrink_ispace.at(cs_name).at(hlevel).at(ds_name) =
            isl_map_subtract(shrink_ispace.at(cs_name).at(hlevel).at(ds_name), shifted_xfer);
          TRACE(1) << "    delta xfer: " << shrink_ispace.at(cs_name).at(hlevel).at(ds_name) << std::endl;
        }
      }
    }

    //
    // Stage REUSE.1.3: UPDATE blocks.
    //

    // For each data space...
    for (string ds_name: problem_->DataSpaceNames())
    {
      // For each hardware level...
      TRACE(1) << "Data space: " << ds_name << std::endl;
      // For each hardware level...
      for (int hlevel = bindings_->LeastLevelStorage(ds_name)-1; hlevel >= 1; hlevel--)
      {
        TRACE(1) << "Update programs for Hardware Level: " << hlevel << std::endl;

        if (update_ispace.at(cs_name).at(hlevel).at(ds_name) != nullptr)
        {
          TRACE(2) << "  Data space: " << ds_name << std::endl;
          TRACE(2) << "    xfer: " << update_ispace.at(cs_name).at(hlevel).at(ds_name) << std::endl;

          pair<int, int> t_dim = getTimeDim(update_ispace.at(cs_name).at(hlevel).at(ds_name));
          string t_out_str_domain = time_vec_str(t_dim.first, "out");
          string t_out_str_range = time_vec_str(t_dim.first, "_out");
          string t_in_str = time_vec_str(t_dim.second, "in");

          std::string parent_instance_name =
              bindings_->MemBinding(hlevel+1, ds_name, cs_name);
          std::string child_instance_name =
              bindings_->MemBinding(hlevel, ds_name, cs_name);

          char time_shift_str[256];
          sprintf(time_shift_str, "{ [%s[s_out,%s] -> %s[s_in,%s]] -> [%s[s_out,%s] -> %s[s_in,%s-1]] }",
                  parent_instance_name.c_str(), t_out_str_domain.c_str(),
                  child_instance_name.c_str(), t_in_str.c_str(),
                  parent_instance_name.c_str(), t_out_str_range.c_str(),
                  child_instance_name.c_str(), t_in_str.c_str());

          TRACE(2) << "    time shift str: " << time_shift_str << std::endl;
          isl_map* time_shift = isl_map_read_from_str(context_, time_shift_str);
          TRACE(2) << "    time shift: " << time_shift << std::endl;

          isl_map* inverse_xfer = isl_map_reverse(isl_map_copy(update_ispace.at(cs_name).at(hlevel).at(ds_name)));
          isl_map* shifted_xfer = isl_map_apply_range(inverse_xfer, time_shift);
          shifted_xfer = isl_map_reverse(shifted_xfer);
          TRACE(2) << "    shifted xfer: " << shifted_xfer << std::endl;

          update_ispace.at(cs_name).at(hlevel).at(ds_name) =
            isl_map_subtract(update_ispace.at(cs_name).at(hlevel).at(ds_name), shifted_xfer);
          TRACE(1) << "    delta xfer: " << update_ispace.at(cs_name).at(hlevel).at(ds_name) << std::endl;
        }
      }
    }
    }

#if 0
    //
    // Stage REUSE.2: Spatial reuse (multicast, spatial reduction).
    //
    //   Multicast is detected simply by inverting the child -> tensor relation
    //   at each parent.
    //
    TRACE(1) << "Beginning spatial reuse analysis." << std::endl;

    // For each hardware level...
    for (int hlevel = num_hardware_levels_-1; hlevel >= 1; hlevel--)
    {
      TRACE(1) << "Read programs for Hardware Level: " << hlevel << std::endl;

      // For each data space...
      for (string ds_name: problem_->DataSpaceNames())
      {
        if (read_ispace.at(hlevel).at(ds_name) != nullptr)
        {
          TRACE(2) << "  Data space: " << ds_name << std::endl;
          TRACE(2) << "    xfer: " << read_ispace.at(hlevel).at(ds_name) << std::endl;

          std::string parent_instance_name =Binding(bindings_, hlevel+1, ds_name);
          std::string child_instance_name = Binding(bindings_, hlevel, ds_name);

          std::stringstream data_space_index;
          std::size_t rank;
          for (rank = 0; rank < data_spaces_.at(ds_name).DataSpaceNumRanks(); rank++)
          {
            if (rank != 0)
              data_space_index << ",";
            data_space_index << "i" << rank;
          }

          // Whenever there is a spatial fanout between a parent and child, we need to
          // instantiate an intermediate "distributor" node.
          // We split the transfer program into two stages: (1) move data from parent level
          // to distributor, and (2) move data from distributor to child level.
          // The distributor is capable of multicasting a piece of data to multiple
          // children.
          // For simplicity and uniformity, we *always* instantiate a distributor even if
          // there isn't a multicast.

          // For parent to distributor, discard [s_c]. In the ReadFill implementation, we
          // will ignore tensor coords for the Fill because it is a FIFO and we ensure
          // (during schedule creation) that it is sync’d with the Read on the other side.

          // For distributor to child, don’t discard anything. However, in the ReadFill
          // implementation we will ignore Read tensor coords because it is a FIFO and we
          // ensure it is sync’d with the Fill on the other side.
          // Note that s_c has to be the innermost loop in the schedule.

          // Transform to: [[P[sp,tp] -> C[tc]] -> [sc]] -> Weights[k,r]
          // Then apply Domain Factor of Domain Product.
          char split_child_transform_str[256];
          sprintf(split_child_transform_str,
                  "{ [[%s[s_out,t_out] -> %s[s_in,t_in]] -> %s[%s]] -> [[[%s[s_out,t_out] -> %s[t_in]] -> [s_in]] -> %s[%s]] }",
                  parent_instance_name.c_str(),
                  child_instance_name.c_str(),
                  problem_->DataSpaceName(dsi).c_str(),
                  data_space_index.str().c_str(),

                  parent_instance_name.c_str(),
                  child_instance_name.c_str(),
                  problem_->DataSpaceName(dsi).c_str(),
                  data_space_index.str().c_str()
            );

          isl_map* split_child_transform = isl_map_read_from_str(context_, split_child_transform_str);
          isl_map* read_ispace_split = isl_set_unwrap(isl_set_apply(isl_map_wrap(isl_map_copy(read_ispace.at(hlevel).at(dsi))), split_child_transform));
          TRACE(2) << "    split xfer: " << read_ispace_split << std::endl;

          isl_map* read_ispace_distributor = isl_map_domain_factor_domain(read_ispace_split);
          TRACE(2) << "    distrib xfer: " << read_ispace_distributor << std::endl;

          auto x = isl_set_card(isl_map_wrap(isl_map_copy(read_ispace.at(hlevel).at(dsi))));
          auto y = isl_set_card(isl_map_wrap(isl_map_copy(read_ispace_distributor)));

          TRACE(2) << "    card(orig) = " << x << std::endl;
          TRACE(2) << "    card(dist) = " << y << std::endl;

          auto diff = isl_pw_qpolynomial_sub(x, y);
          diff = isl_pw_qpolynomial_drop_unused_params(diff);
          TRACE(2) << "      diff = " << diff << std::endl;

          // isl_size num_pieces = isl_pw_qpolynomial_n_piece(diff);
          // TRACE(2) << "        size = " << num_pieces << std::endl;

          // isl_stat stat = isl_pw_qpolynomial_foreach_piece(
          //   diff,
          //   [](__isl_take isl_set* set, __isl_take isl_qpolynomial* qp, void* user)
          //   {
          //     (void) set;
          //     (void) user;
          //     TRACE(2) << "        piece = " << qp << std::endl;
          //     return isl_stat_ok;
          //   },
          //   nullptr);

          // bool multicast_required = (num_pieces != 0);
          // if (multicast_required)
          // {
          //   distributor_input_ispace[hlevel][dsi] = read_ispace_distributor;
          //   // Next: create hardware data structures for distributor.
          // }
        }
      }
    }
#endif // 0

    TRACE(1) << "Reuse analysis complete." << std::endl;
    TRACE(1) << std::endl;
  }


  void GenerateCode(map<string, PHST> & PHSTs, bool generate_compute, Printer& p, Printer& q) {
    if (generate_compute) {
      for (auto it: problems_) {

        // ---------------------------------------------------------------------------------------
        // Stage CODEGEN.1: (emulation only) Print custom implementation of COMPUTE statement because
        //            we don't yet have a macro to support COMPUTEs.
        // ---------------------------------------------------------------------------------------

        // ---- Compute ----
        string cs_name = it.first;
        auto problem_ = problems_.at(cs_name);
        auto compute_program = it.second;
        //TODO: Secret handshake, this should be ok
        // as long as both compute function and compute call are both auto generated
        string compute_name =
          "COMPUTE_" + bindings_->ComputeBinding(cs_name) + "_" + cs_name;
        auto computePHST = PHSTs.at(bindings_->ComputeBindingFQ(cs_name));
        size_t t_dim = computePHST.GetComputeTDim(cs_name);
        CodegenCompute(problems_, data_spaces_, bindings_, cs_name, compute_name, t_dim, p);
      }
    }
    for (auto & it: PHSTs) {
      it.second.CodegenInit(context_, p, data_spaces_, val_save_programs_, val_check_programs_);
    }
    for (auto & it: PHSTs) {
      umap* schedule = rdmap(context_, "{}");
      it.second.ScheduleGen(schedule, problems_, data_spaces_);
      it.second.CodegenMemory(schedule, context_, p, q);
    }
  }

  //This is an optimization pass that merge operand into compute
  void MergeOperandIntoCompute(map<string, PHST> & PHST_map) {
    for (auto it: problems_) {
      string cs_name = it.first;
      string phst_name = bindings_->ComputeBindingFQ(cs_name);
      auto & comp_PHST = PHST_map.at(phst_name);
      //Get read op the level 1 binding for this Einsum
      for (string ds_name: it.second->GetReadDataSpace()) {
        string op_name = bindings_->MemBindingFQ(1, ds_name, cs_name);
        auto op_PHST = PHST_map.at(op_name);
        if (op_PHST.shrink_ispace_map.count(ds_name)) {
          comp_PHST.shrink_ispace_map.insert({ds_name,
            cpy(op_PHST.shrink_ispace_map.at(ds_name))});
          comp_PHST.shrink_src_map.insert({ds_name,
            op_PHST.shrink_src_map.at(ds_name)});
          comp_PHST.shrink_src_binding_map.insert({ds_name,
            op_PHST.shrink_src_binding_map.at(ds_name)});
        }
      }
      for (string ds_name: it.second->GetWriteDataSpace()) {
        string op_name = bindings_->MemBindingFQ(1, ds_name, cs_name);
        auto op_PHST = PHST_map.at(op_name);
        if (op_PHST.update_ispace_map.count(ds_name)) {
          comp_PHST.update_ispace_map.insert({ds_name,
            cpy(op_PHST.update_ispace_map.at(ds_name))});
          comp_PHST.update_dst_map.insert({ds_name,
            op_PHST.update_dst_map.at(ds_name)});
          comp_PHST.update_dst_binding_map.insert({ds_name,
            op_PHST.update_dst_binding_map.at(ds_name)});
        }
      }
    }

    //Second Pass remove all op PHST
    for (auto pit: problems_) {
      string cs_name = pit.first;
      for (auto ds_name: pit.second->DataSpaceNames()) {
        string op_name = bindings_->MemBindingFQ(1, ds_name, cs_name);
        if (PHST_map.count(op_name))
          PHST_map.erase(op_name);
      }
    }
    TRACE(1) << "======After Merging======" << endl;
    //Print out for debugging
    for (auto it: PHST_map) {
      TRACE(1) << it.second << endl;
    }
  }

  map<string, PHST> GeneratePHSTs(
    // Inputs.
    map<string, isl_map*>& init_ispace,
    map<string, isl_map*>& LLS_shrink_ispace,
    map<string, map<int, std::map<string, isl_map*>>>& read_ispace_map,
    map<string, map<int, std::map<string, isl_map*>>>& shrink_ispace,
    map<string, isl_map*>& compute_ispace_map,
    map<string, map<int, std::map<string, isl_map*>>>& update_ispace) {

      map<string, PHST> PHST_map;

      //get max ranks for codegen only
      std::size_t max_ranks = 0;
      for (auto it: problems_) {
        max_ranks = std::max(max_ranks, it.second->ComputeSpaceNumRanks());
      }
      for (auto it: data_spaces_)
      {
        max_ranks = std::max(max_ranks, it.second->DataSpaceNumRanks());
      }

      //unpack all the computation space
      for (auto it: problems_) {
        string cs_name = it.first;
        PHST phst;
        string phst_name = bindings_->ComputeBindingFQ(cs_name);
        string phst_binding = bindings_->ComputeBinding(cs_name);
        if (PHST_map.count(phst_name)) {
          phst = PHST_map.at(phst_name);
        } else {
          phst = PHST(context_, phst_name, phst_binding, max_ranks);
        }
        auto compute_ispace = cpy(compute_ispace_map.at(cs_name));
        auto problem_ = problems_.at(cs_name);
        //We need to intersect the ispace with original iteration space
        compute_ispace = isl_map_intersect_range(compute_ispace,
          isl_set_copy(problem_->IterationSpace()));
        phst.AddCompute(compute_ispace, cs_name);
        PHST_map[phst_name] = phst;
      }

      for (auto it: data_spaces_) {
        string ds_name = it.first;
        int lls_level = bindings_->LeastLevelStorage(ds_name)-1;
        for (int hlevel = lls_level; hlevel >= 0; hlevel --) {
          if (hlevel == lls_level) {
            string phst_name = bindings_->LLSBindingFQ(ds_name);
            string phst_binding = bindings_->LLSBinding(ds_name);
            PHST phst;
            if (PHST_map.count(phst_name)) {
              phst = PHST_map.at(phst_name);
            } else {
              phst = PHST(context_, phst_name, phst_binding, max_ranks);
              phst.SetLLS();
            }

            //Init ispace
            phst.AddInit(cpy(init_ispace.at(ds_name)), ds_name);

            //For the LLS shrink, DRAM does not have shrink
            if (LLS_shrink_ispace.count(ds_name)) {
              string src = bindings_->LLSBindingFQ(ds_name);
              string src_binding = bindings_->LLSBinding(ds_name);
              phst.AddShrink(cpy(LLS_shrink_ispace.at(ds_name)), ds_name, src, src_binding);
            }

            PHST_map[phst_name] = phst;

          } else {
            //Hlevel < lls_level
            //Hlevel = leastlevelstorage - 2

            //update
            for (string cs_name: it.second->GetWriteComputeSpaceNames()) {
              PHST phst;
              string  phst_name = bindings_->MemBindingFQ(hlevel+1, ds_name, cs_name);
              string  phst_binding = bindings_->MemBinding(hlevel+1, ds_name, cs_name);
              if (PHST_map.count(phst_name)) {
                phst = PHST_map.at(phst_name);
              } else {
                phst = PHST(context_, phst_name, phst_binding, max_ranks);
              }
              auto src2dst = std::make_pair(
                bindings_->MemBindingFQ(hlevel+1, ds_name, cs_name),
                bindings_->MemBindingFQ(hlevel+2, ds_name, cs_name));
              auto src2dst_binding = std::make_pair(
                bindings_->MemBinding(hlevel+1, ds_name, cs_name),
                bindings_->MemBinding(hlevel+2, ds_name, cs_name));
              phst.AddUpdate(cpy(update_ispace.at(cs_name).at(hlevel+1).at(ds_name)),
                ds_name, src2dst, src2dst_binding);

              PHST_map[phst_name] = phst;
            }
          }
          for (string cs_name : it.second->GetReadComputeSpaceNames()) {
            PHST phst;
            string phst_name = bindings_->MemBindingFQ(hlevel+1, ds_name, cs_name);
            string phst_binding = bindings_->MemBinding(hlevel+1, ds_name, cs_name);
            if (PHST_map.count(phst_name)) {
              phst = PHST_map.at(phst_name);
            } else {
              phst = PHST(context_, phst_name, phst_binding, max_ranks);
            }

            if (hlevel > 0) {
              //pack the binding information for codegen
              string dst = bindings_->MemBindingFQ(hlevel, ds_name, cs_name);
              string dst_binding =
                bindings_->MemBinding(hlevel, ds_name, cs_name);
              bool isIU =
                (update_ispace.at(cs_name).at(hlevel).at(ds_name) != nullptr);
              auto read_ispace = cpy(read_ispace_map.at(cs_name).at(hlevel).at(ds_name));
              phst.AddRead(read_ispace, cs_name, ds_name, dst, dst_binding, isIU);

            }
            if (hlevel < lls_level) {
              if (shrink_ispace.at(cs_name).at(hlevel + 1).at(ds_name) != nullptr) {
                //phst.shrink_ispace_map[ds_name] =
                //  cpy(shrink_ispace.at(cs_name).at(hlevel + 1).at(ds_name));
                string src = bindings_->MemBindingFQ(hlevel+1, ds_name, cs_name);
                string src_binding = bindings_->MemBinding(hlevel+1, ds_name, cs_name);
                phst.AddShrink(cpy(shrink_ispace.at(cs_name).at(hlevel+1).at(ds_name)),
                  ds_name, src, src_binding);
              }
            }
            PHST_map[phst_name] = phst;
          }
        }
      }
      //Print out for debugging
      for (auto it: PHST_map) {
        TRACE(1) << it.second << endl;
      }
      return PHST_map;
    }

  void PrintPHSTs(map<string, PHST> & PHSTs) {
    ofstream out("test_collaterals/" + test_name + "/phst.info");
    for (auto it: PHSTs) {
      out << "PHST: " << it.first << endl;
      out << DumpAccessCount(it.second) << endl;
    }
  }

  // -----------------------------------------------------------------------------------
  // Stage SCHED: Schedule creation.
  // -----------------------------------------------------------------------------------

  void GenerateSchedules(
    // Inputs.
    map<string, isl_map*>& init_ispace,
    map<string, map<int, std::map<string, isl_map*>>>& read_ispace,
    map<string, map<int, std::map<string, isl_map*>>>& shrink_ispace,
    map<string, isl_map*>& compute_ispace_map,
    map<string, map<int, std::map<string, isl_map*>>>& update_ispace,
    // Outputs.
    map<string, isl_union_map*>& init_programs,
    map<string, map<int, std::map<string, isl_union_map*>>>& read_programs,
    map<string, map<string, isl_union_map*>>& LLS_read_programs,
    map<string, map<int, std::map<string, isl_union_map*>>>& shrink_programs,
    map<string, isl_union_map*>& compute_programs,
    map<string, map<int, std::map<string, isl_union_map*>>>& update_programs)
    //char* compute_name)
  {
    TRACE(1) << "-------------------------" << std::endl;
    TRACE(1) << "    Schedule creation    " << std::endl;
    TRACE(1) << "-------------------------" << std::endl;

    // Before we begin, count the maximum rank across all compute and data spaces.
    // This is because we eventually want to serialize some data-movement blocks
    // with respect to each other. We do this using isomorphic schedule tuples,
    // which means we need to pad all schedule tuples to the same arity.
    // If we used schedule trees instead, we wouldn't need to do this. FIXME.
    std::size_t max_ranks = 0;
    for (auto it: problems_) {
      max_ranks = std::max(max_ranks, it.second->ComputeSpaceNumRanks());
    }
    for (auto it: data_spaces_)
    {
      max_ranks = std::max(max_ranks, it.second->DataSpaceNumRanks());
    }

    //
    // Stage SCHED.1: INIT blocks (emulation only).
    //

    TRACE(1) << "Init programs" << std::endl;

    // For each data space...
    for (auto & it: init_ispace)
    {
      string ds_name = it.first;
      TRACE(2) << "  Data space: " << ds_name << std::endl;

      // Set up the iteration spaces.
      isl_set* data_init_iteration_space = isl_map_wrap(cpy(init_ispace.at(ds_name)));
      data_init_iteration_space = isl_set_set_tuple_name(data_init_iteration_space, "__INIT");
      TRACE(2) << "    init iteration space: " << data_init_iteration_space << std::endl;

      // Create schedules. For now, use the identity relation as a schedule.
      isl_map* data_init_schedule = isl_set_identity(isl_set_copy(data_init_iteration_space));
      TRACE(2) << "    init schedule: " << data_init_schedule << std::endl;

      isl_map* data_init_program = isl_map_intersect_domain(data_init_schedule, data_init_iteration_space);

      // Create a "name" for the program, which contains enough information about
      // the target engine, src/dst buffets and tensors to trigger a statement
      // decoder macro on the target platform.
      char init_xfer_name[256];

      // INITs will be programmed onto engines at the last hardware level.
      //int hlevel = bindings_->LeastLevelStorage(ds_name) - 1;
      std::string init_engine_name = bindings_->LLSBindingFQ(ds_name);

      sprintf(init_xfer_name, "ACTION_INIT[@%s@, @%s@, @%s@, %lu]",
              init_engine_name.c_str(),
              //BindingFQ(bindings_, hlevel+1, ds_name).c_str(),
              init_engine_name.c_str(),
              ds_name.c_str(),
              data_spaces_.at(ds_name)->DataSpaceNumRanks());

      data_init_program = isl_map_set_tuple_name(data_init_program, isl_dim_in, init_xfer_name);
      TRACE(1) << "    init program: " << data_init_program << std::endl;

      init_programs[ds_name] = isl_union_map_from_map(isl_map_copy(data_init_program));

      // Create validation save and check programs from init.
      // TODO: implement this for data space
      if (data_spaces_.at(ds_name)->isOutput())
      {
        char val_save_xfer_name[256];
        char val_check_xfer_name[256];

        sprintf(val_save_xfer_name, "ACTION_INLINE_SAVE[@%s@, @%s@, @__val__@, @%s@, %lu]",
                init_engine_name.c_str(),
              //BindingFQ(bindings_, hlevel+1, ds_name).c_str(),
                init_engine_name.c_str(),
                ds_name.c_str(),
                data_spaces_.at(ds_name)->DataSpaceNumRanks());
        sprintf(val_check_xfer_name, "ACTION_INLINE_VALIDATE[@%s@, @%s@, @__val__@, @%s@, %lu]",
                init_engine_name.c_str(),
              //BindingFQ(bindings_, hlevel+1, ds_name).c_str(),
                init_engine_name.c_str(),
                ds_name.c_str(),
                data_spaces_.at(ds_name)->DataSpaceNumRanks());

        isl_map* val_save_program = isl_map_set_tuple_name(isl_map_copy(data_init_program), isl_dim_in, val_save_xfer_name);
        isl_map* val_check_program = isl_map_set_tuple_name(data_init_program, isl_dim_in, val_check_xfer_name);

        val_save_programs_[ds_name] = isl_union_map_from_map(val_save_program);
        val_check_programs_[ds_name] = isl_union_map_from_map(val_check_program);
      }
      //else
      //{
      //  val_save_programs_[ds_name] = nullptr;
      //  val_check_programs_[ds_name] = nullptr;
      //}
    }

    //Get all the intermediate buffer connect two Einsum
    //The read_prgroms need special serialization
    auto sorted_einsum = topological_sort_einsums(problems_);
    TRACE(1) << "Topological sorted einsums: " << sorted_einsum << std::endl;
    for (auto it: data_spaces_) {
      //Skip the boundary data space
      string ds_name = it.first;
      int hlevel = bindings_->LeastLevelStorage(ds_name) - 1;
      if (it.second->isInput() || it.second->isOutput() || (hlevel == num_hardware_levels_-1)) {
        continue;
      }
      //TODO use topological sort result in to create the schedule
      int cnt = 0;
      string last_cs_name;
      auto einsum_read = it.second->GetReadComputeSpaceNames();
      vector<string> einsum_read_sorted;
      set<string> einsum_read_set(einsum_read.begin(), einsum_read.end());
      for (auto it: sorted_einsum) {
        if (einsum_read_set.count(it)) {
          einsum_read_sorted.push_back(it);
        }
      }
      for (auto cs_name: einsum_read_sorted) {
        last_cs_name = cs_name;

        TRACE(2) << "  Hardware Level: " << hlevel << std::endl;

        assert(read_ispace.at(cs_name).at(hlevel).at(ds_name));
        {
          TRACE(2) << "  Data space: " << ds_name << std::endl;

          bool iu = (update_ispace.at(cs_name).at(hlevel).at(ds_name) != nullptr);

          // Set up the iteration spaces.
          isl_set* read_iteration_space = isl_map_wrap(read_ispace.at(cs_name).at(hlevel).at(ds_name));
          read_iteration_space = isl_set_set_tuple_name(read_iteration_space, "__READ");
          TRACE(2) << "    read iteration space: " << read_iteration_space << std::endl;

          // Create the schedule.

          // For now, use the identity relation as a schedule.
          // isl_map* read_schedule = isl_set_identity(isl_set_copy(read_iteration_space));
          // The above line and comment are deprecated, but we leave them here
          // for debugging purposes. We generate an explicit schedule now.

          // Determine a schedule to transfer the internal contents of a tile.
          // For now we'll simply use the *same* order as the iteration space.
          // So we'll use [i0,i1,...] for both RHS and LHS of the map we're
          // producing. If we wish to change this, we'll have to generate a new
          // different indexing for the RHS.
          std::stringstream data_space_index;
          std::size_t rank;
          for (rank = 0; rank < data_spaces_.at(ds_name)->DataSpaceNumRanks(); rank++)
          {
            if (rank != 0)
              data_space_index << ",";
            data_space_index << "i" << rank;
          }

          // Pad ranks in schedule.
          std::stringstream padded_indices;
          for (; rank < max_ranks; rank++)
          {
            if (rank != 0)
              padded_indices << ",";
            padded_indices << "0";
          }

          //TODO: Get the schedule from a global schedule of Einsum


          char sched_str[256];
          sprintf(sched_str,
                  // Note:
                  // (1) that the 0 is inserted to serialize read-IUs with updates.
                  //     We'll keep them in plain reads too -- they are harmless.
                  "{ %s[[%s[s_out,t_out] -> %s[s_in,t_in]] -> %s[%s]] -> [t_out,%d,t_in,%s%s,s_out,%d,s_in] }",
                  "__READ",
                  bindings_->MemBinding(hlevel+1, ds_name, cs_name).c_str(),
                  bindings_->MemBinding(hlevel, ds_name, cs_name).c_str(),
                  ds_name.c_str(),
                  data_space_index.str().c_str(),
                  cnt,
                  data_space_index.str().c_str(),
                  padded_indices.str().c_str(),
                  cnt);

          isl_map* schedule_projection = isl_map_read_from_str(context_, sched_str);
          TRACE(2) << "    read schedule projection: " << schedule_projection << std::endl;
          TRACE(2) << "    read iteration space: " << read_iteration_space << std::endl;

          isl_map* read_schedule = isl_map_intersect_domain(schedule_projection, isl_set_copy(read_iteration_space));
          TRACE(2) << "    read schedule: " << read_schedule << std::endl;

          isl_map* read_program = isl_map_intersect_domain(read_schedule, isl_set_copy(read_iteration_space));

          // Create a "name" for the program, which contains enough information about
          // the target engine, src/dst buffets and tensors to trigger a statement
          // decoder macro on the target platform.
          char read_xfer_name[256];

          // READs will be programmed onto engines at the source (outer) level.
          // However, to avoid over-serialization we also add the destination
          // name. Perhaps we should be adding the dataspace ID instead.
          std::string read_engine_name =
            bindings_->MemBindingFQ(hlevel+1, ds_name, cs_name);// + "->" +
          //mapping_->Binding(hlevel, problem_->DataSpaceName(dsi));

          sprintf(read_xfer_name, "ACTION_%s[@%s@, @%s@, @%s@, @%s@, %lu]",
                  iu ? "READ_IU" : "READ",
                  read_engine_name.c_str(),
                  bindings_->MemBindingFQ(hlevel+1, ds_name, cs_name).c_str(),
                  bindings_->MemBindingFQ(hlevel, ds_name, cs_name).c_str(),
                  ds_name.c_str(),
                  data_spaces_.at(ds_name)->DataSpaceNumRanks());

          read_program = isl_map_set_tuple_name(read_program, isl_dim_in, read_xfer_name);
          TRACE(1) << "    LLS read program: " << read_program << std::endl;

          LLS_read_programs[ds_name][cs_name] = isl_union_map_from_map(read_program);
          read_programs[cs_name][hlevel][ds_name] = nullptr;
          cnt ++;
        }
      }

      //For generating the shrink block

      if (shrink_ispace.at(last_cs_name).at(hlevel+1).at(ds_name) != nullptr) {
        TRACE(2) << "LLS: " << hlevel+1 << std::endl;
        TRACE(2) << "  Data space: " << ds_name << std::endl;
        TRACE(2) << "  shrink ispace: " <<shrink_ispace.at(last_cs_name).at(hlevel+1).at(ds_name) << std::endl;

        // Set up the iteration spaces.
        // Surprisingly, the space to be shrinked is the new init space that I am calculating
        // This data is saved in the shrink ispace at the proper level in init_tensor method
        auto shrink_space = shrink_ispace.at(last_cs_name).at(hlevel+1).at(ds_name);

        auto init_space = isl_map_wrap(cpy(shrink_space));
        //TRACE(2) << "  Init ispace: " << init_space << std::endl;
        const char* dummy_space = "{ [0,0] }";
        isl_set* root = isl_set_read_from_str(context_, dummy_space);
        isl_map* m = isl_map_from_domain_and_range(root, init_space);
        TRACE(2) << " tmp map: " << m << std::endl;
        TRACE(2) << " after curry: " << isl_map_uncurry(cpy(m)) << std::endl;
        isl_set* shrink_iteration_space = isl_map_wrap(isl_map_uncurry(m));
        //isl_set* shrink_iteration_space = isl_map_wrap(init_space);
        shrink_iteration_space = isl_set_set_tuple_name(shrink_iteration_space, "__SHRINK");
        TRACE(2) << "    shrink iteration space: " << shrink_iteration_space << std::endl;

        // Determine a schedule to shrink the internal contents of a tile.
        // For now we'll simply use the *same* order as the iteration space.
        // So we'll use [i0,i1,...] for both RHS and LHS of the map we're
        // producing. In future if we want to honor contiguous-SHRINK semantics
        // we'll probably have to generate a different indexing for the RHS
        // depending on the memory layout.
        std::stringstream data_space_index;
        std::size_t rank;
        for (rank = 0; rank < data_spaces_.at(ds_name)->DataSpaceNumRanks(); rank++)
        {
          if (rank != 0)
            data_space_index << ",";
          data_space_index << "i" << rank;
        }

        // Pad ranks in schedule.
        std::stringstream padded_indices;
        for (; rank < max_ranks; rank++)
        {
          if (rank != 0)
            padded_indices << ",";
          padded_indices << "0";
        }

        char sched_str[256];
        sprintf(sched_str,
                // Note:
                // (1) that the 2 is inserted to serialize shrinks with reads
                //     (and updates in the compute block).
                // (2) that similar to updates, we place the inner hardware
                //     level at the outer sched level. This is because we will
                //     serialize shrink blocks at level l+1 with read blocks
                //     at level l.
                // (3) The s_out,t_out aren't really useful for shrinks.
                "{ %s[[[0,0] ->%s[s_in,t_in] ]-> %s[%s]] -> [t_in,%d,0,%s%s,s_in,%d,0] }",
                "__SHRINK",
                bindings_->LLSBinding(ds_name).c_str(),
                ds_name.c_str(),
                data_space_index.str().c_str(),
                cnt,
                data_space_index.str().c_str(),
                padded_indices.str().c_str(),
                cnt);

        isl_map* schedule_projection = isl_map_read_from_str(context_, sched_str);
        TRACE(2) << "    shrink schedule projection: " << schedule_projection << std::endl;

        isl_map* shrink_schedule = isl_map_intersect_domain(schedule_projection, isl_set_copy(shrink_iteration_space));
        TRACE(2) << "    shrink schedule: " << shrink_schedule << std::endl;

        isl_map* shrink_program = isl_map_intersect_domain(shrink_schedule, shrink_iteration_space);

        // Create a "name" for the program, which contains enough information about
        // the target engine, src/dst buffets and tensors to trigger a statement
        // decoder macro on the target platform.
        char shrink_xfer_name[256];

        // Keep in mind the off-by-1 difference vs. READs: SHRINK blocks at level l+1
        // correspond to READ blocks at level l.
        // SHRINKs will be programmed onto engines at the *inner* level.
        // Ugh... SHRINKs for the innermost buffet level need to be programmed
        // on the compute level.
        std::string shrink_engine_name = bindings_->LLSBindingFQ(ds_name);

        sprintf(shrink_xfer_name, "ACTION_SHRINK[@%s@, @%s@, @%s@, %lu]",
                shrink_engine_name.c_str(),
                shrink_engine_name.c_str(),
                ds_name.c_str(),
                data_spaces_.at(ds_name)->DataSpaceNumRanks());

        shrink_program = isl_map_set_tuple_name(shrink_program, isl_dim_in, shrink_xfer_name);
        TRACE(1) << "    LLS shrink program: " << shrink_program << std::endl;
        LLS_read_programs[ds_name][last_cs_name + "_shrink"] = isl_union_map_from_map(shrink_program);
      }
    }

    for (auto it: compute_ispace_map) {
    auto cs_name = it.first;
    auto problem_ = problems_.at(cs_name);
    //
    // Stage SCHED.2: READ blocks.
    //

    // For each data space...
    for (string ds_name : problem_->DataSpaceNames())
    {
      // For each hardware level...
      TRACE(1) << "Read programs for data space: " << ds_name << std::endl;
      auto ds = data_spaces_.at(ds_name);
      int hlevel_ub;
      if (ds->isInput() || ds->isOutput()) {
        hlevel_ub = num_hardware_levels_;
      } else {
        //Skip the lls
        int lls = bindings_->LeastLevelStorage(ds_name);
        hlevel_ub =  (lls == num_hardware_levels_) ? num_hardware_levels_ : lls - 1;
      }

      for (int hlevel = hlevel_ub - 1; hlevel >= 1; hlevel--)
      {
        TRACE(2) << "  Hardware Level: " << hlevel << std::endl;

        if (read_ispace.at(cs_name).at(hlevel).at(ds_name) == nullptr)
        {
          read_programs[cs_name][hlevel][ds_name] = nullptr;
        }
        else
        {
          TRACE(2) << "  Data space: " << ds_name << std::endl;

          bool iu = (update_ispace.at(cs_name).at(hlevel).at(ds_name) != nullptr);

          // Set up the iteration spaces.
          isl_set* read_iteration_space = isl_map_wrap(read_ispace.at(cs_name).at(hlevel).at(ds_name));
          read_iteration_space = isl_set_set_tuple_name(read_iteration_space, "__READ");
          TRACE(2) << "    read iteration space: " << read_iteration_space << std::endl;

          // Create the schedule.

          // For now, use the identity relation as a schedule.
          // isl_map* read_schedule = isl_set_identity(isl_set_copy(read_iteration_space));
          // The above line and comment are deprecated, but we leave them here
          // for debugging purposes. We generate an explicit schedule now.

          // Determine a schedule to transfer the internal contents of a tile.
          // For now we'll simply use the *same* order as the iteration space.
          // So we'll use [i0,i1,...] for both RHS and LHS of the map we're
          // producing. If we wish to change this, we'll have to generate a new
          // different indexing for the RHS.
          std::stringstream data_space_index;
          std::size_t rank;
          for (rank = 0; rank < data_spaces_.at(ds_name)->DataSpaceNumRanks(); rank++)
          {
            if (rank != 0)
              data_space_index << ",";
            data_space_index << "i" << rank;
          }

          // Pad ranks in schedule.
          std::stringstream padded_indices;
          for (; rank < max_ranks; rank++)
          {
            if (rank != 0)
              padded_indices << ",";
            padded_indices << "0";
          }

          char sched_str[256];
          sprintf(sched_str,
                  // Note:
                  // (1) that the 0 is inserted to serialize read-IUs with updates.
                  //     We'll keep them in plain reads too -- they are harmless.
                  "{ %s[[%s[s_out,t_out] -> %s[s_in,t_in]] -> %s[%s]] -> [t_out,0,t_in,%s%s,s_out,0,s_in] }",
                  "__READ",
                  bindings_->MemBinding(hlevel+1, ds_name, cs_name).c_str(),
                  bindings_->MemBinding(hlevel, ds_name, cs_name).c_str(),
                  ds_name.c_str(),
                  data_space_index.str().c_str(),
                  data_space_index.str().c_str(),
                  padded_indices.str().c_str());

          isl_map* schedule_projection = isl_map_read_from_str(context_, sched_str);
          TRACE(2) << "    read schedule projection: " << schedule_projection << std::endl;
          TRACE(2) << "    read iteration space: " << read_iteration_space << std::endl;

          isl_map* read_schedule = isl_map_intersect_domain(schedule_projection, isl_set_copy(read_iteration_space));
          TRACE(2) << "    read schedule: " << read_schedule << std::endl;

          isl_map* read_program = isl_map_intersect_domain(read_schedule, isl_set_copy(read_iteration_space));

          // Create a "name" for the program, which contains enough information about
          // the target engine, src/dst buffets and tensors to trigger a statement
          // decoder macro on the target platform.
          char read_xfer_name[256];

          // READs will be programmed onto engines at the source (outer) level.
          // However, to avoid over-serialization we also add the destination
          // name. Perhaps we should be adding the dataspace ID instead.
          std::string read_engine_name =
            bindings_->MemBindingFQ(hlevel+1, ds_name, cs_name);// + "->" +
          //mapping_->Binding(hlevel, problem_->DataSpaceName(dsi));

          sprintf(read_xfer_name, "ACTION_%s[@%s@, @%s@, @%s@, @%s@, %lu]",
                  iu ? "READ_IU" : "READ",
                  read_engine_name.c_str(),
                  bindings_->MemBindingFQ(hlevel+1, ds_name, cs_name).c_str(),
                  bindings_->MemBindingFQ(hlevel, ds_name, cs_name).c_str(),
                  ds_name.c_str(),
                  data_spaces_.at(ds_name)->DataSpaceNumRanks());

          read_program = isl_map_set_tuple_name(read_program, isl_dim_in, read_xfer_name);
          TRACE(1) << "    read program: " << read_program << std::endl;

          read_programs[cs_name][hlevel][ds_name] = isl_union_map_from_map(read_program);
        }
      }
    }


    //
    // Stage SCHED.3: SHRINK blocks.
    //
    // For each data space...
    for (string ds_name : problem_->DataSpaceNames())
    {
      for (int hlevel = bindings_->LeastLevelStorage(ds_name) - 1; hlevel >= 1; hlevel--)
      // For each hardware level...
      {
        TRACE(1) << "Shrink programs for Hardware Level: " << hlevel << std::endl;
        if (shrink_ispace.at(cs_name).at(hlevel).at(ds_name) == nullptr)
        {
          shrink_programs[cs_name][hlevel][ds_name] = nullptr;
        }
        else
        {
          //TRACE(2) << "  shrink ispace: " << std::endl;
          //for (auto it: shrink_ispace.at(cs_name).at(hlevel)) {
          //  TRACE(2) << "ds_name: " <<it.first << ", " << str(it.second) << std::endl;
          //}
          TRACE(2) << "  Data space: " << ds_name << std::endl;

          // Set up the iteration spaces.
          isl_set* shrink_iteration_space = isl_map_wrap(shrink_ispace.at(cs_name).at(hlevel).at(ds_name));
          shrink_iteration_space = isl_set_set_tuple_name(shrink_iteration_space, "__SHRINK");
          TRACE(2) << "    shrink iteration space: " << shrink_iteration_space << std::endl;

          // Determine a schedule to shrink the internal contents of a tile.
          // For now we'll simply use the *same* order as the iteration space.
          // So we'll use [i0,i1,...] for both RHS and LHS of the map we're
          // producing. In future if we want to honor contiguous-SHRINK semantics
          // we'll probably have to generate a different indexing for the RHS
          // depending on the memory layout.
          std::stringstream data_space_index;
          std::size_t rank;
          for (rank = 0; rank < data_spaces_.at(ds_name)->DataSpaceNumRanks(); rank++)
          {
            if (rank != 0)
              data_space_index << ",";
            data_space_index << "i" << rank;
          }

          // Pad ranks in schedule.
          std::stringstream padded_indices;
          for (; rank < max_ranks; rank++)
          {
            if (rank != 0)
              padded_indices << ",";
            padded_indices << "0";
          }

          char sched_str[256];
          sprintf(sched_str,
                  // Note:
                  // (1) that the 2 is inserted to serialize shrinks with reads
                  //     (and updates in the compute block).
                  // (2) that similar to updates, we place the inner hardware
                  //     level at the outer sched level. This is because we will
                  //     serialize shrink blocks at level l+1 with read blocks
                  //     at level l.
                  // (3) The s_out,t_out aren't really useful for shrinks.
                  "{ %s[[%s[s_out,t_out] -> %s[s_in,t_in]] -> %s[%s]] -> [t_in,2,t_out,%s%s,s_in,2,s_out] }",
                  "__SHRINK",
                  bindings_->MemBinding(hlevel+1, ds_name, cs_name).c_str(),
                  bindings_->MemBinding(hlevel, ds_name, cs_name).c_str(),
                  ds_name.c_str(),
                  data_space_index.str().c_str(),
                  data_space_index.str().c_str(),
                  padded_indices.str().c_str());

          isl_map* schedule_projection = isl_map_read_from_str(context_, sched_str);
          TRACE(2) << "    shrink schedule projection: " << schedule_projection << std::endl;

          isl_map* shrink_schedule = isl_map_intersect_domain(schedule_projection, isl_set_copy(shrink_iteration_space));
          TRACE(2) << "    shrink schedule: " << shrink_schedule << std::endl;

          isl_map* shrink_program = isl_map_intersect_domain(shrink_schedule, shrink_iteration_space);

          // Create a "name" for the program, which contains enough information about
          // the target engine, src/dst buffets and tensors to trigger a statement
          // decoder macro on the target platform.
          char shrink_xfer_name[256];

          // Keep in mind the off-by-1 difference vs. READs: SHRINK blocks at level l+1
          // correspond to READ blocks at level l.
          // SHRINKs will be programmed onto engines at the *inner* level.
          // Ugh... SHRINKs for the innermost buffet level need to be programmed
          // on the compute level.
          std::string shrink_engine_name = (hlevel == 1) ?
            bindings_->ComputeBindingFQ(cs_name) :
            bindings_->MemBindingFQ(hlevel, ds_name, cs_name);

          sprintf(shrink_xfer_name, "ACTION_SHRINK[@%s@, @%s@, @%s@, %lu]",
                  shrink_engine_name.c_str(),
                  bindings_->MemBindingFQ(hlevel, ds_name, cs_name).c_str(),
                  ds_name.c_str(),
                  data_spaces_.at(ds_name)->DataSpaceNumRanks());

          shrink_program = isl_map_set_tuple_name(shrink_program, isl_dim_in, shrink_xfer_name);
          TRACE(1) << "    shrink program: " << shrink_program << std::endl;

          shrink_programs[cs_name][hlevel][ds_name] = isl_union_map_from_map(shrink_program);
        }
      }
    }

    //
    // Stage SCHED.4: Compute block.
    //
    auto compute_ispace = it.second;
    TRACE(2) << "  Compute: " << problem_->ComputeSpaceName() << std::endl;
    TRACE(2) << "    compute ispace: " <<  compute_ispace << std::endl;

    // Ugh... we needed to add this intersection, otherwise ISL gives unbounded optimum errors.
    compute_ispace = isl_map_intersect_range(compute_ispace, isl_set_copy(problem_->IterationSpace()));

    // Set up the iteration spaces.
    isl_set* compute_iteration_space = isl_map_wrap(compute_ispace);

    char compute_name[256];
    // Create a name for the program.
    sprintf(compute_name, "COMPUTE_%s_%s",
            bindings_->ComputeBinding(cs_name).c_str(),
            problem_->ComputeSpaceName().c_str());
    compute_iteration_space = isl_set_set_tuple_name(compute_iteration_space, compute_name);

    // Create the schedule.

    // We are experimenting with 2 different approaches:
    // 1. The operand reads, compute, and result writes are all encapsulated in a single statement.
    //    In this case, the emulation code for the statement will need to emit the algebra
    //    for the indexing expressions. This is weird because the expressions are already
    //    implicitly defined in the access relations. However, it's probably more representative
    //    of the hardware's behavior because it captures the atomic nature of the operand-read,
    //    compute and result-write steps.
    // 2. The operand reads, compute, and result writes are separate statements that need to be
    //    interleaved into a single sequential schedule for each compute unit. The good thing
    //    about this approach is that the indexing expressions are captured in the operand-read
    //    schedules, and are therefore reasoned about by ISL instead of having a second user-written
    //    form in the body. The weirdness with this approach is that the operand-read statements
    //    read the operands into magic state variables, which are then accessed by the compute
    //    statements.
    // We are going with approach 1.

    // Here is the final program that we are preparing. We're creating the interleaving
    // and union program data structures needed for approach 2 even though we're only
    // using approach 1.
    isl_union_map* compute_program = nullptr;

    // Begin by creating an identity relation.
    // isl_map* compute_schedule = isl_set_identity(isl_set_copy(compute_iteration_space));
    // The above line and comment are deprecated, but we leave them here
    // for debugging purposes. We generate an explicit schedule now.

    std::stringstream compute_space_index;
    std::size_t rank;
    for (rank = 0; rank < problem_->ComputeSpaceNumRanks(); rank++)
    {
      if (rank != 0)
        compute_space_index << ",";
      compute_space_index << "i" << rank;
    }

    // Pad ranks in schedule.
    std::stringstream padded_indices;
    for (; rank < max_ranks; rank++)
    {
      if (rank != 0)
        padded_indices << ",";
      padded_indices << "0";
    }

    char sched_str[512];
    sprintf(sched_str,
            "{ %s[%s[s,t] -> [%s]] -> [t,0,0,%s%s,s,0,0] }",
            compute_name,
            bindings_->ComputeBinding(cs_name).c_str(),
            compute_space_index.str().c_str(),
            compute_space_index.str().c_str(),
            padded_indices.str().c_str());

    isl_map* schedule_projection = isl_map_read_from_str(context_, sched_str);
    TRACE(2) << "    compute iteration space: " << compute_iteration_space << std::endl;
    TRACE(2) << "    compute schedule projection: " << schedule_projection << std::endl;

    isl_map* compute_schedule = isl_map_intersect_domain(schedule_projection, isl_set_copy(compute_iteration_space));
    TRACE(2) << "    compute schedule: " << compute_schedule << std::endl;

    // Add the computation schedule to the program.
    if (compute_program == nullptr)
      compute_program = isl_union_map_from_map(compute_schedule);
    else
      compute_program = isl_union_map_union(compute_program, isl_union_map_from_map(compute_schedule));
    compute_programs[cs_name] = compute_program;
    TRACE(1) << "    compute program: " << compute_program << std::endl;

    //
    // Stage SCHED.5: UPDATE blocks.
    //

    // For each data space...
    for (string ds_name : problem_->DataSpaceNames())
    {
      // For each hardware level... note that we reverse the hardware level traversal order
      // for updates, again strictly for emulation reasons.
      for (int hlevel = 1; hlevel < bindings_->LeastLevelStorage(ds_name); hlevel++)
      {
        TRACE(1) << "Update programs for Hardware Level: " << hlevel << std::endl;

        if (update_ispace.at(cs_name).at(hlevel).at(ds_name) == nullptr)
        {
          // No updates needed.
          update_programs[cs_name][hlevel][ds_name] = nullptr;
        }
        else
        {
          TRACE(2) << "  Data space: " << ds_name << std::endl;

          // Set up the iteration spaces.
          isl_set* update_iteration_space = isl_map_wrap(update_ispace.at(cs_name).at(hlevel).at(ds_name));
          update_iteration_space = isl_set_set_tuple_name(update_iteration_space, "__UPDATE");
          TRACE(2) << "    update iteration space: " << update_iteration_space << std::endl;

          // Create schedules. For now, use the identity relation as a schedule.
          // isl_map* update_schedule = isl_set_identity(isl_set_copy(update_iteration_space));

          // The above line and comment are deprecated, but we leave them here
          // for debugging purposes. We generate an explicit schedule now.

          // Determine a schedule to transfer the internal contents of a tile.
          // For now we'll simply use the *same* order as the iteration space.
          // So we'll use [i0,i1,...] for both RHS and LHS of the map we're
          // producing. If we wish to change this, we'll have to generate a new
          // different indexing for the RHS.
          std::stringstream data_space_index;
          std::size_t rank;
          for (rank = 0; rank < data_spaces_.at(ds_name)->DataSpaceNumRanks(); rank++)
          {
            if (rank != 0)
              data_space_index << ",";
            data_space_index << "i" << rank;
          }

          // Pad ranks in schedule.
          std::stringstream padded_indices;
          for (; rank < max_ranks; rank++)
          {
            if (rank != 0)
              padded_indices << ",";
            padded_indices << "0";
          }

          char sched_str[256];
          sprintf(sched_str,
                  // Note:
                  // (1) that the 1 is inserted to serialize read-IUs with updates.
                  "{ %s[[%s[s_out,t_out] -> %s[s_in,t_in]] -> %s[%s]] -> [t_in,1,t_out,%s%s,s_in,1,s_out] }",
                  "__UPDATE",
                  bindings_->MemBinding(hlevel+1, ds_name, cs_name).c_str(),
                  bindings_->MemBinding(hlevel, ds_name, cs_name).c_str(),
                  ds_name.c_str(),
                  data_space_index.str().c_str(),
                  data_space_index.str().c_str(),
                  padded_indices.str().c_str());

          isl_map* schedule_projection = isl_map_read_from_str(context_, sched_str);
          TRACE(2) << "    schedule projection: " << schedule_projection << std::endl;
          isl_map* update_schedule = isl_map_intersect_domain(schedule_projection, isl_set_copy(update_iteration_space));
          TRACE(2) << "    update schedule: " << update_schedule << std::endl;

          isl_map* update_program = isl_map_intersect_domain(update_schedule, update_iteration_space);

          // Create a "name" for the program, which contains enough information about
          // the target engine, src/dst buffets and tensors to trigger a statement
          // decoder macro on the target platform.
          char update_xfer_name[256];

          // UPDATEs will be programmed onto engines at the *inner* level.
          // Ugh... UPDATEs for the innermost buffet level need to be programmed
          // on the compute transfer level *if* serialize drains is set to true.
          std::string update_engine_name = (kSerializeDrains && hlevel == 1) ?
            bindings_->ComputeBindingFQ(problem_->ComputeSpaceName()) :
            bindings_->MemBindingFQ(hlevel, ds_name, cs_name);

          sprintf(update_xfer_name, "ACTION_UPDATE[@%s@, @%s@, @%s@, @%s@, %lu]",
                  update_engine_name.c_str(),
                  bindings_->MemBindingFQ(hlevel+1, ds_name, cs_name).c_str(),
                  bindings_->MemBindingFQ(hlevel, ds_name, cs_name).c_str(),
                  ds_name.c_str(),
                  data_spaces_.at(ds_name)->DataSpaceNumRanks());

          update_program = isl_map_set_tuple_name(update_program, isl_dim_in, update_xfer_name);
          TRACE(1) << "    update program: " << update_program << std::endl;

          update_programs[cs_name][hlevel][ds_name] = isl_union_map_from_map(update_program);
        }
      }
    }
    }

    TRACE(1) << "Schedule creation complete." << std::endl;
    TRACE(1) << std::endl;
  }




  // -----------------------------------------------------------------------------------
  // Stage CODEGEN: Code Generation.
  // -----------------------------------------------------------------------------------

  void GenerateCode(
    // Inputs.
    map<string, isl_union_map*>& init_programs,
    map<string, map<int, std::map<string, isl_union_map*>>>& read_program_map,
    map<string, map<string, isl_union_map*>>& LLS_read_program_map,
    map<string, map<int, std::map<string, isl_union_map*>>>& shrink_program_map,
    map<string, isl_union_map*>& compute_program_map,
    map<string, map<int, std::map<string, isl_union_map*>>>& update_program_map,
    Printer& p, Printer& q, bool generate_compute)
  {
    TRACE(1) << "-----------------------" << std::endl;
    TRACE(1) << "    Code generation    " << std::endl;
    TRACE(1) << "-----------------------" << std::endl;


    // ---------------------------------------------
    // Stage CODEGEN.2: Generate and print the ASTs.
    // ---------------------------------------------

    build_ = isl_ast_build_alloc(context_);

    //
    // Stage CODEGEN.2.1: INIT blocks (emulation only).
    //

    // For each data space...
    for (auto it : init_programs)
    {
      string ds_name = it.first;
      //int hlevel = bindings_->LeastLevelStorage(ds_name)-1;

      TRACE(1) << "Init program: " << it.second << std::endl;
      // Generate the AST.
      isl_ast_node* data_init_tree = isl_ast_build_node_from_schedule_map(build_, init_programs.at(ds_name));
      TRACE(2) << "data init tree: " << data_init_tree << std::endl;


      // Print the AST in C++ code form.
      char comment[256];
      sprintf(comment, "  // Program to init %s at %s.\n", ds_name.c_str(),
              bindings_->LLSBinding(ds_name).c_str());
      p << comment;

      p << data_init_tree << "\n";
    }

    //Stage CODGEN: LLS read block, where multi-einsum confluence
    for (auto it: LLS_read_program_map) {
      string ds_name = it.first;
      TRACE(1) << "LLS Read programs for data space: " << ds_name << std::endl;
      umap* merged_program = rdmap(context_, "{}");
      for (auto cs2program: it.second) {
        string cs_name = cs2program.first;
       TRACE(1) << "  Merging:" << std::endl;
       TRACE(1) << "    read  : " << cs2program.second << std::endl;
        merged_program = isl_union_map_union(cpy(cs2program.second), merged_program);
      }

      // Generate the AST.
      isl_ast_node* read_tree = isl_ast_build_node_from_schedule_map(build_, merged_program);

      // Print the AST in C++ code form.
      char comment[256];
      sprintf(comment, "  // Program to for LLS of %s.\n", ds_name.c_str());
      p << comment;

      p << read_tree << "\n";

      q << read_tree << "\n\n";

    }

    for (auto it: compute_program_map) {

    // ---------------------------------------------------------------------------------------
    // Stage CODEGEN.1: (emulation only) Print custom implementation of COMPUTE statement because
    //            we don't yet have a macro to support COMPUTEs.
    // ---------------------------------------------------------------------------------------

    // ---- Compute ----
    string cs_name = it.first;
    auto problem_ = problems_.at(cs_name);
    auto compute_program = it.second;
    string compute_name = domain_name(it.second);
    if (generate_compute)
    {
      CodegenCompute(problems_, data_spaces_, bindings_, cs_name, compute_name, 1, p);
    }
    // ---- End compute ----


    //Deref the programs
    auto read_programs = read_program_map.at(it.first);
    auto update_programs = update_program_map.at(it.first);
    auto shrink_programs = shrink_program_map.at(it.first);

    //
    // Stage CODEGEN.2.2: READ blocks.
    //

    // For each data space...
    for (string ds_name : problem_->DataSpaceNames())
    {
      // For each hardware level...
      TRACE(1) << "Read programs for data space: " << ds_name << std::endl;
      for (int hlevel = bindings_->LeastLevelStorage(ds_name) - 1; hlevel >= 1; hlevel--)
      {
      TRACE(1) << "Read programs for Hardware Level: " << hlevel << std::endl;

        if (read_programs.at(hlevel).at(ds_name) != nullptr)
        {
          // Serialize the drain/update path with reads?
          if (kSerializeDrains)
          {
            if (hlevel != bindings_->LeastLevelStorage(ds_name)-1 &&
                    update_programs.at(hlevel+1).at(ds_name) != nullptr)
              {
                TRACE(1) << "  Merging:" << std::endl;
                TRACE(1) << "    read  : " << read_programs.at(hlevel).at(ds_name) << std::endl;
                TRACE(1) << "    update: " << update_programs.at(hlevel+1).at(ds_name) << std::endl;

                read_programs.at(hlevel).at(ds_name) = isl_union_map_union(
                  read_programs.at(hlevel).at(ds_name),
                  update_programs.at(hlevel+1).at(ds_name));

                update_programs.at(hlevel+1).at(ds_name) = nullptr;
              }
          }

          // Serialize SHRINKs in an identical way as updates.
          if (hlevel != bindings_->LeastLevelStorage(ds_name)-1 &&
                  shrink_programs.at(hlevel+1).at(ds_name) != nullptr)
            {
              TRACE(1) << "  Merging:" << std::endl;
              TRACE(1) << "    read  : " << read_programs.at(hlevel).at(ds_name) << std::endl;
              TRACE(1) << "    shrink: " << shrink_programs.at(hlevel+1).at(ds_name) << std::endl;

              read_programs.at(hlevel).at(ds_name) = isl_union_map_union(
                read_programs.at(hlevel).at(ds_name),
                shrink_programs.at(hlevel+1).at(ds_name));

              shrink_programs.at(hlevel+1).at(ds_name) = nullptr;
            }

          // Generate the AST.
          isl_ast_node* read_tree = isl_ast_build_node_from_schedule_map(build_, read_programs.at(hlevel).at(ds_name));

          // Print the AST in C++ code form.
          char comment[256];
          sprintf(comment, "  // Program to read %s from %s into %s.\n", ds_name.c_str(),
                  bindings_->MemBinding(hlevel+1, ds_name, cs_name).c_str(),
                  bindings_->MemBinding(hlevel, ds_name, cs_name).c_str());
          p << comment;

          p << read_tree << "\n";

          q << read_tree << "\n\n";
        }
      }
    }

    //
    // Stage CODEGEN.2.3: COMPUTE blocks.
    //

    // Serialize the drain/update path with compute?
    if (kSerializeDrains)
    {
      int hlevel = 0;

      // For each data space...
      for (string ds_name : problem_->DataSpaceNames())
      {
        if (hlevel != num_hardware_levels_-1 && update_programs.at(hlevel+1).at(ds_name) != nullptr)
        {
          TRACE(2) << "  Merging:" << std::endl;
          TRACE(2) << "    compute  : " << str(compute_program) << std::endl;
          TRACE(2) << "    update: " << str(update_programs.at(hlevel+1).at(ds_name)) << std::endl;

          compute_program = isl_union_map_union(
            compute_program,
            update_programs.at(hlevel+1).at(ds_name));

          update_programs.at(hlevel+1).at(ds_name) = nullptr;
        }
      }
    }

    // Serialize shrinks.
    {
      int hlevel = 0;

      // For each data space...
      for (string ds_name : problem_->DataSpaceNames())
      {
        if (hlevel != num_hardware_levels_-1 && shrink_programs.at(hlevel+1).at(ds_name) != nullptr)
        {
          TRACE(2) << "  Merging:" << std::endl;
          TRACE(2) << "    compute  : " << compute_program << std::endl;
          TRACE(2) << "    shrink: " << shrink_programs.at(hlevel+1).at(ds_name) << std::endl;

          compute_program = isl_union_map_union(
            compute_program,
            shrink_programs.at(hlevel+1).at(ds_name));

          shrink_programs.at(hlevel+1).at(ds_name) = nullptr;
        }
      }
    }

    char comment[256];
    // Generate the AST.
    isl_ast_node* compute_tree = isl_ast_build_node_from_schedule_map(build_, compute_program); // isl_union_map_from_map(compute_program));

    // Print the AST in C++ code form.
    sprintf(comment, "  // Program to compute %s at %s.\n", problem_->ComputeSpaceName().c_str(),
            bindings_->ComputeBinding(problem_->ComputeSpaceName()).c_str());
    p << comment;

    p << compute_tree << "\n";

    q << compute_tree << "\n\n";

    //
    // Stage CODEGEN.2.4: UPDATE blocks.
    //

    // For each data space...
    for (string ds_name : problem_->DataSpaceNames())
    {
      // For each hardware level... note that we reverse the hardware level traversal order
      // for updates, again strictly for emulation reasons.
      for (int hlevel = 1; hlevel < bindings_->LeastLevelStorage(ds_name); hlevel++)
      {
        if (update_programs.at(hlevel).at(ds_name) != nullptr)
        {
          // Generate the AST.
          isl_ast_node* update_tree = isl_ast_build_node_from_schedule_map(build_, update_programs.at(hlevel).at(ds_name));

          // Print the AST in C++ code form.
          char comment[256];
          sprintf(comment, "  // Program to update %s into %s from %s.\n", ds_name.c_str(),
                  bindings_->MemBinding(hlevel+1, ds_name, cs_name).c_str(),
                  bindings_->MemBinding(hlevel, ds_name, cs_name).c_str());
          p << comment;

          p << update_tree << "\n";

          q << update_tree << "\n\n";
        }
      }
    }

    }
    //
    // All blocks have been emitted.
    //

    TRACE(1) << "Code generation complete." << std::endl;
    TRACE(1) << std::endl;
  }

  // ==============================
  // Main code-generation pipeline.
  // ==============================

  void Generate(Printer& p, Printer& q,
    bool generate_compute = true,
    bool reuse_analysis = true,
    bool dump_phsts = true)
  {
    map<string, isl_map*> compute_ispace_map;
    map<string, map<int, map<string, isl_map*>>> read_ispace_map;
    map<string, map<int, map<string, isl_map*>>> shrink_ispace_map;
    map<string, map<int, map<string, isl_map*>>> update_ispace_map;
    map<string, map<int, map<string, isl_map*>>> T_read_map;
    map<string, map<int, map<string, isl_map*>>> T_update_map;

    for (auto it: problems_) {
      std::map<int, isl_map*> T;
      std::map<int, std::map<string, isl_map*>> T_read;
      std::map<int, std::map<string, isl_map*>> T_update;

      GenerateTs(it.first, T, T_read, T_update); // Outputs.
      T_read_map[it.first] = T_read;
      T_update_map[it.first] = T_update;

      std::map<int, std::map<string, isl_map*>> read_ispace;
      std::map<int, std::map<string, isl_map*>> shrink_ispace;
      isl_map* compute_ispace;
      std::map<int, std::map<string, isl_map*>> update_ispace;

      Decouple(it.first,
        T, T_read, T_update, // Inputs.
        read_ispace, shrink_ispace, compute_ispace, update_ispace); // Outputs.
      read_ispace_map[it.first] = read_ispace;
      shrink_ispace_map[it.first] = shrink_ispace;
      update_ispace_map[it.first] = update_ispace;
      compute_ispace_map[it.first] = compute_ispace;
    }


    std::map<string, isl_map*> init_ispace, LLS_shrink_ispace;
    //For init the intermediate data space, we need to get the union of all operation
    Init_Tensor(init_ispace, LLS_shrink_ispace, shrink_ispace_map, T_read_map, T_update_map);


    if (reuse_analysis)
    {
      AnalyzeReuse(
        read_ispace_map, shrink_ispace_map, update_ispace_map); // Inputs and Outputs.
    }

    //Ispace map is saved as einsum_name->level_of_memory->data_space
    //But when we generate code for EDDO architecture it will be packed
    //into data transfer engine, which was the PHST
    //And that's basically the binding we create,
    //There should be a point when we unpack the information from ispace map
    //and generate the final program,
    //I think this unpack and repack into transfer engine centric-accessing
    //is the schedule creation function

    map<string, isl_union_map*> init_programs;
    map<string, map<int, map<string, isl_union_map*>>> read_programs;
    map<string, map<string, isl_union_map*>> LLS_read_programs;
    map<string, map<int, map<string, isl_union_map*>>> shrink_programs;
    map<string, isl_union_map*> compute_program;
    map<string, map<int, map<string, isl_union_map*>>> update_programs;
    //char compute_name[256];

    auto PHSTs = GeneratePHSTs(init_ispace, LLS_shrink_ispace, read_ispace_map, shrink_ispace_map, compute_ispace_map, update_ispace_map);

    //Optimiaztion passes
    //It's compulsory because we need the generated code match the architecture emulator
    MergeOperandIntoCompute(PHSTs);

    if(dump_phsts)
      PrintPHSTs(PHSTs);

    //Codegen Passes
    GenerateCode(PHSTs, generate_compute, p, q);


    //The Old Codegen That is deprecated
    //GenerateSchedules(
    //  init_ispace, read_ispace_map, shrink_ispace_map, compute_ispace_map, update_ispace_map, // Inputs.
    //  init_programs, read_programs, LLS_read_programs, shrink_programs, compute_program, update_programs); // Outputs.

    //GenerateCode(
    //  init_programs, read_programs, LLS_read_programs, shrink_programs, compute_program, update_programs, // Inputs.
    //  p, q, generate_compute);
  }

  // -----------------------------------
  // Print function call for printing latency.
  // -----------------------------------
  void PrintLatencyMethod(Printer& p)
  {
    // For each data space...
    for (auto it: data_spaces_)
    {
      string ds_name = it.first;
      if (val_save_programs_.count(ds_name))
      {

        char comment[256];
        sprintf(comment, "  // Program to save %s at %s.\n", ds_name.c_str(),
                bindings_->LLSBinding(ds_name).c_str());
        p << comment;

        p << "  arch.PrintLatency(\"" + bindings_->LLSBindingFQ(ds_name) + "\");"<< "\n";

      }
    }
  }

  // -----------------------------------
  // Print save-programs for validation.
  // -----------------------------------
  void PrintValidationSavePrograms(Printer& p)
  {
    // For each data space...
    for (auto it: data_spaces_)
    {
      string ds_name = it.first;
      if (val_save_programs_.count(ds_name))
      {

        isl_ast_build* build_ = isl_ast_build_alloc(context_);
        // Generate the AST.
        isl_ast_node* val_save_tree = isl_ast_build_node_from_schedule_map(build_, val_save_programs_.at(ds_name));

        // Print the AST in C++ code form.
        char comment[256];
        sprintf(comment, "  // Program to save %s at %s.\n", ds_name.c_str(),
                bindings_->LLSBinding(ds_name).c_str());
        p << comment;

        p << val_save_tree << "\n";

        isl_ast_build_free(build_);
      }
    }
  }

  // ------------------------------------
  // Print check-programs for validation.
  // ------------------------------------

  void PrintValidationCheckPrograms(Printer& p)
  {
    // For each data space...
    for (auto it: data_spaces_)
    {
      string ds_name = it.first;
      if (val_check_programs_.count(ds_name))
      {
        isl_ast_build* build_ = isl_ast_build_alloc(context_);
        // Generate the AST.
        isl_ast_node* val_check_tree = isl_ast_build_node_from_schedule_map(build_, val_check_programs_.at(ds_name));

        // Print the AST in C++ code form.
        char comment[256];
        sprintf(comment, "  // Program to check %s at %s.\n", ds_name.c_str(),
                bindings_->LLSBinding(ds_name).c_str());
        p << comment;

        p << val_check_tree << "\n";

        isl_ast_build_free(build_);
      }
    }
  }

};


void TenssellaCompile(isl_ctx* context, string name,
        map<string, ProblemPtr> & einsum_map,
        map<string, DataPtr> & data_space_map,
        map<string, MappingPtr> & mapping_,
        shared_ptr<Architecture> arch, shared_ptr<Binding> binding);

void CodegenCompute(map<string, ProblemPtr> & problems_,
              map<string, DataPtr> & data_spaces_,
		shared_ptr<Binding> bindings_, Printer& p);

void CodegenCompute(map<string, ProblemPtr> & problems_,
        map<string, DataPtr> & data_spaces_, shared_ptr<Binding> bindings_,
        string cs_name, string compute_name, Printer& p);

void GenerateReferenceCode(isl_ctx* context,
        map<string, ProblemPtr> & einsum_map,
        map<string, DataPtr> & data_space_map,
        Printer & p);

