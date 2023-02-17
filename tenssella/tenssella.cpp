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

#include "tenssella.hpp"

pair<int, int> getTimeDim(isl_map* ispace) {
    auto x_rel = isl_set_unwrap(domain(cpy(ispace)));
    return {num_in_dims(x_rel)-1, num_out_dims(x_rel)-1};
}

string time_vec_str(int num_dim, string suffix) {
    vector<string> t_vec;
    for (int i = 0; i < num_dim; i ++) {
        t_vec.push_back("t" + suffix + "_" + str(i));
    }
    return sep_list(t_vec, "", "", ",");
}

isl_map* linear_domain_map_with_index(isl_set* s, unordered_set<int> index) {
  string domain = name(s);
  int dim = num_dims(s);
  vector<string> var_names;
  vector<string> out_var_names;
  vector<string> exprs;
  isl_val* stride = one(ctx(s));
  int index_visited = 0;
  for (int i = 0; i < dim; i++) {
    string var = "d" + std::to_string(i);
    var_names.push_back(var);
    if (std::find(std::begin(index), std::end(index), i)
            != std::end(index)) {
      index_visited ++;
      string var = "d" + std::to_string(i);
      string stridestr = str(stride);
      exprs.push_back(stridestr + "*" + var);
      auto interval = project_all_but(s, dim-1-i);
      isl_val* extend = add(
              sub(lexmaxval(interval), lexminval(interval)),
              one(ctx(s)));
      stride = mul(stride, extend);
      if ((size_t)index_visited == index.size()) {
          out_var_names.push_back(sep_list(exprs, "", "", "+"));
      }
    } else {
        out_var_names.push_back("d" + std::to_string(i));
    }
  }
  //Check if we visited all dimension need to be merged
  assert((size_t)(index_visited) == index.size());
  std::reverse(var_names.begin(), var_names.end());
  std::reverse(out_var_names.begin(), out_var_names.end());

  string map_str = "{" + domain + sep_list(var_names, "[", "]", ", ") + " -> "
      + domain + sep_list(out_var_names, "[", "]", " ,") + " }";
  return isl_map_read_from_str(ctx(s), map_str.c_str());
}


void PackSaveAndValidation(
        string ds_name, size_t num_ranks,
        string name, isl_map* init_sched,
        map<string, umap*> & val_save_prog,
        map<string, umap*> & val_check_prog) {
  char val_save_xfer_name[256];
  char val_check_xfer_name[256];

  sprintf(val_save_xfer_name, "ACTION_INLINE_SAVE[@%s@, @%s@, @__val__@, @%s@, %lu]",
          name.c_str(),
          name.c_str(),
          ds_name.c_str(),
          num_ranks);
  sprintf(val_check_xfer_name, "ACTION_INLINE_VALIDATE[@%s@, @%s@, @__val__@, @%s@, %lu]",
          name.c_str(),
          name.c_str(),
          ds_name.c_str(),
          num_ranks);

  isl_map* val_save_program =
      isl_map_set_tuple_name(cpy(init_sched), isl_dim_in, val_save_xfer_name);
  isl_map* val_check_program =
      isl_map_set_tuple_name(cpy(init_sched), isl_dim_in, val_check_xfer_name);

  val_save_prog[ds_name] = isl_union_map_from_map(val_save_program);
  val_check_prog[ds_name] = isl_union_map_from_map(val_check_program);
}

void PHST::CodegenInit(isl_ctx* ctx, Printer& p, map<string, DataPtr> & data_spaces_,
        map<string, umap*> & val_save_prog, map<string, umap*> & val_check_prog) {

  isl_ast_build* build_ = isl_ast_build_alloc(ctx);
  for (auto & it: init_ispace_map) {
    string ds_name = it.first;
    isl_map* init_sched = SchedGenInit(ds_name, data_spaces_.at(ds_name));
    TRACE(2) << tab(2) << "=====init progrom schedule generated from phst====" << endl
        << tab(4) << init_sched << endl;

    //Also pack the schedule into Tenssella data save & validation for emulation
    if (data_spaces_.at(ds_name)->isOutput()) {
      auto num_ranks = data_spaces_.at(ds_name)->DataSpaceNumRanks();
      PackSaveAndValidation(ds_name, num_ranks, name,
            init_sched, val_save_prog, val_check_prog);
    }

    // Generate the AST.
    isl_ast_node* data_init_tree =
        isl_ast_build_node_from_schedule_map(build_, to_umap(init_sched));
    TRACE(2) << "data init tree: " << data_init_tree << std::endl;

    // Print the AST in C++ code form.
    char comment[256];
    sprintf(comment, "  // Program to init %s at %s.\n",
            ds_name.c_str(), binding.c_str());
    p << comment;

    p << data_init_tree << "\n";
  }
  isl_ast_build_free(build_);
}

void PHST::CodegenMemory(umap* sched, isl_ctx* ctx, Printer& p, Printer& q) {

  isl_ast_build* build_ = isl_ast_build_alloc(ctx);
  isl_ast_node* update_tree =
      isl_ast_build_node_from_schedule_map(build_, sched);

  // Print the AST in C++ code form.
  char comment[256];
  sprintf(comment, "  // Program for controller %s \n", name.c_str());
  p << comment;

  p << update_tree << "\n";

  q << update_tree << "\n\n";

  isl_ast_build_free(build_);
}

//A preprocessing pass the gather the max dimension of t_in
void PHST::Preprocessing() {
  for (auto & cs2it: read_ispace_map) {
    string cs_name = cs2it.first;
    for (auto & it: cs2it.second) {
      string ds_name = it.first;
      isl_map* rd_ispace = it.second;
      pair<int, int> t_dims = getTimeDim(rd_ispace);
      max_tin_dim = std::max(max_tin_dim, (size_t)t_dims.second);
    }
  }
}

//Function for Sched Generation
void PHST::ScheduleGen(umap* schedule,
        map<string, ProblemPtr>& problems_,
        map<string, DataPtr>& data_spaces_) {

  TRACE(1) << "=====Generate progrom for phst <" << name << ">====" << endl;

  Preprocessing();

  for (auto & it: compute_ispace) {
    string cs_name = it.first;
    isl_map* compute_sched = SchedGenCompute(cs_name, problems_.at(cs_name));
    TRACE(2) << tab(2) << "=====compute schedule generated from phst====" << endl
        << tab(4) << compute_sched << endl;
    schedule = isl_union_map_union(schedule, to_umap(compute_sched));
  }

  for (auto & cs2it: read_ispace_map) {
    string cs_name = cs2it.first;
    for (auto & it: cs2it.second) {
      string ds_name = it.first;
      isl_map* rd_sched = SchedGenRead(cs_name, ds_name, data_spaces_);
      TRACE(2) << tab(2) << "=====Read progrom schedule generated from phst====" << endl
          << tab(4) << rd_sched << endl;
      schedule = isl_union_map_union(schedule, to_umap(rd_sched));
    }
  }
  for (auto & it : update_ispace_map) {
    string ds_name = it.first;
    isl_map* ud_sched = SchedGenUpdate(ds_name, data_spaces_.at(ds_name));
    TRACE(2) << tab(2) << "=====update progrom schedule generated from phst====" << endl
        << tab(4) << ud_sched << endl;
    schedule = isl_union_map_union(schedule, to_umap(ud_sched));
  }
  //Adding a shrink here, it's possible that a compute PHST has multiple shrink operation
  int id = 0;
  for (auto & it: shrink_ispace_map) {
    string ds_name = it.first;
    isl_map* shrink_sched = SchedGenShrink(ds_name, data_spaces_.at(ds_name), id);
    TRACE(2) << tab(2) << "=====shrink progrom schedule generated from phst====" << endl
        << tab(4) << shrink_sched << endl;
    schedule = isl_union_map_union(schedule, to_umap(shrink_sched));
    id ++;
  }
  TRACE(1)<< endl;
}

isl_map* PHST::SchedGenCompute(string & cs_name, ProblemPtr problem_) {

  // Set up the iteration spaces.
  int t_dim = num_in_dims(compute_ispace.at(cs_name)) - 1;
  isl_set* compute_iteration_space = isl_map_wrap(cpy(compute_ispace.at(cs_name)));

  char compute_name[256];
  // Create a name for the program.
  sprintf(compute_name, "COMPUTE_%s_%s",
          binding.c_str(),
          cs_name.c_str());
  compute_iteration_space =
      isl_set_set_tuple_name(compute_iteration_space, compute_name);

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
  //isl_union_map* compute_program = nullptr;

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

  vector<string> pad(max_tin_dim, "0,");
  string t_in_padded = sep_list(pad, "", "", "");

  string t_str = time_vec_str(t_dim, "");

  char sched_str[512];
  sprintf(sched_str,
          "{ %s[%s[s,%s] -> [%s]] -> [%s,0,%s%s%s,s,0] }",
          compute_name,
          binding.c_str(),
          t_str.c_str(),
          compute_space_index.str().c_str(),
          t_str.c_str(),
          t_in_padded.c_str(),
          compute_space_index.str().c_str(),
          padded_indices.str().c_str());

  TRACE(2) << "    sched str: " << sched_str<< std::endl;
  isl_map* schedule_projection = isl_map_read_from_str(context_, sched_str);
  TRACE(2) << "    compute iteration space: " << compute_iteration_space << std::endl;
  TRACE(2) << "    compute schedule projection: " << schedule_projection << std::endl;

  isl_map* compute_schedule = isl_map_intersect_domain(schedule_projection, isl_set_copy(compute_iteration_space));
  TRACE(2) << "    compute schedule: " << compute_schedule << std::endl;

  // Add the computation schedule to the program.
  return compute_schedule;
}

//A helper function
isl_map* PHST::SchedGenRead(string& cs_name, string& ds_name, map<string, DataPtr> & data_spaces_) {

  TRACE(2) << "  Data space: " << ds_name << std::endl;

  auto t_dim = getTimeDim(read_ispace_map.at(cs_name).at(ds_name));

  // Set up the iteration spaces.
  isl_set* read_iteration_space = isl_map_wrap(cpy(read_ispace_map.at(cs_name).at(ds_name)));
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
  string t_out_str = time_vec_str(t_dim.first, "out");
  string t_in_str = time_vec_str(t_dim.second, "in");

  vector<string> pad(max_tin_dim - t_dim.second, "0,");
  string t_in_padded = sep_list(pad, "", "", "");

  char sched_str[256];
  sprintf(sched_str,
          // Note:
          // (1) that the 0 is inserted to serialize read-IUs with updates.
          //     We'll keep them in plain reads too -- they are harmless.
          //"{ %s[[%s[s_out,t_out] -> %s[s_in,t_in]] -> %s[%s]] -> [t_out,0,t_in,%s%s,s_out,0,s_in] }",
          "{ %s[[%s[s_out,%s] -> %s[s_in,%s]] -> %s[%s]] -> [%s,0,%s,%s%s%s,s_out,s_in] }",
          "__READ",
          binding.c_str(),
          t_out_str.c_str(),
          read_dst_binding_map.at(cs_name).at(ds_name).c_str(),
          t_in_str.c_str(),
          ds_name.c_str(),
          data_space_index.str().c_str(),
          t_out_str.c_str(),
          t_in_str.c_str(),
          t_in_padded.c_str(),
          data_space_index.str().c_str(),
          padded_indices.str().c_str());

  TRACE(2) << "    sched str: " << sched_str << endl;
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
  std::string read_engine_name = read_dst_map.at(cs_name).at(ds_name);
    //bindings_->MemBindingFQ(hlevel+1, ds_name, cs_name);// + "->" +
  //mapping_->Binding(hlevel, problem_->DataSpaceName(dsi));
  //
  bool iu = update_cs_name.count(make_pair(cs_name, ds_name));

  string stmt_suffix = t_dim.second > 1 ?
      ("_T" + str(t_dim.second)) : "";

  sprintf(read_xfer_name, "ACTION_%s%s[@%s@, @%s@, @%s@, @%s@, %lu]",
          iu ? "READ_IU" : "READ",
          stmt_suffix.c_str(),
          name.c_str(),
          name.c_str(), //PHST name
          read_engine_name.c_str(),
          ds_name.c_str(),
          data_spaces_.at(ds_name)->DataSpaceNumRanks());

  read_program = isl_map_set_tuple_name(read_program, isl_dim_in, read_xfer_name);
  TRACE(1) << "    read program: " << read_program << std::endl;
  return read_program;
}

isl_map* PHST::SchedGenUpdate(string & ds_name, DataPtr data_space) {

  TRACE(2) << "  Data space: " << ds_name << std::endl;

  pair<int, int> t_dim = getTimeDim(update_ispace_map.at(ds_name));

  // Set up the iteration spaces.
  isl_set* update_iteration_space = isl_map_wrap(cpy(update_ispace_map.at(ds_name)));
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
  for (rank = 0; rank < data_space->DataSpaceNumRanks(); rank++)
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

  string t_out_str = time_vec_str(t_dim.first, "out");
  string t_in_str = time_vec_str(t_dim.second, "in");

  vector<string> pad(max_tin_dim, "0,");
  string t_in_padded = sep_list(pad, "", "", "");

  char sched_str[256];
  sprintf(sched_str,
          // Note:
          // (1) that the 1 is inserted to serialize read-IUs with updates.
          // Original vector
          //"{ %s[[%s[s_out,t_out] -> %s[s_in,t_in]] -> %s[%s]] -> [t_in,1,t_out,%s%s,s_in, s_out] }",
          "{ %s[[%s[s_out,%s] -> %s[s_in,%s]] -> %s[%s]] -> [%s,1,%s%s%s,s_in, s_out] }",
          "__UPDATE",
          update_dst_binding_map.at(ds_name).second.c_str(),
          t_out_str.c_str(),
          update_dst_binding_map.at(ds_name).first.c_str(),
          t_in_str.c_str(),
          ds_name.c_str(),
          data_space_index.str().c_str(),
          t_in_str.c_str(),
          t_in_padded.c_str(),
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

  //TODO: need to check if we need to test app without serialization
  //FIXME: adding the serialization pass to merge L0 compute and L1 Memory
  //
  string stmt = t_dim.second > 1 ?
      ("ACTION_UPDATE_T" + str(t_dim.second)) : "ACTION_UPDATE";

  sprintf(update_xfer_name, "%s[@%s@, @%s@, @%s@, @%s@, %lu]",
          stmt.c_str(),
          name.c_str(),
          update_dst_map.at(ds_name).second.c_str(),
          update_dst_map.at(ds_name).first.c_str(),
          ds_name.c_str(),
          data_space->DataSpaceNumRanks());

  update_program = isl_map_set_tuple_name(update_program, isl_dim_in, update_xfer_name);
  TRACE(1) << "    update program: " << update_program << std::endl;
  return update_program;
}

isl_map* PHST::SchedGenInit(string & ds_name, DataPtr data_space) {
  TRACE(2) << "  Data space: " << ds_name << std::endl;

  // Set up the iteration spaces.
  int t_dim = num_in_dims(init_ispace_map.at(ds_name))-1;
  string stmt_suffix = t_dim> 1 ?
      "_T" + str(t_dim) : "";
  isl_set* data_init_iteration_space = isl_map_wrap(cpy(init_ispace_map.at(ds_name)));
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
  string init_method = data_space->isInput() ?
        "ACTION_INIT" : "ACTION_INIT_ZERO";
  init_method = init_method + stmt_suffix;

  sprintf(init_xfer_name, "%s[@%s@, @%s@, @%s@, %lu]",
          init_method.c_str(),
          name.c_str(),
          name.c_str(),
          ds_name.c_str(),
          data_space->DataSpaceNumRanks());

  data_init_program = isl_map_set_tuple_name(data_init_program, isl_dim_in, init_xfer_name);
  TRACE(1) << "    init program: " << data_init_program << std::endl;
  return data_init_program;
}

isl_map* PHST::SchedGenShrink(string& ds_name, DataPtr data_space, int shrink_id) {

  TRACE(2) << "  Data space: " << ds_name << std::endl;

  // Set up the iteration spaces.
  // TODO move this to the shrink ispace creation
  isl_set* rng;
  isl_map* shrink_map = cpy(shrink_ispace_map.at(ds_name));
  if (isl_map_can_curry(shrink_map)) {
    isl_map* curry_map = isl_map_curry(shrink_map);
    TRACE(2) << "origin map: " << shrink_ispace_map.at(ds_name)  << endl;
    TRACE(2) << "curry map: " << curry_map << endl;
    rng = range(curry_map);
  } else {
    rng = isl_map_wrap(shrink_map);
  }

  const char* dummy_space = "{ [0,0] }";
  isl_set* root = isl_set_read_from_str(context_, dummy_space);
  isl_map* m = isl_map_from_domain_and_range(root, rng);
  auto uncurry_m = isl_map_uncurry(m);
  TRACE(2) << "rewrite map: " << m << endl;
  pair<int, int> t_dim = getTimeDim(uncurry_m);

  isl_set* shrink_iteration_space = isl_map_wrap(uncurry_m);
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
  for (rank = 0; rank < data_space->DataSpaceNumRanks(); rank++)
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

  string t_vec = time_vec_str(t_dim.second, "in");

  vector<string> pad(max_tin_dim, "0,");
  string t_in_padded = sep_list(pad, "", "", "");

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
          // (4)Just change it to a dummy space ?
          //"{ %s[[[0,0]->%s[s_in,t_in]] -> %s[%s]] -> [t_in,2,0,%s%s,s_in,2,0] }",
          "{ %s[[[0,0]->%s[s_in,%s]] -> %s[%s]] -> [%s,%d,%s%s%s,s_in,0] }",
          "__SHRINK",
          shrink_src_binding_map.at(ds_name).c_str(),
          t_vec.c_str(),
          ds_name.c_str(),
          data_space_index.str().c_str(),
          t_vec.c_str(),
          2+shrink_id,
          t_in_padded.c_str(),
          data_space_index.str().c_str(),
          padded_indices.str().c_str());

  TRACE(2) << "    shrink schedule str: " << sched_str << std::endl;
  isl_map* schedule_projection = isl_map_read_from_str(context_, sched_str);
  TRACE(2) << "    shrink schedule projection: " << schedule_projection << std::endl;

  //FIXME: this iteration is not the same as the schedule
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

  string stmt = t_dim.second > 1 ?
      ("ACTION_SHRINK_T" + str(t_dim.second)) : "ACTION_SHRINK";
  sprintf(shrink_xfer_name, "%s[@%s@, @%s@, @%s@, %lu]",
          stmt.c_str(),
          name.c_str(),
          shrink_src_map.at(ds_name).c_str(),
          ds_name.c_str(),
          data_space->DataSpaceNumRanks());

  shrink_program = isl_map_set_tuple_name(shrink_program, isl_dim_in, shrink_xfer_name);
  TRACE(1) << "    shrink program: " << shrink_program << std::endl;
  return shrink_program;
}

std::string DumpAccessCount(PHST& phst) {
  std::ostringstream out;
  out << "---" << phst.name << std::endl;
  if (phst.isLLS) {
    assert(phst.init_ispace_map.size());
    for (auto it : phst.init_ispace_map) {
      out << " --init--" << endl;
      out << tab(2) << "data space: " << it.first << endl;
      out << tab(2) << "ispace: " << str(it.second) << endl;
    }
  }
  for (auto it: phst.update_ispace_map) {
    out << " --update--" << endl;
    out << tab(2) << "data space: " << it.first << endl;
    out << tab(2) << "ispace: " << str(it.second) << endl;
    auto t_dim = getTimeDim(it.second);
    out << "T_out dim: " << t_dim.first << ", T_in dim: " << t_dim.second << endl;
    int access_cnt = int_lower_bound(isl_union_set_card(to_uset(isl_map_wrap(cpy(it.second)))));
    out << tab(4) << " count: " << access_cnt << endl;
  }
  for (auto it: phst.shrink_ispace_map) {
    out << " --shrink--" << endl;
    out << tab(2) << "data space: " << it.first << endl;
    out << tab(2) << "ispace: " << str(it.second) << endl;
  }
  for (auto & cs2it: phst.read_ispace_map) {
    string cs_name = cs2it.first;
    out << " --read--" << endl;
    out << tab(2) << "cs_name: " << cs_name << endl;
    for (auto & it: cs2it.second) {
      string ds_name = it.first;
      if (phst.update_cs_name.count({cs_name, ds_name})) {
        out << tab(4) << "data space: " << ds_name << endl;
        out << tab(4) << "read w/t update ispace: " << str(it.second) << endl;
        int access_cnt = int_lower_bound(isl_union_set_card(to_uset(isl_map_wrap(cpy(it.second)))));
        out << tab(4) << "read IU count: " << access_cnt << endl;
      } else {
        out << tab(4) << "data space: " << ds_name << endl;
        out << tab(4) << "read-only ispace: " << str(it.second) << endl;
        int access_cnt = int_lower_bound(isl_union_set_card(to_uset(isl_map_wrap(cpy(it.second)))));
        out << tab(4) << "read count: " << access_cnt << endl;
      }
      auto t_dim = getTimeDim(it.second);
      out << "T_out dim: " << t_dim.first << ", T_in dim: " << t_dim.second << endl;
    }
  }
  for (auto it: phst.compute_ispace) {
    out << " --compute--" << endl;
    out << tab(2) << "cs_name: " << it.first << endl;
    out << tab(4) << "compute ispace: " << str(it.second)<< endl;
    int access_cnt = int_lower_bound(isl_union_set_card(to_uset(isl_map_wrap(cpy(it.second)))));
    out << tab(4) << "compute count: " << access_cnt << endl;
  }
  return out.str();
}

std::ostream& operator<<(std::ostream& out, const PHST& phst) {
  out << "---" << phst.name << std::endl;
  if (phst.isLLS) {
    assert(phst.init_ispace_map.size());
    for (auto it : phst.init_ispace_map) {
      out << " --init--" << endl;
      out << tab(2) << "data space: " << it.first << endl;
      out << tab(2) << "ispace: " << it.second << endl;
    }
  }
  for (auto it: phst.update_ispace_map) {
    out << " --update--" << endl;
    out << tab(2) << "data space: " << it.first << endl;
    out << tab(2) << "ispace: " << it.second << endl;
    auto t_dim = getTimeDim(it.second);
    out << "T_out dim: " << t_dim.first << ", T_in dim: " << t_dim.second << endl;
    int access_cnt = int_lower_bound(isl_union_set_card(to_uset(isl_map_wrap(cpy(it.second)))));
    out << tab(4) << " count: " << access_cnt << endl;
  }
  for (auto it: phst.shrink_ispace_map) {
    out << " --shrink--" << endl;
    out << tab(2) << "data space: " << it.first << endl;
    out << tab(2) << "ispace: " << it.second << endl;
  }
  for (auto & cs2it: phst.read_ispace_map) {
    string cs_name = cs2it.first;
    out << " --read--" << endl;
    out << tab(2) << "cs_name: " << cs_name << endl;
    for (auto & it: cs2it.second) {
      string ds_name = it.first;
      if (phst.update_cs_name.count({cs_name, ds_name})) {
        out << tab(4) << "data space: " << ds_name << endl;
        out << tab(4) << "read w/t update ispace: " << it.second << endl;
        int access_cnt = int_lower_bound(isl_union_set_card(to_uset(isl_map_wrap(cpy(it.second)))));
        out << tab(4) << "read IU count: " << access_cnt << endl;
      } else {
        out << tab(4) << "data space: " << ds_name << endl;
        out << tab(4) << "read-only ispace: " << it.second << endl;
        int access_cnt = int_lower_bound(isl_union_set_card(to_uset(isl_map_wrap(cpy(it.second)))));
        out << tab(4) << "read count: " << access_cnt << endl;
      }
      auto t_dim = getTimeDim(it.second);
      out << "T_out dim: " << t_dim.first << ", T_in dim: " << t_dim.second << endl;
    }
  }
  for (auto it: phst.compute_ispace) {
    out << " --compute--" << it.second<< endl;
    out << tab(2) << "cs_name: " << it.first << endl;
    out << tab(4) << "compute ispace: " << it.second<< endl;
    int access_cnt = int_lower_bound(isl_union_set_card(to_uset(isl_map_wrap(cpy(it.second)))));
    out << tab(4) << "compute count: " << access_cnt << endl;
  }
  return out;
}

void TenssellaCompile(isl_ctx* context,
        string test_name,
        map<string, ProblemPtr> & einsum_map,
        map<string, DataPtr> & data_space_map,
        map<string, MappingPtr> & mapping_,
        shared_ptr<Architecture> arch,
        shared_ptr<Binding> binding) {
  // == Print prelude.

  cmd("mkdir -p test_collaterals/" + test_name);
  Printer p(context, "test_collaterals/" + test_name + "/out.cpp");
  Printer q(context, "test_collaterals/" + test_name + "/out.ast", true);

  p << str_prelude;
  p << str_begin_main;

  // == Run the code generation.
  Tenssella test(context, test_name, arch, einsum_map, mapping_, data_space_map, binding);
  test.Generate(p, q, true/*generate compute*/, true/*reuse analysis*/);

  p << str_run_arch << "\n";
  test.PrintLatencyMethod(p);
  p << "  std::cerr << std::endl << \"=== TEST RUN COMPLETE ===\" << std::endl << std::endl;\n\n";
  test.PrintValidationSavePrograms(p);

  //Reference Codegen
  GenerateReferenceCode(context, einsum_map, data_space_map, p);
  p << str_run_arch << "\n";
  p << "  std::cerr << std::endl << \"=== REF RUN COMPLETE ===\" << std::endl << std::endl;\n\n";

  //Validation Codgen
  test.PrintValidationCheckPrograms(p);
  p << str_print_validation_result;
  p << "  std::cerr << std::endl << \"=== VALIDATION COMPLETE ===\" << std::endl << std::endl;\n\n";

  p << end_main;
}

shared_ptr<Binding> CreateRefBinding(map<string, DataPtr> & data_space_map,
        map<string, ProblemPtr> & einsum_map) {

  //Helper function  to create default one level CPU-Sytle binding

  auto b = make_shared<Binding>();
  size_t cnt = 0;
  for (auto it: einsum_map) {
      b->compute_binding_[it.first] = {"ALU", cnt};
      cnt ++;
  }
  cnt = 0;
  for (auto it: data_space_map) {
      for (auto p_it: einsum_map) {
          auto  einsum = p_it.second;
          if (einsum->ReadDataSpace(it.first) ||
                  einsum->WriteDataSpace(it.first)){
            b->memory_binding_[it.first][1][p_it.first] =
            {"Memory", cnt};
          }
      }
    cnt ++;
  }
  return b;
}


umap* naive_schedule(isl_ctx* ctx,
        map<string, ProblemPtr> & einsum_map) {

  vector<string> sorted_einsum = topological_sort_einsums(einsum_map);
  umap* global_schedule = rdmap(ctx, "{}");
  int cnt = 0;
  size_t max_iter_rank = 0;

  for (auto cs_name: sorted_einsum) {
    auto iter_vars = einsum_map.at(cs_name)->ComputeSpaceSubscripts();
    max_iter_rank = std::max(max_iter_rank, iter_vars.size());
  }
  TRACE(1)  << "Max rank: " << max_iter_rank << std::endl;

  for (auto cs_name: sorted_einsum) {
    auto iter_vars = einsum_map.at(cs_name)->ComputeSpaceSubscripts();
    vector<string> rhs_vec, lhs_vec;
    rhs_vec.push_back(str(cnt));
    for (string ivar: iter_vars) {
      lhs_vec.push_back(ivar);
      rhs_vec.push_back(ivar);
    }
    //Pad 0 to schedule vectors to make it the same depth
    for (size_t i = 0; i < max_iter_rank - iter_vars.size(); i ++ ){
      rhs_vec.push_back(str(0));
    }

    string sched_str = "{" + cs_name + sep_list(lhs_vec, "[", "]", ",")
        + "->" + sep_list(rhs_vec, "[", "]", ",") + "}";
    TRACE(1) << "schedule for einsum: " << cs_name << std::endl
        << tab(2) << sched_str << std:: endl;
    auto sched = rdmap(ctx, sched_str.c_str());
    auto iteration_domain = cpy(einsum_map.at(cs_name)->IterationSpace());
    iteration_domain = isl_set_set_tuple_id(iteration_domain, id(ctx, cs_name));
    auto udom = to_uset(iteration_domain);
    TRACE(1) << " uset domain: " << iteration_domain << std::endl;
    sched = isl_union_map_intersect_domain(sched, udom);
    global_schedule = isl_union_map_union(global_schedule, sched);
    cnt ++;
  }
  TRACE(1) << " naive schedule: " << global_schedule << std::endl;
  return global_schedule;
}

void generate_compute_code(umap* sched, map<string, ProblemPtr> & einsum_map, Printer& p) {

  string code_blk = codegen_c(sched);
  TRACE(1) << "schedule: " <<sched << std::endl;
  TRACE(1) << "Before rewrite: \n" <<code_blk<< std::endl;
  for (auto it: einsum_map) {
    string cs_name = it.first;
    std::regex re("(\n\\s+)" + cs_name+ "\\((.*)\\);");
    code_blk = regex_replace(code_blk, re, "$1ALU_" + cs_name + "(0, 0, $2);");
  }

  std::regex re_("(\n)");
  code_blk = regex_replace(code_blk, re_, "$1  ");
  code_blk = "  " + code_blk;

  TRACE(1) << "After rewrite: \n" << code_blk << std::endl;
  p << "  // Default Program for all Einsum computed in naive schedule "<< "\n";
  p << code_blk << "\n";
}

map<string, isl_set*> generate_naive_init(isl_ctx* ctx,
        map<string, ProblemPtr> & einsum_map,
        map<string, DataPtr> & data_space_map) {
  map<string, isl_set*> init_ispace;
  for (auto it: data_space_map) {
    auto buf = it.second;
    auto init_domain = rdset(ctx, "{}");
    for (string rd_einsum: buf->GetReadComputeSpaceNames()) {
      isl_map* access_map = buf->ReadProjection(rd_einsum);
      isl_set* iter_dom = einsum_map.at(rd_einsum)->IterationSpace();
      access_map = isl_map_intersect_domain(access_map, iter_dom);
      init_domain = isl_union_set_union(init_domain, to_uset(range(access_map)));
    }
    auto init_set = to_set(init_domain);
    TRACE(1) << "Init domain for data space: <"
        << it.first << "> is " << init_set << std::endl;
    init_ispace[it.first] = init_set;
  }
  return init_ispace;
}

void generate_init_code(isl_ctx* ctx,
        map<string, isl_set*> & init_ispace,
        map<string, DataPtr> & data_spaces,
        shared_ptr<Binding> naive_binding, Printer & p) {
  for (auto it: init_ispace) {
    string ds_name = it.first;
    vector<string> var_list;
    for (size_t i = 0; i < data_spaces.at(ds_name)->DataSpaceNumRanks(); i ++) {
      var_list.push_back("d" + str(i));
    }
    auto data_init_sched = to_map(rdmap(ctx,
            "{" + ds_name + sep_list(var_list, "[", "]", ",")
            + "->" +
            sep_list(var_list, "[", "]", ",") + "}" ) );
    data_init_sched =
        isl_map_intersect_domain(data_init_sched, it.second);
    char init_xfer_name[256];

    // INITs will be programmed onto engines at the last hardware level.
    //int hlevel = bindings_->LeastLevelStorage(ds_name) - 1;
    std::string init_engine_name = naive_binding->LLSBindingFQ(ds_name);
    string init_method = data_spaces.at(ds_name)->isInput() ?
        "ACTION_INIT" : "ACTION_INIT_ZERO";

    sprintf(init_xfer_name, "%s[@%s@, @%s@, @%s@, %lu]",
            init_method.c_str(),
            init_engine_name.c_str(),
            //BindingFQ(bindings_, hlevel+1, ds_name).c_str(),
            init_engine_name.c_str(),
            ds_name.c_str(),
            data_spaces.at(ds_name)->DataSpaceNumRanks());

    data_init_sched =
        isl_map_set_tuple_name(data_init_sched, isl_dim_in, init_xfer_name);

    string init_code_blk = codegen_c(to_umap(data_init_sched));
    TRACE(1) << "init code before replace: " <<  init_code_blk << std::endl;
    TRACE(1) << " xfer engine name: " << init_xfer_name << std::endl;

    //Add the default s, t index
    std::regex re("\\((.*)\\);");
    init_code_blk = regex_replace(init_code_blk, re, "(0, 0, $1);");

    //Some syntax transform
    std::regex re_("(\n\\s+)");
    init_code_blk = regex_replace(init_code_blk, re_, "$1  ");

    TRACE(1) << "init code after replace: " << init_code_blk << std::endl;
    // Print the AST in C++ code form.
    char comment[256];
    sprintf(comment, "  // Default Program to init %s at %s.\n", ds_name.c_str(),
            naive_binding->LLSBinding(ds_name).c_str());
    p << comment;
    p << tab(1) << init_code_blk << "\n";
  }
}

void GenerateReferenceCode(isl_ctx* context,
        map<string, ProblemPtr> & einsum_map,
        map<string, DataPtr> & data_space_map,
        Printer & p) {
  //The reference codegen
  umap* naive_sched = naive_schedule(context, einsum_map);
  auto naive_binding = CreateRefBinding(data_space_map, einsum_map);
  map<string, isl_set*> naive_init_domain =
      generate_naive_init(context, einsum_map, data_space_map);

  //Codegen
  generate_init_code(context, naive_init_domain, data_space_map, naive_binding, p);

  //Generate all the compute functions
  CodegenCompute(einsum_map, data_space_map, naive_binding, p);
  generate_compute_code(naive_sched, einsum_map, p);

}

void CodegenCompute(map<string, ProblemPtr> & problems_,
      map<string, DataPtr> & data_spaces_,
      shared_ptr<Binding> bindings_, Printer& p) {
  for (auto it: problems_) {
    string cs_name = it.first;

    string compute_name = "ALU_" + cs_name;
    CodegenCompute(problems_, data_spaces_, bindings_,
            cs_name, compute_name, 1 /*by default always has 1D time*/, p);
  }
}

void CodegenCompute(map<string, ProblemPtr> & problems_,
        map<string, DataPtr> & data_spaces_, shared_ptr<Binding> bindings_,
        string cs_name, string compute_name, size_t t_dim, Printer& p) {

  // ---- Compute ----
  //string cs_name = it.first;
  auto problem_ = problems_.at(cs_name);
  //auto compute_program = it.second;
  {

    char comment[256];
    string cs_name = problem_->ComputeSpaceName();

    sprintf(comment, "  // Compute a single %s at %s.\n", problem_->ComputeSpaceName().c_str(),
            bindings_->ComputeBinding(problem_->ComputeSpaceName()).c_str());
    p << comment;

    vector<string> t_args;
    for (size_t i = 0; i < t_dim; i ++) {
      t_args.push_back("int _t" + str(i) + ", ");
    }

    p << "  auto " << compute_name << " = [](int _s, " + sep_list(t_args, "", "", "");
    auto subscripts = problem_->ComputeSpaceSubscripts();
    for (auto subscript = subscripts.begin(); subscript != subscripts.end(); subscript++)
    {
      if (subscript != subscripts.begin())
        p << ", ";
      p << "int " << *subscript;
    }
    p << ")\n";
    p << "  {\n";

    for (size_t i = 0; i < t_dim; i ++) {
      p << "    (void) _t" + str(i) + ";\n";
    }
    p << "\n";

    for (string ds_name : problem_->DataSpaceNames())
    {
      if (problem_->ReadDataSpace(ds_name))
      {
        p << "    std::stringstream operand_tensor_point_" << ds_name << ";\n";
        p << "    operand_tensor_point_" << ds_name << " << \"" << ds_name << "\"";
        //for (auto& expr: data_spaces_.at(ds_name)->ReadProjectionTxt(cs_name))
        //{
          //TRACE(2) << "\tDEBUG:: << \"_\" << " << expr << std::endl;
        //}
        for (auto& expr: data_spaces_.at(ds_name)->ReadProjectionTxt(problem_->ComputeSpaceName()))
        {
          p << " << \"_\" << " << expr;
        }
        p << ";\n";
      }

      if (problem_->WriteDataSpace(ds_name))
      {
        p << "    std::stringstream result_tensor_point_" << ds_name << ";\n";
        p << "    result_tensor_point_" << ds_name << " << \"" << ds_name << "\"";
        for (auto& expr : data_spaces_.at(ds_name)->WriteProjectionTxt(problem_->ComputeSpaceName()))
        {
          p << " << \"_\" << " << expr;
        }
        p << ";\n";
      }
    }
    p << "    \n";

    p << "    Action<float> action;\n";
    p << "    action.op = Op::COMPUTE;\n";

    for (string ds_name : problem_->DataSpaceNames())
    {
      if (problem_->ReadDataSpace(ds_name))
      {
        bool iu = problem_->WriteDataSpace(ds_name);
        p << "    action.srcs.push_back({_s, \""
          << bindings_->MemBindingFQ(1 /*ugh*/, ds_name, cs_name)
          << "\", operand_tensor_point_" << ds_name << ".str(), "
          << (iu ? "true" : "false")
          << " });\n";
      }
    }

    p << "    action.transform = [](std::vector<float>& operands, std::vector<float>& results)\n";
    p << "    {\n";
    p << problem_->GetComputeSpace().transform_txt;
    p << "    };\n";

    for (string ds_name : problem_->DataSpaceNames())
    {
      if (problem_->WriteDataSpace(ds_name))
      {
        p << "    action.dsts.push_back({_s, \""
          <<  bindings_->MemBindingFQ(1 /*ugh*/, ds_name, cs_name)
          << "\", result_tensor_point_" <<ds_name << ".str(), false });\n";
      }
    }

    p << "    \n";
    p << "    arch[\"" << bindings_->ComputeBindingFQ(problem_->ComputeSpaceName())
      << "\"](_s).AddAction(action);\n";

    p << "  };\n\n";

  }

}
