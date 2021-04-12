/* Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

#include "loop-analysis/sparse-analysis.hpp"
#include "mapping/arch-properties.hpp"
#include "model/topology.hpp"

namespace sparse {

void CollectCompletePointSetsAndSubnests(const problem::Workload *workload,
										 const Mapping &mapping,
										 std::vector <std::vector<problem::OperationSpace>> &maxtile_point_sets,
										 std::vector <std::vector<loop::Descriptor>> &subnests) {

  problem::OperationPoint origin;
  problem::OperationPoint dimension_sizes;
  dimension_sizes.IncrementAllDimensions(); // initialize to { 1, 1, 1... }

  maxtile_point_sets.push_back({});
  subnests.push_back({});

  unsigned tiling_level = 0;
  auto &loops = mapping.complete_loop_nest.loops;
  for (unsigned loop_level = 0; loop_level < loops.size(); loop_level++)
  {
	auto &loop = loops[loop_level];
	dimension_sizes[loop.dimension] *= ceil((loop.end - loop.start) / loop.stride);

	// origin gives us the low corner (inclusive) of the operation space.
	// dimension_sizes gives the high corner (exclusive) of the operation space.
	// We need the inclusive high corner to build the operation space. See
	// OperationSpace constructor for details.
	problem::OperationPoint high = dimension_sizes;
	high.IncrementAllDimensions(-1);
	problem::OperationSpace maxtile(workload, origin, high);
	maxtile_point_sets[tiling_level].push_back(maxtile);
	subnests[tiling_level].push_back(loop);

	if (loop_level == mapping.complete_loop_nest.storage_tiling_boundaries.at(tiling_level))
	{
	  maxtile_point_sets.push_back({});
	  subnests.push_back({});
	  tiling_level++;
	}
  }

}

bool DefineFormatModelsViaMapping(const problem::Workload *workload,
								  const Mapping &mapping,
								  tiling::CompoundDataMovementNest &compound_data_movement_nest,
								  CompressionInfo &compression_info,
								  const model::Topology::Specs &topology_specs,
								  std::vector <model::EvalStatus> &eval_status,
								  const bool break_on_failure) {

  std::vector <std::vector<problem::OperationSpace>> maxtile_point_sets = {};
  std::vector <std::vector<loop::Descriptor>> subnests = {};

  CollectCompletePointSetsAndSubnests(workload, mapping, maxtile_point_sets, subnests);

  bool success = true;

  // nothing needs to be done if no metadata involved
  if (compression_info.all_ranks_default_dense) return success;

  std::ostringstream fail_reason;

  for (unsigned pv = 0; pv < problem::GetShape()->NumDataSpaces; pv++)
  {

	auto &pv_data_movement_nest = compound_data_movement_nest[pv];
	auto dim_ids_in_proj = problem::GetShape()->DataSpaceIDToDimensionIDVector[pv];

	unsigned level = 0;
	while (level < topology_specs.NumStorageLevels())
	{

	  if (pv_data_movement_nest[level].shape == 0)
	  {
		// have not seen the first level that storages the datatype pv yet
		level++;
		continue;
	  }

	  // get the architecture level of the storage level
	  unsigned overall_level_id = topology_specs.StorageMap(level);

	  // collapse all the bypassed levels between parent and child into metadata subnests
	  std::uint64_t child_level_id = pv_data_movement_nest[level].child_level;

	  // handle special case: no child level
	  // then we need to accumulate all levels below (if any)
	  std::uint64_t inner_most_level = child_level_id == std::numeric_limits<unsigned>::max() ? 0 : child_level_id + 1;

	  // check if pre-tiling is required for this level
	  // if so, current level hardware must support the necessary number of ranks associated with the metadata subnest of both this level and its child level
	  // thus, we need to add all the metadata subnest in the child level to this level
	  bool pre_tiling_required = false;
	  bool cur_level_has_metadata = compression_info.has_metadata_masks.at(level).at(pv);
	  bool cur_level_compressed = compression_info.compressed_masks.at(level).at(pv);
	  // if current level is default uncompressed, it has one rank that all rankIDs are flattened into a single rank
	  unsigned cur_level_num_ranks = cur_level_has_metadata ?
									 compression_info.per_level_info_map.at(level).at(pv).rank_formats.size() : 1;

	  bool child_level_has_metadata = false;
	  unsigned child_level_num_ranks = 0;
	  bool child_level_compressed = false;

	  if (child_level_id != std::numeric_limits<unsigned>::max())
	  {
		child_level_has_metadata = compression_info.has_metadata_masks.at(child_level_id).at(pv);
		child_level_compressed = compression_info.compressed_masks.at(child_level_id).at(pv);
		child_level_num_ranks = child_level_has_metadata ?
								compression_info.per_level_info_map.at(child_level_id).at(pv).rank_formats.size() : 1;
	  }

	  // Pretiling checks
	  // only if current level is compressed && current level is not inner most level
	  // && no online tile partition supported, do we need pre-tiling
	  if (cur_level_compressed && child_level_id != std::numeric_limits<unsigned>::max()
		  && !compression_info.tile_partition_supported_masks[level])
	  {

		pre_tiling_required = true;

		if (child_level_num_ranks > cur_level_num_ranks)
		{
		  fail_reason << "pretiling for " << problem::GetShape()->DataSpaceIDToName.at(pv)
					  << "required but level compression format does not align: "
					  << topology_specs.GetStorageLevel(level)->level_name << " "
					  << topology_specs.GetStorageLevel(child_level_id)->level_name
					  << std::endl;

		  success = false;
		  eval_status[overall_level_id].success = false;
		  eval_status[overall_level_id].fail_reason = fail_reason.str();
		  if (break_on_failure) return success;

		}
		for (unsigned r_id = 0; r_id < child_level_num_ranks; r_id++)
		{
		  if (child_level_compressed &&
			  compression_info.per_level_info_map.at(level).at(pv).rank_formats[r_id] !=
				  compression_info.per_level_info_map.at(child_level_id).at(pv).rank_formats[r_id])
		  {
			fail_reason << " pretiling for " << problem::GetShape()->DataSpaceIDToName.at(pv)
						<< "required but level compression format does not align: "
						<< topology_specs.GetStorageLevel(level)->level_name << " "
						<< topology_specs.GetStorageLevel(child_level_id)->level_name
						<< std::endl;
			success = false;
			eval_status[overall_level_id].success = false;
			eval_status[overall_level_id].fail_reason = fail_reason.str();
			if (break_on_failure) return success;
		  }
		}
	  }

	  std::uint64_t inner_rank_id = pre_tiling_required ? child_level_num_ranks : 0;
	  std::uint64_t loop_id = 0;
	  std::vector <loop::Descriptor> flattened_rank_nest;
	  std::set <problem::Shape::DimensionID> flattening_rule;

	  // singleton subnests for current level and bypassed level
	  std::vector <loop::Descriptor> singleton_metadata_subnest;
	  std::vector <std::uint64_t> singleton_metadata_subtile_shape;

	  // Go through the corresponding storage levels to retrieve info
	  for (int l = level; l >= int(inner_most_level); l--)
	  {
		for (int loop_id = subnests[l].size() - 1; loop_id >= 0; loop_id--)
		{
		  auto loop = subnests[l][loop_id];
		  bool trivial_loop = (loop.start + loop.stride) >= loop.end;

		  // pick out loops that are relevant (and non-trivial if pre-tiling is required)
		  if (dim_ids_in_proj.find(loop.dimension) != dim_ids_in_proj.end() &&
			  (!pre_tiling_required || (pre_tiling_required && !trivial_loop)))
		  {
			// pv_data_movement_nest[level].metadata_subnest.insert(pv_data_movement_nest[level].metadata_subnest.begin(), loop);
			singleton_metadata_subnest.insert(singleton_metadata_subnest.begin(), loop);
			auto subtile_shape = maxtile_point_sets[l][loop_id].GetSize(pv);
			// pv_data_movement_nest[level].metadata_subtile_shape.insert(pv_data_movement_nest[level].metadata_subtile_shape.begin(), subtile_shape);
			singleton_metadata_subtile_shape.insert(singleton_metadata_subtile_shape.begin(), subtile_shape);
		  }
		}
	  }

	  if (!pre_tiling_required)
	  {

		// without pretiling, the subnests associated with the current tile
		// must be bounded to the include all subtile sizes in the inner levels
		// only looking at the subnest bounds at the current level is not enough

		// 1) collect all inner level subnests to get the global bound for each dimension
		// FIXME: temporal and spatial loop bounds might get multiplied together and
		//  the loop type will be set to whichever that is at the top,
		//  this behavior does not affect the correctness in terms of fiber tree construction,
		//  but making it cleaner would be helpful
		problem::PerProblemDimension <std::uint64_t> dimension_sizes;
		problem::PerProblemDimension <std::uint64_t> residual_sizes;
		for (unsigned i = 0; i < dimension_sizes.size(); i++)
		{
		  dimension_sizes[i] = 1;
		  residual_sizes[i] = 1;
		}

		for (unsigned i = 0; i < inner_most_level; i++)
		{
		  auto subnest = subnests[i];
		  for (auto loop = subnest.begin(); loop != subnest.end(); loop++)
		  {
			dimension_sizes[loop->dimension] *= ceil((loop->end - loop->start) / loop->stride);
			residual_sizes[loop->dimension] *= loop->residual_end;
		  }
		}

		// 2) scale the current level bound accordingly
		// note that after this step, there can be trivial loops left
		for (unsigned subnest_id = 0; subnest_id < singleton_metadata_subnest.size(); subnest_id++)
		{
		  auto &loop = singleton_metadata_subnest[subnest_id];
		  loop.end = loop.end * loop.stride * dimension_sizes[loop.dimension];
		  loop.residual_end = loop.residual_end * residual_sizes[loop.dimension];

		  // if there are two loops of the same dim in cur level subnest, the upper loop will not be scaled
		  dimension_sizes[loop.dimension] = 1;
		  residual_sizes[loop.dimension] = 1;
		}
	  }

	  // 1) Get rid of potential trivial loops for non-pretiled cases
	  // 2) Flatten necessary loops according to flattening rule
	  loop_id = 0;
	  for (auto r_id = inner_rank_id; r_id < cur_level_num_ranks; r_id++)
	  {
		assert(loop_id < singleton_metadata_subnest.size());
		flattened_rank_nest.clear();
		bool trivial_loop = true;

		while (trivial_loop
			&& loop_id < singleton_metadata_subnest.size())  //get rid of the first set of consecutive trivial loops
		{
		  auto loop = singleton_metadata_subnest[loop_id];
		  trivial_loop = (loop.start + loop.stride) >= loop.end;
		  if (!trivial_loop) break;
		  loop_id++;
		}

		if (loop_id < singleton_metadata_subnest.size())
		{
		  auto loop = singleton_metadata_subnest[loop_id];
		  flattened_rank_nest.push_back(loop);
		  loop_id++;
		  if (!cur_level_has_metadata  // current level default uncompressed, then all loops must be falttened into 1 rank
			  || compression_info.per_level_info_map.at(level).at(pv).FoundDimensionInFlatteningRule(r_id,
																									 loop.dimension,
																									 flattening_rule))
		  {
			while (loop_id < singleton_metadata_subnest.size()
				&& (!cur_level_has_metadata
					|| (flattening_rule.find(singleton_metadata_subnest[loop_id].dimension)
						!= flattening_rule.end())))
			{
			  auto loop = singleton_metadata_subnest[loop_id];
			  trivial_loop = (loop.start + loop.stride) >= loop.end;
			  if (!trivial_loop)
			  {
				flattened_rank_nest.push_back(loop);
			  }
			  loop_id++;
			}
		  }
		}
		pv_data_movement_nest[level].metadata_subnest.push_back(flattened_rank_nest);
		pv_data_movement_nest[level].metadata_subtile_shape.push_back(singleton_metadata_subtile_shape[loop_id - 1]);
	  }

	  // skip the trailing trivial loops (if any)
	  bool more_compression_ranks_needed = false;
	  if (loop_id < singleton_metadata_subnest.size())
	  {
		while (loop_id < singleton_metadata_subnest.size())
		{
		  auto loop = singleton_metadata_subnest[loop_id];
		  bool trivial_loop = (loop.start + loop.stride) >= loop.end;
		  if (!trivial_loop)
		  {
			// if any non-trivial loops occurs, then the hardware support is not compatible
			more_compression_ranks_needed = true;
			break;
		  }
		  loop_id++;
		}
	  }

	  if (more_compression_ranks_needed)
	  {

		fail_reason << "more compression ranks needed than supported in hardware("
					<< singleton_metadata_subnest.size() - loop_id << " loop(s) unmapped)"
					<< " dataspace name: " << problem::GetShape()->DataSpaceIDToName.at(pv);
		success = false;
		eval_status[overall_level_id].success = false;
		eval_status[overall_level_id].fail_reason = fail_reason.str();
		if (break_on_failure) return success;
	  }

	  // sanity check for flattening procedure
	  assert(!more_compression_ranks_needed && loop_id == singleton_metadata_subnest.size());

	  if (pre_tiling_required)
	  {
		// pre-tiling requires the tile to maintain the order and bounds of the metadata subnests from its child level
		for (int loop_id = pv_data_movement_nest[child_level_id].metadata_subnest.size() - 1; loop_id >= 0; loop_id--)
		{
		  auto loop = pv_data_movement_nest[child_level_id].metadata_subnest[loop_id];
		  pv_data_movement_nest[level].metadata_subnest.insert(pv_data_movement_nest[level].metadata_subnest.begin(),
															   loop);

		  auto subtile_shape = pv_data_movement_nest[child_level_id].metadata_subtile_shape[loop_id + 1];
		  pv_data_movement_nest[level].metadata_subtile_shape.insert(pv_data_movement_nest[level].metadata_subtile_shape.begin(),
																	 subtile_shape);
		}
		// subtile shape must have one more element than subtile nest
		// see assert below for more
		pv_data_movement_nest[level].metadata_subtile_shape.insert(pv_data_movement_nest[level].metadata_subtile_shape.begin(),
																   pv_data_movement_nest[child_level_id].metadata_subtile_shape[0]);
	  } else
	  {
		pv_data_movement_nest[level].metadata_subtile_shape.insert(pv_data_movement_nest[level].metadata_subtile_shape.begin(),
																   1);
	  }

	  // validity check on if the required number of ranks == number of hardware supported ranks
	  if (pv_data_movement_nest[level].metadata_subnest.size() != cur_level_num_ranks)
	  {

		fail_reason << "metadata models cannot be defined, dataspace name: "
					<< problem::GetShape()->DataSpaceIDToName.at(pv);
		//for (unsigned rank_id = 0; rank_id < pv_data_movement_nest[level].metadata_subnest.size(); rank_id++)
		//{
		//     std::cout << " --- flattened loops --- " << std::endl;
		//  for (auto loop = pv_data_movement_nest[level].metadata_subnest[rank_id].begin();
		//	   loop != pv_data_movement_nest[level].metadata_subnest[rank_id].end(); loop++)
		//  {
		//	   std::cout << *loop << std::endl;
		//  }
		//}
		success = false;
		eval_status[overall_level_id].success = false;
		eval_status[overall_level_id].fail_reason = fail_reason.str();
		if (break_on_failure) return success;
	  }

	  // calculate the fiber shapes at each rank of metadata
	  for (unsigned rank_id = 0; rank_id < cur_level_num_ranks; rank_id++)
	  {

		problem::OperationPoint origin;
		problem::OperationPoint dimension_sizes;
		dimension_sizes.IncrementAllDimensions(); // initialize to { 1, 1, 1... }

		// going through the flattened ranks (if size is 1, then there is not flattened ranks
		for (auto loop = pv_data_movement_nest[level].metadata_subnest[rank_id].begin();
			 loop != pv_data_movement_nest[level].metadata_subnest[rank_id].end(); loop++)
		{
		  dimension_sizes[loop->dimension] *= ceil((loop->end - loop->start) / loop->stride);
		}

		// project to operation space to get fiber shape
		problem::OperationPoint high = dimension_sizes;
		high.IncrementAllDimensions(-1);
		problem::OperationSpace offset_tile(workload, origin, high);
		compound_data_movement_nest[pv][level].fiber_shape.push_back(offset_tile.GetSize(pv));
	  }

	  // subtile shape must have one more element than subtile nest
	  // as it includes the tile size of the child level:
	  //     important for compressed metadata models to get the prob of empty coordinates in the last level of metadata
	  assert(pv_data_movement_nest[level].metadata_subnest.size() + 1
				 == pv_data_movement_nest[level].metadata_subtile_shape.size());

	  // print info for sanity checks

	  //  std::cout << "\nDataspace name: " << problem::GetShape()->DataSpaceIDToName.at(pv)
	  //			<< "  level: " << level << " " << topology_specs.GetStorageLevel(level)->level_name
	  //			<< "   pretiling required: " << pre_tiling_required
	  //			<< " compressed: " << compression_info.compressed_masks[level][pv] << std::endl;
	  //  for (unsigned i = 0; i < pv_data_movement_nest[level].metadata_subnest.size(); i++)
	  //  {
	  //	std::cout << " ----- rank: " << i << " ------" << std::endl;
	  //	if (compression_info.compressed_masks[level][pv])
	  //	  std::cout << "   rank format: " << compression_info.per_level_info_map.at(level).at(pv).rank_formats[i]
	  //				<< std::endl;
	  //	std::cout << "   rank tile shape: " << pv_data_movement_nest[level].metadata_subtile_shape[i + 1] << std::endl;
	  //	std::cout << "   fiber shape: " << pv_data_movement_nest[level].fiber_shape[i] << std::endl;
	  //	std::cout << "   flattened nests: " << pv_data_movement_nest[level].metadata_subnest[i].size() << std::endl;

	  //	for (auto iter = pv_data_movement_nest[level].metadata_subnest[i].begin();
	  //		 iter != pv_data_movement_nest[level].metadata_subnest[i].end(); iter++)
	  //	{
	  //	  std::cout << "\t" << *iter << std::endl;
	  //	}
	  //  }
	  // look at parent directly in the next round, as we know the levels in the middle are bypassed
	  auto parent_level = pv_data_movement_nest[level].parent_level;
	  level = parent_level;
	}
  }
  return success;

}

bool CheckFormatModelsAndMapping(const tiling::NestOfCompoundMasks &masks,
								 const CompressionInfo &compression_info,
								 const model::Topology::Specs &topology_specs,
								 std::vector <model::EvalStatus> &eval_status,
								 const bool break_on_failure) {

  bool success = true;

  if (compression_info.all_ranks_default_dense) return success;

  for (unsigned pv = 0; pv < unsigned(problem::GetShape()->NumDataSpaces); pv++)
  {
	bool parent_level_compressed = true;
	int parent_level_num_ranks = -1;
	unsigned parent_level = topology_specs.NumStorageLevels() - 1;

	for (int level = topology_specs.NumStorageLevels() - 1; level >= 0; level--)
	{

	  bool mask = masks[level][pv];
	  if (!mask)
	  {
		continue;
	  }

	  // parent-most level for pv
	  if (parent_level_num_ranks == -1 && parent_level_compressed)
	  {
		parent_level_compressed = compression_info.compressed_masks[level][pv];

		// for (auto iter = compression_info.per_level_info_map.at(level).begin(); iter != compression_info.per_level_info_map.at(level).end(); iter++)
		// {
		//   std::cout << iter->first << ", " << iter->second.rank_formats.size()<< std::endl;
		// }

		if (parent_level_compressed)
		{
		  parent_level_num_ranks = compression_info.per_level_info_map.at(level).at(pv).rank_formats.size();
		}
		parent_level = level;
		continue;
	  }

	  // intermediate storage levels
	  bool cur_level_compressed = compression_info.compressed_masks.at(level).at(pv);
	  int cur_level_num_ranks = -1;
	  if (cur_level_compressed)
	  {
		cur_level_num_ranks = compression_info.per_level_info_map.at(level).at(pv).rank_formats.size();
	  }

	  assert((cur_level_compressed && cur_level_num_ranks > 0)
				 || (!cur_level_compressed && cur_level_num_ranks == -1));

	  // get the overall architecture level id for current storage level and its parent level
	  auto overall_level_id = topology_specs.StorageMap(level);
	  auto overall_parent_level_id = topology_specs.StorageMap(parent_level);

	  if (parent_level_compressed && !cur_level_compressed && !compression_info.decompression_supported_masks[level])
	  {
		success = false;
		eval_status[overall_level_id].success = false;
		eval_status[overall_level_id].fail_reason = "decompression (from parent level) needed but not supported";
		if (break_on_failure)
		{ return success; }
	  }


	  // std::cout << "parent level compressed: " << parent_level_compressed
	  // << " cur level compressed: " << cur_level_compressed << " compression support mask: "
	  // << compression_info.compression_supported_masks[level] << std::endl;
	  if (problem::GetShape()->IsReadWriteDataSpace.at(pv) &&
		  parent_level_compressed && !cur_level_compressed && !compression_info.compression_supported_masks[level])
	  {
		success = false;
		eval_status[overall_level_id].success = false;
		eval_status[overall_level_id].fail_reason = "compression (to parent level) needed but not supported";
		if (break_on_failure)
		{ return success; }
	  }

	  if (parent_level_compressed && cur_level_compressed && parent_level_num_ranks < cur_level_num_ranks
		  && !compression_info.tile_partition_supported_masks[parent_level])
	  {
		success = false;
		eval_status[overall_parent_level_id].success = false;
		eval_status[overall_parent_level_id].fail_reason =
			"runtime partition needed but not supported (NOT IMPLEMENTED YET)";
		if (break_on_failure)
		{ return success; }
	  }

	  // prepare for next round of checks
	  parent_level_num_ranks = cur_level_num_ranks;
	  parent_level_compressed = cur_level_compressed;
	  parent_level = level;
	}
  }
  return success; // no level failed
}
}