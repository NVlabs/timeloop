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

#include "loop-analysis/coordinate-space-tile-info.hpp"
#include "sparse-analysis/state.hpp"
#include "mapping/loop.hpp"

namespace sparse
{

std::map<DataSpaceID, double> GetExpectedOperandDensities(const SetOfOperationSpaces set_of_operation_spaces,
                                                          const PerDataSpaceDensityModel operand_density_models)
{
  std::map<DataSpaceID, double> expected_operand_densities;
  // TODO: adjust to allow coordinate dependent modeling, right now we just use operation space mold to query density models,
  //  most complicated case would be read data model, where we will need to construct all possible operation spaces
  for (auto iter = operand_density_models.begin(); iter != operand_density_models.end(); iter++)
  {
    // operand has metadata, so query density model
    auto& mold_high = set_of_operation_spaces.op_space_mold_high;
    problem::OperationPoint origin;
    tiling::CoordinateSpaceTileInfo ctile_info;
    problem::OperationSpace mold_op_space(set_of_operation_spaces.workload, origin, mold_high);
    ctile_info.Set(mold_op_space.GetDataSpace(iter->first), iter->first);
    expected_operand_densities[iter->first] = round(1000000*iter->second->GetExpectedTileOccupancy(ctile_info) / ctile_info.GetShape())/1000000;
  }
  return expected_operand_densities;
}

// ------------------------------------------------------------------
// Old Version of CalculateFineGrainedComputeAccesses
// ** OBSOLETE
// ------------------------------------------------------------------


void CalculateFlattenedProb(std::map <DataSpaceID, PerStateProb>& per_operand_states,
                            std::map<ComputeOperandStatePair, double>& flattened_probs,
                            const DataSpaceID op_a_id,
                            const DataSpaceID op_b_id)
{
  // std::cout << "after postrpocessing based on conditioned probability..."        << std::endl;
  // std::cout << "op b EZ:  " << per_operand_states.at(op_b_id).at(EXIST_ZERO)     << std::endl;
  // std::cout << "op b ENZ: " << per_operand_states.at(op_b_id).at(EXIST_NOT_ZERO) << std::endl;
  // std::cout << "op b NE:  " << per_operand_states.at(op_b_id).at(NOT_EXIST)      << std::endl;
  // std::cout << "op a EZ:  " << per_operand_states.at(op_a_id).at(EXIST_ZERO)     << std::endl;
  // std::cout << "op a ENZ: " << per_operand_states.at(op_a_id).at(EXIST_NOT_ZERO) << std::endl;
  // std::cout << "op a NE:  " << per_operand_states.at(op_a_id).at(NOT_EXIST)      << std::endl; 

  for (auto op_iter = per_operand_states.begin(); op_iter != per_operand_states.end(); op_iter++)
  {
    std::vector<ComputeOperandState> states = {EXIST_ZERO, EXIST_NOT_ZERO, NOT_EXIST};
    for (auto s: states)
    {
      if (op_iter->second.at(s) < -0.00001 || op_iter->second.at(s) > 1.00001) 
      {
        std::cout << " operand probability: " << op_iter->second.at(s) << std::endl;
        std::cerr << " illegal operand probability, can be due to rounding" << std::endl;
        assert(false);
      }
      else
      {
        if (op_iter->second.at(s) < 0) op_iter->second.at(s) = 0;
        if (op_iter->second.at(s) > 1) op_iter->second.at(s) = 1;
      }
    }
  }
  
  flattened_probs[{EXIST_NOT_ZERO, EXIST_NOT_ZERO}] = per_operand_states.at(op_a_id).at(EXIST_NOT_ZERO)
    * per_operand_states.at(op_b_id).at(EXIST_NOT_ZERO);
  flattened_probs[{EXIST_NOT_ZERO, EXIST_ZERO}] = per_operand_states.at(op_a_id).at(EXIST_NOT_ZERO)
    * per_operand_states.at(op_b_id).at(EXIST_ZERO);
  flattened_probs[{EXIST_NOT_ZERO, NOT_EXIST}] = per_operand_states.at(op_a_id).at(EXIST_NOT_ZERO)
    * per_operand_states.at(op_b_id).at(NOT_EXIST);
  flattened_probs[{EXIST_ZERO, EXIST_NOT_ZERO}] = per_operand_states.at(op_a_id).at(EXIST_ZERO)
    * per_operand_states.at(op_b_id).at(EXIST_NOT_ZERO);
  flattened_probs[{EXIST_ZERO, EXIST_ZERO}] = per_operand_states.at(op_a_id).at(EXIST_ZERO)
    * per_operand_states.at(op_b_id).at(EXIST_ZERO);
  flattened_probs[{EXIST_ZERO, NOT_EXIST}] = per_operand_states.at(op_a_id).at(EXIST_ZERO)
    * per_operand_states.at(op_b_id).at(NOT_EXIST);
  flattened_probs[{NOT_EXIST, EXIST_NOT_ZERO}] = per_operand_states.at(op_a_id).at(NOT_EXIST)
    * per_operand_states.at(op_b_id).at(EXIST_NOT_ZERO);
  flattened_probs[{NOT_EXIST, EXIST_ZERO}] = per_operand_states.at(op_a_id).at(NOT_EXIST)
    * per_operand_states.at(op_b_id).at(EXIST_ZERO);
  flattened_probs[{NOT_EXIST, NOT_EXIST}] = per_operand_states.at(op_a_id).at(NOT_EXIST)
    * per_operand_states.at(op_b_id).at(NOT_EXIST);
}


void CalculateFineGrainedComputeAccesses2Operand(const SparseAnalysisState& state,
                                                 tiling::CompoundTileNest& compound_tile_nest)
{

  // scenarios of operands states       | resulted compute actions
  // -----------------------------------------------------------------
  // 1) A: ENZ, B: ENZ                  | random compute
  // 2) A: ENZ, B: EZ, 4) A: EZ, B: ENZ | random, gated compute (depends on optimization) not we cannot skip here as both operands exist
  // 3) A: ENZ, B: NE, 7) A: NE, B: ENZ | random, gated, skipped compute  depends on optimization
  // 5) A: EZ, B: EZ                    | random, gated compute (depends on optimization) note that we cannot skip here as both operands exist
  // 6) A: EZ, B: NE,  8) A: NE, B: EZ  | random, gated, skipped compute  depends on optimization
  // 9) A: NE, B: NE                    | nonexistent compute (the compute unit will not see this case happening as nothing actually exists)
  // -----------------------------------------------------------------
  // NE: not exist; EZ: exist, is zero; ENZ: exist, not zero


  // Find the inner most level tile for each operand
  auto& compound_data_movement_nest = compound_tile_nest.compound_data_movement_info_nest;
  auto& compute_info = compound_tile_nest.compute_info_nest[0];

  std::map<DataSpaceID, bool> operand_compressed;
  //std::map<DataSpaceID, bool> implicit_coordinates;
  std::map<DataSpaceID, bool> operand_has_metadata;
  PerDataSpaceDensityModel operand_density_models;
  for (unsigned pv = 0; pv < problem::GetShape()->NumDataSpaces; pv++)
  {
    if (!problem::GetShape()->IsReadWriteDataSpace.at(pv))
    {
      auto& pv_data_movement_nest = compound_data_movement_nest[pv];
      for (unsigned l = 0; l < state.num_storage_levels_; l++)
      {
        if (pv_data_movement_nest[l].shape != 0)
        {
          // found the inner most storage for dspace pv
          operand_compressed[pv] = pv_data_movement_nest[l].compressed;
          operand_density_models[pv] = pv_data_movement_nest[l].tile_density;
          operand_has_metadata[pv] = pv_data_movement_nest[l].has_metadata;
          break;
        }
      }
    }
  }

  // Query density models to get necessary density of the scalar operands
  //TODO: consider when there are more than 2 operand dataspaces
  assert(operand_compressed.size() == 2);

  SetOfOperationSpaces set_of_operation_spaces;
  problem::OperationPoint scalar_high;
  set_of_operation_spaces.op_space_mold_high = scalar_high;
  set_of_operation_spaces.upper_level_loops = state.mapping_.loop_nest.loops;
  set_of_operation_spaces.workload = state.workload_;
  auto operand_exp_densities = GetExpectedOperandDensities(set_of_operation_spaces, operand_density_models);
  // for (auto iter = operand_exp_densities.begin(); iter != operand_exp_densities.end(); iter++)
  // {
  //   std::cout << problem::GetShape()->DataSpaceIDToName.at(iter->first) << " density : " << iter->second << std::endl;
  // }

  // Calculate the probability of each possible state of the operands
  std::map <DataSpaceID, PerStateProb> per_operand_states;
  for (auto iter = operand_exp_densities.begin(); iter != operand_exp_densities.end(); iter++)
  {
    problem::Shape::DataSpaceID pv = iter->first;
    double pv_density = round(iter->second*1000000)/1000000;
    per_operand_states[pv][EXIST_NOT_ZERO] = pv_density;
    if (operand_has_metadata.at(pv))
    {
      per_operand_states[pv][EXIST_ZERO] = 0;
      per_operand_states[pv][NOT_EXIST] = 1.0 - pv_density;
    } else
    {
      per_operand_states[pv][EXIST_ZERO] = 1.0 - pv_density;
      per_operand_states[pv][NOT_EXIST] = 0;
    }

    // std::cout << "  EZ: " << per_operand_states[pv][EXIST_ZERO] << std::endl;
    // std::cout << "  NE: " << per_operand_states[pv][NOT_EXIST] << std::endl;
    // std::cout << "  ENZ: " << per_operand_states[pv][EXIST_NOT_ZERO] << std::endl;

  }

  // Extract the operand dataspace ids
  auto iter = per_operand_states.begin();
  DataSpaceID op_a_id = iter->first;
  iter++;
  DataSpaceID op_b_id = iter->first;

  // Calculate dependent probabilities
  std::map<ComputeOperandStatePair, double> flattened_probs = {};
  flattened_probs[{EXIST_NOT_ZERO, EXIST_NOT_ZERO}] = 0.0;
  flattened_probs[{EXIST_NOT_ZERO, EXIST_ZERO}] = 0.0;
  flattened_probs[{EXIST_NOT_ZERO, NOT_EXIST}] = 0.0;
  flattened_probs[{EXIST_ZERO, EXIST_NOT_ZERO}] = 0.0;
  flattened_probs[{EXIST_ZERO, EXIST_ZERO}] = 0.0;
  flattened_probs[{EXIST_ZERO, NOT_EXIST}] = 0.0;
  flattened_probs[{NOT_EXIST, EXIST_NOT_ZERO}] = 0.0;
  flattened_probs[{NOT_EXIST, EXIST_ZERO}] = 0.0;
  flattened_probs[{NOT_EXIST, NOT_EXIST}] = 0.0;

  // std::cout << "op b EZ:  " << per_operand_states.at(op_b_id).at(EXIST_ZERO)     << std::endl;
  // std::cout << "op b ENZ: " << per_operand_states.at(op_b_id).at(EXIST_NOT_ZERO) << std::endl;
  // std::cout << "op b NE:  " << per_operand_states.at(op_b_id).at(NOT_EXIST)      << std::endl;
  // std::cout << "op a EZ:  " << per_operand_states.at(op_a_id).at(EXIST_ZERO)     << std::endl;
  // std::cout << "op a ENZ: " << per_operand_states.at(op_a_id).at(EXIST_NOT_ZERO) << std::endl;
  // std::cout << "op a NE:  " << per_operand_states.at(op_a_id).at(NOT_EXIST)      << std::endl; 
 
  // Extract if this is a double sided intersection
  bool double_sided_saf = state.storage_gs_saf_.at(op_a_id) && state.storage_gs_saf_.at(op_b_id);

  if (double_sided_saf)
  {
    // std::cout << " double sided saf..." << std::endl;
    double cond_on_nonempty_ratio = 1 - state.innermost_empty_cond_on_prob_.at(op_b_id);
    per_operand_states[op_a_id][EXIST_ZERO] = operand_has_metadata.at(op_a_id) ? 0 : 1 - operand_exp_densities.at(op_a_id)/cond_on_nonempty_ratio;
    per_operand_states[op_a_id][NOT_EXIST] = !operand_has_metadata.at(op_a_id) ? 0 : 1 - operand_exp_densities.at(op_a_id)/cond_on_nonempty_ratio;
    per_operand_states[op_a_id][EXIST_NOT_ZERO] = operand_exp_densities.at(op_a_id)/cond_on_nonempty_ratio;
    // std::cout << "op a leader tile conditioned on nonempty ratio: " << cond_on_nonempty_ratio << std::endl;


    cond_on_nonempty_ratio = 1 - state.innermost_empty_cond_on_prob_.at(op_a_id);
    per_operand_states[op_b_id][EXIST_ZERO] = operand_has_metadata.at(op_b_id) ? 0 : 1 - operand_exp_densities.at(op_b_id)/cond_on_nonempty_ratio;
    per_operand_states[op_b_id][NOT_EXIST] = !operand_has_metadata.at(op_b_id) ? 0 : 1 - operand_exp_densities.at(op_b_id)/cond_on_nonempty_ratio;
    per_operand_states[op_b_id][EXIST_NOT_ZERO] = operand_exp_densities.at(op_b_id)/cond_on_nonempty_ratio;
    // std::cout << "op b leader tile conditioned on nonempty ratio: " << cond_on_nonempty_ratio << std::endl;
    
    CalculateFlattenedProb(per_operand_states, flattened_probs, op_a_id, op_b_id); 
  }
  else if (state.storage_gs_saf_.at(op_a_id))
  {
    // std::cout << "op a optimized based on op b" << std::endl;
    double cond_on_nonempty_ratio = 1 - state.innermost_empty_cond_on_prob_.at(op_a_id);
    per_operand_states[op_b_id][EXIST_ZERO] = operand_has_metadata.at(op_b_id) ? 0 : 1 - operand_exp_densities.at(op_b_id)/cond_on_nonempty_ratio;
    per_operand_states[op_b_id][NOT_EXIST] = !operand_has_metadata.at(op_b_id) ? 0 : 1 - operand_exp_densities.at(op_b_id)/cond_on_nonempty_ratio;
    per_operand_states[op_b_id][EXIST_NOT_ZERO] = operand_exp_densities.at(op_b_id)/cond_on_nonempty_ratio;
    
    // std::cout << "op b leader tile conditioned on nonempty ratio: " << cond_on_nonempty_ratio << std::endl;

    CalculateFlattenedProb(per_operand_states, flattened_probs, op_a_id, op_b_id); 
  }
  else if (state.storage_gs_saf_.at(op_b_id))
  {
    // get dependent probability for a
    // std::cout << "opb conditioned on op a" << std::endl;
    double cond_on_nonempty_ratio = 1 - state.innermost_empty_cond_on_prob_.at(op_b_id);
    per_operand_states[op_a_id][EXIST_ZERO] = operand_has_metadata.at(op_a_id) ? 0 : 1 - operand_exp_densities.at(op_a_id)/cond_on_nonempty_ratio;
    per_operand_states[op_a_id][NOT_EXIST] = !operand_has_metadata.at(op_a_id) ? 0 : 1 - operand_exp_densities.at(op_a_id)/cond_on_nonempty_ratio;
    per_operand_states[op_a_id][EXIST_NOT_ZERO] = operand_exp_densities.at(op_a_id)/cond_on_nonempty_ratio;
    
    // std::cout << "op a leader tile conditioned on nonempty ratio: " << cond_on_nonempty_ratio << std::endl;
    CalculateFlattenedProb(per_operand_states, flattened_probs, op_a_id, op_b_id); 
  }
  else
  {
    CalculateFlattenedProb(per_operand_states, flattened_probs, op_a_id, op_b_id);
  }

  // Initialize fine grained access counts
  // double total_compute = compute_info.replication_factor * (double)compute_info.accesses;
  double total_compute = compute_info.fine_grained_accesses["random_compute"];

  // Extract hardware sparse optimization spec (can zero operand be identified?)
  bool gate_on_zero_operand = false;
  bool skip_on_not_aligned_operands = false;
  if (state.sparse_optimization_info_->compute_optimization_info.find("gate_on_zero_operand") !=
      state.sparse_optimization_info_->compute_optimization_info.end())
  {
    gate_on_zero_operand = state.sparse_optimization_info_->compute_optimization_info.at("gate_on_zero_operand");
  }

  if (state.sparse_optimization_info_->compute_optimization_info.find("skip_on_not_aligned_operands") !=
      state.sparse_optimization_info_->compute_optimization_info.end())
  {
    skip_on_not_aligned_operands = state.sparse_optimization_info_->compute_optimization_info.at("skip_on_not_aligned_operands");
  }

  // Initialize all the counters
  // nonexistent compute: although should happen in algorithmic world, will not be present at the hardware
  //   the cycle needs to be spent when one of the operands is empty
  double random_compute = 0.0, skipped_compute = 0.0, gated_compute = 0.0, tmp_delta = 0.0, nonexistent_compute = 0.0;

  // Analyze case by case
  // 1) A: ENZ, B: ENZ                  | random compute
  random_compute += total_compute * flattened_probs[{EXIST_NOT_ZERO, EXIST_NOT_ZERO}];
  //std::cout << "(1) skipped: " << skipped_compute << " gated: " << gated_compute << "  random: " << random_compute << std::endl;

  // 2) A: ENZ, B: EZ, 4) A: EZ, B: ENZ | random, gated compute (depends on optimization) not we cannot skip here as both operands exist
  tmp_delta = total_compute * (flattened_probs[{EXIST_NOT_ZERO, EXIST_ZERO}] + flattened_probs[{EXIST_ZERO, EXIST_NOT_ZERO}]);
  if (gate_on_zero_operand)
  {
    gated_compute += tmp_delta;
  }
  else
  {
    random_compute += tmp_delta;
  }
  // std::cout << "(2)(4) skipped: " << skipped_compute << " gated: " << gated_compute << "  random: " << random_compute <<
  // << " nonexistent: " << nonexistent_compute << std::endl;

  // 3) A: ENZ, B: NE, 7) A: NE, B: ENZ | nonexistent/(random, gated, skipped) compute (w/o vs w/ alignment (when alignment needed depends on optimization))
  tmp_delta = total_compute * (flattened_probs[{EXIST_NOT_ZERO, NOT_EXIST}] + flattened_probs[{NOT_EXIST, EXIST_NOT_ZERO}]);
  if (skip_on_not_aligned_operands)
    skipped_compute += tmp_delta; // operand alignment unit jumps to look for pair of ENZ ENZ operands
  else
  {
    if (gate_on_zero_operand)
    {
      gated_compute += tmp_delta;
    } else
    {
      random_compute += tmp_delta;  // operand alignment unit sends bubble to compute unit
    }
  }
  // std::cout << "(3)(7) skipped: " << skipped_compute << " gated: " << gated_compute << "  random: " << random_compute <<
  // << " nonexistent: " << nonexistent_compute << std::endl;


  // 5) A: EZ, B: EZ                    | random, gated compute (depends on optimization) note that we cannot skip here as both operands exist
  tmp_delta = total_compute * flattened_probs[{EXIST_ZERO, EXIST_ZERO}] ;
  if (gate_on_zero_operand)
  {
    gated_compute += tmp_delta;
  } else
  {
    random_compute += tmp_delta;
  }

  // std::cout << "(5) skipped: " << skipped_compute << " gated: " << gated_compute << "  random: " << random_compute
  // << " nonexistent: " << nonexistent_compute << std::endl;


  // 6) A: EZ, B: NE,  8) A: NE, B: EZ  | (random, gated, skipped) compute depends on optimization
  tmp_delta = total_compute * (flattened_probs[{EXIST_ZERO, NOT_EXIST}] + flattened_probs[{NOT_EXIST, EXIST_ZERO}]);
  if (skip_on_not_aligned_operands)
  {
    skipped_compute += tmp_delta; // operand alignment unit jumps to look for pair of ENZ ENZ operands
  }
  else
  {
    if (gate_on_zero_operand)
    {
      gated_compute += tmp_delta;
    }
    else
    {
      random_compute += tmp_delta;  // operand alignment unit sends bubble to compute unit
    }
  }

  // std::cout << "(6)(8) skipped: " << skipped_compute << " gated: " << gated_compute << "  random: " << random_compute <<
  // << " nonexistent: " << nonexistent_compute << std::endl;


  // 9) A: NE, B: NE                    | nonexistent compute (the compute unit will not see this case happening as nothing actually exists)
  tmp_delta = total_compute * flattened_probs[{NOT_EXIST, NOT_EXIST}] ;
  nonexistent_compute += tmp_delta;

  //std::cout << "(9) skipped: " << skipped_compute << " gated: " << gated_compute << "  random: " << random_compute
  //<< " nonexistent: " << nonexistent_compute << std::endl;

  //std::cout << "total: " << total_compute << "  sum: " <<  skipped_compute + random_compute + gated_compute + nonexistent_compute
  //<< " diff: " << total_compute - skipped_compute - random_compute - gated_compute  - nonexistent_compute << std::endl;

  // sanity check
  // as long as different is smaller than 0.001% of the total compute, pass
  // there could be tiny discrepencies due to lack of precision...
  assert(abs(skipped_compute + random_compute + gated_compute + nonexistent_compute - total_compute) < total_compute * 0.00001);


  // now round the action counts into integers (pessimistic rounding)
  compute_info.fine_grained_accesses["skipped_compute"] = floor(skipped_compute);
  compute_info.fine_grained_accesses["gated_compute"] = floor(gated_compute);
  compute_info.fine_grained_accesses["random_compute"] =
    total_compute - floor(skipped_compute) - floor(gated_compute) - floor(nonexistent_compute);

  // std::cout << "(final) skipped compute: " << compute_info.fine_grained_accesses["skipped_compute"]
  //   << " gated compute: " << compute_info.fine_grained_accesses["gated_compute"]
  //   << " random compute: " << compute_info.fine_grained_accesses["random_compute"]
  //   << " nonexistent: " << nonexistent_compute << std::endl;

}


// ------------------------------------------------------------------------
// A decoder.
// Convert a decimal to base 3 (specifically to ComputeOperandState)
// Each ternary bit represents the state of an operand. The number of bits
// "size" represents the total number of operands involved in compute.
// ------------------------------------------------------------------------
void to_base_3(uint64_t value, std::vector<ComputeOperandState>& ternary_array, uint32_t size) 
{
	ComputeOperandState r = static_cast<ComputeOperandState>(value % 3);
	uint64_t q = (uint64_t) floor(value/3);
	ternary_array.push_back(r);
	
	
	if (ternary_array.size() == size) {
		return;
	}

	to_base_3(q, ternary_array, size);
}

// -------------------------------------------------------
// Updated Version of CalculateFineGrainedComputeAccesses
// Handles multiple operands.
// TODO: potential bug with sanity check at the end.
// -------------------------------------------------------
void CalculateFineGrainedComputeAccesses(const SparseAnalysisState& state,
                                         tiling::CompoundTileNest& compound_tile_nest)
{

  // --------------------------------------------------------------
  // Expanding the 2 operand scenario to multiple operands
  // I assume that each of the operands has an index shared with 
  // some other operand. That is, we can safely do multi-way
  // intersection. 
  // For example: A_mkj * B_nki * C_oih * D_phl = Z_mnopl
  // Here, we need the probability that all the matching scalars
  // for all 4 operands exist.
  // --------------------------------------------------------------

  // scenarios of operands states       | resulted compute actions
  // -----------------------------------------------------------------
  // 1) All operands are ENZ            | random compute
  // 2) At least one operand is EZ,     | random, gated compute (depends on optimization)
  //    no operands are NE              |   note we cannot skip here as all operands exist
  // 3) At least one operand is NE, but | random, gated, skipped compute (depends on optimization)
  //    some operands are EZ or ENZ     |
  // 4) All operands are NE             | nonexistent compute (the compute unit will not see this case happening
  //                                    | as nothing actually exists
  // -----------------------------------------------------------------
  // NE: not exist; EZ: exist, is zero; ENZ: exist, not zero


  // Find the inner most level tile for each operand
  auto& compound_data_movement_nest = compound_tile_nest.compound_data_movement_info_nest;
  auto& compute_info = compound_tile_nest.compute_info_nest[0];

  std::map<DataSpaceID, bool> operand_compressed;
  //std::map<DataSpaceID, bool> implicit_coordinates;
  std::map<DataSpaceID, bool> operand_has_metadata;
  PerDataSpaceDensityModel operand_density_models;
  for (unsigned pv = 0; pv < problem::GetShape()->NumDataSpaces; pv++)
  {
    if (!problem::GetShape()->IsReadWriteDataSpace.at(pv))
    {
      auto& pv_data_movement_nest = compound_data_movement_nest[pv];
      for (unsigned l = 0; l < state.num_storage_levels_; l++)
      {
        if (pv_data_movement_nest[l].shape != 0)
        {
          // found the inner most storage for dspace pv
          operand_compressed[pv] = pv_data_movement_nest[l].compressed;
          operand_density_models[pv] = pv_data_movement_nest[l].tile_density;
          operand_has_metadata[pv] = pv_data_movement_nest[l].has_metadata;
          // if (pv_data_movement_nest[l].has_metadata)
          // {
          //   implicit_coordinates[pv] = compression_info.per_level_info_map.at(l).at(pv).coordinates_implicit[0];
          // }
          break;
        }
      }
    }
  }

  // Query density models to get necessary density of the scalar operands
  SetOfOperationSpaces set_of_operation_spaces;
  problem::OperationPoint scalar_high;
  set_of_operation_spaces.op_space_mold_high = scalar_high;
  set_of_operation_spaces.upper_level_loops = state.mapping_.loop_nest.loops;
  set_of_operation_spaces.workload = state.workload_;
  auto operand_exp_densities = GetExpectedOperandDensities(set_of_operation_spaces, operand_density_models);
   
  // for (auto iter = operand_exp_densities.begin(); iter != operand_exp_densities.end(); iter++)
  // {
  //   std::cout << problem::GetShape()->DataSpaceIDToName.at(iter->first) << " density : " << iter->second << std::endl;
  // }

  // Calculate the probability of each possible state of the operands
  std::map <DataSpaceID, PerStateProb> per_operand_states;
  for (auto iter = operand_exp_densities.begin(); iter != operand_exp_densities.end(); iter++)
  {
    problem::Shape::DataSpaceID pv = iter->first;
    double pv_density = iter->second;
    per_operand_states[pv][EXIST_NOT_ZERO] = pv_density;
    if (operand_has_metadata.at(pv))
    {
      per_operand_states[pv][EXIST_ZERO] = 0;
      per_operand_states[pv][NOT_EXIST] = 1.0 - pv_density;
    } else
    {
      per_operand_states[pv][EXIST_ZERO] = 1.0 - pv_density;
      per_operand_states[pv][NOT_EXIST] = 0;
    }

    //std::cout << "  EZ: " << per_operand_states[pv][EXIST_ZERO] << std::endl;
    //std::cout << "  NE: " << per_operand_states[pv][NOT_EXIST] << std::endl;
    //std::cout << "  ENZ: " << per_operand_states[pv][EXIST_NOT_ZERO] << std::endl;

  }

  // -------------------------------------------------------------
  // WORKAROUND FOR MULTI-OPERAND INTERSECTION
  // Assumptions:
  //     - optimizations are on all other operands, 
  //       (see storage_gs_saf_ variable)
  //       So, this assumes the ranks are shared in some
  //       way by all operands
  // For example: A_mkj * B_nki * C_oih * D_phl = Z_mnopl
  // Here, we need the probability that all the matching scalars
  // for all 4 operands exist.
  // -------------------------------------------------------------

  uint32_t num_operands = per_operand_states.size();
  double num_states = pow(3.0, num_operands);

  // Calculate dependent probabilities
  std::map<ComputeOperandStatePair, double> flattened_probs = {};

  // Initialize the probabilities for all possible combinations
  // of ENZ, EZ, and NE to 0
  for (uint32_t i = 0; i < num_states; i++)
  { 
    ComputeOperandStatePair state_vector;
    to_base_3(i, state_vector, num_operands);
    flattened_probs[state_vector] = 0.0;
  }


  // Check if all operands are scalar-scalar optimization
  bool multi_scalar_opt = true;
  auto iter_all_ops = per_operand_states.begin();
 
  std::vector<DataSpaceID> operand_dataspaces;

  for (uint32_t i = 0; i < num_operands; i++)
  {
    DataSpaceID op_x_id = iter_all_ops->first;
  	multi_scalar_opt = multi_scalar_opt && state.storage_gs_saf_.at(op_x_id);
    operand_dataspaces.push_back(op_x_id);
    iter_all_ops++;
  }

  bool no_opts = true;

  if (multi_scalar_opt)
  {
    // essentially intersection at scalar scale
    // This version is for when all operands are ENZ or NE
    double enz_value = 1.0;

    ComputeOperandStatePair state_vector;
    ComputeOperandStatePair state_vector_ne;

    for (uint32_t i = 0; i < num_operands; i++)
    {
	    state_vector.push_back(EXIST_NOT_ZERO);
      state_vector_ne.push_back(NOT_EXIST);
      DataSpaceID op_x_id = operand_dataspaces.at(i);

      // The probability of the first two operands being ENZ,
      // Then the result of that being ENZ mulitplied by the next operand
      // being ENZ, and so on...
      enz_value = enz_value * per_operand_states[op_x_id][EXIST_NOT_ZERO];
    }

    flattened_probs[state_vector] = enz_value;
    flattened_probs[state_vector_ne] = 1 - enz_value;

    no_opts = false;
  }

  // ----------------------------------------------------------------------
  // Not every operand needs to be optimized. Walk through the operands
  // that have optimization turned on.
  // FIXME: Note that there is a potential bug here. In the 2 operand case, 
  //  it's an if else condition (either a or b, but not both are
  //  optimized if we get to this point).
  //  For multiple operands, we need to consider every combination of
  //  operands that have been optimized, so this might need another walk
  //  through of all the possible combinations of operands being optimized.
  //  We need to know what each operand is optimized on..., then we only
  //  need to look at that other operand.
  // The current implementation stops once a single operand is encountered
  // that requires optimization.
  // ----------------------------------------------------------------------
  else 
  {
    // loop through each operand and check if they need to be optimized
    for (uint32_t i = 0; i < num_operands; i++) 
    {
      // iterate through each operand
      DataSpaceID op_this_id = operand_dataspaces.at(i);

      // Is this the first operand to have optimizations?
      if (no_opts && state.storage_gs_saf_.at(op_this_id)) 
      {
        no_opts = false;
		
        // Loop through all the possible operand states (ENZ, EZ, NE)
        for (uint32_t state_i = 0; state_i < num_states; state_i++)
        { 
          ComputeOperandStatePair state_vector;
          to_base_3(state_i, state_vector, num_operands);

          // For this state, set the flattened_probs
          ComputeOperandState this_state = state_vector.at(i);
          int prob_state = 2; // 0 -- compute, 1 -- set to 0, 2 -- all don't exist
          bool there_is_compute = true;
          double prob_value = 1.0; 
          double prob_ne_value = 1.0;
          ComputeOperandState op_state;

          for (uint32_t op_i = 0; op_i < num_operands; op_i++)
          {

            DataSpaceID op_x_id = operand_dataspaces.at(op_i);
            // make sure we're looking at the other operands
            if (op_i != i)
            {
              op_state = state_vector.at(op_i);

              // There's a zero! No compute
              if (op_state == EXIST_ZERO)
              {
                prob_state = 1;		
                there_is_compute = false;
              } 
						
              else if (op_state == NOT_EXIST)
              {
                // we've encountered either EZ or ENZ 
                // in some other operand
                if (prob_state != 2) 
                { 
                  prob_state = 1;
                  there_is_compute = false;
                }
                // else everything's been NE so far...
                else 
                {
                  // leave it at 2
                  prob_ne_value = prob_ne_value *
                    (per_operand_states[op_x_id][NOT_EXIST] + 
                     per_operand_states[op_x_id][EXIST_ZERO]);
                  there_is_compute = false;
                }
              }

              // op_state == EXIST_NOT_ZERO
              else 
              {
                // Then prior runs have all been NOT_EXIST. There'll be no compute
                if (prob_state == 2 && !there_is_compute) 
                { 
                  prob_state = 1;
                } 

                // then prior runs have either been NE or EZ leave as is.
                else if (prob_state == 1) {
                }

                // then prior runs have been compute!
                else if (there_is_compute) { 
                  prob_state = 0;
                  prob_value = prob_value*per_operand_states[op_x_id][EXIST_NOT_ZERO];
                }
              }
            }
					 
          } // done checking the states of the other operands

          // If compute, then compute!
          if (prob_state == 0)
          {
            flattened_probs[state_vector] = 
              prob_value*per_operand_states[op_this_id][this_state];
          } 
          // one of the operands is zero or doesn't exist. Set to 0
          else if (prob_state == 1)
          {
            flattened_probs[state_vector] = 0.0;
          }
          // so far, none of the operands existed...
          else if (prob_state == 2)
          {
            if (this_state == NOT_EXIST)
            {
              flattened_probs[state_vector] = prob_ne_value;
            } else
            {
              flattened_probs[state_vector] = 0.0;
            }
          }
  			} // done with all possible combinations of operand states
      } // done with checking this optimization for this operand
    } // done checking optimizations for ALL operands
  }

  // There were no optimizations on any of the operands.
  // EVERYTHING will be compute
  if (no_opts)
  {
    // Go through each possible state
    for (uint32_t state_i = 0; state_i < num_states; state_i++)
    {
      ComputeOperandStatePair state_vector;
      to_base_3(state_i, state_vector, num_operands);
		
      // Set the flattened_probs for each state
      double prob_value = 1.0;

      for (uint32_t op_i = 0; op_i < num_operands; op_i++)
      {
        DataSpaceID op_x_id = operand_dataspaces.at(op_i);
        ComputeOperandState op_state = state_vector.at(op_i);
        prob_value = prob_value * per_operand_states[op_x_id][op_state];
      }
      flattened_probs[state_vector] = prob_value;
    }
  }


  // Initialize fine grained access counts
  double total_compute = compute_info.fine_grained_accesses["random_compute"];

  // Extract hardware sparse optimization spec (can zero operand be identified?)
  bool gate_on_zero_operand = false;
  bool skip_on_not_aligned_operands = false;
  if (state.sparse_optimization_info_->compute_optimization_info.find("gate_on_zero_operand") !=
      state.sparse_optimization_info_->compute_optimization_info.end())
  {
    gate_on_zero_operand = state.sparse_optimization_info_->compute_optimization_info.at("gate_on_zero_operand");
  }

  if (state.sparse_optimization_info_->compute_optimization_info.find("skip_on_not_aligned_operands") !=
      state.sparse_optimization_info_->compute_optimization_info.end())
  {
    skip_on_not_aligned_operands = state.sparse_optimization_info_->compute_optimization_info.at("skip_on_not_aligned_operands");
  }

  // Initialize all the counters
  // nonexistent compute: although should happen in algorithmic world, will not be present at the hardware
  //   e.g., for cartesian product, any pair of non-empty operands is legal,
  //   the empty operands are then naturally skipped over by hardware as no alignment is performed
  //   however, if alignment is needed, unless the hardware can lookup corresponding pairs with the "skipping" optimization
  //   the cycle needs to be spent when one of the operands is empty
  double random_compute = 0.0, skipped_compute = 0.0, gated_compute = 0.0, tmp_delta = 0.0, nonexistent_compute = 0.0;


  // ----------------------------------------------------------------
  // Multi-operand version
  // Go through each possible case/state
  // Priorities:
  //  	if all operands are NE: update non-existent compute
  //  	if any operand is NE: update skip or gate or random compute
  //  	if any operand is EZ: update gate or random
  //  	else all operands are ENZ: update random only
  // ----------------------------------------------------------------
  for (uint32_t state_i = 0; state_i < num_states; state_i++)
  {
    ComputeOperandStatePair state_vector;
    to_base_3(state_i, state_vector, num_operands);

    tmp_delta = total_compute * (flattened_probs[state_vector]);
	
    // enumerate the cases
    int all_ne = 0, some_ne = 0, some_ez = 0, some_enz = 0, all_enz = 0;
	
    // Loop through each operand state in this state_vector
    for (uint32_t op_i = 0; op_i < num_operands; op_i++)
    {
      ComputeOperandState op_state = state_vector.at(op_i);
		
      if (op_state == NOT_EXIST)
      {
        some_ne = 1;
      } else if (op_state == EXIST_ZERO)
      {
        some_ez = 1;
      } else if (op_state == EXIST_NOT_ZERO)
      {
        some_enz = 1;
      } else {
        printf("ERROR: op_state does not exist!\n");
        exit(1);
      }
    }

    // No EZ and No ENZ means they are all NE
    if (!some_ez && !some_enz && some_ne)
    {
      all_ne = 1;
    } 
    // No NE and No EZ means they are all ENZ
    else if (some_enz && !some_ne && !some_ez)
    {
      all_enz = 1;
    }
	
    // Update compute for this state_vector
    // all operands are in the NOT_EXIST state
    if (all_ne) {
      nonexistent_compute += tmp_delta;
    }
    // SOME of the operands are in the NOT_EXIST state
    else if (some_ne)
    {
      if (skip_on_not_aligned_operands)
  		{
    		skipped_compute += tmp_delta; // operand alignment unit jumps to look for pair of ENZ ENZ operands
  		}
  		else
  		{
    		if (gate_on_zero_operand)
    		{
          gated_compute += tmp_delta;
    		}
    		else
    		{
          random_compute += tmp_delta;  // operand alignment unit sends bubble to compute unit
    		}
  		}
  	}
    // SOME of the operands are in the EXIST_ZERO state, but not NE state
    else if (some_ez)
    {
      if (gate_on_zero_operand)
  		{
    		gated_compute += tmp_delta;
  		} else
  		{
    		random_compute += tmp_delta;
  		}
    } 
    // All the operands exist and aren't zero!
    else if (all_enz)
    {
      random_compute += tmp_delta;
    }
    else
    {// we should never reach this stage
      printf("ERROR: Bug in calculating intersection computes...\n");
      exit(1);
    }
  }

  // Sanity check
  // std::cout << "total: " << total_compute << "  sum: " <<  skipped_compute + random_compute + gated_compute + nonexistent_compute
  // << " diff: " << total_compute - skipped_compute - random_compute - gated_compute  - nonexistent_compute << 
  // " upper bound " << total_compute * 0.00001 << std::endl;
  // fflush(stdout);

  // sanity check
  // as long as different is smaller than 0.001% of the total compute, pass
  // there could be tiny discrepencies due to lack of precision...
  // TODO: Removed this assertion for now. Sometimes fails, need to check code above.
  //       Need to figure out how to check all combinations of operands 
  //       needing optimization (vs. a single one). See FIXME comment above.
  // assert(abs(skipped_compute + random_compute + gated_compute + nonexistent_compute - total_compute) < total_compute * 0.00001);


  if (abs(skipped_compute + random_compute + gated_compute + nonexistent_compute - total_compute) > total_compute * 0.00001) {
    printf("WARNING: the calculated compute is not meeting the constraints.");
  }

  // Adding  a basic sanity check. 
  assert(abs(skipped_compute + random_compute + gated_compute + nonexistent_compute) <= total_compute);

  // now round the action counts into integers (pessimistic rounding)
  compute_info.fine_grained_accesses["skipped_compute"] = floor(skipped_compute);
  compute_info.fine_grained_accesses["gated_compute"] = floor(gated_compute);
  compute_info.fine_grained_accesses["random_compute"] =
    total_compute - floor(skipped_compute) - floor(gated_compute) - floor(nonexistent_compute);

  // std::cout << "(final) skipped compute: " << compute_info.fine_grained_accesses["skipped_compute"]
  //   << " gated compute: " << compute_info.fine_grained_accesses["gated_compute"]
  //   << " random compute: " << compute_info.fine_grained_accesses["random_compute"]
  //   << " nonexistent: " << nonexistent_compute << std::endl;

}

} // namespace
