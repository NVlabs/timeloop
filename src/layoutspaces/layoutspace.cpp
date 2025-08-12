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

 #include "layoutspaces/layoutspace.hpp"
 #include <set>
 #include <algorithm>
 #include <stdexcept>
 #include <cassert>
 #include <functional>
 // #define DEBUG_CONCORDANT_LAYOUT
 // #define DEBUG_BUFFER_CAPACITY_CONSTRAINT
 // #define DEBUG_CONSTRUCTION_LAYOUT
 // #define DEBUG_CREATE_INTRALINE_FACTOR_SPACE
 #define PACKING_PRUNING_RATIO 0.9

 namespace layoutspace
 {

 //------------------------------------------//
 //        Helper Functions                  //
 //------------------------------------------//

  void Legal::Init(model::Engine::Specs arch_specs,
    const Mapping& mapping,
    layout::Layouts& layout)
  {
    arch_specs_ = arch_specs;
    layout_ = layout::Layouts(layout);

    num_storage_levels = mapping.loop_nest.storage_tiling_boundaries.size();
    num_data_spaces = layout_.at(0).intraline.size();
    ParseArchSpecs(arch_specs, mapping);

    // Step 1: Create concordant layout from mapping
    CreateConcordantLayout(mapping);

    // Step 2: Create design spaces for layout optimization
    CreateIntralineFactorSpace(arch_specs, mapping);
  };

  // Helper function to find all divisors of a number
  std::vector<uint32_t> FindDivisors(uint32_t n)
  {
    std::vector<uint32_t> divisors;
    for (uint32_t i = 1; i <= n; ++i)
    {
      if (n % i == 0)
      {
        divisors.push_back(i);
      }
    }
    return divisors;
  }

  // Helper function to generate combinations of ranks for multi-rank splitting
  std::vector<std::vector<std::string>> Legal::GenerateRankCombinations(const std::vector<std::string>& ranks, size_t max_combo_size)
  {
    std::vector<std::vector<std::string>> combinations;

    // Generate all possible combinations all ranks (limited by ranks.size())
    for (size_t combo_size = 1; combo_size <= std::min(max_combo_size, ranks.size()); combo_size++)
    {
      // Generate all combinations of combo_size from ranks
      std::vector<bool> selector(ranks.size());
      std::fill(selector.begin(), selector.begin() + combo_size, true);

      do
      {
        std::vector<std::string> combination;
        for (size_t i = 0; i < ranks.size(); ++i)
        {
          if (selector[i])
          {
            combination.push_back(ranks[i]);
          }
        }
        combinations.push_back(combination);
      } while (std::prev_permutation(selector.begin(), selector.end()));
    }

    return combinations;
  }

  // Helper function to test multi-rank splitting using pre-computed candidate factors
  bool Legal::TestMultiRankSplittingWithCandidates(unsigned lvl, unsigned ds_idx, const std::vector<std::string>& rank_combination,
                                                    const std::map<std::string, std::vector<uint32_t>>& candidate_factors_per_rank,
                                                    const std::vector<std::vector<std::uint64_t>>& intraline_size_per_ds,
                                                    uint64_t line_capacity, MultiRankSplittingOption& option)
  {
    auto& intraline_nest = layout_.at(lvl).intraline.at(ds_idx);

    // Initialize the option
    option.dataspace = ds_idx;
    option.ranks = rank_combination;
    option.original_intraline_factors.clear();
    option.splitting_factors.clear();
    option.total_reduction = 1;

    // Get original factors and candidate splitting factors for each rank in the combination
    std::vector<std::vector<uint32_t>> candidate_factors_list;
    for (const auto& rank : rank_combination) {
      uint32_t original_factor = (intraline_nest.factors.find(rank) != intraline_nest.factors.end()
                                  ? intraline_nest.factors.at(rank) : 1);
      option.original_intraline_factors[rank] = original_factor;

      // Get candidate factors for this rank
      auto candidates_it = candidate_factors_per_rank.find(rank);
      if (candidates_it == candidate_factors_per_rank.end()) {
        return false; // No candidate factors for this rank
      }
      candidate_factors_list.push_back(candidates_it->second);
    }

    // Calculate current intraline size for this dataspace
    uint64_t current_dataspace_intraline_size = intraline_size_per_ds[lvl][ds_idx];

    // Generate all combinations of candidate factors using nested loops
    // This is a more comprehensive approach than the equal distribution method
    std::function<bool(size_t, std::vector<uint32_t>&, uint64_t)> try_combinations =
      [&](size_t rank_idx, std::vector<uint32_t>& current_factors, uint64_t accumulated_reduction) -> bool {

      if (rank_idx == rank_combination.size()) {
        // All ranks have been assigned factors, test if this combination works
        // Calculate the new intraline size for the split dataspace
        assert(accumulated_reduction > 0 && "Division by zero in try_combinations");
        uint64_t new_dataspace_intraline_size = current_dataspace_intraline_size / accumulated_reduction;

        if (new_dataspace_intraline_size <= line_capacity) {
          // This combination works - store it in the option
          option.total_reduction = accumulated_reduction;
          for (size_t i = 0; i < rank_combination.size(); i++) {
            option.splitting_factors[rank_combination[i]] = current_factors[i];
          }
          return true;
        }
        return false;
      }

      // Try each candidate factor for the current rank
      const auto& rank = rank_combination[rank_idx];
      const auto& factors = candidate_factors_list[rank_idx];
      uint32_t original_factor = option.original_intraline_factors.at(rank);

      for (uint32_t factor : factors) {
        // Check if this factor is valid (i.e., divides the original factor)
        if (original_factor % factor == 0) {
          current_factors[rank_idx] = factor;
          uint64_t new_accumulated_reduction = accumulated_reduction * factor;

          // Recursive call for next rank
          if (try_combinations(rank_idx + 1, current_factors, new_accumulated_reduction)) {
            return true; // Found a valid combination
          }
        }
      }

      return false; // No valid combination found with current prefix
    };

    // Start the recursive combination testing
    std::vector<uint32_t> current_factors(rank_combination.size());
    return try_combinations(0, current_factors, 1);
  }


  // Helper function to test multi-rank packing using pre-computed candidate factors
  bool Legal::TestMultiRankPackingWithCandidates(unsigned lvl, unsigned ds_idx, const std::vector<std::string>& rank_combination,
                                                    const std::map<std::string, std::vector<uint32_t>>& candidate_factors_per_rank,
                                                    const std::vector<std::vector<std::uint64_t>>& intraline_size_per_ds,
                                                    uint64_t line_capacity, std::vector<MultiRankPackingOption>& options)
  {
    auto& interline_nest = layout_.at(lvl).interline.at(ds_idx);

    // Initialize the option
    MultiRankPackingOption option;
    option.dataspace = ds_idx;
    option.ranks = rank_combination;
    option.original_interline_factors.clear();
    option.packing_factors.clear();
    option.total_packing = 1;

    // Get original factors and candidate packing factors for each rank in the combination
    std::vector<std::vector<uint32_t>> candidate_factors_list;
    for (const auto& rank : rank_combination) {
      uint32_t original_factor = (interline_nest.factors.find(rank) != interline_nest.factors.end()
                                  ? interline_nest.factors.at(rank) : 1);
      option.original_interline_factors[rank] = original_factor;

      // Get candidate factors for this rank
      auto candidates_it = candidate_factors_per_rank.find(rank);
      if (candidates_it == candidate_factors_per_rank.end()) {
        return false; // No candidate factors for this rank
      }
      candidate_factors_list.push_back(candidates_it->second);
    }

    // Calculate current intraline size for this dataspace
    uint64_t current_dataspace_intraline_size = intraline_size_per_ds[lvl][ds_idx];

    // Generate all combinations of candidate factors using nested loops
    // This is a more comprehensive approach than the equal distribution method
    std::function<bool(size_t, std::vector<uint32_t>&, uint64_t)> try_combinations =
      [&](size_t rank_idx, std::vector<uint32_t>& current_factors, uint64_t accumulated_packing) -> bool {

      if (rank_idx == rank_combination.size()) {
        // All ranks have been assigned factors, test if this combination works
        // Calculate the new intraline size for the packed dataspace
        uint64_t new_dataspace_intraline_size = current_dataspace_intraline_size * accumulated_packing;
        if (new_dataspace_intraline_size <= line_capacity) {
          // This combination works - store it in the option
          option.total_packing = accumulated_packing;
          for (size_t i = 0; i < rank_combination.size(); i++) {
            option.packing_factors[rank_combination[i]] = current_factors[i];
          }
          options.push_back(option);
          option.total_packing = 1;
          option.packing_factors.clear();
          return true;
        }
        return false;
      }

      // Try each candidate factor for the current rank
      const auto& rank = rank_combination[rank_idx];
      const auto& factors = candidate_factors_list[rank_idx];
      uint32_t original_factor = option.original_interline_factors.at(rank);
      bool ret = false;

      for (auto factor = factors.rbegin(); factor != factors.rend(); ++factor) {
        // Check if this factor is valid (i.e., divides the original factor)
        if (original_factor % *factor == 0) {
          current_factors[rank_idx] = *factor;
          uint64_t new_accumulated_packing = accumulated_packing * *factor;

          // Recursive call for next rank
          if (try_combinations(rank_idx + 1, current_factors, new_accumulated_packing)) {
            ret = true; // Found a valid combination
          }
        }
      }

      return ret; 
    };

    // Start the recursive combination testing
    std::vector<uint32_t> current_factors(rank_combination.size());
    return try_combinations(0, current_factors, 1);
  }


  //------------------------------------------//
  //        Initialization and Setup          //
  //------------------------------------------//

  //
  // Init() - called by constructor or derived classes.
  //
  void Legal::ParseArchSpecs(model::Engine::Specs arch_specs, const Mapping& mapping)
  {
    storage_level_keep_factor.resize(num_storage_levels, std::vector<bool>(num_data_spaces, false));

    for (unsigned storage_level = 0; storage_level < num_storage_levels; storage_level++){
      for (unsigned ds_idx = 0; ds_idx < num_data_spaces; ds_idx++){
        storage_level_keep_factor[storage_level][ds_idx] = mapping.datatype_bypass_nest.at(ds_idx).test(storage_level);
      }
    }

    // Initialize the storage level capacity vectors
    storage_level_total_capacity.resize(num_storage_levels, 0);
    storage_level_line_capacity.resize(num_storage_levels, 0);

    // Iterate through each storage level to extract capacity information and bypass information.
    for (unsigned storage_level = 0; storage_level < num_storage_levels; storage_level++)
    {
      auto storage_level_specs = arch_specs.topology.GetStorageLevel(storage_level);

      // Extract total capacity
      std::uint64_t total_capacity = 0;
      if (storage_level_specs->size.IsSpecified())
      {
        total_capacity = storage_level_specs->size.Get();
      }
      else
      {
        #ifdef DEBUG_ARCH_PARSING
          std::cout << "    WARNING: Storage level " << storage_level
                    << " (" << storage_level_specs->name.Get() << ") has unspecified size, treating as infinite" << std::endl;
        #endif
        total_capacity = std::numeric_limits<uint64_t>::max();
      }

      // Determine line capacity (elements that can be accessed in parallel)
      std::uint64_t line_capacity = 0;
      if (storage_level_specs->block_size.IsSpecified())
      {
        line_capacity = storage_level_specs->block_size.Get();
      }
      else
      {
        // Fallback to bandwidth if block size not specified
        double read_bandwidth = storage_level_specs->read_bandwidth.IsSpecified() ?
                                storage_level_specs->read_bandwidth.Get() : 0.0;
        double write_bandwidth = storage_level_specs->write_bandwidth.IsSpecified() ?
                                  storage_level_specs->write_bandwidth.Get() : 0.0;
        line_capacity = static_cast<std::uint64_t>(std::max(read_bandwidth, write_bandwidth));
      }

      // Store capacity values (with safe casting)
      storage_level_total_capacity[storage_level] = (total_capacity > std::numeric_limits<uint32_t>::max()) ?
                                                    std::numeric_limits<uint32_t>::max() :
                                                    static_cast<std::uint32_t>(total_capacity);
      storage_level_line_capacity[storage_level] = (line_capacity > std::numeric_limits<uint32_t>::max()) ?
                                                    std::numeric_limits<uint32_t>::max() :
                                                    static_cast<std::uint32_t>(line_capacity);

    }
  }


  //
  // ConstructLayout() - Two-parameter version with separate layout_splitting_id and layout_packing_id
  //
  std::vector<Status> Legal::ConstructLayout(uint64_t layout_splitting_id, uint64_t layout_packing_id, layout::Layouts* layouts, Mapping mapping, bool break_on_failure)
  {
    (void)break_on_failure; // Suppress unused parameter warning

    // This function takes separate IDs for all three design spaces:
    // - layout_splitting_id: for SplittingSpace (intraline-to-interline splitting)
    // - layout_packing_id: for PackingSpace (interline-to-intraline packing)

    // Create a deep copy of the layout to ensure modifications don't affect the original
    CreateConcordantLayout(mapping);

    #ifdef DEBUG_CONSTRUCTION_LAYOUT
      std::cout << "\n=== LAYOUT CONSTRUCTION START ===" << std::endl;
      std::cout << "Layout IDs: IntraLine=" << layout_splitting_id << ", Packing=" << layout_packing_id << std::endl;
      std::cout << "Initial original layout:" << std::endl;
      layout::PrintOverallLayoutConcise(layout_);
    #endif

    /*
      Step 0: Sanity Checking
    */
    // Validate layout_splitting_id range
    if (layout_splitting_id > splitting_candidates)
    {
      Status error_status;
      error_status.success = false;
      error_status.fail_reason = " layout_splitting_id " + std::to_string(layout_splitting_id) + " exceeds SplittingSpace size " + std::to_string(splitting_candidates);
      return {error_status};
    }

    // Validate layout_packing_id range
    if (layout_packing_id > packing_candidates)
    {
      Status error_status;
      error_status.success = false;
      error_status.fail_reason = " layout_packing_id " + std::to_string(layout_packing_id) + " exceeds PackingSpace size " + std::to_string(packing_candidates);
      return {error_status};
    }

    /*
      Step 1: Decode the design space choices (Updated for both single and multi-rank splitting)
    */

    // Decode SplittingSpace choices using layout_splitting_id (intraline-to-interline splitting)
    std::vector<std::vector<std::uint64_t>> splitting_choice_per_lvl_per_ds(num_storage_levels, std::vector<std::uint64_t>(num_data_spaces, 0));
    for (unsigned lvl = num_storage_levels; lvl-- > 0;) {
      for (unsigned ds_idx = num_data_spaces; ds_idx-- > 0;) {
        uint32_t divide_factor = 0;
        if (splitting_candidates_per_lvl_per_ds[lvl][ds_idx] > 0 && storage_level_keep_factor[lvl][ds_idx]) {
          divide_factor = splitting_candidates_per_lvl_per_ds[lvl][ds_idx];
        }
        else { // bypass
          divide_factor = 1;
        }
        assert(divide_factor > 0 && "Division by zero in layout_splitting_id / divide_factor");
        splitting_choice_per_lvl_per_ds[lvl][ds_idx] = layout_splitting_id % divide_factor;
        layout_splitting_id = layout_splitting_id - splitting_choice_per_lvl_per_ds[lvl][ds_idx];
        layout_splitting_id = layout_splitting_id / divide_factor;
      }
    }

    // Print flattened splitting choices
    #ifdef DEBUG_CONSTRUCTION_LAYOUT
      std::cout << "Splitting choices:" << std::endl;
      std::cout << "Level | DataSpace | Choice" << std::endl;
      std::cout << "------|-----------|--------" << std::endl;
      for (unsigned lvl = 0; lvl < num_storage_levels; lvl++) {
        for (unsigned ds_idx = 0; ds_idx < num_data_spaces; ds_idx++) {
          std::cout << std::setw(6) << lvl << " | "
                    << std::setw(9) << ds_idx << " | "
                    << std::setw(6) << splitting_choice_per_lvl_per_ds[lvl][ds_idx] << std::endl;
        }
      }
      std::cout << std::endl;
    #endif

    // Decode PackingSpace choices using layout_packing_id (interline-to-intraline packing)
    std::vector<std::vector<std::uint64_t>> packing_choice_per_lvl_per_ds(num_storage_levels, std::vector<std::uint64_t>(num_data_spaces, 0));
    for (unsigned lvl = num_storage_levels; lvl-- > 0;) {
      for (unsigned ds_idx = num_data_spaces; ds_idx-- > 0;) {
        uint32_t divide_factor = 0;
        if (packing_candidates_per_lvl_per_ds[lvl][ds_idx] > 0 && storage_level_keep_factor[lvl][ds_idx]) {
          divide_factor = packing_candidates_per_lvl_per_ds[lvl][ds_idx];
        }
        else { // bypass
          divide_factor = 1;
        }
        assert(divide_factor > 0 && "Division by zero in layout_packing_id / divide_factor");
        packing_choice_per_lvl_per_ds[lvl][ds_idx] = layout_packing_id % divide_factor;
        layout_packing_id = layout_packing_id - packing_choice_per_lvl_per_ds[lvl][ds_idx];
        layout_packing_id = layout_packing_id / divide_factor;
      }
    }

    // Print flattened packing choices
    #ifdef DEBUG_CONSTRUCTION_LAYOUT
      std::cout << "Packing choices:" << std::endl;
      std::cout << "Level | DataSpace | Choice" << std::endl;
      std::cout << "------|-----------|--------" << std::endl;
      for (unsigned lvl = 0; lvl < num_storage_levels; lvl++) {
        for (unsigned ds_idx = 0; ds_idx < num_data_spaces; ds_idx++) {
          std::cout << std::setw(6) << lvl << " | "
                    << std::setw(9) << ds_idx << " | "
                    << std::setw(6) << packing_choice_per_lvl_per_ds[lvl][ds_idx] << std::endl;
        }
      }
      std::cout << std::endl;
    #endif

      // Apply SplittingSpace choices (both single-rank and multi-rank splitting: intraline-to-interline)
    #ifdef DEBUG_CONSTRUCTION_LAYOUT
      std::cout << "[SplittingSpace] Applying multi-rank splitting for all dataspaces..." << std::endl;
    #endif

    for (unsigned lvl = 0; lvl < num_storage_levels; lvl++) {
      for (unsigned ds_idx = 0; ds_idx < num_data_spaces; ds_idx++) {
        uint64_t choice = splitting_choice_per_lvl_per_ds[lvl][ds_idx];
        if (!(choice < multi_rank_splitting_options_per_level_per_ds_[lvl][ds_idx].size())){
          #ifdef DEBUG_CONSTRUCTION_LAYOUT
            std::cout << "Note: Do not need to split for storage level " << lvl << ", dataspace " << ds_idx << " because data fits in line capacity." << std::endl;
          #endif
          continue;
        }
        const auto& multi_rank_option = multi_rank_splitting_options_per_level_per_ds_[lvl][ds_idx][choice];

        for (const auto& unique_rank : multi_rank_option.ranks)
        {
          uint32_t splitting_factor = multi_rank_option.splitting_factors.at(unique_rank);
          {
            std::stringstream ss;
            ss << "Layout vector size (" << layout_.size() << ") is smaller than or equal to current level index (" << lvl << ")";
            assert(layout_.size() > lvl && ss.str().c_str());
          }
          {
            std::stringstream ss;
            ss << "Intraline vector size (" << layout_[lvl].intraline.size() << ") is smaller than or equal to current dataspace index (" << ds_idx << ")";
            assert(layout_[lvl].intraline.size() > ds_idx && ss.str().c_str());
          }
          {
            std::stringstream ss;
            ss << "Interline vector size (" << layout_[lvl].interline.size() << ") is smaller than or equal to current dataspace index (" << ds_idx << ")";
            assert(layout_[lvl].interline.size() > ds_idx && ss.str().c_str());
          }
          // Get references to both nests for the specific dataspace
          auto& intraline_nest = layout_[lvl].intraline[ds_idx];
          auto& interline_nest = layout_[lvl].interline[ds_idx];

          if (intraline_nest.factors.find(unique_rank) == intraline_nest.factors.end() || interline_nest.factors.find(unique_rank) == interline_nest.factors.end())
          {
            Status error_status;
            error_status.success = false;
            error_status.fail_reason = "Rank " + unique_rank + " not found in intraline or interline nest for level " + std::to_string(lvl) + ", dataspace " + std::to_string(ds_idx);
            return {error_status};
          }

          // Get current factors from the layout (not from stored original values)
          uint32_t current_intraline_factor = (intraline_nest.factors.find(unique_rank) != intraline_nest.factors.end()
                                              ? intraline_nest.factors.at(unique_rank) : 1);
          uint32_t current_interline_factor = (interline_nest.factors.find(unique_rank) != interline_nest.factors.end()
                                              ? interline_nest.factors.at(unique_rank) : 1);

          // Validate that the splitting factor divides the current intraline factor
          if (current_intraline_factor % splitting_factor != 0)
          {
            Status error_status;
            error_status.success = false;
            error_status.fail_reason = "Multi-rank splitting factor " + std::to_string(splitting_factor) +
                                      " does not divide current intraline factor " + std::to_string(current_intraline_factor) +
                                      " for rank " + unique_rank + " at level " + std::to_string(lvl) + ", dataspace " + std::to_string(ds_idx);
            return {error_status};
          }

          // Split the factor: move splitting_factor from intraline to interline
          assert(splitting_factor > 0 && "Division by zero in splitting factor calculation");
          uint32_t new_intraline_factor = current_intraline_factor / splitting_factor;
          uint32_t new_interline_factor = current_interline_factor * splitting_factor;

          #ifdef DEBUG_CONSTRUCTION_LAYOUT
            std::cout << "[SplittingSpace] Storage storage level " << lvl << ", DataSpace " << ds_idx
                      << ", Rank '" << unique_rank << "': Multi-rank splitting factor " << splitting_factor
                      << " from intraline to interline" << std::endl;
            std::cout << "  - intraline factor: " << current_intraline_factor << " -> " << new_intraline_factor
                      << " (divided by " << splitting_factor << ")" << std::endl;
            std::cout << "  - interline factor: " << current_interline_factor << " -> " << new_interline_factor
                      << " (multiplied by " << splitting_factor << ")" << std::endl;
          #endif

          // Apply the changes
          intraline_nest.factors[unique_rank] = new_intraline_factor;
          interline_nest.factors[unique_rank] = new_interline_factor;
        }
      }
    }

    // Apply PackingSpace choices (multi-rank packing: interline-to-intraline)
    #ifdef DEBUG_CONSTRUCTION_LAYOUT
      std::cout << "[PackingSpace] Applying multi-rank packing for all dataspaces..." << std::endl;
    #endif
    for (unsigned lvl = 0; lvl < num_storage_levels; lvl++) {
      for (unsigned ds_idx = 0; ds_idx < num_data_spaces; ds_idx++) {
        uint64_t choice = packing_choice_per_lvl_per_ds[lvl][ds_idx];
        if (!(choice < multi_rank_packing_options_per_level_per_ds_[lvl][ds_idx].size())){
          #ifdef DEBUG_CONSTRUCTION_LAYOUT
            std::cout << "Note: Do not need to pack for storage level " << lvl << ", dataspace " << ds_idx << " because no data could be fitted into a line capacity." << std::endl;
          #endif
          continue;
        }
        const auto& multi_rank_option = multi_rank_packing_options_per_level_per_ds_[lvl][ds_idx][choice];

        for (const auto& unique_rank : multi_rank_option.ranks)
        {
          uint32_t packing_factor = multi_rank_option.packing_factors.at(unique_rank);
          {
            std::stringstream ss;
            ss << "Layout vector size (" << layout_.size() << ") is smaller than or equal to current level index (" << lvl << ")";
            assert(layout_.size() > lvl && ss.str().c_str());
          }
          {
            std::stringstream ss;
            ss << "Intraline vector size (" << layout_[lvl].intraline.size() << ") is smaller than or equal to current dataspace index (" << ds_idx << ")";
            assert(layout_[lvl].intraline.size() > ds_idx && ss.str().c_str());
          }
          {
            std::stringstream ss;
            ss << "Interline vector size (" << layout_[lvl].interline.size() << ") is smaller than or equal to current dataspace index (" << ds_idx << ")";
            assert(layout_[lvl].interline.size() > ds_idx && ss.str().c_str());
          }
          // Get references to both nests for the specific dataspace
          auto& intraline_nest = layout_[lvl].intraline[ds_idx];
          auto& interline_nest = layout_[lvl].interline[ds_idx];

          if (intraline_nest.factors.find(unique_rank) == intraline_nest.factors.end() || interline_nest.factors.find(unique_rank) == interline_nest.factors.end())
          {
            Status error_status;
            error_status.success = false;
            error_status.fail_reason = "Rank " + unique_rank + " not found in intraline or interline nest for level " + std::to_string(lvl) + ", dataspace " + std::to_string(ds_idx);
            return {error_status};
          }

          // Get current factors from the layout (not from stored original values)
          uint32_t current_intraline_factor = (intraline_nest.factors.find(unique_rank) != intraline_nest.factors.end()
                                              ? intraline_nest.factors.at(unique_rank) : 1);
          uint32_t current_interline_factor = (interline_nest.factors.find(unique_rank) != interline_nest.factors.end()
                                              ? interline_nest.factors.at(unique_rank) : 1);

          // Validate that the packing factor divides the current interline factor
          if (current_interline_factor % packing_factor != 0)
          {
            Status error_status;
            error_status.success = false;
            error_status.fail_reason = "Multi-rank packing factor " + std::to_string(packing_factor) +
                                      " does not divide current interline factor " + std::to_string(current_interline_factor) +
                                      " for rank " + unique_rank + " at level " + std::to_string(lvl) + ", dataspace " + std::to_string(ds_idx);
            return {error_status};
          }

          // Pack the factor: move packing_factor from interline to intraline
          uint32_t new_intraline_factor = current_intraline_factor * packing_factor;
          uint32_t new_interline_factor = current_interline_factor / packing_factor;

          #ifdef DEBUG_CONSTRUCTION_LAYOUT
            std::cout << "[PackingSpace] Storage storage level " << lvl << ", DataSpace " << ds_idx
                      << ", Rank '" << unique_rank << "': Multi-rank packing factor " << packing_factor
                      << " from interline to intraline" << std::endl;
            std::cout << "  - intraline factor: " << current_intraline_factor << " -> " << new_intraline_factor
                      << " (multiplied by " << packing_factor << ")" << std::endl;
            std::cout << "  - interline factor: " << current_interline_factor << " -> " << new_interline_factor
                      << " (divided by " << packing_factor << ")" << std::endl;
          #endif

          // Apply the changes
          intraline_nest.factors[unique_rank] = new_intraline_factor;
          interline_nest.factors[unique_rank] = new_interline_factor;
        }
      }
    }

    // Copy the modified layout to the output parameter
    if (layouts != nullptr)
    {
      *layouts = layout_;
    }

    #ifdef DEBUG_CONSTRUCTION_LAYOUT
      std::cout << "\n=== LAYOUT CONSTRUCTION COMPLETE ===" << std::endl;
      std::cout << "Final modified layout:" << std::endl;
      layout::PrintOverallLayoutConcise(layout_);
    #endif

    for (unsigned lvl = 0; lvl < num_storage_levels; lvl++)
    {
      for (unsigned ds_idx = 0; ds_idx < num_data_spaces; ds_idx++)
      {
        // Check if this dataspace is bypassed at this storage level
        bool is_kept = mapping.datatype_bypass_nest.at(ds_idx).test(lvl);

        if (is_kept)
        {
          uint64_t intraline_per_ds = 1;
          auto intra_nest = layout_.at(lvl).intraline.at(ds_idx);
          for (const auto &r : intra_nest.ranks) // Analyze slowdown per rank
          {
          int factor = (intra_nest.factors.find(r) != intra_nest.factors.end() ? intra_nest.factors.at(r) : 1);
            intraline_per_ds *= factor;
          }
          if (intraline_per_ds > storage_level_line_capacity[lvl]){
            std::cout << "layout not satisfies the internal constraints" << std::endl;
            layout::PrintOverallLayout(layout_);
            throw std::runtime_error("Dataspace[" + std::to_string(ds_idx) + "] intraline size " + std::to_string(intraline_per_ds) + " exceeds storage level line capacity " + std::to_string(storage_level_line_capacity[lvl]) + " at level " + std::to_string(lvl));
          }
        }
      }
    }

    Status success_status;
    success_status.success = true;
    success_status.fail_reason = "";

    return {success_status};
  }


  //
  // CreateConcordantLayout() - Step 1: Create layout from mapping
  //
  void Legal::CreateConcordantLayout(const Mapping& mapping)
  {
    #ifdef DEBUG_CONSTRUCTION_LAYOUT
      std::cout << "Step 1: Create Concordant Layout..." << std::endl;
      std::cout << "Total number of storage levels: " << mapping.loop_nest.storage_tiling_boundaries.size() << std::endl;
      std::cout << "Total number of layout levels: " << layout_.size() << std::endl;
      assert(mapping.loop_nest.storage_tiling_boundaries.size() == layout_.size());
      std::cout << "Total number of data spaces: " << layout_.at(0).intraline.size() << std::endl;
    #endif

    // Build a initialized map that assigns 1 to every dimension ID present in dim_order.
    std::map<std::uint32_t, std::uint32_t> initial_dimid_to_loopend;
    for (char dim_char : layout_.at(0).dim_order)
    {
      // Convert the char stored in dim_order to a std::string so it can be used
      // as a key into the dimensionToDimID map.
      std::string dim_name(1, dim_char);

      // Look up the dimension ID associated with this name.
      auto dim_id_itr = layout_.at(0).dimensionToDimID.find(dim_name);
      if (dim_id_itr == layout_.at(0).dimensionToDimID.end())
      {
        std::cerr << "ERROR: dimension name " << dim_name << " not found in dimensionToDimID map." << std::endl;
        throw std::runtime_error("Invalid dimension name in dim_order");
      }

      initial_dimid_to_loopend[dim_id_itr->second] = 1;
    }

    /*
        Step 1: Collect the interline nested loop and intraline nested loop.
    */
    num_storage_levels = mapping.loop_nest.storage_tiling_boundaries.size();
    num_data_spaces = layout_.at(0).intraline.size();
    unsigned num_loops = mapping.loop_nest.loops.size();
    unsigned inv_storage_level = num_storage_levels;

    // Each storage level vector element starts as a copy of the prototype map.
    std::vector<std::map<std::uint32_t, std::uint32_t>> storage_level_interline_dimid_to_loopend(mapping.loop_nest.storage_tiling_boundaries.size(), initial_dimid_to_loopend);
    std::vector<std::map<std::uint32_t, std::uint32_t>> storage_level_intraline_dimid_to_loopend(mapping.loop_nest.storage_tiling_boundaries.size(), initial_dimid_to_loopend);
    std::vector<std::map<std::uint32_t, std::uint32_t>> storage_level_overall_dimval(mapping.loop_nest.storage_tiling_boundaries.size(), initial_dimid_to_loopend);

    for (unsigned loop_level = num_loops-1; loop_level != static_cast<unsigned>(-1); loop_level--)
    {
      if (inv_storage_level > 0 &&
          mapping.loop_nest.storage_tiling_boundaries.at(inv_storage_level-1) == loop_level)
      {
        inv_storage_level--;
      }

      if (loop::IsSpatial(mapping.loop_nest.loops.at(loop_level).spacetime_dimension))
      {
        storage_level_intraline_dimid_to_loopend[inv_storage_level][mapping.loop_nest.loops.at(loop_level).dimension] = mapping.loop_nest.loops.at(loop_level).end;
      }else{
        storage_level_interline_dimid_to_loopend.at(inv_storage_level)[mapping.loop_nest.loops.at(loop_level).dimension] = mapping.loop_nest.loops.at(loop_level).end;
      }
    }

    for(unsigned lvl=0; lvl < storage_level_intraline_dimid_to_loopend.size(); lvl++){
      for (unsigned i = 0; i < num_data_spaces; i++){ // iterate over all data
        for (const auto& kv : storage_level_interline_dimid_to_loopend[lvl])
        {
          storage_level_overall_dimval[lvl][kv.first] = storage_level_intraline_dimid_to_loopend[lvl][kv.first] * storage_level_interline_dimid_to_loopend[lvl][kv.first];
        }
      }
    }

    // Calculate cumulative product from end to first index
    cumulatively_intraline_dimval.resize(storage_level_intraline_dimid_to_loopend.size());

    // Initialize all levels with the initial map
    for (unsigned lvl = 0; lvl < cumulatively_intraline_dimval.size(); lvl++)
    {
      cumulatively_intraline_dimval[lvl] = initial_dimid_to_loopend;
    }

    // Initialize the last level (no multiplication needed)
    if (!storage_level_intraline_dimid_to_loopend.empty())
    {
      cumulatively_intraline_dimval[0] = storage_level_intraline_dimid_to_loopend[0];

      // Calculate cumulative product from second-to-last level backwards to first level
      for (int lvl = 1; lvl < static_cast<int>(storage_level_intraline_dimid_to_loopend.size()); lvl++)
      {
        bool is_spatial = false;
        for (const auto& kv : storage_level_intraline_dimid_to_loopend[lvl]) {
          if (kv.second > 1) {
            is_spatial = true;
            break;
          }
        }
        for (const auto& kv : storage_level_intraline_dimid_to_loopend[lvl])
        {
          std::uint32_t dim_id = kv.first;
          std::uint32_t current_value = kv.second;

          // Multiply current level value with cumulative product from next level
          if (is_spatial && cumulatively_intraline_dimval[lvl - 1].find(dim_id) != cumulatively_intraline_dimval[lvl - 1].end())
          {
            cumulatively_intraline_dimval[lvl][dim_id] = current_value * cumulatively_intraline_dimval[lvl - 1][dim_id];
          }
          else
          {
            cumulatively_intraline_dimval[lvl][dim_id] = current_value;
          }
        }
      }
    }

    // Calculate cumulative product from end to first index
    cumulatively_product_dimval.resize(storage_level_overall_dimval.size());

    // Initialize all levels with the initial map
    for (unsigned lvl = 0; lvl < cumulatively_product_dimval.size(); lvl++)
    {
      cumulatively_product_dimval[lvl] = initial_dimid_to_loopend;
    }

    // Initialize the last level (no multiplication needed)
    if (!storage_level_overall_dimval.empty())
    {
      cumulatively_product_dimval[0] = storage_level_overall_dimval[0];

      // Calculate cumulative product from second-to-last level backwards to first level
      for (int lvl = 1; lvl < static_cast<int>(storage_level_overall_dimval.size()); lvl++)
      {
        for (const auto& kv : storage_level_overall_dimval[lvl])
        {
          std::uint32_t dim_id = kv.first;
          std::uint32_t current_value = kv.second;

          // Multiply current level value with cumulative product from next level
          if (cumulatively_product_dimval[lvl - 1].find(dim_id) != cumulatively_product_dimval[lvl - 1].end())
          {
            cumulatively_product_dimval[lvl][dim_id] = current_value * cumulatively_product_dimval[lvl - 1][dim_id];
          }
          else
          {
            cumulatively_product_dimval[lvl][dim_id] = current_value;
          }
        }
      }
    }

    /*
        Step 2: Print out the collapsed interline nested loop and intraline nested loop.
    */
    #ifdef DEBUG_CONCORDANT_LAYOUT
      std::cout << "storage_level_interline_dimid_to_loopend:" << std::endl;
      for (unsigned lvl = 0; lvl < storage_level_interline_dimid_to_loopend.size(); lvl++) // iterate over all storage levels
      {
        std::cout << "storage level=" << lvl << std::endl;
        for (const auto& kv : storage_level_interline_dimid_to_loopend[lvl])
        {
          std::cout << layout_.at(0).dim_order[kv.first] << ":" << kv.second << " ";
        }
        std::cout << std::endl;
      }

      std::cout << "storage_level_intraline_dimid_to_loopend:" << std::endl;
      for (unsigned lvl = 0; lvl < storage_level_intraline_dimid_to_loopend.size(); lvl++) // iterate over all storage levels
      {
        std::cout << "storage level=" << lvl << std::endl;
        for (const auto& kv : storage_level_intraline_dimid_to_loopend[lvl])
        {
          std::cout << layout_.at(0).dim_order[kv.first] << ":" << kv.second << " ";
        }
        std::cout << std::endl;
      }

      std::cout << "storage_level_overall_dimval:" << std::endl;
      for (unsigned lvl = 0; lvl < storage_level_overall_dimval.size(); lvl++) // iterate over all storage levels
      {
        std::cout << "storage level=" << lvl << std::endl;
        for (const auto& kv : storage_level_overall_dimval[lvl])
        {
          std::cout << layout_.at(0).dim_order[kv.first] << ":" << kv.second << " ";
        }
        std::cout << std::endl;
      }

      std::cout << "cumulatively_product_dimval:" << std::endl;
      for (unsigned lvl = 0; lvl < cumulatively_product_dimval.size(); lvl++) // iterate over all storage levels
      {
        std::cout << "storage level=" << lvl << std::endl;
        for (const auto& kv : cumulatively_product_dimval[lvl])
        {
          std::cout << layout_.at(0).dim_order[kv.first] << ":" << kv.second << " ";
        }
        std::cout << std::endl;
      }
    #endif

    /*
        Step 3: Assign collapsed nested loop to the layout.
    */
    for(unsigned lvl=0; lvl < cumulatively_intraline_dimval.size(); lvl++){
      for (unsigned i = 0; i < num_data_spaces; i++){ // iterate over all data spaces
        for(auto & rank: layout_.at(lvl).intraline.at(i).ranks){ // iterate over all ranks of the data space
          const auto& dim_ids = layout_.at(lvl).rankToFactorizedDimensionID.at(rank);
          uint32_t total_intraline = 0;
          uint32_t total_rank_size = 0;
          const auto& coefficient = layout_.at(lvl).rankToCoefficientValue[rank];
          uint32_t zero_padding = 0;
          if (lvl == cumulatively_intraline_dimval.size()-1 && 
              layout_.at(lvl).rankToZeroPadding.find(rank) != layout_.at(lvl).rankToZeroPadding.end()) {
            zero_padding = layout_.at(lvl).rankToZeroPadding.at(rank);
          }
          for (unsigned idx=0; idx < dim_ids.size(); idx++){
            auto dim_intraline_value = cumulatively_intraline_dimval[lvl][dim_ids[idx]];
            auto dim_total_value = cumulatively_product_dimval[lvl][dim_ids[idx]];
            if (dim_ids.size() > 1){
              if (dim_intraline_value == 1){
                if (idx < dim_ids.size()-1){
                  total_intraline += dim_intraline_value;
                }
              }
              else{
                if (idx < dim_ids.size()-1){
                  total_intraline += dim_intraline_value*coefficient[idx];
                }
                else{
                  total_intraline += dim_intraline_value*coefficient[idx] - 1;
                }
              }

              if (dim_total_value == 1){
                if (idx < dim_ids.size()-1){
                  total_rank_size += dim_total_value;
                }
              }
              else{
                if (idx < dim_ids.size()-1){
                  total_rank_size += dim_total_value*coefficient[idx];
                }
                else{
                  total_rank_size += dim_total_value*coefficient[idx] - 1;
                }
              }
            }
            else{
              total_intraline += dim_intraline_value;
              total_rank_size += dim_total_value;
            }
          }
          assert(total_intraline > 0 && "Division by zero in total_interline calculation");
          auto total_interline = (total_rank_size - 2*zero_padding + total_intraline - 1) / total_intraline;

          if (mapping.datatype_bypass_nest.at(i).test(lvl)) {
            layout_.at(lvl).intraline.at(i).factors.at(rank) = total_intraline;
            layout_.at(lvl).interline.at(i).factors.at(rank) = total_interline;
          } else {
            layout_.at(lvl).intraline.at(i).factors.at(rank) = 1;
            layout_.at(lvl).interline.at(i).factors.at(rank) = total_rank_size;
          }
          #ifdef DEBUG_CONCORDANT_LAYOUT
            std::cout << "level=" << lvl << " dataspace=" << i << " rank=" << rank << " intraline = " << total_intraline << " interline = " << total_interline << std::endl;
          #endif
        }
      }
    }

    #ifdef DEBUG_CONCORDANT_LAYOUT
    std::cout << "layout_after_concordant_layout:" << std::endl;
    layout::PrintOverallLayout(layout_);
    #endif
  }

  //
  // CreateIntralineFactorSpace() - Step 3: Generate all possible intraline factor combinations (SplittingSpace and PackingSpace)
  //
  void Legal::CreateIntralineFactorSpace(model::Engine::Specs arch_specs, const Mapping& mapping)
  {
    (void) arch_specs; // Suppress unused parameter warning
    assert(num_storage_levels > 0 && "num_storage_levels is out of range");
    assert(num_data_spaces > 0 && "num_data_spaces is out of range");
    assert(storage_level_line_capacity.size() > 0 && "storage_level_line_capacity has members");
    assert(storage_level_keep_factor.size() > 0 && "storage_level_keep_factor has members");

    #ifdef DEBUG_CREATE_INTRALINE_FACTOR_SPACE
      std::cout << "Step 2: Creating SplittingSpace and PackingSpace candidates from intraline factors..." << std::endl;
    #endif

    // Clear previous design spaces
    multi_rank_splitting_options_per_level_per_ds_.clear();
    multi_rank_splitting_options_per_level_per_ds_.resize(num_storage_levels, std::vector<std::vector<MultiRankSplittingOption>>(num_data_spaces, std::vector<MultiRankSplittingOption>()));
    multi_rank_packing_options_per_level_per_ds_.clear();
    multi_rank_packing_options_per_level_per_ds_.resize(num_storage_levels, std::vector<std::vector<MultiRankPackingOption>>(num_data_spaces, std::vector<MultiRankPackingOption>()));
    uint64_t max_intraline_to_interline_factor = 0;

    // Phase 1: Get Memory Line size for all storage levels (What Layout Provide Per Cycle)
    std::vector<std::vector<std::uint64_t>> intraline_size_per_ds(num_storage_levels, std::vector<std::uint64_t>(num_data_spaces, 0));

    for (unsigned lvl = 0; lvl < num_storage_levels; lvl++)
    {
      for (unsigned ds_idx = 0; ds_idx < num_data_spaces; ds_idx++)
      {
        // Check if this dataspace is bypassed at this storage level
        bool is_kept = mapping.datatype_bypass_nest.at(ds_idx).test(lvl);

        if (is_kept)
        {
          uint64_t intraline_per_ds = 1;
          auto intra_nest = layout_.at(lvl).intraline.at(ds_idx);
          for (const auto &r : intra_nest.ranks) // Analyze slowdown per rank
          {
          int factor = (intra_nest.factors.find(r) != intra_nest.factors.end() ? intra_nest.factors.at(r) : 1);
            intraline_per_ds *= factor;
          }
          intraline_size_per_ds[lvl][ds_idx] = intraline_per_ds;
        }
      }
    }

    // Phase 2: Check if the line capacity is sufficient for the intraline size
    // First, determine which levels require splitting (intraline_size > line_capacity)
    level_ds_requires_splitting_.resize(num_storage_levels, std::vector<bool>(num_data_spaces, false));
    #ifdef DEBUG_CREATE_INTRALINE_FACTOR_SPACE
      std::cout << "Phase 2.1: quick glance at the intraline size and line capacity" << std::endl;
      for (unsigned lvl = 0; lvl < num_storage_levels; lvl++){
        for (unsigned ds_idx = 0; ds_idx < num_data_spaces; ds_idx++){
          if(storage_level_line_capacity[lvl] < intraline_size_per_ds[lvl][ds_idx]){
            level_ds_requires_splitting_[lvl][ds_idx] = true;
            std::cout << "  storage level " << lvl << ": dataspace " << ds_idx << " requires splitting (intraline_size=" << intraline_size_per_ds[lvl][ds_idx]
                    << " > line_capacity=" << storage_level_line_capacity[lvl] << ")" << std::endl;
          } else if (storage_level_line_capacity[lvl] > intraline_size_per_ds[lvl][ds_idx]){
            level_ds_requires_splitting_[lvl][ds_idx] = false;
            std::cout << "  storage level " << lvl << ": dataspace " << ds_idx << "  requires packing  (intraline_size=" << intraline_size_per_ds[lvl][ds_idx]
                      << " < line_capacity=" << storage_level_line_capacity[lvl] << ")";
            if (storage_level_keep_factor[lvl][ds_idx] == 0)
              std::cout << " - bypass" << std::endl;
            else
              std::cout << std::endl;
          }
          else{
            std::cout << "  storage level " << lvl << ": dataspace " << ds_idx << "     do nothing     (intraline_size=" << intraline_size_per_ds[lvl][ds_idx]
            << " == line_capacity=" << storage_level_line_capacity[lvl] << ")" << std::endl;
          }
        }
      }
    #endif

    #ifdef DEBUG_CREATE_INTRALINE_FACTOR_SPACE
      std::cout << "Phase 2.2: generate design space for factor conversions..." << std::endl;
    #endif
    for (unsigned lvl = 0; lvl < num_storage_levels; lvl++){
      // First analyze single-rank and multi-rank splitting possibilities for each dataspace.
      for(unsigned ds_idx = 0; ds_idx < num_data_spaces; ds_idx++){ // single
        if(storage_level_line_capacity[lvl] < intraline_size_per_ds[lvl][ds_idx] && storage_level_keep_factor[lvl][ds_idx]){
          // The product of all factors of intraline for a dataspace is too big to fit in the line capacity,
          // so need to reduce the factors of intraline by converting some factors into interline.
          #ifdef DEBUG_CREATE_INTRALINE_FACTOR_SPACE
            std::cout << "  storage level " << lvl << ": dataspace " << ds_idx << " intraline_size (" << intraline_size_per_ds[lvl][ds_idx]
                    << ") exceeds line capacity (" << storage_level_line_capacity[lvl]
                    << "). Generating design space for factor conversions..." << std::endl;
          #endif
          // Calculate maximum packing factor that can be applied
          assert(storage_level_line_capacity[lvl] > 0 && "Division by zero in max_splitting_factor calculation");
          uint32_t max_splitting_factor = static_cast<uint32_t>((static_cast<float>(intraline_size_per_ds[lvl][ds_idx]) + static_cast<float>(storage_level_line_capacity[lvl]) - 1) / static_cast<float>(storage_level_line_capacity[lvl]));
          if (max_splitting_factor > 1){
            #ifdef DEBUG_CREATE_INTRALINE_FACTOR_SPACE
              std::cout << "    Maximum splitting factor: " << max_splitting_factor << std::endl;
            #endif
            auto& intraline_nest = layout_.at(lvl).intraline.at(ds_idx);
            std::map<std::string, std::vector<uint32_t>> all_candidate_factors_per_rank;

            // First, analyze single-rank splitting possibilities for each rank.
            for (const auto& rank : intraline_nest.ranks) {
              uint32_t current_intraline_factor = (intraline_nest.factors.find(rank) != intraline_nest.factors.end()
                                                  ? intraline_nest.factors.at(rank) : 1);
              if (current_intraline_factor > 1) {
                std::vector<uint32_t> divisors = FindDivisors(current_intraline_factor);
                std::vector<uint32_t> valid_factors;

                // Store all divisors > 1 as candidate factors for multi-rank combinations
                for (uint32_t divisor : divisors) {
                  if (divisor > 1) {
                    valid_factors.push_back(divisor);
                  }
                }

                // Collect all possible splitting factors for each rank (for multi-rank combinations)
                if (!valid_factors.empty()) {
                  all_candidate_factors_per_rank[rank] = valid_factors;
                }
              }
            }

            // Multi-rank splitting: find combinations of ranks that together can reduce intraline size to fit
            // Use all candidate factors (including those that don't fit individually)
            #ifdef DEBUG_CREATE_INTRALINE_FACTOR_SPACE
              std::cout << "      Multi-rank splitting analysis:" << std::endl;
            #endif
            std::vector<std::vector<std::string>> rank_combinations = GenerateRankCombinations(intraline_nest.ranks);

            for (const auto& rank_combo : rank_combinations) {
              // Check if all ranks in combination have candidate factors
              bool all_ranks_have_factors = true;
              for (const auto& rank : rank_combo) {
                if (all_candidate_factors_per_rank.find(rank) == all_candidate_factors_per_rank.end()) {
                  all_ranks_have_factors = false;
                  break;
                }
              }

              if (!all_ranks_have_factors) continue;

              MultiRankSplittingOption multi_option;
              if (TestMultiRankSplittingWithCandidates(lvl, ds_idx, rank_combo, all_candidate_factors_per_rank,
                                                    intraline_size_per_ds, storage_level_line_capacity[lvl], multi_option)) {
                multi_rank_splitting_options_per_level_per_ds_[lvl][ds_idx].push_back(multi_option);

                #ifdef DEBUG_CREATE_INTRALINE_FACTOR_SPACE
                  std::cout << "        --> Valid multi-rank option: Ranks [";
                  for (size_t i = 0; i < rank_combo.size(); i++) {
                    std::cout << rank_combo[i];
                    if (i < rank_combo.size() - 1) std::cout << ", ";
                  }
                  std::cout << "], total_reduction for dataspace[" << ds_idx << "] = " << multi_option.total_reduction << std::endl;

                  // Print individual splitting factors
                  for (const auto& rank : rank_combo) {
                    std::cout << "          " << rank << "(intraline): " << multi_option.original_intraline_factors.at(rank)
                              << " -> " << (multi_option.original_intraline_factors.at(rank) / multi_option.splitting_factors.at(rank))
                              << " (split by " << multi_option.splitting_factors.at(rank) << ")" << std::endl;
                  }
                #endif
              }
            }
          }
          else{
            #ifdef DEBUG_CREATE_INTRALINE_FACTOR_SPACE
              std::cout << "  storage level " << lvl << ": splitting factor = 1, no splitting is needed." << std::endl;
            #endif
          }
          // Calculate maximum splitting factor that can be applied
        }
        else if (storage_level_line_capacity[lvl] > intraline_size_per_ds[lvl][ds_idx] && storage_level_keep_factor[lvl][ds_idx]){
          // Intraline has free space to hold more data, could convert some factors of interline into intraline,
          // this creates the overall design spaces

          #ifdef DEBUG_CREATE_INTRALINE_FACTOR_SPACE
            std::cout << "  storage level " << lvl << " Dataspace[" << ds_idx << "] intraline_size (" << intraline_size_per_ds[lvl][ds_idx]
                      << ") has " << (storage_level_line_capacity[lvl] - intraline_size_per_ds[lvl][ds_idx])
                      << " free capacity. Generating design space for data packing..." << std::endl;
          #endif

          // Calculate maximum packing factor that can be applied
          assert(intraline_size_per_ds[lvl][ds_idx] > 0 && "Division by zero in max_packing_factor calculation");
          uint32_t max_packing_factor = static_cast<uint32_t>(static_cast<float>(storage_level_line_capacity[lvl]) / static_cast<float>(intraline_size_per_ds[lvl][ds_idx]));
          if (max_packing_factor > 1){
            #ifdef DEBUG_CREATE_INTRALINE_FACTOR_SPACE
              std::cout << "    Maximum packing factor: " << max_packing_factor << std::endl;
            #endif

            auto& inter_nest = layout_.at(lvl).interline.at(ds_idx);

            // Multi-rank packing analysis within each dataspace
            #ifdef DEBUG_CREATE_INTRALINE_FACTOR_SPACE
              std::cout << "    DataSpace " << ds_idx << " Multi-rank packing analysis:" << std::endl;
            #endif

            // Collect all ranks and their candidate factors from this dataspace
            std::map<std::string, std::vector<uint32_t>> all_candidate_factors_per_rank;
            for (const auto& rank : inter_nest.ranks) {
              uint32_t current_interline_factor = (inter_nest.factors.find(rank) != inter_nest.factors.end()
                                                  ? inter_nest.factors.at(rank) : 1);

              if (current_interline_factor > 1) {
                std::vector<uint32_t> divisors = FindDivisors(current_interline_factor);
                std::vector<uint32_t> valid_factors;

                for (uint32_t divisor : divisors) {
                  valid_factors.push_back(divisor);
                }

                if (!valid_factors.empty()) {
                  all_candidate_factors_per_rank[rank] = valid_factors;
                }
              }
            }

            if (all_candidate_factors_per_rank.size() >= 2) {
              std::vector<std::string> all_ranks;
              for (const auto& entry : all_candidate_factors_per_rank) {
                all_ranks.push_back(entry.first);
              }

              std::vector<std::vector<std::string>> rank_combinations = {all_ranks};

              for (auto rank_combo = rank_combinations.rbegin(); rank_combo != rank_combinations.rend(); ++rank_combo) {
                std::vector<MultiRankPackingOption> multi_options;
                if (TestMultiRankPackingWithCandidates(lvl, ds_idx, *rank_combo, all_candidate_factors_per_rank,
                                                    intraline_size_per_ds, storage_level_line_capacity[lvl], multi_options)) {
                  for (auto multi_option : multi_options) {
                    if ((multi_option.total_packing > PACKING_PRUNING_RATIO*max_intraline_to_interline_factor) && ((intraline_size_per_ds[lvl][ds_idx]*multi_option.total_packing) <= storage_level_line_capacity[lvl]) ){// pruning low packing options
                      multi_rank_packing_options_per_level_per_ds_[lvl][ds_idx].push_back(multi_option);
                      uint32_t max_possible_factor = (storage_level_line_capacity[lvl] + intraline_size_per_ds[lvl][ds_idx] - 1)/ intraline_size_per_ds[lvl][ds_idx];
                      if (multi_option.total_packing > max_intraline_to_interline_factor){
                        max_intraline_to_interline_factor = multi_option.total_packing;
                        if (max_intraline_to_interline_factor > max_possible_factor)
                          max_intraline_to_interline_factor = max_possible_factor;
                      }
                      #ifdef DEBUG_CREATE_INTRALINE_FACTOR_SPACE
                        uint32_t size_option = multi_rank_packing_options_per_level_per_ds_[lvl][ds_idx].size()-1;
                        std::cout << "      --> Valid multi-rank packing option: Ranks [";
                        for (size_t i = 0; i < rank_combo->size(); i++) {
                          std::cout << (*rank_combo)[i];
                          if (i < rank_combo->size() - 1) std::cout << ", ";
                        }
                        std::cout << "], total_packing=" << multi_rank_packing_options_per_level_per_ds_[lvl][ds_idx][size_option].total_packing << std::endl;

                        // Print individual packing factors
                        for (const auto& rank : *rank_combo) {
                          assert(multi_rank_packing_options_per_level_per_ds_[lvl][ds_idx][size_option].packing_factors.at(rank) > 0 && "Division by zero in print packing factor");
                          std::cout << "        " << rank << "(interline): " << multi_rank_packing_options_per_level_per_ds_[lvl][ds_idx][size_option].original_interline_factors.at(rank)
                                    << " -> " << (multi_rank_packing_options_per_level_per_ds_[lvl][ds_idx][size_option].original_interline_factors.at(rank) / multi_rank_packing_options_per_level_per_ds_[lvl][ds_idx][size_option].packing_factors.at(rank))
                                    << " (pack by " << multi_rank_packing_options_per_level_per_ds_[lvl][ds_idx][size_option].packing_factors.at(rank) << ")" << std::endl;
                        }
                      #endif
                    }
                  }
                }
              }
            } else {
              #ifdef DEBUG_CREATE_INTRALINE_FACTOR_SPACE
                std::cout << "      Insufficient ranks for multi-rank combinations in this dataspace" << std::endl;
              #endif
            }

          }
          else{
            #ifdef DEBUG_CREATE_INTRALINE_FACTOR_SPACE
              std::cout << "  storage level " << lvl << ": packing factor = 1, no packing is needed." << std::endl;
            #endif
          }
        }
        // Do nothing if the line capacity is equal to the intraline size
      }
    }

    // cross_dataspace_multi_rank_splitting_options_per_level_
    splitting_candidates_per_lvl_per_ds.clear();
    splitting_candidates_per_lvl_per_ds.resize(num_storage_levels, std::vector<std::uint64_t>(num_data_spaces, 1));
    splitting_candidates = 1;
    for (unsigned lvl = 0; lvl < num_storage_levels; lvl++){
      for (unsigned ds_idx = 0; ds_idx < num_data_spaces; ds_idx++){
        if (multi_rank_splitting_options_per_level_per_ds_[lvl][ds_idx].empty()){
          splitting_candidates_per_lvl_per_ds[lvl][ds_idx] = 0;
        }
        else{
          splitting_candidates_per_lvl_per_ds[lvl][ds_idx] = multi_rank_splitting_options_per_level_per_ds_[lvl][ds_idx].size();
          splitting_candidates *= multi_rank_splitting_options_per_level_per_ds_[lvl][ds_idx].size();
        }
      }
    }

    // Print flattened splitting choices in table format
    #ifdef DEBUG_CREATE_INTRALINE_FACTOR_SPACE
      std::cout << "Breakdown of splitting candidates:" << std::endl;
      std::cout << "Level | DataSpace | Candidates" << std::endl;
      std::cout << "------|-----------|-----------" << std::endl;
      for (unsigned lvl = 0; lvl < num_storage_levels; lvl++) {
        for (unsigned ds_idx = 0; ds_idx < num_data_spaces; ds_idx++) {
          std::cout << std::setw(6) << lvl << " | "
                    << std::setw(9) << ds_idx << " | "
                    << std::setw(9) << splitting_candidates_per_lvl_per_ds[lvl][ds_idx] << std::endl;
        }
      }
      std::cout << std::endl;
    #endif

    // cross_dataspace_multi_rank_packing_options_per_level_
    packing_candidates_per_lvl_per_ds.clear();
    packing_candidates_per_lvl_per_ds.resize(num_storage_levels, std::vector<std::uint64_t>(num_data_spaces, 1));
    packing_candidates = 1;
    for (unsigned lvl = 0; lvl < num_storage_levels; lvl++){
      for (unsigned ds_idx = 0; ds_idx < num_data_spaces; ds_idx++){
        if (multi_rank_packing_options_per_level_per_ds_[lvl][ds_idx].empty()){
          packing_candidates_per_lvl_per_ds[lvl][ds_idx] = 0;
        }
        else{
          packing_candidates_per_lvl_per_ds[lvl][ds_idx] = multi_rank_packing_options_per_level_per_ds_[lvl][ds_idx].size();
          packing_candidates *= multi_rank_packing_options_per_level_per_ds_[lvl][ds_idx].size();
        }
      }
    }

    // Print flattened packing choices in table format
    #ifdef DEBUG_CREATE_INTRALINE_FACTOR_SPACE
      std::cout << "Breakdown of packing candidates:" << std::endl;
      std::cout << "Level | DataSpace | Candidates" << std::endl;
      std::cout << "------|-----------|-----------" << std::endl;
      for (unsigned lvl = 0; lvl < num_storage_levels; lvl++) {
        for (unsigned ds_idx = 0; ds_idx < num_data_spaces; ds_idx++) {
          std::cout << std::setw(6) << lvl << " | "
                    << std::setw(9) << ds_idx << " | "
                    << std::setw(9) << packing_candidates_per_lvl_per_ds[lvl][ds_idx] << std::endl;
        }
      }
      std::cout << std::endl;
    #endif
  }


  void Legal::SequentialFactorizeLayout(layout::Layouts& layout){
    for (unsigned lvl = 0; lvl < num_storage_levels; lvl++)
    {
      for (unsigned ds_idx = 0; ds_idx < num_data_spaces; ds_idx++)
      {
        // Check if this dataspace is bypassed at this storage level
        uint32_t intraline_per_ds = 1;
        bool is_kept = storage_level_keep_factor[lvl][ds_idx];

        if (is_kept)
        {
          auto intra_nest = layout.at(lvl).intraline.at(ds_idx);
          for (const auto &r : intra_nest.ranks) // Analyze slowdown per rank
          {
          int factor = (intra_nest.factors.find(r) != intra_nest.factors.end() ? intra_nest.factors.at(r) : 1);
            intraline_per_ds *= factor;
          }

          float splitting_factor = (float)intraline_per_ds / (float)storage_level_line_capacity[lvl];
          // Check if the intraline product of dataspaces is greater than the storage level line capacity
          for (const auto &r : layout.at(lvl).intraline.at(ds_idx).ranks)
          {
            if (layout.at(lvl).intraline.at(ds_idx).factors[r] > 1)
            {
              layout.at(lvl).interline.at(ds_idx).factors[r] *= layout.at(lvl).intraline.at(ds_idx).factors[r];
              splitting_factor = splitting_factor / (float) layout.at(lvl).intraline.at(ds_idx).factors[r];
              layout.at(lvl).intraline.at(ds_idx).factors[r] = 1;
            }
            if (splitting_factor < 1.0)
            {
              break;
            }
          }
        }
      }
    }
  }

} // namespace layoutspace
