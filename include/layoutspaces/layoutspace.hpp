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

#pragma once

#include <boost/multiprecision/cpp_int.hpp>

#include "util/numeric.hpp"
#include "layout/layout.hpp"
#include "model/engine.hpp"
#include "mapping/mapping.hpp"

using namespace boost::multiprecision;
// #define DEBUG 

namespace layoutspace
{

typedef uint32_t ID;

struct Status
{
  bool success;
  std::string fail_reason;
};

//--------------------------------------------//
//                    Legal                   //
//--------------------------------------------//

class Legal
{
  /*
                                                ┌────────────────────────────┐
                                                │ 1. DEFINE MAPPING SPACE    │
                                                │    (all legal loop‑nests)  │
                                                └────────────┬───────────────┘
                                                             │
                                                             ▼
                     ┌──────────────────────────────────────────────────────────┐
                     │ 2. ITERATE: pick next mapping M ∈ mapping‑space          │
                     │    – Identify the spatial loops at every memory level    │
                     │    – requested_parallelism = ∏ extents of those loops    │
                     └────────────┬─────────────────────────────────────────────┘
                                  │
                                  ▼
               ┌───────────────────────────────────────────────────────┐
               │ 3. FOR each 2‑D on‑chip buffer level L                │
               │    – line_cap(L) = words per line (hardware)          │
               └────────────┬──────────────────────────────────────────┘
                            │
                            ▼
               ┌────────────────────────────────────────────────────────┐
               │ 4. DECIDE: how does line_cap(L) compare to             │
               │    requested_parallelism (RP)?                         │
               └────────────┬──────────────────────┬────────────────────┐
                            │                      │                    │
                            ▼                      ▼                    ▼
            ┌─────────────────────┐  ┌─────────────────────┐   ┌─────────────────────┐
            │ 4A. RP == line_cap  │  │ 4B. RP  > line_cap  │   │ 4C. RP  < line_cap  │
            └──────────┬──────────┘  └──────────┬──────────┘   └──────────┬──────────┘
                       │                        │                         │
                       ▼                        ▼                         ▼
   ┌─────────────────────────────┐   ┌─────────────────────────────┐ ┌─────────────────────────────┐
   │ Case 1: Perfect fit.        │   │ Case 2: Line too small.     │ │ Case 3: Line has slack.     │
   │ • If exactly one dim in RP: │   │ • Enumerate partitions of   │ │ • Pack all RP data first.   │
   │   – Enumerate all factor‑   │   │   RP across ⌈RP/line_cap⌉   │ │ • Enumerate temporal‑loop   │
   │     izations of that dim.   │   │   lines (choose subset per  │ │   dimensions that can be    │
   │ • Else (>1 dims):           │   │   line).                    │ │   packed into remaining     │
   │   – Enumerate all flatten‑  │   │ • Continue until every line │ │   slots.                    │
   │     ings of the RP dims     │   │   layout fits in buffer.    │ │ • Continue until buffer‑    │
   │   – (choose which dims map  │   └─────────────────────────────┘ │   size constraint met.      │
   │     to row, order, etc.)    │                                   └─────────────────────────────┘
   └─────────────────────────────┘
                     │
                     ▼
        ┌──────────────────────────────────────────────────┐
        │ 5. FILTER layouts that violate any constraint:   │
        │    – #lines(layout,L) ≤ #physical_lines(L)       │
        │    – Data required by mapping M is contained     │
        │      within layout rows (no extra stalls).       │
        └────────────┬─────────────────────────────────────┘
                     │
                     ▼
        ┌──────────────────────────────────────────────────┐
        │ 6. EVALUATE each legal (M, layout) pair:         │
        │    – Timeloop cost model → {cycles, energy, …}   │
        │    – Record best‑of‑class metrics or Pareto set  │
        └────────────┬─────────────────────────────────────┘
                     │
                     ▼
        ┌─────────────────────────────────────────────────┐
        │ 7. ANY mappings left?                           │
        └───────┬──────────────────────────────┬──────────┘
                │Yes                           │No
                ▼                              ▼
         (Return to step 2)         ┌────────────────────────┐
                                    │ 8. OUTPUT optimal      │
                                    │    (mapping, layout)   │
                                    │    configurations      │
                                    └────────────────────────┘
*/

  protected:
    model::Engine::Specs arch_specs_;
    const Mapping& mapping_;
    layout::Layouts& layout_;
    
  public:
    std::uint64_t num_layout_candidates;
    std::vector<std::map<std::uint32_t, std::uint32_t>> storage_level_overall_dimval;
    std::vector<std::map<std::uint32_t, std::uint32_t>> cumulatively_intraline_dimval;
    std::vector<std::map<std::uint32_t, std::uint32_t>> cumulatively_product_dimval;
    std::vector<std::uint32_t> storage_level_total_capacity;
    std::vector<std::uint32_t> storage_level_line_capacity;
    std::vector<std::vector<bool>> storage_level_keep_factor; // true as kept, false as bypassed
    
    uint64_t splitting_candidates;
    uint64_t packing_candidates;
    uint64_t authblock_candidates;
    std::vector<std::vector<std::uint64_t>> splitting_candidates_per_lvl_per_ds;
    std::vector<std::vector<std::uint64_t>> packing_candidates_per_lvl_per_ds;

    // AuthBlock factor variation tracking
    std::vector<std::tuple<unsigned, unsigned, std::string, uint32_t>> variable_authblock_factors_; // level, dataspace, rank, max_value
    std::vector<std::vector<uint32_t>> authblock_factor_ranges_; // stores divisors for each factor

    // Intraline-to-interline conversion factor tracking (new level-based structure)
    struct SplittingOption {
      unsigned dataspace;
      std::string rank;
      uint32_t original_intraline_factor;
      uint32_t splitting_factor;
    };

    // Multi-rank splitting option for combinations of ranks
    struct MultiRankSplittingOption {
      unsigned dataspace;
      std::vector<std::string> ranks;  // Multiple ranks involved in the combination
      std::map<std::string, uint32_t> original_intraline_factors;  // Original factors for each rank
      std::map<std::string, uint32_t> splitting_factors;  // Splitting factors for each rank
      uint64_t total_reduction;  // Total reduction in intraline size from this combination
    };
    std::vector<std::vector<std::vector<MultiRankSplittingOption>>> multi_rank_splitting_options_per_level_per_ds_; // [level][ds_idx][option_index]

    // Track which levels and dataspaces require splitting (where intraline_size > line_capacity)
    std::vector<std::vector<bool>> level_ds_requires_splitting_; // [level][ds_idx] -> true if splitting is mandatory

    // Interline-to-intraline packing factor tracking (for unused line capacity)
    // Restructured to support single-rank-per-level packing
    struct PackingOption {
      unsigned dataspace;
      std::string rank;
      uint32_t original_interline_factor;
      uint32_t packing_factor;
    };

    // Multi-rank packing option for combinations of ranks within a single dataspace
    struct MultiRankPackingOption {
      unsigned dataspace;
      std::vector<std::string> ranks;  // Multiple ranks involved in the combination
      std::map<std::string, uint32_t> original_interline_factors;  // Original factors for each rank
      std::map<std::string, uint32_t> packing_factors;  // Packing factors for each rank
      uint64_t total_packing;  // Total packing factor applied
    };
    std::vector<std::vector<std::vector<MultiRankPackingOption>>> multi_rank_packing_options_per_level_per_ds_; // [level][ds_idx][option_index]

    // Packing choices organized by storage level
    // Each level can choose to pack exactly one rank (or no packing)
    unsigned num_storage_levels;
    unsigned num_data_spaces;

    Legal(model::Engine::Specs arch_specs,
          const Mapping& mapping,
          layout::Layouts& layout) :
          arch_specs_(arch_specs),
          mapping_(mapping),
          layout_(layout){};

    Legal(const Legal& other) = default;
    ~Legal();


    layout::Layouts GetLayout()
    {
      return layout_;
    }

    //------------------------------------------//
    //        Initialization and Setup          //
    //------------------------------------------//

    void Init(model::Engine::Specs arch_specs, const Mapping& mapping, layout::Layouts& layout);
    void ParseArchSpecs(model::Engine::Specs arch_specs, const Mapping& mapping);

    // Construct a specific layout using separate IDs for all three design spaces.
    std::vector<Status> ConstructLayout(uint64_t layout_splitting_id, uint64_t layout_packing_id, layout::Layouts* layouts, Mapping mapping, bool break_on_failure = true);

    // Layout constraint methods
    void CreateConcordantLayout(const Mapping& mapping);
    void CreateIntralineFactorSpace(model::Engine::Specs arch_specs, const Mapping& mapping);
    
    void SequentialFactorizeLayout(layout::Layouts& layout);

    // Helper methods for multi-rank splitting
    std::vector<std::vector<std::string>> GenerateRankCombinations(const std::vector<std::string>& ranks, size_t max_combo_size = 3);
    bool TestMultiRankSplittingWithCandidates(unsigned lvl, unsigned ds_idx, const std::vector<std::string>& rank_combination,
                                            const std::map<std::string, std::vector<uint32_t>>& candidate_factors_per_rank,
                                            const std::vector<std::vector<std::uint64_t>>& intraline_size_per_ds,
                                            uint64_t line_capacity, MultiRankSplittingOption& option);


    // Helper methods for multi-rank packing
    bool TestMultiRankPackingWithCandidates(unsigned lvl, unsigned ds_idx, const std::vector<std::string>& rank_combination,
                                          const std::map<std::string, std::vector<uint32_t>>& candidate_factors_per_rank,
                                          const std::vector<std::vector<std::uint64_t>>& intraline_size_per_ds,
                                          uint64_t line_capacity, std::vector<MultiRankPackingOption>& option);

  }; // class Legal
} // namespace layoutspace
