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

#include <regex>
#include <string>
#include <filesystem>
#include <iostream>
#include <fstream>

namespace hyphens2underscores
{

  static std::string HYPHEN_TO_UNDERSCORE_TARGETS[] = {
    "linear_pruned",
    "problem_space_files",
    "log_stats",
    "tile_width",
    "log_interval",
    "CNN_Layer",
    "ordered_accesses",
    "router_energy",
    "tree_like",
    "arch_space_files",
    "log_oave_mappings",
    "penalize_consecutive_bypass_fails",
    "emit_whoop_nest",
    "condition_on",
    "energy_per_hop",
    "max_permutations_per_if_visit",
    "gate_on_zero_operand",
    "gcc_ar",
    "num_ports",
    "multiple_buffering",
    "cluster_area",
    "outer_to_inner",
    "spatial_skip",
    "log_all",
    "wire_energy",
    "metadata_word_bits",
    "optimization_metric",
    "word_bits",
    "action_optimization",
    "cnn_layer",
    "spatial_skip",
    "num_threads",
    "arch_space_sweep",
    "live_status",
    "step_size",
    "min_utilization",
    "network_type",
    "block_size",
    "sync_interval",
    "log_suboptimal",
    "representation_format",
    "flattened_rankIDs",
    "gemm_ABZ",
    "arch_spec",
    "last_level_accesses",
    "optimization_metrics",
    "inner_to_outer",
    "compression_rate",
    "filter_revisits",
    "vector_access_energy",
    "rank_application_order",
    "log_mappings_yaml",
    "log_mappings_verbose",
    "log_all_mappings",
    "read_write",
    "fixed_structured",
    "gcc_ranlib",
    "adder_energy",
    "search_size",
    "victory_condition",
    "random_pruned",
    "access_X",
    "skipping_spatial",
    "payload_word_bits",
    "num_banks",
    "addr_gen_energy",
    "cluster_size",
    "compute_optimization",
    "network_word_bits",
    "data_spaces",
    "timeloop_metric",
    "timeloop_metrics",
    "timeloop_simple_mapper",
    "timeloop_design_space",
    "timeloop_unit_tests",
  };

  std::string hyphens2underscores(std::string input);
  std::string hyphens2underscores_from_file(const char* inputFile);

} // namespace hyphens2underscores
