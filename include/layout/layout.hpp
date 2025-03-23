#pragma once

#include "compound-config/compound-config.hpp"
#include <algorithm>
#include <cctype>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <workload/workload.hpp>
#include <yaml-cpp/yaml.h>

//------------------------------------------------------------------------------
// Layout structure (target-level information)
//------------------------------------------------------------------------------

namespace layout
{

struct LayoutNest
{
  std::string data_space;         // e.g., "Inputs"
  std::string type;               // "interline" or "intraline"
  std::vector<std::string> ranks; // Order of rank names for this nest.
  std::map<std::string, std::uint32_t>
      factors; // Factor for each rank (if specified)
};

struct Layout
{
  std::string target; // e.g., "MainMemory"
  std::vector<LayoutNest>
      interline; // One nest per data space for interline type
  std::vector<LayoutNest>
      intraline; // One nest per data space for intraline type
  std::vector<std::string>
      data_space; // Data space names (e.g., Inputs, Outputs, Weights)
  std::vector<std::string>
      rank_list;          // Overall rank list (derived from a permutation key)
  int num_read_ports = 1; // Configured read ports
  int num_write_ports = 1;     // Configured write ports
  std::vector<char> dim_order; // Dimension order (derived from configuration)

  // Configured mappings
  std::map<std::string, std::vector<std::string> > dataSpaceToRank;
  std::map<std::string, std::vector<std::uint32_t> >
      rankToFactorizedDimensionID;
  std::map<std::string, std::vector<std::string> > rankToDimensionName;
  std::map<std::string, std::uint32_t> dimensionToDimID;
  std::map<std::string, std::vector<std::string> > rankToCoefficient;
  std::map<std::string, std::vector<std::uint32_t> > rankToCoefficientValue;
  std::unordered_map<std::string, std::uint32_t> coefficientToValue;

  bool initialize
      = false; // True if external YAML provided layout for this target
};

typedef std::vector<Layout> Layouts;

//------------------------------------------------------------------------------
// Helper: parseOrderMapping()
// Parses a mapping string (e.g., "C:0, M:1, R:2, S:3, N:4, P:5, Q:6")
// into an unordered_map from char to int.
//------------------------------------------------------------------------------
std::map<std::string, unsigned>
parseOrderMapping(const std::string& mappingString);

//------------------------------------------------------------------------------
// ParseAndConstruct()
// This function uses the compound-config library to read a configuration that
// has a top-level "layout" array. Each entry must contain:
//   - target (string)
//   - type (string): either "interline" or "intraline"
//   - factors (string): e.g., "R=3 S=3 P=7 Q=7 C=3 M=1 N=1"
//   - permutation (string): e.g., "SR CQP MN"
// For interline entries, optional fields "num_read_ports" and
// "num_write_ports" are parsed (defaulting to 1).
//
// For each unique target, a Layout is created holding one interline nest and
// one intraline nest. If a nest is missing, a default nest with all factors
// set to 1 is created. Also, the extra vector factor_order is set from the
// externally provided order mapping. Finally, max_dim_perline is computed from
// the intraline nest.

std::vector<Layout> ParseAndConstruct(
    config::CompoundConfigNode layoutArray, problem::Workload& workload,
    std::map<std::string, std::pair<uint32_t, uint32_t> >& targetToPortValue);

//------------------------------------------------------------------------------
// Helper function to print a Nest's loop order.
//------------------------------------------------------------------------------
void PrintOverallLayout(Layouts layout);

//------------------------------------------------------------------------------
void PrintOneLvlLayout(Layout layout);

//------------------------------------------------------------------------------
void PrintOneLvlLayoutDataSpace(Layout layout, std::string data_space_in);

} // namespace layout