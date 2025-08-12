#include "layout/layout.hpp"

namespace layout
{

  //------------------------------------------------------------------------------
  // Helper: parseOrderMapping()
  // Parses a mapping string (e.g., "C:0, M:1, R:2, S:3, N:4, P:5, Q:6")
  // into an unordered_map from char to int.
  //------------------------------------------------------------------------------
  std::map<std::string, unsigned>
  ParseOrderMapping(const std::string &mappingString)
  {
    std::map<std::string, unsigned> orderMapping;
    std::istringstream iss(mappingString);
    std::string token;
    while (std::getline(iss, token, ','))
    {
      token.erase(std::remove_if(token.begin(), token.end(), ::isspace),
                  token.end());
      if (token.empty())
        continue;
      size_t pos = token.find(':');
      if (pos != std::string::npos)
        orderMapping[token.substr(0, 1)] = static_cast<unsigned>(std::stoi(token.substr(pos + 1)));
    }
    return orderMapping;
  }

  //------------------------------------------------------------------------------
  // ParseAndConstruct()
  // This function uses the compound-config library to read a configuration that
  // has a top-level "layout" array. Each entry must contain:
  //   - target (string)
  //   - type (string): either "interline" or "intraline"
  //   - factors (string): e.g., "R=3 S=3 P=7 Q=7 C=3 M=1 N=1"
  //   - permutation (string): The permutation string is processed by removing
  //   whitespace and then reversed,
  // so that left-to-right order is interpreted as outer-most to inner-most.
  //  e.g., "SR CQP MN" will become "SRCQPMN" with all spaces being ignored
  //
  // For interline entries, optional fields "num_read_ports" and
  // "num_write_ports" are parsed. For each unique target, a Layout is created
  // holding one interline nest and one intraline nest. If a nest is missing, a
  // default nest (with all factors set to 1) is created. Also, factor_order is
  // recorded (from the external mapping) and max_dim_perline is computed from
  // the intraline nest. (as defined by the external order mapping) the product
  // of the interline and intraline factors is >= the corresponding bound in
  // dimension_bound. Finally, the function returns a std::vector<Layout>
  // containing one Layout per unique target.
  Layouts
  ParseAndConstruct(
      config::CompoundConfigNode layoutArray, problem::Workload &workload,
      std::vector<std::pair<std::string, std::pair<uint32_t, uint32_t>>> &targetToPortValue)
  {
    // ToDo: Current memory logic only supports 3 levels, need to directly support more levels defined by architecture file.
    std::map<std::string, std::vector<std::uint32_t>>
        rankToFactorizedDimensionID = workload.GetShape()->RankNameToFactorizedDimensionID;
    std::map<std::string, std::vector<std::string>> rankToDimensionName = workload.GetShape()->RankNameToDimensionName;
    std::map<std::string, std::vector<std::string>> rankToCoefficient = workload.GetShape()->RankNameToCoefficient;
    std::map<std::string, std::string> rankToZeroPadding = workload.GetShape()->RankNameToZeroPadding;
    std::map<std::string, std::vector<std::string>> dataSpaceToRank = workload.GetShape()->DataSpaceNameToRankName;
    for (auto [ds, ranks] : dataSpaceToRank)
    {
      if (ranks.empty())
      {
        std::cerr << "Ranks need to be defined in the problem file for each dataspace. No ranks were provided for "
                  << ds
                  << "."
                  << std::endl;
        exit(1);
      }
    }
    std::vector<std::string> data_space_vec;
    for (unsigned j = 0; j < problem::GetShape()->NumDataSpaces; j++)
    {
      data_space_vec.push_back(problem::GetShape()->DataSpaceIDToName.at(j));
    }

    std::unordered_map<std::string, std::uint32_t> coefficientToValue;
    for (auto &key_pair : workload.GetShape()->CoefficientIDToName)
    {
      coefficientToValue[key_pair.second] = workload.GetCoefficient(key_pair.first);
    }

    std::map<std::string, std::vector<std::uint32_t>> rankToCoefficientValue;
    for (auto map_pair : rankToCoefficient)
    {
      std::vector<std::uint32_t> coefficientValue;
      for (auto coefName : rankToCoefficient.at(map_pair.first))
      {
        coefficientValue.push_back(coefficientToValue.at(coefName));
      }
      rankToCoefficientValue[map_pair.first] = coefficientValue;
    }

    std::map<std::string, std::uint32_t> rankToZeroPaddingValue;
    for (auto map_pair : rankToZeroPadding)
    {
      rankToZeroPaddingValue[map_pair.first] = workload.GetPadding(map_pair.second);
    }

    std::map<std::string, unsigned> dimensionToDimID = workload.GetShape()->FactorizedDimensionNameToID;
    const std::vector<std::int32_t> dimension_bound = workload.GetFactorizedBounds().GetCoordinates();

    std::unordered_map<std::string, Layout> layoutMap;

    int layoutCount = layoutArray.getLength();
    std::string samplePermutation;
    bool foundPermutation = false;
    for (int i = 0; i < layoutCount; i++)
    {
      config::CompoundConfigNode entry = layoutArray[i];
      if (entry.exists("permutation"))
      {
        entry.lookupValue("permutation", samplePermutation);
        foundPermutation = true;
        break;
      }
    }
    if (!foundPermutation)
    {
      std::cerr << "No permutation key found in any layout entry."
                << std::endl;
      exit(1);
    }

    // load targets
    std::vector<std::string> targets;
    for (unsigned i =  0; i < targetToPortValue.size(); i++)
    {
      targets.push_back(targetToPortValue[i].first);
    }

    // Convert the sample permutation string into a vector of single-character
    // strings.
    std::vector<std::string> globalRankList;
    for (char c : samplePermutation)
    {
      globalRankList.push_back(std::string(1, c));
    }

    // Derive the dimension order from dimensionToDimID by sorting the mapping by
    // value.
    std::vector<std::pair<std::string, unsigned>> dims;
    for (auto &p : dimensionToDimID)
    {
      dims.push_back(p);
    }
    std::sort(dims.begin(), dims.end(),
              [](auto &a, auto &b)
              { return a.second < b.second; });
    std::vector<char> computedDimOrder;
    for (auto &p : dims)
    {
      computedDimOrder.push_back(p.first[0]);
    }

    // ----------------------
    // Parse layout configuration entries.
    // ----------------------
    std::map<std::string,
             std::map<std::string,
                      std::pair<std::string,
                                std::map<std::string, std::uint32_t>>>>
        config_layout;

    for (int i = 0; i < layoutCount; i++)
    {
      config::CompoundConfigNode entry = layoutArray[i];
      std::string target, type, permutation, factorsStr;
      entry.lookupValue("target", target);
      entry.lookupValue("type", type);
      entry.lookupValue("permutation", permutation);
      entry.lookupValue("factors", factorsStr);

      // Parse the factors string (e.g., "J=1 K=1 U=1 I=1 V=1 E=1 Z=1 H=1 W=1")
      std::map<std::string, std::uint32_t> factors;
      std::istringstream iss(factorsStr);
      std::string token;
      while (iss >> token)
      {
        auto pos = token.find('=');
        if (pos != std::string::npos)
        {
          std::string rank = token.substr(0, pos);
          std::uint32_t value = std::stoi(token.substr(pos + 1));
          factors[rank] = value;
        }
      }
      config_layout[target][type] = std::make_pair(permutation, factors);
    }

    // ----------------------
    // Create Layout objects for each target.
    // ----------------------
    Layouts layouts;
    for (const auto &t : targets)
    {
      Layout layout;
      layout.target = t;
      // Find the target in the vector
      auto it = std::find_if(targetToPortValue.begin(), targetToPortValue.end(),
                            [&t](const auto& pair) { return pair.first == t; });
      if (it != targetToPortValue.end()) {
        layout.num_read_ports = it->second.first;
        layout.num_write_ports = it->second.second;
      } else {
        // Default values if target not found
        layout.num_read_ports = 1;
        layout.num_write_ports = 1;
      }
      layout.data_space = data_space_vec;
      layout.dataSpaceToRank = dataSpaceToRank;
      layout.rankToCoefficient = rankToCoefficient;
      layout.rankToCoefficientValue = rankToCoefficientValue;
      layout.rankToDimensionName = rankToDimensionName;
      layout.rankToFactorizedDimensionID = rankToFactorizedDimensionID;
      layout.dimensionToDimID = dimensionToDimID;
      layout.coefficientToValue = coefficientToValue;
      layout.rankToZeroPadding = rankToZeroPaddingValue;
      layout.dim_order = computedDimOrder;
      layout.rank_list = globalRankList;

      // ToDo: make these configurable, and also separately configurable per memory level
      layout.assume_zero_padding = true;
      layout.assume_row_buffer = true;
      layout.assume_reuse = true;

      // For each data space, create loop nests.
      for (const auto &ds : layout.data_space)
      {
        // --- Interline nest ---
        LayoutNest internest;
        internest.data_space = ds;
        internest.type = "interline";
        if (config_layout[t].find("interline") != config_layout[t].end())
        {
          std::string perm = config_layout[t]["interline"].first;
          std::map<std::string, std::uint32_t> factors = config_layout[t]["interline"].second;
          std::vector<std::string> order;
          for (char c : perm)
          {
            std::string r(1, c);
            const auto &ranks = layout.dataSpaceToRank[ds];
            if (std::find(ranks.begin(), ranks.end(), r) != ranks.end())
            {
              order.push_back(r);
            }
          }
          std::reverse(order.begin(), order.end());
          internest.ranks = order;
          internest.factors = factors;
        }
        else
        {
          internest.ranks = layout.dataSpaceToRank[ds];
        }
        layout.interline.push_back(internest);

        // --- Intraline nest ---
        LayoutNest intranest;
        intranest.data_space = ds;
        intranest.type = "intraline";
        if (config_layout[t].find("intraline") != config_layout[t].end())
        {
          std::string perm = config_layout[t]["intraline"].first;
          std::map<std::string, std::uint32_t> factors = config_layout[t]["intraline"].second;
          std::vector<std::string> order;
          for (char c : perm)
          {
            std::string r(1, c);
            const auto &ranks = layout.dataSpaceToRank[ds];
            if (std::find(ranks.begin(), ranks.end(), r) != ranks.end())
            {
              order.push_back(r);
            }
          }
          std::reverse(order.begin(), order.end());
          intranest.ranks = order;
          intranest.factors = factors;
        }
        else
        {
          intranest.ranks = layout.dataSpaceToRank[ds];
          for (const auto &r : intranest.ranks)
          {
            intranest.factors[r] = 1;
          }
        }
        layout.intraline.push_back(intranest);
      }

      layouts.push_back(layout);
    }

    return layouts;
  }


  //------------------------------------------------------------------------------
  // InitializeDummyLayout()
  // This function creates a dummy layout for each target.
  //
  // For each unique target, a Layout is created holding one interline nest and 
  // one intraline nest with all factors set to 1. Also, factor_order is
  // recorded and max_dim_perline is computed from the intraline nest.
  Layouts
  InitializeDummyLayout(
      problem::Workload &workload,
      std::vector<std::pair<std::string, std::pair<uint32_t, uint32_t>>> &targetToPortValue)
  {
    // ToDo: Current memory logic only supports 3 levels, need to directly support more levels defined by architecture file.
    std::map<std::string, std::vector<std::uint32_t>>
        rankToFactorizedDimensionID = workload.GetShape()->RankNameToFactorizedDimensionID;
    std::map<std::string, std::vector<std::string>> rankToDimensionName = workload.GetShape()->RankNameToDimensionName;
    std::map<std::string, std::vector<std::string>> rankToCoefficient = workload.GetShape()->RankNameToCoefficient;
    std::map<std::string, std::string> rankToZeroPadding = workload.GetShape()->RankNameToZeroPadding;
    std::map<std::string, std::vector<std::string>> dataSpaceToRank = workload.GetShape()->DataSpaceNameToRankName;
    for (auto [ds, ranks] : dataSpaceToRank)
    {
      if (ranks.empty())
      {
        std::cerr << "Ranks need to be defined in the problem file for each dataspace. No ranks were provided for "
                  << ds
                  << "."
                  << std::endl;
        exit(1);
      }
    }
    std::vector<std::string> data_space_vec;
    for (unsigned j = 0; j < problem::GetShape()->NumDataSpaces; j++)
    {
      data_space_vec.push_back(problem::GetShape()->DataSpaceIDToName.at(j));
    }

    std::unordered_map<std::string, std::uint32_t> coefficientToValue;
    for (auto &key_pair : workload.GetShape()->CoefficientIDToName)
    {
      coefficientToValue[key_pair.second] = workload.GetCoefficient(key_pair.first);
    }

    std::map<std::string, std::vector<std::uint32_t>> rankToCoefficientValue;
    for (auto map_pair : rankToCoefficient)
    {
      std::vector<std::uint32_t> coefficientValue;
      for (auto coefName : rankToCoefficient.at(map_pair.first))
      {
        coefficientValue.push_back(coefficientToValue.at(coefName));
      }
      rankToCoefficientValue[map_pair.first] = coefficientValue;
    }

    std::map<std::string, std::uint32_t> rankToZeroPaddingValue;
    for (auto map_pair : rankToZeroPadding)
    {
      rankToZeroPaddingValue[map_pair.first] = workload.GetPadding(map_pair.second);
    }

    std::map<std::string, unsigned> dimensionToDimID = workload.GetShape()->FactorizedDimensionNameToID;
    const std::vector<std::int32_t> dimension_bound = workload.GetFactorizedBounds().GetCoordinates();

    std::unordered_map<std::string, Layout> layoutMap;

    // Create default permutation based on dimension order
    std::string samplePermutation;
    
    // Derive the dimension order from dimensionToDimID by sorting the mapping by
    // value.
    std::vector<std::pair<std::string, unsigned>> dims;
    for (auto &p : dimensionToDimID)
    {
      dims.push_back(p);
    }
    std::sort(dims.begin(), dims.end(),
              [](auto &a, auto &b)
              { return a.second < b.second; });
    
    // Create default permutation from computed dimension order
    for (auto &p : dims)
    {
      samplePermutation += p.first[0];
    }

    // Use targets from targetToPortValue
    // load targets
    std::vector<std::string> targets;
    for (unsigned i =  0; i < targetToPortValue.size(); i++)
    {
      targets.push_back(targetToPortValue[i].first);
    }

    // Convert the sample permutation string into a vector of single-character
    // strings.
    std::vector<std::string> globalRankList;
    for (char c : samplePermutation)
    {
      globalRankList.push_back(std::string(1, c));
    }

    std::vector<char> computedDimOrder;
    for (auto &p : dims)
    {
      computedDimOrder.push_back(p.first[0]);
    }

    // ----------------------
    // Create Layout objects for each target with dummy values.
    // ----------------------
    Layouts layouts;
    for (const auto &t : targets)
    {
      Layout layout;
      layout.target = t;
      
      // Find the target in the vector
      auto it = std::find_if(targetToPortValue.begin(), targetToPortValue.end(),
                            [&t](const auto& pair) { return pair.first == t; });
      if (it != targetToPortValue.end()) {
        layout.num_read_ports = it->second.first;
        layout.num_write_ports = it->second.second;
      } else {
        // Default values if target not found
        layout.num_read_ports = 1;
        layout.num_write_ports = 1;
      }
      
      layout.data_space = data_space_vec;
      layout.dataSpaceToRank = dataSpaceToRank;
      layout.rankToCoefficient = rankToCoefficient;
      layout.rankToCoefficientValue = rankToCoefficientValue;
      layout.rankToDimensionName = rankToDimensionName;
      layout.rankToFactorizedDimensionID = rankToFactorizedDimensionID;
      layout.dimensionToDimID = dimensionToDimID;
      layout.coefficientToValue = coefficientToValue;
      layout.rankToZeroPadding = rankToZeroPaddingValue;
      layout.dim_order = computedDimOrder;
      layout.rank_list = globalRankList;

      // ToDo: make these configurable, and also separately configurable per memory level
      layout.assume_zero_padding = true;
      layout.assume_row_buffer = true;
      layout.assume_reuse = true;

      // For each data space, create loop nests with dummy values (all factors = 1).
      for (const auto &ds : layout.data_space)
      {
        // --- Interline nest with dummy values ---
        LayoutNest internest;
        internest.data_space = ds;
        internest.type = "interline";
        internest.ranks = layout.dataSpaceToRank[ds];
        // Set all factors to 1 for dummy layout
        for (const auto &r : internest.ranks)
        {
          internest.factors[r] = 1;
        }
        layout.interline.push_back(internest);

        // --- Intraline nest with dummy values ---
        LayoutNest intranest;
        intranest.data_space = ds;
        intranest.type = "intraline";
        intranest.ranks = layout.dataSpaceToRank[ds];
        // Set all factors to 1 for dummy layout
        for (const auto &r : intranest.ranks)
        {
          intranest.factors[r] = 1;
        }
        layout.intraline.push_back(intranest);
      }

      layouts.push_back(layout);
    }

    return layouts;
  }


  //------------------------------------------------------------------------------
  // Helper function to print a Nest's loop order.
  //------------------------------------------------------------------------------

  //------------------------------------------------------------------------------
  // PrintOverallLayout
  // Iterates over the nest's skew descriptors and prints each term (including
  // rank name and, now, all related ranks based on shared dimensions).
  void
  PrintOverallLayout(Layouts layouts)
  {
    std::cout << "Dimension Order: ";
    for (size_t i = 0; i < layouts[0].dim_order.size(); i++)
    {
      char d = layouts[0].dim_order[i];
      std::string dStr(1, d);
      std::cout << d << "-" << layouts[0].dimensionToDimID[dStr];
      if (i != layouts[0].dim_order.size() - 1)
        std::cout << ", ";
    }
    std::cout << std::endl;

    std::cout << "Rank List: ";
    for (const auto &r : layouts[0].rank_list)
      std::cout << r << " ";
    std::cout << std::endl
              << std::endl;

    for (const auto &layout : layouts)
    {
      std::cout << "Target: " << layout.target << std::endl;
      std::cout << " num_read_ports: " << layout.num_read_ports
                << ", num_write_ports: " << layout.num_write_ports
                << std::endl;

      for (const auto &nest : layout.interline)
      {
        std::cout << "  Data space: " << nest.data_space << std::endl;
        std::cout << "  Type: " << nest.type << std::endl;
        for (const auto &r : nest.ranks)
        {
          int factor = (nest.factors.find(r) != nest.factors.end()
                            ? nest.factors.at(r)
                            : 1);
          auto dims = layout.rankToFactorizedDimensionID.at(r);
          std::cout << "    Rank: " << r << " dimension=";
          if (dims.size() == 1)
          {
            std::cout << dims[0] << "-"
                      << layout.rankToDimensionName.at(r)[0];
          }
          else
          {
            std::cout << "(";
            for (size_t i = 0; i < dims.size(); i++)
            {
              std::cout << dims[i] << (i != dims.size() - 1 ? "," : "");
            }
            std::cout << ")-(";
            auto names = layout.rankToDimensionName.at(r);
            for (size_t i = 0; i < names.size(); i++)
            {
              std::cout << names[i]
                        << (i != names.size() - 1 ? "," : "");
            }
            std::cout << ")";
          }
          std::cout << ", factor=" << factor << std::endl;
        }
      }
      for (const auto &nest : layout.intraline)
      {
        std::cout << "  Data space: " << nest.data_space << std::endl;
        std::cout << "  Type: " << nest.type << std::endl;
        for (const auto &r : nest.ranks)
        {
          int factor = (nest.factors.find(r) != nest.factors.end()
                            ? nest.factors.at(r)
                            : 1);
          auto dims = layout.rankToFactorizedDimensionID.at(r);
          std::cout << "    Rank: " << r << " dimension=";
          if (dims.size() == 1)
          {
            std::cout << dims[0] << "-"
                      << layout.rankToDimensionName.at(r)[0];
          }
          else
          {
            std::cout << "(";
            for (size_t i = 0; i < dims.size(); i++)
            {
              std::cout << dims[i] << (i != dims.size() - 1 ? "," : "");
            }
            std::cout << ")-(";
            auto names = layout.rankToDimensionName.at(r);
            for (size_t i = 0; i < names.size(); i++)
            {
              std::cout << names[i]
                        << (i != names.size() - 1 ? "," : "");
            }
            std::cout << ")";
          }
          std::cout << ", factor=" << factor << std::endl;
        }
      }
    }
  }



  //------------------------------------------------------------------------------
  // PrintOverallLayoutConcise
  // Prints layout information in a concise format, grouping by data space and 
  // showing factors in rank=factor format on single lines.
  void
  PrintOverallLayoutConcise(Layouts layouts)
  {
    std::cout << "Dimension Order: ";
    for (size_t i = 0; i < layouts[0].dim_order.size(); i++)
    {
      char d = layouts[0].dim_order[i];
      std::string dStr(1, d);
      std::cout << d << "-" << layouts[0].dimensionToDimID[dStr];
      if (i != layouts[0].dim_order.size() - 1)
        std::cout << ", ";
    }
    std::cout << std::endl;

    std::cout << "Rank List: ";
    for (const auto &r : layouts[0].rank_list)
      std::cout << r << " ";
    std::cout << std::endl << std::endl;

    for (const auto &layout : layouts)
    {
      std::cout << "Target: " << layout.target << std::endl;
      std::cout << " num_read_ports: " << layout.num_read_ports
                << ", num_write_ports: " << layout.num_write_ports << std::endl;

      // Get all unique data spaces from all nest types
      std::set<std::string> data_spaces;
      for (const auto &nest : layout.interline)
        data_spaces.insert(nest.data_space);
      for (const auto &nest : layout.intraline)
        data_spaces.insert(nest.data_space);

      // For each data space, print all nest types in a compact format
      for (const auto &ds : data_spaces)
      {
        std::cout << "  Data space: " << ds << std::endl;
        
        // Print interline factors
        for (const auto &nest : layout.interline)
        {
          if (nest.data_space == ds)
          {
            std::cout << "    interline: ";
            bool first = true;
            for (const auto &r : nest.ranks)
            {
              if (!first) std::cout << ", ";
              int factor = (nest.factors.find(r) != nest.factors.end() ? nest.factors.at(r) : 1);
              std::cout << r << "=" << factor;
              first = false;
            }
            std::cout << std::endl;
            break;
          }
        }
        
        // Print intraline factors
        for (const auto &nest : layout.intraline)
        {
          if (nest.data_space == ds)
          {
            std::cout << "    intraline: ";
            bool first = true;
            for (const auto &r : nest.ranks)
            {
              if (!first) std::cout << ", ";
              int factor = (nest.factors.find(r) != nest.factors.end() ? nest.factors.at(r) : 1);
              std::cout << r << "=" << factor;
              first = false;
            }
            std::cout << std::endl;
            break;
          }
        }
      }
    }
  }


  void
  PrintOverallLayoutConcise(Layouts layouts, std::ostream &os)
  {
    os << "Dimension Order: ";
    for (size_t i = 0; i < layouts[0].dim_order.size(); i++)
    {
      char d = layouts[0].dim_order[i];
      std::string dStr(1, d);
      os << d << "-" << layouts[0].dimensionToDimID[dStr];
      if (i != layouts[0].dim_order.size() - 1)
        os << ", ";
    }
    os << std::endl;

    os << "Rank List: ";
    for (const auto &r : layouts[0].rank_list)
      os << r << " ";
    os << std::endl << std::endl;

    for (const auto &layout : layouts)
    {
      os << "Target: " << layout.target << std::endl;
      os << " num_read_ports: " << layout.num_read_ports
                << ", num_write_ports: " << layout.num_write_ports << std::endl;

      // Get all unique data spaces from all nest types
      std::set<std::string> data_spaces;
      for (const auto &nest : layout.interline)
        data_spaces.insert(nest.data_space);
      for (const auto &nest : layout.intraline)
        data_spaces.insert(nest.data_space);

      // For each data space, print all nest types in a compact format
      for (const auto &ds : data_spaces)
      {
        os << "  Data space: " << ds << std::endl;
        
        // Print interline factors
        for (const auto &nest : layout.interline)
        {
          if (nest.data_space == ds)
          {
            os << "    interline: ";
            bool first = true;
            for (const auto &r : nest.ranks)
            {
              if (!first) os << ", ";
              int factor = (nest.factors.find(r) != nest.factors.end() ? nest.factors.at(r) : 1);
              os << r << "=" << factor;
              first = false;
            }
            os << std::endl;
            break;
          }
        }
        
        // Print intraline factors
        for (const auto &nest : layout.intraline)
        {
          if (nest.data_space == ds)
          {
            os << "    intraline: ";
            bool first = true;
            for (const auto &r : nest.ranks)
            {
              if (!first) os << ", ";
              int factor = (nest.factors.find(r) != nest.factors.end() ? nest.factors.at(r) : 1);
              os << r << "=" << factor;
              first = false;
            }
            os << std::endl;
            break;
          }
        }
      }
    }
  }


  void
  PrintOneLvlLayout(Layout layout)
  {
    std::cout << "Dimension Order: ";
    for (size_t i = 0; i < layout.dim_order.size(); i++)
    {
      char d = layout.dim_order[i];
      std::string dStr(1, d);
      std::cout << d << "-" << layout.dimensionToDimID[dStr];
      if (i != layout.dim_order.size() - 1)
        std::cout << ", ";
    }
    std::cout << std::endl;

    std::cout << "Rank List: ";
    for (const auto &r : layout.rank_list)
      std::cout << r << " ";
    std::cout << std::endl
              << std::endl;
    assert(layout.rank_list.size() == layout.rankToFactorizedDimensionID.size());

    {
      std::cout << "Target: " << layout.target << std::endl;
      std::cout << " num_read_ports: " << layout.num_read_ports
                << ", num_write_ports: " << layout.num_write_ports << std::endl;
      for (const auto &nest : layout.interline)
      {
        std::cout << "  Data space: " << nest.data_space << std::endl;
        std::cout << "  Type: " << nest.type << std::endl;
        for (const auto &r : nest.ranks)
        {
          int factor = (nest.factors.find(r) != nest.factors.end()
                            ? nest.factors.at(r)
                            : 1);
          auto dims = layout.rankToFactorizedDimensionID.at(r);
          std::cout << "    Rank: " << r << " dimension=";
          if (dims.size() == 1)
          {
            std::cout << dims[0] << "-"
                      << layout.rankToDimensionName.at(r)[0];
          }
          else
          {
            std::cout << "(";
            for (size_t i = 0; i < dims.size(); i++)
            {
              std::cout << dims[i] << (i != dims.size() - 1 ? "," : "");
            }
            std::cout << ")-(";
            auto names = layout.rankToDimensionName.at(r);
            for (size_t i = 0; i < names.size(); i++)
            {
              std::cout << names[i]
                        << (i != names.size() - 1 ? "," : "");
            }
            std::cout << ")";
          }
          std::cout << ", factor=" << factor << std::endl;
        }
      }
      for (const auto &nest : layout.intraline)
      {
        std::cout << "  Data space: " << nest.data_space << std::endl;
        std::cout << "  Type: " << nest.type << std::endl;
        for (const auto &r : nest.ranks)
        {
          int factor = (nest.factors.find(r) != nest.factors.end()
                            ? nest.factors.at(r)
                            : 1);
          auto dims = layout.rankToFactorizedDimensionID.at(r);
          std::cout << "    Rank: " << r << " dimension=";
          if (dims.size() == 1)
          {
            std::cout << dims[0] << "-"
                      << layout.rankToDimensionName.at(r)[0];
          }
          else
          {
            std::cout << "(";
            for (size_t i = 0; i < dims.size(); i++)
            {
              std::cout << dims[i] << (i != dims.size() - 1 ? "," : "");
            }
            std::cout << ")-(";
            auto names = layout.rankToDimensionName.at(r);
            for (size_t i = 0; i < names.size(); i++)
            {
              std::cout << names[i]
                        << (i != names.size() - 1 ? "," : "");
            }
            std::cout << ")";
          }
          std::cout << ", factor=" << factor << std::endl;
        }
      }
      std::cout << std::endl;
    }
  }


  void
  PrintOneLvlLayoutDataSpace(Layout layout, std::string data_space_in)
  {
    std::cout << "Dimension Order: ";
    for (size_t i = 0; i < layout.dim_order.size(); i++)
    {
      char d = layout.dim_order[i];
      std::string dStr(1, d);
      std::cout << d << "-" << layout.dimensionToDimID[dStr];
      if (i != layout.dim_order.size() - 1)
        std::cout << ", ";
    }
    std::cout << std::endl;

    std::cout << "Rank List: ";
    for (const auto &r : layout.rank_list)
      std::cout << r << " ";
    std::cout << std::endl
              << std::endl;
    assert(layout.rank_list.size() == layout.rankToFactorizedDimensionID.size());

    {
      std::cout << "Target: " << layout.target << std::endl;
      std::cout << " num_read_ports: " << layout.num_read_ports
                << ", num_write_ports: " << layout.num_write_ports << std::endl;
      for (const auto &nest : layout.interline)
      {
        if (data_space_in == nest.data_space)
        {
          std::cout << "  Data space: " << nest.data_space << std::endl;
          std::cout << "  Type: " << nest.type << std::endl;
          for (const auto &r : nest.ranks)
          {
            int factor = (nest.factors.find(r) != nest.factors.end()
                              ? nest.factors.at(r)
                              : 1);
            auto dims = layout.rankToFactorizedDimensionID.at(r);
            std::cout << "    Rank: " << r << " dimension=";
            if (dims.size() == 1)
            {
              std::cout << dims[0] << "-"
                        << layout.rankToDimensionName.at(r)[0];
            }
            else
            {
              std::cout << "(";
              for (size_t i = 0; i < dims.size(); i++)
              {
                std::cout << dims[i] << (i != dims.size() - 1 ? "," : "");
              }
              std::cout << ")-(";
              auto names = layout.rankToDimensionName.at(r);
              for (size_t i = 0; i < names.size(); i++)
              {
                std::cout << names[i]
                          << (i != names.size() - 1 ? "," : "");
              }
              std::cout << ")";
            }
            std::cout << ", factor=" << factor << std::endl;
          }
        }
      }
      for (const auto &nest : layout.intraline)
      {
        if (data_space_in == nest.data_space)
        {
          std::cout << "  Type: " << nest.type << std::endl;
          for (const auto &r : nest.ranks)
          {
            int factor = (nest.factors.find(r) != nest.factors.end()
                              ? nest.factors.at(r)
                              : 1);
            auto dims = layout.rankToFactorizedDimensionID.at(r);
            std::cout << "    Rank: " << r << " dimension=";
            if (dims.size() == 1)
            {
              std::cout << dims[0] << "-"
                        << layout.rankToDimensionName.at(r)[0];
            }
            else
            {
              std::cout << "(";
              for (size_t i = 0; i < dims.size(); i++)
              {
                std::cout << dims[i] << (i != dims.size() - 1 ? "," : "");
              }
              std::cout << ")-(";
              auto names = layout.rankToDimensionName.at(r);
              for (size_t i = 0; i < names.size(); i++)
              {
                std::cout << names[i]
                          << (i != names.size() - 1 ? "," : "");
              }
              std::cout << ")";
            }
            std::cout << ", factor=" << factor << std::endl;
          }
        }
      }
      std::cout << std::endl;
    }
  }


  //------------------------------------------------------------------------------
  // DumpLayoutToYAML
  // Dumps the layout to a YAML file following the pattern in test_layout.yaml
  //------------------------------------------------------------------------------
  void DumpLayoutToYAML(const Layouts& layouts, const std::string& filename)
  {
    std::ofstream yaml_file(filename);
    if (!yaml_file.is_open())
    {
      std::cerr << "Error: Could not open " << filename << " for writing." << std::endl;
      return;
    }

    yaml_file << "layout:" << std::endl;
    
    for (auto it = layouts.rbegin(); it != layouts.rend(); ++it)
    {
      const auto& layout = *it;
      // Process each nest type (interline, intraline)
      std::vector<std::string> nest_types = {"interline", "intraline"};
      
      for (const auto& nest_type : nest_types)
      {
        // Collect all factors and ranks across all dataspaces for this target and type
        std::map<std::string, uint32_t> combined_factors;
        std::vector<std::string> combined_ranks;
        bool has_data = false;
        
              // Get the appropriate nest vector based on type
        std::vector<layout::LayoutNest> nests;
        if (nest_type == "interline")
          nests = layout.interline;
        else if (nest_type == "intraline")
          nests = layout.intraline;

        // Combine factors from all dataspaces
        for (const auto& nest : nests)
        {

          has_data = true;
          
          // Collect ranks in order (avoid duplicates)
          for (const auto& rank : nest.ranks)
          {
            if (std::find(combined_ranks.begin(), combined_ranks.end(), rank) == combined_ranks.end())
            {
              combined_ranks.push_back(rank);
            }
          }
          
                  // Collect factors (use the factor from each dataspace, taking max if rank appears multiple times)
          for (const auto& rank : nest.ranks)
          {
            auto factor_it = nest.factors.find(rank);
            uint32_t factor = (factor_it != nest.factors.end()) ? factor_it->second : 1;
            
            if (combined_factors.find(rank) == combined_factors.end())
            {
              combined_factors[rank] = factor;
            }
            else
            {
              combined_factors[rank] = std::max(combined_factors[rank], factor); // Take maximum factor across dataspaces
            }
          }
        }
        
        // Write the combined block if there's data
        if (has_data)
        {
          yaml_file << "  - target: " << layout.target << std::endl;
          yaml_file << "    type: " << nest_type << std::endl;
          
          // Generate combined factors string using the combined_ranks order
          std::string factors_str = "";
          for (const auto& rank : combined_ranks)
          {
            auto factor_it = combined_factors.find(rank);
            uint32_t factor = (factor_it != combined_factors.end()) ? factor_it->second : 1;
            if (!factors_str.empty()) factors_str += " ";
            factors_str += rank + "=" + std::to_string(factor);
          }
          yaml_file << "    factors: " << factors_str << std::endl;
          
          // Generate combined permutation string
          std::string permutation_str = "";
          for (const auto& rank : combined_ranks)
          {
            permutation_str += rank;
          }
          yaml_file << "    permutation: " << permutation_str << std::endl;
        }
      }
    }
    
    yaml_file.close();
    std::cout << "Layout dumped to " << filename << std::endl;
  }
} // namespace layout