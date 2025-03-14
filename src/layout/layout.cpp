#include "layout/layout.hpp"

namespace layout
{

//------------------------------------------------------------------------------
// Helper: parseOrderMapping()
// Parses a mapping string (e.g., "C:0, M:1, R:2, S:3, N:4, P:5, Q:6")
// into an unordered_map from char to int.
//------------------------------------------------------------------------------
std::map<std::string, unsigned>
ParseOrderMapping(const std::string& mappingString)
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
        orderMapping[token.substr(0, 1)]
            = static_cast<unsigned>(std::stoi(token.substr(pos + 1)));
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
    config::CompoundConfigNode layoutArray, problem::Workload& workload,
    std::map<std::string, std::pair<uint32_t, uint32_t> >& targetToPortValue)
{
  std::map<std::string, std::vector<std::uint32_t> >
      rankToFactorizedDimensionID
      = workload.GetShape()->RankNameToFactorizedDimensionID;
  std::map<std::string, std::vector<std::string> > rankToDimensionName
      = workload.GetShape()->RankNameToDimensionName;
  std::map<std::string, std::vector<std::string> > rankToCoefficent
      = workload.GetShape()->RankNameToCoefficient;
  std::map<std::string, std::vector<std::string> > dataSpaceToRank
      = workload.GetShape()->DataSpaceNameToRankName;
  std::vector<std::string> data_space_vec;
  for (auto& key_pair : dataSpaceToRank)
    {
      data_space_vec.push_back(key_pair.first);
    }

  std::unordered_map<std::string, std::uint32_t> coefficentToValue;
  for (auto& key_pair : workload.GetShape()->CoefficientIDToName)
    {
      coefficentToValue[key_pair.second]
          = workload.GetCoefficient(key_pair.first);
    }

  std::map<std::string, std::vector<std::uint32_t> > rankToCoefficentValue;
  for (auto map_pair : rankToCoefficent)
    {
      std::vector<std::uint32_t> coefficentValue;
      for (auto coefName : rankToCoefficent.at(map_pair.first))
        {
          coefficentValue.push_back(coefficentToValue.at(coefName));
        }
      rankToCoefficentValue[map_pair.first] = coefficentValue;
    }

  std::map<std::string, unsigned> dimensionToDimID
      = workload.GetShape()->FactorizedDimensionNameToID;
  const std::vector<std::int32_t> dimension_bound
      = workload.GetFactorizedBounds().GetCoordinates();

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
  std::unordered_set<std::string> targets_unique;
  for (int i = 0; i < layoutCount; i++)
    {
      config::CompoundConfigNode entry = layoutArray[i];
      if (entry.exists("target"))
        {
          std::string cur_target;
          entry.lookupValue("target", cur_target);
          targets_unique.insert(cur_target);
        }
    }
  std::vector<std::string> targets;
  targets.insert(
      targets.begin(), targets_unique.begin(),
      targets_unique
          .end()); // = {"RegisterFile", "GlobalBuffer", "MainMemory"};

  // Convert the sample permutation string into a vector of single-character
  // strings.
  std::vector<std::string> globalRankList;
  for (char c : samplePermutation)
    {
      globalRankList.push_back(std::string(1, c));
    }

  // Derive the dimension order from dimensionToDimID by sorting the mapping by
  // value.
  std::vector<std::pair<std::string, unsigned> > dims;
  for (auto& p : dimensionToDimID)
    {
      dims.push_back(p);
    }
  std::sort(dims.begin(), dims.end(),
            [](auto& a, auto& b) { return a.second < b.second; });
  std::vector<char> computedDimOrder;
  for (auto& p : dims)
    {
      computedDimOrder.push_back(p.first[0]);
    }

  // ----------------------
  // Parse layout configuration entries.
  // ----------------------
  std::map<std::string,
           std::map<std::string,
                    std::pair<std::string,
                              std::map<std::string, std::uint32_t> > > >
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
  for (const auto& t : targets)
    {
      Layout layout;
      layout.target = t;
      layout.num_read_ports = targetToPortValue[t].first;
      layout.num_write_ports = targetToPortValue[t].second;
      layout.data_space = data_space_vec;
      layout.dataSpaceToRank = dataSpaceToRank;
      layout.rankToCoefficent = rankToCoefficent;
      layout.rankToCoefficentValue = rankToCoefficentValue;
      layout.rankToDimensionName = rankToDimensionName;
      layout.rankToFactorizedDimensionID = rankToFactorizedDimensionID;
      layout.dimensionToDimID = dimensionToDimID;
      layout.coefficentToValue = coefficentToValue;
      layout.dim_order = computedDimOrder;
      layout.rank_list = globalRankList;

      // For each data space, create loop nests.
      for (const auto& ds : layout.data_space)
        {
          // --- Interline nest ---
          LayoutNest nest;
          nest.data_space = ds;
          nest.type = "interline";
          if (config_layout[t].find("interline") != config_layout[t].end())
            {
              std::string perm = config_layout[t]["interline"].first;
              std::map<std::string, std::uint32_t> factors
                  = config_layout[t]["interline"].second;
              std::vector<std::string> order;
              for (char c : perm)
                {
                  std::string r(1, c);
                  const auto& ranks = layout.dataSpaceToRank[ds];
                  if (std::find(ranks.begin(), ranks.end(), r) != ranks.end())
                    {
                      order.push_back(r);
                    }
                }
              std::reverse(order.begin(), order.end());
              nest.ranks = order;
              nest.factors = factors;
            }
          else
            {
              nest.ranks = layout.dataSpaceToRank[ds];
            }
          layout.interline.push_back(nest);

          // --- Intraline nest ---
          LayoutNest intranest;
          intranest.data_space = ds;
          intranest.type = "intraline";
          if (config_layout[t].find("intraline") != config_layout[t].end())
            {
              std::string perm = config_layout[t]["intraline"].first;
              std::map<std::string, std::uint32_t> factors
                  = config_layout[t]["intraline"].second;
              std::vector<std::string> order;
              for (char c : perm)
                {
                  std::string r(1, c);
                  const auto& ranks = layout.dataSpaceToRank[ds];
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
              for (const auto& r : intranest.ranks)
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
// Helper function to print a Nest's loop order.
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// PrintOverallLayout
// Iterates over the nest’s skew descriptors and prints each term (including
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
  for (const auto& r : layouts[0].rank_list)
    std::cout << r << " ";
  std::cout << std::endl << std::endl;

  for (const auto& layout : layouts)
    {
      std::cout << "Target: " << layout.target << std::endl;
      std::cout << " num_read_ports: " << layout.num_read_ports
                << ", num_write_ports: " << layout.num_write_ports
                << std::endl;
      for (const auto& nest : layout.interline)
        {
          std::cout << "  Data space: " << nest.data_space << std::endl;
          std::cout << "  Type: " << nest.type << std::endl;
          for (const auto& r : nest.ranks)
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
                      std::cout << dims[i]
                                << (i != dims.size() - 1 ? "," : "");
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
      for (const auto& nest : layout.intraline)
        {
          std::cout << "  Data space: " << nest.data_space << std::endl;
          std::cout << "  Type: " << nest.type;
          for (const auto& r : nest.ranks)
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
                          std::cout << dims[i]
                                    << (i != dims.size() - 1 ? "," : "");
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
  for (const auto& r : layout.rank_list)
    std::cout << r << " ";
  std::cout << std::endl << std::endl;
  assert(layout.rank_list.size() == layout.rankToFactorizedDimensionID.size());

  {
    std::cout << "Target: " << layout.target << std::endl;
    std::cout << " num_read_ports: " << layout.num_read_ports
              << ", num_write_ports: " << layout.num_write_ports << std::endl;
    for (const auto& nest : layout.interline)
      {
        std::cout << "  Data space: " << nest.data_space << std::endl;
        std::cout << "  Type: " << nest.type << std::endl;
        for (const auto& r : nest.ranks)
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
    for (const auto& nest : layout.intraline)
      {
        std::cout << "  Data space: " << nest.data_space << std::endl;
        std::cout << "  Type: " << nest.type << std::endl;
        for (const auto& r : nest.ranks)
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

} // namespace layout