#pragma once

#include <vector>

namespace tree_analyzer
{

using Id = size_t;

struct Base
{
  virtual void Calculate(Analyzer& analyzer) = 0;
  Id id;
};

struct Root : public Base
{
  Id child_id;
};

struct Compute : public Base
{
};

struct Branch : public Base
{
  std::vector<Id> children_id;
};

struct Pipeline : public Branch
{
};

struct Sequential : public Branch
{
};

template<typename T>
inline constexpr bool HasOneChildV = std::is_base_of_v<Root, T>;

template<typename T>
inline constexpr bool HasManyChildrenV = std::is_base_of<Pipeline, T>
                                       | std::is_base_of<Sequential, T>;

struct Analyzer
{

 protected:
  Base base;
  std::vector<AnalyzerNodeT>
};

};