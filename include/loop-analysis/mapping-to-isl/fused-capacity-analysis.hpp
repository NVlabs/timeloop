#pragma once

#include <isl/polynomial.h>

#include "loop-analysis/isl-ir.hpp"

namespace capacity_analysis
{

using Id = size_t;
struct Aggregator;

struct Base
{
  virtual void Calculate(Aggregator& agg) = 0;

  Id id;
  isl_pw_qpolynomial* capacity;
};

struct Compute : public Base
{
  size_t start_idx;

  void Calculate(Aggregator&) override;
};

struct Root : public Base
{
  virtual void Calculate(Aggregator& agg) override;
  Id child_id;
};

struct BranchBase : public Base
{
  size_t start_idx;
  std::vector<Id> children_id;
};

struct Pipeline : public BranchBase
{
  void Calculate(Aggregator&) override;
};

struct Sequential : public BranchBase
{
  void Calculate(Aggregator&) override;
};

template<typename T>
inline constexpr bool HasOneChildV = std::is_same_v<T, Root>;

template<typename T>
inline constexpr bool HasManyChildrenV = std::is_same_v<T, Pipeline>
                                       | std::is_same_v<T, Sequential>;

using AggregatorTypes = std::variant<Compute,
                                     Root,
                                     Pipeline,
                                     Sequential>;

struct Aggregator
{
  Aggregator();

  AggregatorTypes& AggregatorAt(Id id);

  void Calculate();

  Id GetRootId() const;

  template<typename T>
  T& AddChild(Id parent)
  {
    Id child_id = aggregators.size();
    auto& child = std::get<T>(aggregators.emplace_back(T()));
    child.id = child_id;

    std::visit(
      [&child_id](auto&& agg)
      {
        using ParentT = std::decay_t<decltype(agg)>;
        if constexpr (HasOneChildV<ParentT>)
        {
          agg.child_id = child_id;
        }
        else if constexpr (HasManyChildrenV<ParentT>)
        {
          agg.children_id.push_back(child_id);
        }
        else
        {
          throw std::logic_error("cannot add child to this node");
        }
      },
      AggregatorAt(parent)
    );
    return child;
  }

  void SetCapacity(mapping::NodeID compute, Id latency);

  void Set(mapping::NodeID compute,
                  __isl_take isl_pw_qpolynomial* latency);

 private:
  Id root;
  std::vector<AggregatorTypes> aggregators;
  std::map<mapping::NodeID, Id> compute_to_aggregator;
};

Aggregator
CreateAggregatorFromMapping(mapping::FusedMapping& mapping);

};