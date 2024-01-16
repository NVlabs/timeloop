#include <vector>

#include <barvinok/isl.h>

#include "mapping/fused-mapping.hpp"

namespace analysis 
{

using LatencyId = size_t;
struct LatencyAggregator;

struct LatencyBase
{
  virtual void CalculateLatency(LatencyAggregator& agg) = 0;

  LatencyId id;
  isl_pw_qpolynomial* latency;
};

struct ComputeLatency : public LatencyBase
{
  size_t start_idx;

  void CalculateLatency(LatencyAggregator&) override;
};

struct RootLatency : public LatencyBase
{
  virtual void CalculateLatency(LatencyAggregator& agg) override;
  LatencyId child_id;
};

struct BranchLatencyBase : public LatencyBase
{
  size_t start_idx;
  std::vector<LatencyId> children_id;
};

struct PipelineLatency : public BranchLatencyBase
{
  void CalculateLatency(LatencyAggregator&) override;
};

struct SequentialLatency : public BranchLatencyBase
{
  void CalculateLatency(LatencyAggregator&) override;
};

template<typename T>
inline constexpr bool HasOneChildV = std::is_same_v<T, RootLatency>;

template<typename T>
inline constexpr bool HasManyChildrenV = std::is_same_v<T, PipelineLatency>
                                       | std::is_same_v<T, SequentialLatency>;

using AggregatorTypes = std::variant<ComputeLatency,
                                     RootLatency,
                                     PipelineLatency,
                                     SequentialLatency>;

struct LatencyAggregator
{
  LatencyAggregator();

  AggregatorTypes& AggregatorAt(LatencyId id);

  double CalculateLatency();

  LatencyId GetRootId() const;

  template<typename T>
  T& AddChild(LatencyId parent)
  {
    LatencyId child_id = aggregators.size();
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

  void SetComputeLatency(mapping::NodeID compute, LatencyId latency);

  void SetLatency(mapping::NodeID compute,
                  __isl_take isl_pw_qpolynomial* latency);

 private:
  LatencyId root_id;
  std::vector<AggregatorTypes> aggregators;
  std::map<mapping::NodeID, LatencyId> compute_to_aggregator;
};

LatencyAggregator
CreateLatencyAggregatorFromMapping(mapping::FusedMapping& mapping);

};