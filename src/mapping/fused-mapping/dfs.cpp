#include "mapping/fused-mapping.hpp"

namespace mapping
{

bool DfsIterator::operator==(const DfsIterator& other)
{
  if (stack_.size() > 0 and other.stack_.size() > 0)
  {
    return stack_.back() == other.stack_.back();
  }
  return stack_.size() == 0 and other.stack_.size() == 0;
}

bool DfsIterator::operator!=(const DfsIterator& other)
{
  return !(*this == other);
}

DfsIterator& DfsIterator::operator++()
{
  auto keep_going = true;
  while (keep_going)
  {
    auto cur = stack_.back();
    stack_.pop_back();
    std::visit(
      [&](auto&& node)
      {
        using T = std::decay_t<decltype(node)>;

        auto cur_n_loops = node_to_n_loops_.at(cur);
        if constexpr (IsLoopV<T> || IsBranchV<T>)
        {
          cur_n_loops++;
        }

        if constexpr (HasOneChildV<T>)
        {
          stack_.emplace_back(*node.child);
          child_to_parent_[*node.child] = cur;
          node_to_n_loops_[*node.child] = cur_n_loops;
        }
        else if constexpr (HasManyChildrenV<T>)
        {
          for (auto child : node.children)
          {
            stack_.emplace_back(child);
            child_to_parent_[child] = cur;
            node_to_n_loops_[child] = cur_n_loops;
          }
        }
      },
      mapping_.NodeAt(cur)
    );

    if (stack_.size() > 0)
    {
      keep_going = filter_(mapping_.NodeAt(stack_.back()));
    }
    else
    {
      keep_going = false;
    }
  }

  return *this;
}

std::tuple<NodeID, NodeID, size_t> DfsIterator::operator*()
{
  auto cur_id = stack_.back();
  return std::tie(cur_id,
                  child_to_parent_.at(cur_id),
                  node_to_n_loops_.at(cur_id));
}

DfsIterator::DfsIterator(FusedMapping& mapping,
                         const std::vector<NodeID>& stack,
                         std::function<bool(const MappingNodeTypes&)> filter) :
  mapping_(mapping), filter_(filter), stack_(stack)
{
}

DfsIterator DfsRange::begin()
{
  return DfsIterator(mapping_, {mapping_.GetRoot().id}, filter_);
}

DfsIterator DfsRange::end()
{
  return DfsIterator(mapping_, std::vector<NodeID>(), filter_);
}

DfsRange::DfsRange(
  FusedMapping& mapping,
  std::function<bool(const MappingNodeTypes&)> filter
) : mapping_(mapping), filter_(filter)
{
}

template<>
DfsRange IterateInDfsOrder<>(FusedMapping& mapping)
{
  return IterateInDfsOrder<Root,
                           For, ParFor,
                           Storage,
                           Sequential, Pipeline,
                           Compute>(mapping);
}


DfsRange
IterateInDfsOrder(FusedMapping& mapping,
                  std::function<bool(const MappingNodeTypes&)> filter)
{
  return DfsRange(mapping, filter);
}

}; // namespace mapping