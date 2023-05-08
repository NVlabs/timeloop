#include "mapping/fused-mapping.hpp"

namespace mapping
{

MappingPath MappingPathsIterator::operator*()
{
  return MappingPath(path_);
}

bool MappingPathsIterator::operator==(const MappingPathsIterator& other) const
{
  // TODO: should check for the same mapping
  return (done_ && other.done_) ||
    (idx_ == other.idx_ && !done_ && !other.done_);
}

bool MappingPathsIterator::operator!=(const MappingPathsIterator& other) const
{
  return !(*this == other);
}

MappingPathsIterator& MappingPathsIterator::operator++()
{
  GetNextPath();
  return *this;
}

void MappingPathsIterator::GetNextPath()
{
  if (dfs_stack_.size() == 0)
  {
    done_ = true;
    return;
  }

  while (dfs_stack_.size() > 0)
  {
    auto& dfs_stack_back = dfs_stack_.back();
    auto backtrack_idx = dfs_stack_back.path_backtrack_idx;
    auto& node = dfs_stack_back.ref_node.get();
    dfs_stack_.pop_back();

    path_.erase(path_.begin() + backtrack_idx, path_.end());
    path_.emplace_back(std::ref(node));

    bool found_leaf = false;
    std::visit(
      [this, &backtrack_idx, &found_leaf](auto&& arg)
      {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, Root>)
        {
          dfs_stack_.emplace_back(
            backtrack_idx + 1,
            mapping_.NodeAt(*arg.child)
          );
        }
        else if constexpr (std::is_same_v<T, For>)
        {
          dfs_stack_.emplace_back(
            backtrack_idx + 1,
            mapping_.NodeAt(*arg.child)
          );
        }
        else if constexpr (std::is_same_v<T, ParFor>)
        {
          dfs_stack_.emplace_back(
            backtrack_idx + 1,
            mapping_.NodeAt(*arg.child)
          );
        }
        else if constexpr (std::is_same_v<T, Storage>)
        {
          dfs_stack_.emplace_back(
            backtrack_idx + 1,
            mapping_.NodeAt(*arg.child)
          );
        }
        else if constexpr (std::is_same_v<T, Compute>)
        {
          found_leaf = true;
        }
        else if constexpr (std::is_same_v<T, Pipeline>)
        {
          for (auto child_id : arg.children)
          {
            dfs_stack_.emplace_back(
              backtrack_idx + 1,
              mapping_.NodeAt(child_id)
            );
          }
        }
      },
      node
    );

    if (found_leaf)
    {
      break;
    }
  }
}

MappingPathsIterator::DfsRecord::DfsRecord(
  size_t backtrack_idx,
  MappingNodeTypes& node
) :
  path_backtrack_idx(backtrack_idx), ref_node(std::ref(node))
{
}

MappingPathsIterator::MappingPathsIterator(FusedMapping& mapping, bool done) :
  mapping_(mapping), dfs_stack_(), path_(), idx_(0), done_(done)
{
  dfs_stack_.emplace_back(0, mapping.NodeAt(mapping.GetRoot().id));
  GetNextPath();
}

MappingPathsIterator MappingPaths::begin()
{
  return MappingPathsIterator(fused_mapping_);
}

MappingPathsIterator MappingPaths::end()
{
  return MappingPathsIterator(fused_mapping_, true);
}

MappingPaths::MappingPaths(FusedMapping& mapping) :
  fused_mapping_(mapping)
{
}

MappingNodeTypes& MappingPathNodeIterator::operator*()
{
  return path_.ref_nodes_.at(idx_).get();
}

bool
MappingPathNodeIterator::operator==(const MappingPathNodeIterator& other) const
{
  return idx_ == other.idx_;
}

bool
MappingPathNodeIterator::operator!=(const MappingPathNodeIterator& other) const
{
  return idx_ != other.idx_;
}

MappingPathNodeIterator& MappingPathNodeIterator::operator++()
{
  idx_ += (idx_ < path_.ref_nodes_.size());
  return *this;
}

MappingPathNodeIterator::MappingPathNodeIterator(
  MappingPath& path,
  size_t idx
) :
  path_(path), idx_(idx)
{
}

MappingPathNodeIterator MappingPath::begin()
{
  return MappingPathNodeIterator(*this, 0);
}

MappingPathNodeIterator MappingPath::end()
{
  return MappingPathNodeIterator(*this, ref_nodes_.size());
}

MappingNodeTypes& MappingPath::back()
{
  return ref_nodes_.back().get();
}

MappingPath::MappingPath(
  std::vector<std::reference_wrapper<MappingNodeTypes>> ref_nodes
) :
  ref_nodes_(ref_nodes)
{
}

MappingPaths GetPaths(FusedMapping& mapping)
{
  return MappingPaths(mapping);
}

}; // namespace mapping