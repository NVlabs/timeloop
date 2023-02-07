#include "mapping/fused-mapping.hpp"

namespace mapping
{
  MappingNodeIterator& MappingNodeIterator::operator++()
  {
    ++cur_;
    return *this;
  }

  bool MappingNodeIterator::operator!=(const MappingNodeIterator& other) const
  {
    return cur_ != other.cur_;
  }

  MappingNodeTypes& MappingNodeIterator::operator*()
  {
    return cur_->second;
  }

  FusedMapping::FusedMapping()
  {
    nodes_.emplace(
      std::make_pair(0, MappingNodeTypes(std::in_place_type<Root>, 0)));
  }

  const MappingNodeTypes& FusedMapping::NodeAt(const NodeID& node_id) const
  {
    return nodes_.at(node_id);
  }

  MappingNodeTypes& FusedMapping::NodeAt(const NodeID& node_id)
  {
    return nodes_.at(node_id);
  }

  const Root& FusedMapping::GetRoot() const
  {
    return std::get<Root>(nodes_.at(0));
  }

  MappingNodeIterator FusedMapping::begin()
  {
    return MappingNodeIterator(nodes_.begin());
  }

  MappingNodeIterator FusedMapping::end()
  {
    return MappingNodeIterator(nodes_.end());
  }


};