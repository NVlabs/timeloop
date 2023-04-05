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

  Root::Root(const NodeID& id) : id(id) {}

  For::For(const NodeID& id,
        const std::string& iterator_name,
        const problem::Shape::FlattenedDimensionID& op_dim,
        std::optional<IslAff>&& begin = std::nullopt,
        std::optional<IslAff>&& end = std::nullopt):
    id(id), 
    iterator_name(iterator_name),
    op_dim(op_dim) 
  {
    // TODO : What is IslAff's relation to size_t
  }

  ParFor::ParFor(const NodeID& id,
         const std::string& iterator_name,
         const problem::Shape::FlattenedDimensionID& op_dim,
         std::optional<IslAff>&& begin = std::nullopt,
         std::optional<IslAff>&& end = std::nullopt):
    id(id),
    iterator_name(iterator_name),
    op_dim(op_dim)
  {
    // TODO : What is IslAff's relation to size_t
  }

  Storage::Storage(const NodeID& id,
          const BufferID& buffer,
          const problem::Shape::DataSpaceID& dspace):
    id(id), buffer(buffer), dspace(dspace) {}
  
  Compute::Compute(const NodeID& id,
          const problem::EinsumID& einsum,
          const std::optional<IslPwMultiAff>&& tiling_spec):
    id(id), kernel(einsum)
  {
    // TODO: std::optional<>&& ?? 
  }

  Pipeline::Pipeline(const NodeID& id) 
  {
    children.push_back(id);
  }
};