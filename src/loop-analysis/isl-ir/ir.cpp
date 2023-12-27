#include "loop-analysis/isl-ir.hpp"

#include "isl-wrapper/isl-functions.hpp"

namespace analysis {

bool IsTemporal(const SpaceTime& st)
{
  return std::holds_alternative<Temporal>(st);
}

LogicalComputeUnit::LogicalComputeUnit(BufferId buffer_id,
                                       mapping::NodeID branch_leaf_id) :
  buffer_id(buffer_id), branch_leaf_id(branch_leaf_id)
{
}

bool LogicalComputeUnit::operator<(const LogicalComputeUnit& other) const
{
  if (buffer_id < other.buffer_id)
  {
    return true;
  }
  else if (buffer_id == other.buffer_id)
  {
    return branch_leaf_id < other.branch_leaf_id;
  }
  return false;
}

bool LogicalComputeUnit::operator==(const LogicalComputeUnit& other) const
{
  return buffer_id == other.buffer_id
         && branch_leaf_id == other.branch_leaf_id;
}

LogicalBuffer::LogicalBuffer(BufferId buffer_id,
                             DataSpaceId dspace_id,
                             mapping::NodeID branch_leaf_id) :
  buffer_id(buffer_id), dspace_id(dspace_id), branch_leaf_id(branch_leaf_id)
{
}

bool LogicalBuffer::operator<(const LogicalBuffer& other) const
{
  if (buffer_id < other.buffer_id)
  {
    return true;
  }
  else if (buffer_id == other.buffer_id && dspace_id < other.dspace_id)
  {
    return true;
  }
  else if (buffer_id == other.buffer_id && dspace_id == other.dspace_id)
  {
    return branch_leaf_id < other.branch_leaf_id;
  }
  return false;
}

bool LogicalBuffer::operator==(const LogicalBuffer& other) const
{
  return buffer_id == other.buffer_id && dspace_id == other.dspace_id
          && branch_leaf_id == other.branch_leaf_id;
}

Skew::Skew()
{
}

Skew::Skew(const std::vector<SpaceTime>& dim_in_tags, isl::map map) :
  dim_in_tags(dim_in_tags), map(std::move(map))
{
  if (dim_in_tags.size() != isl::dim(map, isl_dim_in))
  {
    throw std::logic_error("mismatched space-time tags and map dimensions");
  }
}


Occupancy::Occupancy()
{
}

Occupancy::Occupancy(const std::vector<SpaceTime>& dim_in_tags, isl::map map) :
  dim_in_tags(dim_in_tags), map(std::move(map))
{
}

OpOccupancy::OpOccupancy()
{
}

OpOccupancy::OpOccupancy(const std::vector<SpaceTime>& dim_in_tags,
                         isl::map map) :
  dim_in_tags(dim_in_tags), map(std::move(map))
{
}

Transfers::Transfers()
{
}

Transfers::Transfers(const std::vector<SpaceTime>& dim_in_tags, isl::map map) :
  dim_in_tags(dim_in_tags), map(std::move(map))
{
}


Fill::Fill()
{
}

Fill::Fill(const std::vector<SpaceTime>& dim_in_tags, isl::map map) :
  dim_in_tags(dim_in_tags), map(std::move(map))
{
}

Reads::Reads()
{
}

Reads::Reads(const std::vector<SpaceTime>& dim_in_tags, isl::map map) :
  dim_in_tags(dim_in_tags), map(std::move(map))
{
}

}; // namespace analysis