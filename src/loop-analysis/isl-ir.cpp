#include "loop-analysis/isl-ir.hpp"

namespace analysis {

LogicalBuffer::LogicalBuffer(BufferID buffer_id,
                             DataSpaceID dspace_id,
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

std::ostream& operator<<(std::ostream& os, const LogicalBuffer& buf)
{
  os << "LogicalBuffer(";
  os << buf.buffer_id << ", " << buf.dspace_id << ", " << buf.branch_leaf_id;
  os << ")";
  return os;
}


Skew::Skew()
{
}

Skew::Skew(const std::vector<spacetime::Dimension>& dim_in_tags,
                     isl::map map) :
  dim_in_tags(dim_in_tags), map(std::move(map))
{
}


Occupancy::Occupancy()
{
}

Occupancy::Occupancy(const std::vector<spacetime::Dimension>& dim_in_tags,
                     isl::map map) :
  dim_in_tags(dim_in_tags), map(std::move(map))
{
}

Transfers::Transfers()
{
}

Transfers::Transfers(const std::vector<spacetime::Dimension>& dim_in_tags,
                     isl::map map) :
  dim_in_tags(dim_in_tags), map(std::move(map))
{
}


Fill::Fill()
{
}

Fill::Fill(const std::vector<spacetime::Dimension>& dim_in_tags,
           isl::map map) :
  dim_in_tags(dim_in_tags), map(std::move(map))
{
}

Reads::Reads()
{
}

Reads::Reads(const std::vector<spacetime::Dimension>& dim_in_tags,
             isl::map map) :
  dim_in_tags(dim_in_tags), map(std::move(map))
{
}

}; // namespace analysis