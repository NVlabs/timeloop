#include "loop-analysis/isl-ir.hpp"

#include <boost/algorithm/string/join.hpp>

namespace analysis {

/******************************************************************************
 * Local declarations
 *****************************************************************************/

template<typename T>
std::ostream& StreamOutMapLike(std::ostream& os, const T& map_like)
{
  auto begin = map_like.dim_in_tags.begin();
  auto end = map_like.dim_in_tags.end();

  os << "[";

  if (begin == end)
  {
    os << "] " << map_like.map;
    return os;
  }

  os << *begin;

  for (auto it = begin + 1; it != end; ++it)
  {
    os << ", " << *it;
  }

  os << "] " << map_like.map;
  return os;
}

/******************************************************************************
 * Global function implementations
 *****************************************************************************/

Spatial::Spatial(int spatial_dim) : spatial_dim(spatial_dim)
{
}

std::ostream& operator<<(std::ostream& os, const Temporal&)
{
  os << "Temporal()";
  return os;
}

std::ostream& operator<<(std::ostream& os, const Spatial& t)
{
  os << "Spatial(" << t.spatial_dim << ")";
  return os;
}

std::ostream& operator<<(std::ostream& os, const Sequential&)
{
  os << "Sequential()";
  return os;
}

std::ostream& operator<<(std::ostream& os, const PipelineTemporal&)
{
  os << "PipelineTemporal()";
  return os;
}

std::ostream& operator<<(std::ostream& os, const PipelineSpatial&)
{
  os << "PipelineSpatial()";
  return os;
}

std::ostream& operator<<(std::ostream& os, const SpaceTime& t)
{
  std::visit(
    [&os](auto&& arg)
    {
      os << arg;
    },
    t
  );

  return os;
}

std::ostream& operator<<(std::ostream& os, const LogicalComputeUnit& buf)
{
  os << "LogicalComputeUnit(" << buf.buffer_id << ", " << buf.branch_leaf_id
     << ")";
  return os;
}

std::ostream& operator<<(std::ostream& os, const LogicalBuffer& buf)
{
  os << "LogicalBuffer(";
  os << buf.buffer_id << ", " << buf.dspace_id << ", " << buf.branch_leaf_id;
  os << ")";
  return os;
}

std::ostream& operator<<(std::ostream& os, const Occupancy& s)
{
  return StreamOutMapLike(os, s);
}

std::ostream& operator<<(std::ostream& os, const OpOccupancy& s)
{
  return StreamOutMapLike(os, s);
}

std::ostream& operator<<(std::ostream& os, const Skew& s)
{
  return StreamOutMapLike(os, s);
}

std::ostream& operator<<(std::ostream& os, const Fill& s)
{
  return StreamOutMapLike(os, s);
}

}; // namespace analysis