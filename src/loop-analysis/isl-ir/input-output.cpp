#include "loop-analysis/isl-ir.hpp"

#include <boost/algorithm/string/join.hpp>

namespace analysis {

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

std::ostream& operator<<(std::ostream& os, const LogicalBuffer& buf)
{
  os << "LogicalBuffer(";
  os << buf.buffer_id << ", " << buf.dspace_id << ", " << buf.branch_leaf_id;
  os << ")";
  return os;
}


std::ostream& operator<<(std::ostream& os, const Skew& s)
{
  auto begin = s.dim_in_tags.begin();
  auto end = s.dim_in_tags.end();

  os << "[";

  if (begin == end)
  {
    os << "] " << s.map;
    return os;
  }

  os << *begin;

  for (auto it = begin + 1; it != end; ++it)
  {
    os << ", " << *it;
  }

  os << "] " << s.map;
  return os;
}

}; // namespace analysis