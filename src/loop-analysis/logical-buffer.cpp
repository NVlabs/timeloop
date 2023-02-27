#include "loop-analysis/isl-ir.hpp"

namespace analysis
{

std::ostream& operator<<(std::ostream& os, const LogicalBuffer& buf)
{
  os << "LogicalBuffer(";
  os << buf.buffer_id << ", " << buf.dspace_id << ", " << buf.branch_leaf_id;
  os << ")";
  return os;
}

};