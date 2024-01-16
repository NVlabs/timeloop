#include "mapping/spacetime.hpp"

namespace spacetime
{

std::ostream& operator<<(std::ostream& os, const Dimension& d)
{
  if (d == Dimension::Time)
  {
    os << "Time";
  }
  else if (d == Dimension::SpaceX)
  {
    os << "SpaceX";
  }
  else if (d == Dimension::SpaceY)
  {
    os << "SpaceY";
  }
  else
  {
    throw std::logic_error("unreachable");
  }

  return os;
}

};