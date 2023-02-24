#include "isl-wrapper/tagged.hpp"

std::ostream& operator<<(std::ostream& os, const NoTag& n)
{
  (void) n;
  return os;
}