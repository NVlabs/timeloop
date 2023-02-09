#include "isl-wrapper/isl-wrapper.hpp"

/******************************************************************************
 * Local declarations
 *****************************************************************************/

#define ISL_BINARY_OP_IMPL(NAME, OP, TYPE)        \
  template<>                                      \
  TYPE NAME(TYPE&& map1, TYPE&& map2)             \
  {                                               \
    auto result = TYPE(OP(map1.data, map2.data)); \
    map1.data = nullptr;                          \
    map2.data = nullptr;                          \
    return result;                                \
  }

#define ISL_BASIC_MAP_BINARY_OP_IMPL(NAME, SHORT_OP) \
  ISL_BINARY_OP_IMPL(NAME, isl_basic_map_ ## SHORT_OP, IslBasicMap)

#define ISL_MAP_BINARY_OP_IMPL(NAME, SHORT_OP) \
  ISL_BINARY_OP_IMPL(NAME, isl_map_ ## SHORT_OP, IslMap)

#define ISL_BOTH_MAP_BINARY_OP_IMPL(NAME, SHORT_OP) \
  ISL_BASIC_MAP_BINARY_OP_IMPL(NAME, SHORT_OP)      \
  ISL_MAP_BINARY_OP_IMPL(NAME, SHORT_OP)

/******************************************************************************
 * Global function implementations
 *****************************************************************************/

IslSpace IslSpaceDomain(IslSpace&& space) {
  return IslSpace(isl_space_domain(space.data));
}

IslMap IslMapReverse(IslMap&& map) {
  return IslMap(isl_map_reverse(map.data));
}

ISL_BOTH_MAP_BINARY_OP_IMPL(ApplyRange, apply_range)

ISL_BINARY_OP_IMPL(Subtract, isl_map_subtract, IslMap)
ISL_BINARY_OP_IMPL(Subtract, isl_set_subtract, IslSet)