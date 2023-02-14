#include "isl-wrapper/isl-wrapper.hpp"

/******************************************************************************
 * Local declarations
 *****************************************************************************/

#define ISL_BINARY_OP_IMPL(NAME, OP, TYPE)                                    \
  template<>                                                                  \
  TYPE NAME(TYPE&& map1, TYPE&& map2)                                         \
  {                                                                           \
    auto result = TYPE(OP(map1.data, map2.data));                             \
    map1.data = nullptr;                                                      \
    map2.data = nullptr;                                                      \
    return result;                                                            \
  }

#define ISL_BASIC_MAP_BINARY_OP_IMPL(NAME, SHORT_OP)                          \
  ISL_BINARY_OP_IMPL(NAME, isl_basic_map_ ## SHORT_OP, IslBasicMap)

#define ISL_MAP_BINARY_OP_IMPL(NAME, SHORT_OP)                                \
  ISL_BINARY_OP_IMPL(NAME, isl_map_ ## SHORT_OP, IslMap)

#define ISL_BOTH_MAP_BINARY_OP_IMPL(NAME, SHORT_OP)                           \
  ISL_BASIC_MAP_BINARY_OP_IMPL(NAME, SHORT_OP)                                \
  ISL_MAP_BINARY_OP_IMPL(NAME, SHORT_OP)

#define SWAP_IMPL(TYPE)                                                       \
  void swap(TYPE& obj1, TYPE& obj2) noexcept                                  \
  {                                                                           \
    using std::swap;                                                          \
    swap(obj1.data, obj2.data);                                               \
  }

/******************************************************************************
 * Global function implementations
 *****************************************************************************/

IslCtx& IslCtx::operator=(IslCtx&& other)
{
  data = other.data;
  other.data = nullptr;
  return *this;
}

IslSpace IslSpace::Alloc(IslCtx& ctx, unsigned nparam, unsigned n_in,
                         unsigned n_out)
{
  return IslSpace(isl_space_alloc(ctx.data, nparam, n_in, n_out));
}

IslSpace IslSpaceDomain(IslSpace&& space) {
  return IslSpace(isl_space_domain(space.data));
}

IslAff IslAff::ZeroOnDomainSpace(IslSpace&& domain_space)
{
  auto aff = isl_aff_zero_on_domain_space(domain_space.data);
  domain_space.data = nullptr;
  return IslAff(std::move(aff));
}

IslAff& IslAff::SetCoefficientSi(enum isl_dim_type dim_type, int pos, int v)
{
  data = isl_aff_set_coefficient_si(data, dim_type, pos, v);
  return *this;
}

IslAff& IslAff::SetConstantSi(int v)
{
  data = isl_aff_set_constant_si(data, v);
  return *this;
}

IslMultiAff IslMultiAff::Identity(IslSpace&& space)
{
  auto maff = isl_multi_aff_identity(space.data);
  space.data = nullptr;
  return IslMultiAff(std::move(maff));
}

IslMultiAff IslMultiAff::Zero(IslSpace&& space)
{
  auto maff = isl_multi_aff_zero(space.data);
  space.data = nullptr;
  return IslMultiAff(std::move(maff));
}

IslAff IslMultiAff::GetAff(size_t pos) const
{
  return IslAff(isl_multi_aff_get_aff(data, pos));
}

IslMultiAff& IslMultiAff::SetAff(size_t pos, IslAff&& aff)
{
  data = isl_multi_aff_set_aff(data, pos, aff.data);
  aff.data = nullptr;
  return *this;
}

IslMap::IslMap() : data(nullptr)
{
}
IslMap::IslMap(__isl_take isl_map*&& raw) :
    data(raw)
{
}
IslMap::IslMap(const IslMap& other) :
    data(isl_map_copy(other.data))
{
}
IslMap::IslMap(IslMap&& other) :
    IslMap()
{
  swap(*this, other);
}
IslMap::IslMap(IslMultiAff&& multi_aff) :
    data(isl_map_from_multi_aff(multi_aff.data))
{
  multi_aff.data = nullptr;
}
IslMap::IslMap(IslPwMultiAff&& multi_aff) :
    data(isl_map_from_pw_multi_aff(multi_aff.data))
{
  multi_aff.data = nullptr;
}

IslMap::~IslMap() {
  if (data) {
    isl_map_free(data);
  }
}

IslMap& IslMap::operator=(IslMap&& other)
{
  if (data)
  {
    isl_map_free(data);
  }
  data = other.data;
  other.data = nullptr;
  return *this;
}

size_t IslMap::NumDims(isl_dim_type dim_type) const
{
  return isl_map_dim(data, dim_type);
}

IslMap IslMapReverse(IslMap&& map) {
  auto new_map = IslMap(isl_map_reverse(map.data));
  map.data = nullptr;
  return new_map;
}

IslMap ProjectDims(IslMap&& map, isl_dim_type dim_type, size_t first, size_t n)
{
  auto result = IslMap(isl_map_project_out(map.data, dim_type, first, n));
  map.data = nullptr;
  return result;
}

IslBasicMap ProjectDims(IslBasicMap&& map, isl_dim_type dim_type, size_t first,
                        size_t n)
{
  map.data = isl_basic_map_project_out(map.data, dim_type, first, n);
  return map;
}
IslMap ProjectDimInAfter(IslMap&& map, size_t start)
{
  return ProjectDims(std::move(map),
                     isl_dim_in,
                     start,
                     map.NumDims(isl_dim_in) - start);
}
IslBasicMap ProjectDimInAfter(IslBasicMap&& map, size_t start)
{
  return ProjectDims(std::move(map),
                     isl_dim_in,
                     start,
                     map.NumDims(isl_dim_in) - start);
}

SWAP_IMPL(IslAff);
SWAP_IMPL(IslMultiAff);
SWAP_IMPL(IslPwMultiAff);
SWAP_IMPL(IslBasicMap);
SWAP_IMPL(IslMap);

ISL_BOTH_MAP_BINARY_OP_IMPL(ApplyRange, apply_range)

ISL_BINARY_OP_IMPL(Subtract, isl_map_subtract, IslMap)
ISL_BINARY_OP_IMPL(Subtract, isl_set_subtract, IslSet)