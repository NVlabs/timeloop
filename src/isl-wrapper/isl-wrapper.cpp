#include "isl-wrapper/isl-wrapper.hpp"
#include "isl-wrapper/ctx-manager.hpp"

#include <ostream>

/******************************************************************************
 * Local declarations
 *****************************************************************************/

#define ISL_STREAMOUT_IMPL(TYPE, FUNC)                                        \
  std::ostream& operator<<(std::ostream& os, const TYPE& obj)                 \
  {                                                                           \
    os << FUNC(obj.data);                                                     \
    return os;                                                                \
  }

#define ISL_BINARY_OP_IMPL(NAME, OP, TYPE)                                    \
  TYPE NAME(TYPE&& obj1, TYPE&& obj2)                                         \
  {                                                                           \
    auto result = TYPE(OP(obj1.data, obj2.data));                             \
    obj1.data = nullptr;                                                      \
    obj2.data = nullptr;                                                      \
    return result;                                                            \
  }

#define ISL_BINARY_OP_IMPL_2(NAME, OP, RET_T, T1, T2)                         \
  RET_T NAME(T1&& obj1, T2&& obj2)                                            \
  {                                                                           \
    auto result = RET_T(OP(obj1.data, obj2.data));                            \
    obj1.data = nullptr;                                                      \
    obj2.data = nullptr;                                                      \
    return result;                                                            \
  }

#define ISL_BASIC_MAP_BINARY_OP_IMPL(NAME, SHORT_OP)                          \
  ISL_BINARY_OP_IMPL(NAME, isl_basic_map_ ## SHORT_OP, IslBasicMap)

#define ISL_MAP_BINARY_OP_IMPL(NAME, SHORT_OP)                                \
  ISL_BINARY_OP_IMPL(NAME, isl_map_ ## SHORT_OP, IslMap)

#define ISL_BOTH_MAP_BINARY_OP_IMPL(NAME, SHORT_OP)                           \
  ISL_BASIC_MAP_BINARY_OP_IMPL(NAME, SHORT_OP)                                \
  ISL_MAP_BINARY_OP_IMPL(NAME, SHORT_OP)

#define ISL_BASIC_SET_BINARY_OP_IMPL(NAME, SHORT_OP)                          \
  ISL_BINARY_OP_IMPL(NAME, isl_basic_set_ ## SHORT_OP, IslBasicSet)

#define ISL_SET_BINARY_OP_IMPL(NAME, SHORT_OP)                                \
  ISL_BINARY_OP_IMPL(NAME, isl_set_ ## SHORT_OP, IslSet)

#define ISL_BOTH_SET_BINARY_OP_IMPL(NAME, SHORT_OP)                           \
  ISL_BASIC_SET_BINARY_OP_IMPL(NAME, SHORT_OP)                                \
  ISL_SET_BINARY_OP_IMPL(NAME, SHORT_OP)

#define DEFAULT_CTOR_IMPL(TYPE, ISL_TYPE)                                     \
  TYPE::TYPE() : data(nullptr) {}

#define COPY_CTOR_IMPL(TYPE, ISL_TYPE)                                        \
  TYPE::TYPE(const TYPE& other) :                                             \
    data(isl_ ## ISL_TYPE ## _copy(other.data)) {}

#define MOVE_CTOR_IMPL(TYPE, ISL_TYPE)                                        \
  TYPE::TYPE(TYPE&& other) : TYPE() { swap(*this, other); }

#define RAW_PTR_CTOR_IMPL(TYPE, ISL_TYPE)                                     \
  TYPE::TYPE(__isl_take isl_ ## ISL_TYPE*&& raw) :                            \
    data(raw)                                                                 \
  {                                                                           \
    raw = nullptr;                                                            \
  }

#define CTOR_IMPL(TYPE, ISL_TYPE)                                             \
  DEFAULT_CTOR_IMPL(TYPE, ISL_TYPE)                                           \
  COPY_CTOR_IMPL(TYPE, ISL_TYPE)                                              \
  MOVE_CTOR_IMPL(TYPE, ISL_TYPE)                                              \
  RAW_PTR_CTOR_IMPL(TYPE, ISL_TYPE)                                           \

#define DTOR_IMPL(TYPE, ISL_TYPE)                                             \
  TYPE::~TYPE()                                                               \
  {                                                                           \
    if (data)                                                                 \
    {                                                                         \
      isl_ ## ISL_TYPE ## _free(data);                                        \
    }                                                                         \
  }

#define CTOR_DTOR_IMPL(TYPE, ISL_TYPE)                                        \
  CTOR_IMPL(TYPE, ISL_TYPE)                                                   \
  DTOR_IMPL(TYPE, ISL_TYPE)

#define COPY_ASSIGN_IMPL(TYPE, ISL_TYPE)                                      \
  TYPE& TYPE::operator=(const TYPE& other)                                    \
  {                                                                           \
    data = isl_ ## ISL_TYPE ## _copy(other.data);                             \
  }

#define MOVE_ASSIGN_IMPL(TYPE, ISL_TYPE)                                      \
  TYPE& TYPE::operator=(TYPE&& other)                                         \
  {                                                                           \
    swap(*this, other);                                                       \
  }

#define ASSIGN_IMPL(TYPE, ISL_TYPE)                                           \
  COPY_ASSIGN_IMPL(TYPE, ISL_TYPE)                                            \
  MOVE_ASSIGN_IMPL(TYPE, ISL_TYPE)

#define SWAP_IMPL(TYPE)                                                       \
  void swap(TYPE& obj1, TYPE& obj2) noexcept                                  \
  {                                                                           \
    using std::swap;                                                          \
    swap(obj1.data, obj2.data);                                               \
  }

#define CTOR_DTOR_ASSIGN_SWAP_IMPL(TYPE, ISL_TYPE)                            \
  CTOR_DTOR_IMPL(TYPE, ISL_TYPE)                                              \
  ASSIGN_IMPL(TYPE, ISL_TYPE)                                                 \
  SWAP_IMPL(TYPE)

/******************************************************************************
 * Global class methods
 *****************************************************************************/

IslCtx::IslCtx() : data(isl_ctx_alloc()) {}
IslCtx::IslCtx(__isl_take isl_ctx*&& raw) : data(raw) { raw = nullptr; }

IslCtx::~IslCtx()
{
  // TODO: figure out the RAII part here
}

CTOR_DTOR_ASSIGN_SWAP_IMPL(IslSpace, space)

IslSpace::~IslSpace()
{
  if (data) {
    isl_space_free(data);
  }
}

IslSpace& IslSpace::SetDimName(isl_dim_type dim_type, unsigned pos,
                               const std::string& name) {
  data = isl_space_set_dim_name(data, dim_type, pos, name.c_str());
  return *this;
}

IslSpace IslSpace::Alloc(IslCtx& ctx, unsigned nparam, unsigned n_in,
                         unsigned n_out)
{
  return IslSpace(isl_space_alloc(ctx.data, nparam, n_in, n_out));
}

IslSpace IslSpaceDomain(IslSpace&& space) {
  auto result = IslSpace(isl_space_domain(space.data));
  space.data = nullptr;
  return result;
}

CTOR_DTOR_ASSIGN_SWAP_IMPL(IslVal, val)

IslVal::IslVal(long v) : data(isl_val_int_from_ui(GetIslCtx().data, v)) {}
IslVal::IslVal(int v) : data(isl_val_int_from_ui(GetIslCtx().data, v)) {}
IslVal::IslVal(unsigned long v) :
  data(isl_val_int_from_ui(GetIslCtx().data, v))
{
}

CTOR_DTOR_ASSIGN_SWAP_IMPL(IslAff, aff)

IslAff IslAff::ZeroOnDomainSpace(IslSpace&& domain_space)
{
  auto aff = isl_aff_zero_on_domain_space(domain_space.data);
  domain_space.data = nullptr;
  return IslAff(std::move(aff));
}

IslAff IslAff::ValOnDomainSpace(IslSpace&& domain_space, IslVal&& val)
{
  auto aff = isl_aff_val_on_domain_space(domain_space.data, val.data);
  domain_space.data = nullptr;
  val.data = nullptr;
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

CTOR_DTOR_ASSIGN_SWAP_IMPL(IslMultiAff, multi_aff)

IslMultiAff IslMultiAff::Identity(IslSpace&& space)
{
  auto maff = isl_multi_aff_identity(space.data);
  space.data = nullptr;
  return IslMultiAff(std::move(maff));
}

IslMultiAff IslMultiAff::IdentityOnDomainSpace(IslSpace&& space)
{
  auto maff = isl_multi_aff_identity_on_domain_space(space.data);
  space.data = nullptr;
  return IslMultiAff(std::move(maff));
}

IslMultiAff IslMultiAff::Zero(IslSpace&& space)
{
  auto maff = isl_multi_aff_zero(space.data);
  space.data = nullptr;
  return IslMultiAff(std::move(maff));
}

IslSpace IslMultiAff::GetSpace() const
{
  return IslSpace(isl_multi_aff_get_space(data));
}

IslSpace IslMultiAff::GetDomainSpace() const
{
  return IslSpace(isl_multi_aff_get_domain_space(data));
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

CTOR_DTOR_ASSIGN_SWAP_IMPL(IslPwMultiAff, pw_multi_aff)

CTOR_DTOR_ASSIGN_SWAP_IMPL(IslBasicMap, basic_map)

CTOR_DTOR_ASSIGN_SWAP_IMPL(IslMap, map)

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

IslMap& IslMap::Coalesce()
{
  data = isl_map_coalesce(data);
  return *this;
}

IslSpace IslMap::GetSpace() const
{
  return IslSpace(isl_map_get_space(data));
}
IslSpace IslMap::GetDomainSpace() const
{
  return IslSpace(isl_space_domain(isl_map_get_space(data)));
}

bool IslMap::InvolvesDims(isl_dim_type dim_type, size_t first, size_t n) const
{
  return isl_map_involves_dims(data, dim_type, first, n);
}

size_t IslMap::NumDims(isl_dim_type dim_type) const
{
  return isl_map_dim(data, dim_type);
}

CTOR_DTOR_ASSIGN_SWAP_IMPL(IslBasicSet, basic_set)

CTOR_DTOR_ASSIGN_SWAP_IMPL(IslSet, set)

IslSet IslSet::Universe(IslSpace&& space)
{
  auto set = isl_set_universe(space.data);
  space.data = nullptr;
  return IslSet(std::move(set));
}

IslSet& IslSet::Coalesce()
{
  data = isl_set_coalesce(data);
  return *this;
}

/******************************************************************************
 * Global function implementations
 *****************************************************************************/

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

ISL_STREAMOUT_IMPL(IslAff, isl_aff_to_str);
ISL_STREAMOUT_IMPL(IslMultiAff, isl_multi_aff_to_str);
ISL_STREAMOUT_IMPL(IslBasicMap, isl_basic_map_to_str);
ISL_STREAMOUT_IMPL(IslMap, isl_map_to_str);
ISL_STREAMOUT_IMPL(IslSet, isl_set_to_str);

ISL_BOTH_MAP_BINARY_OP_IMPL(ApplyRange, apply_range)
ISL_BOTH_MAP_BINARY_OP_IMPL(Intersect, intersect)
ISL_BOTH_SET_BINARY_OP_IMPL(Intersect, intersect)

ISL_BINARY_OP_IMPL_2(IntersectDomain,
                     isl_map_intersect_domain,
                     IslMap,
                     IslMap,
                     IslSet)

ISL_BINARY_OP_IMPL_2(GeSet,
                     isl_aff_ge_set,
                     IslSet,
                     IslAff,
                     IslAff)
ISL_BINARY_OP_IMPL_2(LtSet,
                     isl_aff_lt_set,
                     IslSet,
                     IslAff,
                     IslAff)

ISL_BINARY_OP_IMPL_2(LexGeSet,
                     isl_multi_aff_lex_ge_set,
                     IslSet,
                     IslMultiAff,
                     IslMultiAff)
ISL_BINARY_OP_IMPL_2(LexLtSet,
                     isl_multi_aff_lex_lt_set,
                     IslSet,
                     IslMultiAff,
                     IslMultiAff)

ISL_BINARY_OP_IMPL(Subtract, isl_map_subtract, IslMap)
ISL_BINARY_OP_IMPL(Subtract, isl_set_subtract, IslSet)