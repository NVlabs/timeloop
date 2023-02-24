#pragma once

#include <stdexcept>

#include <isl/aff.h>
#include <isl/ctx.h>
#include <isl/map.h>
#include <isl/set.h>
#include <isl/space.h>

/******************************************************************************
 * Macros
 *****************************************************************************/

#define DEFINE_ISL_BINARY_OP(NAME, RET_T, T1, T2)                             \
  RET_T NAME(T1&& arg1, T2&& arg2);

#define DEFINE_ISL_STREAMOUT(TYPE)                                            \
  std::ostream& operator<<(std::ostream& os, const TYPE& obj);

#define DEFINE_ISL_SWAP(TYPE)                                                 \
  friend void swap(TYPE& obj1, TYPE& obj2) noexcept;

#define DEFINE_CTOR_DTOR_ASSIGN_SWAP(TYPE, ISL_TYPE)                          \
  TYPE();                                                                     \
  TYPE(const TYPE& other);                                                    \
  TYPE(TYPE&& other);                                                         \
  TYPE(isl_ ## ISL_TYPE*&& raw);                                              \
  TYPE(const std::string& str);                                               \
  ~TYPE();                                                                    \
  TYPE& operator=(const TYPE& other);                                         \
  TYPE& operator=(TYPE&& other);                                              \
  DEFINE_ISL_SWAP(TYPE)

/******************************************************************************
 * Class Prototypes
 *****************************************************************************/

struct IslCtx;
struct IslSpace;
struct IslAff;
struct IslMultiAff;
struct IslPwMultiAff;
struct IslBasicMap;
struct IslMap;
struct IslSet;

/******************************************************************************
 * Classes
 *****************************************************************************/

struct IslCtx
{
  isl_ctx* data;

  IslCtx();
  IslCtx(__isl_take isl_ctx*&& raw);

  ~IslCtx();
};

struct IslSpace
{
  isl_space* data;

  DEFINE_CTOR_DTOR_ASSIGN_SWAP(IslSpace, space)

  static IslSpace Alloc(IslCtx& ctx, unsigned nparam, unsigned n_in,
                        unsigned n_out);
  static IslSpace SetAlloc(IslCtx& ctx, unsigned nparam, unsigned ndim);

  IslSpace& SetDimName(isl_dim_type dim_type, unsigned pos,
                       const std::string& name);
};

struct IslVal
{
  isl_val* data;

  DEFINE_CTOR_DTOR_ASSIGN_SWAP(IslVal, val)

  IslVal(long v);
  IslVal(int v);
  IslVal(unsigned long v);
};

struct IslAff
{
  isl_aff* data;

  DEFINE_CTOR_DTOR_ASSIGN_SWAP(IslAff, aff)

  static IslAff ZeroOnDomainSpace(IslSpace&& domain_space);
  static IslAff ValOnDomainSpace(IslSpace&& domain_space, IslVal&& val);

  IslAff& SetCoefficientSi(enum isl_dim_type dim_type, int pos, int v);
  IslAff& SetConstantSi(int v);
};

struct IslMultiAff {
  isl_multi_aff* data;

  DEFINE_CTOR_DTOR_ASSIGN_SWAP(IslMultiAff, multi_aff)

  static IslMultiAff Identity(IslSpace&& space);
  static IslMultiAff IdentityOnDomainSpace(IslSpace&& space);
  static IslMultiAff Zero(IslSpace&& space);

  IslSpace GetSpace() const;
  IslSpace GetDomainSpace() const;

  IslAff GetAff(size_t pos) const;
  IslMultiAff& SetAff(size_t pos, IslAff&& aff);
};

struct IslPwMultiAff
{
  isl_pw_multi_aff* data;

  DEFINE_CTOR_DTOR_ASSIGN_SWAP(IslPwMultiAff, pw_multi_aff)
};

struct IslBasicMap {
  isl_basic_map* data;

  DEFINE_CTOR_DTOR_ASSIGN_SWAP(IslBasicMap, basic_map)

  size_t NumDims(isl_dim_type dim_type) const;

  DEFINE_ISL_SWAP(IslBasicMap)
};

struct IslMap {
  isl_map* data;

  DEFINE_CTOR_DTOR_ASSIGN_SWAP(IslMap, map)

  IslMap(IslMultiAff&& multi_aff);
  IslMap(IslPwMultiAff&& multi_aff);

  IslMap& Coalesce();

  IslSpace GetSpace() const;
  IslSpace GetDomainSpace() const;

  bool InvolvesDims(isl_dim_type dim_type, size_t first, size_t n) const;
  size_t NumDims(isl_dim_type dim_type) const;
};

struct IslBasicSet
{
  isl_basic_set* data;

  DEFINE_CTOR_DTOR_ASSIGN_SWAP(IslBasicSet, basic_set)
};

struct IslSet {
  isl_set* data;

  DEFINE_CTOR_DTOR_ASSIGN_SWAP(IslSet, set)

  static IslSet Universe(IslSpace&& space);

  IslSet& Coalesce();

  DEFINE_ISL_SWAP(IslSet)
};

DEFINE_ISL_STREAMOUT(IslAff);
DEFINE_ISL_STREAMOUT(IslMultiAff);
DEFINE_ISL_STREAMOUT(IslBasicMap);
DEFINE_ISL_STREAMOUT(IslMap);
DEFINE_ISL_STREAMOUT(IslSet);

IslSpace IslSpaceDomain(IslSpace&& space);

IslMap IslMapReverse(IslMap&& map);

DEFINE_ISL_BINARY_OP(ApplyRange, IslMap, IslMap, IslMap)
DEFINE_ISL_BINARY_OP(Subtract, IslMap, IslMap, IslMap)
DEFINE_ISL_BINARY_OP(Intersect, IslMap, IslMap, IslMap)
DEFINE_ISL_BINARY_OP(Intersect, IslSet, IslSet, IslSet)
DEFINE_ISL_BINARY_OP(IntersectDomain, IslMap, IslMap, IslSet)
DEFINE_ISL_BINARY_OP(GeSet, IslSet, IslAff, IslAff)
DEFINE_ISL_BINARY_OP(LtSet, IslSet, IslAff, IslAff)
DEFINE_ISL_BINARY_OP(LexGeSet, IslSet, IslMultiAff, IslMultiAff)
DEFINE_ISL_BINARY_OP(LexLtSet, IslSet, IslMultiAff, IslMultiAff)

IslMap ProjectDims(IslMap&& map, isl_dim_type dim_type, size_t first,
                   size_t n);
IslBasicMap ProjectDims(IslBasicMap&& map, isl_dim_type dim_type, size_t first,
                        size_t n);
IslMap ProjectDimInAfter(IslMap&& map, size_t start);
IslBasicMap ProjectDimInAfter(IslBasicMap&& map, size_t start);
