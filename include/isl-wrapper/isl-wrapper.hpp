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

#define DEFINE_ISL_COPY_CTR(TYPE, ISL_TYPE)                                   \
  TYPE(const TYPE& other) : data(isl_ ## ISL_TYPE ## _copy(other.data)) {}

#define DEFINE_ISL_SWAP(TYPE)                                                 \
  friend void swap(TYPE& obj1, TYPE& obj2) noexcept;

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

  IslCtx() : data(isl_ctx_alloc()) {}
  IslCtx(__isl_take isl_ctx* raw) : data(raw) {}

  ~IslCtx() {
  }
};

struct IslSpace
{
  isl_space* data;

  IslSpace(__isl_take isl_space*&& raw) : data(raw) { raw = nullptr; }
  IslSpace(const IslSpace& other) : data(isl_space_copy(other.data)) {}
  IslSpace(IslSpace&& other) : data(other.data) { other.data = nullptr; }

  ~IslSpace() {
    if (data) {
      isl_space_free(data);
    }
  }

  static IslSpace Alloc(IslCtx& ctx, unsigned nparam, unsigned n_in,
                        unsigned n_out);
  static IslSpace SetAlloc(IslCtx& ctx, unsigned nparam, unsigned ndim);
  static IslSpace ParamsAlloc(IslCtx& ctx, unsigned nparam);

  IslSpace& SetDimName(enum isl_dim_type dim_type, unsigned pos,
                       const std::string& name) {
    data = isl_space_set_dim_name(data, dim_type, pos, name.c_str());
    return *this;
  }
};

struct IslVal
{
  isl_val* data;

  IslVal() : data(nullptr) {}
  IslVal(__isl_take isl_val*&& raw) : data(raw) { raw = nullptr; }
  DEFINE_ISL_COPY_CTR(IslVal, val)
  IslVal(IslVal&& other) : IslVal() { swap(*this, other); }

  IslVal(long v);
  IslVal(int v);
  IslVal(unsigned long v);

  DEFINE_ISL_SWAP(IslVal)
};

struct IslAff
{
  isl_aff* data;

  IslAff() : data(nullptr) {}
  IslAff(__isl_take isl_aff*&& raw) : data(raw) { raw = nullptr; }
  IslAff(const IslAff& other) : data(isl_aff_copy(other.data)) {}
  IslAff(IslAff&& other) : IslAff() { swap(*this, other); }

  ~IslAff() {
    if (data) {
      isl_aff_free(data);
    }
  }

  static IslAff ZeroOnDomainSpace(IslSpace&& domain_space);
  static IslAff ValOnDomainSpace(IslSpace&& domain_space, IslVal&& val);

  IslAff& SetCoefficientSi(enum isl_dim_type dim_type, int pos, int v);
  IslAff& SetConstantSi(int v);

  DEFINE_ISL_SWAP(IslAff)
};

struct IslMultiAff {
  isl_multi_aff* data;

  IslMultiAff() : data(nullptr) {}
  IslMultiAff(__isl_take isl_multi_aff*&& raw) : data(raw) { raw = nullptr; }
  DEFINE_ISL_COPY_CTR(IslMultiAff, multi_aff)

  ~IslMultiAff() {
    if (data) {
      isl_multi_aff_free(data);
    }
  }

  static IslMultiAff Identity(IslSpace&& space);
  static IslMultiAff IdentityOnDomainSpace(IslSpace&& space);
  static IslMultiAff Zero(IslSpace&& space);

  IslSpace GetSpace() const;
  IslSpace GetDomainSpace() const;

  IslAff GetAff(size_t pos) const;
  IslMultiAff& SetAff(size_t pos, IslAff&& aff);

  DEFINE_ISL_SWAP(IslMultiAff)
};

struct IslPwMultiAff
{
  isl_pw_multi_aff* data;

  IslPwMultiAff(__isl_take isl_pw_multi_aff*&& raw) : data(raw)
  {
    raw = nullptr;
  }

  IslPwMultiAff(const IslPwMultiAff& other)
    : data(isl_pw_multi_aff_copy(other.data)) {}
  IslPwMultiAff(IslPwMultiAff&& other) : data(other.data)
  {
    other.data = nullptr;
  }

  IslPwMultiAff& operator=(const IslPwMultiAff& other)
  {
    if (data)
    {
      isl_pw_multi_aff_free(data);
    }
    data = isl_pw_multi_aff_copy(other.data);
    return *this;
  }

  DEFINE_ISL_SWAP(IslPwMultiAff)
};

struct IslBasicMap {
  isl_basic_map* data;

  IslBasicMap(__isl_take isl_basic_map*&& raw) : data(raw) {}
  IslBasicMap(const IslBasicMap& other)
      : data(isl_basic_map_copy(other.data)) {}
  IslBasicMap(IslBasicMap&& other) : data(other.data) { data = nullptr; }

  size_t NumDims(isl_dim_type dim_type) const;

  DEFINE_ISL_SWAP(IslBasicMap)
};

struct IslMap {
  isl_map* data;

  IslMap();
  IslMap(__isl_take isl_map*&& raw);
  IslMap(const IslMap& other);
  IslMap(IslMap&& other);

  IslMap(IslMultiAff&& multi_aff);
  IslMap(IslPwMultiAff&& multi_aff);

  ~IslMap();

  IslMap& operator=(IslMap&& other);

  IslMap& Coalesce();

  IslSpace GetSpace() const;

  bool InvolvesDims(isl_dim_type dim_type, size_t first, size_t n) const;
  size_t NumDims(isl_dim_type dim_type) const;

  DEFINE_ISL_SWAP(IslMap)
};

struct IslBasicSet
{
  isl_basic_set* data;

  IslBasicSet() : data(nullptr) {}
  IslBasicSet(__isl_take isl_basic_set*&& raw) : data(raw) { raw = nullptr; }
  DEFINE_ISL_COPY_CTR(IslBasicSet, basic_set)
  IslBasicSet(IslBasicSet&& other) : IslBasicSet() { swap(*this, other); }

  DEFINE_ISL_SWAP(IslBasicSet)
};

struct IslSet {
  isl_set* data;

  IslSet(__isl_take isl_set*&& raw) : data(raw) {}
  IslSet(const IslSet& other) : data(isl_set_copy(other.data)) {}
  IslSet(IslSet&& other) : data(other.data) { other.data = nullptr; }

  ~IslSet() {
    if (data) {
      isl_set_free(data);
    }
  }

  static IslSet Universe(IslSpace&& space);

  IslSet& operator=(IslSet&& other);

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
