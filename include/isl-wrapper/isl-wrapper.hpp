#pragma once

#include <stdexcept>

#include <isl/aff.h>
#include <isl/ctx.h>
#include <isl/map.h>
#include <isl/set.h>
#include <isl/space.h>

struct IslCtx {
  isl_ctx* data;

  IslCtx() : data(isl_ctx_alloc()) {}
  IslCtx(__isl_take isl_ctx*&& raw) : data(raw) {}
  IslCtx(const IslCtx&) = delete;
  IslCtx(IslCtx&& other) : data(other.data) { other.data = nullptr; }

  IslCtx& operator=(IslCtx&& other)
  {
    data = other.data;
    other.data = nullptr;
    return *this;
  }

  ~IslCtx() {
    if (data) {
      isl_ctx_free(data);
    }
  }
};

struct IslSpace {
  isl_space* data;

  IslSpace(__isl_take isl_space*&& raw) : data(raw) {}
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

struct IslAff {
  isl_aff* data;

  IslAff(__isl_take isl_aff*&& raw) : data(raw) {}

  IslAff(const IslAff& other) : data(isl_aff_copy(other.data)) {}

  ~IslAff() {
    if (data) {
      isl_aff_free(data);
    }
  }

  static IslAff ZeroOnDomainSpace(IslSpace&& domain_space);

  IslAff& SetCoefficientSi(enum isl_dim_type dim_type, int pos, int v) {
    data = isl_aff_set_coefficient_si(data, dim_type, pos, v);
    return *this;
  }
  IslAff& SetConstantSi(int v)
  {
    data = isl_aff_set_constant_si(data, v);
    return *this;
  }
};

struct IslMultiAff {
  isl_multi_aff* data;

  IslMultiAff(__isl_take isl_multi_aff*&& raw) : data(raw) {}

  ~IslMultiAff() {
    if (data) {
      isl_multi_aff_free(data);
    }
  }

  IslAff GetAff(size_t pos) const;
  IslMultiAff& SetAff(size_t pos, IslAff&& aff);

  static IslMultiAff Identity(IslSpace&& space);
  static IslMultiAff Zero(IslSpace&& space);
};

struct IslPwMultiAff
{
  isl_pw_multi_aff* data;

  IslPwMultiAff(__isl_take isl_pw_multi_aff*&& raw) : data(raw) {}

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
};

struct IslBasicMap {
  isl_basic_map* data;

  IslBasicMap(__isl_take isl_basic_map*&& raw) : data(raw) {}
  IslBasicMap(const IslBasicMap& other)
      : data(isl_basic_map_copy(other.data)) {}
  IslBasicMap(IslBasicMap&& other) : data(other.data) { data = nullptr; }
};

struct IslMap {
  isl_map* data;

  IslMap(__isl_take isl_map*&& raw) : data(raw) {}
  IslMap(IslBasicMap&& basic_map)
      : data(isl_map_from_basic_map(basic_map.data)) {
    basic_map.data = nullptr;
  }
  IslMap(IslMultiAff&& multi_aff)
      : data(isl_map_from_multi_aff(multi_aff.data)) {
    multi_aff.data = nullptr;
  }

  IslMap(const IslMap& other) : data(isl_map_copy(other.data)) {}
  IslMap(IslMap&& other) : data(other.data) { other.data = nullptr; }

  IslMap& operator=(IslMap&& other)
  {
    if (data)
    {
      isl_map_free(data);
    }
    data = other.data;
    other.data = nullptr;
    return *this;
  }

  ~IslMap() {
    if (data) {
      isl_map_free(data);
    }
  }

  IslSpace GetSpace() const;

  bool InvolvesDims(isl_dim_type dim_type, size_t first, size_t n) const;
  size_t NumDims(isl_dim_type dim_type) const;

  static IslMap FromMultiAff(IslMultiAff&& multi_aff);
  static IslMap FromMultiAff(IslPwMultiAff&& pw_multi_aff);
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
};


IslSpace IslSpaceDomain(IslSpace&& space);

IslMap IslMapReverse(IslMap&& map);

#define DEFINE_ISL_BINARY_OP(NAME)            \
  template<typename T>                        \
  T NAME(T&& arg1, T&& arg2);

DEFINE_ISL_BINARY_OP(ApplyRange)

DEFINE_ISL_BINARY_OP(Subtract)

IslMap ProjectDims(IslMap&& map, isl_dim_type dim_type, size_t first,
                   size_t n);
IslBasicMap ProjectDims(IslBasicMap&& map, isl_dim_type dim_type, size_t first,
                        size_t n);
IslMap ProjectDimInAfter(IslMap&& map, size_t start);
IslBasicMap ProjectDimInAfter(IslBasicMap&& map, size_t start);