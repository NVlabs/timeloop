#pragma once

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

  ~IslCtx() {
    if (data) {
      isl_ctx_free(data);
    }
  }
};

struct IslSpace {
  isl_space* data;

  IslSpace(IslCtx& ctx, unsigned nparam, unsigned n_in, unsigned n_out);
  IslSpace(__isl_take isl_space*&& raw) : data(raw) {}

  IslSpace(const IslSpace& other) : data(isl_space_copy(other.data)) {}
  IslSpace(IslSpace&& other) : data(other.data) { other.data = nullptr; }

  ~IslSpace() {
    if (data) {
      isl_space_free(data);
    }
  }

  IslSpace& SetDimName(enum isl_dim_type dim_type, unsigned pos,
                       const char* name) {
    data = isl_space_set_dim_name(data, dim_type, pos, name);
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

  ~IslMap() {
    if (data) {
      isl_map_free(data);
    }
  }
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

struct IslAff {
  isl_aff* data;

  IslAff(__isl_take isl_aff*&& raw) : data(raw) {}

  ~IslAff() {
    if (data) {
      isl_aff_free(data);
    }
  }

  IslAff& SetCoefficientSi(enum isl_dim_type dim_type, int pos, int v) {
    data = isl_aff_set_coefficient_si(data, dim_type, pos, v);
  }
};

IslSpace IslSpaceDomain(IslSpace&& space) {
  return IslSpace(isl_space_domain(space.data));
}

IslMap IslMapReverse(IslMap&& map) {
  return IslMap(isl_map_reverse(map.data));
}
