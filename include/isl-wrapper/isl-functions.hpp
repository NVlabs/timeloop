#pragma once

#include <isl/cpp.h>

namespace isl {

isl::map
project_dim(isl::map map, isl_dim_type dim_type, size_t start, size_t n);

isl::map project_dim_in_after(isl::map map, size_t start);

isl::map map_from_multi_aff(isl::multi_aff maff);
isl::map map_from_multi_aff(isl::pw_multi_aff maff);

isl::space
space_alloc(isl::ctx ctx, size_t n_params, size_t n_dim_in, size_t n_dim_out);

isl::aff
set_coefficient_si(isl::aff aff, isl_dim_type dim_type, size_t pos, int val);

isl::aff si_on_domain(isl::space space, int val);

isl::map map_to_shifted(isl::space domain_space, size_t pos, int shift);

isl::map fix_si(isl::map map, isl_dim_type dim_type, size_t pos, int val);

};  // namespace isl
