#pragma once

#include <isl/cpp.h>

namespace isl {

size_t dim(const isl::map& map, isl_dim_type dim_type);

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

isl::map add_dims(isl::map map, isl_dim_type dim_type, size_t n_dims);

isl::map insert_dims(isl::map map,
                     isl_dim_type dim_type, size_t pos, size_t n_dims);

isl::map move_dims(isl::map map,
                   isl_dim_type dst_dim_type, size_t dst,
                   isl_dim_type src_dim_type, size_t src,
                   size_t n_dims);

isl::map map_to_shifted(isl::space domain_space, size_t pos, int shift);

isl::map map_to_all_after(isl::space domain_space,
                          isl_dim_type dim_type, size_t pos);

isl::map fix_si(isl::map map, isl_dim_type dim_type, size_t pos, int val);

isl::map
insert_equal_dims(isl::map map, size_t in_pos, size_t out_pos, size_t n);

isl::map insert_dummy_dim_ins(isl::map map, size_t pos, size_t n);

};  // namespace isl
