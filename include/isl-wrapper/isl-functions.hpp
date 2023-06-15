#pragma once

#include <isl/cpp.h>
#include "isl/polynomial.h"

namespace isl {

size_t dim(const isl::map& map, isl_dim_type dim_type);

isl::map
project_dim(isl::map map, isl_dim_type dim_type, size_t start, size_t n);

isl::map project_dim_in_after(isl::map map, size_t start);

isl::map project_last_dim(isl::map map);

isl::map map_from_multi_aff(isl::multi_aff maff);
isl::map map_from_multi_aff(isl::pw_multi_aff maff);

isl::space
space_alloc(isl::ctx ctx, size_t n_params, size_t n_dim_in, size_t n_dim_out);
isl::space space_set_alloc(isl::ctx ctx, size_t n_params, size_t n_dim_set);
isl::space space_from_domain_and_range(isl::space domain, isl::space range);

isl::aff
set_coefficient_si(isl::aff aff, isl_dim_type dim_type, size_t pos, int val);
isl::aff set_constant_si(isl::aff aff, int val);

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

isl::multi_aff
insert_equal_dims(isl::multi_aff maff, size_t in_pos, size_t out_pos, size_t n);

isl_multi_aff*
insert_equal_dims(isl_multi_aff* p_maff, int in_pos, int out_pos, int n);

isl::map insert_dummy_dim_ins(isl::map map, size_t pos, size_t n);

isl_pw_qpolynomial* set_card(isl::set set);

isl_pw_qpolynomial* sum_map_range_card(isl::map map);

double val_to_double(isl_val* val);

isl_val* get_val_from_singular_qpolynomial(isl_pw_qpolynomial* pw_qp);

isl_val* get_val_from_singular_qpolynomial_fold(isl_pw_qpolynomial_fold* pwf);

isl::map ConstraintDimEquals(isl::map map, size_t n_dims);

isl::map MapToPriorData(size_t n_in_dims, size_t top);

};  // namespace isl