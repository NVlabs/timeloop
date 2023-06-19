#pragma once

#include <isl/cpp.h>
#include "isl/polynomial.h"

namespace isl {

size_t dim(const map& map, isl_dim_type dim_type);

map project_dim(map map, isl_dim_type dim_type, size_t start, size_t n);

map project_dim_in_after(map map, size_t start);

map project_last_dim(map map);

map map_from_multi_aff(multi_aff maff);
map map_from_multi_aff(pw_multi_aff maff);

space space_alloc(ctx ctx, size_t n_params, size_t n_dim_in, size_t n_dim_out);
space space_set_alloc(ctx ctx, size_t n_params, size_t n_dim_set);
space space_from_domain_and_range(space domain, space range);

aff set_coefficient_si(aff aff, isl_dim_type dim_type, size_t pos, int val);
aff set_constant_si(aff aff, int val);

aff si_on_domain(space space, int val);

map add_dims(map map, isl_dim_type dim_type, size_t n_dims);

map insert_dims(map map,
                isl_dim_type dim_type, size_t pos, size_t n_dims);

map move_dims(map map,
              isl_dim_type dst_dim_type, size_t dst,
              isl_dim_type src_dim_type, size_t src,
              size_t n_dims);

isl_map*
map_to_shifted(__isl_take isl_space* domain_space, size_t pos, int shift);
map map_to_shifted(space domain_space, size_t pos, int shift);

map map_to_all_at_dim(space domain_space, size_t pos);

map fix_si(map map, isl_dim_type dim_type, size_t pos, int val);

map
insert_equal_dims(map map, size_t in_pos, size_t out_pos, size_t n);

multi_aff
insert_equal_dims(multi_aff maff, size_t in_pos, size_t out_pos, size_t n);

isl_multi_aff*
insert_equal_dims(isl_multi_aff* p_maff, int in_pos, int out_pos, int n);

map insert_dummy_dim_ins(map map, size_t pos, size_t n);

isl_pw_qpolynomial* set_card(set set);

isl_pw_qpolynomial* sum_map_range_card(map map);

double val_to_double(isl_val* val);

isl_val* get_val_from_singular_qpolynomial(isl_pw_qpolynomial* pw_qp);

isl_val* get_val_from_singular_qpolynomial_fold(isl_pw_qpolynomial_fold* pwf);

map ConstraintDimEquals(map map, size_t n_dims);

map MapToPriorData(size_t n_in_dims, size_t top);

};  // namespace isl
