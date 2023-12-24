#pragma once

#include <boost/range/adaptor/indexed.hpp>
#include <boost/range/adaptor/reversed.hpp>
#include <isl/cpp.h>
#include <isl/polynomial.h>

namespace isl {

size_t dim(const map& map, isl_dim_type dim_type);

map dim_projector(space space, isl_dim_type dim_type, size_t start, size_t n);
isl_map* dim_projector(__isl_take isl_space* space, size_t start, size_t n);

/**
 * @brief Reverse of map that out isl_dim_in if element in mask is true.
 * 
 * For example, a space [i0, i1, i2] with mask [True, True, False] yields
 *  [i2] -> [i0, i1, i2]
 */
template<typename RangeT>
isl_map* dim_projector(__isl_take isl_space* space, RangeT mask)
{
  using namespace boost::adaptors;

  auto p_projector = isl_map_identity(isl_space_map_from_set(space));
  for (const auto& [idx, is_projected_out] : mask | indexed(0) | reversed )
  {
    if (is_projected_out)
    {
      p_projector = isl_map_project_out(p_projector, isl_dim_in, idx, 1);
    }
  }
  return p_projector;
}

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

map insert_dims(map map, isl_dim_type dim_type, size_t pos, size_t n_dims);

map move_dims(map map,
              isl_dim_type dst_dim_type, size_t dst,
              isl_dim_type src_dim_type, size_t src,
              size_t n_dims);

isl_map*
map_to_shifted(__isl_take isl_space* domain_space, size_t pos, int shift);
map map_to_shifted(space domain_space, size_t pos, int shift);

map map_to_all_at_dim(space domain_space, size_t pos);

map fix_si(map map, isl_dim_type dim_type, size_t pos, int val);

__isl_give isl_map* bound_dim_si(__isl_take isl_map* map,
                                 isl_dim_type dim_type, size_t pos,
                                 int lower, int upper);

map insert_equal_dims(map map, size_t in_pos, size_t out_pos, size_t n);

__isl_give isl_map*
insert_equal_dims(
  __isl_take isl_map* p_map, size_t in_pos, size_t out_pos, size_t n
);

multi_aff
insert_equal_dims(multi_aff maff, size_t in_pos, size_t out_pos, size_t n);

__isl_give isl_multi_aff*
insert_equal_dims(
  __isl_take isl_multi_aff* p_maff, int in_pos, int out_pos, int n
);

map insert_dummy_dim_ins(map map, size_t pos, size_t n);

isl_pw_qpolynomial* set_card(set set);

isl_pw_qpolynomial* sum_map_range_card(map map);

double val_to_double(isl_val* val);

isl_val* get_val_from_singular(__isl_take isl_pw_qpolynomial* pw_qp);
isl_val* get_val_from_singular(__isl_take isl_pw_qpolynomial_fold* pwf);


map ConstraintDimEquals(map map, size_t n_dims);

map MapToPriorData(size_t n_in_dims, size_t top);


__isl_give isl_map*
lex_lt(__isl_keep isl_space* set_space, size_t start, size_t n_lex);

map map_to_next(const space& space, size_t start, size_t n_lex, size_t n_total);


__isl_give isl_map*
map_to_next(__isl_take isl_set* set, size_t start, size_t n);

map map_to_next(set set, size_t start, size_t n);

__isl_give isl_set*
separate_dependent_bounds(__isl_take isl_set* set, size_t start, size_t n);

std::string pw_qpolynomial_fold_to_str(isl_pw_qpolynomial_fold* pwqf);

__isl_give isl_pw_qpolynomial*
gather_pw_qpolynomial_from_fold(__isl_take isl_pw_qpolynomial_fold* pwqpf);
};  // namespace isl
