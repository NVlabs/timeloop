#include "isl-wrapper/isl-functions.hpp"
#include "isl/constraint.h"
#include "barvinok/isl.h"

namespace isl {

size_t dim(const isl::map& map, isl_dim_type dim_type)
{
  return isl_map_dim(map.get(), dim_type);
}

isl::map
project_dim(isl::map map, isl_dim_type dim_type, size_t start, size_t n)
{
  return isl::manage(isl_map_project_out(map.release(), dim_type, start, n));
}

isl::map project_dim_in_after(isl::map map, size_t start)
{
  auto n_dim_in = isl_map_dim(map.get(), isl_dim_in);
  return project_dim(map, isl_dim_in, start, n_dim_in - start);
}

isl::map project_last_dim(isl::map map)
{
  auto n_dim_in = isl_map_dim(map.get(), isl_dim_in);
  return project_dim(map, isl_dim_in, n_dim_in-1, 1);
}

isl::map map_from_multi_aff(isl::multi_aff maff)
{
  return isl::manage(isl_map_from_multi_aff(maff.release()));
}
isl::map map_from_multi_aff(isl::pw_multi_aff maff)
{
  return isl::manage(isl_map_from_pw_multi_aff(maff.release()));
}

isl::space
space_alloc(isl::ctx ctx, size_t n_params, size_t n_dim_in, size_t n_dim_out)
{
  return isl::manage(isl_space_alloc(ctx.release(),
                                     n_params,
                                     n_dim_in,
                                     n_dim_out));
}

isl::space
space_set_alloc(isl::ctx ctx, size_t n_params, size_t n_dim_set)
{
  return isl::manage(isl_space_set_alloc(ctx.release(), n_params, n_dim_set));
}

isl::space
space_from_domain_and_range(isl::space domain, isl::space range)
{
  return isl::manage(isl_space_map_from_domain_and_range(
    domain.release(),
    range.release()
  ));
}

isl::aff
set_coefficient_si(isl::aff aff, isl_dim_type dim_type, size_t pos, int val)
{
  return isl::manage(isl_aff_set_coefficient_si(aff.release(),
                                                dim_type,
                                                pos,
                                                val));
}

isl::aff set_constant_si(isl::aff aff, int val)
{
  return isl::manage(isl_aff_set_constant_si(aff.release(), val));
}

isl::aff si_on_domain(isl::space space, int val)
{
  return isl::manage(isl_aff_val_on_domain_space(
    space.release(),
    isl_val_int_from_si(space.ctx().get(), val)
  ));
}

isl::map add_dims(isl::map map, isl_dim_type dim_type, size_t n_dims)
{
  return isl::manage(isl_map_add_dims(map.release(), dim_type, n_dims));
}

isl::map insert_dims(isl::map map,
                     isl_dim_type dim_type, size_t pos, size_t n_dims)
{
  return isl::manage(isl_map_insert_dims(map.release(),
                                         dim_type, pos, n_dims));
}

isl::map move_dims(isl::map map,
                   isl_dim_type dst_dim_type, size_t dst,
                   isl_dim_type src_dim_type, size_t src,
                   size_t n_dims)
{
  return isl::manage(
    isl_map_move_dims(map.release(),
                      dst_dim_type, dst,
                      src_dim_type, src,
                      n_dims)
  );
}

isl::map map_to_shifted(isl::space domain_space, size_t pos, int shift)
{
  auto p_maff = isl_multi_aff_identity_on_domain_space(domain_space.release());
  p_maff = isl_multi_aff_set_at(
    p_maff,
    pos,
    isl_aff_set_constant_si(isl_multi_aff_get_at(p_maff, pos), shift)
  );
  return isl::manage(isl_map_from_multi_aff(p_maff));
}

isl::map map_to_all_after(isl::space domain_space,
                          isl_dim_type dim_type, size_t pos)
{
  auto p_aff = isl_aff_zero_on_domain_space(domain_space.release());
  p_aff = isl_aff_set_coefficient_si(p_aff, dim_type, pos, 1);
  auto p_pw_aff = isl_pw_aff_from_aff(p_aff);
  return isl::manage(isl_pw_aff_lt_map(
    p_pw_aff,
    isl_pw_aff_copy(p_pw_aff)
  ));
}

isl::map fix_si(isl::map map, isl_dim_type dim_type, size_t pos, int val)
{
  return isl::manage(isl_map_fix_si(map.release(), dim_type, pos, val));
}

isl::map
insert_equal_dims(isl::map map, size_t in_pos, size_t out_pos, size_t n)
{
  auto p_map = map.release();
  p_map = isl_map_insert_dims(p_map, isl_dim_in, in_pos, n);
  p_map = isl_map_insert_dims(p_map, isl_dim_out, out_pos, n);

  auto p_ls = isl_local_space_from_space(isl_map_get_space(p_map));
  for (size_t i = 0; i < n; ++i)
  {
    auto c = isl_constraint_alloc_equality(isl_local_space_copy(p_ls));
    c = isl_constraint_set_coefficient_si(c, isl_dim_in, i + in_pos, 1);
    c = isl_constraint_set_coefficient_si(c, isl_dim_out, i + out_pos, -1);
    p_map = isl_map_add_constraint(p_map, c);
  }
  isl_local_space_free(p_ls);
  
  return isl::manage(p_map);
}

isl::map insert_dummy_dim_ins(isl::map map, size_t pos, size_t n)
{
  auto p_map = map.release();
  p_map = isl_map_insert_dims(p_map, isl_dim_in, pos, n);

  auto p_ls = isl_local_space_from_space(isl_map_get_space(p_map));
  for (size_t i = 0; i < n; ++i)
  {
    auto c = isl_constraint_alloc_equality(isl_local_space_copy(p_ls));
    c = isl_constraint_set_coefficient_si(c, isl_dim_in, i + pos, 1);
    c = isl_constraint_set_constant_si(c, 0);
    p_map = isl_map_add_constraint(p_map, c);
  }
  isl_local_space_free(p_ls);

  return isl::manage(p_map);
}

isl_pw_qpolynomial* sum_map_range_card(isl::map map)
{
  auto p_domain = map.domain().release();
  auto p_count = isl_map_card(map.release());
  return isl_set_apply_pw_qpolynomial(p_domain, p_count);
}

double val_to_double(isl_val* val)
{
  auto num = isl_val_get_num_si(val);
  auto den = isl_val_get_den_si(val);
  isl_val_free(val);
  return (double)num / (double)den;
}

isl_val* get_val_from_singular_qpolynomial(isl_pw_qpolynomial* pw_qp)
{
  return isl_pw_qpolynomial_eval(
    pw_qp,
    isl_point_zero(isl_pw_qpolynomial_get_domain_space(pw_qp))
  );
}

};  // namespace isl
