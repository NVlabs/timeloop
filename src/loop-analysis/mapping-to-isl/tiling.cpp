#include "loop-analysis/mapping-to-isl/tiling.hpp"
isl_map* AddNewTileDim(isl_map* p_old_tiling, size_t dim_idx, size_t tile_size)
{
  // The new tiling has one extra dimension at the end
  auto p_new_tiling = isl_map_insert_dims(
    p_old_tiling,
    isl_dim_in,
    isl_map_dim(p_old_tiling, isl_dim_in),
    1
  );

  // Min and max of ori. dimension being tiled as function of tiled dimensions.
  auto p_dim_min = isl_map_dim_min(isl_map_copy(p_new_tiling), dim_idx);
  auto p_dim_max = isl_map_dim_max(isl_map_copy(p_new_tiling), dim_idx);

  // Aff. expr. from tiled dimensions space to value of newest dim
  auto p_new_dim_id = isl_aff_var_on_domain(
    isl_local_space_from_space(isl_pw_aff_get_domain_space(p_dim_min)),
    isl_dim_set,
    isl_pw_aff_dim(p_dim_min, isl_dim_in)-1
  );

  // Aff. expr. from tiled dimension space to tile size constant
  auto p_tile_size = isl_aff_val_on_domain_space(
    isl_pw_aff_get_domain_space(p_dim_min),
    isl_val_int_from_ui(GetIslCtx().get(), tile_size)
  );

  // Pw. aff. expr. from tiled dimension space to tile_size*newest_dim
  auto p_tile_translate = isl_pw_aff_from_aff(isl_aff_mul(
    p_new_dim_id,
    isl_aff_copy(p_tile_size)
  ));

  // What dim_min should be given new tiling
  auto p_new_dim_min = isl_pw_aff_add(
    isl_pw_aff_copy(p_dim_min),
    p_tile_translate
  );

  // What dim_max should be given new tiling
  auto p_new_dim_max = isl_pw_aff_add(
    isl_pw_aff_copy(p_new_dim_min),
    isl_pw_aff_from_aff(
      isl_aff_add_constant_val(
        isl_aff_copy(p_tile_size),
        isl_val_negone(GetIslCtx().get())
      )
    )
  );

  // TODO: I think this is the same as p_new_dim_id
  auto p_new_iter_id = isl_pw_aff_from_aff(isl_aff_var_on_domain(
    isl_local_space_from_space(isl_space_domain(
      isl_map_get_space(p_new_tiling)
    )),
    isl_dim_set,
    isl_map_dim(p_new_tiling, isl_dim_in)-1
  ));

  // The set of valid values of the new tiled dimensions
  auto p_iter_set = isl_map_domain(isl_map_copy(p_new_tiling));
  p_iter_set = isl_set_intersect(
    p_iter_set,
    isl_pw_aff_le_set(
      p_new_iter_id,
      isl_pw_aff_ceil(isl_pw_aff_div(
        p_dim_max,
        isl_pw_aff_from_aff(p_tile_size)
      ))
    )
  );
  p_iter_set = isl_set_intersect(
    p_iter_set,
    isl_pw_aff_ge_set(
      isl_pw_aff_copy(p_new_dim_min),
      p_dim_min
    )
  );

  // The value of iter dims cannot exceed what is available before
  // tiling.
  p_new_tiling = isl_map_intersect_domain(
    p_new_tiling,
    isl_set_copy(p_iter_set)
  );

  // The set of operations need to follow the new tiled bounds
  auto p_identity = isl_pw_aff_from_aff(isl_aff_var_on_domain(
    isl_local_space_from_space(isl_space_range(isl_map_get_space(
      p_new_tiling
    ))),
    isl_dim_set,
    dim_idx
  ));
  p_new_tiling = isl_map_intersect(
    p_new_tiling,
    isl_pw_aff_le_map(p_new_dim_min,
                      isl_pw_aff_copy(p_identity))
  );
  p_new_tiling = isl_map_intersect(
    p_new_tiling,
    isl_pw_aff_ge_map(p_new_dim_max, p_identity)
  );

  return p_new_tiling;
}