#pragma once

#include <isl/aff.h>
#include <isl/map.h>

#include "isl-wrapper/ctx-manager.hpp"

/**
 * @brief Adds a new dimension to a tiling relationship.
 *
 * @param p_old_tiling original iteration -> operation relationship.
 * @param dim_idx dimension of operation space being tiled.
 * @param tile_size the size of the tile in the dim_idx direction.
 */
isl_map* AddNewTileDim(isl_map* p_old_tiling, size_t dim_idx, size_t tile_size);