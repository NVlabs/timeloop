/* Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "loop-analysis/coordinate-space-tile-info.hpp"

namespace tiling
{

// additional shape and occupancy constraints added during tiling
void ExtraTileConstraintInfo::Set(const std::uint64_t shape, const std::uint64_t occupancy)
{
  shape_ = shape;
  occupancy_ = occupancy;
  set_ = true;
}

std::uint64_t ExtraTileConstraintInfo::GetShape() const
{
  return shape_;
}

std::uint64_t ExtraTileConstraintInfo::GetOccupancy() const
{
  return occupancy_;
}

// interface object between sparse modeling module and density models
// tells the density models which set of tiles are we looking at
void CoordinateSpaceTileInfo::Clear()
{
  shape_ = 0;
  subnests_ = {};
}

void CoordinateSpaceTileInfo::Set(std::uint64_t shape, problem::Shape::DataSpaceID data_space_id)
{
  shape_ = shape;
  dspace_id_ = data_space_id;
}

void CoordinateSpaceTileInfo::Set(std::uint64_t shape, std::vector <loop::Descriptor> subnests, problem::Shape::DataSpaceID data_space_id)
{
  shape_ = shape;
  subnests_ = subnests;
  dspace_id_ = data_space_id;
}

void CoordinateSpaceTileInfo::Set(std::uint64_t shape, problem::Shape::DataSpaceID data_space_id, ExtraTileConstraintInfo extra_tile_constraint)
{
  shape_ = shape;
  dspace_id_ = data_space_id;
  extra_tile_constraint_ = extra_tile_constraint;
}

std::uint64_t CoordinateSpaceTileInfo::GetShape() const
{
  return shape_;
}

std::vector <loop::Descriptor> CoordinateSpaceTileInfo::GetSubnests() const
{ return subnests_; }

bool CoordinateSpaceTileInfo::HasExtraConstraintInfo() const
{
  return extra_tile_constraint_.set_;
}

ExtraTileConstraintInfo CoordinateSpaceTileInfo::GetExtraConstraintInfo() const
{
  return extra_tile_constraint_;
}

// void Set(problem::OperationSpace mold, problem::DataSpaceID dspace_id)
// {
//   operation_space_mold_ = mold;
//   dspace_id_ = dspace_id;
//   mold_set_ = true;
// }

}
