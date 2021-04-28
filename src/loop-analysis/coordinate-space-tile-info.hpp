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

#pragma once
#include "mapping/loop.hpp"

namespace tiling {

  // interface object between sparse modeling module and density models
  // tells the density models which set of tiles are we looking at
  struct CoordinateSpaceTileInfo {

    // an operation space mold for coordinate space representation needed for more precise representation
	// problem::Operation operation_space_mold_;
    std::vector <loop::Descriptor> subnests_;
    std::uint64_t shape_;
    problem::Shape::DataSpaceID dspace_id_;

    // for compatibility (there are code segments that do not set the mold yet)
    // TODO: maks all usage of coord space tile include mold and remove the check
    bool mold_set_ = false;

    void Clear() {
      shape_ = 0;
      subnests_ = {};
    }

    void Set(std::uint64_t shape, problem::Shape::DataSpaceID data_space_id) {
      shape_ = shape;
      dspace_id_ = data_space_id;
    }

    void Set(std::uint64_t shape, std::vector <loop::Descriptor> subnests, problem::Shape::DataSpaceID data_space_id) {
      shape_ = shape;
      subnests_ = subnests;
      dspace_id_ = data_space_id;
    }

    std::uint64_t GetShape() const {
       return shape_;
    }

    std::vector <loop::Descriptor> GetSubnests() const { return subnests_; }

  };

 // void Set(problem::OperationSpace mold, problem::DataSpaceID dspace_id)
 // {
 //   operation_space_mold_ = mold;
 //   dspace_id_ = dspace_id;
 //   mold_set_ = true;
 // }

}