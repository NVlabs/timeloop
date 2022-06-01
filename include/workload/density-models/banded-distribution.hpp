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

#include "density-distribution.hpp"
#include <boost/serialization/export.hpp>

namespace problem
{

class BandedDistribution : public DensityDistribution
{

 public:

  //
  // Specs
  //

  struct Specs : public DensityDistributionSpecs
  {

    std::string type;
    std::uint32_t band_width;

    const std::string Type() const override
    { return type; }

    // Serialization
    friend class boost::serialization::access;

    template<class Archive>
    void serialize(Archive& ar, const unsigned int version = 0)
    {
      ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(DensityDistributionSpecs);
      if (version == 0)
      {
        ar & BOOST_SERIALIZATION_NVP(type);
      }
    }

   public:
    std::shared_ptr <DensityDistributionSpecs> Clone() const override
    {
      return std::static_pointer_cast<DensityDistributionSpecs>(std::make_shared<Specs>(*this));
    }

  }; // struct Specs

  //
  // Data
  //

 private:
  Specs specs_;
  bool is_specced_;

  std::uint64_t workload_tensor_size_;
  std::vector<std::uint64_t> workload_dim_bounds_;
  double global_density_level_;
  bool workload_tensor_set_ = false;

  std::uint64_t ComputeNNZForSpecificTile(const PointSet& point_set) const;
  std::map<std::uint64_t, double> GetProbabilityDistributionForTileMold
      (const PointSet& point_set_mold, const bool zero_occupancy_only = false) const;
  double GetZeroOccupancyProbForConstrainedTileMold(const PointSet& point_set_mold,
                                                    const PointSet& constraint_point_mold);
  PointSet GetReprPointSetForTileOccupancy(const PointSet& piont_set_mold, const std::uint64_t occupancy);

 public:
  // Serialization
  friend class boost::serialization::access;

  template<class Archive>
  void serialize(Archive& ar, const unsigned int version = 0)
  {
    ar & BOOST_SERIALIZATION_BASE_OBJECT_NVP(DensityDistribution);
    if (version == 0)
    {
      ar & BOOST_SERIALIZATION_NVP(specs_);
    }
  }

  //
  // API
  //

  // constructor and destructors
  BandedDistribution();

  BandedDistribution(const Specs& specs);

  ~BandedDistribution();

  static Specs ParseSpecs(config::CompoundConfigNode density_config);

  void SetWorkloadTensorSize(const PointSet& point_set);

  std::uint64_t GetWorkloadTensorSize() const;
  std::string GetDistributionType() const;
  std::uint64_t GetMaxTileOccupancyByConfidence(const tiling::CoordinateSpaceTileInfo& tensor,
                                                const double confidence);
  std::uint64_t GetMaxTileOccupancyByConfidence_LTW(const std::uint64_t tile_shape,
                                                    const double confidence);
  std::uint64_t GetMaxNumElementByConfidence(const tiling::CoordinateSpaceTileInfo& fiber_tile,
                                             const tiling::CoordinateSpaceTileInfo& element_tile,
                                             const double confidence);
  double GetMaxTileDensityByConfidence(const tiling::CoordinateSpaceTileInfo tile,
                                       const double confidence = 1.0);
  double GetMinTileDensity(const tiling::CoordinateSpaceTileInfo tile);
  double GetTileOccupancyProbability(const tiling::CoordinateSpaceTileInfo& tile,
                                     const std::uint64_t occupancy);
  double GetExpectedTileOccupancy(const tiling::CoordinateSpaceTileInfo tile);
  


}; // class BandedDistribution

} // namespace problem
