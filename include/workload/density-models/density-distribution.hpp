/* Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#include <bitset>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/math/distributions/hypergeometric.hpp>
#include <iostream>
#include <exception>

#include "compound-config/compound-config.hpp"
#include "loop-analysis/coordinate-space-tile-info.hpp"

namespace problem
{

//-------------------------------------------------//
//         Density Distribution Specs              //
//-------------------------------------------------//

struct DensityDistributionSpecs
{
  virtual ~DensityDistributionSpecs();

  virtual std::shared_ptr<DensityDistributionSpecs> Clone() const = 0;

  virtual const std::string Type() const = 0;

  std::string type = "UNSET";

  // Serialization
  friend class boost::serialization::access;

  template <class Archive>
  void serialize(Archive& ar, const unsigned int version = 0)
  {
    if (version == 0)
    {
      ar& BOOST_SERIALIZATION_NVP(type);
    }
  }
};  // struct DensityDistributionSpecs

BOOST_SERIALIZATION_ASSUME_ABSTRACT(DensityDistributionSpecs)


//-------------------------------------------------//
//      Density Distribution (base class)          //
//-------------------------------------------------//

class DensityDistribution
{
public:
  // destructor
  virtual ~DensityDistribution();

  virtual void SetWorkloadTensorSize(const problem::DataSpace& point_set) = 0;

  virtual std::uint64_t GetWorkloadTensorSize() const = 0;
  virtual std::string GetDistributionType() const = 0;

  virtual std::uint64_t GetMaxTileOccupancyByConfidence (const tiling::CoordinateSpaceTileInfo& tile,
                                                         const double confidence = 1.0) = 0;
  // for lightweight pre-evaluation check
  virtual std::uint64_t GetMaxTileOccupancyByConfidence_LTW (const std::uint64_t tile_shape,
                                                             const double confidence = 1.0) = 0;
  virtual std::uint64_t GetMaxNumElementByConfidence(const tiling::CoordinateSpaceTileInfo& fiber_tile,
                                             const tiling::CoordinateSpaceTileInfo& element_tile,
                                             const double confidence = 1.0) = 0;
  virtual double GetMaxTileDensityByConfidence(const tiling::CoordinateSpaceTileInfo tile,
                                               const double confidence = 1.0) = 0;
  virtual double GetMinTileDensity(const tiling::CoordinateSpaceTileInfo tile) = 0;
  virtual double GetTileOccupancyProbability (const tiling::CoordinateSpaceTileInfo& tile,
                                              const std::uint64_t occupancy) = 0;
  virtual double GetExpectedTileOccupancy (const tiling::CoordinateSpaceTileInfo tile) = 0;
  virtual bool OccupancyMoldNeeded() = 0;
  virtual problem::DataSpace GetOccupancyMold(const std::uint64_t occupancy) const = 0;
  
  // Serialization.
  friend class boost::serialization::access;
  template <class Archive>
  void serialize(Archive& ar, const unsigned int version=0)
  {
    (void) ar;
    (void) version;
  }

}; // class DensityDistribution


// exception class to handle requests that cannot be answsered by density model  
class DensityModelIncapability: public std::exception
{
  const char* what() const throw()
  {
    return "Density model cannot anwser specific request";
  }

};


BOOST_SERIALIZATION_ASSUME_ABSTRACT(DensityDistribution)
} // namespace problem
