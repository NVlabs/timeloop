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

#include "metadata-format.hpp"
#include <boost/serialization/export.hpp>

namespace problem {

class UncompressedOffsetPair : public MetaDataFormat {

public:

  //
  // Specs
  //

  struct Specs : public MetaDataFormatSpecs {

    std::string name = "uop";  // uncompressed offset pair
    bool rank_compressed = false;
    bool coordinates_implicit = true;
    std::vector<problem::Shape::FlattenedDimensionID> dimension_ids;
    std::uint32_t metadata_word_bits;
    std::uint32_t payload_word_bits;

    const std::string Name() const override { return name; }
    bool RankCompressed() const override {return rank_compressed;}
    std::vector<problem::Shape::FlattenedDimensionID> DimensionIDs() const override {return dimension_ids;}
    std::uint32_t MetaDataWordBits() const override { return metadata_word_bits; }
    std::uint32_t PayloadWordBits() const override { return payload_word_bits; }
    void SetMetaDataWordBits(std::uint32_t word_bits) override { metadata_word_bits = word_bits; }
    void SetPayloadWordBits(std::uint32_t word_bits) override { payload_word_bits = word_bits; }

    // Serialization
    friend class boost::serialization::access;

    template<class Archive>
    void serialize(Archive &ar, const unsigned int version = 0) {

      ar& BOOST_SERIALIZATION_BASE_OBJECT_NVP(MetaDataFormatSpecs);
      if (version == 0) {
        ar& BOOST_SERIALIZATION_NVP(name);
        ar& BOOST_SERIALIZATION_NVP(dimension_ids);
      }
    }

  public:
    std::shared_ptr<MetaDataFormatSpecs> Clone() const override
    {
      return std::static_pointer_cast<MetaDataFormatSpecs>(std::make_shared<Specs>(*this));
    }

  }; // struct Specs

//
// Data
//

private:
  Specs specs_;
  bool is_specced_;


public:
  // Serialization
  friend class boost::serialization::access;

  template<class Archive>
  void serialize(Archive &ar, const unsigned int version = 0) {
    ar& BOOST_SERIALIZATION_BASE_OBJECT_NVP(MetaDataFormat);
    if (version == 0) {
      ar & BOOST_SERIALIZATION_NVP(specs_);
    }
  }

  //
  // API
  //

  // constructor and destructors
  UncompressedOffsetPair();
  UncompressedOffsetPair(const Specs &specs);
  ~UncompressedOffsetPair();

  static Specs ParseSpecs(config::CompoundConfigNode metadata_config);

  PerRankMetaDataTileOccupancy GetOccupancy(const MetaDataOccupancyQuery& query) const;
  bool RankCompressed () const;
  bool CoordinatesImplicit() const;
  std::vector<problem::Shape::FlattenedDimensionID> GetDimensionIDs() const;
  std::string GetFormatName() const;
  bool MetaDataImplicitAsLowestRank() const {return false;}
  const MetaDataFormatSpecs& GetSpecs() const; 


}; // class UncompressedOffsetPair

} // namespace problem
