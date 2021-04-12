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

#include "metadata-format.hpp"
#include "uncompressed-offset-pair.hpp"
#include "run-length-encoding.hpp"
#include "coordinate-payload.hpp"
#include "uncompressed-bitmask.hpp"
#include "compound-config/compound-config.hpp"


namespace problem
{

// list of lower-case format names
// std::vector<std::string> rank_formats = {"u", "b", "rle", "cp", "uop", "ub"};

//-------------------------------------------------//
//               MetaData Format Factory          //
//-------------------------------------------------//

class MetaDataFormatFactory{

public:
  static std::shared_ptr<MetaDataFormatSpecs> ParseSpecs(config::CompoundConfigNode metadata_rank_config){

    std::shared_ptr<MetaDataFormatSpecs> specs_ptr;
    std::string metadata_format_name = "None";

    if (metadata_rank_config.lookupValue("format", metadata_format_name))
    {
      if (metadata_format_name == "uop" || metadata_format_name == "UOP")
      {
        auto uop_specs = UncompressedOffsetPair::ParseSpecs(metadata_rank_config);
        specs_ptr = std::make_shared<UncompressedOffsetPair::Specs>(uop_specs);
      }
      else if (metadata_format_name == "rle" || metadata_format_name == "RLE")
      {
        auto rle_specs = RunLengthEncoding::ParseSpecs(metadata_rank_config);
        specs_ptr = std::make_shared<RunLengthEncoding::Specs>(rle_specs);
      }
      else if (metadata_format_name == "cp" || metadata_format_name == "CP")
      {
        auto cp_specs = CoordinatePayload::ParseSpecs(metadata_rank_config);
        specs_ptr = std::make_shared<CoordinatePayload::Specs>(cp_specs);
      }
      else if (metadata_format_name == "ub" || metadata_format_name == "UB")
      {
        auto ub_specs = UncompressedBitmask::ParseSpecs(metadata_rank_config);
        specs_ptr = std::make_shared<UncompressedBitmask::Specs>(ub_specs);
      }
      else
      {
        std::cerr << "ERROR: unrecognized metadata format name: " << metadata_format_name<< std::endl;
        exit(1);
      }
    }

    return specs_ptr;
  }

  static std::shared_ptr<MetaDataFormat> Construct(std::shared_ptr<MetaDataFormatSpecs> specs){

    std::shared_ptr<MetaDataFormat> metadata_format_ptr;

    if (specs->Name() == "uop")
    {
      auto specs_ptr = *std::static_pointer_cast<UncompressedOffsetPair::Specs>(specs);
      auto uop_metadata_ptr = std::make_shared<UncompressedOffsetPair>(specs_ptr);
      metadata_format_ptr = std::static_pointer_cast<MetaDataFormat>(uop_metadata_ptr);
    }
    else if (specs->Name() == "rle")
    {
      auto specs_ptr = *std::static_pointer_cast<RunLengthEncoding::Specs>(specs);
      auto rle_metadata_ptr = std::make_shared<RunLengthEncoding>(specs_ptr);
      metadata_format_ptr = std::static_pointer_cast<MetaDataFormat>(rle_metadata_ptr);
    }
    else if (specs->Name() == "cp")
    {
      auto specs_ptr = *std::static_pointer_cast<CoordinatePayload::Specs>(specs);
      auto cp_metadata_ptr = std::make_shared<CoordinatePayload>(specs_ptr);
      metadata_format_ptr = std::static_pointer_cast<MetaDataFormat>(cp_metadata_ptr);
    }
    else if (specs->Name() == "ub")
    {
      auto specs_ptr = *std::static_pointer_cast<UncompressedBitmask::Specs>(specs);
      auto ub_metadata_ptr = std::make_shared<UncompressedBitmask>(specs_ptr);
      metadata_format_ptr = std::static_pointer_cast<MetaDataFormat>(ub_metadata_ptr);
    }
    else
    {
      std::cerr << "ERROR: unrecognized metadata format name: " << specs->Name() << std::endl;
      exit(1);
    }

    return metadata_format_ptr;
  }


};




} // namespace
