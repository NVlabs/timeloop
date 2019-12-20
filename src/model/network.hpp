/* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#include <iostream>

#include "model/util.hpp"
#include "model/level.hpp"
#include "pat/pat.hpp"

namespace model
{

//--------------------------------------------//
//               Network Specs                //
//--------------------------------------------//

struct NetworkSpecs
{
  virtual ~NetworkSpecs() { }

  virtual const std::string Type() const = 0;

  std::string name = "UNSET";

  // Serialization
  friend class boost::serialization::access;

  template <class Archive>
  void serialize(Archive& ar, const unsigned int version = 0)
  {
    if (version == 0)
    {
      ar& BOOST_SERIALIZATION_NVP(name);
    }
  }
}; // struct NetworkSpecs

BOOST_SERIALIZATION_ASSUME_ABSTRACT(NetworkSpecs)

//--------------------------------------------//
//            Network (base class)            //
//--------------------------------------------//

class Network : public Module
{
 public:
  virtual ~Network() { }

  virtual void ConnectSource(std::weak_ptr<Level> source) = 0;
  virtual void ConnectSink(std::weak_ptr<Level> sink) = 0;
  virtual void SetName(std::string name) = 0;

  // STAT_ACCESSOR_HEADER(virtual double, Energy) = 0;
  virtual double Energy(problem::Shape::DataSpaceID pv = problem::GetShape()->NumDataSpaces) const = 0;

  virtual std::string Name() const = 0;
  virtual bool DistributedMulticastSupported() const = 0;
  virtual EvalStatus Evaluate(const tiling::CompoundTile& tile,
                              const double inner_tile_area,
                              const bool break_on_failure,
                              const bool reduction = false) = 0;

  virtual void Print(std::ostream& out) const = 0;

  // Ugly abstraction-breaking probes that should be removed.
  virtual std::uint64_t WordBits() const = 0;

  // Serialization
  friend class boost::serialization::access;

  template <class Archive>
  void serialize(Archive& ar, const unsigned int version = 0)
  {
    (void) ar;
    (void) version;
  }

  friend std::ostream& operator << (std::ostream& out, const Network& network)
  {
    network.Print(out);
    return out;
  }

}; // class Network

BOOST_SERIALIZATION_ASSUME_ABSTRACT(NetworkSpecs)

} // namespace model
