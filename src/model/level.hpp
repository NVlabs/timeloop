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

#include "model/model-base.hpp"
#include "loop-analysis/tiling.hpp"

namespace model
{

struct EvalStatus
{
  bool success;
  std::string fail_reason;
};

//--------------------------------------------//
//                Level Specs                 //
//--------------------------------------------//

struct LevelSpecs
{
  virtual ~LevelSpecs() { }
  
  virtual const std::string Type() const = 0;

  std::string level_name;

  // Serialization
  friend class boost::serialization::access;

  template <class Archive>
  void serialize(Archive& ar, const unsigned int version = 0)
  {
    if (version == 0)
    {
      ar& BOOST_SERIALIZATION_NVP(level_name);
    }
  }
};

BOOST_SERIALIZATION_ASSUME_ABSTRACT(LevelSpecs)

//--------------------------------------------//
//                    Level                   //
//--------------------------------------------//

class Level : public Module
{
 public:
  virtual ~Level() { }

  virtual std::shared_ptr<Level> Clone() const = 0;

  virtual bool HardwareReductionSupported() = 0;

  virtual EvalStatus PreEvaluationCheck(const problem::PerDataSpace<std::size_t> working_set_sizes,
                                        const tiling::CompoundMask mask, const bool break_on_failure) = 0;
  virtual EvalStatus Evaluate(const tiling::CompoundTile& tile, const tiling::CompoundMask& mask,
                              const std::uint64_t compute_cycles,
                              const bool break_on_failure) = 0;
  
  virtual double Energy(problem::Shape::DataSpaceID pv = problem::GetShape()->NumDataSpaces) const = 0;
  
  virtual std::string Name() const = 0;
  virtual double Area() const = 0;
  virtual double AreaPerInstance() const = 0;
  virtual std::uint64_t Cycles() const = 0;
  virtual std::uint64_t Accesses(problem::Shape::DataSpaceID pv = problem::GetShape()->NumDataSpaces) const = 0;
  virtual double CapacityUtilization() const = 0;
  virtual std::uint64_t UtilizedCapacity(problem::Shape::DataSpaceID pv = problem::GetShape()->NumDataSpaces) const = 0;
  virtual std::uint64_t UtilizedInstances(problem::Shape::DataSpaceID pv = problem::GetShape()->NumDataSpaces) const = 0;
  
  virtual void Print(std::ostream& out) const = 0;

  friend std::ostream& operator << (std::ostream& out, const Level& level)
  {
    level.Print(out);
    return out;
  }
  
  // Serialization
  friend class boost::serialization::access;

  template <class Archive>
  void serialize(Archive& ar, const unsigned int version = 0)
  {
    (void) ar;
    (void) version;
  } 
};

BOOST_SERIALIZATION_ASSUME_ABSTRACT(Level)

} // namespace model
