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

namespace problem
{

// notes for future improvements
// this class should be more general, statistical modeling focused, and directly produces the following
//    (1) largest density of the tile (for the occupancy check and slowest compute unit)
//    (2) smallest density of the tile
//    (3) average density of the tile
// the interface should involve coordinate dependent setup, i.e., where is the tile in the workload
// the other parts of the infrastructure should be independent from the distribution format


//
// Constant
//
// all the tiles in the workload tensor will have the same constant density as the workload tensor

struct ConstantDensity{

    std::string type = "constant";
    double constant_density = 1.0;

    ConstantDensity(double density = 1.0){
       constant_density = density;
    }

    double GetTileDensity(std::uint64_t tile_shape, double confidence = 0.99){
       // constant density always return the same constant value
       (void) tile_shape;
       (void) confidence;
       return constant_density;
    }
};

//
// Coordinate uniform average
//
// have a certain workload, and know the amount of involved density, but has no distribution information other than
// the zeros are randomly distributed across the coordinates, i.e., for the same tile size, the sampled tiles will have
// similar property

struct CoordinateUniformAverage{

    std::string type = "coordinate_uniform_average";
    double average_density_ = 1.0;
    std::uint64_t workload_tensor_size_ = 0;
    bool workload_tensor_size_set_ = false;

    CoordinateUniformAverage(double density = 1.0){
      // if workload tensor is not available at the parsing stage, set the average density fist
      average_density_ = density;
    }

    CoordinateUniformAverage(std::uint64_t size, double density = 1.0){
      // constructor that sets both parameters at the same time
      workload_tensor_size_set_ = true;
      workload_tensor_size_ = size;
      average_density_ = density;
    }

    void SetWorkloadTensorSize(std::uint64_t size){
      // setter that allows workload tensor size at a latter stage (topology.cpp, PreEvaluationCheck)
      assert(! workload_tensor_size_set_);
      workload_tensor_size_set_ = true;
      workload_tensor_size_ = size;
    }

    double GetTileDensity(std::uint64_t tile_shape, double confidence = 0.99){

        if (average_density_ == 1.0){
          // dense data does not need distribution
          return 1.0;
        }

        // check necessary parameter is set
        assert(workload_tensor_size_set_);

        // construct a hypergeometric distribution according to
        //    1) N: population of objects -> workload size
        //    2) n: sample size -> tile size
        //    3) r: defective objects -> total number of zero data

        std::uint64_t r = workload_tensor_size_ * average_density_;
        std::uint64_t n = tile_shape;
        std::uint64_t N = workload_tensor_size_;
        boost::math::hypergeometric_distribution<double> distribution(r, n, N);
        std::uint64_t k_percentile = quantile(distribution, confidence);
        double density = 1.0*k_percentile/tile_shape;

        return density;
    }
};


class DataDensity{

  protected:

  bool is_typed_ = false;
  bool distribution_instantiated_= false;
  std::string type_ = "constant";

  // place holder distributions, one of them will be defined
  ConstantDensity constant_distribution;
  CoordinateUniformAverage hypergeometric_distribution;

  public:

  // constructor
  DataDensity(std::string density_type = "constant"){
    is_typed_ = true;
    type_ = density_type;
  }

  bool IsTyped() const { return is_typed_; }

  void SetDensity( double density){

    assert(is_typed_);

    if (type_ == "constant"){
       ConstantDensity distribution(density);
       constant_distribution = distribution;
    } else if (type_ == "coordinate_uniform_average"){
       CoordinateUniformAverage distribution(density);
       hypergeometric_distribution = distribution;
    } else {
       assert(false);
    }

    // a distribution is instantiated
    distribution_instantiated_ = true;

  }


  void SetWorkloadTensorSize( std::uint64_t size){

    assert(is_typed_);
    assert(distribution_instantiated_);
    if (type_ == "coordinate_uniform_average"){
       hypergeometric_distribution.SetWorkloadTensorSize(size);
    }

  }

  std::string GetType(){
     assert(is_typed_);
     return type_;
  }

  double GetTileDensity(std::uint64_t tile_shape, double confidence = 0.99){
    assert(is_typed_);

    double density;
    if (type_ == "constant"){
      density = constant_distribution.GetTileDensity(tile_shape, confidence);
    } else if (type_ == "coordinate_uniform_average"){
      density = hypergeometric_distribution.GetTileDensity(tile_shape, confidence);
    } else {
      assert(false);
    }
    return density;
  }

  // Serialization.
  friend class boost::serialization::access;
  template <class Archive>
  void serialize(Archive& ar, const unsigned int version=0){
    if (version == 0)
     { 
       ar& BOOST_SERIALIZATION_NVP(type_);
     }
  }
};
} // namespace