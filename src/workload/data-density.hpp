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
    double constant_density_ = 1.0;

    ConstantDensity(double density = 1.0){
       constant_density_ = density;
    }

    double GetTileConfidentDensity(std::uint64_t tile_shape, double confidence) const {
       // constant density always return the same constant value
       (void) tile_shape;
       (void) confidence;
       return constant_density_;
    }

    double GetTileExpectedDensity( uint64_t tile_shape ) const {

       (void) tile_shape;

       return constant_density_;
    }

    double GetTileConfidence(std::uint64_t tile_shape, std::uint64_t buffer_size) const {

        double confidence;
        if ( buffer_size >= tile_shape){
            confidence = 1.0; // for sure the tile will fit, no matter what the density is
        } else {

          // binary option -> fit or not fit
          if (tile_shape * constant_density_ > buffer_size){
            confidence = 0.0;
          } else {
            confidence = 1.0;
          }
        }

        return confidence;
    }

};

//
// Coordinate uniform average
//
// have a certain workload, and know the amount of involved density, but has no distribution information other than
// the zeros are randomly distributed across the coordinates, i.e., for the same tile size, the sampled tiles will have
// similar property

struct CoordinateUniform{

    std::string type = "coordinate_uniform";
    double average_density_;
    std::uint64_t workload_tensor_size_;
    bool workload_tensor_size_set_;

    CoordinateUniform(double density = 1.0){
      // if workload tensor is not available at the parsing stage, set the average density fist
      average_density_ = density;
      workload_tensor_size_set_ = false;
      workload_tensor_size_ = 0;
    }

    CoordinateUniform(std::uint64_t size, double density = 1.0){
      // constructor that sets both parameters at the same time
      workload_tensor_size_set_ = true;
      workload_tensor_size_ = size;
      average_density_ = density;
    }

    void SetWorkloadTensorSize(std::uint64_t size){
      // setter that allows workload tensor size at a latter stage (topology.cpp, PreEvaluationCheck)
      // assert(! workload_tensor_size_set_);
      workload_tensor_size_set_ = true;
      workload_tensor_size_ = size;
    }

    double GetTileConfidentDensity(std::uint64_t tile_shape, double confidence) const {

        double tile_density;

        if (average_density_ == 1.0){
          // dense data does not need distribution
          tile_density = 1.0;
        } else if (tile_shape == 0) {
          // zero sized tile does not make sense for distribution
          tile_density = average_density_;
        } else {

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

            tile_density = 1.0*k_percentile/tile_shape;

        }


//        std::cout << "average density: " << average_density_ << " tensor size: " << workload_tensor_size_ << " tile size: " << tile_shape
//                  << " --->" << confidence <<" confident density: " << tile_density << std::endl;

        return tile_density;
    }



    double GetTileExpectedDensity( uint64_t tile_shape ) const {

       (void) tile_shape;

       return average_density_;
    }

 double GetTileConfidence(std::uint64_t tile_shape, std::uint64_t buffer_size) const {

    // given a specific buffer size, return the confidence of tile shape with workload density fitting in
    // the buffer

    double confidence;

    if ( buffer_size >= tile_shape){

          confidence = 1.0; // for sure the tile will fit, no matter what the density is

    } else if (average_density_ == 1.0) {

      confidence = 0.0; // for sure will not fit

    } else {  // buffer size < tile shape, and sparse data

        // construct a hypergeometric distribution according to
        //    1) N: population of objects -> workload size
        //    2) n: sample size -> tile size
        //    3) r: defective objects -> total number of zero data

       std::uint64_t r = workload_tensor_size_ * average_density_;
       std::uint64_t n = tile_shape;
       std::uint64_t N = workload_tensor_size_;

       boost::math::hypergeometric_distribution<double> distribution(r, n, N);
       confidence = cdf(distribution, buffer_size);
    }

    return confidence;

    }
};


class DataDensity{

  protected:

  bool is_typed_ = false;
  bool distribution_instantiated_= false;
  std::string type_ = "constant";
  double confidence_knob_ = 0.5;

  // place holder distributions, one of them will be defined
  ConstantDensity constant_distribution;


  public:
  CoordinateUniform hypergeometric_distribution;

  // constructor
  DataDensity(std::string density_type = "constant", double knob = 1){
    is_typed_ = true;
    type_ = density_type;
    confidence_knob_ = knob;
  }

  bool IsTyped() const { return is_typed_; }

  void SetDensity( double density){

    assert(is_typed_);

    if (type_ == "constant"){
       ConstantDensity distribution(density);
       constant_distribution = distribution;

    } else if (type_ == "coordinate_uniform"){
       CoordinateUniform distribution(density);
       hypergeometric_distribution = distribution;

    } else {
       assert(false);
    }

    // a distribution is instantiated
    distribution_instantiated_ = true;

  }

  double GetTileConfidence(){
     return confidence_knob_;
  }

  void SetWorkloadTensorSize( std::uint64_t size){

    assert(is_typed_);
    assert(distribution_instantiated_);
    if (type_ == "coordinate_uniform"){
       hypergeometric_distribution.SetWorkloadTensorSize(size);
    } else {
       // no need to set workload tensor sizes
    }
  }

  std::uint64_t GetWorkloadTensorSize() const {

    assert(is_typed_);
    assert(distribution_instantiated_);
    if (type_ == "coordinate_uniform"){
      return hypergeometric_distribution.workload_tensor_size_;
    } else {
      std::cout << "WARN: workload tensor size is not used for this distribution, so not set" << std::endl;
      return 0;
    }
  }

  std::string GetType() const{
     assert(is_typed_);
     return type_;
  }

  double GetTileConfidentDensity(std::uint64_t tile_shape, double confidence = 0.5) const {
    assert(is_typed_);
    (void) confidence; // use the user-defined knob that can be defined outside

    double density;
    if (type_ == "constant"){
      density = constant_distribution.GetTileConfidentDensity(tile_shape, confidence_knob_);

    } else if (type_ == "coordinate_uniform"){
      density = hypergeometric_distribution.GetTileConfidentDensity(tile_shape, confidence_knob_);

    } else {
      assert(false);
    }
    return density;
  }

  double GetTileExpectedDensity(std::uint64_t tile_shape) const {
    double density;
    if (type_ == "constant"){
      density = constant_distribution.GetTileExpectedDensity(tile_shape);

    } else if (type_ == "coordinate_uniform"){
      density = hypergeometric_distribution.GetTileExpectedDensity(tile_shape);

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