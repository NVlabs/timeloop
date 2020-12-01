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
// Uniform tile density model
//
// all the tiles in the workload tensor will have the same fixed density as the workload tensor

struct Fixed{

    std::string type = "fixed";
    double fixed_density_ = 1.0;
    std::uint64_t workload_tensor_size_ = 0;

    Fixed(double density = 1.0){
       fixed_density_ = density;
    }

    void SetWorkloadTensorSize(std::uint64_t size){
      workload_tensor_size_ = size;
    }

    double GetProbability(std::uint64_t tile_shape, std::uint64_t nnz_vals) const {
      if (tile_shape == 1){
        // tile shape 1 is a special case, where we use a binomial representation to reflect the probability
        return nnz_vals == 0 ? 1-fixed_density_ : fixed_density_;
      } else if (ceil(tile_shape * fixed_density_) != nnz_vals){
        return 0.0;
      } else {
        // fixed distribution is not stochastic
        return 1.0;
      }
    }

    double GetTileDensityByConfidence(std::uint64_t tile_shape, double confidence, uint64_t buffer_size = 0) const {
       // fixed density always return the same fixed value
       (void) tile_shape;
       (void) confidence;
       (void) buffer_size;
       return fixed_density_;
    }

    std::uint64_t GetTileOccupancyByConfidence(std::uint64_t tile_shape, double confidence) const{

       if (confidence == 0){
         return 0;
       } else {
         return ceil(tile_shape * fixed_density_);
       }

    }

    double GetTileExpectedDensity( uint64_t tile_shape ) const {

       (void) tile_shape;

       return fixed_density_;
    }

    double GetTileConfidence(std::uint64_t tile_shape, std::uint64_t buffer_size) const {

        double confidence;
        if ( buffer_size >= tile_shape){
            confidence = 1.0; // for sure the tile will fit, no matter what the density is
        } else {

          // binary option -> fit or not fit
          if (tile_shape * fixed_density_ > buffer_size){
            confidence = 0.0;
          } else {
            confidence = 1.0;
          }
        }

        return confidence;
    }

};

//
// Hypergeometric
//
// have a certain workload, and know the amount of involved density, but has no distribution information other than
// the zeros are randomly distributed across the coordinates, i.e., for the same tile size, the sampled tiles will have
// similar property

struct Hypergeometric{

    std::string type = "hypergeometric";
    double average_density_;
    std::uint64_t workload_tensor_size_;
    bool workload_tensor_size_set_;

    Hypergeometric(double density = 1.0){
      // if workload tensor is not available at the parsing stage, set the average density fist
      average_density_ = density;
      workload_tensor_size_set_ = false;
      workload_tensor_size_ = 0;
    }

    Hypergeometric(std::uint64_t size, double density = 1.0){
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

    double GetProbability(std::uint64_t tile_shape, std::uint64_t nnz_vals) const{
        assert(workload_tensor_size_set_);
        std::uint64_t  r = ceil(workload_tensor_size_ * average_density_);
        std::uint64_t  n = tile_shape;
        std::uint64_t  N = workload_tensor_size_;

        if ( ((n + r > N) && (nnz_vals < n + r - N))| (nnz_vals > r)){return 0;}

        boost::math::hypergeometric_distribution<double> distribution(r, n, N);
        double prob = pdf(distribution, nnz_vals);

        return prob;
    }

    double GetTileDensityByConfidence(std::uint64_t tile_shape, double confidence, uint64_t allocated_capacity = 0) const {

        double tile_density;

        if (average_density_ == 1.0){
          // dense data does not need distribution
          tile_density = 1.0;

        } else if (tile_shape == 0) {
          // zero sized tile does not make sense for distribution
          tile_density = average_density_;

        } else if (confidence == 1.0){

          if (allocated_capacity <= tile_shape && allocated_capacity > workload_tensor_size_ * average_density_){
             tile_density = workload_tensor_size_ * average_density_/tile_shape;
          } else if (allocated_capacity <= tile_shape) {
             tile_density = double(allocated_capacity)/tile_shape;
          } else if (allocated_capacity > tile_shape && tile_shape > workload_tensor_size_ * average_density_){
             tile_density = workload_tensor_size_ * average_density_/tile_shape;
          } else if (allocated_capacity > tile_shape){
             tile_density = 1.0;
          } else {
             assert(false);
          }

        } else {

            // check necessary parameter is set
            // assert(workload_tensor_size_set_);

            // construct a hypergeometric distribution according to
            //    1) N: population of objects -> workload size
            //    2) n: sample size -> tile size
            //    3) r: defective objects -> total number of zero data

            // std::uint64_t r = workload_tensor_size_ * average_density_;
            // std::uint64_t n = tile_shape;
            // std::uint64_t N = workload_tensor_size_;
            // boost::math::hypergeometric_distribution<double> distribution(r, n, N);
            // std::uint64_t k_percentile = quantile(distribution, confidence);
            // std::cout << "k: " << k_percentile << std::endl;

            // we don't need to re-derive the percentile as above
            // for this case, we know that we are not confident that the entire tile will fit
            // even if we give the entire allocated_capacity to the tile
            // so the max density supported is just allocated_capacity/tile shape
            tile_density = 1.0 * allocated_capacity / tile_shape;
        }


        // std::cout << "average density: " << average_density_ << " tensor size: " << workload_tensor_size_ << " tile size: " << tile_shape
        // << " --->" << confidence <<" confident density: " << tile_density << std::endl;

        return tile_density;
    }


    std::uint64_t GetTileOccupancyByConfidence(std::uint64_t tile_shape, double confidence) const{

       std::uint64_t tile_occupancy;
       std::uint64_t total_nnzs = workload_tensor_size_ * average_density_;

       if (confidence == 1.0){
         if (tile_shape >= total_nnzs){
           tile_occupancy = total_nnzs;
         } else {
           tile_occupancy = tile_shape;
         }
       } else {
         std::uint64_t r = workload_tensor_size_ * average_density_;
         std::uint64_t n = tile_shape;
         std::uint64_t N = workload_tensor_size_;
         boost::math::hypergeometric_distribution<double> distribution(r, n, N);
         tile_occupancy = quantile(distribution, confidence);
       }

       return tile_occupancy;
    }



    double GetTileExpectedDensity( uint64_t tile_shape ) const {

       (void) tile_shape;

       return average_density_;
    }

 double GetTileConfidence(std::uint64_t tile_shape, std::uint64_t buffer_size) const {

    // given a specific buffer size, return the confidence of tile shape with workload density fitting in
    // the buffer

    float confidence;

    //std::cout << "total nonzeros: " << workload_tensor_size_ * average_density_ << std::endl;

    if ( buffer_size >= tile_shape){

      confidence = 1.0; // for sure the tile will fit, no matter what the density is

    } else if ( buffer_size >=  workload_tensor_size_ * average_density_){

      // can store more than total number of nonzeros, tile must fit, check on upper bound min(n,r)
      confidence = 1.0;

    } else if ( buffer_size < tile_shape + workload_tensor_size_ * average_density_ - workload_tensor_size_){

      // check on lower bound max(0, n+r-N)
      confidence = 0.0;

    } else if (average_density_ == 1.0) {
      // if not larger than tile shape or total number of nonzeros, and has a density of 1.0, for sure will not fit
      confidence = 0.0;

    } else {  // buffer size < tile shape, buffer size < total number of nonzeros and sparse data

        // construct a hypergeometric distribution according to
        //    1) N: population of objects -> workload size
        //    2) n: sample size -> tile size
        //    3) r: defective objects -> total number of zero data

       std::uint64_t r = workload_tensor_size_ * average_density_;
       std::uint64_t n = tile_shape;
       std::uint64_t N = workload_tensor_size_;
       // std::cout << "r: " << r << " n: " << tile_shape << " N: " << workload_tensor_size_ << " x: "<< buffer_size << std::endl;
       boost::math::hypergeometric_distribution<double> distribution(r, n, N);
       double cdf_val = cdf(distribution, 1.0 * buffer_size);
       confidence = cdf_val >= 0.9999 ? 1.0 :  cdf_val;
       // std::cout << "print version 1.0 buffer size: " << cdf(distribution, 1.0*buffer_size) << std::endl;
    }

    return confidence;

    }
};


class DataDensity{

  protected:

  bool is_typed_ = false;
  bool distribution_instantiated_= false;
  std::string type_ = "fixed";
  double confidence_constraint_ = 1.0;

  public:

  // place holder distributions, one of them will be defined
  Fixed fixed_distribution;
  Hypergeometric hypergeometric_distribution;

  // constructor
  DataDensity(std::string density_type = "fixed", double confidence_constraint = 1.0){
    // by default, tile density is set as fixed distribution and no constraint
    is_typed_ = true;
    type_ = density_type;
    confidence_constraint_ = confidence_constraint;
  }

  void SetConfidenceConstraint(double confidence_constraint){
     confidence_constraint_ = confidence_constraint;
  }

  bool IsTyped() const { return is_typed_; }

  void SetDensity( double density){

    assert(is_typed_);

    if (type_ == "fixed"){
      Fixed distribution(density);
      fixed_distribution = distribution;

    } else if (type_ == "hypergeometric"){
       Hypergeometric distribution(density);
       hypergeometric_distribution = distribution;

    } else {
       assert(false);
    }

    // a distribution is instantiated
    distribution_instantiated_ = true;

  }

  double GetConfidenceConstraint() const{
     return confidence_constraint_;
  }

  std::string GetDensityType() const {
     assert(is_typed_);
     return type_;
  }

  double GetTileConfidence(std::uint64_t tile_shape, std::uint64_t buffer_size) const{
    assert(is_typed_);
    assert(distribution_instantiated_);

    double confidence;

    if (type_ == "fixed"){
      confidence = fixed_distribution.GetTileConfidence(tile_shape, buffer_size);

    } else if (type_ == "hypergeometric"){
      confidence = hypergeometric_distribution.GetTileConfidence(tile_shape, buffer_size);

    } else {
      assert(false);
    }

    return confidence;

  }

  std::uint64_t GetTileOccupancyByConfidence (std::uint64_t tile_shape, double confidence){

    uint64_t tile_occupancy;

    if (type_ == "fixed"){
      tile_occupancy = fixed_distribution.GetTileOccupancyByConfidence(tile_shape, confidence);

    } else if (type_ == "hypergeometric"){
      tile_occupancy = hypergeometric_distribution.GetTileOccupancyByConfidence(tile_shape, confidence);

    } else {
      assert(false);
    }

    return tile_occupancy;
  }

  void SetWorkloadTensorSize( std::uint64_t size){

    assert(is_typed_);
    assert(distribution_instantiated_);
    if (type_ == "hypergeometric"){
       hypergeometric_distribution.SetWorkloadTensorSize(size);
    } else {
       fixed_distribution.SetWorkloadTensorSize(size);
    }
  }

  std::uint64_t GetWorkloadTensorSize() const {

    assert(is_typed_);
    assert(distribution_instantiated_);
    if (type_ == "hypergeometric"){
      return hypergeometric_distribution.workload_tensor_size_;
    } else {
      return fixed_distribution.workload_tensor_size_;
    }
  }

  std::string GetType() const{
     assert(is_typed_);
     return type_;
  }

  double GetTileDensityByConfidence(std::uint64_t tile_shape, double confidence, uint64_t buffer_size = 0) const {
    assert(is_typed_);

    double density;
    if (type_ == "fixed"){
      density = fixed_distribution.GetTileDensityByConfidence(tile_shape, confidence, buffer_size);

    } else if (type_ == "hypergeometric"){
      density = hypergeometric_distribution.GetTileDensityByConfidence(tile_shape, confidence, buffer_size);

    } else {
      assert(false);
    }
    return density;
  }


  double GetTileExpectedDensity(std::uint64_t tile_shape) const {
    double density;
    if (type_ == "fixed"){
      density = fixed_distribution.GetTileExpectedDensity(tile_shape);

    } else if (type_ == "hypergeometric"){
      density = hypergeometric_distribution.GetTileExpectedDensity(tile_shape);

    } else {
      assert(false);
    }
    return density;
  }


  double GetProbability(std::uint64_t tile_shape, std::uint64_t nnz_vals) const {

    double probability;

    if (type_ == "fixed"){
      probability = fixed_distribution.GetProbability(tile_shape, nnz_vals);

    } else if (type_ == "hypergeometric") {
      probability = hypergeometric_distribution.GetProbability(tile_shape, nnz_vals);

    } else {
      probability = 0;
      assert(false);
    }

    return probability;
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