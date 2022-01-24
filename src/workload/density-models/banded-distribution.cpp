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

#include "workload/density-models/banded-distribution.hpp"
#include <mutex>

#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>

BOOST_CLASS_EXPORT(problem::BandedDistribution)

namespace problem{

BandedDistribution::BandedDistribution(){ }

BandedDistribution::BandedDistribution(const Specs& specs) : specs_(specs) { is_specced_ = true; }

BandedDistribution::~BandedDistribution() { }

BandedDistribution::Specs BandedDistribution::ParseSpecs(config::CompoundConfigNode density_config)
{
  Specs specs;
  density_config.lookupValue("distribution", specs.type);

  specs.band_width = 0; // a plain diagonal matrix has bandwidth of zero
  density_config.lookupValue("band_width", specs.band_width);

  specs.type = "banded-" +  std::to_string(specs.band_width);

  return specs;
}

std::mutex mtx;

void BandedDistribution::SetWorkloadTensorSize(const PointSet& point_set)
{
  if (workload_tensor_set_) return;

  
  // do not support multi-AAHR cases yet
  assert(point_set.numAAHRs() == 1);
  
  // retrieve the first(and only) aahr
  auto workload_aahr = point_set.GetAAHRs()[0];
  
  // only support 2D matrix
  assert(workload_aahr.Max().Order()==2 && workload_aahr.Min().Order()==2);
  
  mtx.lock();
  
  // record workload information per dimension and generate global density level information
  int min_dim_bound = workload_aahr.Max()[0];
  for (unsigned i= 0; i < workload_aahr.Max().Order(); i++)
  {
    workload_dim_bounds_.emplace_back(workload_aahr.Max()[i]);
    assert(min_dim_bound == workload_aahr.Max()[i]); // square matrix only for now
    min_dim_bound = min_dim_bound > workload_aahr.Max()[i] ? workload_aahr.Max()[i] : min_dim_bound;
  }
 
  // total number of nonzeros == the sum of the number of elements in the diagonals
  std::uint64_t total_nnzs = min_dim_bound;
  for (std::uint64_t ud = 1; ud <= specs_.band_width; ud++)
    total_nnzs += (min_dim_bound - ud) * 2;
  
  workload_tensor_size_ = workload_aahr.size(); 
  global_density_level_ = (double)total_nnzs/workload_tensor_size_;
  workload_tensor_set_ = true;
  
  mtx.unlock();
  // std::cout << "total nnzs: " << total_nnzs << "  density: " << global_density_level_ << std::endl;
}

std::uint64_t BandedDistribution::GetWorkloadTensorSize() const { return workload_tensor_size_; }

std::string BandedDistribution::GetDistributionType() const { return specs_.type; }


std::uint64_t BandedDistribution::ComputeNNZForSpecificTile(const PointSet& point_set) const
{
  // do not support multi-AAHR cases yet
  assert(point_set.numAAHRs() == 1);
  
  // retrieve the first(and only) aahr
  auto tile_aahr = point_set.GetAAHRs()[0];
  
  std:: uint64_t nnz = 0; 
  for (int r = tile_aahr.Min()[0]; r < tile_aahr.Max()[0]; r++)
  {
    int min_tile_c_bound = tile_aahr.Min()[1];     // dim_offsets[1];
    int c_start = std::max(min_tile_c_bound, std::max(r - int(specs_.band_width), 0));
    int max_tile_c_bound = tile_aahr.Max()[1]; // upper bound exclusive
    int c_end = std::min(max_tile_c_bound,  r + int(specs_.band_width + 1)); // exculsive end point
    
    // sanity check of the nz points in tile
    // for (int c = c_start; c < c_end; c++) 
    // {
    //   std::cout << "point (" << r << ", " << c << ")";
    //   if (abs(int(r)-int(c)) <= specs_.band_width) 
    //   {
    //     std::cout  << " nonzero" << std::endl;
    //   }
    //   else
    //   {
    //     std::cout << " zero " << std::endl;
    //   }
    // }
    nnz += c_start < c_end ? c_end - c_start : 0;  
  }
  //  std::cout << "tile nnz: " << nnz << std::endl;
  return nnz;
}


double BandedDistribution::GetZeroOccupancyProbForConstrainedTileMold(const PointSet& point_set_mold,
                                                                      const PointSet& constraint_point_set_mold) 
{

  if (constraint_point_set_mold.size() == 0) return 1.0;
 
  // std::cout << "point set:  " << point_set_mold 
  // << "   constriant point set: "
  // << constraint_point_set_mold << std::endl;

  // do not support multi-AAHR cases yet
  assert(point_set_mold.numAAHRs() == 1);
  assert(constraint_point_set_mold.numAAHRs() == 1);
  
  // retrieve the first(and only) aahr
  auto tile_aahr_mold = point_set_mold.GetAAHRs()[0];
  auto constraint_aahr_mold = constraint_point_set_mold.GetAAHRs()[0]; 
   
  for (unsigned i = 0; i < tile_aahr_mold.Max().Order(); i++) 
    assert(workload_dim_bounds_[i] % tile_aahr_mold.Max()[i] == 0);
 
  int constraint_col_bound = constraint_aahr_mold.Max()[1] - constraint_aahr_mold.Min()[1];
  int constraint_row_bound = constraint_aahr_mold.Max()[0] - constraint_aahr_mold.Min()[0];

  int num_col_groups = constraint_col_bound / tile_aahr_mold.Max()[1] ;
  int num_row_groups = constraint_row_bound / tile_aahr_mold.Max()[0] ;
 
  std::vector<int> tile_delta = {0, 0};
  tile_delta[0] = tile_aahr_mold.Max()[0] - tile_aahr_mold.Min()[0];
  tile_delta[1] = tile_aahr_mold.Max()[1] - tile_aahr_mold.Min()[0];

  int row_offset, cur_r_group;
  row_offset = constraint_aahr_mold.Min()[0]; 
  cur_r_group = 0;

  std::uint64_t count = 0;
  
  while(cur_r_group < num_row_groups)
  {
    int col_offset = constraint_aahr_mold.Min()[1];
    for (int cur_c_group = 0; cur_c_group < num_col_groups; cur_c_group++)
    {
      std::vector<Coordinate> min_coordinates = {constraint_aahr_mold.Min()[0] + row_offset, constraint_aahr_mold.Min()[1] + col_offset};
      std::vector<Coordinate> max_coordinates = {tile_delta[0] + row_offset, tile_delta[1] + col_offset};
      PointSet point_set(tile_aahr_mold.Min().Order(), Point(min_coordinates), Point(max_coordinates));
      auto occupancy = ComputeNNZForSpecificTile(point_set);
      if (occupancy == 0) count += 1;      
      
      col_offset += tile_aahr_mold.Max()[1];
    }
    // reset to first column
    col_offset = constraint_aahr_mold.Min()[1];
    row_offset += tile_aahr_mold.Max()[0];
    cur_r_group += 1;
  }
 
  double prob = (double)count/(num_row_groups * num_col_groups);
  // std::cout << "  zero prob: " << prob << std::endl;

  return prob; 
}


std::map<std::uint64_t, double> 
BandedDistribution::GetProbabilityDistributionForTileMold(const PointSet& point_set_mold,
                                                          const bool zero_occupancy_only) const
{
  
  // do not support multi-AAHR cases yet
  assert(point_set_mold.numAAHRs() == 1);
  auto tile_aahr_mold = point_set_mold.GetAAHRs()[0];
  
  // for (unsigned i = 0; i < tile_aahr_mold.Max().Order(); i++) 
  //   assert(workload_dim_bounds_[i] % tile_aahr_mold.Max()[i] == 0);
  
  int num_col_groups = workload_dim_bounds_[1] / tile_aahr_mold.Max()[1] ;
  int num_row_groups = workload_dim_bounds_[0] / tile_aahr_mold.Max()[0] ;
  
  int num_warm_up_row_groups = ceil((double)specs_.band_width/tile_aahr_mold.Max()[0]);
  int num_steady_state_groups = std::max(int(num_row_groups) - 2*int(num_warm_up_row_groups), 0);
  

  int row_offset, cur_r_group;
  row_offset = 0; 
  cur_r_group = 0;

  std::map<std::uint64_t, std::uint64_t> occupancy_count;
  occupancy_count[0] = 0;

  while(cur_r_group < num_warm_up_row_groups + 1 && cur_r_group < num_row_groups)
  {
    int col_offset = 0;
    for (int cur_c_group = 0; cur_c_group < num_col_groups;cur_c_group++)
    {
      if (zero_occupancy_only || col_offset > row_offset + tile_aahr_mold.Max()[0] - 1 + int(specs_.band_width))
      {
        
        if (cur_r_group >= num_warm_up_row_groups)
        {
          // all steay state groups have the same number of empty tiles
          occupancy_count[0] += (num_col_groups - cur_c_group) * num_steady_state_groups;
        } 
        else
        {
          // warm up state is symmetric with cool down state -> 2x
          // if there is no cool down state, then 1x
          occupancy_count[0] += num_warm_up_row_groups != num_row_groups ? 
            (num_col_groups - cur_c_group) * 2 : num_col_groups - cur_c_group;
        }
        
        break;
      }
      else
      {
        // construct exact point set
        // we have not yet reached steady state yet, so it is not possible for a tile to be empty 
        std::vector<Coordinate> min_coordinates = {tile_aahr_mold.Min()[0] + row_offset, tile_aahr_mold.Min()[1] + col_offset};
        std::vector<Coordinate> max_coordinates = {tile_aahr_mold.Max()[0] + row_offset, tile_aahr_mold.Max()[1] + col_offset};
        PointSet point_set(tile_aahr_mold.Min().Order(), Point(min_coordinates), Point(max_coordinates));
        
        auto occupancy = ComputeNNZForSpecificTile(point_set);
        if (occupancy_count.find(occupancy) == occupancy_count.end()) 
        {
          occupancy_count[occupancy] = 0;
        }
        if (cur_r_group >= num_warm_up_row_groups)
        {
          occupancy_count[occupancy] += num_steady_state_groups;
        } 
        else
        {
          occupancy_count[occupancy] += num_warm_up_row_groups != num_row_groups ?  2 : 1;
        }
      }
      col_offset += tile_aahr_mold.Max()[1];
    }
    // reset to first column
    col_offset = 0;
    row_offset += tile_aahr_mold.Max()[0];
    cur_r_group += 1; 
  }
  std::map<std::uint64_t, double> probability_lookup = {};
  for (auto iter = occupancy_count.begin(); iter != occupancy_count.end(); iter++)
  {
    double prob = (double)iter->second/(num_row_groups * num_col_groups);
   probability_lookup[iter->first] = prob;
  }
  //
  // Sanity check for occupancy calaculation
  //
 
  // if (!zero_occupancy_only)
  // {
  //   std::cout << "point set mold occupancy eval: "
  //   << tile_aahr_mold << std::endl;

  //   std::cout << "num col groups: " << num_col_groups 
  //   << "  num row groups: " << num_row_groups 
  //   << "  num warm up row groups: " <<  num_warm_up_row_groups 
  //   << "  num steady state row groups: " << num_steady_state_groups
  //   << std::endl;
  //   
  //   for (auto iter = occupancy_count.begin(); iter != occupancy_count.end(); iter++)
  //   {
  //     // std::cout <<"occu: " << iter->first  << " count: " << iter->second << " prob: " << prob << std::endl;
  //   }
  // }
  
  return probability_lookup;
 }

std::uint64_t BandedDistribution::GetMaxTileOccupancyByConfidence_LTW (const std::uint64_t tile_shape,
                                                                       const double confidence) 
{
  // the buffer should at least hold a one row density (best case possible)
  std::uint64_t nnzs = tile_shape > (specs_.band_width * 2 + 1) ? specs_.band_width * 2 + 1 : tile_shape;
  std::uint64_t max_occupancy = ceil(nnzs * confidence);
  return max_occupancy;
}

std::uint64_t BandedDistribution::GetMaxTileOccupancyByConfidence(const tiling::CoordinateSpaceTileInfo& tile,
                                                                  const double confidence)
{
  // shortcuts
  if (tile.GetShape() == 0)
    return 0;
  
  // more irregular tiles (check if we can use saved lookup table)
  auto tile_point_set = tile.GetPointSetRepr();
  auto probability_lookup = GetProbabilityDistributionForTileMold(tile_point_set);
 
  double accumulated_confidence = 1.0;
  auto iter = probability_lookup.rbegin();
  while (iter != probability_lookup.rend() && accumulated_confidence > confidence)
  {
    iter++;
    accumulated_confidence -= iter->second;
  }
  std::uint64_t max_occupancy = iter->first;  
  
  return max_occupancy;
}


std::uint64_t BandedDistribution::GetMaxNumElementByConfidence(const tiling::CoordinateSpaceTileInfo& fiber_tile,
                                                               const tiling::CoordinateSpaceTileInfo& element_tile,
                                                               const double confidence)
{
  
  assert( fiber_tile.GetShape() % element_tile.GetShape() == 0);
  
  std::uint64_t max_occupancy = GetMaxTileOccupancyByConfidence(fiber_tile, confidence);
  tiling::ExtraTileConstraintInfo new_constraint;
  new_constraint.Set(fiber_tile.GetShape(), max_occupancy);
  new_constraint.SetMold(fiber_tile.GetPointSetRepr());
  
  auto new_element_tile = element_tile;
  new_element_tile.extra_tile_constraint_= new_constraint;

  double prob_empty_element = GetTileOccupancyProbability(new_element_tile, 0);
  std::uint64_t max_num_element = ceil(fiber_tile.GetShape()/element_tile.GetShape() * prob_empty_element); 
  
  return max_num_element;
}


double BandedDistribution::GetMaxTileDensityByConfidence(const tiling::CoordinateSpaceTileInfo tile,
                                                         const double confidence) 
{
  std::uint64_t max_occupancy = GetMaxTileOccupancyByConfidence(tile, confidence);
  return (double)max_occupancy/tile.GetShape();
}

double BandedDistribution::GetMinTileDensity(const tiling::CoordinateSpaceTileInfo tile) 
{
  // shortcuts
  if (tile.GetShape() == 0)
    return 0;

  // more irregular tiles (check if we can use saved lookup table)
  auto tile_point_set = tile.GetPointSetRepr();
  auto probability_lookup = GetProbabilityDistributionForTileMold(tile_point_set);
  std::uint64_t min_occupancy = probability_lookup.begin()->first;
  return (double)min_occupancy/tile.GetShape();
}


PointSet BandedDistribution::GetReprPointSetForTileOccupancy(const PointSet& point_set_mold,
                                                             const std::uint64_t occupancy)
{
 
  // do not support multi-AAHR cases yet
  assert(point_set_mold.numAAHRs() == 1);
  auto tile_aahr_mold = point_set_mold.GetAAHRs()[0];
  
  for (unsigned i = 0; i < tile_aahr_mold.Max().Order(); i++) 
    assert(workload_dim_bounds_[i] % tile_aahr_mold.Max()[i] == 0);
  
  int num_col_groups = workload_dim_bounds_[1] / tile_aahr_mold.Max()[1] ;
  int num_row_groups = workload_dim_bounds_[0] / tile_aahr_mold.Max()[0] ;
  
  int num_warm_up_row_groups = ceil((double)specs_.band_width/tile_aahr_mold.Max()[0]);

  int row_offset, cur_r_group;
  row_offset = 0; 
  cur_r_group = 0;

  while(cur_r_group < num_warm_up_row_groups + 1 && cur_r_group < num_row_groups)
  {
    int col_offset = 0;
    for (int cur_c_group = 0; cur_c_group < num_col_groups;cur_c_group++)
    {
      if (col_offset > row_offset + tile_aahr_mold.Max()[0] - 1 + int(specs_.band_width))
      {
        std::vector<Coordinate> min_coordinates = {tile_aahr_mold.Min()[0] + row_offset, tile_aahr_mold.Min()[1] + col_offset};
        std::vector<Coordinate> max_coordinates = {tile_aahr_mold.Max()[0] + row_offset, tile_aahr_mold.Max()[1] + col_offset};
        
        PointSet point_set(tile_aahr_mold.Min().Order(), Point(min_coordinates), Point(max_coordinates));
        if (occupancy == 0) return point_set;
        else break;
      }
      else
      {
        // construct exact point set
        // we have not yet reached steady state yet, so it is not possible for a tile to be empty 
        std::vector<Coordinate> min_coordinates = {tile_aahr_mold.Min()[0] + row_offset, tile_aahr_mold.Min()[1] + col_offset};
        std::vector<Coordinate> max_coordinates = {tile_aahr_mold.Max()[0] + row_offset, tile_aahr_mold.Max()[1] + col_offset};
        PointSet point_set(tile_aahr_mold.Min().Order(), Point(min_coordinates), Point(max_coordinates));
        
        std::uint64_t point_set_occupancy = ComputeNNZForSpecificTile(point_set);
        if (point_set_occupancy == occupancy) return point_set;
      }
      col_offset += tile_aahr_mold.Max()[1];
    }
    // reset to first column
    col_offset = 0;
    row_offset += tile_aahr_mold.Max()[0];
    cur_r_group += 1; 
  }

  std::cerr << "ERROR: cannot find representative occupancy: "<< occupancy << " for tile mold: " << point_set_mold << std::endl;
  assert(false); 
}

double BandedDistribution::GetTileOccupancyProbability(const tiling::CoordinateSpaceTileInfo& tile,
                                                       const std::uint64_t occupancy)
{
  std::uint64_t tile_shape = tile.GetShape();
  // shortcuts
  if (tile_shape == 0) return occupancy == 0 ? 1.0 : 0;
  
  // more irregular tile shapes
  auto tile_point_set = tile.GetPointSetRepr();
  double prob;
  if (tile.HasExtraConstraintInfo())
  {
    auto extra_constraint_info = tile.GetExtraConstraintInfo();
    PointSet constraint_point_set = GetReprPointSetForTileOccupancy(extra_constraint_info.GetPointSetMold(), 
                                                                    extra_constraint_info.GetOccupancy());
    //FIXME: although there would not be any calls that invovle nonzero occupancy inquery, we still need to be more general 
    assert(occupancy == 0); 
    prob = GetZeroOccupancyProbForConstrainedTileMold(tile_point_set, constraint_point_set);
  }
  else
  {
    std::map<std::uint64_t, double> probability_lookup = GetProbabilityDistributionForTileMold(tile_point_set);  
    if (probability_lookup.find(occupancy) != probability_lookup.end())
      prob = probability_lookup.at(occupancy);
    else
    {
      // if occupancy not recorded, then the probability is zero
      prob = 0;
    }
  }
   
  return prob;
}


double BandedDistribution::GetExpectedTileOccupancy (const tiling::CoordinateSpaceTileInfo tile)
{
  return tile.GetShape() * global_density_level_;
}


}
