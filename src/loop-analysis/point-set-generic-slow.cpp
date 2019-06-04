/* Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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

#include <vector>

#include "point-set.hpp"

// The following function expands two order-dimensional endpoints into a vector
// of order-dimensional points that fill out the axis-aligned hyper-rectangle
// between the endpoints. This is used by a constructor in the point set
// implementation below. However, implementing the logic as a recursive template
// causes GCC to hang for an unknown reason. Therefore, we use a different point
// implementation as an intermediate to construct this space.

std::vector<FlexPoint> Expand(const FlexPoint min, const FlexPoint max)
{
  // Min and max should have same order.
  assert(min.size() == max.size());
  auto order = min.size();
  
  std::vector<FlexPoint> retval;

  if (order == 1)
  {
    // Base case.
    for (Coordinate x = min[0]; x < max[0]; x++)
    {
      FlexPoint point(1);
      point[0] = x;
      retval.push_back(point);
    }
  }
  else
  {
    // Recursive case: iterate over the coordinate of the highest
    // dimension. At each step, create a lower-dimensional point
    // vector and extrude it into this high-dimensional space at the
    // high-dim coordinate.
    for (Coordinate x = min[order-1]; x < max[order-1]; x++)
    {
      // Low-dimension min and max.
      FlexPoint low_dim_min(order-1), low_dim_max(order-1);
      for (unsigned dim = 0; dim < order-1; dim++)
      {
        low_dim_min[dim] = min[dim];
        low_dim_max[dim] = max[dim];
      }

      // Create a new low-dimensional point vector.
      std::vector<FlexPoint> low_dim_points = Expand(low_dim_min, low_dim_max);
      
      // Extrude each point from this low-dim point-vector into the
      // highest dimension and add the higher-dim point into our
      // final point vector.
      for (auto& low_dim_point : low_dim_points)
      {
        // The high-dim point we will insert into this vector.
        FlexPoint point(order);

        // Copy over all low dimension coordinates.
        for (unsigned dim = 0; dim < order-1; dim++)
        {
          point[dim] = low_dim_point[dim];
        }
          
        // Extrude into the highest-order dimension.
        point[order-1] = x;
          
        // Insert the point into the vector.
        retval.push_back(point);
      }
    }
  }

  return retval;
}
