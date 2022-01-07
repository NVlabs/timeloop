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

#include <cassert>

#include "loop-analysis/aahr-carve.hpp"

#define ASSERT(args...) assert(args)

void Increment(Point& point, unsigned rank, const Point& shape)
{
  ASSERT(point.Order() <= shape.Order());
  ASSERT(rank < point.Order());

  for (unsigned r = rank; r < point.Order(); r++)
  {
    if (r == point.Order()-1)
    {
      // Outermost rank, we cannot wrap this.
      ASSERT(point[r] < shape[r]-1);
      point[r] = point[r] + 1;
    }
    else
    {
      point[r] = (point[r] + 1) % shape[r];
      if (point[r] != 0)
        break;
    }
  }
}

void Decrement(Point& point, unsigned rank, const Point& shape)
{
  ASSERT(point.Order() <= shape.Order());
  ASSERT(rank < point.Order());

  for (unsigned r = rank; r < point.Order(); r++)
  {
    if (r == point.Order()-1)
    {
      // Outermost rank, we cannot wrap this.
      ASSERT(point[r] >= 1);
      point[r] = point[r] - 1;
    }
    else
    {
      point[r] = (point[r] + shape[r] - 1) % shape[r];
      if (point[r] != shape[r]-1)
        break;
    }
  }
}

std::vector<std::pair<Point, Point>> Carve(Point base, Point bound, const Point& shape)
{
  // Base and bound must have the same number of ranks.
  ASSERT(base.Order() == bound.Order());
  ASSERT(base.Order() <= shape.Order());

  auto order = base.Order();

  ASSERT(base[order-1] >= 0);
  ASSERT(bound[order-1] < shape[order-1]);

  std::vector<std::pair<Point, Point>> retval;

  if (base[order-1] == bound[order-1])
  {
    // Base and bound are the same coordinate at this rank.
    if (order == 1)
    {
      // Base case: we are the innermost rank.
      retval.push_back(std::make_pair(Point(0), Point(0)));
    }
    else
    {
      // Recurse and carve the next rank.
      retval = Carve(base.DiscardTopRank(), bound.DiscardTopRank(), shape);
    }

    // Concatenate this rank to the low,high points in each AAHR.
    for (auto& aahr: retval)
    {
      aahr.first.AddTopRank(base[order-1]);
      aahr.second.AddTopRank(bound[order-1]);
    }
  }     
  else if (base[order-1] <= bound[order-1])
  {
    // Since base and bound are different coordinates, we can safely
    // march forward from the base and backward from the bound at
    // all ranks. No recursion needed.
        
    // March forward from base. Start processing from innermost rank.
    // Skip the outermost rank for now.
    for (unsigned rank = 0; rank < order-1; rank++)
    {
      if (base[rank] != 0)
      {
        // Need to add an AAHR at this rank.
        auto high = base;
        for (unsigned r = 0; r <= rank; r++)
          high[r] = shape[r]-1;
        retval.push_back(std::make_pair(base, high));
                
        // Update base coord at this rank to 0 (because we've
        // taken care of that section with this AAHR).
        base[rank] = 0;

        // Advance coordinate at the next rank by 1 (which may
        // have a ripple effect on other ranks).
        Increment(base, rank+1, shape);
      }
      // else: coordinate at this rank is 0, no need to add an AAHR.
    }

    // March backward from bound. Start processing from innermost rank.
    // Skip the outermost rank for now.
    for (unsigned rank = 0; rank < order-1; rank++)
    {
      if (bound[rank] != shape[rank]-1)
      {
        // Need to add an AAHR at this rank.
        auto low = bound;
        for (unsigned r = 0; r <= rank; r++)
          low[r] = 0;
        retval.push_back(std::make_pair(low, bound));

        // Update bound coord at this rank to shape-1 (because we've
        // taken care of that section with this AAHR).
        bound[rank] = shape[rank]-1;

        // Regress coordinate at the next rank by 1 (which may
        // have a ripple effect on other ranks).
        Decrement(bound, rank+1, shape);
      }
      // else: coordinate at this rank is max, no need to add an AAHR.
    }

    // Now process outermost rank. This will add a full AAHR.
    // Remember we started with base < bound at this rank. However,
    // we may have recursively advanced the base by 1 *and* the
    // regressed the bound by 1, which would mean base > bound.
    // This is not illegal: remember that we are using an inclusive
    // bound, so base > bound simply means this rank is empty.
    // In other words, the smaller AAHRs completely accounted for
    // our target region, we do not need a full size AAHR.
    if (base[order-1] <= bound[order-1])
    {
      // Note: base/boundcoords at lower ranks have been zeroed out.
      retval.push_back(std::make_pair(base, bound));
    }
  }
  else if (base[order-1] == bound[order-1]+1)
  {
    // Believe it or not, this is legal and is a consequence of using
    // inclusive bounds. This just means the space is empty.
  }
  else
  {
    ASSERT(false);
  }

  return retval;
}
