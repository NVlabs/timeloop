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

#include <boost/multiprecision/cpp_int.hpp>

#include "numeric.hpp"

using namespace boost::multiprecision;

//------------------------------------
//           Miscellaneous
//------------------------------------

// Returns the smallest factor of an integer and the quotient after
// division with the smallest factor.
void SmallestFactor(uint64_t n, uint64_t& factor, uint64_t& residue)
{
  for (uint64_t i = 2; i < n; i++)
  {
    if (n % i == 0)
    {
      factor = i;
      residue = n / i;
      return;
    }
  }
  factor = n;
  residue = 1;
}

// Helper function to get close-to-square layouts of arrays
// containing a given number of nodes.
void GetTiling(uint64_t num_elems, uint64_t& height, uint64_t& width)
{
  std::vector<uint64_t> factors;
  uint64_t residue = num_elems;
  uint64_t cur_factor;
  while (residue > 1)
  {
    SmallestFactor(residue, cur_factor, residue);
    factors.push_back(cur_factor);
  }

  height = 1;
  width = 1;
  for (uint64_t i = 0; i < factors.size(); i++)
  {
    if (i % 2 == 0)
      height *= factors[i];
    else
      width *= factors[i];
  }

  if (height > width)
  {
    uint64_t temp = height;
    height = width;
    width = temp;
  }
}

double LinearInterpolate(double x,
                         double x0, double x1,
                         double q0, double q1)
{
  double slope = (x0 == x1) ? 0 : (q1 - q0) / double(x1 - x0);
  return q0 + slope * (x - x0);
}

double BilinearInterpolate(double x, double y,
                           double x0, double x1,
                           double y0, double y1,
                           double q00, double q01, double q10, double q11)
{
  // Linear interpolate along x dimension.
  double qx0 = LinearInterpolate(x, x0, x1, q00, q10);
  double qx1 = LinearInterpolate(x, x0, x1, q01, q11);

  // Linear interpolate along y dimension.
  return LinearInterpolate(y, y0, y1, qx0, qx1);
}
