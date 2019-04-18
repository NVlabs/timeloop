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

#include <cassert>
#include <iostream>

#include "pat.hpp"

namespace pat
{

double SRAMArea(std::uint64_t height, std::uint64_t width, std::uint64_t num_banks, std::uint64_t num_ports)
{
  (void) height;
  (void) width;
  (void) num_banks;
  (void) num_ports;
  return 0;
}

double SRAMEnergy(std::uint64_t height, std::uint64_t width, std::uint64_t num_banks, std::uint64_t num_ports)
{
  (void) num_banks;
  (void) num_ports;
  
  // Eyeriss data points:

  // 0.5 KB (x0) = 1.0 units (y0)
  const double x0 = 0.5;
  const double y0 = 1.0;
  
  // 100 KB (x1) = 6.0 units (y1)
  const double x1 = 100.0;
  const double y1 = 6.0;

  // slope m = (y1-y0)/(x1-x0)
  const double m = (y1 - y0) / (x1 - x0);

  // y = m(x-x0)+y0    
  double x = double(width * height) / (8 * 1024);
  double y = m * (x - x0) + y0;

  // The baseline data was for a 16b access.
  // Scale this based on the width of the access.
  // This will make the model insensitive to block
  // size.
  double energy = (y * width) / 16;
  return energy;
}

double DRAMEnergy(std::uint64_t width)
{
  double energy = (200.0 * width) / 16;
  return energy;
}

double WireEnergy(std::uint64_t bits, double length_mm)
{
  (void) bits;
  (void) length_mm;
  return 0;
}

double MultiplierEnergy(std::uint64_t bits_A, std::uint64_t bits_B)
{
  double energy = 1.0 * (double(bits_A) / 16.0) * (double(bits_B) / 16.0);
  return energy;
}

double MultiplierArea(std::uint64_t bits_A, std::uint64_t bits_B)
{
  (void) bits_A;
  (void) bits_B;
  return 0;
}

double AdderEnergy(std::uint64_t bits_A, std::uint64_t bits_B)
{
  (void) bits_A;
  (void) bits_B;
  return 0;
}

double AdderArea(std::uint64_t bits_A, std::uint64_t bits_B)
{
  (void) bits_A;
  (void) bits_B;
  return 0;
}

} // namespace pat
