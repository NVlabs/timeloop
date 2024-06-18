/* Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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

#include <iostream>
#include <algorithm>
#include <cstdio>

int min(int a, int b) { return std::min(a, b); }
int max(int a, int b) { return std::max(a, b); }
int floord(int a, int b) { return a/b; }

int main()
{
  int K=7, P=4, R=3;

  // Program to read Weights from DRAM.
  if (P >= 1)
    for (int c2 = 0; c2 <= min(2, R - 1); c2 += 1)
      for (int c4 = 0; c4 <= min(6, K - 1); c4 += 1)
        Read_DRAM_Weights(0, 0, c2, 0, c4, c2);


  // Program to read Inputs from DRAM.
  if (K >= 1 && P >= 1 && R >= 1)
    for (int c2 = 0; c2 <= min(min(min(5, P + 1), P + R - 2), R + 2); c2 += 1)
      Read_DRAM_Inputs(0, 0, c2, 0, c2);


  // Program to read Outputs from DRAM.
  if (R >= 1)
    for (int c2 = 0; c2 <= min(3, P - 1); c2 += 1)
      for (int c4 = 0; c4 <= min(6, K - 1); c4 += 1)
        Read_DRAM_Outputs(0, 0, c2, 0, c4, c2);


  // Program to read Weights from RowBuffer.
  if (P >= 1 && R >= 1)
    for (int c0 = 0; c0 <= 2; c0 += 1)
      for (int c3 = 0; c3 <= min(6, K - 1); c3 += 1)
        Read_RowBuffer_Weights(c0, 0, c0, c3, c3, 0);


  // Program to read Inputs from DiagBuffer.
  if (P >= 1 && R >= 1)
    for (int c0 = 0; c0 <= 5; c0 += 1)
      for (int c3 = 0; c3 <= min(6, K - 1); c3 += 1)
        Read_DiagBuffer_Inputs(c0, 0, c0, c3, 0);


  // Program to read Outputs from ColBuffer.
  if (P >= 1 && R >= 1)
    for (int c0 = 0; c0 <= 3; c0 += 1)
      for (int c3 = 0; c3 <= min(6, K - 1); c3 += 1)
        Read_ColBuffer_Outputs(c0, 0, c0, c3, c3, 0);


  // Program to read Weights from RowBroadcaster.
  if (K >= 1 && P >= 1 && R >= 1)
    for (int c0 = 0; c0 <= 2; c0 += 1)
      for (int c1 = 0; c1 <= 6; c1 += 1)
        for (int c3 = 0; c3 <= 3; c3 += 1)
          Read_RowBroadcaster_Weights(c0, c1, c0, c3, c1, 0, 0);


  // Program to read Inputs from DiagBroadcaster.
  if (K >= 1 && P >= 1 && R >= 1)
    for (int c0 = 0; c0 <= 5; c0 += 1)
      for (int c1 = 0; c1 <= 6; c1 += 1)
        for (int c2 = max(0, c0 - 3); c2 <= min(2, c0); c2 += 1)
          Read_DiagBroadcaster_Inputs(c0, c1, c2, c0 - c2, c1, 0);


  // Program to read Outputs from ColSpatialReducer.
  if (K >= 1 && P >= 1 && R >= 1)
    for (int c0 = 0; c0 <= 3; c0 += 1)
      for (int c1 = 0; c1 <= 6; c1 += 1)
        for (int c2 = 0; c2 <= 2; c2 += 1)
          Read_ColSpatialReducer_Outputs(c0, c1, c2, c0, c1, 0, 0);


  return 0;
}
