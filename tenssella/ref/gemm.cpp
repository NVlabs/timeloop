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
  int M=7, N=13, K=15;

  // Program to read MatrixA from GlobalBuffer.
  if (N >= 1)
    for (int c2 = 0; c2 <= min(min(min(90, 13 * M - 1), 13 * M + N - 14), N + 77); c2 += 1)
      if (N >= (c2 % 13) + 1)
        for (int c3 = 0; c3 <= min(floord(K - 1, 3), 4); c3 += 1)
          for (int c5 = 3 * c3; c5 <= min(K - 1, 3 * c3 + 2); c5 += 1)
            Read_GlobalBuffer_MatrixA(0, 0, c2, c3, c2 / 13, c5);


  // Program to read MatrixB from GlobalBuffer.
  if (N >= 1)
    for (int c2 = 0; c2 <= min(min(min(90, 13 * M - 1), 13 * M + N - 14), N + 77); c2 += 1)
      if (N >= (c2 % 13) + 1)
        for (int c3 = 0; c3 <= min(floord(K - 1, 3), 4); c3 += 1)
          for (int c4 = 3 * c3; c4 <= min(K - 1, 3 * c3 + 2); c4 += 1)
            Read_GlobalBuffer_MatrixB(0, 0, c2, c3, c4, c2 % 13);


  // Program to read MatrixZ from GlobalBuffer.
  if (N >= 1)
    for (int c2 = 0; c2 <= min(min(min(90, 13 * M - 1), 13 * M + N - 14), N + 77); c2 += 1)
      if (N >= (c2 % 13) + 1)
        for (int c3 = 0; c3 <= min(floord(K - 1, 3), 4); c3 += 1)
          Read_GlobalBuffer_MatrixZ(0, 0, c2, c3, c2 / 13, c2 % 13);


  // Program to read MatrixA from RegFile.
  if (M >= 1 && N >= 1)
    for (int c0 = 0; c0 <= 90; c0 += 1)
      for (int c1 = 0; c1 <= 4; c1 += 1)
        for (int c4 = 0; c4 <= min(2, K - 1); c4 += 1)
          Read_RegFile_MatrixA(c0, c1, c0, c1, c4, 0, c4);


  // Program to read MatrixB from RegFile.
  if (M >= 1 && N >= 1)
    for (int c0 = 0; c0 <= 90; c0 += 1)
      for (int c1 = 0; c1 <= 4; c1 += 1)
        for (int c4 = 0; c4 <= min(2, K - 1); c4 += 1)
          Read_RegFile_MatrixB(c0, c1, c0, c1, c4, c4, 0);


  // Program to read MatrixZ from RegFile.
  if (M >= 1 && N >= 1)
    for (int c0 = 0; c0 <= 90; c0 += 1)
      for (int c1 = 0; c1 <= 4; c1 += 1)
        for (int c4 = 0; c4 <= min(2, K - 1); c4 += 1)
          Read_RegFile_MatrixZ(c0, c1, c0, c1, c4, 0, 0);


  return 0;
}
