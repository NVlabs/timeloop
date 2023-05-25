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

#include "../emulation/includes.hpp"
#include "../emulation/functions.hpp"
#include "../emulation/arch.hpp"
#include "../emulation/action-macros.hpp"

Arch<float> arch;

// Compute a single Multiply at Multiplier.
//#define COMPUTE_Multiplier_Multiply(arch, srcA, srcB, dst)            \
//  [](int s, int t, int k, int p, int r)                               \
//  {                                                                   \
//    (void) t;                                                         \

void COMPUTE_Multiplier_Multiply(int s, int t, int k, int p, int r)
{
  const char* srcA = "OperandA";
  const char* srcB = "OperandB";
  const char* dst = "Result";

  (void) t;

  std::stringstream operand_tensor_point_Weights;
  operand_tensor_point_Weights << "Weights" << "_" << k << "_" << r;
  std::stringstream operand_tensor_point_Inputs;
  operand_tensor_point_Inputs << "Inputs" << "_" << p+r;
  std::stringstream operand_tensor_point_Outputs;
  operand_tensor_point_Outputs << "Outputs" << "_" << k << "_" << p;
  std::stringstream result_tensor_point_Outputs;
  result_tensor_point_Outputs << "Outputs" << "_" << k << "_" << p;
  
  Action<float> action;
  action.op = Op::COMPUTE;
  action.src_s = s;
  action.dst_s = s;
  action.srcs.push_back({ srcA, operand_tensor_point_Weights.str(), false });
  action.srcs.push_back({ srcB, operand_tensor_point_Inputs.str(), false });
  action.srcs.push_back({ dst, operand_tensor_point_Outputs.str(), true });
  action.transform = [](std::vector<float>& operands, std::vector<float>& results)
  {
    float a = operands.at(0);
    float b = operands.at(1);
    float z = operands.at(2);
    float x = z + a*b;
    results.push_back(x);
  };
  action.dsts.push_back({ dst, result_tensor_point_Outputs.str(), false });
  
  arch["Multiplier"](s).AddAction(action);
}

int main()
{
  int K=16, P=14, R=3;

  //
  // Run test.
  //

  // Program to init Weights at DRAM.
  if (P >= 1)
    for (int c2 = 0; c2 < K; c2 += 1)
      for (int c3 = 0; c3 < R; c3 += 1)
        ACTION_INIT("DRAM", "DRAM", "Weights", 2)(0, 0, c2, c3);

  // Program to init Inputs at DRAM.
  if (K >= 1 && P >= 1 && R >= 1)
    for (int c2 = 0; c2 < P + R - 1; c2 += 1)
      ACTION_INIT("DRAM", "DRAM", "Inputs", 1)(0, 0, c2);

  // Program to init Outputs at DRAM.
  if (R >= 1)
    for (int c2 = 0; c2 < K; c2 += 1)
      for (int c3 = 0; c3 < P; c3 += 1)
        ACTION_INIT("DRAM", "DRAM", "Outputs", 2)(0, 0, c2, c3);

  // Program to read Weights from DRAM into RowBuffer.
  if (P >= 1)
    for (int c3 = 0; c3 <= min(15, K - 1); c3 += 1)
      for (int c4 = 0; c4 <= min(2, R - 1); c4 += 1)
        ACTION_READ("DRAM", "DRAM", "RowBuffer", "Weights", 2)(0, 0, c4, 0, c3, c4);

  // Program to read Inputs from DRAM into DiagBuffer.
  if (K >= 1 && P >= 1 && R >= 1)
    for (int c3 = 0; c3 <= min(min(min(15, P + 1), P + R - 2), R + 12); c3 += 1)
      ACTION_READ("DRAM", "DRAM", "DiagBuffer", "Inputs", 1)(0, 0, c3, 0, c3);

  // Program to read Outputs from DRAM into ColBuffer.
  if (R >= 1)
    for (int c3 = 0; c3 <= min(15, K - 1); c3 += 1)
      for (int c4 = 0; c4 <= min(13, P - 1); c4 += 1)
        ACTION_READ_IU("DRAM", "DRAM", "ColBuffer", "Outputs", 2)(0, 0, c4, 0, c3, c4);

  // Program to read Weights from RowBuffer into RowBroadcaster.
  if (P >= 1) {
    for (int c2 = 0; c2 <= min(15, K - 1); c2 += 1)
      for (int c4 = 0; c4 <= min(2, R - 1); c4 += 1)
        ACTION_READ("RowBuffer", "RowBuffer", "RowBroadcaster", "Weights", 2)(c4, 0, c4, c2, c2, c4);
    for (int c3 = 0; c3 <= min(15, K - 1); c3 += 1)
      for (int c4 = 0; c4 <= min(2, R - 1); c4 += 1)
        ACTION_SHRINK("RowBuffer", "RowBuffer", "Weights", 2)(0, 0, c4, 0, c3, c4);
  }

  // Program to read Inputs from DiagBuffer into DiagBroadcaster.
  if (K >= 1 && P >= 1 && R >= 1) {
    for (int c3 = 0; c3 <= min(min(min(15, P + 1), P + R - 2), R + 12); c3 += 1)
      ACTION_READ("DiagBuffer", "DiagBuffer", "DiagBroadcaster", "Inputs", 1)(c3, 0, c3, 0, c3);
    for (int c3 = 0; c3 <= min(min(min(15, P + 1), P + R - 2), R + 12); c3 += 1)
      ACTION_SHRINK("DiagBuffer", "DiagBuffer", "Inputs", 1)(0, 0, c3, 0, c3);
  }

  // Program to read Outputs from ColBuffer into ColSpatialReducer.
  if (R >= 1) {
    for (int c2 = 0; c2 <= min(15, K - 1); c2 += 1)
      for (int c4 = 0; c4 <= min(13, P - 1); c4 += 1)
        ACTION_READ_IU("ColBuffer", "ColBuffer", "ColSpatialReducer", "Outputs", 2)(c4, 0, c4, c2, c2, c4);
    for (int c3 = 0; c3 <= min(15, K - 1); c3 += 1)
      for (int c4 = 0; c4 <= min(13, P - 1); c4 += 1)
        ACTION_UPDATE("ColBuffer", "DRAM", "ColBuffer", "Outputs", 2)(0, 0, c4, 0, c3, c4);
  }

  // Program to read Weights from RowBroadcaster into OperandA.
  if (P >= 1)
    for (int c0 = 0; c0 <= min(15, K - 1); c0 += 1) {
      for (int c4 = 0; c4 <= min(2, R - 1); c4 += 1)
        for (int c8 = 5 * c4; c8 <= min(P + 5 * c4 - 1, 5 * c4 + 4); c8 += 1)
          ACTION_READ("RowBroadcaster", "RowBroadcaster", "OperandA", "Weights", 2)(c4, c0, c8, c0, c0, c4);
      for (int c4 = 0; c4 <= min(2, R - 1); c4 += 1)
        ACTION_SHRINK("RowBroadcaster", "RowBroadcaster", "Weights", 2)(c4, 0, c4, c0, c0, c4);
    }

  // Program to read Inputs from DiagBroadcaster into OperandB.
  if (K >= 1) {
    for (int c3 = 0; c3 <= min(min(min(6, P + 1), P + R - 2), R + 3); c3 += 1)
      for (int c8 = max(max(5 * c3 - 16, c3), -4 * P + 5 * c3 + 4); c8 <= min(min(4 * R + c3 - 4, 5 * c3), c3 + 8); c8 += 4)
        ACTION_READ("DiagBroadcaster", "DiagBroadcaster", "OperandB", "Inputs", 1)(c3, 0, c8, 0, c3);
    if (K >= 16 && P >= 1 && R >= 1) {
      for (int c3 = 0; c3 <= min(min(min(15, P + 1), P + R - 2), R + 12); c3 += 1)
        ACTION_SHRINK("DiagBroadcaster", "DiagBroadcaster", "Inputs", 1)(c3, 0, c3, 15, c3);
    } else if (K <= 15 && P >= 1 && R >= 1) {
      for (int c3 = 0; c3 <= min(min(min(15, P + 1), P + R - 2), R + 12); c3 += 1)
        ACTION_SHRINK("DiagBroadcaster", "DiagBroadcaster", "Inputs", 1)(c3, 0, c3, K - 1, c3);
    }
  }

  // Program to read Outputs from ColSpatialReducer into Result.
  if (R >= 1)
    for (int c0 = 0; c0 <= min(15, K - 1); c0 += 1) {
      for (int c4 = 0; c4 <= min(4, P - 1); c4 += 1)
        for (int c8 = c4; c8 <= min(5 * R + c4 - 5, c4 + 10); c8 += 5)
          ACTION_READ_IU("ColSpatialReducer", "ColSpatialReducer", "Result", "Outputs", 2)(c4, c0, c8, c0, c0, c4);
      for (int c4 = 0; c4 <= min(13, P - 1); c4 += 1)
        ACTION_UPDATE("ColSpatialReducer", "ColBuffer", "ColSpatialReducer", "Outputs", 2)(c4, 0, c4, c0, c0, c4);
    }

  // Program to compute Multiply at Multiplier.
  for (int c0 = 0; c0 <= 15; c0 += 1) {
    for (int c4 = 0; c4 <= 4; c4 += 1)
      for (int c5 = 0; c5 <= 2; c5 += 1)
        COMPUTE_Multiplier_Multiply(c4 + 5 * c5, c0, c0, c4, c5);
    if (K >= c0 + 1) {
      for (int c4 = 0; c4 <= min(4, P - 1); c4 += 1)
        for (int c6 = c4; c6 <= min(5 * R + c4 - 5, c4 + 10); c6 += 5)
          ACTION_UPDATE("Multiplier", "ColSpatialReducer", "Result", "Outputs", 2)(c4, c0, c6, c0, c0, c4);
      if (K <= 15 && c0 + 1 == K) {
        for (int c3 = 0; c3 <= min(min(min(min(6, K - 2), P + 1), P + R - 2), R + 3); c3 += 1)
          for (int c6 = max(max(5 * c3 - 16, c3), -4 * P + 5 * c3 + 4); c6 <= min(min(4 * R + c3 - 4, 5 * c3), c3 + 8); c6 += 4)
            ACTION_SHRINK("Multiplier", "OperandB", "Inputs", 1)(c3, K - 1, c6, K - 1, c3);
      } else if (c0 == 15) {
        for (int c3 = 0; c3 <= min(min(min(6, P + 1), P + R - 2), R + 3); c3 += 1)
          for (int c6 = max(max(5 * c3 - 16, c3), -4 * P + 5 * c3 + 4); c6 <= min(min(4 * R + c3 - 4, 5 * c3), c3 + 8); c6 += 4)
            ACTION_SHRINK("Multiplier", "OperandB", "Inputs", 1)(c3, 15, c6, 15, c3);
      }
      for (int c4 = 0; c4 <= min(2, R - 1); c4 += 1) {
        for (int c6 = 5 * c4; c6 <= min(P + 5 * c4 - 1, 5 * c4 + 4); c6 += 1) {
          ACTION_SHRINK("Multiplier", "OperandA", "Weights", 2)(c4, c0, c6, c0, c0, c4);
          if (c0 + 1 == K && c4 == 0 && c6 + 1 == K)
            ACTION_SHRINK("Multiplier", "OperandB", "Inputs", 1)(K - 1, K - 1, K - 1, K - 1, K - 1);
        }
        if (c0 + 1 == K && c4 == 0)
          for (int c6 = max(max(5 * K - 21, 5 * K - 4 * P - 1), ((K + 2) % 4) + 5); c6 <= min(min(5 * K - 5, K + 7), K + 4 * R - 5); c6 += 4)
            ACTION_SHRINK("Multiplier", "OperandB", "Inputs", 1)(K - 1, K - 1, c6, K - 1, K - 1);
      }
      if (c0 + 1 == K)
        for (int c3 = K; c3 <= min(min(min(6, P + 1), P + R - 2), R + 3); c3 += 1)
          for (int c6 = max(max(5 * c3 - 16, c3), -4 * P + 5 * c3 + 4); c6 <= min(min(4 * R + c3 - 4, 5 * c3), c3 + 8); c6 += 4)
            ACTION_SHRINK("Multiplier", "OperandB", "Inputs", 1)(c3, K - 1, c6, K - 1, c3);
    }
  }

  arch_<float> = &arch;  
  arch.Run();
  arch.Wait();

  std::cerr << "\n === TEST RUN COMPLETE ===\n" << std::endl;

  // Program to save Outputs into validation buffer.
  if (R >= 1)
    for (int c2 = 0; c2 < K; c2 += 1)
      for (int c3 = 0; c3 < P; c3 += 1)
        ACTION_INLINE_SAVE("DRAM", "DRAM", "__val__", "Outputs", 2)(0, 0, c2, c3);

  std::cerr << "\n === OUTPUTS SAVED ===\n" << std::endl;

  //
  // Run reference.
  //

  arch.Reset();

  // Program to init Weights at DRAM.
  if (P >= 1)
    for (int c2 = 0; c2 < K; c2 += 1)
      for (int c3 = 0; c3 < R; c3 += 1)
        ACTION_INIT("DRAM", "DRAM", "Weights", 2)(0, 0, c2, c3);

  // Program to init Inputs at DRAM.
  if (K >= 1 && P >= 1 && R >= 1)
    for (int c2 = 0; c2 < P + R - 1; c2 += 1)
      ACTION_INIT("DRAM", "DRAM", "Inputs", 1)(0, 0, c2);

  // Program to init Outputs at DRAM.
  if (R >= 1)
    for (int c2 = 0; c2 < K; c2 += 1)
      for (int c3 = 0; c3 < P; c3 += 1)
        ACTION_INIT("DRAM", "DRAM", "Outputs", 2)(0, 0, c2, c3);

  // Program to read Weights from DRAM into RowBuffer.
  if (P >= 1)
    for (int c3 = 0; c3 <= min(15, K - 1); c3 += 1)
      for (int c4 = 0; c4 <= min(2, R - 1); c4 += 1)
        ACTION_READ("DRAM", "DRAM", "RowBuffer", "Weights", 2)(0, 0, c4, 0, c3, c4);

  // Program to read Inputs from DRAM into DiagBuffer.
  if (K >= 1 && P >= 1 && R >= 1)
    for (int c3 = 0; c3 <= min(min(min(15, P + 1), P + R - 2), R + 12); c3 += 1)
      ACTION_READ("DRAM", "DRAM", "DiagBuffer", "Inputs", 1)(0, 0, c3, 0, c3);

  // Program to read Outputs from DRAM into ColBuffer.
  if (R >= 1)
    for (int c3 = 0; c3 <= min(15, K - 1); c3 += 1)
      for (int c4 = 0; c4 <= min(13, P - 1); c4 += 1)
        ACTION_READ_IU("DRAM", "DRAM", "ColBuffer", "Outputs", 2)(0, 0, c4, 0, c3, c4);

  // Program to read Weights from RowBuffer into RowBroadcaster.
  if (P >= 1) {
    for (int c2 = 0; c2 <= min(15, K - 1); c2 += 1)
      for (int c4 = 0; c4 <= min(2, R - 1); c4 += 1)
        ACTION_READ("RowBuffer", "RowBuffer", "RowBroadcaster", "Weights", 2)(c4, 0, c4, c2, c2, c4);
    for (int c3 = 0; c3 <= min(15, K - 1); c3 += 1)
      for (int c4 = 0; c4 <= min(2, R - 1); c4 += 1)
        ACTION_SHRINK("RowBuffer", "RowBuffer", "Weights", 2)(0, 0, c4, 0, c3, c4);
  }

  // Program to read Inputs from DiagBuffer into DiagBroadcaster.
  if (K >= 1 && P >= 1 && R >= 1) {
    for (int c3 = 0; c3 <= min(min(min(15, P + 1), P + R - 2), R + 12); c3 += 1)
      ACTION_READ("DiagBuffer", "DiagBuffer", "DiagBroadcaster", "Inputs", 1)(c3, 0, c3, 0, c3);
    for (int c3 = 0; c3 <= min(min(min(15, P + 1), P + R - 2), R + 12); c3 += 1)
      ACTION_SHRINK("DiagBuffer", "DiagBuffer", "Inputs", 1)(0, 0, c3, 0, c3);
  }

  // Program to read Outputs from ColBuffer into ColSpatialReducer.
  if (R >= 1) {
    for (int c2 = 0; c2 <= min(15, K - 1); c2 += 1)
      for (int c4 = 0; c4 <= min(13, P - 1); c4 += 1)
        ACTION_READ_IU("ColBuffer", "ColBuffer", "ColSpatialReducer", "Outputs", 2)(c4, 0, c4, c2, c2, c4);
    for (int c3 = 0; c3 <= min(15, K - 1); c3 += 1)
      for (int c4 = 0; c4 <= min(13, P - 1); c4 += 1)
        ACTION_UPDATE("ColBuffer", "DRAM", "ColBuffer", "Outputs", 2)(0, 0, c4, 0, c3, c4);
  }

  // Program to read Weights from RowBroadcaster into OperandA.
  if (P >= 1)
    for (int c0 = 0; c0 <= min(15, K - 1); c0 += 1) {
      for (int c4 = 0; c4 <= min(2, R - 1); c4 += 1)
        for (int c8 = 5 * c4; c8 <= min(P + 5 * c4 - 1, 5 * c4 + 4); c8 += 1)
          ACTION_READ("RowBroadcaster", "RowBroadcaster", "OperandA", "Weights", 2)(c4, c0, c8, c0, c0, c4);
      for (int c4 = 0; c4 <= min(2, R - 1); c4 += 1)
        ACTION_SHRINK("RowBroadcaster", "RowBroadcaster", "Weights", 2)(c4, 0, c4, c0, c0, c4);
    }

  // Program to read Inputs from DiagBroadcaster into OperandB.
  if (K >= 1) {
    for (int c3 = 0; c3 <= min(min(min(6, P + 1), P + R - 2), R + 3); c3 += 1)
      for (int c8 = max(max(5 * c3 - 16, c3), -4 * P + 5 * c3 + 4); c8 <= min(min(4 * R + c3 - 4, 5 * c3), c3 + 8); c8 += 4)
        ACTION_READ("DiagBroadcaster", "DiagBroadcaster", "OperandB", "Inputs", 1)(c3, 0, c8, 0, c3);
    if (K >= 16 && P >= 1 && R >= 1) {
      for (int c3 = 0; c3 <= min(min(min(15, P + 1), P + R - 2), R + 12); c3 += 1)
        ACTION_SHRINK("DiagBroadcaster", "DiagBroadcaster", "Inputs", 1)(c3, 0, c3, 15, c3);
    } else if (K <= 15 && P >= 1 && R >= 1) {
      for (int c3 = 0; c3 <= min(min(min(15, P + 1), P + R - 2), R + 12); c3 += 1)
        ACTION_SHRINK("DiagBroadcaster", "DiagBroadcaster", "Inputs", 1)(c3, 0, c3, K - 1, c3);
    }
  }

  // Program to read Outputs from ColSpatialReducer into Result.
  if (R >= 1)
    for (int c0 = 0; c0 <= min(15, K - 1); c0 += 1) {
      for (int c4 = 0; c4 <= min(4, P - 1); c4 += 1)
        for (int c8 = c4; c8 <= min(5 * R + c4 - 5, c4 + 10); c8 += 5)
          ACTION_READ_IU("ColSpatialReducer", "ColSpatialReducer", "Result", "Outputs", 2)(c4, c0, c8, c0, c0, c4);
      for (int c4 = 0; c4 <= min(13, P - 1); c4 += 1)
        ACTION_UPDATE("ColSpatialReducer", "ColBuffer", "ColSpatialReducer", "Outputs", 2)(c4, 0, c4, c0, c0, c4);
    }

  // Program to compute Multiply at Multiplier.
  for (int c0 = 0; c0 <= 15; c0 += 1) {
    for (int c4 = 0; c4 <= 4; c4 += 1)
      for (int c5 = 0; c5 <= 2; c5 += 1)
        COMPUTE_Multiplier_Multiply(c4 + 5 * c5, c0, c0, c4, c5);
    if (K >= c0 + 1) {
      for (int c4 = 0; c4 <= min(4, P - 1); c4 += 1)
        for (int c6 = c4; c6 <= min(5 * R + c4 - 5, c4 + 10); c6 += 5)
          ACTION_UPDATE("Multiplier", "ColSpatialReducer", "Result", "Outputs", 2)(c4, c0, c6, c0, c0, c4);
      if (K <= 15 && c0 + 1 == K) {
        for (int c3 = 0; c3 <= min(min(min(min(6, K - 2), P + 1), P + R - 2), R + 3); c3 += 1)
          for (int c6 = max(max(5 * c3 - 16, c3), -4 * P + 5 * c3 + 4); c6 <= min(min(4 * R + c3 - 4, 5 * c3), c3 + 8); c6 += 4)
            ACTION_SHRINK("Multiplier", "OperandB", "Inputs", 1)(c3, K - 1, c6, K - 1, c3);
      } else if (c0 == 15) {
        for (int c3 = 0; c3 <= min(min(min(6, P + 1), P + R - 2), R + 3); c3 += 1)
          for (int c6 = max(max(5 * c3 - 16, c3), -4 * P + 5 * c3 + 4); c6 <= min(min(4 * R + c3 - 4, 5 * c3), c3 + 8); c6 += 4)
            ACTION_SHRINK("Multiplier", "OperandB", "Inputs", 1)(c3, 15, c6, 15, c3);
      }
      for (int c4 = 0; c4 <= min(2, R - 1); c4 += 1) {
        for (int c6 = 5 * c4; c6 <= min(P + 5 * c4 - 1, 5 * c4 + 4); c6 += 1) {
          ACTION_SHRINK("Multiplier", "OperandA", "Weights", 2)(c4, c0, c6, c0, c0, c4);
          if (c0 + 1 == K && c4 == 0 && c6 + 1 == K)
            ACTION_SHRINK("Multiplier", "OperandB", "Inputs", 1)(K - 1, K - 1, K - 1, K - 1, K - 1);
        }
        if (c0 + 1 == K && c4 == 0)
          for (int c6 = max(max(5 * K - 21, 5 * K - 4 * P - 1), ((K + 2) % 4) + 5); c6 <= min(min(5 * K - 5, K + 7), K + 4 * R - 5); c6 += 4)
            ACTION_SHRINK("Multiplier", "OperandB", "Inputs", 1)(K - 1, K - 1, c6, K - 1, K - 1);
      }
      if (c0 + 1 == K)
        for (int c3 = K; c3 <= min(min(min(6, P + 1), P + R - 2), R + 3); c3 += 1)
          for (int c6 = max(max(5 * c3 - 16, c3), -4 * P + 5 * c3 + 4); c6 <= min(min(4 * R + c3 - 4, 5 * c3), c3 + 8); c6 += 4)
            ACTION_SHRINK("Multiplier", "OperandB", "Inputs", 1)(c3, K - 1, c6, K - 1, c3);
    }
  }

  arch_<float> = &arch;  
  arch.Run();
  arch.Wait();

  std::cerr << "\n === REF RUN COMPLETE ===\n" << std::endl;

  // Program to validate Outputs.
  if (R >= 1)
    for (int c2 = 0; c2 < K; c2 += 1)
      for (int c3 = 0; c3 < P; c3 += 1)
        ACTION_INLINE_VALIDATE("DRAM", "DRAM", "__val__", "Outputs", 2)(0, 0, c2, c3);

  std::cerr << "\n === VALIDATION COMPLETE ===\n" << std::endl;

  return 0;
}
