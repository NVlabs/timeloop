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

#include <cstdarg>

// std::function<void(int src_s, int src_t, int dst_s, int dst_t, int indices...)>
// READ(std::string transfer_engine_level,
//      std::string src_buffet_level,
//      std::string dst_buffet_level,
//      std::string tensor_name)
// {
//   return [&](int src_s, int src_t, int dst_s, int dst_t, int indices...)
//   {
//     (void) src_t;
//     (void) dst_t;

//     std::stringstream tensor_point;
//     tensor_point << tensor_name; // << "_" << w;

//     Action<float> action;
//     action.op = Op::READ;
//     action.src_s = src_s;
//     action.dst_s = dst_s;
//     action.srcs.push_back({ src_buffet_level, tensor_point.str(), false});
//     action.transform = [](std::vector<float>& operands, std::vector<float>& results)
//       {
//         float x = operands.at(0);
//         results.push_back(x);
//       };
//     action.dsts.push_back({ dst_buffet_level, tensor_point.str(), false });

//     arch[transfer_engine_level](src_s).AddAction(action);
//   };
// }

template<const char* transfer_engine_level,
         const char* src_buffet_level,
         const char* dst_buffet_level,
         const char* tensor_name,
         int num_ranks>
void READ(int src_s, int src_t, int dst_s, int dst_t...)
{
  (void) src_t;
  (void) dst_t;

  std::stringstream tensor_point;
  tensor_point << tensor_name;

  va_list args;
  va_start(args, dst_t);
  for (int rank = 0; rank < num_ranks; rank++)
  {
    tensor_point << "_" << va_arg(args, int);
  }
  va_end(args);

  Action<float> action;
  action.op = Op::READ;
  action.src_s = src_s;
  action.dst_s = dst_s;
  action.srcs.push_back({ src_buffet_level, tensor_point.str(), false});
  action.transform = [](std::vector<float>& operands, std::vector<float>& results)
    {
      float x = operands.at(0);
      results.push_back(x);
    };
  action.dsts.push_back({ dst_buffet_level, tensor_point.str(), false });

  arch[transfer_engine_level](src_s).AddAction(action);
}




// Read a single Inputs from DiagBroadcaster and Fill into OperandB.
std::function<void(int src_s, int src_t, int dst_s, int dst_t, int w)>
READ(std::string transfer_engine_level,
     std::string src_buffet_level,
     std::string dst_buffet_level,
     std::string tensor_name)
{
  return [=]
    (int src_s, int src_t, int dst_s, int dst_t, int w)
  {
    (void) src_t;
    (void) dst_t;

    std::stringstream tensor_point;
    tensor_point << "Inputs" << "_" << w;

    Action<float> action;
    action.op = Op::READ;
    action.src_s = src_s;
    action.dst_s = dst_s;
    action.srcs.push_back({ src_buffet_level, tensor_point.str(), false});
    action.transform = [](std::vector<float>& operands, std::vector<float>& results)
      {
        float x = operands.at(0);
        results.push_back(x);
      };
    action.dsts.push_back({ dst_buffet_level, tensor_point.str(), false });

    arch[transfer_engine_level](src_s).AddAction(action);
  }
}



// Read a single Inputs from DiagBroadcaster and Fill into OperandB.
void READ_DiagBroadcaster_to_OperandB_Inputs(int src_s, int src_t, int dst_s, int dst_t, int w)
{
  (void) src_t;
  (void) dst_t;

  std::stringstream tensor_point;
  tensor_point << "Inputs" << "_" << w;

  Action<float> action;
  action.op = Op::READ;
  action.src_s = src_s;
  action.dst_s = dst_s;
  action.srcs.push_back({ "DiagBroadcaster", tensor_point.str(), false});
  action.transform = [](std::vector<float>& operands, std::vector<float>& results)
  {
    float x = operands.at(0);
    results.push_back(x);
  };
  action.dsts.push_back({ "OperandB", tensor_point.str(), false });

  arch["DiagBroadcaster"](src_s).AddAction(action);
}
