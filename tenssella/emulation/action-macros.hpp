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

#pragma once

#include <cstdarg>
#include <string>
#include <unordered_map>

#define ACTION_READ_T3(transfer_engine_level, src_buffet_level, dst_buffet_level, tensor_name, num_ranks)     \
  [](int src_s, int src_t, int src_t1, int dst_s, int dst_t, int dst_t2, int dst_t3...)                                                        \
  {                                                                                                        \
    (void) src_t;                                                                                          \
    (void) src_t1;                                                                                          \
    (void) dst_t;                                                                                          \
    (void) dst_t2;                                                                                          \
    (void) dst_t3;                                                                                          \
                                                                                                           \
    std::stringstream tensor_point;                                                                        \
    tensor_point << tensor_name;                                                                           \
                                                                                                           \
    va_list args;                                                                                          \
    va_start(args, dst_t3);                                                                                 \
    for (int rank = 0; rank < num_ranks; rank++)                                                           \
    {                                                                                                      \
      tensor_point << "_" << va_arg(args, int);                                                            \
    }                                                                                                      \
    va_end(args);                                                                                          \
                                                                                                           \
    Action<float> action;                                                                                  \
    action.op = Op::READ;                                                                                  \
    action.srcs.push_back({src_s, src_buffet_level, tensor_point.str(), false});                                 \
    action.transform = [](std::vector<float>& operands, std::vector<float>& results)                       \
      {                                                                                                    \
        float x = operands.at(0);                                                                          \
        results.push_back(x);                                                                              \
      };                                                                                                   \
    action.dsts.push_back({dst_s, dst_buffet_level, tensor_point.str(), false });                                \
                                                                                                           \
    arch[transfer_engine_level](src_s).AddAction(action);                                                  \
  }

#define ACTION_READ_T2(transfer_engine_level, src_buffet_level, dst_buffet_level, tensor_name, num_ranks)     \
  [](int src_s, int src_t, int dst_s, int dst_t, int dst_t2...)                                                        \
  {                                                                                                        \
    (void) src_t;                                                                                          \
    (void) dst_t;                                                                                          \
    (void) dst_t2;                                                                                          \
                                                                                                           \
    std::stringstream tensor_point;                                                                        \
    tensor_point << tensor_name;                                                                           \
                                                                                                           \
    va_list args;                                                                                          \
    va_start(args, dst_t2);                                                                                 \
    for (int rank = 0; rank < num_ranks; rank++)                                                           \
    {                                                                                                      \
      tensor_point << "_" << va_arg(args, int);                                                            \
    }                                                                                                      \
    va_end(args);                                                                                          \
                                                                                                           \
    Action<float> action;                                                                                  \
    action.op = Op::READ;                                                                                  \
    action.srcs.push_back({src_s, src_buffet_level, tensor_point.str(), false});                                 \
    action.transform = [](std::vector<float>& operands, std::vector<float>& results)                       \
      {                                                                                                    \
        float x = operands.at(0);                                                                          \
        results.push_back(x);                                                                              \
      };                                                                                                   \
    action.dsts.push_back({dst_s, dst_buffet_level, tensor_point.str(), false });                                \
                                                                                                           \
    arch[transfer_engine_level](src_s).AddAction(action);                                                  \
  }

#define ACTION_READ(transfer_engine_level, src_buffet_level, dst_buffet_level, tensor_name, num_ranks)     \
  [](int src_s, int src_t, int dst_s, int dst_t...)                                                        \
  {                                                                                                        \
    (void) src_t;                                                                                          \
    (void) dst_t;                                                                                          \
                                                                                                           \
    std::stringstream tensor_point;                                                                        \
    tensor_point << tensor_name;                                                                           \
                                                                                                           \
    va_list args;                                                                                          \
    va_start(args, dst_t);                                                                                 \
    for (int rank = 0; rank < num_ranks; rank++)                                                           \
    {                                                                                                      \
      tensor_point << "_" << va_arg(args, int);                                                            \
    }                                                                                                      \
    va_end(args);                                                                                          \
                                                                                                           \
    Action<float> action;                                                                                  \
    action.op = Op::READ;                                                                                  \
    action.srcs.push_back({src_s, src_buffet_level, tensor_point.str(), false});                                 \
    action.transform = [](std::vector<float>& operands, std::vector<float>& results)                       \
      {                                                                                                    \
        float x = operands.at(0);                                                                          \
        results.push_back(x);                                                                              \
      };                                                                                                   \
    action.dsts.push_back({dst_s, dst_buffet_level, tensor_point.str(), false });                                \
                                                                                                           \
    arch[transfer_engine_level](src_s).AddAction(action);                                                  \
  }

#define ACTION_INLINE_SAVE(transfer_engine_level, src_buffet_level, dst_buffet_level, tensor_name, num_ranks) \
  [](int _s, int _t...)                                                                                       \
  {                                                                                                           \
    (void) _t;                                                                                                \
                                                                                                              \
    std::stringstream tensor_point;                                                                           \
    tensor_point << tensor_name;                                                                              \
                                                                                                              \
    va_list args;                                                                                             \
    va_start(args, _t);                                                                                       \
    for (int rank = 0; rank < num_ranks; rank++)                                                              \
    {                                                                                                         \
      tensor_point << "_" << va_arg(args, int);                                                               \
    }                                                                                                         \
    va_end(args);                                                                                             \
                                                                                                              \
    Action<float> action;                                                                                     \
    action.op = Op::READ;                                                                                     \
    action.srcs.push_back({_s, src_buffet_level, tensor_point.str(), false});                                    \
    action.transform = [](std::vector<float>& operands, std::vector<float>& results)                          \
      {                                                                                                       \
        float x = operands.at(0);                                                                             \
        results.push_back(x);                                                                                 \
      };                                                                                                      \
    action.dsts.push_back({_s, dst_buffet_level, tensor_point.str(), false });                                   \
                                                                                                              \
    arch[transfer_engine_level](_s).ProcessAction(action);                                                    \
  }

#define ACTION_INLINE_VALIDATE(transfer_engine_level, src_buffet_level, dst_buffet_level, tensor_name, num_ranks) \
  [](int _s, int _t...)                                                                                           \
  {                                                                                                               \
    (void) _t;                                                                                                    \
                                                                                                                  \
    std::stringstream tensor_point;                                                                               \
    tensor_point << tensor_name;                                                                                  \
                                                                                                                  \
    va_list args;                                                                                                 \
    va_start(args, _t);                                                                                           \
    for (int rank = 0; rank < num_ranks; rank++)                                                                  \
    {                                                                                                             \
      tensor_point << "_" << va_arg(args, int);                                                                   \
    }                                                                                                             \
    va_end(args);                                                                                                 \
                                                                                                                  \
    Action<float> action;                                                                                         \
    action.op = Op::VALIDATE;                                                                                     \
    action.srcs.push_back({_s, src_buffet_level, tensor_point.str(), false});                                        \
    action.transform = [](std::vector<float>& operands, std::vector<float>& results)                              \
      {                                                                                                           \
        float x = operands.at(0);                                                                                 \
        results.push_back(x);                                                                                     \
      };                                                                                                          \
    action.dsts.push_back({_s, dst_buffet_level, tensor_point.str(), false });                                       \
                                                                                                                  \
    arch[transfer_engine_level](_s).ProcessAction(action);                                                        \
  }

#define ACTION_READ_IU(transfer_engine_level, src_buffet_level, dst_buffet_level, tensor_name, num_ranks)  \
  [](int src_s, int src_t, int dst_s, int dst_t...)                                                        \
  {                                                                                                        \
    (void) src_t;                                                                                          \
    (void) dst_t;                                                                                          \
                                                                                                           \
    std::stringstream tensor_point;                                                                        \
    tensor_point << tensor_name;                                                                           \
                                                                                                           \
    va_list args;                                                                                          \
    va_start(args, dst_t);                                                                                 \
    for (int rank = 0; rank < num_ranks; rank++)                                                           \
    {                                                                                                      \
      tensor_point << "_" << va_arg(args, int);                                                            \
    }                                                                                                      \
    va_end(args);                                                                                          \
                                                                                                           \
    Action<float> action;                                                                                  \
    action.op = Op::READ;                                                                                  \
    action.srcs.push_back({ src_s, src_buffet_level, tensor_point.str(), true});                                  \
    action.transform = [](std::vector<float>& operands, std::vector<float>& results)                       \
      {                                                                                                    \
        float x = operands.at(0);                                                                          \
        results.push_back(x);                                                                              \
      };                                                                                                   \
    action.dsts.push_back({ dst_s, dst_buffet_level, tensor_point.str(), false });                                \
                                                                                                           \
    arch[transfer_engine_level](src_s).AddAction(action);                                                  \
  }

#define ACTION_READ_IU_T2(transfer_engine_level, src_buffet_level, dst_buffet_level, tensor_name, num_ranks)  \
  [](int src_s, int src_t, int dst_s, int dst_t, int dst_t1...)                                                        \
  {                                                                                                        \
    (void) src_t;                                                                                          \
    (void) dst_t;                                                                                          \
    (void) dst_t1;                                                                                          \
                                                                                                           \
    std::stringstream tensor_point;                                                                        \
    tensor_point << tensor_name;                                                                           \
                                                                                                           \
    va_list args;                                                                                          \
    va_start(args, dst_t1);                                                                                 \
    for (int rank = 0; rank < num_ranks; rank++)                                                           \
    {                                                                                                      \
      tensor_point << "_" << va_arg(args, int);                                                            \
    }                                                                                                      \
    va_end(args);                                                                                          \
                                                                                                           \
    Action<float> action;                                                                                  \
    action.op = Op::READ;                                                                                  \
    action.srcs.push_back({ src_s, src_buffet_level, tensor_point.str(), true});                                  \
    action.transform = [](std::vector<float>& operands, std::vector<float>& results)                       \
      {                                                                                                    \
        float x = operands.at(0);                                                                          \
        results.push_back(x);                                                                              \
      };                                                                                                   \
    action.dsts.push_back({ dst_s, dst_buffet_level, tensor_point.str(), false });                                \
                                                                                                           \
    arch[transfer_engine_level](src_s).AddAction(action);                                                  \
  }

#define ACTION_READ_IU_T3(transfer_engine_level, src_buffet_level, dst_buffet_level, tensor_name, num_ranks)  \
  [](int src_s, int src_t, int src_t1, int dst_s, int dst_t, int dst_t1, int dst_t2...)                                                        \
  {                                                                                                        \
    (void) src_t;                                                                                          \
    (void) src_t1;                                                                                          \
    (void) dst_t;                                                                                          \
    (void) dst_t1;                                                                                          \
    (void) dst_t2;                                                                                          \
                                                                                                           \
    std::stringstream tensor_point;                                                                        \
    tensor_point << tensor_name;                                                                           \
                                                                                                           \
    va_list args;                                                                                          \
    va_start(args, dst_t2);                                                                                 \
    for (int rank = 0; rank < num_ranks; rank++)                                                           \
    {                                                                                                      \
      tensor_point << "_" << va_arg(args, int);                                                            \
    }                                                                                                      \
    va_end(args);                                                                                          \
                                                                                                           \
    Action<float> action;                                                                                  \
    action.op = Op::READ;                                                                                  \
    action.srcs.push_back({ src_s, src_buffet_level, tensor_point.str(), true});                                  \
    action.transform = [](std::vector<float>& operands, std::vector<float>& results)                       \
      {                                                                                                    \
        float x = operands.at(0);                                                                          \
        results.push_back(x);                                                                              \
      };                                                                                                   \
    action.dsts.push_back({ dst_s, dst_buffet_level, tensor_point.str(), false });                                \
                                                                                                           \
    arch[transfer_engine_level](src_s).AddAction(action);                                                  \
  }

#define ACTION_SHRINK(transfer_engine_level, buffet_level, tensor_name, num_ranks)                         \
  [](int parent_s, int parent_t, int _s, int _t...)                                                        \
  {                                                                                                        \
    (void) parent_s;                                                                                       \
    (void) parent_t;                                                                                       \
    (void) _t;                                                                                             \
                                                                                                           \
    std::stringstream tensor_point;                                                                        \
    tensor_point << tensor_name;                                                                           \
                                                                                                           \
    va_list args;                                                                                          \
    va_start(args, _t);                                                                                    \
    for (int rank = 0; rank < num_ranks; rank++)                                                           \
    {                                                                                                      \
      tensor_point << "_" << va_arg(args, int);                                                            \
    }                                                                                                      \
    va_end(args);                                                                                          \
                                                                                                           \
    Action<float> action;                                                                                  \
    action.op = Op::SHRINK;                                                                                \
    action.srcs.push_back({_s, buffet_level, tensor_point.str(), false});                                     \
    action.transform = [](std::vector<float>& operands, std::vector<float>& results)                       \
    {                                                                                                      \
      (void) operands;                                                                                     \
      (void) results;                                                                                      \
    };                                                                                                     \
                                                                                                           \
    arch[transfer_engine_level](_s).AddAction(action);                                                     \
  }

#define ACTION_SHRINK_T2(transfer_engine_level, buffet_level, tensor_name, num_ranks)                         \
  [](int parent_s, int parent_t, int _s, int _t0, int _t1...)                                                        \
  {                                                                                                        \
    (void) parent_s;                                                                                       \
    (void) parent_t;                                                                                       \
    (void) _t0;                                                                                             \
    (void) _t1;                                                                                             \
                                                                                                           \
    std::stringstream tensor_point;                                                                        \
    tensor_point << tensor_name;                                                                           \
                                                                                                           \
    va_list args;                                                                                          \
    va_start(args, _t1);                                                                                    \
    for (int rank = 0; rank < num_ranks; rank++)                                                           \
    {                                                                                                      \
      tensor_point << "_" << va_arg(args, int);                                                            \
    }                                                                                                      \
    va_end(args);                                                                                          \
                                                                                                           \
    Action<float> action;                                                                                  \
    action.op = Op::SHRINK;                                                                                \
    action.srcs.push_back({_s, buffet_level, tensor_point.str(), false});                                     \
    action.transform = [](std::vector<float>& operands, std::vector<float>& results)                       \
    {                                                                                                      \
      (void) operands;                                                                                     \
      (void) results;                                                                                      \
    };                                                                                                     \
                                                                                                           \
    arch[transfer_engine_level](_s).AddAction(action);                                                     \
  }

#define ACTION_SHRINK_T3(transfer_engine_level, buffet_level, tensor_name, num_ranks)                         \
  [](int parent_s, int parent_t, int _s, int _t0, int _t1, int _t2...)                                                        \
  {                                                                                                        \
    (void) parent_s;                                                                                       \
    (void) parent_t;                                                                                       \
    (void) _t0;                                                                                             \
    (void) _t1;                                                                                             \
    (void) _t2;                                                                                             \
                                                                                                           \
    std::stringstream tensor_point;                                                                        \
    tensor_point << tensor_name;                                                                           \
                                                                                                           \
    va_list args;                                                                                          \
    va_start(args, _t2);                                                                                    \
    for (int rank = 0; rank < num_ranks; rank++)                                                           \
    {                                                                                                      \
      tensor_point << "_" << va_arg(args, int);                                                            \
    }                                                                                                      \
    va_end(args);                                                                                          \
                                                                                                           \
    Action<float> action;                                                                                  \
    action.op = Op::SHRINK;                                                                                \
    action.srcs.push_back({_s, buffet_level, tensor_point.str(), false});                                     \
    action.transform = [](std::vector<float>& operands, std::vector<float>& results)                       \
    {                                                                                                      \
      (void) operands;                                                                                     \
      (void) results;                                                                                      \
    };                                                                                                     \
                                                                                                           \
    arch[transfer_engine_level](_s).AddAction(action);                                                     \
  }

#define ACTION_UPDATE(transfer_engine_level, dst_buffet_level, src_buffet_level, tensor_name, num_ranks)   \
  [](int dst_s, int dst_t, int src_s, int src_t...)                                                        \
  {                                                                                                        \
    (void) src_t;                                                                                          \
    (void) dst_t;                                                                                          \
                                                                                                           \
    std::stringstream tensor_point;                                                                        \
    tensor_point << tensor_name;                                                                           \
                                                                                                           \
    va_list args;                                                                                          \
    va_start(args, src_t);                                                                                 \
    for (int rank = 0; rank < num_ranks; rank++)                                                           \
    {                                                                                                      \
      tensor_point << "_" << va_arg(args, int);                                                            \
    }                                                                                                      \
    va_end(args);                                                                                          \
                                                                                                           \
    Action<float> action;                                                                                  \
    action.op = Op::UPDATE;                                                                                \
    action.srcs.push_back({src_s, src_buffet_level, tensor_point.str(), false});                                 \
    action.transform = [](std::vector<float>& operands, std::vector<float>& results)                       \
      {                                                                                                    \
        float x = operands.at(0);                                                                          \
        results.push_back(x);                                                                              \
      };                                                                                                   \
    action.dsts.push_back({dst_s, dst_buffet_level, tensor_point.str(), false });                                \
                                                                                                           \
    arch[transfer_engine_level](src_s).AddAction(action);                                                  \
  }

#define ACTION_UPDATE_T2(transfer_engine_level, dst_buffet_level, src_buffet_level, tensor_name, num_ranks)   \
  [](int dst_s, int dst_t, int src_s, int src_t, int src_t1...)                                                        \
  {                                                                                                        \
    (void) src_t;                                                                                          \
    (void) src_t1;                                                                                          \
    (void) dst_t;                                                                                          \
                                                                                                           \
    std::stringstream tensor_point;                                                                        \
    tensor_point << tensor_name;                                                                           \
                                                                                                           \
    va_list args;                                                                                          \
    va_start(args, src_t1);                                                                                 \
    for (int rank = 0; rank < num_ranks; rank++)                                                           \
    {                                                                                                      \
      tensor_point << "_" << va_arg(args, int);                                                            \
    }                                                                                                      \
    va_end(args);                                                                                          \
                                                                                                           \
    Action<float> action;                                                                                  \
    action.op = Op::UPDATE;                                                                                \
    action.srcs.push_back({src_s, src_buffet_level, tensor_point.str(), false});                                 \
    action.transform = [](std::vector<float>& operands, std::vector<float>& results)                       \
      {                                                                                                    \
        float x = operands.at(0);                                                                          \
        results.push_back(x);                                                                              \
      };                                                                                                   \
    action.dsts.push_back({dst_s, dst_buffet_level, tensor_point.str(), false });                                \
                                                                                                           \
    arch[transfer_engine_level](src_s).AddAction(action);                                                  \
  }


#define ACTION_UPDATE_T3(transfer_engine_level, dst_buffet_level, src_buffet_level, tensor_name, num_ranks)\
  [](int dst_s, int dst_t, int dst_t1, int src_s, int src_t, int src_t1, int src_t2...)                                \
  {                                                                                                        \
    (void) src_t;                                                                                          \
    (void) src_t1;                                                                                         \
    (void) src_t2;                                                                                         \
    (void) dst_t;                                                                                          \
    (void) dst_t1;                                                                                          \
                                                                                                           \
    std::stringstream tensor_point;                                                                        \
    tensor_point << tensor_name;                                                                           \
                                                                                                           \
    va_list args;                                                                                          \
    va_start(args, src_t2);                                                                                \
    for (int rank = 0; rank < num_ranks; rank++)                                                           \
    {                                                                                                      \
      tensor_point << "_" << va_arg(args, int);                                                            \
    }                                                                                                      \
    va_end(args);                                                                                          \
                                                                                                           \
    Action<float> action;                                                                                  \
    action.op = Op::UPDATE;                                                                                \
    action.srcs.push_back({src_s, src_buffet_level, tensor_point.str(), false});                           \
    action.transform = [](std::vector<float>& operands, std::vector<float>& results)                       \
      {                                                                                                    \
        float x = operands.at(0);                                                                          \
        results.push_back(x);                                                                              \
      };                                                                                                   \
    action.dsts.push_back({dst_s, dst_buffet_level, tensor_point.str(), false });                                \
                                                                                                           \
    arch[transfer_engine_level](src_s).AddAction(action);                                                  \
  }

#define ACTION_INIT(transfer_engine_level, buffet_level, tensor_name, num_ranks)                           \
  [](int _s, int _t...)                                                                                    \
  {                                                                                                        \
    (void) _t;                                                                                             \
                                                                                                           \
    std::stringstream tensor_point; float val = 1;                                                         \
    tensor_point << tensor_name;                                                                           \
                                                                                                           \
    va_list args;                                                                                          \
    va_start(args, _t);                                                                                    \
    for (int rank = 0; rank < num_ranks; rank++)                                                           \
    {                                                                                                      \
      int x = va_arg(args, int); tensor_point << "_" << x;  val += rand() % 256;                          \
    }                                                                                                      \
    va_end(args);                                                                                          \
                                                                                                           \
    Action<float> action;                                                                                  \
    action.op = Op::INIT;                                                                                  \
    auto tpstr = tensor_point.str();                                                                       \
    action.transform = [tpstr,val](std::vector<float>& operands, std::vector<float>& results)              \
      {                                                                                                    \
        (void) operands;                                                                                   \
        /* std::hash<std::string> hasher;  */                                                              \
        float x = val;  /* static_cast<float>(hasher(tpstr))/1.0e18; */                                    \
        results.push_back(x);                                                                              \
      };                                                                                                   \
    action.dsts.push_back({_s, buffet_level, tensor_point.str(), false });                                    \
                                                                                                           \
    std::string suffix = "_fill";                                                                          \
    arch[transfer_engine_level + suffix](_s).AddAction(action);                                            \
  }

#define ACTION_INIT_ZERO(transfer_engine_level, buffet_level, tensor_name, num_ranks)                           \
  [](int _s, int _t...)                                                                                    \
  {                                                                                                        \
    (void) _t;                                                                                             \
                                                                                                           \
    std::stringstream tensor_point; float val = 1;                                                         \
    tensor_point << tensor_name;                                                                           \
                                                                                                           \
    va_list args;                                                                                          \
    va_start(args, _t);                                                                                    \
    for (int rank = 0; rank < num_ranks; rank++)                                                           \
    {                                                                                                      \
      int x = va_arg(args, int); tensor_point << "_" << x;  val  = 0;                                      \
    }                                                                                                      \
    va_end(args);                                                                                          \
                                                                                                           \
    Action<float> action;                                                                                  \
    action.op = Op::INIT;                                                                                  \
    auto tpstr = tensor_point.str();                                                                       \
    action.transform = [tpstr,val](std::vector<float>& operands, std::vector<float>& results)              \
      {                                                                                                    \
        (void) operands;                                                                                   \
        /* std::hash<std::string> hasher;  */                                                              \
        float x = val;  /* static_cast<float>(hasher(tpstr))/1.0e18; */                                    \
        results.push_back(x);                                                                              \
      };                                                                                                   \
    action.dsts.push_back({_s, buffet_level, tensor_point.str(), false });                                    \
                                                                                                           \
    std::string suffix = "_fill";                                                                          \
    arch[transfer_engine_level + suffix](_s).AddAction(action);                                            \
  }

#define ACTION_INIT_ZERO_T2(transfer_engine_level, buffet_level, tensor_name, num_ranks)                           \
  [](int _s, int _t1, int _t2...)                                                                                    \
  {                                                                                                        \
    (void) _t1;                                                                                             \
    (void) _t2;                                                                                             \
                                                                                                           \
    std::stringstream tensor_point; float val = 1;                                                         \
    tensor_point << tensor_name;                                                                           \
                                                                                                           \
    va_list args;                                                                                          \
    va_start(args, _t2);                                                                                    \
    for (int rank = 0; rank < num_ranks; rank++)                                                           \
    {                                                                                                      \
      int x = va_arg(args, int); tensor_point << "_" << x;  val  = 0;                                      \
    }                                                                                                      \
    va_end(args);                                                                                          \
                                                                                                           \
    Action<float> action;                                                                                  \
    action.op = Op::INIT;                                                                                  \
    auto tpstr = tensor_point.str();                                                                       \
    action.transform = [tpstr,val](std::vector<float>& operands, std::vector<float>& results)              \
      {                                                                                                    \
        (void) operands;                                                                                   \
        /* std::hash<std::string> hasher;  */                                                              \
        float x = val;  /* static_cast<float>(hasher(tpstr))/1.0e18; */                                    \
        results.push_back(x);                                                                              \
      };                                                                                                   \
    action.dsts.push_back({_s, buffet_level, tensor_point.str(), false });                                    \
                                                                                                           \
    std::string suffix = "_fill";                                                                          \
    arch[transfer_engine_level + suffix](_s).AddAction(action);                                            \
  }

