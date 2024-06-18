/* Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#include <deque>
#include <queue>
#include <mutex>
#include <cassert>
#include <vector>
#include <functional>
#include "../utils.hpp"

enum class Op
{
  INIT, READ, MULTICAST, SHRINK, UPDATE, COMPUTE, VALIDATE
};

std::map<Op, std::string> OpName =
{
  { Op::INIT, "INIT" },
  { Op::READ, "READ" },
  { Op::MULTICAST, "MULTICAST" },
  { Op::SHRINK, "SHRINK" },
  { Op::UPDATE, "UPDATE" },
  { Op::COMPUTE, "COMPUTE" },
  { Op::VALIDATE, "VALIDATE" }
};

struct TensorAccessDescriptor
{
  //Spatial factor for the tensor
  int space_id;
  std::string instance_name;
  std::string tensor_point;
  bool iu;
};

template<typename T>
struct Action
{
  Op op;

  std::vector<TensorAccessDescriptor> srcs;
  std::function<void(std::vector<T>&, std::vector<T>&)> transform;
  std::vector<TensorAccessDescriptor> dsts;

  std::string ToString()
  {
    std::stringstream out;
    out << OpName.at(op) << ": ";
    for (auto& desc: srcs)
    {
      out << desc.instance_name << "[" << desc.space_id << "][" << desc.tensor_point << "] ";
    }
    out << "--> ";
    for (auto& desc: dsts)
    {
      out << desc.instance_name << "[" << desc.space_id << "][" << desc.tensor_point << "] ";
    }
    return out.str();
  }

  size_t GetLatency() {
    if (op == Op::INIT)
        return 0;
    else if (op == Op::COMPUTE) {
        return 1;
    }

    //DRAM latency
    for (auto& desc: srcs) {
      std::string ins_name = desc.instance_name;
      if (ins_name.find("DRAM") != std::string::npos)
        return 10;
    }
    for (auto& desc: dsts) {
      std::string ins_name = desc.instance_name;
      if (ins_name.find("DRAM") != std::string::npos)
        return 10;
    }

    //Regular onchip memory interconnect
    return 1;
  }

};

template<typename T>
class TransferEngine
{
 private:
  enum class State { Idle, Running };

  std::string name_;
  State state_ = State::Idle;
  std::thread run_thread_;
  std::deque<Action<T>> action_queue_;

  //Use to trace time
  size_t time_stamp;

 public:
  TransferEngine() {time_stamp = 0;}

  TransferEngine(std::string name) : name_(name) {time_stamp = 0;}

  void AddAction(Action<T>& action)
  {
    assert(state_ == State::Idle);
    action_queue_.push_back(action);
  }

  void ProcessAction(Action<T>& action)
  {
    std::thread::id tid = std::this_thread::get_id();

    TRACE_LOCK(1);
    TRACE(1) << "[" << std::hex << tid << std::dec << "]  engine " << name_ <<
        " (Issue) " << "  (timestamp = " << time_stamp << ")"  << std::endl <<
        tab(8) << "  actions: " << action.ToString() << std::endl;
    TRACE_UNLOCK(1);

    std::vector<T> operands;
    std::vector<T> results;

    //init Src time to be the transfer engine issue time
    size_t src_t = time_stamp;

    // Read srcs.
    for (auto& desc: action.srcs)
    {
      switch (action.op)
      {
        case Op::READ:
        case Op::COMPUTE:
        case Op::VALIDATE:
        case Op::MULTICAST:
        {
          T val = desc.iu ?
            (*arch_<T>)[desc.instance_name][desc.space_id].read_iu(desc.tensor_point) :
            (*arch_<T>)[desc.instance_name][desc.space_id].read(desc.tensor_point);
          operands.push_back(val);
          src_t = std::max(src_t,
                  (*arch_<T>)[desc.instance_name][desc.space_id].getTimeStamp(desc.tensor_point));
          break;
        }

        case Op::SHRINK:
        {
          (*arch_<T>)[desc.instance_name][desc.space_id].shrink(desc.tensor_point);
          src_t = std::max(src_t,
                  (*arch_<T>)[desc.instance_name][desc.space_id].getTimeStamp(desc.tensor_point));
          break;
        }

        case Op::UPDATE:
        {
          T val = (*arch_<T>)[desc.instance_name][desc.space_id].drain(desc.tensor_point);
          operands.push_back(val);
          src_t = std::max(src_t,
                  (*arch_<T>)[desc.instance_name][desc.space_id].getTimeStamp(desc.tensor_point));
          break;
        }

        case Op::INIT:
        default:
        {
          std::cerr << "ERROR: invalid opcode for operand." << std::endl;
          exit(1);
        }
      }
    }

    TRACE_LOCK(1);
    TRACE(1) << "[" << std::hex << tid << std::dec << "]  engine " << name_ << std::endl
        << tab(8) << "  action: " << action.ToString() << " (T start = " << src_t << ")" << std::endl;
    TRACE_UNLOCK(1);

    // Perform transformation.
    action.transform(operands, results);

    //Calculate the latency
    size_t latency = action.GetLatency();
    size_t dst_t = src_t + latency;

    // Write dsts.
    assert(results.size() == action.dsts.size() || action.op == Op::MULTICAST);

    for (std::size_t i = 0; i < results.size(); i++)
    {
      T val = results.at(i);

      switch (action.op)
      {
        case Op::MULTICAST:
        {
          //broadcast to all consumers
          for (auto & desc : action.dsts) {
            (*arch_<T>)[desc.instance_name][desc.space_id].fill(desc.tensor_point, val);
            (*arch_<T>)[desc.instance_name][desc.space_id].setTimeStamp(desc.tensor_point, dst_t);
          }
          break;
        }
        case Op::INIT:
        case Op::READ:
        {
          auto& desc = action.dsts.at(i);
          (*arch_<T>)[desc.instance_name][desc.space_id].fill(desc.tensor_point, val);
          (*arch_<T>)[desc.instance_name][desc.space_id].setTimeStamp(desc.tensor_point, dst_t);
          break;
        }

        case Op::COMPUTE:
        case Op::UPDATE:
        {
          auto& desc = action.dsts.at(i);
          (*arch_<T>)[desc.instance_name][desc.space_id].update(desc.tensor_point, val);
          (*arch_<T>)[desc.instance_name][desc.space_id].setTimeStamp(desc.tensor_point, dst_t);
          break;
        }

        case Op::VALIDATE:
        {
          auto& desc = action.dsts.at(i);
          (*arch_<T>)[desc.instance_name][desc.space_id].validate(desc.tensor_point, val);
          break;
        }

        case Op::SHRINK:
        default:
        {
          std::cerr << "ERROR: invalid opcode for operand." << std::endl;
          exit(1);
        }
      }
    }

    //FIXME: we should support a plug in architecture model
    //Support outstanding issue?
    //if ((action.op != Op::INIT) && (action.op != Op::SHRINK) && (action.op != Op::UPDATE))
    //  time_stamp = src_t + TRANSFER_ENGINE_II;
    if (action.op == Op::READ || action.op == Op::MULTICAST || action.op == Op::UPDATE) {
        time_stamp = src_t + 0.25; //FIXME: issue rate 4
    }
    if (action.op == Op::COMPUTE) {
        time_stamp = src_t + 1;
    }

    TRACE_LOCK(1);
    TRACE(1) << "[" << std::hex << tid << std::dec << "]  engine " << name_
            << " (next_issue) (time stamp = " << time_stamp << ")" << std::endl
            << tab(8) << "  action:" << action.ToString() << " (T end = " << dst_t << ")" << std::endl;
    TRACE_UNLOCK(1);
  }

  void RunThread()
  {
    assert(state_ == State::Running);
    while (action_queue_.size() != 0)
    {
      auto& action = action_queue_.front();
      ProcessAction(action);
      action_queue_.pop_front();
    }
  }

  Action<T> CreateMulticastAction(std::queue<Action<T>> & to_be_merged) {

      Action<T> action;
    action.op = Op::MULTICAST;

    //Add the only src
    auto top_a = to_be_merged.front();
    assert(top_a.srcs.size() == 1);
    for (auto src: top_a.srcs)
      action.srcs.push_back(src);

    action.transform = top_a.transform;

    //Add all the dsts
    int cnt = 0;
    while(to_be_merged.size()) {
      auto merge_act = to_be_merged.front();
      assert(merge_act.dsts.size() == 1);
      action.dsts.push_back(pick(merge_act.dsts));

      //TRACE(2) << "Optimization\n\t ==> [" + str(cnt) + "]"
      //    << " Tobe merged: " << action.ToString() << std::endl;

      to_be_merged.pop();
      cnt ++;
    }
    //TRACE(2) << std::endl;

    return action;
  }

  void CreateMulticast(std::queue<Action<T>> & to_be_merged,
          std::deque<Action<T>> & opt_action_queue) {
    if (to_be_merged.size() == 0) {
      //chances are that nothing need to be merged
      return;
    } else if (to_be_merged.size() == 1) {
      //no merging, just move to the newly created queue
      opt_action_queue.push_back(to_be_merged.front());
      //TRACE(1) << "\tPush read action: " << to_be_merged.front().ToString() << std::endl;
      to_be_merged.pop();
    } else {
      opt_action_queue.push_back(CreateMulticastAction(to_be_merged));
      //TRACE(1) << "\tPush MULTICAST action: " << opt_action_queue.back().ToString() << std::endl;
    }
  }

  //FIXME: This is a hack
  // We go through all the actions,
  // finding a subsequence reading the same location
  // and merge them into one src multi destination action,
  // AKA multi-cast (Is this broadcast)?
  void MergeActionsIntoMulticast() {
    std::queue<Action<T>> to_be_merged;
    std::deque<Action<T>> opt_action_queue;
    for (auto action: action_queue_) {
      //TRACE(1) << "Get action: " << action.ToString() << std::endl;
      if (action.op == Op::READ) {
        if (to_be_merged.size()) {

          auto loc = pick(to_be_merged.back().srcs).tensor_point;
          auto next_loc = pick(action.srcs).tensor_point;

          if (next_loc == loc) {
            to_be_merged.push(action);
          } else {
            CreateMulticast(to_be_merged, opt_action_queue);
            to_be_merged.push(action);
          }

        } else {
          to_be_merged.push(action);
        }
      //Other operand
      } else {
        //Merge the current read queue
        CreateMulticast(to_be_merged, opt_action_queue);
        //Push the following operand
        opt_action_queue.push_back(action);
        //TRACE(1) << "\tPush other action: " << action.ToString() << std::endl;
      }
    }
    //Handle the Tail
    CreateMulticast(to_be_merged, opt_action_queue);
    action_queue_ = opt_action_queue;
  }

  void Optimizations() {
    MergeActionsIntoMulticast();
  }

  void Run()
  {
    assert(state_ == State::Idle);
    state_ = State::Running;
    run_thread_ = std::thread(&TransferEngine::RunThread, this);
  }

  void Wait()
  {
    assert(state_ == State::Running);
    run_thread_.join();
    state_ = State::Idle;
  }
};
