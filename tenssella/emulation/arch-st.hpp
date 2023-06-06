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

#include <map>
#include <string>

template<typename T>
class Buffet
{
 private:
  std::string name_;
  std::map<std::string, bool> lock_;
  std::map<std::string, T> contents_;

 public:
  Buffet() {}
  Buffet(std::string name) : name_(name) {}

  T& fill(std::string addr)
  {
    return contents_[addr]; // note: allocating if not present.
  }

  const T& read_iu(std::string addr)
  {
    auto lock_it = lock_.find(addr);
    if (lock_it != lock_.end() && lock_it->second)
    {
      std::cerr << "ERROR: buffet " << name_ << ": attempt to read_iu locked location: " << addr << std::endl;
      exit(1);
    }
    lock_[addr] = true;
    return contents_.at(addr);
  }

  const T& read(std::string addr) const
  {
    auto lock_it = lock_.find(addr);
    if (lock_it != lock_.end() && lock_it->second)
    {
      std::cerr << "ERROR: buffet " << name_ << ": attempt to read locked location: " << addr << std::endl;
      exit(1);
    }
    return contents_.at(addr);
  }

  const T& drain(std::string addr) const
  {
    return read(addr);
  }

  T& update(std::string addr)
  {
    auto lock_it = lock_.find(addr);
    if (lock_it != lock_.end() && !lock_it->second)
    {
      std::cerr << "ERROR: buffet " << name_ << ": attempt to update unlocked location: " << addr << std::endl;
      exit(1);
    }
    lock_[addr] = false;
    return contents_.at(addr);
  }
};

template<typename T>
class PhysicalLevel
{
 private:
  std::string name_ = "";
  std::map<int, Buffet<T>> buffets_;

 public:
  PhysicalLevel() {}
  PhysicalLevel(std::string name) : name_(name) {}

  Buffet<T>& operator [](int space_coord)
  {
    auto it = buffets_.find(space_coord);
    if (it == buffets_.end())
    {
      char buffet_name[256];
      sprintf(buffet_name, "%s[%d]", name_.c_str(), space_coord);
      buffets_[space_coord] = Buffet<T>(std::string(buffet_name));
      return buffets_[space_coord];
    }
    else
    {
      return it->second;
    }
    // return buffets_[space_coord];
  }
};

template<typename T>
class Arch
{
 private:
  std::map<std::string, PhysicalLevel<T>> levels_;

 public:
  PhysicalLevel<T>& operator [](std::string level_name)
  {
    auto it = levels_.find(level_name);
    if (it == levels_.end())
    {
      levels_[level_name] = PhysicalLevel<T>(level_name);
      return levels_[level_name];
    }
    else
    {
      return it->second;
    }
    // return levels_[level_name];
  }

  // const PhysicalLevel<T>& at(std::string level_name) const
  // {
  //   return levels_.at(level_name);
  // }
};
