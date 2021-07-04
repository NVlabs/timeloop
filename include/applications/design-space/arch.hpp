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

#pragma once

#include <fstream>
#include <iomanip>
#include <string>
#include <sstream>
#include <vector>

#include "compound-config/compound-config.hpp"

class ArchSweepNode
{
 public:
  std::string name_;
  int val_curr_;   
  int val_min_;
  int val_max_;
  int val_step_size_;

  ArchSweepNode();
  ArchSweepNode(std::string n, int min, int max, int step);
};


class ArchSpaceNode
{
 public:
  std::string name_; //descriptive name
  YAML::Node yaml_; //text version of YAML/ should be YAML

  ArchSpaceNode();
  ArchSpaceNode(std::string n, YAML::Node a);
};


class ArchSpace
{
 protected:
  std::string name_;
  std::vector<ArchSpaceNode> architectures_;

 public:
  ArchSpace();
  ArchSpace(std::string n);

  void InitializeFromFile(std::string filename);
  void InitializeFromFileList(YAML::Node list_yaml);
  void InitializeFromFileSweep(YAML::Node sweep_yaml);

  int GetSize();

  ArchSpaceNode& GetNode(int index);
};


