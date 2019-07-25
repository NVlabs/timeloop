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

#include "compound-config.hpp"

namespace config
{

/* CompoundConfigNode */

CompoundConfigNode::CompoundConfigNode(libconfig::Setting* _lnode, YAML::Node* _ynode) {
  LNode = _lnode;
  YNode = _ynode;
}

CompoundConfigNode CompoundConfigNode::lookup(const char *path) const {
  if (LNode) {
    libconfig::Setting& nextNode = LNode->lookup(path);
    return CompoundConfigNode(&nextNode, nullptr);
  } else {
    assert(false);
  }
}


bool CompoundConfigNode::lookupValue(const char *name, bool &value) const {
  if (LNode) return LNode->lookupValue(name, value);
  else {
    assert(false);
    return false;
  }
}

bool CompoundConfigNode::lookupValue(const char *name, int &value) const {
  if (LNode) return LNode->lookupValue(name, value);
  else {
    assert(false);
    return false;
  }
}

bool CompoundConfigNode::lookupValue(const char *name, unsigned int &value) const {
  if (LNode) return LNode->lookupValue(name, value);
  else {
    assert(false);
    return false;
  }
}

bool CompoundConfigNode::lookupValue(const char *name, long long &value) const {
  if (LNode) return LNode->lookupValue(name, value);
  else {
    assert(false);
    return false;
  }
}

bool CompoundConfigNode::lookupValue(const char *name, unsigned long long &value) const {
  if (LNode) return LNode->lookupValue(name, value);
  else {
    assert(false);
    return false;
  }
}

bool CompoundConfigNode::lookupValue(const char *name, double &value) const {
  if (LNode) return LNode->lookupValue(name, value);
  else {
    assert(false);
    return false;
  }
}

bool CompoundConfigNode::lookupValue(const char *name, float &value) const {
  if (LNode) return LNode->lookupValue(name, value);
  else {
    assert(false);
    return false;
  }
}

bool CompoundConfigNode::lookupValue(const char *name, const char *&value) const {
  if (LNode) return LNode->lookupValue(name, value);
  else {
    assert(false);
    return false;
  }
}

bool CompoundConfigNode::lookupValue(const char *name, std::string &value) const {
  if (LNode) return LNode->lookupValue(name, value);
  else {
    assert(false);
    return false;
  }
}

bool CompoundConfigNode::exists(const char *name) const {
  if (LNode) return LNode->exists(name);
  else {
    assert(false);
    return false;
  }

}

bool CompoundConfigNode::lookupArrayValue(const char* name, std::vector<std::string> &vectorValue) const {
  if (LNode) {
    assert(LNode->lookup(name).isArray());
    for (const std::string& m: LNode->lookup(name))
    {
      vectorValue.push_back(m);
    }
    return true;
  } else {
    assert(false);
    return false;
  }
}

bool CompoundConfigNode::isList() const {
  if(LNode) return LNode->isList();
  else {
    assert(false);
    return false;
  }
}

bool CompoundConfigNode::isArray() const {
  if(LNode) return LNode->isArray();
  else {
    assert(false);
    return false;
  }
}

int CompoundConfigNode::getLength() const {
  if(LNode) return LNode->getLength();
  else {
    assert(false);
    return false;
  }
}

CompoundConfigNode CompoundConfigNode::operator [](int idx) const {
  assert(isList() || isArray());
  if(LNode) return CompoundConfigNode(&(*LNode)[idx], nullptr);
  else {
    assert(false);
    return CompoundConfigNode(nullptr, nullptr);
  }
}

bool CompoundConfigNode::getArrayValue(std::vector<std::string> &vectorValue) {
  if (LNode) {
    for (const std::string& m: *LNode)
    {
      vectorValue.push_back(m);
    }
    return true;
  } else {
    assert(false);
    return false;
  }
}

/* CompoundConfig */

CompoundConfig::CompoundConfig(const char* inputFile) {
  //FIXME: parse the input to decide which format it is
  LConfig.readFile(inputFile);
  auto& lroot = LConfig.getRoot();
  useLConfig = true;
  root = CompoundConfigNode(&lroot, nullptr);
}

libconfig::Config& CompoundConfig::getLConfig() {
  return LConfig;
}

YAML::Node& CompoundConfig::getYConfig() {
  return YConfig;
}

CompoundConfigNode CompoundConfig::getRoot() const {
  return root;
}

} // namespace config
