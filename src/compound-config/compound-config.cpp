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

#include <iostream>
#include <fstream>
#include <cstring>

#define EXCEPTION_PROLOGUE                                                          \
    try { 

#define EXCEPTION_EPILOGUE                                                          \
    }                                                                               \
    catch (const libconfig::SettingTypeException& e)                                \
    {                                                                               \
      std::cerr << "ERROR: setting type exception at: " << e.getPath() << std::endl;\
      exit(1);                                                                      \
    }                                                                               \
    catch (const libconfig::SettingNotFoundException& e)                            \
    {                                                                               \
      std::cerr << "ERROR: setting not found: " << e.getPath() << std::endl;        \
      exit(1);                                                                      \
    }                                                                               \
    catch (const libconfig::SettingNameException& e)                                \
    {                                                                               \
      std::cerr << "ERROR: setting name exception at: " << e.getPath() << std::endl;\
      exit(1);                                                                      \
    }                                                                               
    /*catch (YAML::Exception e)                                                       \
    {                                                                               \
      std::cerr << "ERROR: YAML exception: " << e.msg << std::endl;                 \
      exit(1);                                                                      \
    } */

namespace config
{

/* CompoundConfigNode */

CompoundConfigNode::CompoundConfigNode(libconfig::Setting* _lnode, YAML::Node _ynode) {
  LNode = _lnode;
  YNode = _ynode;
}

CompoundConfigNode CompoundConfigNode::lookup(const char *path) const {
  EXCEPTION_PROLOGUE;
  if (LNode) {
    libconfig::Setting& nextNode = LNode->lookup(path);
    return CompoundConfigNode(&nextNode, YAML::Node());
  } else if (YNode) {
    YAML::Node nextNode = YNode[path];
    return CompoundConfigNode(nullptr, nextNode);
  } else {
    assert(false);
  }
  EXCEPTION_EPILOGUE;
}


bool CompoundConfigNode::lookupValue(const char *name, bool &value) const {
  EXCEPTION_PROLOGUE;
  if (LNode) return LNode->lookupValue(name, value);
  else if (YNode) {
    if (YNode.IsScalar() || !YNode[name].IsDefined()) return false;
    value = YNode[name].as<bool>();
    return true;
  }
  else {
    assert(false);
    return false;
  }
  EXCEPTION_EPILOGUE;
}

bool CompoundConfigNode::lookupValue(const char *name, int &value) const {
  EXCEPTION_PROLOGUE;
  if (LNode) return LNode->lookupValue(name, value);
  else if (YNode) {
    if (YNode.IsScalar() || !YNode[name].IsDefined()) return false;
    value = YNode[name].as<int>();
    return true;
  }
  else {
    assert(false);
    return false;
  }
  EXCEPTION_EPILOGUE;
}

bool CompoundConfigNode::lookupValue(const char *name, unsigned int &value) const {
  EXCEPTION_PROLOGUE;
  if (LNode) return LNode->lookupValue(name, value);
  else if (YNode) {
    if (YNode.IsScalar() || !YNode[name].IsDefined()) return false;
    value = YNode[name].as<unsigned int>();
    return true;
  }
  else {
    assert(false);
    return false;
  }
  EXCEPTION_EPILOGUE;

}

bool CompoundConfigNode::lookupValue(const char *name, long long &value) const {
  EXCEPTION_PROLOGUE;
  if (LNode) return LNode->lookupValue(name, value);
  else if (YNode) {
    if (YNode.IsScalar() || !YNode[name].IsDefined()) return false;
    value = YNode[name].as<long long>();
    return true;
  }
  else {
    assert(false);
    return false;
  }
  EXCEPTION_EPILOGUE;
}

bool CompoundConfigNode::lookupValue(const char *name, unsigned long long &value) const {
  EXCEPTION_PROLOGUE;
  if (LNode) return LNode->lookupValue(name, value);
  else if (YNode) {
    if (YNode.IsScalar() || !YNode[name].IsDefined()) return false;
    value = YNode[name].as<unsigned long long>();
    return true;
  }
  else {
    assert(false);
    return false;
  }
  EXCEPTION_EPILOGUE;
}

bool CompoundConfigNode::lookupValue(const char *name, double &value) const {
  EXCEPTION_PROLOGUE;
  if (LNode) return LNode->lookupValue(name, value);
  else if (YNode) {
    if (YNode.IsScalar() || !YNode[name].IsDefined()) return false;
    value = YNode[name].as<double>();
    return true;
  }
  else {
    assert(false);
    return false;
  }
  EXCEPTION_EPILOGUE;
}

bool CompoundConfigNode::lookupValue(const char *name, float &value) const {
  EXCEPTION_PROLOGUE;
  if (LNode) return LNode->lookupValue(name, value);
  else if (YNode) {
    if (YNode.IsScalar() || !YNode[name].IsDefined()) return false;
    value = YNode[name].as<float>();
    return true;
  }
  else {
    assert(false);
    return false;
  }
  EXCEPTION_EPILOGUE;
}

bool CompoundConfigNode::lookupValue(const char *name, const char *&value) const {
  EXCEPTION_PROLOGUE;
  if (LNode) return LNode->lookupValue(name, value);
  else if (YNode) {
    if (YNode.IsScalar() || !YNode[name].IsDefined()) return false;
    value = YNode[name].as<std::string>().c_str();
    return true;
  }
  else {
    assert(false);
    return false;
  }
  EXCEPTION_EPILOGUE;
}

bool CompoundConfigNode::lookupValue(const char *name, std::string &value) const {
  EXCEPTION_PROLOGUE;
  if (LNode) return LNode->lookupValue(name, value);
  else if (YNode) {
    if (YNode.IsScalar() || !YNode[name].IsDefined()) return false;
    value = YNode[name].as<std::string>();
    return true;
  }
  else {
    assert(false);
    return false;
  }
  EXCEPTION_EPILOGUE;
}

bool CompoundConfigNode::exists(const char *name) const {
  EXCEPTION_PROLOGUE;
  if (LNode) return LNode->exists(name);
  else if (YNode) return !YNode.IsScalar() && YNode[name].IsDefined();
  else {
    assert(false);
    return false;
  }
  EXCEPTION_EPILOGUE;
}

bool CompoundConfigNode::lookupArrayValue(const char* name, std::vector<std::string> &vectorValue) const {
  EXCEPTION_PROLOGUE;

  if (LNode) {
    assert(LNode->lookup(name).isArray());
    for (const std::string& m: LNode->lookup(name))
    {
      vectorValue.push_back(m);
    }
    return true;
  } else if (YNode) {
    assert(YNode[name].IsSequence());
    for (auto n : YNode[name]) {
      vectorValue.push_back(n.as<std::string>());
    }
    return true;
  } else {
    assert(false);
    return false;
  }
  EXCEPTION_EPILOGUE;
}

bool CompoundConfigNode::isList() const {
  if(LNode) return LNode->isList();
  else if (YNode) return YNode.IsSequence() && !YNode[0].IsScalar();
  else {
    assert(false);
    return false;
  }
}

bool CompoundConfigNode::isArray() const {
  if(LNode) return LNode->isArray();
  else if (YNode) return YNode.IsSequence() && YNode[0].IsScalar();
  else {
    assert(false);
    return false;
  }
}

int CompoundConfigNode::getLength() const {
  if(LNode) return LNode->getLength();
  else if (YNode) return YNode.size();
  else {
    assert(false);
    return false;
  }
}

CompoundConfigNode CompoundConfigNode::operator [](int idx) const {
  assert(isList() || isArray());
  if(LNode) return CompoundConfigNode(&(*LNode)[idx], YAML::Node());
  else if (YNode) {
      auto yIter = YNode.begin();
      for (int i = 0; i < idx; i++) yIter++;
      auto nextNode = *yIter;
      return CompoundConfigNode(nullptr, nextNode);
  }
  else {
    assert(false);
    return CompoundConfigNode(nullptr, YAML::Node());
  }
}

bool CompoundConfigNode::getArrayValue(std::vector<std::string> &vectorValue) {
  if (LNode) {
    assert(isArray());
    for (const std::string& m: *LNode)
    {
      vectorValue.push_back(m);
    }
    return true;
  } else if (YNode) {
    assert(isArray());
    for (auto n : YNode)
    {
      vectorValue.push_back(n.as<std::string>());
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
  if (std::strstr(inputFile, ".cfg")) {
    LConfig.readFile(inputFile);
    auto& lroot = LConfig.getRoot();
    useLConfig = true;
    root = CompoundConfigNode(&lroot, YAML::Node());
  } else if (std::strstr(inputFile, ".yml") || std::strstr(inputFile, ".yaml")) {
    std::ifstream f;
    f.open(inputFile);
    YConfig = YAML::Load(f);
    root = CompoundConfigNode(nullptr, YConfig);
    useLConfig = false;
    std::cout << YConfig << std::endl;
  } else {
    std::cerr << "ERROR: Input configuration file does not end with .cfg, .yml, or .yaml" << std::endl;
    exit(1);
  }
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
