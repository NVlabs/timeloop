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

#include<optional>

#include <libconfig.h++>
#include <yaml-cpp/yaml.h>
#include <cassert>

namespace config
{

class CompoundConfig; // forward declaration

class CompoundConfigNode
{
 private:
  libconfig::Setting* LNode = nullptr;
  YAML::Node YNode;
  CompoundConfig* cConfig = nullptr;

 public:
  CompoundConfigNode(){}
  CompoundConfigNode(libconfig::Setting* _lnode, YAML::Node _ynode);
  CompoundConfigNode(libconfig::Setting* _lnode, YAML::Node _ynode, CompoundConfig* _cConfig);

  libconfig::Setting& getLNode() {return *LNode;}
  YAML::Node getYNode() {return YNode;}

  /*!
   * @brief return compound config node corresponding with `path`.
   */
  CompoundConfigNode lookup(const char *path) const;
  inline CompoundConfigNode lookup(const std::string &path) const
  { return(lookup(path.c_str())); }

  bool lookupValue(const char *name, bool &value) const;
  bool lookupValue(const char *name, int &value) const;
  bool lookupValue(const char *name, unsigned int &value) const;
  bool lookupValueLongOnly(const char *name, long long &value) const; // Only for values with an L like 123L
  bool lookupValueLongOnly(const char *name, unsigned long long &value) const; // Only for values with an L like 123L
  bool lookupValue(const char *name, long long &value) const;
  bool lookupValue(const char *name, unsigned long long &value) const;
  bool lookupValue(const char *name, double &value) const;
  bool lookupValue(const char *name, float &value) const;
  bool lookupValue(const char *name, const char *&value) const;
  bool lookupValue(const char *name, std::string &value) const;
  
  /// @brief Null type setter at name.
  bool setNull(const char *name);
  /// @brief Scalar setter at name (template).
  template <typename T>
  bool setScalar(const char *name, const T value);
  /// @brief Creates/appends to sequence at name (template).
  template <typename T>
  bool push_back(const T value);

  inline bool lookupValue(const std::string &name, bool &value) const
  { return(lookupValue(name.c_str(), value)); }

  inline bool lookupValue(const std::string &name, int &value) const
  { return(lookupValue(name.c_str(), value)); }

  inline bool lookupValue(const std::string &name, unsigned int &value) const
  { return(lookupValue(name.c_str(), value)); }

  inline bool lookupValue(const std::string &name, long long &value) const
  { return(lookupValue(name.c_str(), value)); }

  inline bool lookupValue(const std::string &name,
                          unsigned long long &value) const
  { return(lookupValue(name.c_str(), value)); }

  inline bool lookupValue(const std::string &name, double &value) const
  { return(lookupValue(name.c_str(), value)); }

  inline bool lookupValue(const std::string &name, float &value) const
  { return(lookupValue(name.c_str(), value)); }

  inline bool lookupValue(const std::string &name, const char *&value) const
  { return(lookupValue(name.c_str(), value)); }

  inline bool lookupValue(const std::string &name, std::string &value) const
  { return(lookupValue(name.c_str(), value)); }

  inline bool setNull(const std::string &name)
  { return setNull(name.c_str()); }

  template <typename T>
  inline bool setScalar(const std::string &name, const T value)
  { return setScalar<T>(name.c_str(), value); }

  bool exists(const char *name) const;

  inline bool exists(const std::string &name) const
  { return(exists(name.c_str())); }

  bool lookupArrayValue(const char* name, std::vector<std::string> &vectorValue) const;

  inline bool lookupArrayValue(const std::string &name, std::vector<std::string> &vectorValue) const
  { return(lookupArrayValue(name.c_str(), vectorValue));}

  bool isList() const;
  bool isArray() const;
  bool isMap() const;
  int getLength() const;

  CompoundConfigNode operator [](int idx) const;

  bool getArrayValue(std::vector<std::string> &vectorValue) const;
  // iterate through all maps and get the keys within a node
  bool getMapKeys(std::vector<std::string> &mapKeys) const;
};

class CompoundConfig
{
 private:
  bool useLConfig;
  libconfig::Config LConfig;
  YAML::Node YConfig;
  CompoundConfigNode root;
  CompoundConfigNode variableRoot;

 public:
  CompoundConfig(){assert(false);}
  CompoundConfig(const char* inputFile);
  CompoundConfig(char* inputFile) : CompoundConfig((const char*) inputFile) {}
  CompoundConfig(std::vector<std::string> inputFiles);
  CompoundConfig(std::string input, std::string format); // yaml file given as string

  ~CompoundConfig(){}

  libconfig::Config& getLConfig();
  YAML::Node& getYConfig();
  CompoundConfigNode getRoot() const;
  CompoundConfigNode getVariableRoot() const;

  bool hasLConfig() { return useLConfig;}

  std::vector<std::string> inFiles;

};

  std::uint64_t parseElementSize(std::string name);
  std::string parseName(std::string name);

} // namespace config
