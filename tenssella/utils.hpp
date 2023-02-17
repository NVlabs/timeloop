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

#include <regex>
#include <assert.h>
#include <stdlib.h>
#include <iostream>
#include <sstream>
#include <memory>

#include <vector>
#include <map>
#include <set>
#include <string>
#include <list>
#include <unordered_set>
#include <unordered_map>

using std::vector;
using std::list;
using std::pair;
using std::map;
using std::unordered_map;
using std::string;
using std::set;
using std::unordered_set;

using std::shared_ptr;
using std::make_shared;


template<typename A, typename B>
static inline
pair<A, B> pick(const unordered_map<A, B>& s) {
  assert(s.size() > 0);
  return *(begin(s));
}

template<typename A, typename B>
static inline
pair<A, B> pick(const map<A, B>& s) {
  assert(s.size() > 0);
  return *(begin(s));
}

template<typename T>
static inline
T pick(const std::vector<T>& s) {
  assert(s.size() > 0);
  return *(begin(s));
}

template<typename T>
static inline
T pick(const std::set<T>& s) {
  assert(s.size() > 0);
  return *(begin(s));
}

template<typename A>
std::set<A>
intersection(const std::set<A>& l, const std::set<A>& r) {
  std::set<A> it;
  set_intersection(std::begin(l), std::end(l),
  	     std::begin(r), std::end(r),
  	     std::inserter(it, std::end(it)));
  return it;
}

static inline
std::vector<std::string> split_at(const std::string& t, const std::string& delimiter) {
  std::string s = t;
  size_t pos = 0;
  std::string token;
  std::vector<std::string> tokens;
  while ((pos = s.find(delimiter)) != std::string::npos) {
    token = s.substr(0, pos);
    tokens.push_back(token);
    s.erase(0, pos + delimiter.length());
  }

  tokens.push_back(s);

  return tokens;
}

std::vector<std::string> parse_aff(const std::string& aff);

std::map<std::string, int> ParseFactorLine(std::string line);

static inline
const char* str(const std::string & s) {
  return s.c_str();
}


static inline
string str(const int i) {
  return std::to_string(i);
}

static inline
int ReduceFactors(const std::map<std::string, int>& factors)
{
  int retval = 1;
  for (auto& factor: factors)
    retval *= factor.second;
  return retval;
}


static inline
int cmd(const std::string& cm) {
  std::cout << "cmd: " << cm << std::endl;
  return system(cm.c_str());
}


// Inputs:
//   factors: "C" -> 32
//            "K" -> 16
//            "R" -> 3
//   permutation: <"K", "R", "C"> (big-endian)
// Output:
//   "(k*R+r)*C+c"
static inline
std::string Flatten(const std::vector<std::string>& permutation, const std::map<std::string, int>& factors)
{
  // Ugh... if the permutation is null, return a hard-coded expression "0".
  if (permutation.empty())
    return "0";

  std::string flattening;
  for (auto& var: permutation)
  {
    int factor = factors.at(var);
    if (flattening.empty())
    {
      flattening += var;
    }
    else
    {
      flattening = "(" + flattening + ")";
      flattening += "*" + std::to_string(factor) + " + " + var;
    }
  }
  return flattening;
}

static inline
string tab(const int n) {
  string t = "";
  for (int i = 0; i < n; i++) {
    t += "  ";
  }
  return t;
}

static inline
std::string sep_list(const std::vector<std::string>& strs, const std::string& ldelim, const std::string& rdelim, const std::string& sep) {

  //cout << "Starting sep list" << endl;
  string res = ldelim;

  if (strs.size() > 0) {
    for (size_t i = 0; i < strs.size(); i++) {
      res += strs[i];
      if (strs.size() > 1 && i < strs.size() - 1) {
        res += sep;
      }
    }
  }
  res += rdelim;

  //cout << "Done with sep list" << endl;

  return res;
}


template<typename T>
static inline
std::string sep_list(const std::vector<T>& vals, const std::string& ldelim, const std::string& rdelim, const std::string& sep) {
  vector<string> strs;
  for (auto v : vals) {
      std::ostringstream ss;
    ss << v;
    strs.push_back(ss.str());
  }
  return sep_list(strs, ldelim, rdelim, sep);
}

template<typename T>
static inline
std::ostream& operator<<(std::ostream& out, vector<T>& v) {
  out << sep_list(v, "{", "}", ", ");
  return out;
}

template<typename T>
static inline
std::ostream& operator<<(std::ostream& out, std::set<T>& v) {
  vector<T> vv(v.begin(), v.end());
  out << sep_list(vv, "{", "}", ", ");
  return out;
}

template <typename T>
static inline
std::ostream& operator<< (std::ostream& out, const std::map<string, T>& m) {
    if ( !m.empty()  ) {
      for (const auto &p : m)
      {
        out << p.first << ": ";
        out << p.second << ' ';
        out << std::endl;
      }

    }
    return out;
}


// Point-wise multiply.
template<typename KEY, typename VALUE>
void PointwiseMultiplyUpdate(const std::map<KEY, VALUE>& a, const std::map<KEY, VALUE>& b,
                             std::map<KEY, VALUE>& z)
{
  // Note that a and b may have disjoint key sets. We assume value=1 for missing keys.
  for (auto& pair: a)
  {
    auto& key = pair.first;
    auto& value = pair.second;
    assert(z.find(key) == z.end());
    z[key] = value;
  }

  for (auto& pair: b)
  {
    auto& key = pair.first;
    auto& value = pair.second;
    auto it = z.find(key);
    if (it != z.end())
      it->second *= value;
    else
      z[key] = value;
  }
}

// Add a prefix and suffix to each element in a vector of strings.
static inline
std::vector<std::string> AddPrefixSuffix(const std::vector<std::string>& v,
                                         const std::string& prefix, const std::string& suffix)
{
  std::vector<std::string> retval;
  for (auto& x: v)
  {
    std::string x_ = prefix + x + suffix;
    retval.push_back(x_);
  }
  return retval;
}

template<typename T>
void Print(const T& container, std::string separator = " ",
           std::ostream& out = std::cout)
{
  for (auto& element: container)
  {
    out << element << separator;
  }
}

template<typename T>
void PrintLn(const T& container, std::string separator = " ",
             std::ostream& out = std::cout)
{
  for (auto& element: container)
  {
    out << element << separator;
  }
  out << std::endl;
}

