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

#include <array>
#include <vector>
#include <utility>
#include <iostream>
#include <cstdint>
#include <map>
#include <random>
#include <set>
#include <boost/multiprecision/cpp_int.hpp>

using namespace boost::multiprecision;

//------------------------------------
//              Factors
//------------------------------------

typedef std::pair<unsigned long, int> Factor;

class Factors
{
 private:
  unsigned long n_;
  std::vector<unsigned long> all_factors_;
  std::vector<std::vector<unsigned long>> cofactors_;

  unsigned long ISqrt_(unsigned long x);

  void CalculateAllFactors_(unsigned long of);

  // Return a vector of all order-way cofactor sets of n.
  std::vector<std::vector<unsigned long>> MultiplicativeSplitRecursive_(unsigned long n, int order);

 public:
  Factors();
  Factors(const unsigned long n, const int order);
  Factors(const unsigned long n, const int order, std::map<unsigned, unsigned long> given);

  void PruneMax(std::map<unsigned, unsigned long>& max);
  void PruneMin(std::map<unsigned, unsigned long>& min);

  std::vector<unsigned long>& operator[](int index);

  std::size_t size();

  void Print();
  void PrintAllFactors();
  void PrintCoFactors();

  friend std::ostream& operator<<(std::ostream& out, const Factors& f);
};

//------------------------------------
//              ResidualFactors
//------------------------------------

typedef std::pair<unsigned long, int> Factor;

class ResidualFactors
{
 private:
  unsigned long n_;
  std::vector<unsigned long> remainder_bounds_;
  std::vector<unsigned long> remainder_ix_;
  std::set<unsigned long> all_factors_;
  std::vector<std::vector<unsigned long>> pruned_product_factors_;
  std::vector<std::vector<unsigned long>> pruned_residuals_;
  std::vector<std::vector<unsigned long>> cofactors_;
  std::vector<std::vector<unsigned long>> rfactors_;
  std::vector<std::vector<unsigned long>> replicated_factors_;
  

  unsigned long ISqrt_(unsigned long x);

  void ClearAllFactors_();

  void CalculateAllFactors_(unsigned long of);

  void CalculateAdditionalFactors_(unsigned long of);
  std::vector<std::vector<unsigned long>> CartProduct_ (const std::vector<std::vector<unsigned long>> v);

  void GenerateFactorProduct_(const unsigned long n, const int order);
  void GenerateResidual_(const unsigned long n, const int order);
  void ValidityChecker_(const unsigned long n, std::map<unsigned, unsigned long> given, std::map<unsigned, unsigned long> given_residuals);

  // Return a vector of all order-way cofactor sets of n.

 public:
  ResidualFactors();
  ResidualFactors(const unsigned long n, const int order, std::vector<unsigned long> remainder_bounds, std::vector<unsigned long> remainder_ix);
  ResidualFactors(const unsigned long n, const int order, std::vector<unsigned long> remainder_bounds, std::vector<unsigned long> remainder_ix, std::map<unsigned, unsigned long> given);


  void PruneMax(std::map<unsigned, unsigned long>& max);
  void PruneMin(std::map<unsigned, unsigned long>& min);

  std::vector<std::vector<unsigned long>> operator[](int index);

  std::size_t size();

  void Print();
  void PrintAllFactors();
  void PrintCoFactors();

  friend std::ostream& operator<<(std::ostream& out, const ResidualFactors& f);
};
//------------------------------------
//        Cartesian Counter
//------------------------------------

template <int order>
class CartesianCounter
{
 private:
  // Note: base is variable for each position!
  std::array<uint128_t, order> base_;
  std::array<uint128_t, order> value_;
  checked_uint128_t integer_;
  checked_uint128_t endint_;

  bool IncrementRecursive_(int position)
  {
    if (value_[position] < base_[position] - 1)
    {
      value_[position]++;
      return true;
    }
    else if (position < order - 1)
    {
      value_[position] = 0;
      return IncrementRecursive_(position + 1);
    }
    else
    {
      // Overflow! We could do one of 3 things:
      // (a) Throw an exception.
      // (b) Wrap-around to 0.
      // (c) Saturate and return a code.
      // We choose (c) for this implementation.
      return false;
    }
  }

  void UpdateIntegerFromValue()
  {
    integer_ = 0;
    checked_uint128_t base = 1;
    for (int i = 0; i < order; i++)
    {
      integer_ += (checked_uint128_t(value_[i]) * base);
      base *= base_[i];
    }
  }

  void UpdateValueFromInteger()
  {
    uint128_t n = integer_;
    for (int i = 0; i < order; i++)
    {
      value_[i] = n % base_[i];
      n = n / base_[i];
    }
  }

 public:
  CartesianCounter(std::array<uint128_t, order> base = {})
  {
    Init(base);
  }

  template<typename T>
  void Init(T base)
  {
    value_ = {};
    integer_ = 0;
    endint_ = 1;
    for (int i = 0; i < order; i++)
    {
      base_[i] = base[i];
      endint_ *= checked_uint128_t(base_[i]);
    }
  }

  bool Increment()
  {
    integer_++;
    return IncrementRecursive_(0);
  }

  std::array<uint128_t, order> Read() const
  {
    return value_;
  }

  std::array<uint128_t, order> Base() const
  {
    return base_;
  }

  uint128_t operator[] (int dim)
  {
    return value_[dim];
  }

  void Set(uint128_t n)
  {
    assert(checked_uint128_t(n) < endint_);
    integer_ = n;
    UpdateValueFromInteger();
  }

  void Set(int dim, uint128_t v)
  {
    value_[dim] = v;
    UpdateIntegerFromValue();
    assert(integer_ < endint_);
  }

  void Set(std::array<uint128_t, order> v)
  {
    value_ = v;
    UpdateIntegerFromValue();
    assert(integer_ < endint_);
  }

  uint128_t EndInteger() const
  {
    return endint_;
  }

  uint128_t Integer() const
  {
    return integer_;
  }
};

//------------------------------------
//    Cartesian Counter (dynamic)
//------------------------------------

class CartesianCounterDynamic
{
 private:
  // Note: base is variable for each position!
  int order_;
  std::vector<uint128_t> base_;
  std::vector<uint128_t> value_;
  checked_uint128_t integer_;
  checked_uint128_t endint_;

  bool IncrementRecursive_(int position)
  {
    if (value_[position] < base_[position] - 1)
    {
      value_[position]++;
      return true;
    }
    else if (position < order_ - 1)
    {
      value_[position] = 0;
      return IncrementRecursive_(position + 1);
    }
    else
    {
      // Overflow! We could do one of 3 things:
      // (a) Throw an exception.
      // (b) Wrap-around to 0.
      // (c) Saturate and return a code.
      // We choose (c) for this implementation.
      return false;
    }
  }

  void UpdateIntegerFromValue()
  {
    integer_ = 0;
    checked_uint128_t base = 1;
    for (int i = 0; i < order_; i++)
    {
      integer_ += (checked_uint128_t(value_[i]) * base);
      base *= base_[i];
    }
  }

  void UpdateValueFromInteger()
  {
    uint128_t n = integer_;
    for (int i = 0; i < order_; i++)
    {
      value_[i] = n % base_[i];
      n = n / base_[i];
    }
  }

 public:
  CartesianCounterDynamic() = delete;

  CartesianCounterDynamic(unsigned order) :
      order_(order)
  {
    base_.resize(order_);
    value_.resize(order_);
  }
  
  CartesianCounterDynamic(std::vector<uint128_t> base) :
      CartesianCounterDynamic(base.size())
  {
    Init(base);
  }

  template<typename T>
  void Init(T base)
  {
    integer_ = 0;
    endint_ = 1;
    for (int i = 0; i < order_; i++)
    {
      base_[i] = base[i];
      endint_ *= checked_uint128_t(base_[i]);
      value_[i] = 0;
    }
  }

  bool Increment()
  {
    integer_++;
    return IncrementRecursive_(0);
  }

  std::vector<uint128_t> Read() const
  {
    return value_;
  }

  std::vector<uint128_t> Base() const
  {
    return base_;
  }

  uint128_t operator[] (int dim)
  {
    return value_[dim];
  }

  void Set(uint128_t n)
  {
    assert(order_ != 0);
    assert(checked_uint128_t(n) < endint_);
    integer_ = n;
    UpdateValueFromInteger();
  }

  void Set(int dim, uint128_t v)
  {
    assert(dim < order_);
    value_[dim] = v;
    UpdateIntegerFromValue();
    assert(integer_ < endint_);
  }

  void Set(std::vector<uint128_t> v)
  {
    order_ = v.size();
    value_ = v;
    UpdateIntegerFromValue();
    assert(integer_ < endint_);
  }

  uint128_t EndInteger() const
  {
    return endint_;
  }

  uint128_t Integer() const
  {
    return integer_;
  }
};

//------------------------------------
//    Cartesian Counter (generic)
//------------------------------------

template<typename T>
class CartesianCounterGeneric
{
 private:
  // Note: base is variable for each position!
  int order_;
  std::vector<T> base_;
  std::vector<T> value_;
  T integer_;
  T endint_;

  bool IncrementRecursive_(int position)
  {
    if (value_[position] < base_[position] - 1)
    {
      value_[position]++;
      return true;
    }
    else if (position < order_ - 1)
    {
      value_[position] = 0;
      return IncrementRecursive_(position + 1);
    }
    else
    {
      // Overflow! We could do one of 3 things:
      // (a) Throw an exception.
      // (b) Wrap-around to 0.
      // (c) Saturate and return a code.
      // We choose (c) for this implementation.
      return false;
    }
  }

  void UpdateIntegerFromValue()
  {
    integer_ = 0;
    T base = 1;
    for (int i = 0; i < order_; i++)
    {
      integer_ += (T(value_[i]) * base);
      base *= base_[i];
    }
  }

  void UpdateValueFromInteger()
  {
    T n = integer_;
    for (int i = 0; i < order_; i++)
    {
      value_[i] = n % base_[i];
      n = n / base_[i];
    }
  }

 public:
  CartesianCounterGeneric() = delete;

  CartesianCounterGeneric(unsigned order) :
      order_(order)
  {
    base_.resize(order_);
    value_.resize(order_);
  }
  
  CartesianCounterGeneric(std::vector<T> base) :
      CartesianCounterGeneric(base.size())
  {
    Init(base);
  }

  template<typename S>
  void Init(S base)
  {
    integer_ = 0;
    endint_ = 1;
    for (int i = 0; i < order_; i++)
    {
      base_[i] = base[i];
      endint_ *= T(base_[i]);
      value_[i] = 0;
    }
  }

  bool Increment()
  {
    integer_++;
    return IncrementRecursive_(0);
  }

  std::vector<T> Read() const
  {
    return value_;
  }

  std::vector<T> Base() const
  {
    return base_;
  }

  T operator[] (int dim)
  {
    return value_[dim];
  }

  void Set(T n)
  {
    assert(order_ != 0);
    assert(T(n) < endint_);
    integer_ = n;
    UpdateValueFromInteger();
  }

  void Set(int dim, T v)
  {
    assert(dim < order_);
    value_[dim] = v;
    UpdateIntegerFromValue();
    assert(integer_ < endint_);
  }

  void Set(std::vector<T> v)
  {
    order_ = v.size();
    value_ = v;
    UpdateIntegerFromValue();
    assert(integer_ < endint_);
  }

  T EndInteger() const
  {
    return endint_;
  }

  T Integer() const
  {
    return integer_;
  }
};

//------------------------------------
//             Factoradic
//------------------------------------

template <class T>
class Factoradic
{
 private:
  const static std::size_t MaxLength = 20;  // 21! > 2^64.
  std::uint64_t factorial_table_[MaxLength + 1];

 public:
  Factoradic()
  {
    factorial_table_[0] = 1;
    std::uint64_t fact = 1;
    for (std::uint64_t i = 1; i <= MaxLength; i++)
    {
      fact = fact * i;
      factorial_table_[i] = fact;
    }
  }

  std::uint64_t Factorial(std::uint64_t n)
  {
    assert(n <= MaxLength);
    return factorial_table_[n];
  }

  void Permute(T* buffer, std::size_t length, std::uint64_t index)
  {
    std::uint64_t scale = factorial_table_[length];
    assert(index < scale);

    for (std::size_t i = 0; i < length - 1; i++)
    {
      scale /= (std::uint64_t)(length - i);
      std::size_t d = index / scale;
      index %= scale;
      if (d > 0)
      {
        const T c = buffer[i + d];
        memmove(buffer + i + 1, buffer + i, d * sizeof(T));
        buffer[i] = c;
      }
    }
  }
};

//------------------------------------
//        PatternGenerator128
//------------------------------------

class PatternGenerator128
{
 protected:
  const std::uint64_t uint64_max_ = std::numeric_limits<std::uint64_t>::max();
  uint128_t bound_;

 public:
  PatternGenerator128(uint128_t bound);

  virtual uint128_t Next() = 0;
};

class SequenceGenerator128 final : public PatternGenerator128
{
 private:
  bool autoloop_;
  uint128_t cur_;

 public:
  SequenceGenerator128(uint128_t bound, bool autoloop = true);

  uint128_t Next();
};

class RandomGenerator128 final : public PatternGenerator128
{
 private:
  bool use_two_generators_;

  std::default_random_engine engine_;
  std::uniform_int_distribution<std::uint64_t> low_gen_;
  std::uniform_int_distribution<std::uint64_t> high_gen_;

 public:
  RandomGenerator128(uint128_t bound);

  uint128_t Next();
};

//------------------------------------
//           Miscellaneous
//------------------------------------

// Returns the smallest factor of an integer and the quotient after
// division with the smallest factor.
void SmallestFactor(uint64_t n, uint64_t& factor, uint64_t& residue);

// Helper function to get close-to-square layouts of arrays
// containing a given number of nodes.
void GetTiling(uint64_t num_elems, uint64_t& height, uint64_t& width);

double LinearInterpolate(double x,
                         double x0, double x1,
                         double q0, double q1);

double BilinearInterpolate(double x, double y,
                           double x0, double x1,
                           double y0, double y1,
                           double q00, double q01, double q10, double q11);

//---------------------------------------------
// STL Utility Functions: move to another file
//---------------------------------------------

template<typename K>
struct TaggedBound
{
  bool valid;
  K bound;
};

template<typename K, typename V>
std::tuple<TaggedBound<K>, TaggedBound<K>>
FindBounds(std::map<K, V> map, K key)
{
  TaggedBound<K> lower = { false, K() };
  TaggedBound<K> upper = { false, K() };

  if (!map.empty())
  {
    // Find consecutive entries in the map that bound the width parameter.
    typename std::map<K, V>::iterator it;
    for (it = map.begin(); it != map.end() && key > it->first; it++);

    if (it != map.end())
    {
      upper.valid = true;
      upper.bound = it->first;
    }
    
    if (it != map.begin())
    {
      it--;
      lower.valid = true;
      lower.bound = it->first;
    }
    else if (key == it->first)
    {
      lower.valid = true;
      lower.bound = it->first;
    }
  }
  
  return std::make_tuple(lower, upper);
}
