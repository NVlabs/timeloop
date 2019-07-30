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

#pragma once

#include <array>
#include <vector>
#include <utility>
#include <iostream>
#include <cstdint>
#include <map>
#include <random>
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

  unsigned long ISqrt_(unsigned long x)
  {
    register unsigned long op, res, one;

    op = x;
    res = 0;

    one = 1 << 30;
    while (one > op) one >>= 2;

    while (one != 0)
    {
      if (op >= res + one)
      {
        op -= res + one;
        res += one << 1;
      }
      res >>= 1;
      one >>= 2;
    }
    return res;
  }

  void CalculateAllFactors_()
  {
    all_factors_.clear();
    for (unsigned long i = 1; i <= ISqrt_(n_); i++)
    {
      if (n_ % i == 0)
      {
        all_factors_.push_back(i);
        if (i * i != n_)
        {
          all_factors_.push_back(n_ / i);
        }
      }
    }
  }

  // Return a vector of all order-way cofactor sets of n.
  std::vector<std::vector<unsigned long>> MultiplicativeSplitRecursive_(
      unsigned long n, int order)
  {
    if (order == 0)
    {
      return {{}};
    }
    else if (order == 1)
    {
      return {{n}};
    }
    else
    {
      std::vector<std::vector<unsigned long>> retval;
      for (auto factor = all_factors_.begin(); factor != all_factors_.end(); factor++)
      {
        // This factor is only acceptable if the residue is divisible by it.
        if (n % (*factor) == 0)
        {
          // Recursive call.
          std::vector<std::vector<unsigned long>> subproblem =
              MultiplicativeSplitRecursive_(n / (*factor), order - 1);

          // Append this factor to the end of each vector returned by the
          // recursive call.
          for (auto vector = subproblem.begin(); vector != subproblem.end(); vector++)
          {
            vector->push_back(*factor);
          }

          // Add all these appended vectors to my growing vector of vectors.
          retval.insert(retval.end(), subproblem.begin(), subproblem.end());
        }
        else
        {
          // Discard this factor.
        }
      }
      return retval;
    }
  }

 public:
  Factors() : n_(0) {}

  Factors(const unsigned long n, const int order) : n_(n), cofactors_()
  {
    CalculateAllFactors_();
    cofactors_ = MultiplicativeSplitRecursive_(n, order);
  }

  Factors(const unsigned long n, const int order, std::map<unsigned, unsigned long> given)
      : n_(n), cofactors_()
  {
    assert(given.size() <= std::size_t(order));

    // If any of the given factors is not a factor of n, forcibly reset that to
    // be a free variable, otherwise accumulate them into a partial product.
    unsigned long partial_product = 1;
    for (auto& f : given)
    {
      auto factor = f.second;
      if (n % (factor * partial_product) == 0)
      {
        partial_product *= factor;
      }
      else
      {
        std::cerr << "WARNING: cannot accept " << factor << " as a factor of " << n
                  << " with current partial product " << partial_product;
#define SET_NONFACTOR_TO_FREE_VARIABLE
#ifdef SET_NONFACTOR_TO_FREE_VARIABLE
        // Ignore mapping constraint and set the factor to a free variable.
        std::cerr << ", ignoring mapping constraint and setting to a free variable."
                  << std::endl;
        given.erase(f.first);
#else       
        // Try to find the next *lower* integer that is a factor of n.
        // FIXME: there are multiple exceptions that can cause us to be here:
        // (a) factor doesn't divide into n.
        // (b) factor does divide into n but causes partial product to exceed n.
        // (c) factor does divide into n but causes partial product to not
        //     divide into n.
        // The following code only solves case (a).
        std::cerr << "FIXME: please fix this code." << std::endl;
        assert(false);
        
        for (; factor >= 1 && (n % factor != 0); factor--);
        std::cerr << ", setting this to " << factor << " instead." << std::endl;
        f.second = factor;
        partial_product *= factor;
#endif
      }
      assert(n % partial_product == 0);
    }

    CalculateAllFactors_();

    cofactors_ = MultiplicativeSplitRecursive_(n / partial_product, order - given.size());

    // Insert the given factors at the specified indices of each of the solutions.
    for (auto& cofactors : cofactors_)
    {
      for (auto& given_factor : given)
      {
        // Insert the given factor, pushing all existing factors back.
        auto index = given_factor.first;
        auto value = given_factor.second;
        assert(index <= cofactors.size());
        cofactors.insert(cofactors.begin() + index, value);
      }
    }
  }

  void PruneMax(std::map<unsigned, unsigned long>& max)
  {
    // Prune the vector of cofactor sets by removing those sets that have factors
    // outside user-specified min/max range. We should really have done this during
    // MultiplicativeSplitRecursive. However, the "given" map complicates things
    // because given factors may be scattered, and we'll need a map table to
    // find the original rank from the "compressed" rank seen by
    // MultiplicativeSplitRecursive. Doing it now is slower but cleaner and less
    // bug-prone.

    auto cofactors_it = cofactors_.begin();
    while (cofactors_it != cofactors_.end())
    {
      bool illegal = false;
      for (auto& max_factor : max)
      {
        auto index = max_factor.first;
        auto max = max_factor.second;
        assert(index <= cofactors_it->size());
        auto value = cofactors_it->at(index);
        if (value > max)
        {
          illegal = true;
          break;
        }
      }
      
      if (illegal)
        cofactors_it = cofactors_.erase(cofactors_it);
      else
        cofactors_it++;
    }
  }

  std::vector<unsigned long>& operator[](int index)
  {
    return cofactors_[index];
  }

  std::size_t size() { return cofactors_.size(); }

  void Print()
  {
    PrintAllFactors();
    PrintCoFactors();
  }

  void PrintAllFactors()
  {
    std::cout << "All factors of " << n_ << ": ";
    bool first = true;
    for (auto f = all_factors_.begin(); f != all_factors_.end(); f++) {
      if (first) {
        first = false;
      } else {
        std::cout << ", ";
      }
      std::cout << (*f);
    }
    std::cout << std::endl;
  }

  void PrintCoFactors() { std::cout << *this; }

  friend std::ostream& operator<<(std::ostream& out, const Factors& f) {
    out << "Co-factors of " << f.n_ << " are: " << std::endl;
    for (auto cset = f.cofactors_.begin(); cset != f.cofactors_.end(); cset++) {
      out << "    " << f.n_ << " = ";
      bool first = true;
      for (auto i = cset->begin(); i != cset->end(); i++) {
        if (first) {
          first = false;
        } else {
          out << " * ";
        }
        out << (*i);
      }
      out << std::endl;
    }
    return out;
  }
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
  PatternGenerator128(uint128_t bound) :
      bound_(bound)
  {
  }

  virtual uint128_t Next() = 0;
};

class SequenceGenerator128 final : public PatternGenerator128
{
 private:
  bool autoloop_;
  uint128_t cur_;

 public:
  SequenceGenerator128(uint128_t bound, bool autoloop = true) :
      PatternGenerator128(bound),
      autoloop_(autoloop),
      cur_(0)
  {
  }

  uint128_t Next()
  {
    auto retval = cur_;
    if (cur_ == bound_-1)
    {
      assert(autoloop_);
      cur_ = 0;
    }
    else
    {
      cur_++;
    }
    return retval;
  }
};

class RandomGenerator128 final : public PatternGenerator128
{
 private:
  bool use_two_generators_;

  std::default_random_engine engine_;
  std::uniform_int_distribution<std::uint64_t> low_gen_;
  std::uniform_int_distribution<std::uint64_t> high_gen_;

 public:
  RandomGenerator128(uint128_t bound) :
      PatternGenerator128(bound),
      use_two_generators_(bound > uint128_t(uint64_max_)),
      low_gen_(0, use_two_generators_ ? uint64_max_ : (std::uint64_t)(bound - 1)),
      high_gen_(0, (std::uint64_t)(bound/uint64_max_ - 1))
  {
  }

  uint128_t Next()
  {
    std::uint64_t low = low_gen_(engine_);
    std::uint64_t high = 0;
    
    if (use_two_generators_)
    {
      high = high_gen_(engine_);
    }

    uint128_t rand = low + ((uint128_t)high * uint64_max_);
    assert(rand < bound_);
    
    return rand;
  }
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
