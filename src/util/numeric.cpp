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

#include <boost/multiprecision/cpp_int.hpp>

#include "util/numeric.hpp"

using namespace boost::multiprecision;

//------------------------------------
//              Factors
//------------------------------------

unsigned long Factors::ISqrt_(unsigned long x)
{
  unsigned long op, res, one;

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

void Factors::CalculateAllFactors_()
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
std::vector<std::vector<unsigned long>>
Factors::MultiplicativeSplitRecursive_(unsigned long n, int order)
{
  if (order == 0)
  {
    if (n != 1)
    {
      std::cerr << "ERROR: Factors: cannot split n=" << n << " into 0 cofactors." << std::endl;
      assert(false);
    }
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

Factors::Factors() :
    n_(0)
{
}

Factors::Factors(const unsigned long n, const int order) :
    n_(n), cofactors_()
{
  CalculateAllFactors_();
  cofactors_ = MultiplicativeSplitRecursive_(n, order);
}

Factors::Factors(const unsigned long n, const int order, std::map<unsigned, unsigned long> given) :
    n_(n), cofactors_()
{
  assert(given.size() <= std::size_t(order));

  // If any of the given factors is not a factor of n, forcibly reset that to
  // be a free variable, otherwise accumulate them into a partial product.
  unsigned long partial_product = 1;
  for (auto f = given.begin(); f != given.end(); f++)
  {
    auto factor = f->second;
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
      f = given.erase(f);
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
      f->second = factor;
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

void Factors::PruneMax(std::map<unsigned, unsigned long>& max)
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

std::vector<unsigned long>& Factors::operator[](int index)
{
  return cofactors_[index];
}

std::size_t Factors::size()
{
  return cofactors_.size();
}

void Factors::Print()
{
  PrintAllFactors();
  PrintCoFactors();
}

void Factors::PrintAllFactors()
{
  std::cout << "All factors of " << n_ << ": ";
  bool first = true;
  for (auto f = all_factors_.begin(); f != all_factors_.end(); f++)
  {
    if (first)
    {
      first = false;
    }
    else
    {
      std::cout << ", ";
    }
    std::cout << (*f);
  }
  std::cout << std::endl;
}

void Factors::PrintCoFactors()
{
  std::cout << *this;
}

std::ostream& operator<<(std::ostream& out, const Factors& f)
{
  out << "Co-factors of " << f.n_ << " are: " << std::endl;
  for (auto cset = f.cofactors_.begin(); cset != f.cofactors_.end(); cset++)
  {
    out << "    " << f.n_ << " = ";
    bool first = true;
    for (auto i = cset->begin(); i != cset->end(); i++)
    {
      if (first)
      {
        first = false;
      }
      else
      {
        out << " * ";
      }
      out << (*i);
    }
    out << std::endl;
  }
  return out;
}

//------------------------------------
//        PatternGenerator128
//------------------------------------

PatternGenerator128::PatternGenerator128(uint128_t bound) :
    bound_(bound)
{
}

SequenceGenerator128::SequenceGenerator128(uint128_t bound, bool autoloop) :
    PatternGenerator128(bound),
    autoloop_(autoloop),
    cur_(0)
{
}

uint128_t SequenceGenerator128::Next()
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


RandomGenerator128::RandomGenerator128(uint128_t bound) :
    PatternGenerator128(bound),
    use_two_generators_(bound > uint128_t(uint64_max_)),
    low_gen_(0, use_two_generators_ ? uint64_max_ : (std::uint64_t)(bound - 1)),
    high_gen_(0, (std::uint64_t)(bound/uint64_max_ - 1))
{
}

uint128_t RandomGenerator128::Next()
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


//------------------------------------
//           Miscellaneous
//------------------------------------

// Returns the smallest factor of an integer and the quotient after
// division with the smallest factor.
void SmallestFactor(uint64_t n, uint64_t& factor, uint64_t& residue)
{
  for (uint64_t i = 2; i < n; i++)
  {
    if (n % i == 0)
    {
      factor = i;
      residue = n / i;
      return;
    }
  }
  factor = n;
  residue = 1;
}

// Helper function to get close-to-square layouts of arrays
// containing a given number of nodes.
void GetTiling(uint64_t num_elems, uint64_t& height, uint64_t& width)
{
  std::vector<uint64_t> factors;
  uint64_t residue = num_elems;
  uint64_t cur_factor;
  while (residue > 1)
  {
    SmallestFactor(residue, cur_factor, residue);
    factors.push_back(cur_factor);
  }

  height = 1;
  width = 1;
  for (uint64_t i = 0; i < factors.size(); i++)
  {
    if (i % 2 == 0)
      height *= factors[i];
    else
      width *= factors[i];
  }

  if (height > width)
  {
    uint64_t temp = height;
    height = width;
    width = temp;
  }
}

double LinearInterpolate(double x,
                         double x0, double x1,
                         double q0, double q1)
{
  double slope = (x0 == x1) ? 0 : (q1 - q0) / double(x1 - x0);
  return q0 + slope * (x - x0);
}

double BilinearInterpolate(double x, double y,
                           double x0, double x1,
                           double y0, double y1,
                           double q00, double q01, double q10, double q11)
{
  // Linear interpolate along x dimension.
  double qx0 = LinearInterpolate(x, x0, x1, q00, q10);
  double qx1 = LinearInterpolate(x, x0, x1, q01, q11);

  // Linear interpolate along y dimension.
  return LinearInterpolate(y, y0, y1, qx0, qx1);
}
