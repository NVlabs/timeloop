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
//              ResidualFactors
//------------------------------------


unsigned long ResidualFactors::ISqrt_(unsigned long x)
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

void ResidualFactors::ClearAllFactors_()
{
  all_factors_.clear();
}
void ResidualFactors::CalculateAllFactors_()
{
  for (unsigned long i = 1; i <= ISqrt_(n_); i++)
  {
    if (n_ % i == 0)
    {
      all_factors_.insert(i);
      if (i * i != n_)
      {
        all_factors_.insert(n_ / i);
      }
    }
  }
}


// Generate all additional potential factors given the remainder bounds  
void ResidualFactors::CalculateAdditionalFactors_()
{
  std::vector<unsigned long> reaminder_possible; 
  for (auto& n : remainder_bounds_){
    for(unsigned long i = 1; i <= n; i++)
      reaminder_possible.push_back(i);
  }

  for (auto& n : reaminder_possible){
    unsigned long g = n * n_ * ceil((double)n_/(double)n);
    for (unsigned long i = 1; i <= n_; i++)
    {
      if (g % i == 0)
      {
        if (i < n_)
        all_factors_.insert(i);
        if (i * i != g)
        {
          if ((g/i) < n_)
          all_factors_.insert(g / i);
        }
      }
    }
  }
}

std::vector<std::vector<unsigned long>> ResidualFactors::CartProduct_ (const std::vector<std::vector<unsigned long>> v) {
  std::vector<std::vector<unsigned long>> s = {{}};
  for (const auto u : v) {
      std::vector<std::vector<unsigned long>> r;
      for (const auto x : s) {
          for (const auto y : u) {
              r.push_back(x);
              r.back().push_back(y);
          }
      }
      s = r;
  }
  return s;
}

// Replicate all factors possibilities accross each level n, disregarding sets with that include more than dimension size n * sqrt(n)
void ResidualFactors::GenerateFactorProduct_(const unsigned long n, const int order)
{
  for(auto rec = 0; rec < order; rec++){
    std::vector<std::vector<unsigned long>> inter_factors;
    std::vector<std::vector<unsigned long>> product_factors;
    for (auto i = 0; i < order; i++)
    {
      std::vector<unsigned long> v2;
      for(auto a : all_factors_){
        if (i == 0 && rec == 0){
          v2.push_back(a);
        }else if(a <= ((unsigned int)(pow((double)n_, 1.0/(2.0)) + 1.5)) && i > 0){
          v2.push_back(a);
        }else if (rec > 0 && i == 0 && a >= ((unsigned int)(pow((double)n_, 1.0/(2.0)) + 1.5))){
          v2.push_back(a);
        }
      }
      inter_factors.push_back(v2);
    }

    std::swap(inter_factors[0], inter_factors[rec]);

    product_factors = CartProduct_(inter_factors);
    replicated_factors_.reserve(replicated_factors_.size() + product_factors.size());
    replicated_factors_.insert(replicated_factors_.end(), product_factors.begin(), product_factors.end());
  }



  for(auto t : replicated_factors_){

      unsigned long product = 1;
      for(auto p : t){
        if (p != 1){
          product *= (p-1);
        }
      }
      if (product <= n){
          pruned_product_factors_.push_back(t);
      }
  }
  

}

// Replicate all possible residual combiniations, and remove possibilities that sum up to size greater than n
void ResidualFactors::GenerateResidual_(const unsigned long n, const int order)
{
  std::vector<std::vector<unsigned long>> residuals;

  for (auto i : remainder_bounds_){
    std::vector<unsigned long> r;
    for (unsigned j = 1; j <= i; j++){
      r.push_back(j);
    }
    residuals.push_back(r);
  }

  residuals = CartProduct_(residuals);

  for(auto t : residuals){
    unsigned long sum = 0;
    for(auto p : t){
      sum += p;
    }
    if(sum <= n+(unsigned)order){
      pruned_residuals_.push_back(t);
    }
  }

}

// Replicate all possible residual combiniations, and remove possibilities that sum up to size greater than n
void ResidualFactors::ValidityChecker_(const unsigned long n, std::map<unsigned, unsigned long> given)
{

  for (unsigned i = pruned_product_factors_.size(); i > 0 ; i--)
  {


    for (auto it = given.begin(); it != given.end(); it++)
    {
      // Insert the given factor, pushing all existing factors back.
      auto index = it->first;
      auto value = it->second;
      pruned_product_factors_[i-1].insert(pruned_product_factors_[i-1].begin() + index, value);
    }


  }
  

  for(auto f : pruned_product_factors_){
    for(auto r : pruned_residuals_){
      std::vector<unsigned long> valid_residuals;
      int s_i = 0;
      bool valid = true;
      for(unsigned long i = 0; i < (f.size()); i++){
        if(std::count(remainder_ix_.begin(),remainder_ix_.end(), i) > 0){
          valid_residuals.push_back(r.at(s_i));
          
          if(f.at(i) > remainder_bounds_[s_i]){
            valid = false;
          }
          s_i++;
        }else{
          valid_residuals.push_back(f.at(i));
        }
      }

      //Solve for generic is L_{n} = L{n+1}*P{n} + R_{n} - 1
      unsigned long equation_answer = 0;
      for(unsigned j = (f.size()); j > 0; j--){ 
          equation_answer = f.at(j-1)*equation_answer + (valid_residuals.at(j-1) - 1);
      
          if (f.at(j-1) < valid_residuals.at(j-1))
            valid = false;
          if(equation_answer == 0 && valid_residuals.at(j-1) != f.at(j-1))
            valid = false;
        }

        if ((equation_answer + 1 == n) and valid){

          cofactors_.push_back(f);
          rfactors_.push_back(valid_residuals);

      }
    }
  }


}

void ResidualFactors::PruneMax()
{
  // Prune the vector of cofactor sets by removing those sets that have factors
  // outside user-specified min/max range. We should really have done this during
  // MultiplicativeSplitRecursive. However, the "given" map complicates things
  // because given factors may be scattered, and we'll need a map table to
  // find the original rank from the "compressed" rank seen by
  // MultiplicativeSplitRecursive. Doing it now is slower but cleaner and less
  // bug-prone.

}

ResidualFactors::ResidualFactors() : n_(0) {}

/***
First, we cacluate all the factors same as Uber
Next, we add in additional factors based on user defined loop bounds and take the cross product of these (eliminating impossible mappings)
Then, we calculate all the valid mapspace points that fit the expanded formula L_{n} = L{n+1}*P{n} + R_{n} - 1
***/
ResidualFactors::ResidualFactors(const unsigned long n, const int order, std::vector<unsigned long> remainder_bounds, 
    std::vector<unsigned long> remainder_ix) : n_(n), remainder_bounds_(remainder_bounds), remainder_ix_(remainder_ix)
{
  ClearAllFactors_();
  CalculateAllFactors_();
  CalculateAdditionalFactors_();
  GenerateFactorProduct_(n, order);
  GenerateResidual_(n, order);
  std::map<unsigned, unsigned long> given = {{}};
  ValidityChecker_(n, given);

  for (unsigned i = 0; i < cofactors_.size(); i++)
  {
    std::reverse(cofactors_[i].begin(), cofactors_[i].end());
    std::reverse(rfactors_[i].begin(), rfactors_[i].end());
  }
}

ResidualFactors::ResidualFactors(const unsigned long n, const int order, std::vector<unsigned long> remainder_bounds, 
    std::vector<unsigned long> remainder_ix, std::map<unsigned, unsigned long> given)
    : n_(n), remainder_bounds_(remainder_bounds), remainder_ix_(remainder_ix)
{

  const unsigned int given_size = given.size();

  assert(given_size <= std::size_t(order));
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

  ClearAllFactors_();
  CalculateAllFactors_();
  CalculateAdditionalFactors_();



  GenerateFactorProduct_(n / partial_product, order - given.size());

  GenerateResidual_(n / partial_product, order - given.size());
  


  // Insert the given factors at the specified indices of each of the solutions.

  ValidityChecker_(n, given);

  remainder_bounds_.resize(0);
  remainder_ix_.resize(0);
  pruned_product_factors_.resize(0);
  pruned_residuals_.resize(0);
  replicated_factors_.resize(0);

}

std::vector<std::vector<unsigned long>> ResidualFactors::operator[](int index)
{
  std::vector<std::vector<unsigned long>> ret;
  std::vector<unsigned long> cfm = cofactors_.at(index);
  std::vector<unsigned long> rfm = rfactors_.at(index);

  ret.push_back(cfm);
  ret.push_back(rfm);
  return ret;
}

std::size_t ResidualFactors::size() { return cofactors_.size(); }


void ResidualFactors::Print()
{
  PrintAllFactors();
  PrintCoFactors();
}

void ResidualFactors::PrintAllFactors()
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

void ResidualFactors::PrintCoFactors() { std::cout << *this; }

std::ostream& operator<<(std::ostream& out, const ResidualFactors& f) {
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
