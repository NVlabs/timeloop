/* Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

#include <isl/set.h>

#include "loop-analysis/point-set-isl.hpp"

std::mutex ISLPointSet::mutex;
std::unordered_map<pthread_t, isl_ctx*> ISLPointSet::contexts;
std::unordered_map<pthread_t, isl_printer*> ISLPointSet::consoles;

//isl_ctx* ISLPointSet::context = isl_ctx_alloc();
//isl_printer* ISLPointSet::console = isl_printer_to_file(ISLPointSet::context, stdout);

// Convert point to isl_point.
isl_point* ISLPointSet::ToISL(const Point p)
{
  auto order = p.Order();
  isl_space* space = isl_space_set_alloc(Context(), 0, order);
  isl_point* point = isl_point_zero(space);

  for (unsigned dim = 0; dim < order; dim++)
  {
    isl_val* v = isl_val_int_from_si(Context(), p[dim]);
    point = isl_point_set_coordinate_val(point, isl_dim_set, dim, v);
  }

  return point;
}

// Get the thread-local ISL context.
isl_ctx* ISLPointSet::Context()
{
  mutex.lock();
  auto thread_id = pthread_self();
  isl_ctx* retval;
  auto it = contexts.find(thread_id);
  if (it == contexts.end())
  {
    retval = isl_ctx_alloc();
    contexts[thread_id] = retval;
    consoles[thread_id] = isl_printer_to_file(retval, stdout);
  }
  else
  {
    retval = it->second;
  }
  mutex.unlock();
  return retval;
}

ISLPointSet::~ISLPointSet()
{
  if (set_ != nullptr)
    isl_set_free(set_);
}

ISLPointSet::ISLPointSet(std::uint32_t order) :
    order_(order)
{
  isl_space* space = isl_space_set_alloc(Context(), 0, order);
  set_ = isl_set_empty(space);
}

ISLPointSet::ISLPointSet(std::uint32_t order, isl_set* set) :
    order_(order)
{
  set_ = isl_set_copy(set);
}

ISLPointSet::ISLPointSet(std::uint32_t order, const Point unit) :
    order_(order)
{
  // Construct a singleton set.
  set_ = isl_set_from_point(ToISL(unit));
}

ISLPointSet::ISLPointSet(std::uint32_t order, const Point min, const Point max) :
    order_(order)
{
  // Construct an AAHR (or "box" in ISL parlance). Both points are inclusive
  // in the ISL call, so we need to adjust our max.
  Point incl_max = max;
  incl_max.IncrementAllDimensions(-1);
  set_ = isl_set_box_from_points(ToISL(min), ToISL(incl_max));
}

ISLPointSet::ISLPointSet(const ISLPointSet& a) :
    order_(a.order_)
{
  set_ = isl_set_copy(a.set_);
}

// Copy-and-swap idiom.
ISLPointSet& ISLPointSet::operator = (ISLPointSet other)
{
  // Note: the copy constructor fired because this function is call-by value.
  // This means other now has an isl_copy of the isl_set that was in the
  // parameter in the function call. The swap() call below will simply swap
  // the set_ pointers between us and the copy in other, giving us that
  // pointer. When this function ends, other will be destroyed, calling
  // isl_set_free on the set_ pointer we were holding (if any).
  swap(*this, other);
  return *this;
}

void swap(ISLPointSet& first, ISLPointSet& second)
{
  using std::swap;
  swap(first.order_, second.order_);
  swap(first.set_, second.set_);
}

std::size_t ISLPointSet::size() const
{
  isl_set* copy = isl_set_copy(set_);

  // Unfortunately, barvinok does not appear to be thread-safe. Even though
  // we have per-thread context management, we need to wrap this call
  // with a mutex, which is unfortunate because this call is the most
  // expensive step in the entire execution.
  mutex.lock();    
  isl_pw_qpolynomial* cardinality = isl_set_card(copy);
  mutex.unlock();

  isl_size num_pieces = isl_pw_qpolynomial_n_piece(cardinality);

  ASSERT(num_pieces <= 1);

  size_t size = 0;

  isl_pw_qpolynomial_foreach_piece(
    cardinality,
    [ ](__isl_take isl_set* set, __isl_take isl_qpolynomial* qp, void* user)
    {
      (void) set;

      isl_val* constant = isl_qpolynomial_get_constant_val(qp);
      long numerator = isl_val_get_num_si(constant);
      long denominator = isl_val_get_den_si(constant);

      ASSERT(numerator % denominator == 0);
      *static_cast<size_t*>(user) += (numerator / denominator);

      isl_set_free(set);
      isl_qpolynomial_free(qp);

      return isl_stat_ok;
    },
    static_cast<void*>(&size));

  return size;
}

bool ISLPointSet::empty() const
{
  ASSERT(set_ != nullptr);
  return (isl_set_is_empty(set_) == isl_bool_true);
}

void ISLPointSet::Reset()
{
  if (set_ != nullptr)
    isl_set_free(set_);
  isl_space* space = isl_space_set_alloc(Context(), 0, order_);
  set_ = isl_set_empty(space);
}

ISLPointSet& ISLPointSet::operator += (const Point& p)
{
  isl_set* singleton = isl_set_from_point(ToISL(p));
  set_ = isl_set_union(set_, singleton);
  return *this;
}

ISLPointSet ISLPointSet::operator - (const ISLPointSet& s)
{
  // This implementation performs too many copies. FIXME: optimize.
  isl_set* delta = isl_set_subtract(isl_set_copy(set_), isl_set_copy(s.set_));
  ISLPointSet retval(order_, delta);
  return retval;
}

bool ISLPointSet::operator == (const ISLPointSet& s) const
{
  return (isl_set_is_equal(set_, s.set_) == isl_bool_true);
}

Point ISLPointSet::GetTranslation(const ISLPointSet& s) const
{
  (void) s;
  assert(false);
  return Point(order_);
}

void ISLPointSet::Translate(const Point& p)
{
  (void) p;
  assert(false);
}

void ISLPointSet::Print(std::ostream& out = std::cout) const
{
  (void) out;
  mutex.lock();
  consoles.at(pthread_self()) = isl_printer_print_set(consoles.at(pthread_self()), set_);
  mutex.unlock();
}
