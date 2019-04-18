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

#error fix API incompatibility before using this point set implementation 

// ---------------------------------------------
//     Generic-Fast Point Set implementation
// ---------------------------------------------

class DenseSet {
 private:
  std::uint32_t lower_bound;
  std::uint32_t upper_bound;

 public:
  DenseSet() {
    lower_bound = 0;
    upper_bound = 0;
  }

  DenseSet(const std::uint32_t _lower_bound, const std::uint32_t _upper_bound) {
    lower_bound = _lower_bound;
    upper_bound = _upper_bound;
  }

  DenseSet(const std::uint32_t _lower_bound) {
    lower_bound = _lower_bound;
    upper_bound = lower_bound + 1;
  }

  std::size_t size() const { return (upper_bound - lower_bound); }

  bool empty() const { return upper_bound == lower_bound; }

  void Reset() {
    lower_bound = 0;
    upper_bound = 0;
  }

  DenseSet operator+(const DenseSet& s) {
    DenseSet r(*this);

    if (!s.empty()) {
      if (r.empty()) {
        r.lower_bound = s.lower_bound;
        r.upper_bound = s.upper_bound;
      } else {
        if (r.lower_bound != s.lower_bound || r.upper_bound != s.upper_bound) {
          if (r.upper_bound == s.lower_bound) {
            r.upper_bound = s.upper_bound;
          } else if (r.lower_bound == s.upper_bound) {
            r.lower_bound = s.lower_bound;
          } else {
            assert(false);
          }
        }
      }
    }

    return r;
  }

  DenseSet& operator+=(const DenseSet& s) {
    if (!s.empty()) {
      if (empty()) {
        lower_bound = s.lower_bound;
        upper_bound = s.upper_bound;
      } else {
        if (lower_bound != s.lower_bound || upper_bound != s.upper_bound) {
          if (upper_bound == s.lower_bound) {
            upper_bound = s.upper_bound;
          } else if (lower_bound == s.upper_bound) {
            lower_bound = s.lower_bound;
          } else {
            assert(false);
          }
        }
      }
    }

    return *this;
  }

  bool NoOverlap(const DenseSet& s) const {
    return (lower_bound >= s.upper_bound || upper_bound <= s.lower_bound);
  }

  DenseSet operator-(const DenseSet& s) const {
    DenseSet r(*this);

    if (!r.empty() && !s.empty()) {
      // If r, s have zero overlap, r - s = r
      bool no_overlap = NoOverlap(s);

      if (!no_overlap) {
        if (r.lower_bound == s.lower_bound && r.upper_bound == s.upper_bound) {
          // the two subspaces are equal, result is empty set.
          r.Reset();
        } else {
          if (r.lower_bound <= s.lower_bound &&
              r.upper_bound <= s.upper_bound) {
            r.upper_bound = s.lower_bound;
          } else if (r.lower_bound >= s.lower_bound &&
                     r.upper_bound >= s.upper_bound) {
            r.lower_bound = s.upper_bound;
          } else {
            assert(false);
          }
        }
      }
    }

    return r;
  }

  DenseSet& operator-=(const DenseSet& s) {
    if (!empty() && !s.empty()) {
      // If r, s have zero overlap, r - s = r
      bool no_overlap = NoOverlap(s);

      if (!no_overlap) {
        if (lower_bound == s.lower_bound && upper_bound == s.upper_bound) {
          // the two subspaces are equal, result is empty set.
          Reset();
        } else {
          if (lower_bound <= s.lower_bound && upper_bound <= s.upper_bound) {
            upper_bound = s.lower_bound;
          } else if (lower_bound >= s.lower_bound &&
                     upper_bound >= s.upper_bound) {
            lower_bound = s.upper_bound;
          } else {
            assert(false);
          }
        }
      }
    }

    return (*this);
  }

  bool operator==(const DenseSet& rhs) const {
    DenseSet lhs = *(this);
    bool lhs_empty = lhs.empty();
    bool rhs_empty = rhs.empty();

    if (!lhs_empty && !rhs_empty) {
      if (lhs.lower_bound != rhs.lower_bound ||
          lhs.upper_bound != rhs.upper_bound) {
        return false;
      }
    } else if (lhs_empty ^ rhs_empty) {
      return false;
    }
    return true;
  }

  void Print() const {
    std::cout << "[" << lower_bound << ", " << upper_bound << ")" << std::endl;
  }
};

namespace point {
static const uint32_t sparse_set_words = 4; // WAS: 4, then 8
static const uint32_t sparse_set_word_size = 32; // WAS: 32, then 64
};

template <std::uint32_t num_words>
class SparseSet {
 private:
  static const std::uint32_t bits_per_word = point::sparse_set_word_size;
  std::array<std::uint32_t, num_words> bit_vector;

 public:
  SparseSet() { bit_vector.fill(0); }

  std::size_t size() const {
    std::size_t total = 0;
    for (std::uint32_t i = 0; i < num_words; i++) {
      std::uint32_t a = bit_vector[i];
      total += __builtin_popcount(a);
    }
    return total;
  }

  bool empty() const {
    bool empty = true;
    for (std::uint32_t i = 0; i < num_words; i++) {
      if (bit_vector[i] != 0) {
        empty = false;
        break;
      }
    }
    return empty;
  }

  void Reset() { bit_vector.fill(0); }

  bool NoOverlap(const SparseSet<num_words>& s) const {
    bool no_overlap = true;
    for (std::uint32_t i = 0; i < num_words; i++) {
      if ((bit_vector[i] & s.bit_vector[i]) != 0) {
        no_overlap = false;
        break;
      }
    }
    return no_overlap;
  }

  SparseSet<num_words> operator+(const std::uint32_t& p) const {
    ASSERT(p < num_words * bits_per_word);
    SparseSet<num_words> r(*this);
    std::uint32_t word_pos = p / bits_per_word;
    std::uint32_t word_offset = p & (bits_per_word - 1);
    r.bit_vector[word_pos] = r.bit_vector[word_pos] | (1 << word_offset);
    return r;
  }

  SparseSet<num_words> operator+(const SparseSet<num_words>& s) const {
    SparseSet<num_words> r(*this);
    for (std::uint32_t i = 0; i < num_words; i++) {
      r.bit_vector[i] = r.bit_vector[i] | s.bit_vector[i];
    }
    return r;
  }

  SparseSet<num_words>& operator+=(const std::uint32_t& p) {
    ASSERT(p < num_words * bits_per_word);
    std::uint32_t word_pos = p / bits_per_word;
    std::uint32_t word_offset = p & (bits_per_word - 1);
    bit_vector[word_pos] = bit_vector[word_pos] | (1 << word_offset);
    return *this;
  }

  SparseSet<num_words>& operator+=(const SparseSet<num_words>& s) {
    for (std::uint32_t i = 0; i < num_words; i++) {
      bit_vector[i] = bit_vector[i] | s.bit_vector[i];
    }
    return *this;
  }

  SparseSet<num_words> operator-(const SparseSet<num_words>& s) const {
    SparseSet<num_words> r(*this);
    for (std::uint32_t i = 0; i < num_words; i++) {
      r.bit_vector[i] = r.bit_vector[i] & (~s.bit_vector[i]);
    }
    return r;
  }

  SparseSet<num_words>& operator-=(const SparseSet<num_words>& s) {
    for (std::uint32_t i = 0; i < num_words; i++) {
      bit_vector[i] = bit_vector[i] & (~s.bit_vector[i]);
    }
    return *this;
  }

  bool operator==(const SparseSet<num_words>& rhs) const {
    bool equal = true;
    for (std::uint32_t i = 0; i < num_words; i++) {
      if (bit_vector[i] != rhs.bit_vector[i]) {
        equal = false;
        break;
      }
    }
    return equal;
  }

  void Print() const {
    std::cout << "< ";
    for (std::uint32_t i = 0; i < num_words; i++) {
      for (std::uint32_t j = 0; j < bits_per_word; j++) {
        if (bit_vector[i] & (1 << j)) {
          std::cout << i* bits_per_word + j << " ";
        }
      }
    }
    std::cout << "> " << std::endl;
  }
};

template <std::uint32_t dense_order, std::uint32_t sparse_order>
class PointSet {
 private:
  std::array<DenseSet, dense_order> dense_sets;
  std::array<SparseSet<point::sparse_set_words>, sparse_order> sparse_sets;

 public:
  PointSet() : dense_sets(), sparse_sets() {}

  PointSet(const std::array<DenseSet, dense_order>& _dense_sets,
                    const std::array<SparseSet<point::sparse_set_words>,
                                     sparse_order>& _sparse_sets) {
    dense_sets = _dense_sets;
    sparse_sets = _sparse_sets;
  }

  std::size_t size() const {
    std::size_t retval = 1;
    for (std::uint32_t i = 0; i < dense_order && retval != 0; i++) {
      retval *= dense_sets[i].size();
    }
    for (std::uint32_t i = 0; i < sparse_order && retval != 0; i++) {
      retval *= sparse_sets[i].size();
    }
    return retval;
  }

  bool empty() const {
    for (std::uint32_t i = 0; i < dense_order; i++) {
      if (dense_sets[i].empty()) {
        return true;
      }
    }
    for (std::uint32_t i = 0; i < sparse_order; i++) {
      if (sparse_sets[i].empty()) {
        return true;
      }
    }
    return false;
  }

  void Reset() {
    for (std::uint32_t i = 0; i < dense_order; i++) {
      dense_sets[i].Reset();
    }
    for (std::uint32_t i = 0; i < sparse_order; i++) {
      sparse_sets[i].Reset();
    }
  }

  PointSet<dense_order, sparse_order> operator+(
      const PointSet<dense_order, sparse_order>& s) {
    PointSet<dense_order, sparse_order> r(*this);
    if (!s.empty()) {
      if (r.empty()) {
        r.dense_sets = s.dense_sets;
        r.sparse_sets = s.sparse_sets;
      } else {
        int diff_dim = -1;
        for (std::uint32_t i = 0; i < dense_order; i++) {
          if (!(r.dense_sets[i] == s.dense.sets[i])) {
            ASSERT(diff_dim == -1);  // FIXME
            diff_dim = i;
          }
        }
        for (std::uint32_t i = 0; i < sparse_order; i++) {
          if (!(r.sparse_sets[i] == s.sparse.sets[i])) {
            // ASSERT(diff_dim == -1);  // FIXME
            diff_dim = i + dense_order;
          }
        }
        if (diff_dim != -1) {
          if (diff_dim < dense_order) {
            r.dense_sets[diff_dim] += s.dense_sets[diff_dim];
          } else {
            r.sparse_sets[diff_dim - dense_order] +=
                s.sparse_sets[diff_dim - dense_order];
          }
        }
      }
    }

    return r;
  }

  PointSet<dense_order, sparse_order>& operator+=(
    const PointSet<dense_order, sparse_order>& s)
  {
    if (!s.empty())
    {
      if (empty())
      {
        dense_sets = s.dense_sets;
        sparse_sets = s.sparse_sets;
      }
      else
      {
        std::uint32_t diff_dim = -1U;
        for (std::uint32_t i = 0; i < dense_order; i++)
        {
          if (!(dense_sets[i] == s.dense_sets[i]))
          {
            ASSERT(diff_dim == -1U);  // FIXME
            diff_dim = i;
          }
        }
        for (std::uint32_t i = 0; i < sparse_order; i++)
        {
          if (!(sparse_sets[i] == s.sparse_sets[i]))
          {
            // ASSERT(diff_dim == -1U);  // FIXME
            diff_dim = i + dense_order;
          }
        }
        if (diff_dim != -1U)
        {
          if (diff_dim < dense_order)
          {
            dense_sets[diff_dim] += s.dense_sets[diff_dim];
          }
          else
          {
            sparse_sets[diff_dim - dense_order] += s.sparse_sets[diff_dim - dense_order];
          }
        }
      }
    }
    return *this;
  }

  PointSet<dense_order, sparse_order> operator-(
      const PointSet<dense_order, sparse_order>& s) const {
    PointSet r(*this);

    if (!r.empty() && !s.empty()) {
      // Check if ranges for any one dimension have no-overlap.
      // In such a case, r, s also have zero overlap and r - s = r
      bool no_overlap = false;
      for (std::uint32_t i = 0; i < dense_order; i++) {
        if (r.dense_sets[i].NoOverlap(s.dense_sets[i])) {
          no_overlap = true;
          break;
        }
      }
      for (std::uint32_t i = 0; i < sparse_order; i++) {
        if (r.sparse_sets[i].NoOverlap(s.sparse_sets[i])) {
          no_overlap = true;
          break;
        }
      }

      if (!no_overlap) {
        std::uint32_t diff_dim = -1U;
        for (std::uint32_t i = 0; i < dense_order; i++) {
          if (!(r.dense_sets[i] == s.dense_sets[i])) {
            ASSERT(diff_dim == -1U);  // FIXME
            diff_dim = i;
          }
        }
        for (std::uint32_t i = 0; i < sparse_order; i++) {
          if (!(r.sparse_sets[i] == s.sparse_sets[i])) {
            // ASSERT(diff_dim == -1U);  // FIXME
            diff_dim = i + dense_order;
          }
        }

        if (diff_dim == -1U) {
          // the two subspaces are equal, result is empty set.
          r.Reset();
        } else {
          if (diff_dim < dense_order) {
            r.dense_sets[diff_dim] -= s.dense_sets[diff_dim];
          } else {
            r.sparse_sets[diff_dim - dense_order] -=
                s.sparse_sets[diff_dim - dense_order];
          }
          if (r.empty()) {
            r.Reset();
          }
        }
      }
    }

    return r;
  }

  PointSet<dense_order, sparse_order>& operator-=(
      const PointSet<dense_order, sparse_order>& s) {
    if (!empty() && !s.empty()) {
      // Check if ranges for any one dimension have no-overlap.
      // In such a case, r, s also have zero overlap and r - s = r
      bool no_overlap = false;
      for (std::uint32_t i = 0; i < dense_order; i++) {
        if (dense_sets[i].NoOverlap(s.dense_sets[i])) {
          no_overlap = true;
          break;
        }
      }
      for (std::uint32_t i = 0; i < sparse_order; i++) {
        if (sparse_sets[i].NoOverlap(s.sparse_sets[i])) {
          no_overlap = true;
          break;
        }
      }

      if (!no_overlap) {
        std::uint32_t diff_dim = -1U;
        for (std::uint32_t i = 0; i < dense_order; i++) {
          if (!(dense_sets[i] == s.dense.sets[i])) {
            ASSERT(diff_dim == -1U);  // FIXME
            diff_dim = i;
          }
        }
        for (std::uint32_t i = 0; i < sparse_order; i++) {
          if (!(sparse_sets[i] == s.sparse.sets[i])) {
            // ASSERT(diff_dim == -1U);  // FIXME
            diff_dim = i + dense_order;
          }
        }

        if (diff_dim == -1U) {
          // the two subspaces are equal, result is empty set.
          Reset();
        } else {
          if (diff_dim < dense_order) {
            dense_sets[diff_dim] -= s.dense_sets[diff_dim];
          } else {
            sparse_sets[diff_dim - dense_order] -=
                s.sparse_sets[diff_dim - dense_order];
          }
          if (empty()) {
            Reset();
          }
        }
      }
    }

    return (*this);
  }

  bool operator==(const PointSet<dense_order, sparse_order>& rhs)
      const {
    PointSet<dense_order, sparse_order> lhs = *(this);
    // DenseSet operations are faster, so make dense_sets
    // equality check first
    for (std::uint32_t i = 0; i < dense_order; i++) {
      if (!(lhs.dense_sets[i] == rhs.dense_sets[i])) {
        return false;
      }
    }
    for (std::uint32_t i = 0; i < sparse_order; i++) {
      if (!(lhs.sparse_sets[i] == rhs.sparse_sets[i])) {
        return false;
      }
    }
    return true;
  }

  void Print() const {
    for (std::uint32_t i = 0; i < dense_order; i++) {
      dense_sets[i].Print();
    }
    for (std::uint32_t i = 0; i < sparse_order; i++) {
      sparse_sets[i].Print();
    }
  }
};
