#pragma once

#include <functional>
#include <iostream>
#include <set>

#include <boost/algorithm/string/join.hpp>

template<typename T>
struct HashableSet;

template<typename T>
struct std::hash<HashableSet<T>>;

/**
 * @brief Immutable set that can be hashed.
 * 
 * Functions `Emplace` and `Erase` can be used to create a _new_ set. To avoid
 * copying, the original set can be moved into these functions.
 */
template<typename T>
struct HashableSet
{
  HashableSet() : set_(), hash_(0)
  {
  }

  template<typename RangeT>
  HashableSet(RangeT init_range) :
    set_(init_range.begin(), init_range.end()), hash_(0)
  {
  }

  auto begin() const
  {
    return set_.begin();
  }

  auto end() const
  {
    return set_.end();
  }

  auto Size() const
  {
    return set_.size();
  }

 private:
  std::set<T> set_;
  size_t hash_;

  template<typename... Args>
  void Emplace(Args&&... v)
  {
    auto [it, inserted] = set_.emplace(std::forward<Args>(v)...);
    if (inserted)
    {
      hash_ ^= std::hash<T>()(*it);
    }
  }

  void Erase(const T& key)
  {
    auto num_erased = set_.erase(key);
    if (num_erased == 1)
    {
      hash_ ^= std::hash<T>()(key);
    }
  }

  size_t Hash() const
  {
    return hash_;
  }

  template<typename... Args>
  friend HashableSet<T> Emplace(HashableSet<T> set, Args&&... args)
  {
    set.Emplace(std::forward<Args>(args)...);
    return set;
  }

  friend HashableSet<T> Erase(HashableSet<T> set, const T& key)
  {
    set.Erase(key);
    return set;
  }

  friend std::hash<HashableSet<T>>;

  friend
  bool operator==(const HashableSet<T>& set1, const HashableSet<T>& set2)
  {
    if (set1.Hash() != set2.Hash())
    {
      return false;
    }

    return set1.set_ == set2.set_;
  }
};

template<typename T>
struct std::hash<HashableSet<T>>
{
  std::size_t operator()(const HashableSet<T>& set) const noexcept
  {
    return set.Hash();
  }
};

template<typename T>
std::ostream& operator<<(std::ostream& os, const HashableSet<T>& set)
{
  std::vector<std::string> set_str;
  for (auto key : set)
  {
    set_str.push_back(std::to_string(key));
  }
  os << "{" << boost::algorithm::join(set_str, ", ") << "}";
  return os;
}
