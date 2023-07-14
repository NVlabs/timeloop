#pragma once

#include <functional>

template<typename T>
struct ParetoFrontier
{
  struct Iterator
  {
    bool operator!=(const Iterator& other)
    {
      return it_ != other.it_;
    }
    bool operator==(const Iterator& other)
    {
      return it_ == other.it_;
    }

    const T& operator*()
    {
      return *(*it_);
    }

    Iterator& operator++()
    {
      ++it_;
      while (it_ != end_ && !(*it_))
      {
        ++it_;
      }
      return *this;
    }

   private:
    Iterator(typename std::vector<std::optional<T>>::const_iterator it,
             typename std::vector<std::optional<T>>::const_iterator end) :
      it_(std::move(it)), end_(std::move(end))
    {
    }

    typename std::vector<std::optional<T>>::const_iterator it_;
    typename std::vector<std::optional<T>>::const_iterator end_;

    friend ParetoFrontier;
  };

  ParetoFrontier() : arr_(), size_(0) {}

  Iterator begin() const
  {
    return Iterator(arr_.begin(), arr_.end());
  }

  Iterator end() const
  {
    return Iterator(arr_.end(), arr_.end());
  }

  void Insert(const T& element)
  {
    bool inserted = false;
    for (auto& e : arr_)
    {
      if (!e)
      {
        continue;
      }

      if (element > *e)
      {
        return;
      }
      else if (element < *e)
      {
        e = std::nullopt;
        size_ -= 1;
        if (!inserted)
        {
          e = element;
          inserted = true;
          size_ += 1;
        }
      }
    }
    if (!inserted)
    {
      arr_.emplace_back(element);
    }

    if (size_ < arr_.size() / 2)
    {
      size_t i = 0;
      for (; i < arr_.size(); ++i)
      {
        if (!arr_.at(i))
        {
          ++i;
          break;
        }
      }

      for (size_t j = i; j < arr_.size(); ++j)
      {
        if (arr_.at(j))
        {
          arr_.at(i) = arr_.at(j);
        }
      }

      assert(i == size_);
      arr_.resize(i);
    }
  }

  void Insert(const ParetoFrontier<T>& other)
  {
    for (const auto& e : other)
    {
      Insert(e);
    }
  }

 private:
  std::vector<std::optional<T>> arr_;
  size_t size_;
};

template<typename T>
ParetoFrontier<T>
CombineParetoFrontiers(const ParetoFrontier<T>& frontier1,
                       const ParetoFrontier<T>& frontier2,
                       const std::function<T(const T&, const T&)>& op)
{
  auto pareto = ParetoFrontier<T>();
  for (const auto& e1 : frontier1)
  {
    for (const auto& e2 : frontier2)
    {
      pareto.Insert(op(e1, e2));
    }
  }
  return pareto;
}