#pragma once

#include <vector>

template<typename T>
class TaggedMapDimIterator
{
 public:
  std::pair<size_t, T> operator*()
  {
    return std::make_pair(dim_, *iterator_);
  }

  TaggedMapDimIterator<T>& operator++()
  {
    ++iterator_;
    ++dim_;
    return *this;
  }

  bool operator!=(const TaggedMapDimIterator& other)
  {
    return iterator_ != other.iterator_;
  }

 private:
  using BaseIter = typename std::vector<T>::iterator;

  BaseIter iterator_;
  size_t dim_;
  size_t size_;

  TaggedMapDimIterator(const BaseIter& iter,
                       size_t start)
  : iterator_(iter), dim_(start) {}

  static TaggedMapDimIterator start(std::vector<T>& vec)
  {
    return TaggedMapDimIterator(vec.begin(), 0UL);
  }

  static TaggedMapDimIterator end(std::vector<T>& vec)
  {
    return TaggedMapDimIterator(vec.end(), 0UL);
  }
};

template<typename T>
class ReverseTaggedMapDimIterator
{
 public:
  std::pair<size_t, T> operator*()
  {
    return std::make_pair(dim_, *iterator_);
  }

  TaggedMapDimIterator<T>& operator++()
  {
    ++iterator_;
    --dim_;
    return *this;
  }

  bool operator!=(const ReverseTaggedMapDimIterator& other)
  {
    return iterator_ != other.iterator_;
  }

 private:
  using BaseIter = typename std::vector<T>::reverse_iterator;

  BaseIter iterator_;
  size_t dim_;
  size_t size_;

  ReverseTaggedMapDimIterator(const BaseIter& iter,
                              size_t start)
  : iterator_(iter), dim_(start) {}

  static ReverseTaggedMapDimIterator start(std::vector<T>& vec)
  {
    return ReverseTaggedMapDimIterator(vec.rbegin(), vec.size()-1);
  }

  static ReverseTaggedMapDimIterator end(std::vector<T>& vec)
  {
    return ReverseTaggedMapDimIterator(vec.rend(), 0UL);
  }
};

struct NoTag
{
};

template<typename MapT, typename InTagT, typename OutTagT = NoTag>
struct TaggedMap
{
  using Map    = MapT;
  using InTag  = InTagT;
  using OutTag = OutTagT;

  Map map;
  std::vector<InTag> in_tags;
  std::vector<OutTag> out_tags;

  TaggedMap(Map&& map,
            std::vector<InTag>&& in_tags,
            std::vector<OutTag>&& out_tags)
    : map(std::move(map)), in_tags(std::move(in_tags)),
      out_tags(std::move(out_tags))
  {
  }
  TaggedMap(const TaggedMap<MapT, InTagT, OutTagT>& other)
    : map(other.map), in_tags(other.in_tags), out_tags(other.out_tags)
  {
  }

  TaggedMap<MapT, InTagT, OutTagT>&
  operator=(TaggedMap<MapT, InTagT, OutTagT>&& other)
  {
    map = std::move(other.map);
    in_tags = std::move(other.in_tags);
    out_tags = std::move(other.out_tags);
    return *this;
  }

  TaggedMap<MapT, InTag, OutTag> Copy() const;
  TaggedMap<MapT, InTag, OutTag> TagLikeThis(MapT&& map) const;

  TaggedMapDimIterator<InTag> in_begin();
  TaggedMapDimIterator<InTag> in_end();
  TaggedMapDimIterator<InTag> in_rbegin();
  TaggedMapDimIterator<InTag> in_rend();

  TaggedMapDimIterator<OutTag> out_begin();
  TaggedMapDimIterator<OutTag> out_end();
  TaggedMapDimIterator<OutTag> out_rbegin();
  TaggedMapDimIterator<OutTag> out_rend();


  bool InvolvesDims(isl_dim_type dim_type, size_t first, size_t n) const
  {
    return map.InvolvesDims(dim_type, first, n);
  }
  size_t NumDims(isl_dim_type dim_type) const
  {
    return map.NumDims(dim_type);
  }
};

#define ISL_TAGGED_MAP_BINARY_OP_IMPL(NAME)                  \
  template<typename MapT, typename InTagT, typename OutTagT> \
  TaggedMap<MapT, InTagT, OutTagT>                           \
  NAME(TaggedMap<MapT, InTagT, OutTagT>&& map1,              \
       MapT&& map2)                                          \
  {                                                          \
    return TaggedMap<MapT, InTagT, OutTagT>(                 \
      NAME(std::move(map1.map), std::move(map2)),            \
      std::move(map1.in_tags),                               \
      std::move(map1.out_tags)                               \
    );                                                       \
  }                                                          \
  template<typename MapT, typename InTagT, typename OutTagT> \
  TaggedMap<MapT, InTagT, OutTagT>                           \
  NAME(TaggedMap<MapT, InTagT, OutTagT>&& map1,              \
       TaggedMap<MapT, InTagT, OutTagT>&& map2)              \
  {                                                          \
    return TaggedMap<MapT, InTagT, OutTagT>(                 \
      NAME(std::move(map1.map), std::move(map2.map)),        \
      std::move(map1.in_tags),                               \
      std::move(map1.out_tags)                               \
    );                                                       \
  }

ISL_TAGGED_MAP_BINARY_OP_IMPL(ApplyRange)
ISL_TAGGED_MAP_BINARY_OP_IMPL(Subtract)

template<typename MapT, typename InTagT, typename OutTagT>
TaggedMap<MapT, InTagT, OutTagT>
ProjectDims(TaggedMap<MapT, InTagT, OutTagT>&& map,
            isl_dim_type dim_type,
            size_t first,
            size_t n)
{
  if (dim_type == isl_dim_in)
  {
    auto first_it = map.in_tags.begin() + first;
    auto last_it = map.in_tags.begin() + first + n;
    map.in_tags.erase(first_it, last_it);
  }
  else if (dim_type == isl_dim_out)
  {
    auto first_it = map.out_tags.begin() + first;
    auto last_it = map.out_tags.begin() + first + n;
    map.out_tags.erase(first_it, last_it);
  }
  else
  {
    throw std::logic_error("unimplemented");
  }

  return TaggedMap<MapT, InTagT, OutTagT>(
    ProjectDims(std::move(map.map), dim_type, first, n),
    std::move(map.in_tags),
    std::move(map.out_tags)
  );
}
