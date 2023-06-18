// represents YAML maps
#include <map>
// for type-safe unions of YAML types
#include <variant>
// for YAML arrays
#include <vector>
// for std errors from variants
#include <stdexcept>
#include <memory>
#include <iostream>
// for asserts
#include <cassert>

namespace structured_config {
class CCRet; // forward definition

// Literal value types possible in a YAML file
using YAMLLiteral = std::variant<
  std::string, long long, double, bool
>;
// YAML vector representation
using YAMLVector = std::vector<std::unique_ptr<CCRet>>;
// YAML map representation
using YAMLMap = std::map<std::string, std::unique_ptr<CCRet>>;
// YAML value types possible
using YAMLType = std::variant<
  std::monostate,
  YAMLLiteral,
  YAMLVector,
  YAMLMap
>;

class CCRet
{
  private:
    YAMLType data_ = nullptr;

  public:
    /** VALUE RESOLUTION **/
    // unpacks a literal CCRet
    inline YAMLLiteral& GetValue()
    { return std::get<YAMLLiteral>(data_); } 
    // resolves a map CCRet
    inline CCRet& At(const std::string& key)
    { return *std::get<YAMLMap>(data_).at(key); }
    // resolves a list CCRet
    inline CCRet& At(YAMLVector::size_type index)
    { return *std::get<YAMLVector>(data_).at(index); }

    CCRet& operator [](int idx);

    std::vector<std::string> getMapKeys();

    /** TYPE RESOLUTION **/
    // we do not need isArray as isArray is only used for LNode, which is bypassed
    // when we do dynamic checking
    inline bool isLiteral()
    { return std::holds_alternative<YAMLLiteral>(data_); }
    inline bool isList() const
    { return std::holds_alternative<YAMLVector>(data_); }
    inline bool isMap() const
    { return std::holds_alternative<YAMLMap>(data_); }

    /** contained **/
    bool exists(std::string name) const;

    // Michael added this, no clue what it does for now
    template<typename... ArgsT>
    void EmplaceBack(ArgsT&&... args)
    {
      std::get<YAMLVector>(data_).push_back(
        std::make_unique<CCRet>(std::forward<ArgsT>(args)...)
      );
    }

    size_t Size() const
    {
      return std::visit(
        [] (auto&& data)
        {
          using DataT = std::decay_t<decltype(data)>;
          if constexpr (std::is_same_v<DataT, YAMLLiteral>)
          {
            return (size_t)1;
          } else if constexpr(std::is_same_v<DataT, std::monostate>) {
            return (size_t)0;
          } else
          {
            return data.size();
          }
        },
        data_
      );
    }

    // Michael added this, no clue what it does for now
    template<typename T>
    static CCRet Literal(const T& val)
    {
      auto ret = CCRet();
      ret.data_ = val;
      return ret;
    }

    static CCRet Vector()
    {
      auto ret = CCRet();
      ret.data_ = YAMLVector();
      return ret;
    }

    static CCRet Map()
    {
      auto ret = CCRet();
      ret.data_ = YAMLMap();
      return ret;
    }

    CCRet() : data_() {}
};
}