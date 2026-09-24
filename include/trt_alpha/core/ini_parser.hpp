// =============================================================================
//  trt_alpha :: core :: ini_parser
// -----------------------------------------------------------------------------
//  IniParser —— 极简 INI 解析器（支持节）。
//
//  支持的语法：
//    [section]              节头
//    key = value            键值对（在节内）
//    key = value            键值对（无节，全局）
//    # 注释                 整行注释
//    ; 注释                 整行注释
//    行内 # 注释            value # comment
//
//  不支持：
//    * 多行值
//    * 引号
//    * 嵌套节（[a.b.c] 当普通节名处理）
//    * 转义
//
//  查询：
//    ini.getString("key")                // 全局 key
//    ini.getString("section.key")        // 节内 key
//    ini.getString("key", "section")     // 同上（另一种写法）
//
//  读取失败 / 语法错误 → 抛 std::runtime_error（消息带上下文）。
// =============================================================================
#pragma once

#include <string>
#include <unordered_map>
#include <vector>

namespace trt_alpha::core {

class IniParser
{
public:
    //! 解析文件。文件不可读 / 语法非法抛异常。
    static IniParser load(const std::string& path);

    //! 解析字符串（测试用）。
    static IniParser parseString(const std::string& content);

    [[nodiscard]] bool has(const std::string& key) const;
    [[nodiscard]] bool has(const std::string& key, const std::string& section) const;

    [[nodiscard]] std::string getString(const std::string& key,
                                       const std::string& fallback = "") const;
    [[nodiscard]] std::string getString(const std::string& key,
                                       const std::string& section,
                                       const std::string& fallback) const;

    [[nodiscard]] int getInt(const std::string& key, int fallback) const;
    [[nodiscard]] int getInt(const std::string& key, const std::string& section,
                            int fallback) const;

    [[nodiscard]] float getFloat(const std::string& key, float fallback) const;
    [[nodiscard]] float getFloat(const std::string& key, const std::string& section,
                                float fallback) const;

    //! 逗号分隔字符串 → vector<string>。
    [[nodiscard]] std::vector<std::string>
    getStringList(const std::string& key, const std::string& section = "",
                  char sep = ',') const;
    
    //! 返回全部 key-value（key 格式："section.key" 或 "key"）。
    [[nodiscard]] const std::unordered_map<std::string, std::string>& all() const noexcept
    {
        return m_kv;
    }

private:
    //! 内部 key 格式："section.key" 或 "key"（无节）
    std::unordered_map<std::string, std::string> m_kv;

    static std::string makeKey(const std::string& section, const std::string& key);
};

}  // namespace trt_alpha::core