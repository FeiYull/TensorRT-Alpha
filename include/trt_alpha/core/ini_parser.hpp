// =============================================================================
//  trt_alpha :: core :: ini_parser
// -----------------------------------------------------------------------------
//  IniParser -- a minimal INI parser (sections supported).
//
//  Supported syntax:
//    [section]              section header
//    key = value            key/value pair (inside a section)
//    key = value            key/value pair (no section, global)
//    # comment              whole-line comment
//    ; comment              whole-line comment
//    inline # comment       value # comment
//
//  Not supported:
//    * multi-line values
//    * quotes
//    * nested sections ([a.b.c] is treated as an ordinary section name)
//    * escapes
//
//  Lookup:
//    ini.getString("key")                // global key
//    ini.getString("section.key")        // key inside a section
//    ini.getString("key", "section")     // same as above (alternative form)
//
//  A read failure / syntax error -> throws std::runtime_error (with context).
// =============================================================================
#pragma once

#include <string>
#include <unordered_map>
#include <vector>

namespace trt_alpha::core {

class IniParser
{
public:
    //! Parse a file. An unreadable file or invalid syntax throws.
    static IniParser load(const std::string& path);

    //! Parse a string (for tests).
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

    //! Comma-separated string -> vector<string>.
    [[nodiscard]] std::vector<std::string>
    getStringList(const std::string& key, const std::string& section = "",
                  char sep = ',') const;

    //! Return every key/value pair (keys look like "section.key" or "key").
    [[nodiscard]] const std::unordered_map<std::string, std::string>& all() const noexcept
    {
        return m_kv;
    }

    //! [File order] of all keys (first occurrence; duplicate keys are recorded
    //! once). Used to display the config exactly as the ini file has it (an
    //! unordered_map is unordered by itself).
    [[nodiscard]] const std::vector<std::string>& keys() const noexcept
    {
        return m_order;
    }

private:
    //! Internal key format: "section.key", or "key" when there is no section.
    std::unordered_map<std::string, std::string> m_kv;
    //! First-occurrence order of the keys in m_kv (kept strictly in sync with
    //! m_kv: append-only).
    std::vector<std::string> m_order;

    static std::string makeKey(const std::string& section, const std::string& key);
};

}  // namespace trt_alpha::core
