// =============================================================================
//  trt_alpha :: core :: ini_parser（实现）
// =============================================================================
#include "trt_alpha/core/ini_parser.hpp"

#include "trt_alpha/core/logger.hpp"   //  新增
#include "trt_alpha/core/paths.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>

namespace trt_alpha::core {
namespace {

std::string trim(const std::string& s)
{
    const auto begin = s.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) { return {}; }
    const auto end = s.find_last_not_of(" \t\r\n");
    return s.substr(begin, end - begin + 1);
}

//! 去掉行内注释（# 或 ; 之后的内容）。简单处理，不识别引号。
std::string stripInlineComment(const std::string& s)
{
    auto pos = s.find_first_of("#;");
    return (pos == std::string::npos) ? s : s.substr(0, pos);
}

int parseInt(const std::string& key, const std::string& text)
{
    try
    {
        std::size_t consumed = 0;
        int v = std::stoi(text, &consumed);
        if (consumed != text.size())
        {
            throw std::invalid_argument("trailing");
        }
        return v;
    }
    catch (const std::exception&)
    {
        throw std::runtime_error("INI key '" + key + "' expects int, got '" + text + "'");
    }
}

float parseFloat(const std::string& key, const std::string& text)
{
    try
    {
        std::size_t consumed = 0;
        float v = std::stof(text, &consumed);
        if (consumed != text.size())
        {
            throw std::invalid_argument("trailing");
        }
        return v;
    }
    catch (const std::exception&)
    {
        throw std::runtime_error("INI key '" + key + "' expects float, got '" + text + "'");
    }
}

}  // namespace

std::string IniParser::makeKey(const std::string& section, const std::string& key)
{
    return section.empty() ? key : (section + "." + key);
}

IniParser IniParser::parseString(const std::string& content)
{
    IniParser ini;
    std::istringstream iss(content);
    std::string line;
    int lineNo = 0;
    std::string currentSection;

    while (std::getline(iss, line))
    {
        ++lineNo;
        // 去行尾注释
        line = stripInlineComment(line);
        line = trim(line);
        if (line.empty())
        {
            continue;
        }

        // 节头
        if (line.front() == '[')
        {
            if (line.back() != ']')
            {
                TRT_LOG_ERROR("INI: section header not closed at line " << lineNo);
                throw std::runtime_error("INI syntax error at line " +
                                         std::to_string(lineNo) +
                                         ": section header not closed");
            }
            currentSection = trim(line.substr(1, line.size() - 2));
            if (currentSection.empty())
            {
                TRT_LOG_ERROR("INI: empty section name at line " << lineNo);
                throw std::runtime_error("INI syntax error at line " +
                                         std::to_string(lineNo) +
                                         ": empty section name");
            }
            continue;
        }

        // 键值对
        const auto eq = line.find('=');
        if (eq == std::string::npos)
        {
            TRT_LOG_ERROR("INI: missing '=' at line " << lineNo);
            throw std::runtime_error("INI syntax error at line " +
                                     std::to_string(lineNo) + " (missing '=')");
        }
        const std::string key = trim(line.substr(0, eq));
        const std::string value = trim(line.substr(eq + 1));
        if (key.empty())
        {
            TRT_LOG_ERROR("INI: missing '=' at line " << lineNo);
            throw std::runtime_error("INI syntax error at line " +
                                     std::to_string(lineNo) + ": empty key");
        }
        ini.m_kv[makeKey(currentSection, key)] = value;
    }
    return ini;
}

IniParser IniParser::load(const std::string& path)
{
    const auto file = Paths::resolve(path);
    std::ifstream in(file);
    if (!in.is_open())
    {
        TRT_LOG_ERROR("INI: cannot open file " << Paths::toDisplay(file));
        throw std::runtime_error("cannot open INI file: " + Paths::toDisplay(file));
    }
    std::ostringstream oss;
    oss << in.rdbuf();

    TRT_LOG_DEBUG("INI: loaded " << Paths::toDisplay(file));

    return parseString(oss.str());
}

bool IniParser::has(const std::string& key) const
{
    return m_kv.count(key) != 0;
}

bool IniParser::has(const std::string& key, const std::string& section) const
{
    return m_kv.count(makeKey(section, key)) != 0;
}

std::string IniParser::getString(const std::string& key,
                                const std::string& fallback) const
{
    const auto it = m_kv.find(key);
    return (it == m_kv.end()) ? fallback : it->second;
}

std::string IniParser::getString(const std::string& key,
                                const std::string& section,
                                const std::string& fallback) const
{
    return getString(makeKey(section, key), fallback);
}

int IniParser::getInt(const std::string& key, int fallback) const
{
    const auto it = m_kv.find(key);
    if (it == m_kv.end() || it->second.empty()) { return fallback; }
    return parseInt(key, it->second);
}

int IniParser::getInt(const std::string& key, const std::string& section,
                     int fallback) const
{
    return getInt(makeKey(section, key), fallback);
}

float IniParser::getFloat(const std::string& key, float fallback) const
{
    const auto it = m_kv.find(key);
    if (it == m_kv.end() || it->second.empty()) { return fallback; }
    return parseFloat(key, it->second);
}

float IniParser::getFloat(const std::string& key, const std::string& section,
                         float fallback) const
{
    return getFloat(makeKey(section, key), fallback);
}

std::vector<std::string> IniParser::getStringList(const std::string& key,
                                                  const std::string& section,
                                                  char sep) const
{
    const std::string raw = getString(makeKey(section, key));
    if (raw.empty()) { return {}; }

    std::vector<std::string> out;
    std::istringstream iss(raw);
    std::string token;
    while (std::getline(iss, token, sep))
    {
        token = trim(token);
        if (!token.empty()) { out.push_back(token); }
    }
    return out;
}

}  // namespace trt_alpha::core