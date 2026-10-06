// =============================================================================
//  trt_alpha :: core :: model_config（实现）
// =============================================================================
#include "trt_alpha/core/model_config.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace trt_alpha::core {

void ModelConfig::setOrigin(const std::string& fullKey, const std::string& source)
{
    const auto it = std::find_if(origins.begin(), origins.end(),
                                 [&fullKey](const auto& p) { return p.first == fullKey; });
    if (it == origins.end())
    {
        origins.emplace_back(fullKey, source);   // 首次出现：追加，保序
    }
    else
    {
        it->second = source;                     // 已存在：就地改写，不改变顺序
    }
}

std::string ModelConfig::originOf(const std::string& fullKey,
                                  const std::string& fallback) const
{
    const auto it = std::find_if(origins.begin(), origins.end(),
                                 [&fullKey](const auto& p) { return p.first == fullKey; });
    return (it == origins.end()) ? fallback : it->second;
}

bool ModelConfig::wasRead(const std::string& fullKey) const
{
    if (readKeys.count(fullKey) != 0) { return true; }
    // 短名命中：读取方一般用短名（cfg.getFloat("conf_thresh", ...)），
    // 而展示用的是长名（"postprocess.conf_thresh"）。
    const auto dot = fullKey.find('.');
    return dot != std::string::npos && readKeys.count(fullKey.substr(dot + 1)) != 0;
}

std::string ModelConfig::getString(const std::string& key,
                                   const std::string& fallback) const
{
    markRead(key);
    const auto it = extras.find(key);
    return (it == extras.end()) ? fallback : it->second;
}

int ModelConfig::getInt(const std::string& key, int fallback) const
{
    markRead(key);
    const auto it = extras.find(key);
    if (it == extras.end() || it->second.empty())
    {
        return fallback;
    }
    try
    {
        std::size_t consumed = 0;
        const int v = std::stoi(it->second, &consumed);
        if (consumed != it->second.size())
        {
            throw std::invalid_argument("trailing");
        }
        return v;
    }
    catch (const std::exception&)
    {
        throw std::runtime_error("ModelConfig: key '" + key +
                                 "' expects int, got '" + it->second + "'");
    }
}

float ModelConfig::getFloat(const std::string& key, float fallback) const
{
    markRead(key);
    const auto it = extras.find(key);
    if (it == extras.end() || it->second.empty())
    {
        return fallback;
    }
    try
    {
        std::size_t consumed = 0;
        const float v = std::stof(it->second, &consumed);
        if (consumed != it->second.size())
        {
            throw std::invalid_argument("trailing");
        }
        return v;
    }
    catch (const std::exception&)
    {
        throw std::runtime_error("ModelConfig: key '" + key +
                                 "' expects float, got '" + it->second + "'");
    }
}

bool ModelConfig::getBool(const std::string& key, bool fallback) const
{
    markRead(key);
    const auto it = extras.find(key);
    if (it == extras.end() || it->second.empty())
    {
        return fallback;
    }
    const std::string& s = it->second;
    if (s == "true"  || s == "1" || s == "yes" || s == "on")  { return true;  }
    if (s == "false" || s == "0" || s == "no"  || s == "off") { return false; }
    throw std::runtime_error("ModelConfig: key '" + key +
                             "' expects bool, got '" + s + "'");
}

}  // namespace trt_alpha::core