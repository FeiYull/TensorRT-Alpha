// =============================================================================
//  trt_alpha :: core :: model_config（实现）
// =============================================================================
#include "trt_alpha/core/model_config.hpp"

#include <stdexcept>
#include <string>

namespace trt_alpha::core {

std::string ModelConfig::getString(const std::string& key,
                                   const std::string& fallback) const
{
    const auto it = extras.find(key);
    return (it == extras.end()) ? fallback : it->second;
}

int ModelConfig::getInt(const std::string& key, int fallback) const
{
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