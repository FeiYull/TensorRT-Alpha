// =============================================================================
//  trt_alpha :: core :: model_registry（实现）
// =============================================================================
#include "trt_alpha/core/model_registry.hpp"

#include "trt_alpha/core/logger.hpp"

#include <sstream>

namespace trt_alpha {

ModelRegistry& ModelRegistry::instance() noexcept
{
    static ModelRegistry registry;
    return registry;
}

bool ModelRegistry::add(const std::string& name, Factory factory)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    const bool inserted = m_factories.emplace(name, factory).second;
    if (inserted)
    {
        TRT_LOG_DEBUG("ModelRegistry: registered '" << name << "'");
    }
    else
    {
        TRT_LOG_WARN("ModelRegistry: duplicate registration for '" << name
                     << "' (ignored)");
    }
    return inserted;
}

std::unique_ptr<IModel> ModelRegistry::create(const std::string& name) const
{
    std::lock_guard<std::mutex> lock(m_mutex);
    const auto it = m_factories.find(name);
    if (it == m_factories.end())
    {
        std::ostringstream oss;
        oss << "unknown model '" << name << "'. Registered models:";
        for (const auto& [key, factory] : m_factories)
        {
            oss << " " << key;
        }
        if (m_factories.empty())
        {
            oss << " (none - model library not linked with WHOLE_ARCHIVE?)";
        }
        const std::string msg = oss.str();
        TRT_LOG_ERROR("ModelRegistry: " << msg);
        throw std::runtime_error(msg);
    }
    return it->second();
}

std::vector<std::string> ModelRegistry::names() const
{
    std::lock_guard<std::mutex> lock(m_mutex);
    std::vector<std::string> result;
    result.reserve(m_factories.size());
    for (const auto& [key, factory] : m_factories)
    {
        result.push_back(key);
    }
    return result;
}

}  // namespace trt_alpha