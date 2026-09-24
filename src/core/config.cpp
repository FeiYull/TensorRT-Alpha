// =============================================================================
//  trt_alpha :: core :: config（实现）
// =============================================================================
#include "trt_alpha/core/config.hpp"

#include "trt_alpha/core/ini_parser.hpp"
#include "trt_alpha/core/logger.hpp"
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

//! 从 INI 读一个必填字符串。缺失则抛异常。
std::string requireString(const IniParser& ini, const std::string& section,
                          const std::string& key, const std::string& iniPath)
{
    const std::string v = ini.getString(key, section, "");
    if (v.empty())
    {
        const std::string fullKey = section.empty() ? key : (section + "." + key);
        TRT_LOG_ERROR("Config: INI missing required key '" << fullKey
                      << "' in " << Paths::toDisplay(Paths::resolve(iniPath)));
        throw std::runtime_error("INI missing required key: " + fullKey +
                                 "  (file: " + Paths::toDisplay(Paths::resolve(iniPath)) +
                                 ")");
    }
    return v;
}

}  // namespace

ModelConfig loadModelConfig(const std::string& iniPath)
{
    const IniParser ini = IniParser::load(iniPath);

    ModelConfig cfg;

    // ---- 通用字段 ----
    cfg.engine = requireString(ini, "model", "engine", iniPath);
    cfg.classNamesFile = requireString(ini, "model", "class_names_file", iniPath);
    cfg.inputOutputNames = ini.getStringList("input_output_names", "model");
    if (cfg.inputOutputNames.empty())
    {
        TRT_LOG_ERROR("Config: model.input_output_names is empty in " << iniPath);
        throw std::runtime_error("INI missing required key: model.input_output_names");
    }
    cfg.batchSize = ini.getInt("batch_size", "model", cfg.batchSize);
    cfg.dstH = ini.getInt("dst_h", "input", cfg.dstH);
    cfg.dstW = ini.getInt("dst_w", "input", cfg.dstW);

    // ---- extras：把 INI 全部 key-value 存起来，模型自己取 ----
    // INI 内部 key 是 "section.key" 或 "key"（无节）
    // 同时存"短名"（去 section 前缀），方便模型用 "num_class" 取
    for (const auto& [k, v] : ini.all())
    {
        cfg.extras[k] = v;
        const auto dot = k.find('.');
        if (dot != std::string::npos)
        {
            const std::string shortKey = k.substr(dot + 1);
            // 短名不覆盖已有的长名（同名不同 section 时保留第一个）
            cfg.extras.emplace(shortKey, v);
        }
    }

    TRT_LOG_INFO("Config: loaded " << iniPath
                 << " (engine=" << cfg.engine
                 << ", batch=" << cfg.batchSize
                 << ", dst=" << cfg.dstW << "x" << cfg.dstH
                 << ", extras=" << cfg.extras.size() << " entries)");

    return cfg;
}

std::vector<ClassInfo> loadClassNamesFile(const std::string& txtPath)
{
    const auto file = Paths::resolve(txtPath);
    std::ifstream in(file);
    if (!in.is_open())
    {
        TRT_LOG_ERROR("Config: cannot open class names file "
                      << Paths::toDisplay(file));
        throw std::runtime_error("cannot open class names file: " +
                                 Paths::toDisplay(file));
    }

    std::vector<ClassInfo> out;
    std::string line;
    int lineNo = 0;
    while (std::getline(in, line))
    {
        ++lineNo;
        // 去注释和空白（# 和 ; 都当注释符）
        const auto hash = line.find_first_of("#;");
        if (hash != std::string::npos) { line = line.substr(0, hash); }
        line = trim(line);
        if (line.empty()) { continue; }

        std::istringstream iss(line);
        ClassInfo ci;
        int r = 0, g = 0, b = 0;
        if (!(iss >> ci.name >> r >> g >> b))
        {
            TRT_LOG_ERROR("Config: class names syntax error at line " << lineNo
                          << " in " << Paths::toDisplay(file)
                          << " (expected: name R G B)");
            throw std::runtime_error("class names file syntax error at line " +
                                     std::to_string(lineNo) + ": " +
                                     Paths::toDisplay(file) +
                                     "  (expected: name R G B)");
        }
        if (r < 0 || r > 255 || g < 0 || g > 255 || b < 0 || b > 255)
        {
            TRT_LOG_ERROR("Config: class names RGB out of [0,255] at line " << lineNo);
            throw std::runtime_error("class names file: RGB out of [0,255] at line " +
                                     std::to_string(lineNo));
        }
        ci.r = static_cast<std::uint8_t>(r);
        ci.g = static_cast<std::uint8_t>(g);
        ci.b = static_cast<std::uint8_t>(b);
        out.push_back(std::move(ci));
    }

    TRT_LOG_INFO("Config: class names file " << txtPath
                 << " (" << out.size() << " classes)");

    return out;
}

}  // namespace trt_alpha::core