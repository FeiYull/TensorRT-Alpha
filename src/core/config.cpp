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

//! 从合并后的 map 里读一个必填字符串。缺失则抛异常。
std::string requireString(
    const std::unordered_map<std::string, std::string>& merged,
    const std::string& fullKey,
    const std::string& iniPath)
{
    const auto it = merged.find(fullKey);
    if (it == merged.end() || it->second.empty())
    {
        TRT_LOG_ERROR("Config: INI missing required key '" << fullKey
                      << "' in " << Paths::toDisplay(Paths::resolve(iniPath)));
        throw std::runtime_error("INI missing required key: " + fullKey +
                                 "  (file: " + Paths::toDisplay(Paths::resolve(iniPath)) +
                                 ")");
    }
    return it->second;
}

//! 从合并后的 map 里读 int，缺省返回 fallback。
int getIntOr(
    const std::unordered_map<std::string, std::string>& merged,
    const std::string& fullKey, int fallback)
{
    const auto it = merged.find(fullKey);
    if (it == merged.end() || it->second.empty()) { return fallback; }
    try {
        std::size_t consumed = 0;
        const int v = std::stoi(it->second, &consumed);
        if (consumed != it->second.size()) throw std::invalid_argument("trailing");
        return v;
    } catch (...) {
        throw std::runtime_error("INI key '" + fullKey +
                                 "' expects int, got '" + it->second + "'");
    }
}

}  // namespace

ModelConfig loadModelConfig(const std::string& iniPath)
{
    // 1. 读公共配置
    const std::string basePath = "configs/base.ini";
    const IniParser base = IniParser::load(basePath);

    // 2. 读特殊配置
    const IniParser special = IniParser::load(iniPath);

    // 3. 合并：special 覆盖 base
    std::unordered_map<std::string, std::string> merged = base.all();
    for (const auto& [k, v] : special.all()) {
        merged[k] = v;
    }

    // 4. 从合并结果解出 ModelConfig
    ModelConfig cfg;

    // 必填
    cfg.engine = requireString(merged, "model.engine", iniPath);
    cfg.classNamesFile = requireString(merged, "model.class_names_file", iniPath);

    // input_output_names（逗号分隔）
    {
        const auto it = merged.find("model.input_output_names");
        if (it == merged.end() || it->second.empty()) {
            TRT_LOG_ERROR("Config: model.input_output_names is empty in " << iniPath);
            throw std::runtime_error("INI missing required key: model.input_output_names");
        }
        std::istringstream iss(it->second);
        std::string token;
        while (std::getline(iss, token, ',')) {
            token = trim(token);
            if (!token.empty()) cfg.inputOutputNames.push_back(token);
        }
        if (cfg.inputOutputNames.empty()) {
            throw std::runtime_error("INI: input_output_names is empty: " + iniPath);
        }
    }

    // 可选（从合并结果读）
    cfg.batchSize = getIntOr(merged, "input.batch_size", cfg.batchSize);
    cfg.maxBatchSize = getIntOr(merged, "input.max_batch_size", cfg.maxBatchSize);

    // 输入逻辑维序（可选；缺省用模型的规范布局）
    if (const auto it = merged.find("input.layout"); it != merged.end())
    {
        const std::string v = trim(it->second);
        if (!v.empty() && !Layout::tryParse(v, cfg.layout))
        {
            TRT_LOG_ERROR("Config: invalid input.layout '" << v << "' in " << iniPath);
            throw std::runtime_error(
                "INI: invalid input.layout '" + v +
                "' (expect e.g. nchw / nhwc / ncdhw / ndhwc / chw / hwc)");
        }
    }

    // 空间维"意图值"：一般不需要设置（仅引擎该维为动态时有意义）
    cfg.dstH = getIntOr(merged, "input.dst_h", cfg.dstH);
    cfg.dstW = getIntOr(merged, "input.dst_w", cfg.dstW);

    // 5. extras：把合并后的 key-value 全存起来（短名 + 长名都存）
    for (const auto& [k, v] : merged) {
        cfg.extras[k] = v;
        const auto dot = k.find('.');
        if (dot != std::string::npos) {
            cfg.extras.emplace(k.substr(dot + 1), v);
        }
    }

    TRT_LOG_INFO("Config: loaded " << iniPath
                 << " (with " << basePath << ") "
                 << "(engine=" << cfg.engine
                 << ", batch=" << cfg.batchSize
                 << (cfg.maxBatchSize > 0 ? " (max=" + std::to_string(cfg.maxBatchSize) + ")" : "")
                 << ", layout=" << (cfg.layout.empty() ? "auto" : cfg.layout.str())
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