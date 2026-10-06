// =============================================================================
//  trt_alpha :: core :: config（实现）
// =============================================================================
#include "trt_alpha/core/config.hpp"

#include "trt_alpha/core/engine.hpp"
#include "trt_alpha/core/ini_parser.hpp"
#include "trt_alpha/core/logger.hpp"
#include "trt_alpha/core/paths.hpp"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace trt_alpha::core {
namespace {

//! 取路径最后一段（"configs/u2net.ini" → "u2net.ini"）。
std::string baseNameOf(const std::string& path)
{
    const auto pos = path.find_last_of("/\\");
    return (pos == std::string::npos) ? path : path.substr(pos + 1);
}

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

    // 6. origins：记录每个【长名】键来自哪个文件（保序，供配置展示用）。
    //    先 base（骨架顺序），再模型 ini（新增追加；覆盖只改来源、不动位置）。
    const std::string baseName    = baseNameOf(basePath);
    const std::string specialName = baseNameOf(iniPath);
    for (const auto& k : base.keys())    { cfg.setOrigin(k, baseName); }
    for (const auto& k : special.keys()) { cfg.setOrigin(k, specialName); }

    // 7. 登记"本函数直接消费"的键。
    //    这些键是直接走 IniParser / cfg 字段读的（不走 getInt/getFloat/...）；
    //    其余的键（num_class、conf_thresh、scale、mean…）由模型 init 时
    //    调 getXxx 自动登记 —— 两处合起来 = "本次真正被读到"的全集。
    static const char* const kDirectKeys[] = {
        "model.engine", "model.class_names_file", "model.input_output_names",
        "input.batch_size", "input.max_batch_size", "input.layout",
        "input.dst_h", "input.dst_w",
    };
    for (const char* k : kDirectKeys) { cfg.markRead(k); }

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

// =============================================================================
//  logConfigBox —— 把"本次实际生效的配置"整体框出来
// =============================================================================
namespace {

//! 已知 section 的展示顺序。新增 section 时在这里补一项即可；
//! 漏补也不会丢内容（未列出的 section 会按首次出现顺序排到末尾）。
const char* const kSectionOrder[] = {
    "model", "input", "normalize", "postprocess", "pool", "output", "kernels",
};

struct KvRow
{
    std::string label;
    std::string value;
    std::string origin;
    std::string tag;      //!< 异常标记（目前只有 "unused"），空 = 正常
};

struct Block
{
    std::string        header;
    std::vector<KvRow> rows;
};

std::string padRight(const std::string& s, std::size_t width)
{
    return (s.size() >= width) ? s : (s + std::string(width - s.size(), ' '));
}

//! nvinfer1::Dims → "1 x 3 x 320 x 320"（动态维原样显示 -1）。
std::string dimsToString(const nvinfer1::Dims& d)
{
    if (d.nbDims <= 0) { return "scalar"; }
    std::ostringstream oss;
    for (int i = 0; i < d.nbDims; ++i)
    {
        if (i > 0) { oss << " x "; }
        oss << d.d[i];
    }
    return oss.str();
}

//! 物理内存格式的【短名】。TRT 的 formatDesc 是一长串（"Row major linear FP32
//! format (kLINEAR)"），会把框撑爆；而框架只接受 kLINEAR，所以简写即可，
//! 非 kLINEAR 用醒目的 non-linear[N] 标出（正常路径不会出现，出现就是有问题）。
std::string formatTag(const TensorDesc& t)
{
    return (t.format == nvinfer1::TensorFormat::kLINEAR)
               ? std::string("linear")
               : ("non-linear[" + std::to_string(static_cast<int>(t.format)) + "]");
}

}  // namespace

void logConfigBox(const ModelConfig& cfg,
                  const std::string& netName,
                  const std::string& iniPath,
                  const std::vector<TensorDesc>* io,
                  int resolvedBatch)
{
    // ---- 1. 收集全部【长名】键，保持 ini 出现顺序 ----
    std::vector<std::string> keys;
    keys.reserve(cfg.origins.size());
    for (const auto& [k, src] : cfg.origins)
    {
        (void)src;
        if (k.find('.') != std::string::npos) { keys.push_back(k); }
    }
    // extras 里可能有 origins 未覆盖的长名键（例如代码里直接塞进去的）：
    // 按字典序补在末尾，保证"框里看到的就是全部"。
    {
        std::vector<std::string> extra;
        for (const auto& [k, v] : cfg.extras)
        {
            (void)v;
            if (k.find('.') == std::string::npos) { continue; }
            if (std::find(keys.begin(), keys.end(), k) == keys.end()) { extra.push_back(k); }
        }
        std::sort(extra.begin(), extra.end());
        keys.insert(keys.end(), extra.begin(), extra.end());
    }

    // ---- 2. section 顺序：kSectionOrder 优先，未列出的按首次出现顺序追加 ----
    const auto sectionOf = [](const std::string& k) { return k.substr(0, k.find('.')); };

    std::vector<std::string> sections;
    for (const char* s : kSectionOrder)
    {
        const std::string prefix = std::string(s) + ".";
        const bool present = std::any_of(keys.begin(), keys.end(),
            [&prefix](const std::string& k) { return k.rfind(prefix, 0) == 0; });
        if (present) { sections.emplace_back(s); }
    }
    for (const auto& k : keys)
    {
        const std::string s = sectionOf(k);
        if (std::find(sections.begin(), sections.end(), s) == sections.end())
        {
            sections.push_back(s);
        }
    }

    // ---- 3. 组装分块（每个 section 一块；再加一块 [resolved]）----
    std::vector<Block> blocks;
    for (const auto& s : sections)
    {
        Block b;
        b.header = "[" + s + "]";
        for (const auto& k : keys)
        {
            if (sectionOf(k) != s) { continue; }
            const auto it = cfg.extras.find(k);
            if (it == cfg.extras.end()) { continue; }
            // 没被任何消费者读过的键 → 标 [unused]。
            // 这就是"ini 里写了但本次路径根本不生效"的可见证据。
            const std::string tag = cfg.wasRead(k) ? std::string() : std::string("unused");
            b.rows.push_back({ k.substr(s.size() + 1), it->second, cfg.originOf(k), tag });
        }
        if (!b.rows.empty()) { blocks.push_back(std::move(b)); }
    }

    if (io != nullptr)
    {
        Block b;
        b.header = "[resolved]   engine truth (shape = engine declaration; -1 = dynamic axis)";
        if (resolvedBatch > 0)
        {
            const int iniBatch = cfg.getInt("input.batch_size", 0);
            const std::string note =
                (iniBatch > 0 && iniBatch != resolvedBatch)
                    ? "(ini " + std::to_string(iniBatch) + ", corrected by engine)"
                    : "(engine-confirmed)";
            b.rows.push_back({ "batch", std::to_string(resolvedBatch), note, "" });
        }
        for (const auto& t : *io)
        {
            std::ostringstream v;
            v << t.name << "   " << dimsToString(t.shape) << "   "
              << nameOf(t.dtype) << " (" << sizeOf(t.dtype) << "B)   "
              << formatTag(t);
            b.rows.push_back({ t.isInput ? "in" : "out", v.str(), "", "" });
        }
        if (!b.rows.empty()) { blocks.push_back(std::move(b)); }
    }

    if (blocks.empty()) { return; }

    // ---- 4. 列宽 ----
    std::size_t labelW  = 0;
    std::size_t valueW  = 0;
    std::size_t originW = 0;
    for (const auto& b : blocks)
    {
        for (const auto& r : b.rows)
        {
            labelW  = std::max(labelW,  r.label.size());
            valueW  = std::max(valueW,  r.value.size());
            originW = std::max(originW, r.origin.size());
        }
    }

    // ---- 5. 拼正文行 ----
    std::vector<std::string> lines;
    bool hasUnused = false;
    for (std::size_t i = 0; i < blocks.size(); ++i)
    {
        if (i > 0) { lines.emplace_back(); }          // 块间空行
        lines.push_back(blocks[i].header);
        for (const auto& r : blocks[i].rows)
        {
            std::string l = "  " + padRight(r.label, labelW) + " = " +
                            padRight(r.value, valueW);
            if (originW > 0) { l += "   " + padRight(r.origin, originW); }
            // tag 不占固定列：label/value/origin 都已定宽，
            // 于是 tag 天然落在同一列；没有 tag 的行只是更短（尾部空格会被裁掉）。
            if (!r.tag.empty())
            {
                l += "   [" + r.tag + "]";
                hasUnused = hasUnused || (r.tag == "unused");
            }
            while (!l.empty() && l.back() == ' ') { l.pop_back(); }
            lines.push_back(std::move(l));
        }
    }
    if (hasUnused)
    {
        lines.emplace_back();
        lines.push_back("  [unused] = 该键写在 ini 里，但本次路径没有任何消费者（写了不生效）");
    }

    // ---- 6. 标题行：左 = [CONFIG] <net>，右 = ini 路径 ----
    const std::string left  = "[CONFIG] " + netName;
    const std::string right = iniPath;

    std::size_t width = 0;
    for (const auto& l : lines) { width = std::max(width, l.size()); }
    width = std::max(width, left.size() + right.size() + 2);

    lines.insert(lines.begin(), padRight(left, width - right.size()) + right);

    detail::logBox(lines, '=');
}

}  // namespace trt_alpha::core