// =============================================================================
//  trt_alpha :: core :: layout
// -----------------------------------------------------------------------------
//  Layout —— 张量的【逻辑维度次序】描述（与物理内存排布无关）。
//
//  为什么不用 enum：
//    维序在生态里是开放的 —— 2D 图像 NCHW / NHWC，3D NCDHW / NDHWC，
//    无 batch 的 CHW / HWC，1D 音频 NCW / NWC，视频 NTCHW，
//    以及昇腾 CANN 真实存在的 CHWN / HWCN / DHWCN …… 枚举写死就永远追不上。
//    因此用"轴字母串"表达：字符串长度 == 张量秩，每个字母标识一根轴的语义。
//    → 任意排列、任意秩（≤ 8）都支持，新增布局【零代码改动】。
//
//  约定字母（不区分大小写）：
//    N = batch   C = channel   D = depth   H = height
//    W = width   T = time      E = embed   ? = 未标注（对应昇腾 "ND" 任意格式）
//  约束：每个字母至多出现一次（'?' 可重复）。
//
//  ⚠ 这里描述的是【逻辑次序】（ICudaEngine::getTensorShape 返回的次序），
//     不是【物理格式】（TensorRT 的 TensorFormat、昇腾的 NC1HWC0 等分块格式）。
//     物理格式由后端自己处理，框架只做守卫，见 core::TensorDesc::format。
// =============================================================================
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>

namespace trt_alpha::core {

class Layout
{
public:
    static constexpr int kMaxRank = 8;   // 与 nvinfer1::Dims::MAX_DIMS 一致

    // 轴字母
    static constexpr char kBatch   = 'N';
    static constexpr char kChannel = 'C';
    static constexpr char kDepth   = 'D';
    static constexpr char kHeight  = 'H';
    static constexpr char kWidth   = 'W';
    static constexpr char kTime    = 'T';
    static constexpr char kEmbed   = 'E';
    static constexpr char kAny     = '?';   // 未标注轴（长度仍计入秩）

    //! 空布局 = 未指定。
    constexpr Layout() noexcept = default;

    //! 从轴字母串构造（不区分大小写）。
    //! 非法（空 / 超长 / 未知字母 / 字母重复）抛 std::runtime_error。
    constexpr explicit Layout(std::string_view axes)
    {
        if (axes.empty() || axes.size() > static_cast<std::size_t>(kMaxRank))
        {
            throw std::runtime_error(
                "Layout: rank must be in [1, " + std::to_string(kMaxRank) + "]");
        }
        m_rank = static_cast<std::int8_t>(axes.size());
        for (int i = 0; i < m_rank; ++i)
        {
            const char c = toUpper(axes[static_cast<std::size_t>(i)]);
            if (!isKnownAxis(c))
            {
                throw std::runtime_error(
                    std::string("Layout: unknown axis letter '") + c +
                    "' (expect one of N C D H W T E ?)");
            }
            for (int j = 0; j < i; ++j)
            {
                if (m_axes[static_cast<std::size_t>(j)] == c && c != kAny)
                {
                    throw std::runtime_error(
                        std::string("Layout: duplicate axis letter '") + c + "'");
                }
            }
            m_axes[static_cast<std::size_t>(i)] = c;
        }
    }

    //! 非抛出版本：失败返回 false 且 out 保持不变。
    [[nodiscard]] static bool tryParse(std::string_view axes, Layout& out) noexcept
    {
        try
        {
            const Layout parsed(axes);
            out = parsed;
            return true;
        }
        catch (...)
        {
            return false;
        }
    }

    //! 按秩给出约定默认布局：3 → CHW、4 → NCHW、5 → NCDHW；其余返回空（调用方须显式声明）。
    [[nodiscard]] static Layout defaultForRank(int rank) noexcept
    {
        switch (rank)
        {
        case 3:  return Layout{"CHW"};
        case 4:  return Layout{"NCHW"};
        case 5:  return Layout{"NCDHW"};
        default: return Layout{};
        }
    }

    [[nodiscard]] constexpr bool empty() const noexcept { return m_rank == 0; }
    [[nodiscard]] constexpr int  rank()  const noexcept { return m_rank; }

    //! 轴字母所在的位置下标；不存在返回 -1。
    [[nodiscard]] constexpr int indexOf(char axis) const noexcept
    {
        const char c = toUpper(axis);
        for (int i = 0; i < m_rank; ++i)
        {
            if (m_axes[static_cast<std::size_t>(i)] == c) { return i; }
        }
        return -1;
    }
    [[nodiscard]] constexpr bool has(char axis) const noexcept { return indexOf(axis) >= 0; }

    [[nodiscard]] constexpr char at(int index) const noexcept
    {
        return (index >= 0 && index < m_rank) ? m_axes[static_cast<std::size_t>(index)] : '\0';
    }

    //! 规范化的轴字母串（日志 / 异常信息用）。
    [[nodiscard]] std::string str() const
    {
        return std::string(m_axes.data(), static_cast<std::size_t>(m_rank));
    }

    [[nodiscard]] constexpr bool operator==(const Layout& o) const noexcept
    {
        if (m_rank != o.m_rank) { return false; }
        for (int i = 0; i < m_rank; ++i)
        {
            if (m_axes[static_cast<std::size_t>(i)] != o.m_axes[static_cast<std::size_t>(i)])
            {
                return false;
            }
        }
        return true;
    }
    [[nodiscard]] constexpr bool operator!=(const Layout& o) const noexcept { return !(*this == o); }

    // ---- 常用布局常量 ----
    // 声明与定义分离：类内是自身的不完整类型，无法就地初始化；
    // 定义见 src/core/layout.cpp（编译期常量初始化，无静态初始化顺序问题）。
    static const Layout NCHW;
    static const Layout NHWC;
    static const Layout NCDHW;
    static const Layout NDHWC;
    static const Layout CHW;
    static const Layout HWC;

private:
    static constexpr char toUpper(char c) noexcept
    {
        return (c >= 'a' && c <= 'z') ? static_cast<char>(c - 'a' + 'A') : c;
    }
    static constexpr bool isKnownAxis(char c) noexcept
    {
        return c == kBatch || c == kChannel || c == kDepth || c == kHeight ||
               c == kWidth || c == kTime || c == kEmbed || c == kAny;
    }

    std::array<char, kMaxRank> m_axes{};   // 已大写；前 m_rank 个有效
    std::int8_t                m_rank = 0;
};

}  // namespace trt_alpha::core
