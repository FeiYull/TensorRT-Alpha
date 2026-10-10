// =============================================================================
//  trt_alpha :: core :: layout
// -----------------------------------------------------------------------------
//  Layout -- the [logical axis order] of a tensor (unrelated to how it is laid
//  out physically in memory).
//
//  Why not an enum:
//    Axis orders are open-ended in the ecosystem -- 2D images are NCHW / NHWC,
//    3D is NCDHW / NDHWC, batchless is CHW / HWC, 1D audio is NCW / NWC, video
//    is NTCHW, and Ascend CANN really has CHWN / HWCN / DHWCN ... a fixed enum
//    can never keep up. So it is expressed as an axis letter string: the string
//    length equals the tensor rank and every letter names one axis's meaning.
//    -> Any permutation and any rank (<= 8) is supported; adding a layout takes
//       [zero code changes].
//
//  Letters (case-insensitive):
//    N = batch   C = channel   D = depth   H = height
//    W = width   T = time      E = embed   ? = unlabelled (Ascend's "ND" any-format)
//  Constraint: each letter appears at most once ('?' may repeat).
//
//  NOTE This describes the [logical order] (the order returned by
//  ICudaEngine::getTensorShape), not the [physical format] (TensorRT's
//  TensorFormat, Ascend's blocked NC1HWC0, ...). The physical format is handled
//  by the backend; the framework only guards it, see core::TensorDesc::format.
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
    static constexpr int kMaxRank = 8;   // same as nvinfer1::Dims::MAX_DIMS

    // Axis letters
    static constexpr char kBatch   = 'N';
    static constexpr char kChannel = 'C';
    static constexpr char kDepth   = 'D';
    static constexpr char kHeight  = 'H';
    static constexpr char kWidth   = 'W';
    static constexpr char kTime    = 'T';
    static constexpr char kEmbed   = 'E';
    static constexpr char kAny     = '?';   // unlabelled axis (still counts towards the rank)

    //! An empty layout = unspecified.
    constexpr Layout() noexcept = default;

    //! Construct from an axis letter string (case-insensitive).
    //! Invalid input (empty / too long / unknown letter / duplicate letter)
    //! throws std::runtime_error.
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

    //! Throwing-free variant: on failure it returns false and leaves out unchanged.
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

    //! The conventional default layout for a rank: 3 -> CHW, 4 -> NCHW,
    //! 5 -> NCDHW; anything else returns empty (the caller must declare one).
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

    //! Position index of an axis letter; -1 when absent.
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

    //! The normalised axis letter string (for logs / exception messages).
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

    // ---- Common layout constants ----
    // Declaration and definition are separated: inside the class the type is
    // still incomplete, so in-place initialization is impossible. The definitions
    // live in src/core/layout.cpp (constant-initialized at compile time, so there
    // is no static initialization order problem).
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

    std::array<char, kMaxRank> m_axes{};   // upper-cased; the first m_rank entries are valid
    std::int8_t                m_rank = 0;
};

}  // namespace trt_alpha::core
