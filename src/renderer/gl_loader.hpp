// =============================================================================
//  gl_loader -- minimal OpenGL 3.3 core loader (no GLAD / no GLEW), header-only
// -----------------------------------------------------------------------------
//  Why hand-written:
//    * Windows' <GL/gl.h> only declares OpenGL 1.1. Everything from 1.2 up
//      (VBO/VAO/shaders/glActiveTexture) is a function pointer obtained via
//      glfwGetProcAddress. GLAD would do this, but it needs a Khronos XML
//      download at generation time; a small hand-written set keeps the build
//      fully offline.
//
//  Header-only (inline variables + inline load()): both gl_context.cpp and
//  gl_renderer.cpp share the same pointers through the one definition per
//  translation unit that C++17 inline variables guarantee. There is no .cpp to
//  compile, so the GL=OFF build (which excludes gl_*.cpp) never pulls in GLFW.
//
//  Usage:
//      #define GLFW_INCLUDE_NONE
//      #include "gl_loader.hpp"
//      #include <GLFW/glfw3.h>
//
//      glfwMakeContextCurrent(window);
//      if (!trt_gl::load()) { ... }
//
//  After load() succeeds the macros at the bottom let ordinary GL spelling
//  (glGenBuffers(...) etc.) resolve to the loaded pointers. The GL 1.1 entry
//  points (glClear / glTexImage2D / glDrawArrays / ...) are NOT touched here:
//  they come straight from opengl32 / the system libGL.
// =============================================================================
#pragma once

#ifdef _WIN32
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#endif

#include <GL/gl.h>

//! The inline load() below calls glfwGetProcAddress, so this header pulls in
//! GLFW itself. GLFW_INCLUDE_NONE stops glfw3.h from including its own GL / GLU
//! headers (we already included <GL/gl.h> above and want only the 1.1 surface).
#ifndef GLFW_INCLUDE_NONE
#  define GLFW_INCLUDE_NONE
#endif
#include <GLFW/glfw3.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace trt_gl {

#ifdef _WIN32
#  define TRT_GL_CALL __stdcall
#else
#  define TRT_GL_CALL
#endif

using GLsizeiptrT  = std::ptrdiff_t;
using GLintptrT    = std::ptrdiff_t;
using GLcharT      = char;
using GLbooleanT   = unsigned char;

// ---- constants that <GL/gl.h> (1.1) does not provide --------------------
#ifndef GL_ARRAY_BUFFER
#  define GL_ARRAY_BUFFER                 0x8892
#endif
#ifndef GL_ELEMENT_ARRAY_BUFFER
#  define GL_ELEMENT_ARRAY_BUFFER         0x8893
#endif
#ifndef GL_PIXEL_UNPACK_BUFFER
#  define GL_PIXEL_UNPACK_BUFFER          0x88EC
#endif
#ifndef GL_STATIC_DRAW
#  define GL_STATIC_DRAW                  0x88E4
#endif
#ifndef GL_DYNAMIC_DRAW
#  define GL_DYNAMIC_DRAW                 0x88E8
#endif
#ifndef GL_STREAM_DRAW
#  define GL_STREAM_DRAW                  0x88E0
#endif
#ifndef GL_VERTEX_SHADER
#  define GL_VERTEX_SHADER                0x8B31
#endif
#ifndef GL_FRAGMENT_SHADER
#  define GL_FRAGMENT_SHADER              0x8B30
#endif
#ifndef GL_COMPILE_STATUS
#  define GL_COMPILE_STATUS               0x8B81
#endif
#ifndef GL_LINK_STATUS
#  define GL_LINK_STATUS                  0x8B82
#endif
#ifndef GL_INFO_LOG_LENGTH
#  define GL_INFO_LOG_LENGTH              0x8B84
#endif
#ifndef GL_TEXTURE0
#  define GL_TEXTURE0                     0x84C0
#endif
#ifndef GL_CLAMP_TO_EDGE
#  define GL_CLAMP_TO_EDGE                0x812F
#endif
#ifndef GL_MULTISAMPLE
#  define GL_MULTISAMPLE                  0x809D
#endif
#ifndef GL_FRAMEBUFFER_SRGB
#  define GL_FRAMEBUFFER_SRGB             0x8DB9
#endif
#ifndef GL_SHADING_LANGUAGE_VERSION
#  define GL_SHADING_LANGUAGE_VERSION     0x8B8C
#endif
#ifndef GL_NUM_EXTENSIONS
#  define GL_NUM_EXTENSIONS               0x821D
#endif
#ifndef GL_BLEND
#  define GL_BLEND                        0x0BE2
#endif
#ifndef GL_SRC_ALPHA
#  define GL_SRC_ALPHA                    0x0302
#endif
#ifndef GL_ONE_MINUS_SRC_ALPHA
#  define GL_ONE_MINUS_SRC_ALPHA          0x0303
#endif
#ifndef GL_FUNC_ADD
#  define GL_FUNC_ADD                     0x8006
#endif
#ifndef GL_NEAREST
#  define GL_NEAREST                      0x2600
#endif
#ifndef GL_LINEAR
#  define GL_LINEAR                       0x2601
#endif
#ifndef GL_RED
#  define GL_RED                          0x1903
#endif
#ifndef GL_R8
#  define GL_R8                           0x8229
#endif
#ifndef GL_RGB
#  define GL_RGB                          0x1907
#endif
#ifndef GL_RGB8
#  define GL_RGB8                         0x8051
#endif
#ifndef GL_RGBA
#  define GL_RGBA                         0x1908
#endif
#ifndef GL_RGBA8
#  define GL_RGBA8                        0x8058
#endif
#ifndef GL_UNSIGNED_BYTE
#  define GL_UNSIGNED_BYTE                0x1401
#endif
#ifndef GL_FLOAT
#  define GL_FLOAT                        0x1406
#endif
#ifndef GL_TRIANGLES
#  define GL_TRIANGLES                    0x0004
#endif
#ifndef GL_LINES
#  define GL_LINES                        0x0001
#endif
#ifndef GL_FALSE
#  define GL_FALSE                        0
#endif
#ifndef GL_TRUE
#  define GL_TRUE                         1
#endif

// ---- function pointer types (only those >= GL 1.2) -----------------------
using PFN_glGenBuffers             = void (TRT_GL_CALL*)(GLsizei, GLuint*);
using PFN_glDeleteBuffers          = void (TRT_GL_CALL*)(GLsizei, const GLuint*);
using PFN_glBindBuffer             = void (TRT_GL_CALL*)(GLenum, GLuint);
using PFN_glBufferData             = void (TRT_GL_CALL*)(GLenum, GLsizeiptrT, const void*, GLenum);
using PFN_glBufferSubData          = void (TRT_GL_CALL*)(GLenum, GLintptrT, GLsizeiptrT, const void*);
using PFN_glGenVertexArrays        = void (TRT_GL_CALL*)(GLsizei, GLuint*);
using PFN_glDeleteVertexArrays     = void (TRT_GL_CALL*)(GLsizei, const GLuint*);
using PFN_glBindVertexArray        = void (TRT_GL_CALL*)(GLuint);
using PFN_glVertexAttribPointer    = void (TRT_GL_CALL*)(GLuint, GLint, GLenum, GLbooleanT, GLsizei, const void*);
using PFN_glEnableVertexAttribArray  = void (TRT_GL_CALL*)(GLuint);
using PFN_glDisableVertexAttribArray = void (TRT_GL_CALL*)(GLuint);
using PFN_glCreateShader           = GLuint (TRT_GL_CALL*)(GLenum);
using PFN_glDeleteShader           = void (TRT_GL_CALL*)(GLuint);
using PFN_glShaderSource           = void (TRT_GL_CALL*)(GLuint, GLsizei, const GLcharT* const*, const GLint*);
using PFN_glCompileShader          = void (TRT_GL_CALL*)(GLuint);
using PFN_glGetShaderiv            = void (TRT_GL_CALL*)(GLuint, GLenum, GLint*);
using PFN_glGetShaderInfoLog       = void (TRT_GL_CALL*)(GLuint, GLsizei, GLsizei*, GLcharT*);
using PFN_glCreateProgram          = GLuint (TRT_GL_CALL*)(void);
using PFN_glDeleteProgram          = void (TRT_GL_CALL*)(GLuint);
using PFN_glAttachShader           = void (TRT_GL_CALL*)(GLuint, GLuint);
using PFN_glLinkProgram            = void (TRT_GL_CALL*)(GLuint);
using PFN_glGetProgramiv           = void (TRT_GL_CALL*)(GLuint, GLenum, GLint*);
using PFN_glGetProgramInfoLog      = void (TRT_GL_CALL*)(GLuint, GLsizei, GLsizei*, GLcharT*);
using PFN_glUseProgram             = void (TRT_GL_CALL*)(GLuint);
using PFN_glGetUniformLocation     = GLint (TRT_GL_CALL*)(GLuint, const GLcharT*);
using PFN_glUniform1i              = void (TRT_GL_CALL*)(GLint, GLint);
using PFN_glUniform2f              = void (TRT_GL_CALL*)(GLint, GLfloat, GLfloat);
using PFN_glUniform4f              = void (TRT_GL_CALL*)(GLint, GLfloat, GLfloat, GLfloat, GLfloat);
using PFN_glUniformMatrix4fv       = void (TRT_GL_CALL*)(GLint, GLsizei, GLbooleanT, const GLfloat*);
using PFN_glActiveTexture          = void (TRT_GL_CALL*)(GLenum);
using PFN_glBlendFuncSeparate      = void (TRT_GL_CALL*)(GLenum, GLenum, GLenum, GLenum);
using PFN_glGenerateMipmap         = void (TRT_GL_CALL*)(GLenum);

// ---- the pointers (C++17 inline: one object across all TUs) ---------------
inline PFN_glGenBuffers               GenBuffers               = nullptr;
inline PFN_glDeleteBuffers            DeleteBuffers            = nullptr;
inline PFN_glBindBuffer               BindBuffer               = nullptr;
inline PFN_glBufferData               BufferData               = nullptr;
inline PFN_glBufferSubData            BufferSubData            = nullptr;
inline PFN_glGenVertexArrays          GenVertexArrays          = nullptr;
inline PFN_glDeleteVertexArrays       DeleteVertexArrays       = nullptr;
inline PFN_glBindVertexArray          BindVertexArray          = nullptr;
inline PFN_glVertexAttribPointer      VertexAttribPointer      = nullptr;
inline PFN_glEnableVertexAttribArray  EnableVertexAttribArray  = nullptr;
inline PFN_glDisableVertexAttribArray DisableVertexAttribArray = nullptr;
inline PFN_glCreateShader             CreateShader             = nullptr;
inline PFN_glDeleteShader             DeleteShader             = nullptr;
inline PFN_glShaderSource             ShaderSource             = nullptr;
inline PFN_glCompileShader            CompileShader            = nullptr;
inline PFN_glGetShaderiv              GetShaderiv              = nullptr;
inline PFN_glGetShaderInfoLog         GetShaderInfoLog         = nullptr;
inline PFN_glCreateProgram            CreateProgram            = nullptr;
inline PFN_glDeleteProgram            DeleteProgram            = nullptr;
inline PFN_glAttachShader             AttachShader             = nullptr;
inline PFN_glLinkProgram              LinkProgram              = nullptr;
inline PFN_glGetProgramiv             GetProgramiv             = nullptr;
inline PFN_glGetProgramInfoLog        GetProgramInfoLog        = nullptr;
inline PFN_glUseProgram               UseProgram               = nullptr;
inline PFN_glGetUniformLocation       GetUniformLocation       = nullptr;
inline PFN_glUniform1i                Uniform1i                = nullptr;
inline PFN_glUniform2f                Uniform2f                = nullptr;
inline PFN_glUniform4f                Uniform4f                = nullptr;
inline PFN_glUniformMatrix4fv         UniformMatrix4fv         = nullptr;
inline PFN_glActiveTexture            ActiveTexture            = nullptr;
inline PFN_glBlendFuncSeparate        BlendFuncSeparate        = nullptr;
inline PFN_glGenerateMipmap           GenerateMipmap           = nullptr;

//! Resolve every pointer. Requires a current GL context.
//! Returns false and leaves `missing` set to the first unresolved name.
inline bool load(const char** missing = nullptr)
{
    const char* firstBad = nullptr;
    bool ok = true;

    auto R = [&firstBad, &ok](auto& slot, const char* name) {
        using T = std::decay_t<decltype(slot)>;
        slot = reinterpret_cast<T>(glfwGetProcAddress(name));
        if (slot == nullptr)
        {
            if (firstBad == nullptr) { firstBad = name; }
            ok = false;
        }
    };

    R(GenBuffers,               "glGenBuffers");
    R(DeleteBuffers,            "glDeleteBuffers");
    R(BindBuffer,               "glBindBuffer");
    R(BufferData,               "glBufferData");
    R(BufferSubData,            "glBufferSubData");
    R(GenVertexArrays,          "glGenVertexArrays");
    R(DeleteVertexArrays,       "glDeleteVertexArrays");
    R(BindVertexArray,          "glBindVertexArray");
    R(VertexAttribPointer,      "glVertexAttribPointer");
    R(EnableVertexAttribArray,  "glEnableVertexAttribArray");
    R(DisableVertexAttribArray, "glDisableVertexAttribArray");
    R(CreateShader,             "glCreateShader");
    R(DeleteShader,             "glDeleteShader");
    R(ShaderSource,             "glShaderSource");
    R(CompileShader,            "glCompileShader");
    R(GetShaderiv,              "glGetShaderiv");
    R(GetShaderInfoLog,         "glGetShaderInfoLog");
    R(CreateProgram,            "glCreateProgram");
    R(DeleteProgram,            "glDeleteProgram");
    R(AttachShader,             "glAttachShader");
    R(LinkProgram,              "glLinkProgram");
    R(GetProgramiv,             "glGetProgramiv");
    R(GetProgramInfoLog,        "glGetProgramInfoLog");
    R(UseProgram,               "glUseProgram");
    R(GetUniformLocation,       "glGetUniformLocation");
    R(Uniform1i,                "glUniform1i");
    R(Uniform2f,                "glUniform2f");
    R(Uniform4f,                "glUniform4f");
    R(UniformMatrix4fv,         "glUniformMatrix4fv");
    R(ActiveTexture,            "glActiveTexture");
    R(BlendFuncSeparate,        "glBlendFuncSeparate");
    R(GenerateMipmap,           "glGenerateMipmap");

    if (missing != nullptr) { *missing = firstBad; }
    return ok;
}

}  // namespace trt_gl

// ---- macros so ordinary GL spelling works ----------------------------------
#define glGenBuffers               ::trt_gl::GenBuffers
#define glDeleteBuffers            ::trt_gl::DeleteBuffers
#define glBindBuffer               ::trt_gl::BindBuffer
#define glBufferData               ::trt_gl::BufferData
#define glBufferSubData            ::trt_gl::BufferSubData
#define glGenVertexArrays          ::trt_gl::GenVertexArrays
#define glDeleteVertexArrays       ::trt_gl::DeleteVertexArrays
#define glBindVertexArray          ::trt_gl::BindVertexArray
#define glVertexAttribPointer      ::trt_gl::VertexAttribPointer
#define glEnableVertexAttribArray  ::trt_gl::EnableVertexAttribArray
#define glDisableVertexAttribArray ::trt_gl::DisableVertexAttribArray
#define glCreateShader             ::trt_gl::CreateShader
#define glDeleteShader             ::trt_gl::DeleteShader
#define glShaderSource             ::trt_gl::ShaderSource
#define glCompileShader            ::trt_gl::CompileShader
#define glGetShaderiv              ::trt_gl::GetShaderiv
#define glGetShaderInfoLog         ::trt_gl::GetShaderInfoLog
#define glCreateProgram            ::trt_gl::CreateProgram
#define glDeleteProgram            ::trt_gl::DeleteProgram
#define glAttachShader             ::trt_gl::AttachShader
#define glLinkProgram              ::trt_gl::LinkProgram
#define glGetProgramiv             ::trt_gl::GetProgramiv
#define glGetProgramInfoLog        ::trt_gl::GetProgramInfoLog
#define glUseProgram               ::trt_gl::UseProgram
#define glGetUniformLocation       ::trt_gl::GetUniformLocation
#define glUniform1i                ::trt_gl::Uniform1i
#define glUniform2f                ::trt_gl::Uniform2f
#define glUniform4f                ::trt_gl::Uniform4f
#define glUniformMatrix4fv         ::trt_gl::UniformMatrix4fv
#define glActiveTexture            ::trt_gl::ActiveTexture
#define glBlendFuncSeparate        ::trt_gl::BlendFuncSeparate
#define glGenerateMipmap           ::trt_gl::GenerateMipmap
