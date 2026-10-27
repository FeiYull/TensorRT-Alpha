# =============================================================================
#  build_windows_tensorrt-opencv.ps1
# -----------------------------------------------------------------------------
#  Windows build :  CUDA + TensorRT + OpenCV          (no FFmpeg, no OpenGL)
#  Build dir     :  build
#  Output        :  build\bin\<Config>\trt_alpha.exe
#
#  Usage:
#      .\scripts\build_windows_tensorrt-opencv.ps1
#      .\scripts\build_windows_tensorrt-opencv.ps1 -Config Debug
#      .\scripts\build_windows_tensorrt-opencv.ps1 -WithTests -NoSamples
# =============================================================================
[CmdletBinding()]
param(
    [ValidateSet('Release', 'Debug', 'RelWithDebInfo', 'MinSizeRel')]
    [string] $Config = 'Release',

    # Third-party roots -- override if yours live elsewhere.
    [string] $TensorRTRoot = 'D:/ThirdParty/TensorRT-10.16.1.11',
    [string] $OpenCVDir    = 'D:/ThirdParty/opencv/build/x64/vc16/lib',

    # Empty = <repo>\build
    [string] $BuildDir = '',
    # 0 = number of logical cores
    [int]    $Jobs = 0,

    [switch] $WithTests,   # TRT_ALPHA_BUILD_TESTS   (default OFF)
    [switch] $NoSamples    # TRT_ALPHA_BUILD_SAMPLES (default ON)
)

# cmake writes warnings to stderr; with 'Stop' PowerShell turns the first such
# line into an exception and aborts a build that is actually fine. Fatal cases
# are caught by the $LASTEXITCODE checks below.
$ErrorActionPreference = 'Continue'

$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
if (-not $BuildDir) { $BuildDir = Join-Path $root 'build' }
if ($Jobs -le 0)    { $Jobs = [Environment]::ProcessorCount }

# -----------------------------------------------------------------------------
# MSB6001 guard: "CL.exe: invalid command line switch".
# ProcessStartInfo.EnvironmentVariables is a case-sensitive Hashtable, so having
# both HTTP_PROXY and http_proxy makes MSBuild throw "duplicate key" and wrap it
# as MSB6001. Keep the uppercase spelling, drop the lowercase twin.
# -----------------------------------------------------------------------------
foreach ($n in 'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'NO_PROXY') {
    if ([Environment]::GetEnvironmentVariable($n) -and
        [Environment]::GetEnvironmentVariable($n.ToLowerInvariant())) {
        [Environment]::SetEnvironmentVariable($n.ToLowerInvariant(), $null)
        Write-Host "[env] removed duplicate $($n.ToLowerInvariant()) (MSB6001 guard)" -ForegroundColor DarkYellow
    }
}

$tests   = if ($WithTests) { 'ON' } else { 'OFF' }
$samples = if ($NoSamples) { 'OFF' } else { 'ON' }

Write-Host "[build] CUDA + TensorRT + OpenCV  ->  $BuildDir" -ForegroundColor Cyan

# Note the quoting: an unquoted -DNAME=$var token is passed to cmake verbatim
# (PowerShell does not expand it), which makes cmake look for a path literally
# named "$OpenCVDir". Always quote every -D argument.
cmake -S $root -B $BuildDir `
      -G "Visual Studio 17 2022" -A x64 `
      "-DTensorRT_ROOT=$TensorRTRoot" `
      "-DOpenCV_DIR=$OpenCVDir" `
      "-DTRT_ALPHA_WITH_OPENCV=ON" `
      "-DTRT_ALPHA_WITH_FFMPEG=OFF" `
      "-DTRT_ALPHA_WITH_GL=OFF" `
      "-DTRT_ALPHA_BUILD_TESTS=$tests" `
      "-DTRT_ALPHA_BUILD_SAMPLES=$samples"
if ($LASTEXITCODE -ne 0) { throw "cmake configure failed ($LASTEXITCODE)" }

cmake --build $BuildDir --config $Config -j $Jobs
if ($LASTEXITCODE -ne 0) { throw "cmake build failed ($LASTEXITCODE)" }

$exe = Join-Path $BuildDir "bin\$Config\trt_alpha.exe"
$lines = 0
try { $lines = (& $exe list 2>$null | Measure-Object -Line).Lines } catch { }
Write-Host "[ok] $exe   ($lines lines from 'list')" -ForegroundColor Green
