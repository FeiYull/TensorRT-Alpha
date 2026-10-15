# =============================================================================
#  Release 全测试 —— 一次性跑完，输出到 release_test_log.txt
# =============================================================================
$ErrorActionPreference = "Continue"

$root   = "D:\VS2022_Project\TensorRT-Alpha"
$bin    = "$root\build\bin\Release"
$engine = "D:\ThirdParty\TensorRT-10.16.1.11\bin\yolov8n.trt"
$image  = "$root\data\bus.jpg"
$video  = "$root\data\people.mp4"
$log    = "$root\release_test_log.txt"

Set-Location $root
"" | Out-File -FilePath $log -Encoding utf8

function Run-One {
    param(
        [string]   $Title,
        [string]   $Exe,
        [string[]] $ExtraArgs = @()
    )
    $sep = "=" * 70
    $sep | Out-File -FilePath $log -Append -Encoding utf8
    "=== $Title ===" | Out-File -FilePath $log -Append -Encoding utf8
    "exe : $Exe"     | Out-File -FilePath $log -Append -Encoding utf8
    "args: $($ExtraArgs -join ' ')" | Out-File -FilePath $log -Append -Encoding utf8
    $sep | Out-File -FilePath $log -Append -Encoding utf8

    if (-not (Test-Path $Exe)) {
        "[SKIP] not found: $Exe" | Out-File -FilePath $log -Append -Encoding utf8
        return
    }

    $output = & $Exe @ExtraArgs 2>&1
    $code = $LASTEXITCODE
    $output | Out-File -FilePath $log -Append -Encoding utf8
    "" | Out-File -FilePath $log -Append -Encoding utf8
    "[exit code] $code" | Out-File -FilePath $log -Append -Encoding utf8
}

# ---- 1. 单元测试（无参数）----
$unitTests = @(
    "test_logger",
    "test_bounded_queue",
    "test_buffer",
    "test_buffer_view",
    "test_batch",
    "test_batch_result",
    "test_memory_pool",
    "test_datasource",
    "test_renderer",
    "test_kernels",
    "test_config",
    "test_paths",
    "test_ini_parser",
    "test_engine",
    "test_task_types",
    "test_model_registry"
)

foreach ($t in $unitTests) {
    Run-One -Title "unit: $t" -Exe "$bin\$t.exe"
}

# ---- 2. 真推理 ----
Run-One -Title "infer: test_yolov8"   -Exe "$bin\test_yolov8.exe"   -ExtraArgs @($engine, $image)
Run-One -Title "infer: test_pipeline" -Exe "$bin\test_pipeline.exe" -ExtraArgs @($engine, $image)

# ---- 3. sample_yolov8 ----
Run-One -Title "sample: image" -Exe "$bin\sample_yolov8.exe" -ExtraArgs @($engine, $image, "--save")
Run-One -Title "sample: video" -Exe "$bin\sample_yolov8.exe" -ExtraArgs @($engine, $video)

# ---- 4. CLI ----
Run-One -Title "cli: list"      -Exe "$bin\trt_alpha.exe" -ExtraArgs @("list")
Run-One -Title "cli: run image" -Exe "$bin\trt_alpha.exe" -ExtraArgs @("run", "--image", $image, "--config", "configs/yolov8.ini", "--save")
Run-One -Title "cli: run video" -Exe "$bin\trt_alpha.exe" -ExtraArgs @("run", "--video", $video, "--config", "configs/yolov8.ini")
Run-One -Title "cli: bench"     -Exe "$bin\trt_alpha.exe" -ExtraArgs @("bench", "--engine", $engine, "--config", "configs/yolov8.ini", "--iters", "50", "--warmup", "5")

"`nALL DONE." | Out-File -FilePath $log -Append -Encoding utf8
Write-Host "log written to: $log" -ForegroundColor Green