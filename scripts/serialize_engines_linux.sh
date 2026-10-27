#!/usr/bin/env bash
# =============================================================================
#  scripts/linux/serialize_engines_linux.sh
# -----------------------------------------------------------------------------
#  Serialize every ONNX under  onnx/  into  engines/  with TensorRT 10 on Linux.
#
#  Rules encoded here (every one traced back to a source -- nothing guessed):
#
#   * yolov26*  ->  `--noTF32`.  These ONNX carry a *static* batch, so no
#                  optimization profile may be supplied at all.
#                  (per the project owner's instruction)
#                  Precision variants:  yolov26n_fp16.onnx  ->  --fp16
#                                      yolov26n_int8.onnx  ->  --int8
#                  (`--fp16` is NOT optional -- without it TensorRT computes
#                   the fp16 weights in FP32 and the speed-up is lost.
#                   `--noTF32` is for FP32 engines only -- never add it here.)
#
#   * everything else  ->  the flags documented in the `main` branch README of
#                  FeiYull/TensorRT-Alpha, with two deliberate changes:
#                     - `--buildOnly` dropped   (removed in TensorRT 10)
#                     - max batch clipped 8 -> 4
#                  Dynamic-batch models therefore get
#                     --minShapes/--optShapes/--maxShapes = 1 / 2 / 4
#                  (where a doc asked for some other range it is clipped too)
#
#   * static-batch models that the docs cover with a bare
#     `--onnx --saveEngine` get no shape flags at all.
#
#  Engine name = ONNX basename with every '-' replaced by '_'.  The project
#  owner chose this over the `yolov26n_*.trt` spelling that configs/yolov26*.ini
#  refers to, so those configs need their `engine =` field adjusted separately.
#
#  Input tensor names are NOT assumed: they were read straight out of each
#  ONNX with a protobuf walk and are reproduced below verbatim
#  (images / input / x / Input3 -- four different spellings across 31 files).
#
#  ON LD_LIBRARY_PATH -- READ THIS:
#     trtexec ships without an RPATH, so the loader must be told where
#     libnvinfer*.so / libnvonnxparser*.so live.  This script passes it
#     PER CHILD PROCESS ONLY (see run_trtexec below).  Nothing is exported,
#     nothing is written to the user's shell profile or to any system config.
#
#  Usage:
#     bash scripts/linux/serialize_engines_linux.sh                  # build all
#     bash scripts/linux/serialize_engines_linux.sh --only u2net     # subset
#     TRT_ROOT=/opt/TensorRT bash scripts/linux/serialize_engines_linux.sh
#     REPO=/path/to/TensorRT-Alpha bash scripts/linux/serialize_engines_linux.sh
#
#  Logs: build/engine_logs/<engine>.log  (one per model, trtexec stdout+stderr)
# =============================================================================
set -uo pipefail

# --- locate the repository ---------------------------------------------------
# This file normally lives in <repo>/scripts/linux/, i.e. two levels below the
# repository root, but the depth is not assumed: walk upwards from the script
# directory (then from $PWD, then from the conventional home checkout) until a
# directory that owns both onnx/ and CMakeLists.txt is found.  REPO=... wins.
find_repo() {
    local d="$1" last=""
    while [[ -n "$d" && "$d" != "$last" ]]; do
        if [[ -d "$d/onnx" && -f "$d/CMakeLists.txt" ]]; then printf '%s\n' "$d"; return 0; fi
        last="$d"; d="$(dirname "$d")"
    done
    return 1
}

if [[ -z "${REPO:-}" ]]; then
    for cand in "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" "$PWD" "$HOME/workspace/TensorRT-Alpha"; do
        if REPO="$(find_repo "$cand")"; then break; fi
        REPO=""
    done
fi

TRT_ROOT="${TRT_ROOT:-/home/feiyull/thirdparty/TensorRT-10.16.1.11}"
TRTEXEC="$TRT_ROOT/bin/trtexec"
ONNX_DIR="$REPO/onnx"
ENGINE_DIR="$REPO/engines"
LOG_DIR="$REPO/build/engine_logs"

ONLY=""
if [[ "${1:-}" == "--only" ]]; then ONLY="${2:-}"; fi

if [[ ! -d "$ONNX_DIR" ]]; then
    echo "ERROR: no onnx/ directory under REPO=${REPO:-<unresolved>}" >&2
    echo "       (pass REPO=/path/to/TensorRT-Alpha)" >&2
    exit 2
fi
if [[ ! -x "$TRTEXEC" ]]; then
    echo "ERROR: trtexec not found at $TRTEXEC  (override with TRT_ROOT=...)" >&2
    exit 2
fi

mkdir -p "$ENGINE_DIR" "$LOG_DIR"

ok=0; fail=0; skip=0
FAILED=()

# Per-invocation LD_LIBRARY_PATH: scoped to the child process, never exported.
run_trtexec() {
    LD_LIBRARY_PATH="$TRT_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" "$TRTEXEC" "$@"
}

serialize() {
    local onnx="$1" engine="$2"
    shift 2

    if [[ -n "$ONLY" && "$onnx" != *"$ONLY"* ]]; then return; fi
    if [[ ! -f "$ONNX_DIR/$onnx" ]]; then
        printf 'SKIP  %-52s (onnx not present)\n' "$onnx"
        skip=$((skip + 1)); return
    fi

    printf 'BUILD %-52s -> %s\n' "$onnx" "$engine"
    local log="$LOG_DIR/${engine%.trt}.log" t0 t1
    t0=$(date +%s)
    if run_trtexec --onnx="$ONNX_DIR/$onnx" --saveEngine="$ENGINE_DIR/$engine" "$@" >"$log" 2>&1; then
        t1=$(date +%s)
        printf '  OK    %-6s  %ss\n' "$(du -h "$ENGINE_DIR/$engine" | cut -f1)" "$((t1 - t0))"
        ok=$((ok + 1))
    else
        t1=$(date +%s)
        printf '  FAIL  %ss  -- see %s\n' "$((t1 - t0))" "$log"
        fail=$((fail + 1)); FAILED+=("$onnx")
    fi
}

echo "repo    : $REPO"
echo "trtexec : $TRTEXEC"
echo "onnx    : $ONNX_DIR"
echo "engines : $ENGINE_DIR"
echo "logs    : $LOG_DIR"
echo

# -----------------------------------------------------------------------------
# 1) yolov26 family -- static batch (2), no shapes, --noTF32
#    input name: images        (yolov26n 640 / -depth 768 / -obb -sem 1024)
# -----------------------------------------------------------------------------
serialize yolov26n.onnx          yolov26n.trt          --noTF32
serialize yolov26n-cls.onnx      yolov26n_cls.trt      --noTF32
serialize yolov26n-depth.onnx    yolov26n_depth.trt    --noTF32
serialize yolov26n-obb.onnx      yolov26n_obb.trt      --noTF32
serialize yolov26n-pose.onnx     yolov26n_pose.trt     --noTF32
serialize yolov26n-seg.onnx      yolov26n_seg.trt      --noTF32
serialize yolov26n-sem.onnx      yolov26n_sem.trt      --noTF32
serialize yolov26n_fp16.onnx     yolov26n_fp16.trt     --fp16
serialize yolov26n_fp32.onnx     yolov26n_fp32.trt     --noTF32
serialize yolov26n_int8.onnx     yolov26n_int8.trt     --int8

# -----------------------------------------------------------------------------
# 2) other static-batch models -- bare build, no profile possible
#    yolox_s / yolox_tiny : documented as a bare `--onnx --saveEngine`
#    yolov8n_static (1x3x640x640), human_seg...oct (x 1x3x192x192),
#    mnist (Input3 1x1x28x28) : ONNX batch is already fixed
# -----------------------------------------------------------------------------
serialize yolox_s.onnx                                yolox_s.trt
serialize yolox_tiny.onnx                             yolox_tiny.trt
serialize yolov8n_static.onnx                         yolov8n_static.trt
serialize human_segmentation_pphumanseg_2021oct.onnx  human_segmentation_pphumanseg_2021oct.trt
serialize mnist.onnx                                  mnist.trt

# -----------------------------------------------------------------------------
# 3) dynamic-batch models -- min/opt/max = 1/2/4
# -----------------------------------------------------------------------------
# alpha_yolov3-spp: images[batch,3,height,width]
serialize alpha_yolov3-spp.onnx  alpha_yolov3_spp.trt \
    --minShapes=images:1x3x640x640 --optShapes=images:2x3x640x640 --maxShapes=images:4x3x640x640

# alpha_yolov4: input[batch_size,3,608,608]  (H/W fixed at 608)
serialize alpha_yolov4_-1_3_608_608_dynamic.onnx  alpha_yolov4_1_3_608_608_dynamic.trt \
    --minShapes=input:1x3x608x608 --optShapes=input:2x3x608x608 --maxShapes=input:4x3x608x608

# YuNet: input[batch_size,3,height,width].  H/W deliberately stay dynamic
# (the face size comes from the source image), hence the wide 120..2000 range
# taken from the libfacedetection doc, with max batch clipped 8 -> 4.
serialize alpha_yunet_yunet_final_dynamic_simplify.onnx  alpha_yunet_yunet_final_dynamic_simplify.trt \
    --minShapes=input:1x3x120x120 --optShapes=input:2x3x320x320 --maxShapes=input:4x3x2000x2000

# EfficientDet: input[N,512,512,3] -- NHWC, batch first, no `batch_size` alias
serialize efficientdet0.onnx  efficientdet0.trt \
    --minShapes=input:1x512x512x3 --optShapes=input:2x512x512x3 --maxShapes=input:4x512x512x3

# PP-HumanSeg dynamic: x[batch_size,3,192,192]  (H/W fixed at 192)
serialize human_segmentation_pphumanseg_2021oct_dynamic.onnx  human_segmentation_pphumanseg_2021oct_dynamic.trt \
    --minShapes=x:1x3x192x192 --optShapes=x:2x3x192x192 --maxShapes=x:4x3x192x192

# u2net / u2netp: images[batch_size,3,320,320]  (H/W fixed; doc says 1/4/8 -> 1/2/4)
serialize u2net.onnx   u2net.trt \
    --minShapes=images:1x3x320x320 --optShapes=images:2x3x320x320 --maxShapes=images:4x3x320x320

serialize u2netp.onnx  u2netp.trt \
    --minShapes=images:1x3x320x320 --optShapes=images:2x3x320x320 --maxShapes=images:4x3x320x320

# yolo-nas: images[batch_size,3,640,640]
serialize yolonas_s.onnx  yolonas_s.trt \
    --minShapes=images:1x3x640x640 --optShapes=images:2x3x640x640 --maxShapes=images:4x3x640x640

# yolor: images[batch,3,height,width]
serialize yolor_csp.onnx  yolor_csp.trt \
    --minShapes=images:1x3x640x640 --optShapes=images:2x3x640x640 --maxShapes=images:4x3x640x640

# yolov5 / yolov7 / yolov8: images[batch,3,height,width]
serialize yolov5n.onnx  yolov5n.trt \
    --minShapes=images:1x3x640x640 --optShapes=images:2x3x640x640 --maxShapes=images:4x3x640x640

serialize yolov5s.onnx  yolov5s.trt \
    --minShapes=images:1x3x640x640 --optShapes=images:2x3x640x640 --maxShapes=images:4x3x640x640

serialize yolov7-tiny.onnx  yolov7_tiny.trt \
    --minShapes=images:1x3x640x640 --optShapes=images:2x3x640x640 --maxShapes=images:4x3x640x640

# yolov8 (new deployment): STATIC batch = 2, no dynamic shapes.
# The legacy dynamic engines are kept as *_old.trt for the yolov8_old registration.
# --noTF32 keeps the engine on true fp32: with TF32 on (the Ampere+ default the
# reference numbers were taken with) the boxes drift enough to reshuffle NMS.
serialize yolov8n.onnx  yolov8n.trt  --noTF32

serialize yolov8n-pose.onnx  yolov8n_pose.trt  --noTF32

serialize yolov8n-seg.onnx  yolov8n_seg.trt  --noTF32

serialize yolov8n-cls.onnx  yolov8n_cls.trt  --noTF32

serialize yolov8n-obb.onnx  yolov8n_obb.trt  --noTF32

# yolov9: same static batch = 2.  Only yolov9c-seg / yolov9e-seg exist for seg,
# so the seg engine is named after the `c` checkpoint.
serialize yolov9t.onnx      yolov9t.trt      --noTF32

serialize yolov9c-seg.onnx  yolov9c_seg.trt  --noTF32

# yolov10: one2many only ([B, 84, 8400]).  The official release ONNX is end2end
# ([B, 300, 6]) and is rejected at startup, so export with `yolo export`.
serialize yolov10n.onnx     yolov10n.trt     --noTF32

# yolov6n: images[batch,3,640,640]  (H/W fixed at 640)
serialize yolov6n.onnx  yolov6n.trt \
    --minShapes=images:1x3x640x640 --optShapes=images:2x3x640x640 --maxShapes=images:4x3x640x640

# -----------------------------------------------------------------------------
# summary
# -----------------------------------------------------------------------------
echo
echo "============================================================"
printf '  ok=%d  fail=%d  skip=%d\n' "$ok" "$fail" "$skip"
if (( fail > 0 )); then
    echo "  failed ONNX:"
    printf '    %s\n' "${FAILED[@]}"
fi
echo "============================================================"
(( fail == 0 ))
