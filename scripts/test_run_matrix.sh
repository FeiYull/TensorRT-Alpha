#!/usr/bin/env bash
# =============================================================================
#  test_run_matrix.sh
#  ---------------------------------------------------------------------------
#  Exhaustively exercise `trt_alpha run` by combining every registered model
#  with every kind of input source and every CLI option, then report exactly
#  which combinations fail and why.
#
#  It never modifies the repository sources: everything it writes goes under
#  save/_run_matrix/ (already covered by .gitignore).
#
#  Usage:
#      scripts/linux/test_run_matrix.sh              # full matrix
#      scripts/linux/test_run_matrix.sh --quick      # one image per model only
#      scripts/linux/test_run_matrix.sh --only yolov26
#      scripts/linux/test_run_matrix.sh --list       # print the plan and exit
#
#  Result files (under save/_run_matrix/<timestamp>/):
#      results.tsv     one row per command
#      failures.txt    only the failing commands
#      summary.txt     per-model roll-up
#      logs/*.log      full stdout + stderr per command
#
#  Exit status: 0 when every case passed, 1 when at least one case failed.
# =============================================================================
set -u

# --------------------------------------------------------------------------
# locate the repository root (this script lives two levels down)
# --------------------------------------------------------------------------
find_repo() {
    local d="$1" last=""
    while [[ -n "$d" && "$d" != "$last" ]]; do
        if [[ -d "$d/engines" && -d "$d/configs" && -f "$d/CMakeLists.txt" ]]; then
            printf '%s\n' "$d"; return 0
        fi
        last="$d"; d="$(dirname "$d")"
    done
    return 1
}

QUICK=0
ONLY=""
LIST_ONLY=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --quick)   QUICK=1; shift ;;
        --only)    ONLY="${2:-}"; shift 2 ;;
        --list)    LIST_ONLY=1; shift ;;
        -h|--help) sed -n '2,26p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

REPO="${REPO:-}"
if [[ -n "$REPO" ]]; then
    REPO="$(cd "$REPO" && pwd)"
else
    REPO=""
    for cand in "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" "$PWD" "$HOME/workspace/TensorRT-Alpha"; do
        if REPO="$(find_repo "$cand")"; then break; fi
        REPO=""
    done
fi
[[ -z "$REPO" ]] && { echo "ERROR: cannot locate the repository root" >&2; exit 2; }

cd "$REPO" || exit 2
# Executable now lives under a per-config subdir: build/bin/{Release,Debug}/trt_alpha
# (the older flat build/bin/trt_alpha is still accepted). Override with BIN=/path.
if [[ -z "${BIN:-}" ]]; then
    for cand in "$REPO/build/bin/Release/trt_alpha" \
                "$REPO/build/bin/Debug/trt_alpha" \
                "$REPO/build/bin/RelWithDebInfo/trt_alpha" \
                "$REPO/build/bin/trt_alpha"; do
        [[ -x "$cand" ]] && { BIN="$cand"; break; }
    done
fi
if [[ -z "${BIN:-}" || ! -x "$BIN" ]]; then
    echo "ERROR: trt_alpha not found under $REPO/build/bin/ - build it first" >&2
    echo "       (looked in Release / Debug / RelWithDebInfo / flat)" >&2
    exit 2
fi

TS="$(date +%Y%m%d-%H%M%S)"
OUT="$REPO/save/_run_matrix/$TS"
LOGDIR="$OUT/logs"
mkdir -p "$LOGDIR" || exit 2
RESULTS="$OUT/results.tsv"
: > "$RESULTS"

# --------------------------------------------------------------------------
# Ctrl+C must stop the whole run. Without a trap, `timeout` (which puts the
# command into its own process group by default) makes the terminal signal
# unreliable: it may only kill the current case and the loops keep going.
# --------------------------------------------------------------------------
on_signal() {
    echo "" >&2
    echo ">>> INTERRUPTED - stopping the matrix. Finished cases are kept in:" >&2
    echo "    $RESULTS" >&2
    exit 130
}
trap 'on_signal INT'  INT
trap 'on_signal TERM' TERM
trap 'on_signal HUP'  HUP
trap 'on_signal QUIT' QUIT

# --------------------------------------------------------------------------
# Ctrl+C must stop the whole run. Without a trap, `timeout` (which puts the
# command into its own process group by default) makes the terminal signal
# unreliable: it may only kill the current case and the loops keep going.
# --------------------------------------------------------------------------
on_signal() {
    echo "" >&2
    echo ">>> INTERRUPTED - stopping the matrix. Finished cases are kept in:" >&2
    echo "    $RESULTS" >&2
    exit 130
}
trap 'on_signal INT'  INT
trap 'on_signal TERM' TERM
trap 'on_signal HUP'  HUP
trap 'on_signal QUIT' QUIT

TIMEOUT=150
TOTAL=0; N_PASS=0; N_FAIL=0

# --------------------------------------------------------------------------
# run_case <model-or-dash> <case-id> <run-args...>
# --------------------------------------------------------------------------
run_case() {
    local model="$1"; shift
    local id="$1";    shift
    TOTAL=$((TOTAL + 1))

    local slug log t0 t1 rc dur nerr first reason status
    slug="$(printf '%03d' "$TOTAL")_$(printf '%s' "$id" | tr -c 'A-Za-z0-9._-' '_')"
    log="$LOGDIR/$slug.log"

    t0=$(date +%s.%N)
    # --foreground: keep trt_alpha in our own process group so the terminal Ctrl+C
    # reaches it directly (pairs with the trap above).
    timeout -k 5 --foreground "$TIMEOUT" "$BIN" run "$@" >"$log" 2>&1
    rc=$?
    t1=$(date +%s.%N)
    dur=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.2f", b-a}')

    nerr=$(grep -cE '\[(ERROR|FATAL)\]' "$log")
    first=$(grep -m1 -E '\[(ERROR|FATAL)\]' "$log" \
            | sed -E 's/^.*\[(ERROR|FATAL)\] //; s/^\[[A-Za-z]+ [A-Za-z]+\] //' \
            | cut -c1-170)
    reason="-"
    [[ "$rc" -eq 124 ]] && reason="TIMEOUT>${TIMEOUT}s"

    if [[ "$rc" -eq 0 && "$nerr" -eq 0 ]]; then
        status=PASS; N_PASS=$((N_PASS + 1))
    else
        status=FAIL; N_FAIL=$((N_FAIL + 1))
    fi

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$status" "$model" "$id" "$rc" "$nerr" "$dur" "$reason" "$*" "$first" >> "$RESULTS"

    printf '[%-4s] %-50s rc=%-3s err=%-2s %6ss\n' "$status" "$id" "$rc" "$nerr" "$dur"
    [[ "$status" == FAIL && -n "$first" ]] && printf '         -> %s\n' "$first"
    return 0
}

hr() { printf '%s\n' "------------------------------------------------------------------------"; }

# --------------------------------------------------------------------------
# discover models / sources
# --------------------------------------------------------------------------
mapfile -t MODELS < <("$BIN" list 2>/dev/null | sed -n 's/^  \([A-Za-z0-9_]\{1,\}\)$/\1/p')
[[ ${#MODELS[@]} -eq 0 ]] && { echo "ERROR: no registered models found" >&2; exit 2; }

mapfile -t IMAGES < <(find data -maxdepth 1 -type f \
    \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.bmp' \) | sort)
mapfile -t VIDEOS < <(find data -maxdepth 1 -type f \
    \( -iname '*.mp4' -o -iname '*.avi' -o -iname '*.mkv' \) | sort)

PRIMARY="data/bus.jpg"
[[ -f "$PRIMARY" ]] || PRIMARY="${IMAGES[0]:-}"

if [[ $QUICK -eq 1 ]]; then
    IMAGES=("$PRIMARY")
    VIDEOS=()
fi

engine_of() {
    local ini="$REPO/configs/$1.ini"
    [[ -f "$ini" ]] || { printf ''; return; }
    grep -iE '^[[:space:]]*engine[[:space:]]*=' "$ini" | head -n1 \
        | cut -d= -f2- | tr -d ' \r'
}

echo "repo        : $REPO"
echo "binary      : $BIN"
echo "timestamp   : $TS"
echo "models (${#MODELS[@]}) : ${MODELS[*]}"
echo "primary img : $PRIMARY"
echo "images (${#IMAGES[@]})"
echo "videos (${#VIDEOS[@]}) : ${VIDEOS[*]:-none}"
hr

if [[ $LIST_ONLY -eq 1 ]]; then
    echo "PLAN: phase1 $(( ${#MODELS[@]} * ${#IMAGES[@]} )) + phase2 $(( ${#MODELS[@]} * ${#VIDEOS[@]} ))" \
         "+ phase3 ${#MODELS[@]} + phase4 ~$(( ${#MODELS[@]} * 11 )) + phase5 ~27 cases"
    exit 0
fi

# ==========================================================================
echo "PHASE 0  baseline: primary image x every model"
for m in "${MODELS[@]}"; do
    [[ -n "$ONLY" && "$m" != *"$ONLY"* ]] && continue
    run_case "$m" "p0_${m}" --source "$PRIMARY" --net "$m"
done
hr

# ==========================================================================
echo "PHASE 1  image matrix: ${#IMAGES[@]} image(s) x ${#MODELS[@]} models"
for m in "${MODELS[@]}"; do
    [[ -n "$ONLY" && "$m" != *"$ONLY"* ]] && continue
    for img in "${IMAGES[@]}"; do
        run_case "$m" "p1_${m}_$(basename "$img")" --source "$img" --net "$m"
    done
done
hr

# ==========================================================================
echo "PHASE 2  video matrix: ${#VIDEOS[@]} video(s) x ${#MODELS[@]} models"
for m in "${MODELS[@]}"; do
    [[ -n "$ONLY" && "$m" != *"$ONLY"* ]] && continue
    for v in "${VIDEOS[@]}"; do
        run_case "$m" "p2_${m}_$(basename "$v")" --source "$v" --net "$m"
    done
done
hr

# ==========================================================================
echo "PHASE 3  directory source: data/ x ${#MODELS[@]} models"
for m in "${MODELS[@]}"; do
    [[ -n "$ONLY" && "$m" != *"$ONLY"* ]] && continue
    run_case "$m" "p3_${m}_dir" --source data/ --net "$m"
done
hr

# ==========================================================================
echo "PHASE 4  option matrix on the primary image"
for m in "${MODELS[@]}"; do
    [[ -n "$ONLY" && "$m" != *"$ONLY"* ]] && continue

    run_case "$m" "p4_${m}_save"          --source "$PRIMARY" --net "$m" --save
    run_case "$m" "p4_${m}_workers2"      --source "$PRIMARY" --net "$m" --workers 2
    run_case "$m" "p4_${m}_workers4"      --source "$PRIMARY" --net "$m" --workers 4
    run_case "$m" "p4_${m}_batch1"        --source "$PRIMARY" --net "$m" --batch 1
    run_case "$m" "p4_${m}_batch2"        --source "$PRIMARY" --net "$m" --batch 2
    run_case "$m" "p4_${m}_batch4"        --source "$PRIMARY" --net "$m" --batch 4
    run_case "$m" "p4_${m}_batch8"        --source "$PRIMARY" --net "$m" --batch 8
    run_case "$m" "p4_${m}_cfg_explicit"  --source "$PRIMARY" --net "$m" --config "configs/$m.ini"
    run_case "$m" "p4_${m}_workers2_batch2" --source "$PRIMARY" --net "$m" --workers 2 --batch 2

    eng="$(engine_of "$m")"
    if [[ -n "$eng" && "$eng" != *'<'* && -f "$eng" ]]; then
        run_case "$m" "p4_${m}_eng_explicit" --source "$PRIMARY" --net "$m" --engine "$eng"
    else
        run_case "$m" "p4_${m}_eng_missing"  --source "$PRIMARY" --net "$m" --engine "$eng"
    fi

    if [[ ${#VIDEOS[@]} -gt 0 ]]; then
        run_case "$m" "p4_${m}_video_save" --source "${VIDEOS[0]}" --net "$m" --save
    fi
done
hr

# ==========================================================================
echo "PHASE 5  error injection / argument validation"
ecase() { run_case "-" "$@"; }

ecase "p5_no_source"           --net yolov26
ecase "p5_bad_source_path"     --source data/__nope__.jpg --net yolov26
ecase "p5_bad_source_dir"      --source data/__nope__/ --net yolov26
ecase "p5_bad_net"             --source "$PRIMARY" --net __nope__
ecase "p5_default_net"         --source "$PRIMARY"
ecase "p5_bad_engine"          --source "$PRIMARY" --net yolov26 --engine engines/__nope__.trt
ecase "p5_bad_config"          --source "$PRIMARY" --net yolov26 --config configs/__nope__.ini
ecase "p5_engine_is_dir"       --source "$PRIMARY" --net yolov26 --engine engines
ecase "p5_config_is_dir"       --source "$PRIMARY" --net yolov26 --config configs
ecase "p5_engine_not_engine"   --source "$PRIMARY" --net yolov26 --engine data/bus.jpg
ecase "p5_batch0"              --source "$PRIMARY" --net yolov26 --batch 0
ecase "p5_batch_neg"           --source "$PRIMARY" --net yolov26 --batch -1
ecase "p5_batch_huge"          --source "$PRIMARY" --net yolov26 --batch 64
ecase "p5_batch_nonnum"        --source "$PRIMARY" --net yolov26 --batch abc
ecase "p5_workers0"            --source "$PRIMARY" --net yolov26 --workers 0
ecase "p5_workers_neg"         --source "$PRIMARY" --net yolov26 --workers -1
ecase "p5_workers_nonnum"      --source "$PRIMARY" --net yolov26 --workers abc
ecase "p5_unknown_option"      --source "$PRIMARY" --net yolov26 --totally-unknown 1
ecase "p5_source_bad_video"    --source data/__nope__.mp4 --net yolov26
ecase "p5_source_empty"        --source "" --net yolov26
ecase "p5_save_dir_nested"     --source "$PRIMARY" --net yolov26 --save save/__tmp__/a/b
ecase "p5_dup_source"          --source "$PRIMARY" --source "$PRIMARY" --net yolov26
ecase "p5_dup_net"             --source "$PRIMARY" --net yolov8 --net yolov26
ecase "p5_engine_cross_model"  --source "$PRIMARY" --net yolov8 --engine engines/yolov26n.trt
ecase "p5_root_tmp"            --source "$PRIMARY" --net yolov26 --root /tmp
ecase "p5_source_nonmedia"     --source CMakeLists.txt --net yolov26
ecase "p5_source_classes_dir"  --source data/classes/ --net yolov26

TIMEOUT=25
ecase "p5_camera_999"          --source 999 --net yolov26
ecase "p5_stream_dead"         --source "rtsp://127.0.0.1:1/none" --net yolov26
hr

# ==========================================================================
# report
# ==========================================================================
SUMFILE="$OUT/summary.txt"
FAILFILE="$OUT/failures.txt"

awk -F'\t' '$1=="FAIL"{ printf "%-34s %-40s rc=%-4s err=%-3s %7ss  %s\n", $2, $3, $4, $5, $6, $9 }' \
    "$RESULTS" > "$FAILFILE"

{
    echo "trt_alpha run matrix  --  $TS"
    echo "repo   : $REPO"
    echo "binary : $BIN"
    echo "total  : $TOTAL    PASS: $N_PASS    FAIL: $N_FAIL"
    echo
    echo "== PER-MODEL ROLL-UP ===================================================="
    awk -F'\t' '
        { tot[$2]++; if ($1=="FAIL") bad[$2]++ }
        END {
            printf "%-22s %7s %7s %5s\n", "model", "cases", "fails", "ok?"
            for (k in tot) printf "%-22s %7d %7d %5s\n", k, tot[k], bad[k]+0, (bad[k]+0==0?"YES":"NO")
        }' "$RESULTS" | { read -r h; echo "$h"; sort; }
    echo
    echo "== FAILED CASES ($N_FAIL) ==============================================="
    if [[ $N_FAIL -eq 0 ]]; then
        echo "(none)"
    else
        cat "$FAILFILE"
    fi
} > "$SUMFILE"

echo
cat "$SUMFILE"
echo
echo "results  -> $RESULTS"
echo "failures -> $FAILFILE"
echo "summary  -> $SUMFILE"
echo "logs     -> $LOGDIR/"

[[ $N_FAIL -eq 0 ]] && exit 0 || exit 1
