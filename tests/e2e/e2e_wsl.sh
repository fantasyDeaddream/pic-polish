#!/usr/bin/env bash
# =============================================================================
# PicPolish WSL / Linux full-chain E2E test.
#
# Run this INSIDE Ubuntu-24.04 (from Windows:
#   pwsh: $wsl = "$env:USERPROFILE\.dsh\tools\wsl.ps1"
#         & $wsl run -WorkDir '/mnt/c/Users/Deaddream/Desktop/pic_polish' \
#                    -Cmd 'bash tests/e2e/e2e_wsl.sh'
# ):
#
#   1. scripts/setup_env.sh   (idempotent; also verifies CUDA torch in .venv-linux)
#   2. PIC_POLISH_ENGINE=auto bash start.sh --no-browser --port 8798
#      (R2: WSL now ships CUDA torch -> engine MUST be realesrgan/cuda)
#   3. curl /api/health, /api/models, /api/targets, /api/plan (exact output-size table)
#   4. POST /api/jobs with 2 samples (720p photo + tiny png) -> poll -> download ->
#      python3+PIL asserts exact size/format/content validity + model_used ->
#      ZIP -> DELETE, and records the 720p->1080p GPU wall/elapsed time
#   5. bash stop.sh -> assert cache is empty, port closed, runtime file gone,
#      server PID dead, no leftover uvicorn process
#
# Ports: WSL test owns 8798 (Windows E2E owns 8796).  The script never touches
# other ports and only force-kills a process whose /proc cmdline proves it is
# this project's uvicorn for $PORT.
#
# Env overrides:
#   PIC_POLISH_E2E_PORT=8798  PIC_POLISH_E2E_ARTIFACT_DIR=/tmp/...
#   PIC_POLISH_E2E_SKIP_SETUP=1  PIC_POLISH_E2E_START_TIMEOUT=120
#   PIC_POLISH_E2E_JOB_TIMEOUT=180
# =============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT" || { echo "FATAL: cannot cd to $ROOT"; exit 2; }

PORT="${PIC_POLISH_E2E_PORT:-8798}"
BASE="http://127.0.0.1:${PORT}"
ART="${PIC_POLISH_E2E_ARTIFACT_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/picpolish-e2e-XXXXXX")}"
WORK="$ART/work"
mkdir -p "$WORK"
START_TIMEOUT="${PIC_POLISH_E2E_START_TIMEOUT:-120}"
JOB_TIMEOUT="${PIC_POLISH_E2E_JOB_TIMEOUT:-180}"
SKIP_SETUP="${PIC_POLISH_E2E_SKIP_SETUP:-0}"

PASS=0
FAIL=0
SKIP=0
SERVER_PID=""
SERVER_STARTED=0
FAILED=()

pass() { PASS=$((PASS + 1)); printf 'PASS  %s  %s\n' "$1" "${2:-}"; }
fail() { FAIL=$((FAIL + 1)); FAILED+=("$1"); printf 'FAIL  %s  %s\n' "$1" "${2:-}"; }
skip() { SKIP=$((SKIP + 1)); printf 'SKIP  %s  %s\n' "$1" "${2:-}"; }
expect_eq() { if [ "$2" = "$3" ]; then pass "$1" "$2"; else fail "$1" "expected '$3', got '$2'"; fi; }
expect_one_of() { # name actual allowed1 allowed2 ...
    local name="$1" actual="$2"
    shift 2
    for allowed in "$@"; do
        if [ "$actual" = "$allowed" ]; then pass "$name" "$actual"; return; fi
    done
    fail "$name" "expected one of [$*], got '$actual'"
}

# ---------------------------------------------------------------------------
# tiny JSON reader (python3 stdlib only, prints nothing on error)
# ---------------------------------------------------------------------------
json_get() {
    python3 - "$1" "$2" <<'PY' 2>/dev/null
import json, sys
try:
    data = json.load(open(sys.argv[1], encoding="utf-8"))
    cur = data
    for part in sys.argv[2].split("."):
        if part == "":
            continue
        cur = cur[int(part)] if isinstance(cur, list) else cur[part]
    if isinstance(cur, bool):
        print("1" if cur else "0")
    elif cur is None:
        print("")
    elif isinstance(cur, (dict, list)):
        print(json.dumps(cur, ensure_ascii=False))
    else:
        print(cur)
except Exception:
    print("")
PY
}

# ---------------------------------------------------------------------------
# HTTP helpers (curl is always present in this distro)
# ---------------------------------------------------------------------------
CURL_ERR="$ART/curl.err"
: > "$CURL_ERR"
http_get() { # url outfile
    curl -sS -o "$2" -w '%{http_code}' --max-time 60 "$1" 2>>"$CURL_ERR" || true
}
http_post_json() { # url outfile payload
    curl -sS -o "$2" -w '%{http_code}' --max-time 60 -X POST \
        -H 'Content-Type: application/json' -d "$3" "$1" 2>>"$CURL_ERR" || true
}
http_form() { # url outfile form-args...
    local url="$1" out="$2"
    shift 2
    curl -sS -o "$out" -w '%{http_code}' --max-time 120 -X POST "$@" "$url" 2>>"$CURL_ERR" || true
}
http_delete() { # url outfile
    curl -sS -o "$2" -w '%{http_code}' --max-time 60 -X DELETE "$1" 2>>"$CURL_ERR" || true
}

port_listening() {
    if command -v ss >/dev/null 2>&1; then
        ss -ltn 2>/dev/null | awk -v p=":$PORT" '$4 ~ (p "$") {found=1} END {exit(found ? 0 : 1)}'
        return $?
    fi
    (exec 3<>"/dev/tcp/127.0.0.1/$PORT") >/dev/null 2>&1 && { exec 3>&-; return 0; }
    return 1
}

# Assert one downloaded image: exact size/format + content validity.
assert_image() { # name path width height format
    local name="$1" path="$2" want_w="$3" want_h="$4" want_fmt="$5"
    local out code
    out=$(python3 - "$path" "$want_w" "$want_h" "$want_fmt" <<'PY' 2>&1
import sys
from PIL import Image, ImageStat

path, want_w, want_h, want_fmt = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
try:
    with Image.open(path) as im:
        im.load()
        size = im.size
        fmt = (im.format or "").upper()
        gray = im.convert("L")
        hist = gray.histogram()
        total = sum(hist) or 1
        black_frac = sum(hist[:10]) / total
        mean = ImageStat.Stat(gray).mean[0]
    problems = []
    if size != (want_w, want_h):
        problems.append(f"size={size} want={(want_w, want_h)}")
    if fmt != want_fmt:
        problems.append(f"format={fmt} want={want_fmt}")
    if black_frac > 0.005:
        problems.append(f"black_frac={black_frac:.2%} > 0.5%")
    if mean < 10:
        problems.append(f"mean={mean:.1f} < 10")
    print(f"size={size} format={fmt} mean={mean:.1f} black_frac={black_frac:.3%}")
    sys.exit(0 if not problems else 1)
except Exception as exc:
    print(f"{type(exc).__name__}: {exc}")
    sys.exit(1)
PY
)
    code=$?
    if [ "$code" -eq 0 ]; then pass "$name" "$out"; else fail "$name" "$out"; fi
}

header_match() { # name headers-file regex
    if grep -qiE "$3" "$2" 2>/dev/null; then
        pass "$1" "$(grep -iE "$3" "$2" | head -1 | tr -d '\r')"
    else
        fail "$1" "header '${3}' not found in $2"
    fi
}

# ---------------------------------------------------------------------------
# Cleanup: guarantee the server started by this script is stopped.
# ---------------------------------------------------------------------------
cleanup() {
    local code=$?
    if [ "$SERVER_STARTED" = "1" ] || [ -f .runtime/server.json ] || port_listening; then
        if [ -f stop.sh ]; then
            echo "cleanup: invoking bash stop.sh --port $PORT"
            bash stop.sh --port "$PORT" >"$ART/cleanup-stop.log" 2>&1 || true
            if port_listening || [ -f .runtime/server.json ]; then
                bash stop.sh >"$ART/cleanup-stop2.log" 2>&1 || true
            fi
        fi
    fi
    if port_listening; then
        local pids pid cmdline
        pids=$(ss -ltnp 2>/dev/null | awk -v p=":$PORT" '$4 ~ (p "$")' \
            | grep -oE 'pid=[0-9]+' | cut -d= -f2 | sort -u)
        for pid in $pids; do
            if [ -r "/proc/$pid/cmdline" ] && \
               tr '\0' ' ' <"/proc/$pid/cmdline" | grep -q 'app.main:app'; then
                echo "cleanup: force-killing verified uvicorn PID $pid on port $PORT"
                kill -9 "$pid" 2>/dev/null || true
            fi
        done
    fi
    printf '==============================================================================\n'
    printf 'WSL E2E SUMMARY: PASS=%d FAIL=%d SKIP=%d\n' "$PASS" "$FAIL" "$SKIP"
    if [ "$FAIL" -gt 0 ]; then printf 'FAILED: %s\n' "${FAILED[*]}"; fi
    printf 'artifacts: %s\n' "$ART"
    printf '==============================================================================\n'
    if [ "$FAIL" -gt 0 ]; then
        exit 1
    fi
    exit "$code"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

echo "=============================================================================="
echo "PicPolish WSL E2E  root=$ROOT"
echo "base_url=$BASE  artifacts=$ART"
echo "engine: PIC_POLISH_ENGINE=auto (R2 expectation: realesrgan/cuda via WSL GPU passthrough)"
echo "=============================================================================="

# ---------------------------------------------------------------------------
# 0. Preconditions
# ---------------------------------------------------------------------------
echo "--- preconditions ---"
command -v curl >/dev/null 2>&1 && pass 'precondition.curl' "$(curl --version | head -1)" \
    || fail 'precondition.curl' 'curl missing'
if ! command -v python3 >/dev/null 2>&1; then
    fail 'precondition.python3' 'python3 missing'
else
    pass 'precondition.python3' "$(python3 -V 2>&1)"
    if python3 -c 'import PIL' >/dev/null 2>&1; then
        pass 'precondition.pillow' "PIL $(python3 -c 'import PIL; print(PIL.__version__)')"
    else
        fail 'precondition.pillow' 'python3 PIL missing -- assertions impossible'
    fi
fi
for required in start.sh stop.sh scripts/setup_env.sh; do
    if [ -e "$required" ]; then pass "precondition.$required" 'present'
    else fail "precondition.$required" "missing $required"; fi
done
for sample in samples/tiny_320x240.png samples/portrait_720x1280.jpg \
        samples/photo_1280x720.jpg samples/anime_1280x720.png \
        samples/noisy_gauss_1280x720.jpg; do
    if [ -f "$sample" ]; then pass "precondition.$sample" 'present'
    else fail "precondition.$sample" "missing $sample"; fi
done
if [ "$FAIL" -gt 0 ]; then
    echo "aborting: preconditions failed"
    exit 1
fi

# ---------------------------------------------------------------------------
# 0b. Port must be free (or cleanly stoppable through stop.sh)
# ---------------------------------------------------------------------------
if port_listening; then
    echo "port $PORT busy before start -- trying bash stop.sh first"
    bash stop.sh --port "$PORT" >"$ART/prestop.log" 2>&1 || true
    sleep 1
fi
if port_listening; then
    fail 'lifecycle.port_free' "port $PORT still busy before start"
    exit 1
fi
pass 'lifecycle.port_free' "port $PORT free"

# ---------------------------------------------------------------------------
# 1. Linux venv setup (only when missing)
# ---------------------------------------------------------------------------
echo "--- environment ---"
if [ "$SKIP_SETUP" = "1" ] && [ ! -x .venv-linux/bin/python ]; then
    fail 'setup.venv_linux' 'missing and PIC_POLISH_E2E_SKIP_SETUP=1'
    exit 1
fi
setup_preexisting=0
if [ -x .venv-linux/bin/python ]; then setup_preexisting=1; fi
if [ "$SKIP_SETUP" = "1" ]; then
    pass 'setup.venv_linux' "skipped by PIC_POLISH_E2E_SKIP_SETUP=1 (pre-existing=$setup_preexisting)"
else
    echo "running scripts/setup_env.sh (idempotent; first run may install deps)..."
    setup_start=$(date +%s)
    timeout 900 bash scripts/setup_env.sh >"$WORK/setup_env.log" 2>&1
    setup_code=$?
    setup_sec=$(( $(date +%s) - setup_start ))
    if [ "$setup_code" -eq 0 ] && [ -x .venv-linux/bin/python ]; then
        if [ "$setup_preexisting" = "1" ]; then
            pass 'setup.venv_linux' "idempotent re-run on existing .venv-linux OK (exit 0, ${setup_sec}s)"
        else
            pass 'setup.venv_linux' "setup_env.sh created .venv-linux (exit 0, ${setup_sec}s)"
        fi
        grep -E 'Python:|Web 依赖|torch' "$WORK/setup_env.log" 2>/dev/null | tail -3 | sed 's/^/    /' || true
    else
        fail 'setup.venv_linux' "setup_env.sh exit=$setup_code"
        tail -30 "$WORK/setup_env.log" | sed 's/^/    /'
        exit 1
    fi
fi

# ---------------------------------------------------------------------------
# 1b. WSL GPU sanity: .venv-linux must have CUDA torch (R2 requirement)
# ---------------------------------------------------------------------------
echo "--- gpu environment ---"
if [ -x .venv-linux/bin/python ]; then
    gpu_info=$(.venv-linux/bin/python - <<'PY' 2>&1
import sys
try:
    import torch
    ok = torch.cuda.is_available()
    name = torch.cuda.get_device_name(0) if ok else None
    print(f"torch={torch.__version__} cuda_available={ok} device={name}")
    sys.exit(0 if ok else 1)
except Exception as exc:
    print(f"torch-error: {type(exc).__name__}: {exc}")
    sys.exit(1)
PY
)
    gpu_rc=$?
    if [ "$gpu_rc" -eq 0 ]; then
        pass 'setup.torch_cuda' "$gpu_info"
    else
        fail 'setup.torch_cuda' "$gpu_info (WSL GPU regression requires CUDA torch; see CONTRACT §1)"
    fi
    venv_cv2=$(.venv-linux/bin/python -c 'import cv2; print(cv2.__version__)' 2>&1 | tail -1)
    if [ -n "$venv_cv2" ] && ! printf '%s' "$venv_cv2" | grep -qi 'error'; then
        pass 'setup.opencv' "cv2 $venv_cv2 (denoise available)"
    else
        fail 'setup.opencv' "cv2 missing/broken: $venv_cv2"
    fi
else
    fail 'setup.torch_cuda' '.venv-linux/bin/python missing'
fi

# ---------------------------------------------------------------------------
# 2. Start through the real start script (auto engine -> realesrgan/cuda)
# ---------------------------------------------------------------------------
echo "--- start ---"
PIC_POLISH_ENGINE=auto bash start.sh --no-browser --port "$PORT" \
    >"$WORK/start.log" 2>&1
start_code=$?
SERVER_STARTED=1
if [ "$start_code" -eq 0 ]; then
    pass 'lifecycle.start_exit_code' "start.sh exit=0"
else
    fail 'lifecycle.start_exit_code' "start.sh exit=$start_code"
    tail -20 "$WORK/start.log" | sed 's/^/    /'
fi

health_ok=0
for _ in $(seq 1 "$START_TIMEOUT"); do
    if curl -sf --max-time 3 -o "$WORK/health.json" "$BASE/api/health" 2>/dev/null; then
        health_ok=1
        break
    fi
    sleep 1
done
if [ "$health_ok" = "1" ]; then
    pass 'lifecycle.health_reachable' "$BASE/api/health"
else
    fail 'lifecycle.health_reachable' 'health endpoint unreachable within timeout'
    tail -20 "$WORK/start.log" | sed 's/^/    /'
    exit 1
fi

if [ -f .runtime/server.json ]; then
    pass 'lifecycle.runtime_json_created' '.runtime/server.json'
    SERVER_PID="$(json_get .runtime/server.json pid)"
    expect_eq 'lifecycle.runtime_json_port' "$(json_get .runtime/server.json port)" "$PORT"
    if [ -n "$SERVER_PID" ] && [ "$SERVER_PID" -gt 0 ] 2>/dev/null; then
        pass 'lifecycle.runtime_json_pid' "pid=$SERVER_PID"
    else
        fail 'lifecycle.runtime_json_pid' "invalid pid='$SERVER_PID'"
    fi
else
    fail 'lifecycle.runtime_json_created' '.runtime/server.json missing after start'
fi

# ---------------------------------------------------------------------------
# 3. Health / targets / plan
# ---------------------------------------------------------------------------
echo "--- health ---"
curl -sS --max-time 10 -o "$WORK/health.json" "$BASE/api/health" 2>>"$CURL_ERR" || true
expect_eq 'health.status' "$(json_get "$WORK/health.json" status)" 'ok'
expect_eq 'health.app' "$(json_get "$WORK/health.json" app)" 'PicPolish'
backend="$(json_get "$WORK/health.json" engine.backend)"
expect_eq 'health.engine.backend' "$backend" 'realesrgan'
expect_eq 'health.engine.device' "$(json_get "$WORK/health.json" engine.device)" 'cuda'
gpu_name="$(json_get "$WORK/health.json" engine.gpu)"
if [ -n "$gpu_name" ]; then pass 'health.engine.gpu' "$gpu_name"; else fail 'health.engine.gpu' 'empty'; fi
engine_model="$(json_get "$WORK/health.json" engine.model)"
if [ -n "$engine_model" ]; then pass 'health.engine.model' "$engine_model"; else fail 'health.engine.model' 'empty'; fi
native="$(json_get "$WORK/health.json" engine.native_scale)"
expect_one_of 'health.engine.native_scale' "$native" '2' '4'
expect_eq 'health.models.default' "$(json_get "$WORK/health.json" models.default)" 'auto'
health_models="$(json_get "$WORK/health.json" models.available)"
if printf '%s' "$health_models" | grep -q 'realcugan-conservative'; then
    pass 'health.models.available' "$health_models"
else
    fail 'health.models.available' "$health_models"
fi
for key in max_files max_file_mb max_total_mb max_pixels; do
    value="$(json_get "$WORK/health.json" "limits.$key")"
    if [ -n "$value" ]; then pass "health.limits.$key" "$value"; else fail "health.limits.$key" 'missing'; fi
done

echo "--- models ---"
curl -sS --max-time 10 -o "$WORK/models.json" "$BASE/api/models" 2>>"$CURL_ERR" || true
python3 - "$WORK/models.json" <<'PY'
import json, sys
EXPECTED = ["auto", "realesrgan-x4plus", "realesrgan-x2plus", "realesrgan-anime6b",
            "realcugan", "realcugan-conservative"]
try:
    payload = json.load(open(sys.argv[1], encoding="utf-8"))
except Exception as exc:
    print(f"FAIL models.parse {type(exc).__name__}: {exc}")
    sys.exit(0)
models = payload.get("models") or []
problems = []
if payload.get("default") != "auto":
    problems.append(f"default={payload.get('default')!r}")
if [m.get("id") for m in models] != EXPECTED:
    problems.append(f"ids={[m.get('id') for m in models]}")
for model in models:
    for key in ("label", "family", "native_scales", "denoise", "license", "size_mb", "source"):
        if key not in model:
            problems.append(f"{model.get('id')} missing {key}")
    if model.get("available") is not True:
        problems.append(f"{model.get('id')} available={model.get('available')!r}")
    if not isinstance((model.get("denoise") or {}).get("levels"), list):
        problems.append(f"{model.get('id')} denoise.levels invalid")
cugan = next((m for m in models if m.get("id") == "realcugan"), {})
if (cugan.get("denoise") or {}).get("levels") != [0, 1, 2, 3]:
    problems.append(f"realcugan denoise.levels={(cugan.get('denoise') or {}).get('levels')!r}")
if (cugan.get("denoise") or {}).get("native") is not True:
    problems.append("realcugan denoise.native should be True")
if problems:
    print("FAIL models.contract " + "; ".join(problems))
else:
    print("PASS models.contract (6 models fixed order/available/denoise fields)")
PY

echo "--- targets ---"
curl -sS --max-time 10 -o "$WORK/targets.json" "$BASE/api/targets" 2>>"$CURL_ERR" || true
python3 - "$WORK/targets.json" <<'PY'
import json, sys
try:
    targets = json.load(open(sys.argv[1], encoding="utf-8")).get("targets") or []
except Exception as exc:
    print("FAIL targets.parse", exc)
    sys.exit(0)
expected = [("720p", "720P", "short", 720), ("1080p", "1080P", "short", 1080),
            ("2k", "2K", "long", 2560), ("4k", "4K", "long", 3840)]
ok = len(targets) == 4
for index, (tid, label, rule, value) in enumerate(expected):
    if index >= len(targets):
        ok = False
        print(f"FAIL targets[{index}].present missing")
        continue
    actual = targets[index]
    if (actual.get("id"), actual.get("label"), actual.get("rule"), actual.get("value")) != (tid, label, rule, value):
        ok = False
        print(f"FAIL targets[{index}] got {actual!r}")
if ok:
    print("PASS targets.table (720p/1080p/2k/4k short/long values exact)")
PY

echo "--- plan ---"
plan_case() { # sample-name w h tier -> asserts output w/h
    local sample="$1" w="$2" h="$3" tier="$4"
    local payload code
    payload="{\"target\":\"$tier\",\"align\":8,\"images\":[{\"name\":\"$sample\",\"width\":$w,\"height\":$h}]}"
    code=$(http_post_json "$BASE/api/plan" "$WORK/plan.json" "$payload")
    if [ "$code" != "200" ]; then
        fail "plan.${w}x${h}@${tier}" "http $code"
        return
    fi
    local ow oh
    ow="$(json_get "$WORK/plan.json" images.0.output.width)"
    oh="$(json_get "$WORK/plan.json" images.0.output.height)"
    if [ -n "$ow" ] && [ -n "$oh" ]; then
        printf '%sx%s' "$ow" "$oh"
    else
        printf 'MISSING'
    fi
}
expect_eq 'plan.1280x720@1080p' "$(plan_case photo_1280x720.jpg 1280 720 1080p)" '1920x1080'
expect_eq 'plan.1000x667@1080p' "$(plan_case odd_1000x667.jpg 1000 667 1080p)" '1616x1080'
expect_eq 'plan.1080x1920@720p' "$(plan_case portrait_720x1280.jpg 1080 1920 720p)" '720x1280'
expect_eq 'plan.3840x2160@1080p' "$(plan_case large_3840x2160.jpg 3840 2160 1080p)" '1920x1080'
bad_code=$(http_post_json "$BASE/api/plan" "$WORK/plan-bad.json" \
    '{"target":"foo","images":[{"name":"a.jpg","width":100,"height":100}]}')
expect_eq 'plan.invalid_target.status' "$bad_code" '400'

# ---------------------------------------------------------------------------
# 4. GPU job: 720p photo -> 1080p (timed) + tiny png -> PIL, headers, ZIP, DELETE
# ---------------------------------------------------------------------------
echo "--- job (gpu) ---"
if [ ! -f samples/photo_1280x720.jpg ] || [ ! -f samples/tiny_320x240.png ]; then
    fail 'job.samples' 'missing samples'
else
    job_start_ms=$(date +%s%3N)
    create_code=$(http_form "$BASE/api/jobs" "$WORK/job.json" \
        -F "files=@samples/photo_1280x720.jpg;type=image/jpeg" \
        -F "files=@samples/tiny_320x240.png;type=image/png" \
        -F 'target=1080p' -F 'format=auto')
    expect_eq 'job.create.http_status' "$create_code" '202'
    JOB_ID="$(json_get "$WORK/job.json" job_id)"
    if [ -z "$JOB_ID" ]; then
        fail 'job.create.job_id' "no job_id in $(head -c 300 "$WORK/job.json")"
    else
        pass 'job.create.job_id' "$JOB_ID"
        expect_eq 'job.create.total' "$(json_get "$WORK/job.json" total)" '2'
        expect_eq 'job.create.files0.output' \
            "$(json_get "$WORK/job.json" files.0.output.width)x$(json_get "$WORK/job.json" files.0.output.height)" \
            '1920x1080'
        expect_eq 'job.create.files1.output' \
            "$(json_get "$WORK/job.json" files.1.output.width)x$(json_get "$WORK/job.json" files.1.output.height)" \
            '1440x1080'
        # R2 model echo fields (docs/API.md 14.4).
        expect_eq 'job.create.model.id' "$(json_get "$WORK/job.json" model.id)" 'auto'
        expect_eq 'job.create.files0.model' "$(json_get "$WORK/job.json" files.0.model)" 'auto'
        expect_eq 'job.create.files1.model' "$(json_get "$WORK/job.json" files.1.model)" 'auto'

        job_status=""
        polls=0
        for _ in $(seq 1 $((JOB_TIMEOUT * 2))); do
            polls=$((polls + 1))
            curl -sS --max-time 10 -o "$WORK/job-status.json" "$BASE/api/jobs/$JOB_ID" 2>>"$CURL_ERR" || true
            job_status="$(json_get "$WORK/job-status.json" status)"
            case "$job_status" in
                done | error | cancelled) break ;;
            esac
            sleep 0.5
        done
        expect_eq 'job.poll.terminal_status' "$job_status" 'done'
        job_elapsed="$(json_get "$WORK/job-status.json" elapsed_sec)"
        if [ -n "$job_elapsed" ]; then
            pass 'job.poll.elapsed_sec' "${job_elapsed}s (polls=$polls)"
        else
            fail 'job.poll.elapsed_sec' 'missing'
        fi
        expect_eq 'job.files0.status' "$(json_get "$WORK/job-status.json" files.0.status)" 'done'
        expect_eq 'job.files1.status' "$(json_get "$WORK/job-status.json" files.1.status)" 'done'

        # ---- WSL GPU performance record (720p -> 1080p) -----------------
        job_end_ms=$(date +%s%3N)
        wall_ms=$((job_end_ms - job_start_ms))
        wall_sec=$(awk -v ms="$wall_ms" 'BEGIN {printf "%.2f", ms / 1000.0}')
        pass 'job.gpu.720p_to_1080p_wall' "${wall_sec}s wall (2 files: 1280x720->1920x1080 + 320x240->1440x1080)"
        if [ "$wall_ms" -le 60000 ]; then
            pass 'job.gpu.within_45s_plus_overhead' "wall=${wall_sec}s <= 60s (CONTRACT §5 target 720p->1080p <= 45s)"
        else
            fail 'job.gpu.within_45s_plus_overhead' "wall=${wall_sec}s > 60s"
        fi
        used0="$(json_get "$WORK/job-status.json" files.0.model_used)"
        if printf '%s' "$used0" | grep -qi 'x2'; then
            pass 'job.files0.model_used' "$used0"
        else
            fail 'job.files0.model_used' "expected an x2 model for ratio 1.5, got '$used0'"
        fi
        expect_eq 'job.files0.denoise_level' "$(json_get "$WORK/job-status.json" files.0.denoise_level)" '0'

        # preview (inline) + download (attachment) for both results
        preview0_code=$(curl -sS --max-time 60 -D "$WORK/headers0.txt" -o "$WORK/result0.jpg" \
            -w '%{http_code}' "$BASE/api/jobs/$JOB_ID/files/0" 2>>"$CURL_ERR" || true)
        expect_eq 'job.preview0.http_status' "$preview0_code" '200'
        assert_image 'job.preview0.image' "$WORK/result0.jpg" 1920 1080 'JPEG'
        preview1_code=$(curl -sS --max-time 60 -D "$WORK/headers1.txt" -o "$WORK/result1.png" \
            -w '%{http_code}' "$BASE/api/jobs/$JOB_ID/files/1" 2>>"$CURL_ERR" || true)
        expect_eq 'job.preview1.http_status' "$preview1_code" '200'
        assert_image 'job.preview1.image' "$WORK/result1.png" 1440 1080 'PNG'
        header_match 'job.preview0.content_type' "$WORK/headers0.txt" '^content-type: *image/jpeg'
        header_match 'job.preview1.content_type' "$WORK/headers1.txt" '^content-type: *image/png'
        header_match 'job.preview0.cache_control' "$WORK/headers0.txt" '^cache-control: *.*no-cache'

        download_code=$(http_get "$BASE/api/jobs/$JOB_ID/files/0/download" "$WORK/download0.jpg")
        expect_eq 'job.download0.http_status' "$download_code" '200'
        curl -sS --max-time 60 -D "$WORK/download-headers0.txt" -o /dev/null \
            "$BASE/api/jobs/$JOB_ID/files/0/download" 2>>"$CURL_ERR" || true
        header_match 'job.download0.disposition' "$WORK/download-headers0.txt" '^content-disposition: *attachment'
        if grep -qiE 'photo_1280x720_1080p\.jpg' "$WORK/download-headers0.txt"; then
            pass 'job.download0.filename' 'photo_1280x720_1080p.jpg'
        else
            fail 'job.download0.filename' "$(grep -i 'content-disposition' "$WORK/download-headers0.txt" | head -1)"
        fi

        # ZIP
        zip_code=$(http_get "$BASE/api/jobs/$JOB_ID/download.zip" "$WORK/results.zip")
        expect_eq 'job.zip.http_status' "$zip_code" '200'
        curl -sS --max-time 60 -D "$WORK/zip-headers.txt" -o /dev/null \
            "$BASE/api/jobs/$JOB_ID/download.zip" 2>>"$CURL_ERR" || true
        header_match 'job.zip.content_type' "$WORK/zip-headers.txt" '^content-type: *application/zip'
        python3 - "$WORK/results.zip" "$JOB_ID" <<'PY'
import sys, zipfile
path, job_id = sys.argv[1], sys.argv[2]
try:
    with zipfile.ZipFile(path) as archive:
        names = sorted(archive.namelist())
        bad = archive.testzip()
    expected = ["photo_1280x720_1080p.jpg", "tiny_320x240_1080p.png"]
    if bad is not None:
        print(f"FAIL job.zip.integrity corrupt entry {bad}")
    elif names != expected:
        print(f"FAIL job.zip.entries got {names} want {expected}")
    else:
        print(f"PASS job.zip.entries {names}")
except Exception as exc:
    print(f"FAIL job.zip.opens {type(exc).__name__}: {exc}")
PY

        # DELETE
        delete_code=$(http_delete "$BASE/api/jobs/$JOB_ID" "$WORK/job-delete.json")
        expect_eq 'job.delete.http_status' "$delete_code" '200'
        expect_eq 'job.delete.deleted' "$(json_get "$WORK/job-delete.json" deleted)" '1'
        after_code=$(http_get "$BASE/api/jobs/$JOB_ID" "$WORK/job-after.json")
        expect_eq 'job.delete.get_after_404' "$after_code" '404'
    fi
fi

# ---------------------------------------------------------------------------
# 5. Dummy cache files, then stop through the real stop script
# ---------------------------------------------------------------------------
echo "--- stop ---"
for sub in uploads results tmp; do
    mkdir -p "cache/$sub"
    printf 'qa-dummy' >"cache/$sub/qa_dummy.bin"
done
pass 'lifecycle.cache_dummy_created' '3 dummy cache files created'

bash stop.sh --port "$PORT" >"$WORK/stop.log" 2>&1
stop_code=$?
if [ "$stop_code" -ne 0 ]; then
    echo "stop.sh --port failed (exit $stop_code); retrying plain bash stop.sh"
    bash stop.sh >"$WORK/stop2.log" 2>&1
    stop_code=$?
fi
SERVER_STARTED=0
if [ "$stop_code" -eq 0 ]; then
    pass 'lifecycle.stop_exit_code' "stop.sh exit=0"
else
    fail 'lifecycle.stop_exit_code' "stop.sh exit=$stop_code"
    tail -20 "$WORK/stop.log" | sed 's/^/    /'
fi

settled=0
for _ in $(seq 1 20); do
    if ! port_listening; then settled=1; break; fi
    sleep 0.5
done
if [ "$settled" = "1" ]; then
    pass 'lifecycle.port_closed' "port $PORT not listening"
else
    fail 'lifecycle.port_closed' "port $PORT still listening after stop"
fi

cache_files=$(find cache/uploads cache/results cache/tmp -type f 2>/dev/null | wc -l | tr -d ' ')
expect_eq 'lifecycle.cache_cleared' "$cache_files" '0'

if [ -f .runtime/server.json ]; then
    fail 'lifecycle.runtime_json_removed' '.runtime/server.json still exists'
else
    pass 'lifecycle.runtime_json_removed' 'runtime file deleted'
fi

pid_alive=0
if [ -n "$SERVER_PID" ]; then
    for _ in $(seq 1 20); do
        if kill -0 "$SERVER_PID" 2>/dev/null; then
            sleep 0.5
        else
            pid_alive=0
            break
        fi
        pid_alive=1
    done
    if [ "$pid_alive" = "0" ]; then
        pass 'lifecycle.server_pid_dead' "pid $SERVER_PID gone"
    else
        fail 'lifecycle.server_pid_dead' "pid $SERVER_PID still running"
    fi
else
    skip 'lifecycle.server_pid_dead' 'pid unknown'
fi

leftover=$(pgrep -af 'app.main:app' 2>/dev/null | grep -E -- "--port $PORT" || true)
if [ -z "$leftover" ]; then
    pass 'lifecycle.no_uvicorn_residue' "no app.main:app process for port $PORT"
else
    fail 'lifecycle.no_uvicorn_residue' "$leftover"
fi

echo "=============================================================================="
# trap cleanup prints the summary and the exit code
exit "$([ "$FAIL" -eq 0 ] && echo 0 || echo 1)"
