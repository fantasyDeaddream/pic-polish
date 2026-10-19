#!/usr/bin/env bash
# =============================================================================
#  PicPolish 服务管理脚本（Linux / WSL Ubuntu）
#
#  用法（一般不需要直接调用，用根目录的 start.sh / stop.sh 即可）：
#      bash scripts/server.sh start   [--port 8798] [--host 127.0.0.1] [--no-browser]
#      bash scripts/server.sh stop
#      bash scripts/server.sh status
#      bash scripts/server.sh restart [--port ...] [--no-browser]
#      bash scripts/server.sh logs    [--lines 40]
#
#  约定（docs/CONTRACT.md §4、docs/API.md §12）：
#    * Python 固定使用项目 venv：.venv-linux/bin/python
#    * 启动命令: python -m uvicorn app.main:app --host <host> --port <port>
#    * nohup 后台运行，stdout/stderr → logs/server.log / logs/server.err.log
#    * 运行信息 → .runtime/server.json（{pid,host,port,url,started_at,engine,log}）
#    * stop 校验 PID 命令行（防误杀），清空 cache/uploads|results|tmp 后删 runtime；
#      幂等：未运行时也清理并返回 0
#    * 引擎可用环境变量 PIC_POLISH_ENGINE 强制：auto | realesrgan | pil
#    * 分发版核心：.venv-linux 缺失 → 自动运行 scripts/setup_env.sh；
#      模型权重缺失/不完整 → 自动运行 scripts/fetch_models.py 下载（最长 20 分钟，
#      失败/超时只提示不阻塞，引擎会自动降级）；可用 PIC_POLISH_AUTO_SETUP=0 /
#      PIC_POLISH_AUTO_DOWNLOAD=0 关闭自动行为
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VENV_DEFAULT="$ROOT/.venv-linux/bin/python"
PY="${PIC_POLISH_VENV_PYTHON:-$VENV_DEFAULT}"

CACHE_DIR="$ROOT/cache"
UPLOAD_DIR="$CACHE_DIR/uploads"
RESULT_DIR="$CACHE_DIR/results"
TMP_DIR="$CACHE_DIR/tmp"
LOG_DIR="$ROOT/logs"
OUT_LOG="$LOG_DIR/server.log"
ERR_LOG="$LOG_DIR/server.err.log"
RUNTIME_DIR="$ROOT/.runtime"
RUNTIME_FILE="$RUNTIME_DIR/server.json"

# 首次启动自动配置
SETUP_SCRIPT="$ROOT/scripts/setup_env.sh"
FETCH_MODELS="$ROOT/scripts/fetch_models.py"
MODELS_DIR="${PIC_POLISH_MODELS_DIR:-$ROOT/models}"
MODEL_DL_LOG="$LOG_DIR/models_download.log"
MODEL_DL_ERR="$LOG_DIR/models_download.err.log"
MODEL_DL_TIMEOUT="${PIC_POLISH_MODEL_DOWNLOAD_TIMEOUT:-1200}"

DEFAULT_PORT="${PIC_POLISH_PORT:-8765}"
DEFAULT_HOST="${PIC_POLISH_HOST:-127.0.0.1}"
START_TIMEOUT="${PIC_POLISH_START_TIMEOUT:-90}"

if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then
    C_TITLE=$'\033[36m'; C_OK=$'\033[32m'; C_WARN=$'\033[33m'; C_ERR=$'\033[31m'; C_DIM=$'\033[2m'; C_RST=$'\033[0m'
else
    C_TITLE=''; C_OK=''; C_WARN=''; C_ERR=''; C_DIM=''; C_RST=''
fi
title() { printf '\n%s==== %s ====%s\n' "$C_TITLE" "$1" "$C_RST"; }
info()  { printf '  %s\n' "$1"; }
ok()    { printf '  %s[OK] %s%s\n' "$C_OK" "$1" "$C_RST"; }
warn()  { printf '  %s[警告] %s%s\n' "$C_WARN" "$1" "$C_RST"; }
fail()  { printf '  %s[错误] %s%s\n' "$C_ERR" "$1" "$C_RST"; }

usage() {
    cat <<'EOF'
PicPolish 服务管理（Linux/WSL）

用法:
  bash scripts/server.sh <action> [选项]
  bash start.sh [选项]     # 等价于 server.sh start
  bash stop.sh             # 等价于 server.sh stop

动作:
  start      后台启动服务（默认 127.0.0.1:8765，端口被占用自动顺延）
  stop       停止服务并清空 cache/uploads|results|tmp（幂等）
  restart    先 stop 再 start
  status     查看运行状态 / 引擎 / 缓存
  logs       查看日志尾部

选项:
  --port <n>       监听端口（默认 $PIC_POLISH_PORT 或 8765）
  --host <addr>    监听地址（默认 $PIC_POLISH_HOST 或 127.0.0.1）
  --no-browser     启动后不自动打开浏览器
  --foreground     前台运行 uvicorn（调试用）
  --lines <n>      logs 动作显示的行数（默认 40）

环境变量:
  PIC_POLISH_ENGINE=auto|realesrgan|pil   强制引擎（WSL 无 torch 建议 pil）
  PIC_POLISH_PORT / PIC_POLISH_HOST       默认端口/地址
  PIC_POLISH_VENV_PYTHON                  覆盖 Python 解释器路径
EOF
}

# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------
ensure_dirs() {
    mkdir -p "$CACHE_DIR" "$UPLOAD_DIR" "$RESULT_DIR" "$TMP_DIR" "$LOG_DIR" "$RUNTIME_DIR"
}

format_bytes() {
    local bytes="${1:-0}"
    awk -v b="$bytes" 'BEGIN {
        if (b >= 1073741824) printf "%.2f GB", b/1073741824;
        else if (b >= 1048576) printf "%.2f MB", b/1048576;
        else if (b >= 1024) printf "%.1f KB", b/1024;
        else printf "%d B", b;
    }'
}

# 端口是否可绑定（在目标 host 上试 bind；默认 127.0.0.1）
port_free() {
    local port="$1" host="${2:-127.0.0.1}"
    python3 - "$port" "$host" <<'PY'
import socket, sys
port = int(sys.argv[1])
host = sys.argv[2]
if host in ("0.0.0.0", "*"):
    host = "0.0.0.0"
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    sock.bind((host, port))
except OSError:
    sys.exit(1)
finally:
    sock.close()
sys.exit(0)
PY
}

# 打印当前所有监听 TCP 的 PID（每行一个）
listener_pids() {
    if command -v ss >/dev/null 2>&1; then
        # shellcheck disable=SC2016
        ss -ltnp 2>/dev/null | sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' | sort -u
        return 0
    fi
    python3 - <<'PY'
import glob, os

inodes = set()
for path in ("/proc/net/tcp", "/proc/net/tcp6"):
    try:
        with open(path, encoding="ascii") as fh:
            lines = fh.read().splitlines()[1:]
    except OSError:
        continue
    for line in lines:
        parts = line.split()
        if len(parts) > 9 and parts[3] == "0A":  # 0A = LISTEN
            inodes.add(parts[9])

pids = set()
for fd_dir in glob.glob("/proc/[0-9]*/fd"):
    try:
        entries = os.listdir(fd_dir)
    except OSError:
        continue
    proc_id = fd_dir.split("/")[2]
    for fd in entries:
        try:
            target = os.readlink(os.path.join(fd_dir, fd))
        except OSError:
            continue
        if target.startswith("socket:[") and target[8:-1] in inodes:
            pids.add(proc_id)
            break

for proc_id in sorted(pids, key=int):
    print(proc_id)
PY
}

proc_cmdline() {
    tr '\0' ' ' < "/proc/$1/cmdline" 2>/dev/null || true
}

# 校验 PID 是本项目的 uvicorn（防 PID 复用误杀）
is_own_process() {
    local proc_id="$1" cmd
    [ -n "$proc_id" ] || return 1
    kill -0 "$proc_id" 2>/dev/null || return 1
    cmd="$(proc_cmdline "$proc_id")"
    case "$cmd" in
        *uvicorn*) ;;
        *) return 1 ;;
    esac
    case "$cmd" in
        *"$ROOT"*|*"$PY"*) return 0 ;;
        *) return 1 ;;
    esac
}

# 沿祖先链向上最多 4 层寻找“本项目的 uvicorn”进程，成功打印其 PID
resolve_own_process() {
    local proc_id="$1" depth=0 rest ppid
    while [ "$depth" -lt 4 ] && [ -n "$proc_id" ] && [ "$proc_id" -gt 1 ]; do
        if is_own_process "$proc_id"; then
            printf '%s\n' "$proc_id"
            return 0
        fi
        [ -r "/proc/$proc_id/stat" ] || return 1
        rest="$(sed -n 's/^[0-9][0-9]* (.*) //p' "/proc/$proc_id/stat" 2>/dev/null || true)"
        ppid="$(printf '%s' "$rest" | awk '{print $2}')"
        [ -n "$ppid" ] || return 1
        proc_id="$ppid"
        depth=$((depth + 1))
    done
    return 1
}

# 终止进程（先 TERM，最多等 8 秒，再 KILL），成功返回 0
terminate_pid() {
    local proc_id="$1" deadline
    [ -n "$proc_id" ] || return 1
    # 先尝试子进程
    local children
    children="$(pgrep -P "$proc_id" 2>/dev/null || true)"
    if [ -n "$children" ]; then
        # shellcheck disable=SC2086
        kill -TERM $children 2>/dev/null || true
    fi
    kill -TERM "$proc_id" 2>/dev/null || true
    deadline=$((SECONDS + 8))
    while kill -0 "$proc_id" 2>/dev/null && [ "$SECONDS" -lt "$deadline" ]; do
        sleep 0.2
    done
    if kill -0 "$proc_id" 2>/dev/null; then
        if [ -n "$children" ]; then
            # shellcheck disable=SC2086
            kill -KILL $children 2>/dev/null || true
        fi
        kill -KILL "$proc_id" 2>/dev/null || true
        sleep 0.3
    fi
    ! kill -0 "$proc_id" 2>/dev/null
}

# 读取 runtime JSON 字段；不存在返回 1
runtime_get() {
    local key="$1"
    [ -f "$RUNTIME_FILE" ] || return 1
    python3 - "$RUNTIME_FILE" "$key" <<'PY'
import json, sys
try:
    with open(sys.argv[1], encoding="utf-8") as fh:
        data = json.load(fh)
except Exception:
    sys.exit(1)
value = data.get(sys.argv[2])
if value is None:
    sys.exit(1)
if isinstance(value, (dict, list)):
    print(json.dumps(value, ensure_ascii=False))
else:
    print(value)
PY
}

# 写 runtime JSON
write_runtime() {
    local server_pid="$1" host="$2" port="$3" url="$4" started="$5" engine_json="${6:-}"
    ensure_dirs
    python3 - "$RUNTIME_FILE" "$server_pid" "$host" "$port" "$url" "$started" "$engine_json" "$PY" <<'PY'
import json, sys
path, pid, host, port, url, started, engine_json, python = sys.argv[1:9]
data = {
    "pid": int(pid),
    "host": host,
    "port": int(port),
    "url": url,
    "started_at": started,
    "engine": json.loads(engine_json) if engine_json else None,
    "python": python,
    "log": {"stdout": "logs/server.log", "stderr": "logs/server.err.log"},
}
with open(path, "w", encoding="utf-8") as fh:
    json.dump(data, fh, ensure_ascii=False, indent=2)
    fh.write("\n")
PY
}

remove_runtime() {
    if [ -f "$RUNTIME_FILE" ]; then
        rm -f "$RUNTIME_FILE"
        return 0
    fi
    return 1
}

health_get() {
    local url="$1"
    curl -fsS --max-time 2 "$url/api/health" 2>/dev/null || true
}

# 轮询 /api/health；成功打印 JSON 并返回 0
wait_health() {
    local url="$1" timeout="$2" watch_pid="${3:-0}" deadline payload
    deadline=$((SECONDS + timeout))
    while [ "$SECONDS" -lt "$deadline" ]; do
        payload="$(health_get "$url")"
        if [ -n "$payload" ]; then
            printf '%s\n' "$payload"
            return 0
        fi
        if [ "$watch_pid" -gt 0 ] && ! kill -0 "$watch_pid" 2>/dev/null; then
            return 1
        fi
        sleep 0.5
    done
    return 1
}

engine_json_from_health() {
    python3 -c 'import json, sys
try:
    data = json.load(sys.stdin)
except Exception:
    print("null")
    raise SystemExit(0)
print(json.dumps(data.get("engine")))' 2>/dev/null || printf 'null'
}

format_engine() {
    python3 - "$1" <<'PY'
import json, sys
try:
    engine = json.loads(sys.argv[1]) if sys.argv[1] else None
except Exception:
    engine = None
if not engine:
    print("未知（服务未返回引擎信息）")
    raise SystemExit(0)
if engine.get("backend") == "realesrgan":
    parts = ["AI 超分", "device=%s" % engine.get("device")]
    if engine.get("gpu"):
        parts.append("GPU=%s" % engine.get("gpu"))
    if engine.get("model"):
        parts.append("model=%s" % engine.get("model"))
    print(" / ".join(parts))
else:
    print("兼容模式（pil: Lanczos + 轻锐化）（backend=%s）" % engine.get("backend"))
PY
}

# 输出 "文件数 字节数"
cache_stats() {
    local files=0 bytes=0 line dir
    for dir in "$UPLOAD_DIR" "$RESULT_DIR" "$TMP_DIR"; do
        [ -d "$dir" ] || continue
        line="$(find "$dir" -type f -printf '%s\n' 2>/dev/null | awk '{n += 1; s += $1} END {printf "%d %d", n + 0, s + 0}')" || line="0 0"
        files=$((files + ${line% *}))
        bytes=$((bytes + ${line#* }))
    done
    printf '%d %d\n' "$files" "$bytes"
}

tail_log() {
    local file="$1" lines="$2"
    info "--- $file（最后 $lines 行）---"
    if [ -f "$file" ]; then
        tail -n "$lines" "$file" 2>/dev/null || true
    else
        info '（文件不存在）'
    fi
}

open_browser() {
    local url="$1"
    if [ "${NO_BROWSER:-0}" -eq 1 ]; then
        return 0
    fi
    if command -v cmd.exe >/dev/null 2>&1 && [ -n "${WSL_DISTRO_NAME:-}" ]; then
        cmd.exe /c start "" "$url" >/dev/null 2>&1 || true
        info '已在 Windows 默认浏览器中打开页面（--no-browser 可关闭）。'
    elif command -v xdg-open >/dev/null 2>&1; then
        xdg-open "$url" >/dev/null 2>&1 || true
        info '已尝试用 xdg-open 打开页面（--no-browser 可关闭）。'
    else
        info '未找到可用的浏览器打开方式（可用 --no-browser 关闭此提示）。'
    fi
}

# ---------------------------------------------------------------------------
# 首次启动自动配置：环境准备 + 模型权重下载
# ---------------------------------------------------------------------------
auto_disabled() {
    local value="${1:-}"
    case "${value,,}" in
        0 | false | no | off) return 0 ;;
        *) return 1 ;;
    esac
}

# .venv-linux 缺失时自动运行 setup_env.sh；成功后返回 0
auto_setup_if_needed() {
    if [ -x "$PY" ]; then return 0; fi
    if auto_disabled "${PIC_POLISH_AUTO_SETUP:-}"; then
        info '已禁用自动环境准备（PIC_POLISH_AUTO_SETUP=0）。'
        return 1
    fi
    if [ ! -f "$SETUP_SCRIPT" ]; then
        fail "未找到 $SETUP_SCRIPT，无法自动准备环境。"
        return 1
    fi
    title '首次启动：自动准备运行环境'
    info "未检测到 $PY，自动运行 scripts/setup_env.sh（首次可能需要几分钟）..."
    info '可用 PIC_POLISH_AUTO_SETUP=0 关闭该行为。'
    if ! bash "$SETUP_SCRIPT"; then
        fail '自动环境准备失败，未找到 Python 解释器。'
        info '可手动运行: bash scripts/setup_env.sh'
        return 1
    fi
    if [ -x "$PY" ]; then
        ok '运行环境已自动准备完成'
        return 0
    fi
    fail '自动环境准备失败，未找到 Python 解释器。'
    info '可手动运行: bash scripts/setup_env.sh'
    return 1
}

# 模型权重是否全部就绪（fetch_models.py --check；只检查文件，不加载模型）
models_ready() {
    [ -f "$FETCH_MODELS" ] || return 0
    local args=("$FETCH_MODELS" "--check")
    if [ -n "${PIC_POLISH_MODELS_DIR:-}" ]; then
        args+=("--models-dir" "$PIC_POLISH_MODELS_DIR")
    fi
    "$PY" "${args[@]}" >/dev/null 2>&1
}

# 自动下载缺失的模型权重；成功返回 0，失败/超时返回 1（调用方决定是否继续启动）
auto_download_models() {
    if models_ready; then return 0; fi
    if auto_disabled "${PIC_POLISH_AUTO_DOWNLOAD:-}"; then
        info '已禁用自动下载模型权重（PIC_POLISH_AUTO_DOWNLOAD=0）。'
        return 1
    fi
    if [ ! -f "$FETCH_MODELS" ]; then
        warn "未找到 $FETCH_MODELS，跳过模型权重下载。"
        return 1
    fi
    title '首次启动：自动下载模型权重'
    info "权重目录: $MODELS_DIR"
    info "下载日志: $MODEL_DL_LOG"
    info "最长等待 $((MODEL_DL_TIMEOUT / 60)) 分钟（PIC_POLISH_MODEL_DOWNLOAD_TIMEOUT 可调，PIC_POLISH_AUTO_DOWNLOAD=0 关闭自动下载）"
    ensure_dirs
    : > "$MODEL_DL_LOG"
    : > "$MODEL_DL_ERR"
    local fetch_args=("$FETCH_MODELS")
    if [ -n "${PIC_POLISH_MODELS_DIR:-}" ]; then
        fetch_args+=("--models-dir" "$PIC_POLISH_MODELS_DIR")
    fi
    "$PY" "${fetch_args[@]}" >>"$MODEL_DL_LOG" 2>>"$MODEL_DL_ERR" &
    local dl_pid=$!
    local start_ts=$SECONDS last_report=-15 elapsed tail_line rc=0
    while kill -0 "$dl_pid" 2>/dev/null; do
        elapsed=$((SECONDS - start_ts))
        if [ "$elapsed" -ge "$MODEL_DL_TIMEOUT" ]; then
            warn "模型下载超过 ${MODEL_DL_TIMEOUT} 秒，终止下载进程（已下载内容会保留为 .part，可稍后重试）。"
            kill -TERM "$dl_pid" 2>/dev/null || true
            sleep 1
            kill -KILL "$dl_pid" 2>/dev/null || true
            wait "$dl_pid" 2>/dev/null || true
            warn '服务将继续启动；缺失的模型会自动降级。'
            info "可手动运行: $PY scripts/fetch_models.py"
            return 1
        fi
        if [ $((elapsed - last_report)) -ge 10 ]; then
            last_report=$elapsed
            tail_line="$(tail -n 1 "$MODEL_DL_LOG" 2>/dev/null | tr -d '\r' | cut -c1-110 || true)"
            if [ -n "$tail_line" ]; then
                info "[${elapsed}s] $tail_line"
            else
                info "[${elapsed}s] 下载中..."
            fi
        fi
        sleep 1
    done
    wait "$dl_pid" || rc=$?
    if [ "$rc" -eq 0 ]; then
        tail -n 6 "$MODEL_DL_LOG" 2>/dev/null | sed 's/^/    /' || true
        ok '模型权重下载完成'
        return 0
    fi
    warn "模型权重下载失败（退出码 $rc）。"
    tail -n 12 "$MODEL_DL_LOG" 2>/dev/null | sed 's/^/    /' || true
    tail -n 6 "$MODEL_DL_ERR" 2>/dev/null | sed 's/^/    /' || true
    warn '服务将继续启动；缺失的模型会自动降级（可稍后手动重试下载）。'
    info "可手动运行: $PY scripts/fetch_models.py"
    return 1
}

# ---------------------------------------------------------------------------
# start
# ---------------------------------------------------------------------------
do_start() {
    ensure_dirs
    title 'PicPolish 启动'
    info "项目根目录: $ROOT"
    info "引擎模式:    ${PIC_POLISH_ENGINE:-auto}（auto=自动选择 realesrgan/pil）"

    # 0) 分发版核心：首次启动自动准备 Python 环境
    auto_setup_if_needed || return 1
    info "Python:      $PY"

    # 1) 幂等：runtime + 进程校验 + health 都通过 → 汇报现有实例
    local runtime_pid runtime_url health
    runtime_pid="$(runtime_get pid 2>/dev/null || true)"
    if [ -n "$runtime_pid" ] && is_own_process "$runtime_pid"; then
        runtime_url="$(runtime_get url 2>/dev/null || printf 'http://%s:%s' "$DEFAULT_HOST" "$(runtime_get port 2>/dev/null || echo "$DEFAULT_PORT")")"
        info "检测到已在运行（PID=$runtime_pid），等待健康检查..."
        if health="$(wait_health "$runtime_url" 30 "$runtime_pid")"; then
            ok 'PicPolish 已在运行（本次 start 为幂等调用，未重复启动）'
            info "URL:   $runtime_url"
            info "PID:   $runtime_pid"
            info "引擎:  $(format_engine "$(printf '%s' "$health" | engine_json_from_health)")"
            return 0
        fi
        fail "已有进程 PID=$runtime_pid 存活但健康检查失败，请用 logs 查看日志，或先 stop 再启动。"
        return 1
    elif [ -n "$runtime_pid" ]; then
        warn "发现陈旧的运行记录（PID=$runtime_pid 已不是本项目的 uvicorn 进程），清理后继续启动。"
        remove_runtime || true
    fi

    # 1.5) 分发版核心：首次启动自动下载缺失的模型权重（失败/超时只警告，不阻塞启动）
    if models_ready; then
        info "模型权重:   已就绪（$MODELS_DIR）"
    else
        warn "检测到模型权重缺失/不完整（$MODELS_DIR），尝试自动下载..."
        auto_download_models || true
        if models_ready; then
            ok '模型权重已就绪'
        else
            warn '仍有模型权重缺失；对应模型将由引擎自动降级。可稍后手动重试下载。'
        fi
    fi

    # 2) 选端口
    local actual_port="$PORT" free_port="" candidate own_pid
    if ! port_free "$actual_port" "$HOST"; then
        own_pid=""
        if command -v ss >/dev/null 2>&1; then
            local lp
            lp="$(ss -ltnp "sport = :$actual_port" 2>/dev/null | sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' | head -n 1)"
            if [ -n "$lp" ]; then
                own_pid="$(resolve_own_process "$lp" 2>/dev/null || true)"
            fi
        fi
        if [ -n "$own_pid" ]; then
            local url="http://$DEFAULT_HOST:$actual_port"
            health="$(wait_health "$url" 30 "$own_pid" || true)"
            ok "端口 $actual_port 上已有本项目实例在运行（PID=$own_pid），未重复启动。"
            info "URL:   $url"
            info "PID:   $own_pid"
            [ -n "$health" ] && info "引擎:  $(format_engine "$(printf '%s' "$health" | engine_json_from_health)")"
            return 0
        fi
        for candidate in $(seq 1 100); do
            free_port="$((actual_port + candidate))"
            if [ "$free_port" -gt 65535 ]; then break; fi
            if port_free "$free_port" "$HOST"; then break; fi
            free_port=""
        done
        if [ -z "$free_port" ]; then
            fail "端口 $actual_port 被占用，且向后顺延 100 个端口仍无空闲端口。"
            return 1
        fi
        warn "端口 $actual_port 已被其他程序占用，自动顺延到 $free_port"
        actual_port="$free_port"
    fi
    info "监听地址:    http://$DEFAULT_HOST:$actual_port （bind $HOST）"

    # 3) 后台启动 uvicorn
    ensure_dirs
    : > "$OUT_LOG"
    : > "$ERR_LOG"
    export PYTHONUNBUFFERED=1
    export PYTHONUTF8=1
    export PYTHONIOENCODING=utf-8
    export PIC_POLISH_HOST="$HOST"
    export PIC_POLISH_PORT="$actual_port"
    if [ -n "${PIC_POLISH_ENGINE:-}" ]; then export PIC_POLISH_ENGINE; fi

    if [ "$FOREGROUND" -eq 1 ]; then
        info '前台调试模式（Ctrl+C 结束），不写 .runtime 文件。'
        exec "$PY" -m uvicorn app.main:app --host "$HOST" --port "$actual_port"
    fi

    nohup "$PY" -m uvicorn app.main:app --host "$HOST" --port "$actual_port" >>"$OUT_LOG" 2>>"$ERR_LOG" &
    local server_pid=$!
    disown "$server_pid" 2>/dev/null || true
    info "已后台启动，PID=$server_pid"

    local started url
    started="$(date -Iseconds)"
    url="http://$DEFAULT_HOST:$actual_port"
    write_runtime "$server_pid" "$HOST" "$actual_port" "$url" "$started" ''

    # 4) 等待 health
    info "等待服务就绪（最多 ${START_TIMEOUT} 秒）..."
    if ! health="$(wait_health "$url" "$START_TIMEOUT" "$server_pid")"; then
        fail "服务在 ${START_TIMEOUT} 秒内未通过健康检查。"
        if kill -0 "$server_pid" 2>/dev/null; then
            info "清理残留进程 PID=$server_pid..."
            terminate_pid "$server_pid" || true
        fi
        remove_runtime || true
        tail_log "$ERR_LOG" 25
        tail_log "$OUT_LOG" 15
        return 1
    fi

    # 5) 回写 engine
    write_runtime "$server_pid" "$HOST" "$actual_port" "$url" "$started" "$(printf '%s' "$health" | engine_json_from_health)"

    ok 'PicPolish 启动成功'
    info "URL:    $url"
    info "PID:    $server_pid"
    info "引擎:   $(format_engine "$(printf '%s' "$health" | engine_json_from_health)")"
    info "日志:   $OUT_LOG"
    info "        $ERR_LOG"
    info "缓存:   $CACHE_DIR"
    info '停止:   bash stop.sh'
    open_browser "$url"
    return 0
}

# ---------------------------------------------------------------------------
# stop
# ---------------------------------------------------------------------------
do_stop() {
    ensure_dirs
    title 'PicPolish 停止'
    local runtime_pid runtime_port stopped=0 killed="" reason
    runtime_pid="$(runtime_get pid 2>/dev/null || true)"
    runtime_port="$(runtime_get port 2>/dev/null || true)"

    # 先快照缓存再杀进程：服务是优雅退出时（TERM）其 shutdown hook 会先清空缓存，
    # 若之后再统计就会得到 0，无法报告真实释放字节。
    local before after before_files before_bytes after_files after_bytes removed freed
    before="$(cache_stats)"
    before_files="${before% *}"; before_bytes="${before#* }"

    if [ -n "$runtime_pid" ]; then
        if is_own_process "$runtime_pid"; then
            info "终止进程 PID=$runtime_pid（来自 .runtime/server.json）..."
            if terminate_pid "$runtime_pid"; then
                stopped=$((stopped + 1))
                killed="$runtime_pid"
                ok "已停止进程 PID=$runtime_pid"
            else
                warn "进程 PID=$runtime_pid 未能停止，请手动检查。"
            fi
        else
            reason="运行记录中的 PID=$runtime_pid 已不是本项目的 uvicorn 进程（可能已退出或 PID 被复用），不会误杀"
            info "$reason"
        fi
    else
        info '未找到 .runtime/server.json'
    fi

    # 兜底：扫描所有监听进程，沿祖先链校验后终止本项目的实例
    local list_pid resolved lp
    while IFS= read -r list_pid; do
        [ -n "$list_pid" ] || continue
        resolved="$(resolve_own_process "$list_pid" 2>/dev/null || true)"
        [ -n "$resolved" ] || continue
        case " $killed " in
            *" $resolved "*) continue ;;
        esac
        info "按端口兜底发现本项目的监听进程 PID=$resolved（监听 PID=$list_pid）"
        if terminate_pid "$resolved"; then
            stopped=$((stopped + 1))
            killed="$killed $resolved"
        fi
    done <<< "$(listener_pids || true)"

    if [ "$stopped" -gt 0 ]; then
        ok "已停止 PicPolish 服务（共停止 $stopped 个进程：$(printf '%s' "$killed" | sed 's/^ *//')）"
        # 等待端口释放（最多 5 秒）
        if [ -n "$runtime_port" ]; then
            local deadline=$((SECONDS + 5))
            while [ "$SECONDS" -lt "$deadline" ]; do
                if port_free "$runtime_port" 2>/dev/null; then break; fi
                sleep 0.25
            done
        fi
    else
        info '服务未在运行。'
    fi

    # 清空缓存图片（未运行也执行）
    title '清理缓存图片'
    local dir
    for dir in "$UPLOAD_DIR" "$RESULT_DIR" "$TMP_DIR"; do
        mkdir -p "$dir"
        find "$dir" -mindepth 1 -delete 2>/dev/null || true
    done
    after="$(cache_stats)"
    after_files="${after% *}"; after_bytes="${after#* }"
    removed=$((before_files - after_files))
    freed=$((before_bytes - after_bytes))
    [ "$freed" -lt 0 ] && freed=0
    ok '已清空 cache/uploads、cache/results、cache/tmp 下的文件（保留目录）'
    info "删除文件: $removed 个，释放: $(format_bytes "$freed")"
    info "当前缓存: $after_files 个文件 / $(format_bytes "$after_bytes")"

    # 删除 runtime 文件
    if remove_runtime; then
        ok '已删除 .runtime/server.json'
    else
        info '无 .runtime/server.json 需要删除'
    fi

    title '摘要'
    info '服务状态: 已停止'
    info "释放空间: $(format_bytes "$freed")（$removed 个文件）"
    return 0
}

# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------
do_status() {
    ensure_dirs
    title 'PicPolish 状态'
    local runtime_pid runtime_url running=0 health cache files bytes
    runtime_pid="$(runtime_get pid 2>/dev/null || true)"
    runtime_url="$(runtime_get url 2>/dev/null || true)"
    cache="$(cache_stats)"
    files="${cache% *}"; bytes="${cache#* }"

    if [ -n "$runtime_pid" ] && is_own_process "$runtime_pid"; then
        running=1
        [ -n "$runtime_url" ] || runtime_url="http://$(runtime_get host 2>/dev/null || echo "$DEFAULT_HOST"):$(runtime_get port 2>/dev/null || echo "$DEFAULT_PORT")"
        ok '服务正在运行'
        info "URL:    $runtime_url"
        info "PID:    $runtime_pid"
        health="$(health_get "$runtime_url")"
        if [ -n "$health" ]; then
            info "引擎:   $(format_engine "$(printf '%s' "$health" | engine_json_from_health)")"
            local app version
            app="$(printf '%s' "$health" | python3 -c 'import json,sys
try: print(json.load(sys.stdin).get("app", ""))
except Exception: print("")' 2>/dev/null || true)"
            version="$(printf '%s' "$health" | python3 -c 'import json,sys
try: print(json.load(sys.stdin).get("version", ""))
except Exception: print("")' 2>/dev/null || true)"
            [ -n "$app" ] && info "版本:   $app $version"
        else
            warn '进程存活但 /api/health 无响应（可能正在启动或卡住）'
        fi
        local started
        started="$(runtime_get started_at 2>/dev/null || true)"
        [ -n "$started" ] && info "启动于:  $started"
        info "日志:   $OUT_LOG"
    else
        if [ -n "$runtime_pid" ]; then
            warn "运行记录里的 PID=$runtime_pid 已不是本项目的 uvicorn 进程（陈旧记录，可运行 stop 清理）"
        fi
        info '服务未运行。'
    fi
    info "缓存:   $files 个文件 / $(format_bytes "$bytes")"
    return 0
}

# ---------------------------------------------------------------------------
# logs
# ---------------------------------------------------------------------------
do_logs() {
    title 'PicPolish 日志'
    tail_log "$OUT_LOG" "$LINES"
    tail_log "$ERR_LOG" "$LINES"
    return 0
}

# ---------------------------------------------------------------------------
# 参数解析与分发
# ---------------------------------------------------------------------------
ACTION="start"
PORT="$DEFAULT_PORT"
HOST="$DEFAULT_HOST"
NO_BROWSER=0
FOREGROUND=0
LINES=40

if [ "$#" -gt 0 ]; then
    case "$1" in
        start|stop|status|restart|logs) ACTION="$1"; shift ;;
        -h|--help|help) usage; exit 0 ;;
    esac
fi

while [ "$#" -gt 0 ]; do
    case "$1" in
        --port) PORT="${2:?--port 需要一个端口值}"; shift 2 ;;
        --port=*) PORT="${1#*=}"; shift ;;
        --host) HOST="${2:?--host 需要一个地址值}"; shift 2 ;;
        --host=*) HOST="${1#*=}"; shift ;;
        --no-browser|--no_browser) NO_BROWSER=1; shift ;;
        --foreground) FOREGROUND=1; shift ;;
        --lines) LINES="${2:?--lines 需要一个数字}"; shift 2 ;;
        --lines=*) LINES="${1#*=}"; shift ;;
        -h|--help) usage; exit 0 ;;
        *) fail "未知参数: $1"; usage; exit 2 ;;
    esac
done

case "$PORT" in
    ''|*[!0-9]*) fail "端口必须是数字: $PORT"; exit 2 ;;
esac
case "$LINES" in
    ''|*[!0-9]*) fail "--lines 必须是数字: $LINES"; exit 2 ;;
esac

RC=0
case "$ACTION" in
    start)   do_start   || RC=$? ;;
    stop)    do_stop    || RC=$? ;;
    status)  do_status  || RC=$? ;;
    logs)    do_logs    || RC=$? ;;
    restart)
        do_stop || true
        do_start || RC=$?
        ;;
    *)
        fail "未知动作: $ACTION"
        usage
        RC=2
        ;;
esac
exit "$RC"
