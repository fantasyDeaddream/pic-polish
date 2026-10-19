# PicPolish 部署与运维手册（v1.1 · 多模型 / WSL GPU / 首次启动自动配置）

本文面向**部署/运维人员**，覆盖首次部署与自动配置、启动停止、运行记录、日志、端口与防火墙、
引擎与多模型、离线安装（wheelhouse）、缓存治理、升级与卸载、故障排查。
接口字段请见 [API.md](API.md)，工程契约见 [CONTRACT.md](CONTRACT.md)，日常使用请看 [README.md](../README.md)。

---

## 1. 运行架构

```
浏览器 / curl
     │  HTTP (默认 http://127.0.0.1:8765)
     ▼
uvicorn + FastAPI (app.main:app, 单进程单 worker, GPU 串行)
     ├── 静态资源: web/                        (GET /, /web/*)
     ├── REST API : app/jobs.py 任务队列        (单 worker FIFO)
     ├── 模型注册表: app/models.py              (6 个模型 id / 推荐 / 许可 / 可用性)
     └── 超分引擎 : app/engine.py
            ├── rrdb  : app/rrdbnet.py  + RealESRGAN x4plus / x2plus / anime6B
            ├── cugan : app/cugan.py    + Real-CUGAN v3（2x/3x/4x 原生降噪）
            └── pil   : Lanczos + UnsharpMask（无 torch / 无权重时自动降级）
                 └── PyTorch 2.5.1 + CUDA 12.1（Windows: conda 复用或自动安装；WSL: GPU 直通）
```

- Python 解释器**固定使用项目 venv**：Windows `.venv\Scripts\python.exe`、Linux/WSL
  `.venv-linux/bin/python`；不依赖 conda 激活、不需要管理员权限。
- 服务是**单实例**模型：运行信息写在 `.runtime/server.json`，启动/停止脚本据此管理。
- 所有运行产物都在项目目录内：`cache/`、`logs/`、`.runtime/`；模型权重在 `models/`（不随代码分发）。
- **分发版核心**：解压后直接 `start.bat` / `bash start.sh` 即可，首次启动自动完成环境准备与权重下载。

---

## 2. 环境要求

| 项 | Windows | Linux / WSL |
|----|---------|-------------|
| 系统 | Windows 10 / 11（x64） | Ubuntu 22.04+（已验证 Ubuntu 24.04 / WSL2） |
| Shell | **PowerShell 7+（`pwsh`）** | `bash` |
| Python | 3.10–3.12（3.12 已验证）；可选带 torch 的 conda 环境（优先复用） | 3.10–3.12（3.12 已验证）；`python3` + `venv`/`virtualenv` |
| GPU（可选） | NVIDIA 显卡 + 驱动；自动装 cu121 torch | 同左；WSL2 需 Windows 侧 NVIDIA 驱动支持 WSL（`nvidia-smi` 可见即可），**不需要**在 WSL 内装 CUDA Toolkit |
| 磁盘 | 建议 ≥ 8 GB 空闲（torch 依赖约 5–6 GB + 模型 0.2 GB） | 同左 |
| 网络 | 首次缺 torch/权重时需要；内网可离线（见 §5） | 同左 |
| 其他 | - | `curl`、`ss`（iproute2，通常自带） |

> 没有 GPU 也能用：安装 CPU 版 torch 后 AI 超分可跑但慢；或直接 `PIC_POLISH_ENGINE=pil` 使用兼容引擎。
> 没有 torch 时服务也会自动降级 pil，**不会启动失败**。

---

## 3. 首次部署（自动配置）

### 3.1 Windows：双击 `start.bat` 即可

首次启动流程（全程窗口输出，无需手工）：

1. `.venv\Scripts\python.exe` 缺失 → 自动运行 `scripts\setup_env.ps1`：
   - 按候选顺序寻找**自带 torch** 的解释器：
     `$env:PIC_POLISH_PYTHON` → `E:\ProgramData\anaconda3new\envs\gnn\python.exe` → `novel` → `openui`
     → PATH `python3.12` / `python` / `python3`（跳过 Windows 商店占位符）；
   - 找到 → `--system-site-packages` 建 venv，复用其 torch；
   - 都没有 torch → 用第一个可用 Python 建 venv，自动 `pip install torch torchvision`：
     有 `nvidia-smi` → `https://download.pytorch.org/whl/cu121`；否则 CPU 版；
     可用 `PIC_POLISH_TORCH_INDEX` 覆盖；失败只警告（服务降级 pil，不阻塞）；
   - 补装 `requirements.txt` 缺失依赖（fastapi/uvicorn/python-multipart/pillow/numpy/opencv-python-headless）。
2. 模型权重缺失/不完整 → 自动运行 `scripts\fetch_models.py`（详见 §9），默认最长等 20 分钟；
   超时/失败打印手动命令并继续启动（缺失模型自动降级）。
3. 后台启动 uvicorn → 轮询 `/api/health`（最多 90s）→ 打印 URL/引擎 → 打开浏览器。

也可手动分步执行：

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File scripts\setup_env.ps1
& .venv\Scripts\python.exe scripts\fetch_models.py --check
& .venv\Scripts\python.exe scripts\fetch_models.py
```

### 3.2 Linux / WSL：`bash start.sh`

首次启动流程：

1. `.venv-linux/bin/python` 缺失 → 自动运行 `scripts/setup_env.sh`：
   - `python3 -m venv` 创建；系统缺 `ensurepip`（未装 `python3-venv`）时用
     `pip --user --break-system-packages virtualenv` 兜底；
   - **优先离线安装 torch**：本地 wheelhouse（`$PIC_POLISH_WHEELHOUSE` → `~/picpolish_wheels`
     → `<项目>/wheelhouse` → `~/.cache/pip` 中的 wheel）→ 直接 `zipfile` 解包到 venv
     site-packages（绕过 pip 重下 2.9GB）→ `pip install --no-deps` 补装少量小依赖
     （`nvidia-cuda-runtime-cu12==12.1.105`、`nvidia-nvtx-cu12==12.1.105`、`mpmath`、`filelock`、
     `jinja2`、`fsspec`）；
   - 无 wheelhouse / 离线失败 → 联网安装 `torch==2.5.1 torchvision==0.20.1`（`PIC_POLISH_TORCH_INDEX`，
     默认 `https://download.pytorch.org/whl/cu121`，再兜底 PyPI）；
   - 安装 `requirements-linux.txt`（含 opencv）；校验 `torch.cuda.is_available()`。
2. 模型权重缺失 → 自动运行 `scripts/fetch_models.py`（同 Windows，最长 20 分钟）。
3. `nohup` 启动 → 轮询 `/api/health` → 打印 URL/引擎。

```bash
cd /path/to/pic_polish
PIC_POLISH_ENGINE=auto bash start.sh --no-browser --port 8798   # GPU 可用时默认 realesrgan/cuda
bash scripts/setup_env.sh                                       # 手动准备/修复（幂等）
```

### 3.3 首次启动开关与超时

| 环境变量 | 默认 | 作用 |
|----------|------|------|
| `PIC_POLISH_AUTO_SETUP` | `1` | venv 缺失时自动运行 setup_env；设 `0` 关闭 |
| `PIC_POLISH_AUTO_DOWNLOAD` | `1` | 权重缺失时自动下载；设 `0` 关闭（只提示） |
| `PIC_POLISH_MODEL_DOWNLOAD_TIMEOUT` | `1200` | 自动下载最长等待秒数 |
| `PIC_POLISH_MODELS_DIR` | `<项目>/models` | 权重目录（测试/多套权重隔离时可用） |

首次启动日志：`logs/models_download.log`（下载进度）、`logs/models_download.err.log`（下载错误）。

---

## 4. 日常运维命令

### 4.1 Windows（`scripts/server.ps1`）

| 操作 | 命令 |
|------|------|
| 启动 | `pwsh -NoProfile -ExecutionPolicy Bypass -File scripts\server.ps1 -Action start [-Port 8795] [-Host 127.0.0.1] [-NoBrowser]` |
| 停止 + 清缓存 | `pwsh -NoProfile -ExecutionPolicy Bypass -File scripts\server.ps1 -Action stop` |
| 状态 | `pwsh -NoProfile -ExecutionPolicy Bypass -File scripts\server.ps1 -Action status` |
| 日志 | `pwsh -NoProfile -ExecutionPolicy Bypass -File scripts\server.ps1 -Action logs [-Lines 80]` |
| 重启 | `pwsh -NoProfile -ExecutionPolicy Bypass -File scripts\server.ps1 -Action restart [-NoBrowser]` |
| 前台调试 | 追加 `-Foreground`（Ctrl+C 退出，不写 runtime 文件） |
| 双击 | 根目录 `start.bat` / `stop.bat`（`.bat` 仅 ASCII，参数会透传） |

退出码：`0` 成功（含"本来就没运行"的 stop、幂等 start）；`1` 失败（环境准备失败、启动超时等）。

### 4.2 Linux / WSL（`scripts/server.sh`）

| 操作 | 命令 |
|------|------|
| 启动 | `bash start.sh [--port 8798] [--host 127.0.0.1] [--no-browser]` |
| 停止 + 清缓存 | `bash stop.sh` |
| 状态 | `bash scripts/server.sh status` |
| 日志 | `bash scripts/server.sh logs [--lines 80]` |
| 重启 | `bash scripts/server.sh restart [--no-browser]` |
| 前台调试 | `bash start.sh --foreground` |
| 强制引擎 | `PIC_POLISH_ENGINE=pil bash start.sh --no-browser` |

`start.sh` / `stop.sh` 只是 `scripts/server.sh <action>` 的薄包装，参数原样透传。

---

## 5. 离线安装 / 内网分发（wheelhouse）

### 5.1 Linux / WSL（推荐）

1. 在**联网机器**准备 wheelhouse（2.9 GB 左右，含 torch/torchvision/全部 `nvidia_*` 依赖）：

   ```bash
   mkdir -p picpolish_wheels
   # 方式 A：直接下载 cu121 全家桶
   pip download "torch==2.5.1+cu121" "torchvision==0.20.1+cu121" \
       -d picpolish_wheels --index-url https://download.pytorch.org/whl/cu121
   # 方式 B：从已有 pip 缓存提取（本机采用的方式）
   find ~/.cache/pip -name '*.whl' -exec cp -n {} picpolish_wheels/ \;
   ```

2. 拷贝 wheelhouse 到目标机器（例如 `~/picpolish_wheels` 或项目内 `wheelhouse/`），可选也放入
   `requirements-linux.txt` 的依赖 wheel（`pip download -r requirements-linux.txt -d picpolish_wheels`）。

3. 目标机器执行：

   ```bash
   PIC_POLISH_WHEELHOUSE=~/picpolish_wheels bash scripts/setup_env.sh
   # 或 wheelhouse 放在默认位置（~/picpolish_wheels）时：bash scripts/setup_env.sh
   ```

   脚本会用 `zipfile` 把 wheel 直接解包到 venv `site-packages`（数秒），再 pip 补装
   `nvidia-cuda-runtime-cu12 / nvidia-nvtx-cu12 / mpmath / filelock / jinja2 / fsspec`。
   这些包很小；若完全无网，可提前把它们的 wheel 也放进 `/tmp/nvwheels` 或 wheelhouse。

4. 校验：`.venv-linux/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`

> 脚本是幂等的：torch 已可用时直接跳过安装，不会破坏现有 venv。

### 5.2 Windows

- 若有带 torch 的 conda 环境 → 无需下载，`setup_env.ps1` 自动复用；
- 纯新机器 → 自动从 PyTorch 官方源安装 cu121（有 NVIDIA）/cpu（无 NVIDIA）版；
- 内网镜像：

  ```powershell
  $env:PIC_POLISH_TORCH_INDEX = 'https://内网镜像/pytorch/whl/cu121'
  pwsh -File scripts\setup_env.ps1
  ```

- 完全离线：准备一份已建好的 `.venv`（含 torch）随包分发，或把 torch wheel 放到本地 HTTP 目录并用
  `PIC_POLISH_TORCH_INDEX` 指向它。

### 5.3 模型权重离线

联网机器执行 `python scripts/fetch_models.py`（或 Windows venv 解释器）下载全部 4 组权重，
把 `models/` 目录整体拷贝到目标机器；`--check` 按文件大小校验，已存在不会重复下载。

---

## 6. 运行记录 `.runtime/server.json`

启动成功（健康检查通过）后写入，`stop` 删除。格式：

```json
{
  "pid": 12345,
  "host": "127.0.0.1",
  "port": 8795,
  "url": "http://127.0.0.1:8795",
  "started_at": "2026-10-19T14:42:33.1234567+08:00",
  "engine": {
    "backend": "realesrgan",
    "device": "cuda",
    "gpu": "NVIDIA GeForce RTX 4060 Laptop GPU",
    "model": "RealESRGAN_x4plus",
    "native_scale": 4,
    "models_available": ["auto", "realesrgan-x4plus", "realesrgan-x2plus", "realesrgan-anime6b", "realcugan", "realcugan-conservative"]
  },
  "python": "C:\\Users\\...\\pic_polish\\.venv\\Scripts\\python.exe",
  "log": { "stdout": "logs/server.log", "stderr": "logs/server.err.log" }
}
```

| 字段 | 说明 |
|------|------|
| `pid` | 启动脚本 `Start-Process`/`nohup` 得到的进程 PID（Windows 上可能是 venv 解释器转发器，子进程才是真正监听者，`taskkill /T` 会整树终止） |
| `host` / `port` | 实际绑定地址（端口冲突顺延后的**真实**端口） |
| `url` | 浏览器可访问地址（`host=0.0.0.0` 时用 `127.0.0.1`） |
| `started_at` | 启动时间（ISO 8601，带时区） |
| `engine` | `/api/health` 返回的引擎信息（含 `models_available`）；健康检查失败时可能为 `null` |
| `log` | 日志文件相对路径 |

**单实例语义与安全校验**：

- `start`：若 runtime 文件存在、PID 存活、且进程命令行同时包含**本项目路径**与 `uvicorn` →
  幂等返回现有实例；PID 被复用/进程已退出 → 视为陈旧记录，清理后继续启动。
- `stop`：先按 runtime 的 PID 校验后 `taskkill /PID <pid> /T /F`（Windows）/ `TERM→KILL`（Linux）；
  若 runtime 缺失或校验失败，则**兜底扫描监听端口**，沿进程祖先链校验"本项目路径 + uvicorn"
  后才终止，避免误杀其他 Python 程序。
- runtime 文件丢失但服务仍在运行时，`stop` 仍可通过端口兜底找到并停止本项目实例。

> 请勿手工伪造 `.runtime/server.json` 指向无关进程；即使伪造，命令行校验也会拒绝终止。

---

## 7. 日志

| 文件 | 内容 |
|------|------|
| `logs/server.log` | uvicorn **访问日志**（stdout：每个 HTTP 请求一行） |
| `logs/server.err.log` | uvicorn **启动信息 / 未捕获异常 / 引擎告警**（stderr） |
| `logs/models_download.log` | 首次启动自动下载模型权重的进度（含镜像重试） |
| `logs/models_download.err.log` | 模型下载的子进程错误输出 |

- 每次 `start` 会重新创建 `server.log` / `server.err.log`；模型下载日志每次下载时重建。
- 排查顺序：`server.err.log` → `server.log` → `models_download.log` → `.runtime/server.json` → GPU 驱动。
- 启动失败时 `start` 会自动打印两个 server 日志的尾部辅助定位。

---

## 8. 端口、网络与防火墙

- 默认 `127.0.0.1:8765`，**仅本机可访问**。端口优先级：
  `-Port` / `--port` > 环境变量 `PIC_POLISH_PORT` > 默认 `8765`。
- 端口被占用时自动从指定端口向后顺延（最多 100 个），实际地址以输出/`status`/runtime 文件为准。
- 自动化脚本请使用 `-NoBrowser` / `--no-browser`，并从 runtime 文件读取实际 `url`。

**局域网暴露（谨慎）**：

```powershell
# Windows（本工具无认证，请仅在可信网络使用）
pwsh -File scripts\server.ps1 -Action start -Host 0.0.0.0 -NoBrowser
New-NetFirewallRule -DisplayName "PicPolish 8765" -Direction Inbound -Protocol TCP -LocalPort 8765 -Action Allow
```

```bash
# Linux / WSL
bash start.sh --host 0.0.0.0 --no-browser
sudo ufw allow 8765/tcp          # 如启用 ufw
```

WSL2 说明：

- WSL2 默认通过 `localhost` 转发，Windows 侧访问 `http://127.0.0.1:端口` 通常可直接工作；
- 若使用 mirrored networking 或需要局域网访问，请按 WSL 官方文档配置 `netsh interface portproxy`
  或 `.wslconfig` 的 `networkingMode=mirrored`，并在 Windows 防火墙放行；
- WSL 实例关闭（`wsl --shutdown`）后，Linux 侧用 `nohup` 启动的服务会随发行版停止而消失；
  需要常驻请保持 WSL 运行，或改用 Windows 侧脚本。

---

## 9. 引擎、模型与性能

### 9.1 引擎后端

| 后端 | 触发条件 | 效果 |
|------|----------|------|
| `realesrgan` | 有 torch + 权重；`PIC_POLISH_ENGINE=auto`/`realesrgan` | AI 超分（RRDB 或 CUGAN），GPU 优先，CPU 也可（慢） |
| `pil` | 无 torch / 无权重 / 强制 `PIC_POLISH_ENGINE=pil` | Lanczos + 温和 UnsharpMask，速度快、无显存压力 |

- 强制引擎：启动前设置 `PIC_POLISH_ENGINE=auto|realesrgan|pil`（Windows `.ps1` 与 `.sh` 均生效，子进程继承）。
- 显式 `realesrgan` 但环境不满足时，`/api/health` 会体现实际后端；服务不会崩溃。
- 分块推理：`PIC_POLISH_TILE`（默认 256）；CUDA OOM 时引擎会自动把 tile 减半重试（最小 64）。
  大图 4K 建议 tile ≤ 256。

### 9.2 模型矩阵

| 模型 id | 名称 | 原生倍率 | 大小 | 降噪 | 许可 |
|---------|------|----------|------|------|------|
| `auto` | 智能推荐 | 2/4 | ≈127.9 MB | OpenCV | BSD-3-Clause |
| `realesrgan-x4plus` | Real-ESRGAN 通用照片 | 4 | 63.9 MB | OpenCV | BSD-3-Clause |
| `realesrgan-x2plus` | Real-ESRGAN 低倍率 | 2 | 64.0 MB | OpenCV | BSD-3-Clause |
| `realesrgan-anime6b` | Real-ESRGAN 动漫 | 4 | 17.1 MB | OpenCV | BSD-3-Clause |
| `realcugan` | Real-CUGAN（动漫/视频） | 2/3/4 | 55.4 MB | **原生 0..3** | MIT |
| `realcugan-conservative` | Real-CUGAN 保守 | 2/3/4 | 15.2 MB | 固定 | MIT |

- 推荐：照片/风景/人像 → `auto` / `realesrgan-x4plus`；≤2 倍 → `realesrgan-x2plus`；
  动漫/插画 → `realesrgan-anime6b` / `realcugan`；噪点明显 → `realcugan`（降噪 1–3）；
  不想风格化 → `realcugan-conservative`。
- 降噪语义见 [API.md](API.md) §14.3：rrdb 类用 OpenCV 预处理（`denoise=true`）；
  Real-CUGAN 用原生降噪权重（`denoise_level` 0..3）；conservative 固定策略。
- 模型选择逻辑由 `app/models.py` 注册表 + `app/engine.py::resolve_model` 实现；`GET /api/models`
  与 `/api/health.models.available` 反映本地权重是否就绪（不加载模型）。

### 9.3 性能参考

- Windows RTX 4060 Laptop 8GB：1280×720 → 1920×1080（auto/x2plus）约 **2–3 秒**；
  契约目标 ≤ 45s/张（其他模型 ≤ 90s），1080P→4K ≤ 180s。
- WSL2 GPU 直通（同卡）：本机实测 torch 2.5.1+cu121、`realesrgan/cuda` 正常；
  例如 640×360 → 2560×1440 约 **2.2 秒**、显存约 2.1 GB（Lead 实测）。
- 显存不足：减小 `PIC_POLISH_TILE`（128/64）或改用较小档位/2x 模型；必要时 `PIC_POLISH_ENGINE=pil`。
- CPU 版 torch 有 AI 效果但速度显著变慢；建议大图改用 pil 或 GPU。

---

## 10. 模型权重管理（下载 / 更换 / 新增）

- 目录结构：

  ```
  models/
    RealESRGAN_x4plus.pth               # auto / realesrgan-x4plus
    RealESRGAN_x2plus.pth               # auto / realesrgan-x2plus
    RealESRGAN_x4plus_anime_6B.pth      # realesrgan-anime6b
    realcugan/up2x-latest-*.pth         # realcugan / realcugan-conservative（11 个文件）
    realcugan/up3x-latest-*.pth
    realcugan/up4x-latest-*.pth
  ```

- 下载/校验（幂等）：

  ```bash
  python scripts/fetch_models.py --check            # 只检查，缺失退出码 1
  python scripts/fetch_models.py                    # 下载缺失项
  python scripts/fetch_models.py --only realcugan   # 只处理某一组
  python scripts/fetch_models.py --force            # 强制重新下载
  python scripts/fetch_models.py --models-dir DIR   # 指定权重目录
  ```

  下载源：GitHub release 主源 → `ghfast.top` 代理 → `hf-mirror.com` 兜底；下载原子化
  （`.part` 校验大小后改名），失败不会破坏已有文件。首次启动的自动下载会走同一脚本。

- **更换权重**：停止服务 → 用同名、同结构文件覆盖 → 重启。Real-CUGAN 文件必须放
  `models/realcugan/` 且文件名符合注册表（按 2x/3x/4x 与降噪档位区分）。
- **新增模型**：需要修改 `app/models.py`（注册表）与 `app/engine.py`（加载/推理映射），
  属开发改动；运维侧只需保证权重文件与注册表一致。
- **权重不随代码分发**：`models/*.pth`、`*.zip` 被 `.gitignore` 忽略；停止脚本**只清 `cache/`**，
  绝不删除 `models/`。许可信息见 [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)。

---

## 11. 缓存与磁盘治理

```
cache/
  uploads/<job_id>/<index>_<safe_name>     # 上传原图
  results/<job_id>/<stem>_<tier>.<ext>     # 超分结果
  tmp/<job_id>/                            # 中间文件
```

- **stop 必定清空**三个子目录下所有文件（保留目录），并打印删除文件数与释放字节数；
  服务未运行也执行清理并返回 0（幂等）。
- **服务正常退出**：`PIC_POLISH_CLEAN_ON_EXIT=1`（默认）时清理缓存图片。
- **TTL 清理线程**：每小时删除超过 `PIC_POLISH_KEEP_HOURS`（默认 24h）的任务及目录。
- 手工清理（先 stop 更安全）：

  ```powershell
  Remove-Item -Recurse -Force cache\uploads\*, cache\results\*, cache\tmp\* -ErrorAction SilentlyContinue
  ```

  ```bash
  find cache/uploads cache/results cache/tmp -mindepth 1 -delete
  ```

- 磁盘构成：`.venv-linux`（含 torch 全家桶约 5–6 GB）、`models/`（约 0.2 GB）、`cache/`（随任务增长，
  stop 即清空）。WSL 与 Windows 共用同一份项目目录，但各自有独立 venv。

---

## 12. 升级与更新

1. `bash stop.sh` / `stop.bat`（同时清缓存）；
2. 更新项目文件（代码 / `web/` / `requirements*.txt` / 模型注册表）；
3. 依赖或环境有变化时重新执行 `scripts\setup_env.ps1`（Windows）或 `bash scripts/setup_env.sh`
   （Linux/WSL）——两者幂等，会补装缺失依赖、必要时补装/更新 torch；
4. `python scripts/fetch_models.py --check`；缺权重时首次启动会自动下载；
5. 重新 `start.bat` / `bash start.sh`，确认 `/api/health` 的 `engine` 与 `models.available` 正常。

---

## 13. 卸载 / 彻底清理

```powershell
# Windows
pwsh -File scripts\server.ps1 -Action stop
Remove-Item -Recurse -Force cache, logs, .runtime -ErrorAction SilentlyContinue   # 运行产物
Remove-Item -Recurse -Force .venv, models -ErrorAction SilentlyContinue           # venv + 权重
# 彻底移除：删除整个项目目录
```

```bash
# Linux / WSL
bash stop.sh
rm -rf .venv-linux cache logs .runtime
rm -rf models            # 如需删除权重
# 彻底移除：rm -rf /path/to/pic_polish
```

本工具**不写注册表、不装 Windows 服务/计划任务、不创建系统用户**；删除项目目录即完全卸载。
WSL 中删除 `/mnt/c/...` 下目录时 Windows 侧同样生效（同一份文件）。

---

## 14. 故障排查

| 症状 | 可能原因 | 处理 |
|------|----------|------|
| 首次启动停在“自动准备运行环境”/“自动下载模型权重” | 无网络、镜像慢、wheelhouse 缺失 | 看窗口输出与 `logs/models_download.log`；可设 `PIC_POLISH_MODEL_DOWNLOAD_TIMEOUT`；或离线准备 venv+models 后再启动 |
| 自动安装 torch 失败 | 网络/镜像不可达 | 服务仍会以 pil 启动；稍后重试 `setup_env.ps1`，或设置 `PIC_POLISH_TORCH_INDEX` / 用 wheelhouse |
| WSL 中 `torch.cuda.is_available()=False` | Windows NVIDIA 驱动过旧 / 未支持 WSL；装了 CPU 版 torch | 在 Windows 更新 NVIDIA 驱动，`nvidia-smi` 应在 WSL 内可见；重跑 `setup_env.sh`（会优先 cu121 wheelhouse） |
| WSL 离线解包后 `import torch` 失败（缺 libcudart 等） | wheelhouse 缺少 nvidia 运行时小包 | 联网执行 `pip install --no-deps nvidia-cuda-runtime-cu12==12.1.105 nvidia-nvtx-cu12==12.1.105`，或把对应 wheel 放入 `/tmp/nvwheels` 后重跑 setup_env.sh |
| `start` 提示 `.venv`/`.venv-linux` 不存在且自动准备失败 | 无 Python / 依赖安装失败 / 权限 | 手动跑 setup 脚本看具体错误；确认 Python 3.10–3.12 与网络 |
| `start` 提示 90 秒内未通过健康检查 | 应用导入失败、端口竞争、CUDA/模型初始化异常 | 看 `logs/server.err.log` 尾部（start 已自动打印）；修好后重试 |
| 启动打印的端口与传入不同 | 目标端口被占用，已自动顺延 | 属预期行为；以打印的 URL / runtime 文件为准 |
| 请求某模型返回 400 “权重缺失” | 该组权重未下载 | 运行 `python scripts/fetch_models.py --only <组>` 或完整 fetch；重启无需，接口实时判断可用性 |
| 页面模型卡片置灰 | 权重缺失 / 环境不满足 | 下载权重；`/api/models` 的 `available=false` 即此原因 |
| `stop` 说“服务未在运行”但端口仍被占用 | runtime 丢失 / 占用者是别的程序 | `stop` 已做端口兜底（仅终止命令行含本项目+uvicorn 的进程）；其他程序需自行处理 |
| `stop` 提示 “PID 已不是本项目的 uvicorn 进程，不会误杀” | PID 被系统复用 / 进程已退出 | 正常保护行为；可再执行一次 stop 让端口兜底接管 |
| 页面打不开 / 浏览器未自动打开 | 端口不是 8765 / 浏览器互操作不可用 | 用 `status`/runtime 确认 URL；WSL 手动在 Windows 浏览器打开；无头加 `--no-browser` |
| CUDA out of memory | 大图 + 4K 输出显存不足 | 减小 `PIC_POLISH_TILE`（128/64），改用 2x/较小档位，或 `PIC_POLISH_ENGINE=pil` |
| 结果很“糊”、明显不是 AI 超分 | 引擎降级 pil（无 torch/权重） | 检查 `/api/health` 的 `engine.backend` 与 `models_available`；补齐后重试 |
| 上传报 413 / 415 / 400 | 超过体积/数量/像素限制或格式不支持 | 调整 `PIC_POLISH_MAX_FILE_MB` / `MAX_TOTAL_MB` / `MAX_FILES`，或换图（jpg/jpeg/png/webp/bmp/tif/tiff） |
| 中文输出乱码（重定向场景） | 终端代码页非 UTF-8 | 脚本输出 UTF-8；读取文件时按 UTF-8；`chcp 65001` |
| WSL 里启动后 `wsl --shutdown` 服务就没了 | WSL 发行版生命周期 | 需要常驻用 Windows 侧脚本启动 |
| 双击 `.bat` 闪退 | 缺少 PowerShell 7 | 按提示安装 <https://aka.ms/powershell>；窗口有 pause 可阅读 |

### 快速自检命令

Windows：

```powershell
pwsh -File scripts\server.ps1 -Action status
pwsh -File scripts\setup_env.ps1
curl -s http://127.0.0.1:8765/api/health
curl -s http://127.0.0.1:8765/api/models
& .venv\Scripts\python.exe scripts\fetch_models.py --check
pwsh -File tests\e2e\e2e_windows.ps1
```

Linux / WSL：

```bash
bash scripts/server.sh status
bash scripts/setup_env.sh
curl -s http://127.0.0.1:8765/api/health
curl -s http://127.0.0.1:8765/api/models
.venv-linux/bin/python scripts/fetch_models.py --check
.venv-linux/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
bash tests/e2e/e2e_wsl.sh
```

如果 `/api/health` 返回 `"status": "ok"`、`engine.backend`/`engine.device` 符合预期且
`models.available` 含目标模型，说明服务已正常就绪。

---

## 15. 目录与文件权限约定

- 脚本与代码目录只读运行；运行时只写 `cache/`、`logs/`、`.runtime/`；`models/` 只由
  `fetch_models.py` 写入（运维可手动替换）。
- Windows 上 `.bat` 必须保持 **ASCII 编码**（`start.bat` / `stop.bat`），中文一律放 `.ps1`。
- 所有脚本统一 UTF-8（无 BOM）；`.sh` 使用 LF 换行；PowerShell 脚本以
  `$ErrorActionPreference = 'Stop'` 运行，Shell 脚本以 `set -euo pipefail` 运行。
- 不要手工修改 `.runtime/server.json` 指向其他进程（校验会拒绝执行，但可能造成困惑）。

---

## 16. 许可（分发注意）

| 组件 | 许可 |
|------|------|
| 本项目代码 | MIT（见 [LICENSE](../LICENSE)） |
| Real-ESRGAN 代码/权重 | BSD-3-Clause（Copyright (c) 2021, Xintao Wang） |
| BasicSR（RRDBNet 结构来源） | Apache-2.0 |
| Real-CUGAN 代码/权重 | MIT（Copyright (c) 2022 bilibili） |

- **模型权重不随代码/分发包分发**：首次启动由 `scripts/fetch_models.py` 从上游官方发布页下载
  （或手动放置）。再分发/商用前请阅读 [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)
  与各上游 LICENSE，并遵守其署名要求。
- 若随包分发 wheelhouse（仅含 PyTorch 等第三方二进制 wheel），同样需要遵守 PyTorch 及其
  NVIDIA 依赖的许可条款；建议以"联网下载"或"用户自备 wheelhouse"方式分发。
