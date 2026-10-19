# PicPolish · AI 图片超分

PicPolish 是一个**本地运行**的图像超分辨率 Web 工具：上传 1..N 张图片，选择一个目标档位
（720P / 1080P / 2K / 4K），自动保持原始宽高比、按 8 像素对齐微调另一条边，使用
**Real-ESRGAN / Real-CUGAN（PyTorch + CUDA）** 平滑而真实地提升清晰度。网页中可查看原图/结果
像素尺寸、预览对比、单张"另存为"或全部打包 ZIP 下载，同时也提供完整的 REST API。

- **一条命令即可用（分发版核心）**：解压后双击 `start.bat`（Windows）或 `bash start.sh`
  （Linux/WSL），首次启动会**自动准备 Python/torch 环境、自动下载模型权重**，随后打开网页。
- **多模型可选**：智能推荐 / Real-ESRGAN 通用照片 / 低倍率 / 动漫 / Real-CUGAN（原生降噪）/
  Real-CUGAN 保守，网页里按场景推荐。
- 单机、离线可用：UI 不依赖任何 CDN / 外网资源，图片不出本机（仅首次下载环境/权重需要网络）。
- Windows 与 WSL/Linux 一键启动 / 停止；**WSL2 支持 NVIDIA GPU 直通**（CUDA 12.1 torch）。
- 没有 GPU / 没有 torch / 权重缺失时自动降级到 `pil` 兼容引擎（Lanczos + 温和锐化），永不崩溃。
- 停止时自动清空缓存图片，不长期占用磁盘。

---

## 功能特性

| 特性 | 说明 |
|------|------|
| 4 档目标分辨率 | `720p`（短边 720）/ `1080p`（短边 1080）/ `2k`（长边 2560）/ `4k`（长边 3840） |
| 保持宽高比 | 只固定一条边，另一条边按比例缩放；非整除时吸附到最近的 8 像素倍数（`align` 可配 2..64） |
| 6 种模型 | `auto` / `realesrgan-x4plus` / `realesrgan-x2plus` / `realesrgan-anime6b` / `realcugan` / `realcugan-conservative` |
| AI 超分 | Real-ESRGAN x4plus / x2plus / anime6B（RRDBNet）与 Real-CUGAN v3（2x/3x/4x 多档降噪权重） |
| 降噪 | Real-ESRGAN 类：OpenCV 预处理降噪（`denoise=true`）；Real-CUGAN：原生降噪等级 0..3 |
| 智能降级 | 无 torch / 无 CUDA / 权重缺失 → 自动 `pil`；请求不可用模型时给出明确提示 |
| 显存友好 | 分块（tiling）推理，CUDA OOM 自动把 tile 减半重试；大图 4K 输出不崩溃 |
| 多图批量 | 一次最多 20 张（可配），单 worker 串行处理，逐文件进度与状态 |
| 网页 UI | 拖拽上传、尺寸预览、模型与降噪选择、总/单文件进度、取消、预览对比、另存为（File System Access API，自动降级 `a[download]`）、ZIP 打包 |
| REST API | `/api/health`、`/api/models`、`/api/targets`、`/api/plan`、`/api/jobs` 等，详见 [docs/API.md](docs/API.md) |
| 首次启动自动配置 | venv 缺失 → 自动 `setup_env`；权重缺失 → 自动 `fetch_models.py`（最长 20 分钟，可配置） |
| 离线分发 | 支持本地 wheelhouse 离线安装 torch（Linux/WSL）/自定义 torch index（Windows） |
| 缓存治理 | 任务默认保留 24h；停止脚本必定清空 `cache/uploads|results|tmp` |
| 跨平台脚本 | Windows：`start.bat` / `stop.bat` + `scripts/server.ps1`；Linux/WSL：`start.sh` / `stop.sh` + `scripts/server.sh` |

---

## 模型列表与推荐

网页「模型」区与 `GET /api/models` 均来自后端模型注册表（权重缺失时 `available=false` 并置灰）：

| 模型 id | 名称 | 原生倍率 | 大小 | 降噪 | 许可 | 推荐场景 |
|---------|------|----------|------|------|------|----------|
| `auto` | 智能推荐 | 2 / 4 | ≈127.9 MB | OpenCV | BSD-3-Clause | 不确定选哪个时（按目标倍率自动选 x2plus/x4plus） |
| `realesrgan-x4plus` | Real-ESRGAN 通用照片 | 4 | 63.9 MB | OpenCV | BSD-3-Clause | 照片 / 风景 / 人像，细节自然 |
| `realesrgan-x2plus` | Real-ESRGAN 低倍率 | 2 | 64.0 MB | OpenCV | BSD-3-Clause | 只需要 ≤2 倍放大（速度快、细节好） |
| `realesrgan-anime6b` | Real-ESRGAN 动漫 | 4 | 17.1 MB | OpenCV | BSD-3-Clause | 动漫截图 / 插画 / 二次元 |
| `realcugan` | Real-CUGAN（动漫/视频） | 2 / 3 / 4 | 55.4 MB | **原生 0..3** | MIT | 动漫/视频截图、噪点明显、需要降噪 |
| `realcugan-conservative` | Real-CUGAN 保守 | 2 / 3 / 4 | 15.2 MB | 固定保守策略 | MIT | 不想过度锐化 / 风格化 |

> 权重文件**不随代码/分发包提供**：首次启动由 `scripts/fetch_models.py` 从上游官方发布页
> （GitHub release / ghfast 镜像 / hf-mirror）下载，也可手动放置到 `models/`。

---

## 目录结构

```
pic_polish/
├── app/                     # 后端（FastAPI + 引擎 + 任务队列 + 模型注册表）
│   ├── main.py              # 路由 / 静态托管 / 生命周期
│   ├── jobs.py              # 任务队列、进度、取消、TTL 清理
│   ├── engine.py            # 超分引擎（realesrgan rrdb + realcugan + pil 降级）
│   ├── models.py            # 模型注册表（id/label/推荐/降噪/许可/可用性）
│   ├── cugan.py             # Real-CUGAN v3 网络结构（MIT，vendored）
│   ├── rrdbnet.py           # RRDBNet 网络结构（无 basicsr 依赖）
│   ├── target.py            # 档位与输出尺寸计算
│   └── config.py            # 配置常量（环境变量入口）
├── web/                     # 纯原生 HTML/CSS/JS 前端（模型卡片 / 降噪控件）
├── scripts/
│   ├── server.ps1           # [Windows] start/stop/status/restart/logs + 首启自动配置
│   ├── server.sh            # [Linux/WSL] 同上
│   ├── setup_env.ps1        # [Windows] 创建/校验/修复 .venv（自动装 torch 兜底）
│   ├── setup_env.sh         # [Linux/WSL] 创建/校验 .venv-linux（wheelhouse 离线装 torch）
│   └── fetch_models.py      # 幂等下载/校验全部模型权重（--check/--only/--force）
├── models/                  # 模型权重（*.pth / realcugan/*.pth，不提交，缺失自动下载）
├── cache/                   # 运行缓存：uploads/ results/ tmp/（stop 时清空）
├── logs/                    # server.log / server.err.log / models_download*.log
├── .runtime/server.json     # 运行信息（pid/host/port/url/started_at/engine/log）
├── tests/unit/ tests/e2e/   # 单元测试 / 端到端测试
├── samples/                 # 测试图片（含噪点、动漫、竖图、4K 等）
├── docs/                    # API.md / CONTRACT.md / DEPLOY.md / RESEARCH.md
├── start.bat stop.bat       # [Windows] 双击友好薄包装（内容仅 ASCII）
├── start.sh  stop.sh        # [Linux/WSL] 薄包装
├── requirements.txt         # Windows venv 依赖（torch 复用系统环境或自动安装）
├── requirements-linux.txt   # Linux/WSL venv 依赖（不含 torch，torch 由 setup_env.sh 安装）
├── THIRD_PARTY_NOTICES.md   # 第三方代码/权重许可声明
└── LICENSE                  # 本项目 MIT 许可
```

---

## 快速开始

### 环境要求

| 平台 | 要求 |
|------|------|
| Windows | Windows 10/11；**PowerShell 7+（`pwsh`）**；可选 Python 3.10–3.12（若已有带 torch 的 conda 环境会自动复用）；NVIDIA GPU + 驱动可选（自动装 cu121 torch） |
| Linux / WSL | Ubuntu 22.04+（本机验证 Ubuntu 24.04 / WSL2）；`python3` 3.10+、`curl`；可选 PyTorch（setup_env.sh 自动安装）；NVIDIA GPU 可选 |
| 网络 | 首次启动下载 torch（仅缺 torch 时，约 2 GB）与模型权重（约 200 MB）；离线分发见下文「离线安装」 |
| 浏览器 | 新版 Edge / Chrome / Firefox（另存为优先 File System Access API，不支持时自动降级） |
| 磁盘 | 建议 ≥ 8 GB 空闲（torch + 依赖约 5–6 GB，模型权重约 0.2 GB） |

> **不需要** `conda activate`、**不需要**管理员权限，也不会在全局安装任何 Python 包。

### Windows：解压后直接双击 `start.bat` 即可（首次启动自动配置）

首次启动会依次自动完成（全程打印在窗口中，无需手工操作）：

1. 检测 `.venv`：缺失 → 自动运行 `scripts/setup_env.ps1`
   （优先复用已有带 torch 的 conda 环境；没有任何 torch 时自动 `pip install torch`：
   有 `nvidia-smi` → CUDA 12.1 版，否则 CPU 版；失败只警告并降级 `pil`）；
2. 补装 `requirements.txt` 中缺失的依赖（含 `opencv-python-headless`）；
3. 检测模型权重：缺失/不完整 → 自动运行 `scripts/fetch_models.py` 下载
   （最长等待 20 分钟，超时会提示手动命令并继续启动，缺失模型自动降级）；
4. 启动 uvicorn 并轮询 `/api/health`（最多 90 秒），成功后自动打开浏览器；
5. 关闭窗口不影响服务；**停止请双击 `stop.bat`**（会同时清空缓存图片）。

> 自动化/无头场景可用参数：`start.bat -NoBrowser -Port 8795`；
> 设置 `PIC_POLISH_NO_PAUSE=1` 可跳过结尾 pause。
> 也可手动分步执行：`pwsh -File scripts\setup_env.ps1`、`python scripts\fetch_models.py`。

### Linux / WSL：`bash start.sh` 即可（首次启动自动配置）

```bash
cd /path/to/pic_polish

# 直接启动：venv 缺失会自动跑 setup_env.sh；权重缺失会自动 fetch_models.py
bash start.sh --no-browser --port 8798        # 本机验证：WSL2 + RTX 4060 走 realesrgan/cuda

# 可选：手动准备 / 修复（幂等，可重复执行）
bash scripts/setup_env.sh
.venv-linux/bin/python scripts/fetch_models.py --check
.venv-linux/bin/python scripts/fetch_models.py           # 下载缺失权重

# 状态 / 日志 / 停止
bash scripts/server.sh status
bash scripts/server.sh logs
bash stop.sh                                   # 停止并清空缓存
```

WSL 的 GPU 由 Windows 驱动直通（`nvidia-smi` 可见），`setup_env.sh` 会优先用本地 wheelhouse
（`$PIC_POLISH_WHEELHOUSE` → `~/picpolish_wheels` → 项目内 `wheelhouse/`）**离线**安装
CUDA 12.1 版 torch；装好后 `/api/health` 显示 `realesrgan/cuda`。

### 离线安装（无外网 / 内网分发）

- **torch**：准备一个 wheelhouse 目录（含 torch/torchvision/全部 nvidia_* 依赖 wheel，
  可从联网机器的 `pip download` 或 `~/.cache/pip` 提取），然后：

  ```bash
  # Linux / WSL
  PIC_POLISH_WHEELHOUSE=/path/to/picpolish_wheels bash scripts/setup_env.sh
  ```

  ```powershell
  # Windows（使用本地/内网 torch 源）
  $env:PIC_POLISH_TORCH_INDEX = 'https://内网镜像/pytorch/whl/cu121'
  pwsh -File scripts\setup_env.ps1
  ```

- **模型权重**：在联网机器执行 `python scripts/fetch_models.py` 后，把整个 `models/` 目录
  拷到目标机器即可（`--check` 会校验文件大小，已存在则不会重复下载）。

---

## API 摘要

完整接口契约、字段与错误码见 **[docs/API.md](docs/API.md)**。

| 方法 | 路径 | 用途 |
|------|------|------|
| GET | `/api/health` | 健康检查 + 引擎 / GPU / 模型可用性 / 限制信息 |
| GET | `/api/models` | 模型列表（推荐用途 / 降噪能力 / 许可 / 是否已下载） |
| GET | `/api/targets` | 档位定义（720p/1080p/2k/4k） |
| POST | `/api/plan` | 只计算输出尺寸（不上传文件） |
| POST | `/api/jobs` | multipart 上传图片并创建超分任务（支持 `model` / `denoise` / `denoise_level`） |
| GET | `/api/jobs/{job_id}` | 任务状态 / 进度 / 每张图结果（含 `model_used`） |
| GET | `/api/jobs/{job_id}/files/{index}` | 结果图（inline 预览） |
| GET | `/api/jobs/{job_id}/files/{index}/download` | 结果图（另存为 / 下载） |
| GET | `/api/jobs/{job_id}/download.zip` | 全部结果打包下载 |
| POST | `/api/jobs/{job_id}/cancel` | 取消任务 |
| DELETE | `/api/jobs/{job_id}` | 删除任务及其缓存文件 |
| GET | `/`、`/web/*` | 网页 UI 与静态资源 |

```bash
# 健康检查（含引擎与已就绪模型）
curl -s http://127.0.0.1:8765/api/health
# 模型列表与推荐
curl -s http://127.0.0.1:8765/api/models
# 只算尺寸（1280x720 → 1080P 应得 1920x1080）
curl -s -X POST http://127.0.0.1:8765/api/plan \
  -H "Content-Type: application/json" \
  -d '{"target":"1080p","images":[{"name":"a.jpg","width":1280,"height":720}]}'
# 创建任务（指定模型 + 降噪等级）
curl -s -X POST http://127.0.0.1:8765/api/jobs \
  -F "files=@samples/photo_1280x720.jpg" -F "target=1080p" \
  -F "model=realcugan" -F "denoise_level=2"
curl -s http://127.0.0.1:8765/api/jobs/<job_id>
curl -sO -J http://127.0.0.1:8765/api/jobs/<job_id>/files/0/download
```

---

## 环境变量

应用/引擎配置由 `app/config.py` 读取（启动脚本启动的子进程会继承这些变量）：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `PIC_POLISH_HOST` | `127.0.0.1` | 监听地址 |
| `PIC_POLISH_PORT` | `8765` | 监听端口（被占用时脚本自动顺延） |
| `PIC_POLISH_ENGINE` | `auto` | 引擎：`auto` / `realesrgan` / `pil` |
| `PIC_POLISH_TILE` | `256` | 推理分块大小（像素），OOM 时引擎还会自动减半 |
| `PIC_POLISH_MAX_FILES` | `20` | 单次任务最多图片数 |
| `PIC_POLISH_MAX_FILE_MB` | `50` | 单张图片体积上限（MB） |
| `PIC_POLISH_MAX_TOTAL_MB` | `200` | 单次任务总体积上限（MB） |
| `PIC_POLISH_MAX_PIXELS` | `20000000` | 单张图片最大像素数（约 20MP） |
| `PIC_POLISH_JPEG_QUALITY` | `95` | 默认 JPEG/WebP 质量（70..100） |
| `PIC_POLISH_KEEP_HOURS` | `24` | 任务缓存保留时长（小时） |
| `PIC_POLISH_CLEAN_ON_EXIT` | `1` | 服务正常退出时是否清空缓存（`0`/`false` 关闭） |
| `PIC_POLISH_ROOT` / `PIC_POLISH_MODELS_DIR` / `PIC_POLISH_CACHE_DIR` / `PIC_POLISH_WEB_DIR` / `PIC_POLISH_LOG_DIR` / `PIC_POLISH_RUNTIME_DIR` | 项目内对应目录 | 路径覆盖（高级用法，一般不需要） |

脚本级变量：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `PIC_POLISH_AUTO_SETUP` | `1` | 启动时 venv 缺失是否自动运行 setup_env（`0` 关闭） |
| `PIC_POLISH_AUTO_DOWNLOAD` | `1` | 启动时权重缺失是否自动下载（`0` 关闭） |
| `PIC_POLISH_MODEL_DOWNLOAD_TIMEOUT` | `1200` | 自动下载模型权重的最长等待秒数（默认 20 分钟） |
| `PIC_POLISH_PYTHON` | 空 | `setup_env.ps1` 首选候选解释器（需能 `import torch` 时优先复用） |
| `PIC_POLISH_TORCH_INDEX` | 按 CUDA 可用性自动选 cu121/cpu | `setup_env.ps1` 安装 torch 的源 |
| `PIC_POLISH_SKIP_TORCH` | 空 | 设为 `1` 时跳过自动安装 torch（强制 `pil`） |
| `PIC_POLISH_WHEELHOUSE` | 空 | Linux/WSL 本地 wheelhouse 目录（离线安装 torch） |
| `PIC_POLISH_TORCH_VERSION` | `2.5.1` | Linux/WSL 在线安装 torch 的版本 |
| `PIC_POLISH_TORCHVISION_VERSION` | `0.20.1` | 同上 torchvision 版本 |
| `PIC_POLISH_PYPI_INDEX` | pip 默认配置 | Linux/WSL 补装小依赖用的 PyPI 镜像（可选） |
| `PIC_POLISH_VENV_PYTHON` | `.venv\Scripts\python.exe` / `.venv-linux/bin/python` | 覆盖服务使用的 Python 解释器（高级用法） |
| `PIC_POLISH_START_TIMEOUT` | `90` | `server.sh` 等待健康检查的秒数 |
| `PIC_POLISH_NO_PAUSE` | 空 | `.bat` 设置后跳过结尾 `pause`（自动化用） |

---

## 常用操作

```powershell
# Windows ---------------------------------------------------------------
# 状态 / 日志 / 重启
pwsh -File scripts\server.ps1 -Action status
pwsh -File scripts\server.ps1 -Action logs -Lines 80
pwsh -File scripts\server.ps1 -Action restart -NoBrowser

# 换端口（端口占用会自动顺延，并打印实际地址）
pwsh -File scripts\server.ps1 -Action start -Port 9000 -NoBrowser

# 强制兼容引擎（无 GPU / 排查问题）
$env:PIC_POLISH_ENGINE = 'pil'; pwsh -File scripts\server.ps1 -Action start -NoBrowser

# 模型权重：检查 / 只下某一组 / 强制重下
& .venv\Scripts\python.exe scripts\fetch_models.py --check
& .venv\Scripts\python.exe scripts\fetch_models.py --only realcugan
& .venv\Scripts\python.exe scripts\fetch_models.py --force

# Linux / WSL -----------------------------------------------------------
bash scripts/server.sh status
bash scripts/server.sh logs --lines 80
bash scripts/server.sh restart --no-browser
PIC_POLISH_ENGINE=pil bash start.sh --port 8798 --no-browser
bash stop.sh
```

---

## 首次启动会发生什么？

| 场景 | 行为 |
|------|------|
| `.venv` 已就绪 | 跳过环境准备（只做秒级校验）；若缺 Web 依赖/openCV 会自动补装 |
| `.venv` 缺失 | 自动运行 `setup_env.ps1` / `setup_env.sh`（可能需要几分钟；缺 torch 会自动安装） |
| 权重齐全 | 跳过下载，直接启动 |
| 权重缺失/不完整 | 自动运行 `fetch_models.py` 下载（默认最长 20 分钟）；失败/超时 → 提示手动命令并继续启动，缺失模型自动降级 |
| 已有一个实例在运行 | `start` 幂等：打印现有 URL，不会重复启动 |
| 想跳过自动行为 | `PIC_POLISH_AUTO_SETUP=0` / `PIC_POLISH_AUTO_DOWNLOAD=0` |

首次启动的日志：`logs/models_download.log`（权重下载进度）、`logs/server.err.log`（服务启动）。

---

## 常见问题（FAQ）

**1. 为什么不需要 `conda activate`，也不需要管理员权限？**
项目 venv 里记录的是绝对路径解释器，脚本直接以 `<项目>\.venv\Scripts\python.exe`（Windows）或
`.venv-linux/bin/python`（Linux）调用，不依赖 conda 的激活状态；服务只监听 `127.0.0.1`、只写项目内目录，
所有操作都在用户权限内完成，无需提权。Windows 上 venv 通过 `--system-site-packages` 复用 conda 环境里
的 PyTorch；没有可复用的 torch 时才自动安装。

**2. 首次启动很慢 / 一直在下载？**
首次需要下载 torch（约 2 GB，仅在没有可用 torch 时）和模型权重（约 200 MB，仅缺失时）。
下载有 20 分钟上限（`PIC_POLISH_MODEL_DOWNLOAD_TIMEOUT` 可调），进度见窗口输出与
`logs/models_download.log`。想跳过：先在有网机器准备好 `.venv` + `models/`，或使用离线 wheelhouse。

**3. 端口被占用怎么办？**
默认 `8765`（可用 `PIC_POLISH_PORT` / `-Port` / `--port` 覆盖）。启动脚本检测到端口被占用时会从
指定端口开始向后自动顺延（最多 100 个），并打印实际使用的 URL；如果占用者正是本项目的旧实例，
`start` 会幂等返回现有实例而不重复启动。

**4. 模型权重从哪来？要手动下载吗？**
共 4 组权重（x4plus / x2plus / anime6B / Real-CUGAN v3 共 11 个文件，合计约 200 MB），
首次启动由 `scripts/fetch_models.py` 自动下载（GitHub release 主源 + ghfast.top 代理 +
hf-mirror 兜底），幂等：已存在且大小正确的文件不会重复下载。也可手动放置到 `models/`
（CUGAN 放 `models/realcugan/`）。权重**不随代码分发**，商用/再分发请阅读
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)：Real-ESRGAN 为 BSD-3-Clause，Real-CUGAN 为 MIT。

**5. WSL / Linux 能用 GPU 吗？**
可以。本机 WSL2 + RTX 4060 已实测：`torch 2.5.1+cu121`、`torch.cuda.is_available()=True`，
`/api/health` 返回 `realesrgan/cuda`。Linux 侧默认不装 torch，`setup_env.sh` 会优先用本地
wheelhouse **离线**安装 cu121 版 torch（约 2.9 GB wheel，解包数秒），没有 wheelhouse 时联网安装。
没有 GPU/装不上 torch 时自动降级 `pil`（Lanczos + 温和锐化），仍可正常使用。

**6. 缓存图片什么时候清理？**
三个时机：① 双击 `stop.bat` / 执行 `bash stop.sh` 时**必定**清空 `cache/uploads`、`cache/results`、
`cache/tmp` 下所有文件（保留目录）；② 服务正常退出时（`PIC_POLISH_CLEAN_ON_EXIT=1`，默认开启）；
③ 后台线程每小时清理超过 `PIC_POLISH_KEEP_HOURS`（默认 24h）的过期任务。缓存目录删除不会影响
`models/` 里的权重。

**7. 如何重新生成 venv？**
Windows：先停服务，删除 `.venv` 目录后重新运行 `scripts\setup_env.ps1`（脚本本身**不会**删除已有
`.venv`，损坏时会提示你手动删除）。Linux/WSL：`rm -rf .venv-linux && bash scripts/setup_env.sh`。
Windows 重新生成会复用 conda 里的 torch；Linux 重新生成会再走一次 wheelhouse/联网安装。

**8. 日志在哪里？**
`logs/server.log`（uvicorn 访问日志，stdout）、`logs/server.err.log`（启动/异常/引擎告警，stderr）、
`logs/models_download.log`（权重下载进度）。每次 `start` 会重新创建 server 日志；
Windows 可用 `pwsh -File scripts\server.ps1 -Action logs`，Linux 用 `bash scripts/server.sh logs`
查看尾部。排查启动失败时优先看 `server.err.log`。

**9. 双击 `.bat` 窗口一闪而过 / 中文乱码？**
`start.bat` / `stop.bat` 结尾有 `pause`，正常情况下窗口会停留；若被自动化环境设置了
`PIC_POLISH_NO_PAUSE=1` 则不会停留。`.bat` 只包含 ASCII 字符（避免 cmd 代码页问题），中文提示由
PowerShell 输出（脚本已强制 UTF-8）。若系统缺少 PowerShell 7，会显示英文提示并等待按键。

**10. 如何在局域网访问？**
默认只监听 `127.0.0.1`（仅本机）。需要局域网访问时用 `-Host 0.0.0.0`（Windows）/ `--host 0.0.0.0`
（Linux）启动，并在系统防火墙放行该端口。注意本工具没有认证，请只在可信网络中使用。WSL2 的端口
转发/镜像网络配置见 [docs/DEPLOY.md](docs/DEPLOY.md)。

**11. 为什么只允许一个实例？**
`.runtime/server.json` 充当单实例锁：启动脚本先校验该文件里的 PID 是否为"本项目的 uvicorn"
（命令行同时包含项目路径与 uvicorn），是则幂等返回现有实例；PID 复用/陈旧记录不会被误杀。
停止脚本同样先校验再杀，并按监听端口做兜底扫描。

**12. 想用 CPU 跑 / 显存不够怎么办？**
`PIC_POLISH_ENGINE=pil` 可强制兼容引擎（最快、无显存）；或用 4x/2K 较小档位、减小
`PIC_POLISH_TILE`（如 128/64）避免 OOM。若 torch 是 CPU 版，AI 超分仍可用但速度慢，建议改用 pil。

更多部署、离线 wheelhouse、防火墙、模型替换、卸载与故障排查见 **[docs/DEPLOY.md](docs/DEPLOY.md)**。

---

## 开发与测试

```powershell
# 单元测试（Windows）
.\.venv\Scripts\python.exe -m pytest tests\unit -q

# 端到端（Windows，端口 8796；会自己 start/stop）
pwsh -File tests\e2e\e2e_windows.ps1

# 端到端（WSL，端口 8798）
bash tests/e2e/e2e_wsl.sh
```

工程结构与模块接口契约见 [docs/CONTRACT.md](docs/CONTRACT.md)。

---

## 许可

- 本项目代码：[MIT](LICENSE)
- Real-ESRGAN 代码/权重：BSD-3-Clause；BasicSR（RRDBNet 结构来源）：Apache-2.0；
  Real-CUGAN 代码/权重：MIT（Copyright (c) 2022 bilibili）
- 模型权重**不随本项目分发**，首次启动从上游官方发布页下载（详见
  [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 与 [docs/API.md](docs/API.md) §14.5）。
