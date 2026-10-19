# PicPolish 工程契约（模块接口 / 环境 / 分工）

> 本文件 + `docs/API.md` 是全体成员的单一事实来源。动手前必读，冲突时以本文件为准；
> 需要改契约先报告 Lead。

## 0. 本项目是什么

图像超分（super-resolution）Web 工具：上传 1..N 张图片，选一个目标档位（720P/1080P/2K/4K），
保持宽高比自动算另一条边（允许 8px 对齐的细小差异），用 Real-ESRGAN（PyTorch + CUDA）
平滑且真实地提升清晰度，网页展示原图/结果像素尺寸并支持"另存为"/ZIP 下载；同时提供 REST API。
Windows 用 `.bat` + PowerShell 7 启动/终止，Linux/WSL 用 `.sh` 启动/终止；终止时自动清空缓存图片。

## 1. 运行环境事实（已实测，勿重复探测）

- 工作目录：`C:\Users\Deaddream\Desktop\pic_polish`（WSL 内：`/mnt/c/Users/Deaddream/Desktop/pic_polish`）
- **Python 解释器统一用项目 venv**：`.venv\Scripts\python.exe`（Windows，Python 3.12.13）/
  `.venv-linux/bin/python`（WSL，Python 3.12.3，virtualenv）
  - Windows venv 由 `E:\ProgramData\anaconda3new\envs\gnn\python.exe` 以 `--system-site-packages` 创建，
    因此可直接 `import torch`（**torch 2.5.1+cu121，CUDA 可用，RTX 4060 Laptop 8GB**）、
    `torchvision 0.20.1`、`PIL 12.2`、`numpy 2.4.4`；venv 内已装 `fastapi 0.146.0`、`uvicorn 0.54.0`、
    `python-multipart`、`pytest`、`httpx` 与 **`opencv-python-headless 5.0.0`**（denoise 可用）。
  - WSL `.venv-linux` 已安装 **torch 2.5.1+cu121 + torchvision 0.20.1+cu121（CUDA 可用，GPU 直通）**、
    fastapi/uvicorn/python-multipart/opencv 等；本地 wheelhouse `/home/deaddream/picpolish_wheels`
    可离线复用（torch 全家桶来自 pip 缓存，无需下载）。
  - **不要 conda activate、不要管理员权限**；除 scripts-dev 的 setup 脚本外，其他成员不要 pip install。
- 模型权重（Lead 已下载，**勿动**）：
  - `models/RealESRGAN_x4plus.pth`（4x, 63.9MB）、`models/RealESRGAN_x2plus.pth`（2x, 64.0MB）、
    `models/RealESRGAN_x4plus_anime_6B.pth`（动漫 4x, 17.1MB）
  - `models/realcugan/up{2,3,4}x-latest-*.pth`（Real-CUGAN v3，11 个文件共 55.4MB）
  - RRDBNet 权重加载：`torch.load(path, map_location='cpu')` 取 `params_ema`（无则 `params`），
    strict 加载；anime6B 用 `num_block=6`。CUGAN 权威参考：`.probe2/upcunet_v3.py`（已实测可 strict 加载）；
    许可文本在 `.probe2/licenses/`。
- Windows shell：pwsh 7.6.7。`.bat` 里 **不要写中文**（cmd 代码页问题），中文提示放 `.ps1`。
- WSL：`Ubuntu-24.04`（不是 `ubuntu`）。从 PowerShell 调用 helper：
  `$wsl = "$env:USERPROFILE\.dsh\tools\wsl.ps1"; & $wsl run -Cmd 'bash 命令'`；
  `& $wsl run -Argv @('python3','-c','print(1)')` 传字面 argv。
  WSL 内 python3 3.12.3 + pip 24，有网络；**没有 node/docker**。
- 端口：默认 `8765`（`PIC_POLISH_PORT` 可覆盖），启动脚本冲突时自动顺延。

## 2. 目录结构与写权限（严格按此分工，勿越界写别人的文件）

```
pic_polish/
├── app/                     # [api-dev] main.py jobs.py；[engine-dev] engine.py rrdbnet.py target.py
│   ├── config.py            # [Lead 已建] 配置常量，勿改
│   └── __init__.py          # [Lead 已建]
├── web/                     # [frontend-dev] index.html app.js style.css
├── scripts/                 # [scripts-dev] server.ps1 server.sh setup_env.ps1 setup_env.sh；[engine-dev] fetch_models.py
├── models/                  # [Lead] 权重
├── cache/                   # 运行时缓存（任何代码不得提交/手写固定文件）
├── logs/ .runtime/          # 运行时产物，由脚本管理
├── tests/unit/              # [api-dev] test_api.py；[engine-dev] test_target.py test_engine.py
├── tests/e2e/               # [qa-tester]
├── samples/                 # [qa-tester] 测试图片
├── docs/API.md CONTRACT.md  # [Lead]
├── docs/RESEARCH.md         # [research subagent]
├── README.md                # [scripts-dev]
├── start.bat stop.bat start.sh stop.sh          # [scripts-dev]
├── requirements.txt requirements-linux.txt .gitignore   # [Lead]
└── .venv/                   # [Lead] 勿动
```

**冲突规则**：只写自己 scope 内的文件；需要别人文件改动 → 发消息给 Lead 协调。
编辑前先 `read`；不要 `git` 操作；不要删除别人未完成文件。

## 3. 模块接口契约（必须一致，api-dev 与 engine-dev 各自按此实现）

### 3.1 `app/config.py`（Lead 已完成，只读）

导出常量：`ROOT, MODELS_DIR, CACHE_DIR, UPLOAD_DIR, RESULT_DIR, TMP_DIR, WEB_DIR, LOG_DIR,
RUNTIME_DIR, APP_NAME, VERSION, HOST, PORT, ENGINE, TILE, MAX_FILES, MAX_FILE_MB,
MAX_TOTAL_MB, MAX_PIXELS, DEFAULT_ALIGN, JPEG_QUALITY, KEEP_HOURS`，以及函数 `ensure_dirs()`。

### 3.2 `app/target.py`（engine-dev）

```python
TIERS: dict[str, dict]           # id -> {id,label,rule,value,hint}
class InvalidTarget(ValueError): ...
def get_tier(tier: str) -> dict  # 非法抛 InvalidTarget
def plan_target(width: int, height: int, tier: str, align: int = 8) -> dict
# 返回 {"tier","label","rule","value","origin":{"width","height"},
#       "output":{"width","height"},"scale":float,"mode":"upscale|downscale|same",
#       "aligned":bool,"note":str}（不含 index/name，由调用方补）
def plan_many(images: list[dict], tier: str, align: int = 8) -> list[dict]
# images: [{"name","width","height"}] -> 每项加 "index","name"
```

算法严格按 `docs/API.md` 第 1.1 节；输出宽高最小 `align`，非法尺寸抛 `InvalidTarget`。

### 3.3 `app/rrdbnet.py`（engine-dev）

暴露 `RRDBNet`（basicsr/Real-ESRGAN 原版结构，num_feat=64, num_block=23, num_grow_ch=32，
scale=4 或 2；x2 模型 forward 先 pixel_unshuffle(x,2) 再两次 2x 上采样 = 总 2x）以及
`load_rrdbnet(path, scale, device) -> RRDBNet`（eval + no_grad 友好）。
**只依赖 torch，不依赖 basicsr/realesrgan 包。**

### 3.4 `app/engine.py`（engine-dev）

```python
@dataclass(frozen=True)
class EngineInfo:
    backend: str      # "realesrgan" | "pil"
    device: str       # "cuda" | "cpu"
    gpu: str | None
    model: str | None # 主模型显示名（如 "RealESRGAN_x4plus"）
    native_scale: int # 4 / 2 / 1
    models_available: tuple[str, ...] = ()   # 已就绪模型 id 列表

class EngineError(RuntimeError): ...
class BaseEngine:
    info: EngineInfo
    def upscale(self, img: PIL.Image.Image, out_w: int, out_h: int, *,
                model: str = "auto", denoise: bool = False, denoise_level: int = 0,
                tile: int = 0,
                progress_cb: Callable[[float], None] | None = None) -> PIL.Image.Image
    def resolve_model(self, model: str, ratio: float, denoise_level: int = 0) -> dict
    # -> {"id","label","used","family","native_scale","weight_file","denoise_level"}
    def close(self) -> None

def get_engine(force: str | None = None) -> BaseEngine   # 单例缓存；force: auto|realesrgan|pil
def engine_info() -> EngineInfo
def list_models() -> dict          # 转发 app.models.list_models()，供 /api/models 与 /api/health
```

实现要求：
- `upscale` **保证返回 PIL Image，尺寸精确等于 `(out_w, out_h)`**；输入 RGB/RGBA 均可，
  RGBA 时 alpha 用 Lanczos 单独缩放后合并。返回模式 RGB（原图无 alpha）或 RGBA。
- 模型选择（`model` 参数，见 API.md §14）：
  - `auto`：ratio ≤ 2 且 x2plus 存在 → x2plus，否则 x4plus；
  - `realesrgan-x4plus` / `-x2plus` / `-anime6b`：RRDBNet（anime6b 用 num_block=6）；
  - `realcugan` / `realcugan-conservative`：按 ratio 选 CUGAN native scale（≤2→2，≤3→3，否则 4）；
    `denoise_level` 映射：2x → no-denoise/denoise1x/2x/3x；3x/4x → no-denoise 或 denoise3x；
    conservative 固定 `*-conservative.pth`；CUGAN 输出 uint8 [0,255] 直接转 PIL；
  - 请求模型权重缺失：`resolve_model` 抛 `EngineError`（jobs 映射为 400）。`auto` 降级为可用模型。
- 目标 >= 原尺寸：走 SR，输出用 `Image.LANCZOS` 精确缩放到目标（SR 后可温和 unsharp）。
  目标 < 原尺寸或相等：不跑 SR，直接 LANCZOS 精确缩放。
- 推理必须 `torch.inference_mode()` + tiling（默认 tile 256，rrdb pad 10，CUGAN pad ≥32）；
  CUDA OOM 自动 tile 减半重试（最小 64）；结束 `torch.cuda.empty_cache()`。
- `denoise=True`：rrdb 模型用 `cv2.fastNlMeansDenoisingColored` 预处理（本机 cv2 5.0.0 已装，
  `denoise_available=True`）；CUGAN 走原生降噪权重（`denoise_level`）；无 cv2 时忽略并让 jobs 写 note。
- `progress_cb(0..1)` 分块循环中回调；取消异常原样向上传播（jobs 用 `RuntimeError`/CancelToken）。
- 无 torch/CUDA/权重时自动降级 `pil` 后端（永不抛异常）；`EngineError` 仅在显式 force / 请求缺失模型时抛。
- `pil` 后端的 `upscale` 用 `Image.LANCZOS` + `UnsharpMask(radius=1.2, percent=45, threshold=3)`
  （downscale 不锐化），支持 progress_cb 与 alpha。

### 3.5 `app/jobs.py`（api-dev）

```python
class CancelToken:
    def cancel(self) -> None: ...
    @property
    def cancelled(self) -> bool: ...

class JobManager:
    def submit(self, files: list[tuple[str, bytes]], target: str,
               options: dict) -> dict            # 校验+落盘+入队，返回 API.md 第6节 job dict
    def get(self, job_id: str) -> dict | None      # 无则 None
    def list_jobs(self) -> list[dict]              # 概要（可选）
    def result_path(self, job_id: str, index: int) -> Path | None
    def cancel(self, job_id: str) -> dict          # 状态冲突抛 JobConflictError
    def delete(self, job_id: str) -> dict          # 含 freed_bytes
    def shutdown(self) -> None
manager: JobManager                               # 模块级单例
```

- 单 worker 线程 + FIFO 队列（GPU 串行）；`engine = get_engine()` 懒加载；每次处理前把
  `file.status` 改 `running`、`progress` 回写（线程安全用 `threading.Lock`）。
- 每张图：`engine.upscale(PIL.open(bytes), plan["output"])` → 按 `format/quality` 保存到
  `cache/results/<job_id>/<stem>_<tier>.<ext>` → 更新 `output_size/output_format`。
- 取消：`CancelToken` 传给 worker，每文件开始前检查；运行中取消 → 当前文件标 `cancelled`，
  任务标 `cancelled`。`preview/download` 只在文件 `done` 时可用（否则 409）。
- TTL 清理线程：每小时扫一次，删除 `created_at` 超过 `KEEP_HOURS` 的任务及目录。
- 任务持久化到内存即可；重启后 `.runtime` 空。
- 文件名 sanitize：只保留安全字符，输出名 `{stem}_{tier}.{ext}`（防路径穿越）。

### 3.6 `app/main.py`（api-dev）

FastAPI app 单例 `app`，路由严格按 `docs/API.md`；静态：`/web` 挂 `web/`，`/` 返回
`FileResponse(WEB_DIR/"index.html")`。`startup` 调 `config.ensure_dirs()`、清理过期 tmp；
`shutdown` 调 `manager.shutdown()` 并按 `PIC_POLISH_CLEAN_ON_EXIT`（默认 1）清空 cache 图片。
CORS 不需要（同源）。**uvicorn 启动命令固定为 `.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port <port>`**。

### 3.7 `app/models.py`（engine-dev）

```python
MODEL_REGISTRY: dict[str, dict]   # id -> {id,label,description,tags,recommended_for,family,
                                  #        native_scales,weights,denoise,license,size_mb,source}
class InvalidModelError(ValueError): ...
def get_model(model_id: str) -> dict              # 非法抛 InvalidModelError
def list_models() -> dict                          # API.md §14.1 结构，顺序固定，不加载模型
def resolve_weights(model_id: str, ratio: float, denoise_level: int = 0) -> dict
# -> {"model_id","label","family","native_scale","weight_paths":[Path],"denoise_level"}
def available_model_ids() -> list[str]
```

`family`：`rrdb`（x4plus/x2plus/anime6b/auto）或 `cugan`（realcugan/conservative）。
`available` 只按权重文件是否存在（+ 合理大小）判断，禁止在 `/api/models` 里加载 torch。

### 3.8 `app/cugan.py`（engine-dev）

从 `.probe2/upcunet_v3.py`（MIT，Copyright (c) 2022 bilibili）**原样 vendor** Real-CUGAN v3 架构：
`SEBlock / UNetConv / UNet1 / UNet1x3 / UNet2 / UpCunet2x / UpCunet3x / UpCunet4x` + `q/dq`
（删除 demo / `__main__` / `RealWaifuUpScaler` 中不需要的部分，保留许可头与出处注释）。
`load_cugan(scale: int, path: Path, device) -> nn.Module`：`torch.load(..., map_location='cpu', weights_only=True)`
→ `load_state_dict(strict=True)` → `eval()` → `to(device)`；调用签名 `model(tensor, 0, 0, 1.0, False)`，
输入 float NCHW [0,1]，输出 **uint8 NCHW [0,255]**（引擎直接转 PIL，不要再次乘 255）。
**禁止依赖 basicsr / realcugan pip 包。**

## 4. 启动/终止脚本契约（scripts-dev）

- `scripts/server.ps1` 参数：`-Action start|stop|status|restart|logs`、`-Port`、`-Host`、`-NoBrowser`、
  `-Foreground`（调试用，不推荐）。root 的 `start.bat`/`stop.bat` 是双击友好的薄包装，
  **`.bat` 内容只允许 ASCII**，内部调用 `pwsh -NoProfile -ExecutionPolicy Bypass -File scripts\server.ps1 ...`；
  `pwsh` 不存在时报错提示（用户要求 PowerShell 7+）。
- start 流程：检查 `.venv\Scripts\python.exe`（缺失→提示先跑 `scripts\setup_env.ps1`）；选空闲端口；
  `Start-Process` 后台启动 uvicorn，stdout/stderr 重定向到 `logs\server.log`/`logs\server.err.log`；
  写 `.runtime\server.json`（格式见 API.md §12）；轮询 `/api/health` 最多 90s；打印 URL/引擎信息；
  默认打开浏览器（`-NoBrowser` 关闭）。重复 start 已运行时直接打印现有 URL。
- stop 流程：读 `.runtime\server.json` → 校验 PID 命令行含本项目路径 & `uvicorn`（防 PID 复用误杀）
  → 杀进程树（`taskkill /PID <pid> /T /F`）→ 兜底按端口找监听者并校验后杀 → **清空
  `cache\uploads`、`cache\results`、`cache\tmp` 内容并统计释放字节** → 删 `.runtime\server.json`
  → 打印摘要。未运行也执行清理并 exit 0。**任何情况下 stop.bat 双击后窗口可读、不闪退。**
- `start.sh`/`stop.sh`（WSL/Linux）：等价逻辑；venv `.venv-linux/bin/python`（缺失→自动运行
  `scripts/setup_env.sh`）；WSL 现装 CUDA torch，auto 默认走 `realesrgan`（GPU 直通），无 CUDA/权重时
  自动降级 `pil`；`nohup ... &` + `.runtime/server.json`；`kill`/`kill -9` 进程树；缺模型时自动运行
  `scripts/fetch_models.py`（最长等 20 分钟，超时给可读提示）；清理缓存同 Windows；打开浏览器可选
  （WSL 用 `cmd.exe /c start`，测试一律 `--no-browser`）。
- `scripts/setup_env.ps1`：幂等；`.venv` 存在时补装 `requirements.txt` 缺失项（含 opencv-python-headless）
  并校验；不存在时按候选（`PIC_POLISH_PYTHON` → gnn → novel → openui → PATH python）建 venv
  （`--system-site-packages` 复用已有 torch），无任何候选 torch 时自动 `pip install torch`
  （有 nvidia-smi→cu121，否则 cpu；`PIC_POLISH_TORCH_INDEX` 可覆盖），失败则提示降级 pil 不阻塞。
- `scripts/setup_env.sh`：Linux venv `.venv-linux`；优先用本地 wheelhouse 离线安装 torch cu121
  （`/home/deaddream/picpolish_wheels` 或 pip 缓存），否则按 `PIC_POLISH_TORCH_INDEX`/官方源下载；
  然后装 `requirements-linux.txt`（含 opencv）；最后校验 `torch.cuda.is_available()`。幂等。
- **首次启动自动完成**（分发版核心要求）：start.bat/start.ps1/start.sh 检测到 venv 缺失→自动跑
  setup_env；检测到权重缺失/不完整→自动跑 fetch_models.py；然后启动。解压后一条命令即可用。
- 所有脚本：`-ErrorAction Stop` / `set -euo pipefail`；中文输出 OK（.ps1/.sh），`.bat` 只 ASCII。

## 5. 引擎/质量验收目标

- 720P（1280x720）→ 1080p：GPU 单张 ≤ 45s（x2plus/auto），其他模型 ≤ 90s；输出 1920x1080；
  质量需明显优于 Lanczos（用 samples 里真实照片目测 + 对比 `pil` 兜底）。
- 1080P → 4K：≤ 180s（x2/CUGAN 3x 优先）；CUGAN 4x 大图允许更慢但不得 OOM。
- 模型矩阵全部可跑：`auto / realesrgan-x4plus / -x2plus / -anime6b / realcugan / realcugan-conservative`，
  输出尺寸 **必须精确等于** `/api/plan` 计算值（QA 逐像素断言），model_used 正确。
- 降噪：rrdb 模型 `denoise=true` 生效（与 false 有可测差异）；realcugan `denoise_level` 0..3 各档
  输出合法且 0 与 3 有可测差异；CUGAN uint8 输出不得发灰/溢出（黑占比 <0.5%，通道均值合理）。
- 内存/显存：8GB 显存内不出 OOM（tiling），大图 4K 输出不崩溃。
- WSL：GPU 直通下 `/api/health` 应为 `realesrgan/cuda`，720p→1080p 尺寸精确并记录耗时。

## 6. 测试要求（每个 dev 自测 + QA 验）

- 统一用 `.venv\Scripts\python.exe` 跑 pytest：`& .venv\Scripts\python.exe -m pytest tests\unit -q`。
- api-dev/engine-dev 的单元测试不得依赖网络/模型权重（权重缺失时测 `pil` 后端与 plan 逻辑；
  有权重时测真实 GPU 推理 1 张小图）。
- 启动服务自测（任一 dev 可用）：`.venv\Scripts\python.exe -m uvicorn app.main:app --port 8799`，
  然后 curl `/api/health`。测试结束务必杀掉自己起的进程。
- QA 负责 `tests/e2e/`（Windows + WSL 全链路），不依赖人工点击。

## 7. 沟通协议

- 先 `team_task_list` → `team_task_get` → 用当前 revision `claim`，完成后 `complete`。
- 遇到契约冲突/阻塞 → `send_message` 给 `lead`（说明：卡点、已尝试、需要什么）；不要擅自改契约。
- 交付前自测通过再 complete；complete 后在消息里给 Lead 一段**简短证据**（命令 + 关键输出）。
- 不要并发跑会互相抢端口的测试；用 8799/8798 等临时端口，结束即杀。
- 不要动 `.venv`、`models`、`cache`、别人 scope 的文件；不要 `pip install` 到 `.venv`（Lead 统一管理）。
