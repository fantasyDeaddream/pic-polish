# PicPolish E2E 测试（tests/e2e）

本目录是 QA 的全链路验收套件（R2：多模型 + 降噪 + WSL GPU）：在 **Windows（GPU /
realesrgan/cuda）** 与 **WSL Ubuntu-24.04（GPU 直通 / CUDA torch）** 上，对真实集成系统做
「启动 → API 全链路 → 多模型矩阵 → 降噪矩阵 → 真实照片质量/性能 → 浏览器 DOM 冒烟 →
停止 → 清理断言」，并输出可复现的报告数据。

> 只允许写 `samples/` 与 `tests/e2e/`。本套件不会修改 `app/`、`web/`、`scripts/`；
> 发现缺陷请报告 Lead + 对应 owner，而不是就地改代码。

---

## 1. 文件一览

| 文件 | 作用 |
|------|------|
| `make_samples.py` | 生成/重建 `samples/` 测试素材（真实照片 + 派生图 + 合成图 + anime/noisy 探针），总量约 9 MB |
| `_common.py` | 公共库：HTTP 客户端、检查结果登记、PIL 统计（black_ratio/Laplacian）、模型期望表、图像差异/噪声度量、Content-Disposition 解析 |
| `e2e_windows.py` | Windows API 全链路功能测试（health/health.models/models API/targets/plan/tile/tiers/jobs/下载/zip/DELETE/取消/静态页） |
| `quality_check.py` | 真实照片「输出 vs 同尺寸 Lanczos 基线」的清晰度/内容有效性/色偏检查 + 720p→1080p GPU 性能 |
| `model_matrix.py` | **R2 模型矩阵**：逐模型 720p→1080p 真实任务 + 逐模型 4K 性能 + realcugan denoise_level 0..3 差异 + anime6b vs x4plus 差异 + rrdb denoise on/off 差异 |
| `e2e_windows.ps1` | Windows 生命周期薄包装：`start` → 功能 → 质量 → 模型矩阵 → 浏览器冒烟 → `stop` → 清理断言 |
| `browser_smoke.ps1` | 探测 Edge/Chrome，headless `--dump-dom` 抓页面并断言关键 DOM（标题/上传/档位/模型卡片/推荐文案/降噪控件/另存为），留截图 |
| `e2e_wsl.sh` | WSL 内全链路：`setup_env.sh`（幂等 + CUDA torch 校验）→ `PIC_POLISH_ENGINE=auto start.sh` → curl health/models/targets/plan → 720p→1080p GPU 任务（计时）→ PIL 断言 → `stop.sh` → 清理断言 |
| `selftest_checks.py` | **离线自检**：验证断言逻辑本身（plan 期望表、模型注册表、黑块/tile/接缝检测、噪声/差异度量、alpha 合成、header 解析）；不需要服务 |
| `artifacts/` | 运行时产物（日志、JSON、下载结果、矩阵产物、质量对比图、浏览器截图）；`artifacts/latest` 每次运行会被清空重建 |
| `REPORT.md` | 最终验收报告（模型矩阵表/降噪矩阵/WSL GPU/性能/遗留问题/复现命令） |

---

## 2. 前置条件

* 项目根目录为当前工作区（脚本会自动定位到 `tests/e2e/../..`）。
* Windows：`.venv\Scripts\python.exe` 存在（`httpx`、`PIL`、`numpy`、`opencv-python-headless`）；
  模型权重齐全：`models\RealESRGAN_x4plus.pth` / `x2plus.pth` / `RealESRGAN_x4plus_anime_6B.pth` /
  `models\realcugan\up{2,3,4}x-latest-*.pth`；验收要求 `realesrgan`+`cuda`。
* WSL：Ubuntu-24.04 内 `curl`、`python3`（带 PIL）、`ss`；`.venv-linux` 需有 **CUDA torch**
  （`torch.cuda.is_available()==True`，WSL GPU 直通）与 `opencv-python-headless`；每次运行都会
  执行 `scripts/setup_env.sh`（已存在 venv 时只校验，同时验证幂等；`PIC_POLISH_E2E_SKIP_SETUP=1` 可跳过）。
* 素材：若 `samples/` 缺失，先运行
  `.venv\Scripts\python.exe tests\e2e\make_samples.py`。

### 素材清单（`samples/MANIFEST.json` 有完整来源）

| 文件 | 尺寸 | 用途 |
|------|------|------|
| `photo_beach_1920x1200.jpg` | 1920×1200 | 真实照片（海滩/海蚀柱） |
| `photo_cliff_1920x1200.jpg` | 1920×1200 | 真实照片（岩壁攀岩，细节丰富） |
| `photo_night_1920x1200.jpg` | 1920×1200 | 真实照片（银河帐篷，暗调） |
| `photo_panorama_3840x1200.jpg` | 3840×1200 | 真实 4K 宽幅全景（downscale 输入） |
| `photo_1280x720.jpg` | 1280×720 | **720p→1080p / 模型矩阵主样本**（真实照片裁切） |
| `portrait_720x1280.jpg` | 720×1280 | 竖图（720p 短边→720×1280 / 1080p→1080×1920） |
| `tiny_320x240.png` | 320×240 | 小图放大 |
| `alpha_800x600.png` | 800×600 RGBA | 透明通道单独缩放 |
| `large_3840x2160.jpg` | 3840×2160 | 4K 大图（1080p 下为 downscale） |
| `odd_1000x667.jpg` | 1000×667 | 8px 对齐吸附（1080p→1616×1080） |
| `text_gradient_1280x720.png` | 1280×720 | 渐变+文字+1px 网格，主观清晰度对比 |
| `solid_gradient_700x700.png` | 700×700 | 近均匀渐变：多 tile 黑块 / 接缝探针（1080p→1080×1080） |
| `anime_1280x720.png` | 1280×720 | **R2** 动漫风格合成帧（大色块+粗线条+渐变）：anime6b / CUGAN 与 x4plus 差异对比 |
| `noisy_gauss_1280x720.jpg` | 1280×720 | **R2** 真实照片 + 高斯噪声 σ=18：rrdb denoise on/off、realcugan denoise_level 差异 |
| `noisy_saltpepper_1280x720.jpg` | 1280×720 | **R2** 真实照片 + 椒盐噪声 2%：降噪鲁棒性样本 |

---

## 3. 运行方式

### 3.1 Windows 全链路验收（推荐，一条命令）

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File tests\e2e\e2e_windows.ps1
```

默认端口 **8796**（QA 专用），自动完成 start → 功能 E2E → 质量/性能 → **模型矩阵/降噪矩阵/4K
性能** → 浏览器冒烟 → 停止 → 清理断言，退出码 `0` 表示全部阻塞项 PASS。产物在
`tests/e2e/artifacts/latest/`（`lifecycle.json`、`functional/`、`quality/`、`matrix/`、
`browser/`、`*.log`）。4K 矩阵较重（6 个模型各跑一张 3840×2160），整体约 3–6 分钟。

常用参数：

```powershell
# 只验功能 + 生命周期，跳过质量/矩阵/浏览器
pwsh -File tests\e2e\e2e_windows.ps1 -SkipQuality -SkipMatrix -SkipBrowser

# 跳过 4K 性能段（矩阵只跑 720p→1080p）
pwsh -File tests\e2e\e2e_windows.ps1 -Skip4KPerf

# 换端口 / 换产物目录
pwsh -File tests\e2e\e2e_windows.ps1 -Port 8796 -ArtifactRoot tests\e2e\artifacts\run-20261019
```

### 3.2 只跑 API 功能测试（服务已在跑时）

```powershell
.venv\Scripts\python.exe tests\e2e\e2e_windows.py `
    --base-url http://127.0.0.1:8796 `
    --expect-engine realesrgan --expect-device cuda --expect-fresh
```

可分段调试：`--sections health,targets,models,plan,tile,tiers,job,cancel,ui`；离线校验
plan 期望表：`--self-test-plan`（直接对 `app/target.py` 断言，不需要服务）。

### 3.3 多模型 / 降噪 / 4K 性能矩阵（服务已在跑时）

```powershell
.venv\Scripts\python.exe tests\e2e\model_matrix.py --base-url http://127.0.0.1:8796 `
    --artifact-dir tests\e2e\artifacts\latest\matrix
```

可选参数：`--models auto,realcugan,...` 选模型；`--no-4k` 跳过 4K 段；
`--sections matrix,perf4k,denoise,anime,rrdb` 选段；`--job-timeout 300`。
输出 `model_matrix.json`（含 perf 表、模型差异/降噪差异数值）+ 各模型产物。

### 3.4 质量 / 性能抽检

```powershell
.venv\Scripts\python.exe tests\e2e\quality_check.py --base-url http://127.0.0.1:8796 `
    --artifact-dir tests\e2e\artifacts\latest\quality
```

输出：`quality_check.json` + 每张照片的 `*_lanczos_vs_result.png` 左右对比图。

### 3.5 WSL（Ubuntu-24.04，GPU 直通）全链路

从 Windows 调 WSL helper（2026-10 起 WSL 已装 CUDA torch，预期 `realesrgan/cuda`）：

```powershell
$wsl = "$env:USERPROFILE\.dsh\tools\wsl.ps1"
& $wsl run -WorkDir '/mnt/c/Users/Deaddream/Desktop/pic_polish' -Cmd 'bash tests/e2e/e2e_wsl.sh'
```

或在 WSL 终端内：

```bash
cd /mnt/c/Users/Deaddream/Desktop/pic_polish
bash tests/e2e/e2e_wsl.sh
```

默认端口 **8798**；脚本会断言 `torch.cuda.is_available()`、`/api/health` 为
`backend=realesrgan / device=cuda`，并记录 720p→1080p GPU 任务的 wall/elapsed 耗时。
若 WSL CUDA 不可用 → 直接 FAIL（R2 要求）。可用环境变量覆盖：
`PIC_POLISH_E2E_PORT`、`PIC_POLISH_E2E_ARTIFACT_DIR`、`PIC_POLISH_E2E_SKIP_SETUP=1`、
`PIC_POLISH_E2E_START_TIMEOUT`、`PIC_POLISH_E2E_JOB_TIMEOUT`。

### 3.6 浏览器冒烟（服务已在跑时单独执行）

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File tests\e2e\browser_smoke.ps1 `
    -Url http://127.0.0.1:8796/ -ArtifactDir tests\e2e\artifacts\latest\browser
```

自动探测并**逐个尝试** Edge 与 Chrome（都没装 → exit 2，记录为 WARN）；每个浏览器都会抓
DOM（含模型卡片/推荐文案/降噪控件）并留一张 `screenshot_<Browser>.png` 作为报告证据。

### 3.7 离线自检（不需要服务，验证断言本身）

```powershell
.venv\Scripts\python.exe tests\e2e\selftest_checks.py
```

会校验 plan 期望表、`app/models.py` 模型注册表（若存在）、用合成图模拟「92% 黑块 / 单个黑
tile / 细黑接缝 / 干净渐变 / 真实 GPU 输出」确认内容有效性检查、噪声/降噪度量与
`model_used` 匹配规则。

---

## 4. 端口约定（避免互相踩）

| 端口 | 归属 | 说明 |
|------|------|------|
| 8765 | 默认 | 用户日常使用 |
| 8796 | **QA Windows E2E** | `e2e_windows.ps1` 独占 |
| 8798 | **QA WSL E2E** | `e2e_wsl.sh` 独占 |
| 8795 | scripts-dev | 启动/停止脚本自测 |
| 8797 | frontend-dev | 前端联调 |
| 8799 | api-dev / engine-dev | 单元/手工联调 |

测试结束必须保证端口无监听、无残留 uvicorn 进程；套件在 `finally` 中做了兜底清理，
且只会强杀「命令行含 `app.main:app` 且端口匹配」的本项目进程。

---

## 5. 断言清单（关键验收点）

### 5.1 API 契约（`e2e_windows.py`）

* `/api/health`：`status/app/version/engine.{backend,device,gpu,model,native_scale}/limits/cache`
  结构齐全；**R2 新增 `models: {default:"auto", available:[6 个模型 id]}`**；
  Windows 断言 `backend=realesrgan, device=cuda`。
* `/api/targets`：恰好 4 档，`id/label/rule/value` 与 `docs/API.md` 完全一致。
* `/api/models`（R2）：200、秒回（<2s，不加载模型）、`default=auto`、**6 个模型固定顺序**
  `auto / realesrgan-x4plus / realesrgan-x2plus / realesrgan-anime6b / realcugan /
  realcugan-conservative`；每个模型的 `label/description/family/native_scales/tags/
  recommended_for/denoise{supported,native,levels,default}/license/size_mb/source/available`
  结构正确；`realcugan.denoise.levels == [0,1,2,3]`、rrdb family `denoise.native=False`、
  cugan family `denoise.native=True`；全部 `available=True`（权重齐全）。
* `/api/jobs` 模型参数：非法 `model` → 400 中文 detail；`realcugan` 的 `denoise_level=4` → 400；
  202 响应的 `model` 对象（id/label/denoise/denoise_level）与文件的 `model/model_used/
  denoise_level` 字段存在/回显正确。
* `/api/plan`：`docs/API.md §1.1` 精确输出表：
  `1280×720@1080p→1920×1080`、`1920×1080@2k→2560×1440`、`1000×667@1080p→1616×1080`、
  `1080×1920@720p→720×1280`（downscale）、`3840×2160@1080p→1920×1080`（downscale）、
  `700×700@1080p→1080×1080` 等，以及 `scale/mode/aligned/note`；非法 target/尺寸/align → 400。
* `POST /api/jobs`：4 张（jpg / png / 竖图 / RGBA，`format=auto`）→ 202 → 轮询 → `done`；
  逐张下载用 PIL 断言 **尺寸精确等于 plan 值**、格式正确、`model=auto`、`model_used` 为 x2 模型。
* **4 档位 + 特殊几何的真实产物断言**：`1280×720@720p→1280×720`、`@2k→2560×1440`、
  `@4k→3840×2160`、`1000×667@1080p→1616×1080`（8px 吸附）、`720×1280@720p→720×1280`、
  `3840×2160@1080p→1920×1080`（downscale）。
* **内容有效性硬断言（可捕获 tiling 黑块 bug）**：每张结果 `black_ratio`（RGB 三通道
  最大值 < 10 的像素占比）**< 0.5%**、均值 **> 10**、白饱和 < 90%；真实/暗调照片按输入参考
  放宽（黑占比不得比输入多 2 个百分点、均值不得低于输入 60%）；RGBA 输出保留透明区。
* **多 tile 黑块/接缝断言**：`solid_gradient_700x700.png→1080×1080`：4×4 网格块均值差 <5、
  32×32 块均值差 <5、相邻列/行均值跳变 <3。
* 下载头 / ZIP / DELETE / 取消 / 静态页 / 负路径：同 R1（`Cache-Control: no-cache`、
  `attachment; filename="{stem}_{tier}.{ext}"`、ZIP 条目与字节一致、`freed_bytes>0` 等）。

### 5.2 模型矩阵 / 降噪矩阵（`model_matrix.py`）

* **逐模型 720p→1080p**（6 个模型各 1 次真实 GPU 任务）：输出 `1920×1080` JPEG、`model` 回显、
  `model_used` 与请求模型一致（auto→x2 系列；x4plus/x2plus/anime6b/cugan/conservative 各自
  关键字）、内容有效性、记录 elapsed/wall。
* **逐模型 4K**（`photo_1280x720.jpg → 3840×2160`）：尺寸精确、内容有效、记录耗时；
  契约要求单张 ≤180s。
* **realcugan denoise_level 0/1/2/3**（高斯噪声样本）：4 档全部合法、`denoise_level` 回显正确，
  **level 0 与 level 3 输出 mean_abs_diff > 0.5 级**（可测差异），并记录相邻档位差异。
* **anime6b vs x4plus**（动漫样本）：两者输出 mean_abs_diff > 1.0 级（证明 anime 模型确实生效）。
* **rrdb denoise on/off**（高斯噪声样本，x4plus）：输出有可测差异（>0.3 级），并记录
  Laplacian 高频能量变化（降噪应降低噪声能量）。
* 性能表：每模型 720p→1080p 与 4K 的 elapsed/wall 写入 `model_matrix.json.perf`。

### 5.3 质量 / 性能（`quality_check.py`）

* 内容有效性：`black_ratio < 0.5%`、均值 > 10、白饱和 < 90%（真实照片，按输入参考）。
* 清晰度：结果 Laplacian 方差 vs **同尺寸 Lanczos 基线**；硬门槛 **> 0.4×**（JPEG 噪声
  容差），`>1.0×` 记为「显著更清晰」，否则 WARN 并建议人工目视。
* 色偏：逐通道均值与 Lanczos 基线差 **< 20**；通道均值极差无异常膨胀。
* 性能：`720p→1080p` GPU 单张 `elapsed_sec ≤ 45s`（`docs/CONTRACT.md` §5）。
* 对比图：`artifacts/latest/quality/*_lanczos_vs_result.png`（中心区域 Lanczos | 结果）。

### 5.4 生命周期 / 清理（`e2e_windows.ps1`）

启动前端口空闲 → `scripts\server.ps1 -Action start` exit 0 → `.runtime\server.json`
（pid/port/url）→ `/api/health` ok → 启动后 cache 为空 → 测试结束向
`cache\uploads|results|tmp` 投放 3 个 dummy 文件 → `-Action stop` exit 0 →
**cache 三目录 0 文件**、**8796 无监听**、**runtime 文件已删**、**服务 PID 已退出**、
**无 `app.main:app --port 8796` 残留进程**。任何一步失败都会在 `finally` 中兜底 stop。

### 5.5 WSL GPU（`e2e_wsl.sh`）

`.venv-linux` CUDA torch 校验（FAIL if `cuda_available=False`）→ `opencv` 校验 →
`setup_env.sh` 幂等复跑 → `PIC_POLISH_ENGINE=auto start.sh --no-browser --port 8798` →
health 断言 **`backend=realesrgan / device=cuda`** + `models.available` 6 个模型 →
`/api/models` 契约（6 模型顺序/available/realcugan denoise 档位）→ curl targets/plan
精确尺寸表 → 2 文件 GPU 任务（`photo_1280x720.jpg→1920×1080` JPEG +
`tiny_320x240.png→1440×1080` PNG）→ 轮询 done + **记录 720p→1080p 耗时**（wall ≤60s）+
`model_used` 为 x2 系列 → python3+PIL 断言尺寸/格式/black_frac/均值 → preview/download 头与
文件名 → ZIP 条目 → DELETE → 投放 dummy 缓存 → `stop.sh` → cache 空、8798 无监听、
runtime 文件删除、PID 消失、无残留 uvicorn。

---

## 6. 通过标准与产物

* 阻塞项（FAIL）为 0 时退出码 `0`；`WARN` 只记录不阻塞（浏览器冒烟整体非阻塞）。
* 关键产物：
  * `tests/e2e/artifacts/latest/lifecycle.json` — 生命周期逐项结果 + 汇总（含 functional/quality/matrix）；
  * `tests/e2e/artifacts/latest/functional/` — `e2e_windows.json`、下载结果、`tier_outputs/`、`tile/`；
  * `tests/e2e/artifacts/latest/quality/` — `quality_check.json` + `*_lanczos_vs_result.png`；
  * `tests/e2e/artifacts/latest/matrix/` — `model_matrix.json` + 各模型/降噪/4K 真实产物；
  * `tests/e2e/artifacts/latest/browser/` — `browser_smoke.json`、`dom_*.html`、`screenshot_*.png`；
  * `tests/e2e/artifacts/latest/*.log` — 启动/停止/各步骤完整输出；
  * `tests/e2e/artifacts/wsl-latest/work/` — WSL 的 setup/start/stop 日志、health/models/job JSON、下载结果；
  * `tests/e2e/REPORT.md` — 最终验收报告（模型矩阵表、降噪矩阵、WSL GPU、性能、遗留问题、复现命令）。

---

## 7. 已知限制 / 注意事项

1. **独占端口**：运行前确保 8796（Windows）/ 8798（WSL）空闲；套件检测到端口被非本项目
   进程占用时会直接失败而不是去杀它。
2. **不要并发跑两套 E2E**：GPU 串行、cache 目录共享；Windows/WSL 同一时间只跑一套。
3. **WSL 现在必须有 CUDA torch**：R2 起 WSL 验收要求 `realesrgan/cuda`，CUDA 不可用即 FAIL
   （不再是 pil 兜底验收）。
4. **模型矩阵较重**：6 模型 × (720p + 4K) + denoise/anime/rrdb 差异任务，单次约 3–6 分钟；
   调试可用 `-Skip4KPerf` 或 `model_matrix.py --no-4k/--models/--sections`。
5. **质量指标是统计量**：Laplacian 方差受 JPEG 编码噪声影响，因此用「≥0.4× Lanczos」硬门槛
   + 对比图人工目视；模型差异阈值同理（mean_abs_diff 判「可测差异」，不判「更好看」）。
6. **浏览器冒烟尽力而为**：无 Edge/Chrome 时 exit 2 记为 WARN；headless 首次启动较慢。
7. **首次 WSL 运行**会执行 `scripts/setup_env.sh`（离线 wheelhouse/联网安装 torch），耗时可能数分钟。
8. `artifacts/latest` 每次运行会被清空重建（`-ArtifactRoot` 可指定其他目录）。
9. `samples/` 中的真实照片来自 `C:\Windows\Web\Wallpaper\Theme1`，仅本机使用；
   素材总量约 9 MB（预算 60 MB）。
