# PicPolish R2 全链路验收报告（task-11：多模型 + 降噪 + WSL GPU）

* **结论：PASS** —— Windows GPU（realesrgan/cuda）+ WSL GPU（realesrgan/cuda）双平台全链路通过，
  **1185 项验收断言 PASS / 0 FAIL / 3 WARN（均非阻塞）**；另有 34 项断言逻辑离线自检全 PASS。
* 验收时间：2026-10-19（15:49–15:58，Asia/Shanghai）
* 应用版本：`PicPolish 1.1.0`（R2：多模型 + 降噪）
* 验收对象：`docs/CONTRACT.md`（v1.1）+ `docs/API.md`（v1.1，§6/§7/§14）契约下的完整系统
* QA 范围：`tests/e2e/`（本报告 + 测试框架）、`samples/`；**未修改** `app/`、`web/`、`scripts/`、
  `docs/`、`.venv/`、`models/`
* 产物：`tests/e2e/artifacts/latest/`（Windows）、`tests/e2e/artifacts/wsl-latest/`（WSL）

> R1（task-6）基线报告已被本报告更新；R1 的全部回归断言仍在 R2 套件中执行并通过（§3.7）。

---

## 0. R2 验收目标逐条对照（task-11 验收标准）

| # | task-11 验收点 | 结果 | 关键证据 |
|---|----------------|------|----------|
| 1 | `/api/models` 结构/顺序/available/denoise；前端模型卡片 DOM（Edge+Chrome） | ✅ PASS | 6 模型固定顺序、全部 available、`realcugan.denoise.levels=[0,1,2,3]`；`/api/models` 15 ms 秒回；Edge/Chrome DOM 命中「智能推荐/Real-ESRGAN/Real-CUGAN/动漫 + 推荐 + 降噪控件」并留截图（§3.2、§3.9） |
| 2 | 逐模型真实 GPU 任务：尺寸/格式/`model_used`/black_ratio；CUGAN denoise 0..3；anime6b≠x4plus | ✅ PASS | 6 模型 × (720p→1080p + 4K) 全部精确到 1920×1080 / 3840×2160 且 `model_used` 正确；CUGAN L0 vs L3 **18.64 级**差异；anime6b vs x4plus **3.94 级**（§3.3–§3.6） |
| 3 | rrdb denoise on/off 有可测差异 | ✅ PASS | x4plus denoise=false vs true：mean_abs_diff **7.26 级**；含 lapvar 观察 WARN（§3.5） |
| 4 | WSL `backend=realesrgan device=cuda` + 720p→1080p 尺寸精确/耗时 | ✅ PASS | WSL health = realesrgan/cuda（RTX 4060 直通）；720p→1080p **elapsed 4.8 s / wall 5.14 s**，输出 1920×1080（§4） |
| 5 | 原回归项保持；生命周期 start→job→stop→cache 清空/端口/runtime | ✅ PASS | Windows 功能 672 项（含 R1 全部回归）；生命周期 36/36：cache 0 文件、8796 无监听、runtime 删除、PID 退出、无残留（§3.1） |
| 6 | 各模型 720p→1080p 与 4K 性能；写入 REPORT | ✅ PASS | 性能总表 §6（auto/x4plus/x2plus/anime6b/realcugan/conservative 全部记录） |
| 7 | 只改 tests/e2e 与 samples；bug 报 lead+owner | ✅ 遵守 | 新增 samples/anime+noisy、扩展 tests/e2e；验收中 **0 产品 FAIL**；并发环境问题已在过程消息中报告并加固 |

---

## 1. 测试环境

| 项目 | Windows（主验收） | WSL / Linux |
|------|-------------------|-------------|
| OS | Windows 10 专业版 10.0.19045 | Ubuntu 24.04.4 LTS（WSL2，kernel 6.18.40.1-microsoft-standard-WSL2） |
| Shell | PowerShell 7.6.7 | bash 5.2 |
| Python | 3.12.13（`.venv\Scripts\python.exe`） | 3.12.3（`.venv-linux/bin/python`） |
| 引擎 | **realesrgan + cuda**（health: RealESRGAN_x4plus / native_scale=4） | **realesrgan + cuda**（WSL GPU 直通，torch CUDA 可用） |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU，8188 MiB，驱动 596.21 | 同一 GPU（WSL 直通） |
| torch | 2.5.1+cu121，`cuda_available=True` | 2.5.1+cu121，`cuda_available=True` |
| 关键依赖 | PIL 12.2.0 / numpy 2.4.4 / **opencv 5.0.0** / httpx 0.28.1 / fastapi 0.146.0 / uvicorn 0.54.0 | PIL 13.0.0 / numpy 2.5.4 / **opencv 5.0.0** / fastapi 0.146.0 / uvicorn 0.54.0 |
| 模型权重 | x4plus 63.9MB、x2plus 64.0MB、anime6b 17.1MB、realcugan 11 个权重 55.4MB（全部 available） | 同左 |
| 测试端口 | **8796**（QA 独占） | **8798**（QA 独占） |

测试素材：`samples/` 共 **15 张图 + MANIFEST.json（8.99 MB）**。R2 新增：
`anime_1280x720.png`（动漫风格合成帧：大色块+粗线条+渐变）、
`noisy_gauss_1280x720.jpg`（真实照片 + 高斯噪声 σ=18）、
`noisy_saltpepper_1280x720.jpg`（椒盐噪声 2%）。

---

## 2. 执行与总览

### 2.1 执行命令

```powershell
# 1) Windows 全链路（start → functional → quality → model matrix → browser → stop → 清理断言）
pwsh -NoProfile -ExecutionPolicy Bypass -File tests\e2e\e2e_windows.ps1

# 2) WSL GPU 全链路（PIC_POLISH_ENGINE=auto，端口 8798）
$wsl = "$env:USERPROFILE\.dsh\tools\wsl.ps1"
& $wsl run -WorkDir '/mnt/c/Users/Deaddream/Desktop/pic_polish' `
    -Cmd 'PIC_POLISH_E2E_ARTIFACT_DIR=/mnt/c/Users/Deaddream/Desktop/pic_polish/tests/e2e/artifacts/wsl-latest bash tests/e2e/e2e_wsl.sh'

# 3) 断言逻辑离线自检（不需要服务）
.venv\Scripts\python.exe tests\e2e\selftest_checks.py
```

### 2.2 结果总览

| 套件 | 平台/端口 | 断言数 | PASS | FAIL | WARN | 耗时 |
|------|-----------|-------:|-----:|-----:|-----:|------|
| `e2e_windows.py`（API 全链路 + 模型字段回归） | Win / 8796 / GPU | 672 | **672** | 0 | 0 | 41.2 s |
| `quality_check.py`（真实照片质量 + 性能） | Win / 8796 / GPU | 27 | **27** | 0 | 0 | 6.9 s |
| `model_matrix.py`（6 模型 × 720p/4K + 降噪矩阵） | Win / 8796 / GPU | 348 | **347** | 0 | 1 | 101.0 s |
| `browser_smoke.ps1`（Edge + Chrome，含模型 DOM） | Win / 8796 | 29 | **27** | 0 | 2 | ~5 s |
| `e2e_windows.ps1`（生命周期 + 清理断言） | Win / 8796 | 36 | **36** | 0 | 0 | 174.2 s（含全部） |
| `e2e_wsl.sh`（WSL GPU 全链路） | WSL / 8798 | 76 | **76** | 0 | 0 | 约 1 分钟（含 setup 幂等 36 s） |
| **验收合计** | | **1188** | **1185** | **0** | **3** | |
| `selftest_checks.py`（断言逻辑自检，离线） | — | 34 | **34** | 0 | 0 | ~3 s |

> 3 个 WARN 均为设计内/说明性观察（见 §7），不影响验收结论：
> ① rrdb denoise 后 Laplacian 高频能量反升（只要求“可测差异”，已满足）；
> ②③ Edge/Chrome 静态 DOM 冒烟拿不到「上传后结果卡片」里的 model_used（预期）。

功能套件 672 项构成：health 32、targets 23、**models 109**、plan 210、negative 11、tile 16、
tiers 66、job 187、cancel 8、ui 10。
矩阵套件 348 项构成：models 16、matrix 96、matrix4k 102、**denoise 102**、anime_model 32。

---

## 3. Windows 全链路验收（GPU / 8796）

### 3.1 生命周期与清理证据（36/36 PASS）

| 检查项 | 结果 | 证据 |
|--------|------|------|
| 独占性前置检查 | PASS | 无其他 PicPolish dev 服务监听（8765/8795/8797/8799 空闲） |
| 启动前端口空闲 | PASS | 8796 free |
| `scripts\server.ps1 -Action start -NoBrowser -Port 8796` | PASS | exit 0；健康检查通过 |
| `.runtime\server.json` | PASS | port=8796、url、pid=36736 |
| `/api/health` 引擎 | PASS | `backend=realesrgan device=cuda gpu=NVIDIA GeForce RTX 4060 Laptop GPU` |
| 启动后 cache 为空 + 并发金丝雀 | PASS | 0 files；`cache/tmp/qa_e2e_marker.bin` 创建成功且全程未被清 |
| functional / quality / matrix / browser | PASS | 672 / 27 / 347 / 27 PASS |
| stop 脚本 | PASS | exit 0，耗时 **2.79 s** |
| 停止后 cache 清空 | PASS | 删除 dummy 5 个文件/161.2 KB → **0 个文件** |
| 端口释放 / runtime 删除 / PID 退出 / 无 uvicorn 残留 | PASS | 8796 无监听；`.runtime\server.json` 不存在；pid 36736 消失；无 `app.main:app --port 8796` |

### 3.2 模型列表 API 契约（109 项，全 PASS）

`GET /api/models` 200，**15 ms** 返回（不加载模型），`default=auto`，6 个模型固定顺序：

| # | id | label | family | available | denoise（supported/native/levels） | size_mb | license |
|---|----|-------|--------|-----------|-------------------------------------|--------:|---------|
| 1 | `auto` | 智能推荐 | rrdb | ✅ | ✅ / ✘ / [0,1] | 127.9 | BSD-3-Clause |
| 2 | `realesrgan-x4plus` | Real-ESRGAN 通用照片 | rrdb | ✅ | ✅ / ✘ / [0,1] | 63.9 | BSD-3-Clause |
| 3 | `realesrgan-x2plus` | Real-ESRGAN 低倍率 | rrdb | ✅ | ✅ / ✘ / [0,1] | 64.0 | BSD-3-Clause |
| 4 | `realesrgan-anime6b` | Real-ESRGAN 动漫 | rrdb | ✅ | ✅ / ✘ / [0,1] | 17.1 | BSD-3-Clause |
| 5 | `realcugan` | Real-CUGAN（动漫/视频） | cugan | ✅ | ✅ / **✔** / **[0,1,2,3]** | 55.4 | MIT |
| 6 | `realcugan-conservative` | Real-CUGAN 保守 | cugan | ✅ | ✘ / ✔ / [0] | 15.2 | MIT |

同时验证：每个模型的 `description/tags/recommended_for/native_scales/source` 完整；
`/api/health.models = {default:"auto", available:[6 ids]}`；非法 model → 400 中文 detail；
`realcugan` 的 `denoise_level=4` → 400；202 响应 `model` 对象与文件 `model/model_used/denoise_level`
字段回显正确（默认 auto → 文件 `model_used=RealESRGAN_x2plus`）。

### 3.3 逐模型真实任务：720p → 1080p（全 PASS）

统一输入 `photo_1280x720.jpg`，目标 1080p：**实际输出全部精确 = 1920×1080 JPEG**，`model_used` 正确：

| 请求模型 | `model_used`（实际解析） | 输出 | 均值 | black_ratio | 耗时（elapsed/wall） |
|----------|--------------------------|------|-----:|------------:|---------------------|
| `auto`（ratio 1.5→2x） | RealESRGAN_x2plus | 1920×1080 JPEG | 81.16 | 0.082% | **2.2 s / 2.55 s** |
| `realesrgan-x4plus` | RealESRGAN_x4plus | 1920×1080 JPEG | 77.60 | 0.145% | 8.7 s / 8.86 s |
| `realesrgan-x2plus` | RealESRGAN_x2plus | 1920×1080 JPEG | 81.16 | 0.082% | 2.0 s / 2.11 s |
| `realesrgan-anime6b` | RealESRGAN_x4plus_anime_6B | 1920×1080 JPEG | 81.04 | 0.143% | 3.2 s / 3.42 s |
| `realcugan` | Real-CUGAN_2x_no-denoise | 1920×1080 JPEG | 83.61 | 0.043% | **1.0 s / 1.28 s** |
| `realcugan-conservative` | Real-CUGAN_2x_conservative | 1920×1080 JPEG | 84.08 | 0.043% | **0.9 s / 1.30 s** |

全部满足内容有效性（black_ratio < 0.5%、非全白、通道均值合理；CUGAN uint8 输出无发灰/溢出）、
`job.model` 回显、`denoise_level=0`、`model` 字段等于请求值。

### 3.4 逐模型真实任务：4K（3840×2160，全 PASS）

| 请求模型 | `model_used` | 实际输出 | 均值 | black_ratio | 耗时（elapsed/wall） |
|----------|--------------|----------|-----:|------------:|---------------------|
| `auto`（ratio 3.0→4x） | RealESRGAN_x4plus | 3840×2160 JPEG | 77.59 | 0.082% | 10.1 s / 10.13 s |
| `realesrgan-x4plus` | RealESRGAN_x4plus | 3840×2160 JPEG | 77.59 | 0.082% | 9.7 s / 9.95 s |
| `realesrgan-x2plus` | RealESRGAN_x2plus | 3840×2160 JPEG | 81.13 | 0.025% | 2.5 s / 2.61 s |
| `realesrgan-anime6b` | RealESRGAN_x4plus_anime_6B | 3840×2160 JPEG | 81.02 | 0.062% | 3.3 s / 3.31 s |
| `realcugan`（native 3x） | Real-CUGAN_3x_no-denoise | 3840×2160 JPEG | 83.74 | 0.000% | 1.5 s / 1.66 s |
| `realcugan-conservative` | Real-CUGAN_3x_conservative | 3840×2160 JPEG | 83.28 | 0.004% | 1.6 s / 1.66 s |

契约要求 4K 单张 ≤180 s：全部远低于上限（最大 10.1 s）；8 GB 显存无 OOM。

### 3.5 降噪矩阵（102 项，全 PASS）

**A. realcugan `denoise_level` 0/1/2/3**（输入 `noisy_gauss_1280x720.jpg`，1080p）：

| denoise_level | `model_used` | 输出 | 均值 | black_ratio | 耗时 |
|---------------|--------------|------|-----:|------------:|-----:|
| 0 | Real-CUGAN_2x_no-denoise | 1920×1080 JPEG | 82.55 | 0.076% | 0.8 s |
| 1 | Real-CUGAN_2x_denoise1x | 1920×1080 JPEG | 84.44 | 0.076% | 0.8 s |
| 2 | Real-CUGAN_2x_denoise2x | 1920×1080 JPEG | 83.83 | 0.001% | 0.8 s |
| 3 | Real-CUGAN_2x_denoise3x | 1920×1080 JPEG | 84.19 | 0.029% | 0.8 s |

差异（硬断言 L0 vs L3 > 0.5 级）：`mean_abs_diff(L0,L3)=18.64`、RMSE=23.51、
Laplacian 高频能量 **3219 → 391**（降噪显著）；相邻档差异 6.27 / 13.52 / 8.53 级。

**B. rrdb 模型 denoise=false vs true**（x4plus + 高斯噪声样本，1080p）：
输出差异 `mean_abs_diff=7.26`、RMSE=9.91；均值 75.96 → 76.40、black_ratio 0.224% → 0.608%；
Laplacian 826 → 1119（WARN，见 §7-②）。**结论：denoise 参数确实生效且可测。**

对比图（左：no-denoise / 右：denoise3x）：

![Real-CUGAN 降噪对比](artifacts/latest/matrix/model_matrix/compare_cugan_denoise_L0_vs_L3.png)

### 3.6 anime6b vs x4plus（动漫样本，32 项全 PASS）

输入 `anime_1280x720.png` → 1920×1080 PNG：

| 模型 | `model_used` | 均值 | black_ratio | 耗时 |
|------|--------------|-----:|------------:|-----:|
| `realesrgan-anime6b` | RealESRGAN_x4plus_anime_6B | 143.01 | 0.683%* | 3.6 s |
| `realesrgan-x4plus` | RealESRGAN_x4plus | 144.90 | 0.136% | 11.1 s |

差异 `mean_abs_diff=3.94 级`（阈值 >1.0）、RMSE=6.22、lapvar 1043 vs 783 →
**anime 模型确实被使用且输出明显不同**。
\* 动漫样张本身有粗黑描边，内容有效性按输入参考判定（非固定 0.5% 阈值），已通过。

![anime6b vs x4plus 对比](artifacts/latest/matrix/model_matrix/compare_anime6b_vs_x4plus.png)

### 3.7 R1 回归（原有验收项保持通过）

* **`/api/plan` 精确尺寸表**（210 项）：1280×720@1080p→1920×1080、1920×1080@2k→2560×1440、
  1000×667@1080p→**1616×1080**、1080×1920@720p→720×1280、3840×2160@1080p→1920×1080（downscale）、
  1280×720@720p→1280×720、700×700@1080p→1080×1080 等；非法输入 400。
* **4 档位/特殊几何真实产物**（66 项）：720p→1280×720、2k→2560×1440、4k→3840×2160、
  odd→1616×1080、竖图@720p→720×1280、4K 输入 downscale→1920×1080；耗时 0.42/2.89/**11.00**/1.69/0.44/0.42 s。
* **4 文件混合任务**（187 项）：jpg/png/竖图/RGBA → 202 → done，
  总耗时 **11.05 s（elapsed 10.8 s）**；输出 1920×1080 JPEG(982,277 B)、1920×1080 PNG(3,157,242 B)、
  1080×1920 JPEG(466,659 B)、1440×1080 PNG(1,490,079 B)；**RGBA `alpha_min=0/alpha_max=255/
  透明区 9.4%`** → 透明通道保留；preview/download 头、filename、ZIP 4 条目字节一致、DELETE/
  freed_bytes、取消、409/404/415/400 负路径全部通过。
* **内容有效性 + tiling 黑块回归**（16 项）：`solid_gradient_700x700→1080×1080`：
  4×4 块均值偏差 **1.89**（<5）、32×32 块偏差 **3.08**（<5）、相邻列/行跳变
  **0.693 / 0.698**（<3）、black_ratio 0.000% → 无黑块/无接缝；R1 的 92.33% 黑块 bug 仍被防护。
* 健康检查/limits/静态页（health/targets/ui 共 65 项）全部通过。

### 3.8 真实照片质量与性能（27/27 PASS）

| 照片 | 输入→输出 | Laplacian SR / Lanczos / bicubic | 相对 Lanczos | 均值/std | black_ratio | 通道均值差 |
|------|-----------|----------------------------------|--------------|----------|-------------|------------|
| photo_1280x720.jpg | 1280×720 → 1920×1080 | **1373.05** / 428.92 / 365.77 | **3.20×** | 81.16 / 45.56 | 0.082% | 3.75/2.71/1.96 |
| portrait_720x1280.jpg | 720×1280 → 1080×1920 | **189.63** / 36.57 / 32.98 | **5.19×** | 140.54 / 39.95 | 0.000% | 0.15/0.42/0.85 |

* 720p→1080p GPU 单张（auto→x2plus）：**elapsed 2.3 s / wall 2.53 s**（契约目标 ≤45 s，约 20× 余量）。
* 无全黑/全白/严重色偏；对比图：

![真实照片清晰度对比](artifacts/latest/quality/photo_1280x720_lanczos_vs_result.png)

![竖图清晰度对比](artifacts/latest/quality/portrait_720x1280_lanczos_vs_result.png)

### 3.9 浏览器冒烟（Edge + Chrome，27 PASS / 2 WARN）

`--headless=new --dump-dom --virtual-time-budget=8000` 抓取 `http://127.0.0.1:8796/`：

| 检查项 | Edge | Chrome |
|--------|------|--------|
| DOM 抓取 | ✅ 12,674 chars | ✅ 12,674 chars |
| 标题 | ✅ `PicPolish · AI 图片超分` | ✅ 同左 |
| 上传区（file input/拖拽） | ✅ | ✅ |
| 4 档位卡片可见 | ✅ 720P/1080P/2K/4K | ✅ |
| **模型选择区** | ✅ | ✅ |
| **模型卡片文案** | ✅ 智能推荐 / Real-ESRGAN / Real-CUGAN / 动漫 | ✅ 同左 |
| **推荐文案** | ✅ | ✅ |
| **降噪控件** | ✅ 降噪文案 + checkbox/等级控件 | ✅ |
| 另存为/下载控件 | ✅ | ✅ |
| 引擎徽章 | ✅ `AI 超分` | ✅ |
| 截图 | ✅ 150,238 B | ✅ 150,238 B |

（WARN：`model_used_hint` —— 静态 DOM 冒烟未上传，结果卡片尚未渲染，预期内，见 §7-③）

![Edge 浏览器截图（R2 模型选择区）](artifacts/latest/browser/screenshot_Edge.png)

---

## 4. WSL / Ubuntu 24.04 GPU 全链路（76/76 PASS）

WSL 侧现为 **CUDA torch + GPU 直通**，验收要求 `realesrgan/cuda`：

| 阶段 | 结果 | 证据 |
|------|------|------|
| `scripts/setup_env.sh` 幂等 | ✅ PASS | 检测到已存在 `.venv-linux`（不重建），**36 s** 校验通过；`torch 2.5.1+cu121（CUDA 可用: True，GPU: RTX 4060）`；Web 依赖 + opencv 可用 |
| `.venv-linux` GPU/opencv 校验 | ✅ PASS | `torch.cuda.is_available()=True`、`device=NVIDIA GeForce RTX 4060 Laptop GPU`、`cv2 5.0.0` |
| `PIC_POLISH_ENGINE=auto bash start.sh --no-browser --port 8798` | ✅ PASS | exit 0；`引擎: AI 超分 / device=cuda / GPU=NVIDIA GeForce RTX 4060 Laptop GPU / model=RealESRGAN_x4plus`；runtime port=8798 pid=4668 |
| `/api/health` | ✅ PASS | `realesrgan / cuda`，`models.available` 6 个，limits 正常，version 1.1.0 |
| `/api/models` 契约 | ✅ PASS | 6 模型固定顺序 / 全部 available / realcugan denoise levels [0,1,2,3] / family 正确 |
| `/api/targets` + `/api/plan` | ✅ PASS | 4 档位；1280×720@1080p→1920×1080、1000×667@1080p→1616×1080、竖图@720p→720×1280、4K 输入 downscale→1920×1080；非法 target→400 |
| GPU 任务（720p 照片 + tiny png） | ✅ PASS | 2 文件 done；**elapsed 4.8 s / wall 5.14 s**（目标 ≤45 s + overhead）；`model_used=RealESRGAN_x2plus`；`denoise_level=0` |
| PIL 断言 | ✅ PASS | 1920×1080 JPEG（mean 80.7 / black 0.214%）、1440×1080 PNG（mean 79.1 / black 0.202%） |
| 响应头/文件名/ZIP | ✅ PASS | `image/jpeg`/`image/png`、`cache-control: no-cache`、`attachment; filename="photo_1280x720_1080p.jpg"`、ZIP 条目 `photo_1280x720_1080p.jpg` + `tiny_320x240_1080p.png` |
| DELETE 任务 | ✅ PASS | 200 `deleted=true`，随后 GET 404 |
| `bash stop.sh --port 8798` | ✅ PASS | exit 0；终止 PID 4668；清理 3 个 dummy 文件（24 B） |
| 停止后 cache/端口/runtime/PID/残留 | ✅ PASS | cache 三目录 **0 文件**、8798 无监听、runtime 删除、pid 4668 消失、无 `app.main:app` 残留 |

> WSL 的 720p→1080p 耗时（4.8 s）高于 Windows（2.2 s），主要来自 WSL 首次 CUDA 模型加载与
> `/mnt/c` I/O 开销；仍远低于 45 s 契约目标。

---

## 5. 断言逻辑自检（`selftest_checks.py`，34/34 PASS，离线）

* plan 期望表 vs `app/target.py`（13 用例）✅
* `app/models.py` 注册表：6 模型顺序、family、权重可用性 ✅
* 内容有效性：干净渐变/真实 GPU 输出不误报；92% 黑块、单黑 tile、细黑接缝均被捕获 ✅
* 暗调照片按输入参考放行、额外黑块被捕获 ✅
* 噪声/降噪度量：噪声样本 lapvar 4586 > 干净 1710；高斯模糊（模拟降噪）后 lapvar 4586→38；
  clean vs noisy mean_abs_diff=14.81 ✅
* `model_used` 匹配规则：接受正确 family、拒绝错误 family（realcugan vs x4plus、x4plus vs anime 等）✅
* alpha 合成、Content-Disposition（普通/RFC 5987 中文）✅

---

## 6. 性能总表（Windows GPU）

**720p → 1080p（1920×1080）**：

| 模型 | elapsed (s) | wall (s) | 契约 |
|------|------------:|---------:|------|
| auto（→x2plus） | **2.2** | 2.55 | ≤45 s（通用/auto） |
| realesrgan-x2plus | 2.0 | 2.11 | — |
| realesrgan-x4plus | 8.7 | 8.86 | ≤90 s（其他模型） |
| realesrgan-anime6b | 3.2 | 3.42 | ≤90 s |
| realcugan | 1.0 | 1.28 | ≤90 s |
| realcugan-conservative | 0.9 | 1.30 | ≤90 s |

**4K（3840×2160）**：

| 模型 | elapsed (s) | wall (s) | 契约 |
|------|------------:|---------:|------|
| auto（→x4plus） | 10.1 | 10.13 | ≤180 s |
| realesrgan-x4plus | 9.7 | 9.95 | ≤180 s |
| realesrgan-x2plus | 2.5 | 2.61 | ≤180 s |
| realesrgan-anime6b | 3.3 | 3.31 | ≤180 s |
| realcugan | 1.5 | 1.66 | ≤180 s |
| realcugan-conservative | 1.6 | 1.66 | ≤180 s |

其他：4 文件混合任务 1080p 总耗时 11.05 s；4 档位/特殊几何 0.42–11.00 s；
WSL 720p→1080p 4.8 s / 5.14 s（含首次加载）。
**显存**：`/api/health` 不暴露显存；本机 8 GB 下 6 模型 × 4K 全部完成、无 OOM
（engine-dev 侧单张 720p 峰值约 364 MB，R1 报告口径）。

---

## 7. 遗留问题 / 已知限制（无阻塞 FAIL）

1. **共享 cache + 单实例锁的运行环境风险（已加固）**：验收前预演时曾观察到并发测试的
   stop 脚本清空 `cache/uploads`，导致正在处理的任务报「无法读取上传图片」（瞬时 FAIL，非产品缺陷）。
   最终验收前已加：① wrapper 启动前检查 8765/8795/8797/8799 是否有其他实例（有则拒绝开跑）；
   ② `cache/tmp/qa_e2e_marker.bin` 并发金丝雀（运行期被清则 WARN）；③ 运行中并发服务检测。
   最终 run 独占环境，金丝雀全程存活。**运维建议：同一时间只运行一个 PicPolish 实例/验收。**
2. **rrdb denoise 的 Laplacian 指标 WARN**：`denoise=true` 与 `false` 的输出差异已硬断言通过
   （7.26 级），但 OpenCV 预降噪后再经 SR 重锐化，高频能量反而从 826 升到 1119（WARN 仅记录）。
   如需量化“降噪收益”，建议后续用 PSNR/SSIM 对齐干净参考图；当前按契约只要求“可测差异”。
3. **浏览器 result-card `model_used` WARN**：静态 headless dump-dom 不执行“上传→任务→结果”流程，
   结果卡片尚未渲染，因此该可选断言记 WARN；模型卡片/推荐/降噪控件均已 PASS，截图已留证。
   （本机无 node/Playwright，未做交互式 E2E。）
4. **显存不可通过 API 观测**：性能表记录耗时；显存来自 engine-dev 侧实测（R1 口径 ~364 MB）。
5. **WSL setup_env.sh 幂等复跑约 36 s**：主要耗时在 pip 依赖校验；可接受，未优化。
6. **模型缺失场景未做 E2E**：6 模型权重全部存在（`available=true`）；“权重缺失→400/
   available=false”由 engine-dev 单测覆盖，QA 未移走权重（不写 models/ scope）。
7. **模型质量为主观维度**：矩阵验收断言“可测差异 + 内容合法”，不评判“哪个更好看”；
   已留 CUGAN 降噪、anime6b vs x4plus 对比图供人工目视。
8. **动漫样张 black_ratio 0.683%（> 固定 0.5% 数值）属正常**：样张含粗黑描边，内容有效性按
   **输入参考**判定（输入黑占比 +2% 容差、均值不低于输入 60%），已通过且 R1 的 92% 黑块仍会被捕获。

---

## 8. 复现命令

```powershell
# 生成/刷新测试素材（含 R2 anime/noisy）
.venv\Scripts\python.exe tests\e2e\make_samples.py

# 断言逻辑离线自检
.venv\Scripts\python.exe tests\e2e\selftest_checks.py
.venv\Scripts\python.exe tests\e2e\e2e_windows.py --self-test-plan

# Windows 全链路（含 4K 模型矩阵；确保 8795/8797/8799 无其他实例）
pwsh -NoProfile -ExecutionPolicy Bypass -File tests\e2e\e2e_windows.ps1
#   调试：-Skip4KPerf / -SkipMatrix / -SkipQuality / -SkipBrowser / -SkipFunctional
#   指定矩阵子集：.venv\Scripts\python.exe tests\e2e\model_matrix.py --base-url http://127.0.0.1:8796 --models realcugan --sections denoise

# WSL GPU 全链路
$wsl = "$env:USERPROFILE\.dsh\tools\wsl.ps1"
& $wsl run -WorkDir '/mnt/c/Users/Deaddream/Desktop/pic_polish' -Cmd 'bash tests/e2e/e2e_wsl.sh'

# 单跑质量/性能、浏览器冒烟（服务运行中）
.venv\Scripts\python.exe tests\e2e\quality_check.py --base-url http://127.0.0.1:8796
pwsh -NoProfile -ExecutionPolicy Bypass -File tests\e2e\browser_smoke.ps1 -Url http://127.0.0.1:8796/
```

---

## 9. 产物索引（供 Lead 复核）

| 路径 | 内容 |
|------|------|
| `tests/e2e/REPORT.md` | 本报告（R2） |
| `tests/e2e/artifacts/latest/lifecycle.json` | Windows 生命周期 36 项逐项结果 + functional/quality/matrix/browser 汇总 |
| `tests/e2e/artifacts/latest/functional/e2e_windows.json` | 672 项 API/回归断言明细（含模型字段、tile、tiers、RGBA、headers/ZIP/DELETE） |
| `tests/e2e/artifacts/latest/quality/` | `quality_check.json` + 2 张真实照片 Lanczos vs 结果对比图 |
| `tests/e2e/artifacts/latest/matrix/model_matrix.json` | 6 模型 × 720p/4K、降噪矩阵、anime/rrdb 差异、perf 表 |
| `tests/e2e/artifacts/latest/matrix/model_matrix/` | 各模型/降噪/4K 真实产物 + `compare_cugan_denoise_L0_vs_L3.png`、`compare_anime6b_vs_x4plus.png` |
| `tests/e2e/artifacts/latest/browser/` | Edge/Chrome DOM 抓取、截图（各 150 KB）、`browser_smoke.json` |
| `tests/e2e/artifacts/latest/*.log` | start/stop/functional/quality/matrix/browser 完整日志 |
| `tests/e2e/artifacts/wsl-latest/work/` | WSL：setup/start/stop 日志、health/models/job JSON、下载结果、ZIP |
| `samples/MANIFEST.json` | 15 张测试素材的来源与用途（含 R2 anime/noisy） |

---

## 10. QA 结论

在 Windows 10 + RTX 4060 Laptop（realesrgan/cuda）与 Ubuntu 24.04 WSL GPU 直通
（realesrgan/cuda）双平台，按 v1.1 契约完成「模型列表/推荐/降噪参数 → 6 模型真实任务矩阵 →
CUGAN 降噪 0..3 → anime/通用模型差异 → rrdb denoise on/off → 真实照片质量 → 4K 性能 →
浏览器模型 DOM → 停止清理」全链路验收：

* **1185 项验收断言 PASS / 0 FAIL**（Windows 1109 项含 3 WARN，WSL 76 项全 PASS），
  另有 34 项断言逻辑离线自检全 PASS；
* 6 个模型全部可跑，输出尺寸精确 = `/api/plan`，`model_used` 正确，
  auto 按倍率正确选择 x2/x4（720p→x2plus、4K→x4plus）；
* CUGAN denoise 0/1/2/3 全部合法且差异显著（L0 vs L3 = 18.64 级，lapvar 3219→391）；
  anime6b vs x4plus = 3.94 级；rrdb denoise on/off = 7.26 级（均满足“可测差异”）；
* 性能：720p→1080p auto **2.2 s**（≤45 s）、4K 全模型 ≤10.1 s（≤180 s）；WSL GPU 4.8 s；
* 生命周期：停止后 **cache 清空、端口释放、runtime 删除、进程无残留**；
* R1 的 tiling 黑块防护、plan 精确尺寸、4 档位真实产物、RGBA 透明通道、headers/ZIP/DELETE
  等回归项全部保持通过。

**QA 签署：PASS（task-11 完成）** — qa-tester，2026-10-19。
