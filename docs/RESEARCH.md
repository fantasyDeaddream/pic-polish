# 开源图像超分（Super-Resolution）项目调研

> 项目：PicPolish（本地 Web 图片超分工具）
> 调研日期：2026-10-19
> 调研方式：GitHub REST API（仓库许可证、release 资产名与字节数）、仓库 README/源码、PyPI JSON API；网络受限，`web_fetch` 对 github.com / huggingface.co / docs.opencv.org 均返回 "resolves to a non-public IP address"，因此改用本机 PowerShell 直连 `api.github.com`、`pypi.org`、`huggingface.co/api` 取数，未使用任何镜像站内容作为事实来源。
>
> 可信度标记：**✅ 已验证** = 本次通过 API/源码/HTTP HEAD 直接确认；**⚠️ 推测/经验** = 基于架构与工程常识的判断，未逐项实测。

---

## 1. 调研结论摘要

1. **本期选型（PyTorch 2.5.1 + 官方 Real-ESRGAN x4plus/x2plus + 自建 RRDBNet）是当前约束下的最优解**：权重仅 63.9 MiB/个、BSD-3 许可、纯离线、单卡 8GB 可通过 tiling 跑 4K；同时彻底绕开 `basicsr` 在 torchvision ≥ 0.17 上的导入崩溃（✅ 已验证，见 §3.3）。
2. **不要 `pip install realesrgan`**：PyPI 上 `realesrgan 0.3.0` 依赖 `basicsr>=1.4.2`、`facexlib`、`gfpgan`，会连带拉入已损坏的 basicsr（以及可能重装 torch/torchvision），与项目 venv 的 torchvision 0.20.1 冲突（✅ 已验证依赖声明）。vendored RRDBNet 是唯一干净路径。
3. **Real-ESRGAN 的 x4/x2 权重分属不同 release tag**（x4plus 在 v0.1.0，x2plus 在 v0.2.1，anime6B 在 v0.2.2.4，general-x4v3 在 v0.2.5.0），下载脚本不能假设单一 tag（✅ 已验证）。
4. **学术 SOTA（HAT/SwinIR/DAT）不适合作为本期默认后端**：HAT 20.8M 参数、64×64 输入即 102.4G MACs，官方 README 明确 batch=4 约需 20GB 显存；SwinIR real-world L 权重 271.8MB。它们可作为"高质量慢速档"后续接入，而不是 8GB 笔记本的默认路径（✅ 参数/体积已验证）。
5. **动漫向方案（waifu2x / Real-CUGAN）更轻、更适合二次元，但通用照片会过度平滑或风格化**；Real-CUGAN 为 MIT、原生 2x/3x/4x、带多档降噪，是最值得作为"动漫模式"二期接入的后端（✅ 许可/倍率已验证）。
6. **许可红线**：Upscayl 与 upscayl-ncnn 为 **AGPL-3.0**、chaiNNer 为 **GPL-3.0**，只可借鉴设计与交互，不能把代码复制进本项目；Real-ESRGAN 为 BSD-3、BasicSR 为 Apache-2.0，vendored RRDBNet 结构需保留版权与许可声明（✅ 已验证）。

---

## 2. 候选项目对比表

> 体积单位：MiB（GitHub release 资产实测字节数换算）；"模型体积"指主要权重文件，不含运行时。
> 表中"离线部署"= 权重与推理代码均可本地留存、无需联网。

| 项目（仓库） | 任务类型 | 许可 | 模型体积 | 可离线部署 | 推理依赖 / 硬件 | 优点 | 缺点 / 坑 |
|---|---|---|---|---|---|---|---|
| **Real-ESRGAN**（PyTorch）<br>[xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) | 通用照片（含动漫 6B 变体） | BSD-3-Clause ✅ | x4plus 63.94、x2plus 63.96、anime6B 17.11、general-x4v3 4.66 ✅ | ✅ 完全离线 | PyTorch + CUDA（或 CPU）；8GB 显存需 tiling | 画质/泛化均衡；x2/x4 双原生倍率；tiling 成熟；权重小；文档全 | 官方代码依赖 basicsr（torchvision≥0.17 直接 ImportError）；包管理会拉 gfpgan/facexlib；默认 alpha 走模型导致耗时翻倍 |
| **Real-ESRGAN-ncnn-vulkan**<br>[xinntao/Real-ESRGAN-ncnn-vulkan](https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan) | 通用照片 / 动漫 | 仓库 API 返回 NOASSERTION，主仓库 BSD-3 ✅（分发前需人工确认） | Windows 便携包 43.37（含模型）✅ | ✅ 解压即用 | ncnn + Vulkan GPU（NVIDIA/AMD/Intel），无需 Python/CUDA | 零依赖、跨显卡、启动快、适合分发 | 模型是 .param/.bin，与 PyTorch 权重不通用；仅 4 个 x4 模型（无 x2plus/general-v3）；无 Python API，只能进程调用+临时文件；仓库更新停滞 |
| **ESRGAN（初代）**<br>[xinntao/ESRGAN](https://github.com/xinntao/ESRGAN) | 通用照片 | Apache-2.0 ✅ | 约 64MB 级（RRDB 16 blocks）⚠️ | ✅ | PyTorch | 经典基线、结构简单 | 仅 x4；被 Real-ESRGAN 全面超越；2022 后停更 ✅ |
| **BSRGAN**<br>[cszn/BSRGAN](https://github.com/cszn/BSRGAN) | 通用照片（真实退化） | Apache-2.0 ✅ | 权重经 Google Drive/Baidu 分发（体积未核） | ✅（需手动下载） | PyTorch + KAIR 代码库 | 盲退化训练，真实低质图表现好 | 仅 x4；依赖 KAIR 工程；无 pip 包；集成成本高 |
| **SwinIR**<br>[JingyunLiang/SwinIR](https://github.com/JingyunLiang/SwinIR) | 通用照片 / 去压缩伪影 | Apache-2.0 ✅ | classical x4 56.8–64.7；lightweight 16.4；real-world L 271.8 ✅ | ✅ | PyTorch + CUDA；`--tile 400` 可降显存 ✅ | Transformer 画质强；任务覆盖广（SR/去噪/JPEG） | 推理慢；real-world 权重巨大；无产品级 API；x4 为主 |
| **HAT**<br>[XPixelGroup/HAT](https://github.com/XPixelGroup/HAT) | 通用照片（含 real-world GAN） | Apache-2.0 ✅ | HAT 20.8M 参数；权重经 Google Drive（未核体积）✅ | ✅ | PyTorch + CUDA；官方 batch=4 约 20GB 显存 ✅；有 tile 模式 | 当前 PSNR/感知上限之一；有 Real_HAT_GAN_SRx4（sharper） | 显存/算力门槛高；x4 为主；部署重；8GB 上 4K 不现实 |
| **DAT**<br>[zhengchen1999/DAT](https://github.com/zhengchen1999/DAT) | 通用照片 | Apache-2.0 ✅ | DAT-S/DAT-2 11.21M；DAT-light 573K ✅ | ✅（Google Drive/Baidu） | PyTorch + CUDA，basicsr 训练框架 | 质量/效率平衡好；有轻量 DAT-light | 权重托管在网盘；x4 为主；basicsr 生态 |
| **waifu2x（原版）**<br>[nagadomi/waifu2x](https://github.com/nagadomi/waifu2x) | 动漫插画 | MIT ✅ | 模型随仓库/发行包，体积小 ✅ | ✅ | 原版 Torch/Lua；社区多为移植版 | 动漫降噪+2x 的经典基线 | 仅 1x/2x；原仓库维护弱（最后 push 2023-05）✅；不适合照片 |
| **waifu2x-ncnn-vulkan**<br>[nihui/waifu2x-ncnn-vulkan](https://github.com/nihui/waifu2x-ncnn-vulkan) | 动漫插画 | MIT ✅ | Windows 包 33.9（含 cunet/upconv 模型）✅ | ✅ 解压即用 | ncnn + Vulkan（A/N/I 卡） | 极轻、跨平台、CPU 也能跑（慢）；模型分 cunet/upconv_7 两族 | 仅 2x（内部 1x 去噪）；照片效果差；非 Python |
| **Real-CUGAN**<br>[bilibili/ailab/Real-CUGAN](https://github.com/bilibili/ailab/tree/main/Real-CUGAN) | 动漫插画/视频 | MIT ✅ | `updated_weights.zip` 51.5（多模型合集）✅ | ✅（GitHub Release/网盘） | PyTorch + CUDA；低显存模式（>1.5GB）✅ | 2x/3x/4x 原生；多档降噪（2x 四档+保守）；速度快于 Real-ESRGAN anime ✅（作者对比表） | 仅动漫；依赖旧版 torch 的发行包较大；官方代码非 pip 工程 |
| **realcugan-ncnn-vulkan**<br>[nihui/realcugan-ncnn-vulkan](https://github.com/nihui/realcugan-ncnn-vulkan) | 动漫 | MIT ✅ | Windows 包 43.8（20220728，含模型）✅ | ✅ | ncnn + Vulkan | 无 Python 依赖、支持 pro 模型 | 2023 后未更新 ✅；仅 2/3/4x 动漫 |
| **Upscayl**<br>[upscayl/upscayl](https://github.com/upscayl/upscayl) | 通用 + 动漫（多模型） | **AGPL-3.0** ✅ | 内置多模型；[custom-models](https://github.com/upscayl/custom-models) 24 个模型 1.2–66.8MB ✅ | ✅ 安装后离线 | Electron + [upscayl-ncnn](https://github.com/upscayl/upscayl-ncnn)（AGPL-3.0）+ Vulkan | 产品完成度高：队列、批量、模型管理、UI/UX 极佳；模型生态大 | AGPL 传染，代码不可复用；必须 Vulkan，多数 iGPU/CPU 不可用 ✅；安装包约 249MB |
| **chaiNNer**<br>[chaiNNer-org/chaiNNer](https://github.com/chaiNNer-org/chaiNNer) | 通用/动漫，节点式 | **GPL-3.0** ✅ | 不内置模型，用户自选（OpenModelDB 生态） | ✅ | PyTorch(CUDA/ROCm/MPS)、ONNX、NCNN、TensorRT ✅ | 架构支持最广、可编排预处理/后处理、TensorRT 加速 | 节点式学习曲线；GPL-3.0；桌面应用非服务；不适合直接嵌入 Web 后端 |
| **OpenCV dnn_superres**<br>[opencv_contrib/dnn_superres](https://github.com/opencv/opencv_contrib/tree/4.x/modules/dnn_superres) | 通用照片 | Apache-2.0 ✅ | EDSR ≈38.5MB、ESPCN ≈0.1MB、FSRCNN ≈0.04MB、LapSRN 1.3–3.9MB ✅ | ✅ | OpenCV DNN（CPU/CUDA/OpenCL），无需 PyTorch | 依赖极轻、模型极小、CPU 可跑、部署简单 | 画质落后；仅整数倍率（EDSR/ESPCN/FSRCNN 为 2/3/4，LapSRN 为 2/4/8）✅；无内建 tiling；无 alpha 处理 |
| **Waifu2x-Extension-GUI**<br>[AaronFeng753/Waifu2x-Extension-GUI](https://github.com/AaronFeng753/Waifu2x-Extension-GUI) | 通用/动漫，Windows GUI | NOASSERTION（自定义）⚠️ | 捆绑多引擎 | ✅ | Windows + 多后端 | 功能全、中文友好、适合参考产品形态 | 许可不清晰；仅 Windows；工程庞大难复用 |

**商用/在线方案（仅记录，不参与选型）**：Topaz Photo AI（付费本地，画质与产品化标杆）、Magnific AI / Let's Enhance / VanceAI（云端 SaaS，需上传图片，隐私与成本不可控）、腾讯 ARC 在线 demo（仅 anime6B，[Real-ESRGAN README 提及](https://github.com/xinntao/Real-ESRGAN)）。本项目要求本地离线 + REST API，故全部排除。

---

## 3. Real-ESRGAN 深入

### 3.1 模型家族（✅ 体积与下载地址均经 HEAD 200 + Content-Length 验证）

| 模型名 | 原生倍率 | 结构 | 权重体积 | 官方下载地址（release tag） | 适用 |
|---|---|---|---|---|---|
| `RealESRGAN_x4plus` | 4x | RRDBNet（num_feat=64, num_block=23, grow=32，约 16.7M 参数，由体积推算 ⚠️） | 63.94 MiB（67,040,989 B） | [v0.1.0](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth) | 通用照片默认档 |
| `RealESRGAN_x2plus` | 2x | 同为 23 blocks；forward 先 `pixel_unshuffle(x,2)`，再两次 2x 上采样 + `pixel_shuffle`，净 2x | 63.96 MiB（67,061,725 B） | [v0.2.1](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth) | 1080p→2K/4K 等低倍率、省算力 |
| `RealESRGAN_x4plus_anime_6B` | 4x | RRDBNet 仅 6 blocks，约 4.5M 参数 ⚠️ | 17.11 MiB（17,938,799 B） | [v0.2.2.4](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth) | 动漫插画 |
| `realesr-general-x4v3` | 4x | SRVGG 小模型（S 号） | 4.66 MiB（4,885,111 B） | [v0.2.5.0](https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth) | 快速/弱显卡；支持 `-dn` 降噪强度 |
| `realesr-general-wdn-x4v3` | 4x | SRVGG（去噪增强变体） | 4.66 MiB | [v0.2.5.0](https://github.com/xinntao/Real-ESRGAN/releases/tag/v0.2.5.0) | 噪声大的图 |
| `realesr-animevideov3` | 4x（可 2/3/4 输出） | VGG 式 XS 小模型 | 2.39 MiB | [v0.2.5.0](https://github.com/xinntao/Real-ESRGAN/releases/tag/v0.2.5.0) | 动漫视频帧 |
| `RealESRNet_x4plus` | 4x | RRDBNet（PSNR 导向） | 63.94 MiB | [v0.1.1](https://github.com/xinntao/Real-ESRGAN/releases/tag/v0.1.1) | 更保守、少"AI 味" |

> **坑 1**：x4plus 与 x2plus 不在同一个 release tag，自动化下载需分别写 URL。
> **坑 2**：x4 模型的 `--outscale` 可设 1–4 任意浮点，但实现是"先 4x 推理，再 `cv2.resize(..., INTER_LANCZOS4)` 到目标"，不是原生任意倍率（✅ 源码验证，[utils.py](https://github.com/xinntao/Real-ESRGAN/blob/master/realesrgan/utils.py)）。
> **坑 3**：`realesr-general-x4v3` 的 `-dn` 通过 DNI 混权实现，只在加载时生效，切换强度需保留原始 state_dict。

### 3.2 许可协议

- **Real-ESRGAN 仓库：BSD-3-Clause**（GitHub API `license.spdx_id` 验证 ✅，[LICENSE](https://github.com/xinntao/Real-ESRGAN/blob/master/LICENSE)）。允许商用、修改、再分发，条件是保留版权声明、条款与免责声明，且不得用作者名义背书。
- **权重**：官方以同一仓库 release 分发，按仓库 BSD-3 使用（✅ 事实层面）；**训练数据来源与肖像/商标风险不因权重许可而消失**（⚠️ 法务判断，商用前建议法务确认）。
- **BasicSR：Apache-2.0**（[XPixelGroup/BasicSR](https://github.com/XPixelGroup/BasicSR) ✅）。本项目的 `RRDBNet` 结构源自 basicsr/Real-ESRGAN，属 Apache-2.0 代码：**vendored 时必须在文件头保留版权与 Apache-2.0 声明，并附 NOTICE**（⚠️ 合规要求，非可选项）。
- **Hugging Face 镜像**：[amd/realesrgan-x4plus](https://huggingface.co/amd/realesrgan-x4plus) 提供同名 `RealESRGAN_x4plus.pth`，卡片 license 为 `bsd-3-clause`（✅ API 验证），可作为 GitHub 下载失败时的备用源；`hf-mirror.com` 仅作网络镜像候选，**可达性与文件一致性未验证** ⚠️。

### 3.3 `basicsr` 依赖坑（本项目绕开它的直接原因）

1. **torchvision 兼容性（致命）**：PyPI 上 `basicsr` 最后版本为 **1.4.2（2022-08-30）**（✅ PyPI API）。其 `basicsr/data/degradations.py` 顶部执行
   `from torchvision.transforms.functional_tensor import rgb_to_grayscale`；
   torchvision 0.17（2024-03）移除 `functional_tensor` 后，任何 `import basicsr` 都直接抛
   `ModuleNotFoundError: No module named 'torchvision.transforms.functional_tensor'`
   （✅ 真实报错栈见 [issue #768](https://github.com/xinntao/Real-ESRGAN/issues/768)，2024-03-17）。
   本项目 venv 为 **torchvision 0.20.1**，必然触发。
2. **作者补丁而非官方修复**：2025-01-20 作者发布 `basicsr-fixed`（PyPI 版本号仍写 1.4.2）覆盖安装（✅ [issue #891](https://github.com/xinntao/Real-ESRGAN/issues/891)、[PyPI](https://pypi.org/project/basicsr-fixed/)）；社区另有 shim 方案（把 `rgb_to_grayscale` 从 `torchvision.transforms.functional` 转导，见 [issue #859](https://github.com/xinntao/Real-ESRGAN/issues/859)）。两者都是"打补丁"，不是上游修复。
3. **`pip install realesrgan` 会扩大爆炸半径**：PyPI `realesrgan 0.3.0` 声明依赖 `basicsr>=1.4.2`、`facexlib>=0.2.5`、`gfpgan>=1.3.5`、`torch`、`torchvision`、`opencv-python` 等（✅ PyPI API）。这意味着装一个推理包会顺带装 basicsr（导入即炸），还可能触碰 venv 里的 torch/CUDA 版本。
4. **numpy 2.x**：basicsr 1.4.2 发布于 numpy 2.0（2024-06）之前，未声明 numpy 2 兼容；项目 venv 为 numpy 2.4.4，basicsr 的部分旧式写法/扩展在 numpy 2 下存在额外风险（⚠️ 未逐路径实测，但已无必要实测——我们不引入 basicsr）。
5. **结论**：只保留 `torch.load(...)["params_ema"]` + 自建 `RRDBNet`，**零 basicsr、零 gfpgan/facexlib**，同时把整条推理链固定为 torch + PIL + numpy，环境可复现。

### 3.4 tiling / 分块推理参数（✅ 源码验证）

Real-ESRGAN 官方 `inference_realesrgan.py` 默认值：

| 参数 | 默认 | 含义与建议 |
|---|---|---|
| `--tile` | `0` | 0 = 整图推理。8GB 显存下建议 256–512（4K 输出用 256 更稳）⚠️ |
| `--tile_pad` | `10` | 每块向四周额外扩 10px 再推理，推理后裁掉，**消除块间接缝**。不要设 0 |
| `--pre_pad` | `0`（CLI） | 整图四周预填充；utils 类默认 10，但 CLI 显式传 0。填 0 可避免边缘重复计算 |
| `--outscale` | `4` | 最终缩放；非原生倍率时用 LANCZOS4 二次重采样 |
| `--alpha_upsampler` | `realesrgan` | 透明图 alpha 的处理方式，见 §3.5 |

实现要点（`realesrgan/utils.py` 的 `RealESRGANer.tile_process`）：

- 按 `tile_size` 切成 `tiles_x × tiles_y` 块，每块坐标向外扩 `tile_pad`，并用 `max(0, ...)` 截断到图像边界；
- 输出按原块位置拼回，因此 **tile 越小显存越省、接缝风险越高（靠 tile_pad 抵消）**；
- 推理前按 `mod_scale = scale` 用 reflect padding 把宽高补到倍率的整数倍（`mod_pad`），结束后裁掉，避免最后一行/列出现半像素伪影；
- OOM 时官方提示就是"调小 `--tile`"（源码里的 `print`）。

**对本项目的建议**：`tile_pad=10`、`pre_pad=0`、`tile` 按输入像素自动（例如 <1MP 用 0/512，1–4MP 用 512，>4MP 用 256），并把 `tile` 暴露为 API 参数（`docs/API.md` 已定义 64..1024）。

### 3.5 alpha 通道处理（✅ 源码验证）

`RealESRGANer.enhance()` 对 RGBA 的处理：

- 分离 alpha，RGB 三通道走模型；
- `alpha_upsampler='realesrgan'`（**CLI 默认**）：把 alpha 复制成 3 通道再跑一遍模型 → **推理时间约翻倍**，且 alpha 会被"脑补"出纹理；
- `alpha_upsampler='bicubic'` 分支：`cv2.resize(..., interpolation=cv2.INTER_LINEAR)`（注意：名字叫 bicubic，实际是**双线性**）；
- 最后合并回 BGRA 输出。

**本项目取舍**：alpha 直接用 **PIL LANCZOS** 缩放（比双线性锐、比再跑一遍模型快一倍），只让 RGB 进网络；带透明通道的输入强制输出 PNG，避免 alpha 被 JPEG 丢弃。这一取舍与 `docs/API.md` 的 `format=auto` 规则一致。

### 3.6 PyTorch 版 vs ncnn-vulkan 版

| 维度 | PyTorch（官方 `inference_realesrgan.py`） | ncnn-vulkan（便携包） |
|---|---|---|
| 运行依赖 | Python + PyTorch + CUDA + basicsr（官方脚本） | 无，解压即用 ✅ |
| 包体积 | 权重 64MiB + 环境数 GB | Windows 包 43.37MiB（含模型）✅ |
| 模型格式 | `.pth`（state_dict） | `.param` + `.bin`，不通用 ✅ |
| 可用模型 | x4plus / x2plus / anime6B / general-x4v3 / animevideov3 / RealESRNet | 仅 realesrgan-x4plus、realesrnet-x4plus、realesrgan-x4plus-anime、realesr-animevideov3（✅ README） |
| 倍率 | 原生 2/4 + `--outscale` 任意（LANCZOS4 收尾） | `-s 2/3/4`（固定整数档）✅ |
| 显存控制 | `--tile` / `--tile_pad` / `--pre_pad` | `-t tile-size`（0=auto，≥32）✅ |
| 性能技巧 | fp16、channels_last、TensorRT（需改造） | `-j load:proc:save` 多线程流水线、`-g` 多卡、`-x` TTA ✅ |
| GPU 兼容 | NVIDIA CUDA（ROCm/CPU 需另配） | NVIDIA/AMD/Intel，Vulkan 即可 ✅ |
| 适合场景 | 本机服务、可编程、可扩展 | 绿色分发、客户机无 Python、快速对照验证 |

> **结论**：本期主后端用 PyTorch（可控、可扩展、与 FastAPI 同进程）；ncnn 版保留为"分发/应急后端"的候选，但需要额外的文件协议与进程管理，不进本期范围。

---

## 4. 本项目的选型理由与取舍

### 4.1 为什么 vendored RRDBNet + 官方权重

- **环境已定死**：venv 为 torch 2.5.1+cu121 / torchvision 0.20.1 / numpy 2.4.4 / Python 3.12.13（项目 `docs/CONTRACT.md` 记录）。basicsr 1.4.2 在该组合下 100% 导入失败（§3.3），而 `pip install realesrgan` 会引入 basicsr + gfpgan + facexlib。
- **代码量极小**：RRDBNet 前向 + `pixel_unshuffle/pixel_shuffle` 约 150–200 行；`load_rrdbnet` 只需 `torch.load(path, map_location="cpu")` 取 `params_ema`（回退 `params`）后 `load_state_dict(strict=True)`，可完全离线、可审计。
- **许可干净**：BSD-3 权重 + Apache-2.0 结构代码，保留声明即可商用；不引入 AGPL/GPL 组件。
- **可测试**：无 basicsr 的隐式全局副作用（日志、分布式、CUDA 初始化），单元测试可直接跑 CPU 前向。

### 4.2 为什么 x4 / x2 双模型

目标档位是 720P/1080P/2K/4K（短边或长边定值），输入可能从 480p 到 4K：

| 场景 | 所需倍率 | 选模型 | 理由 |
|---|---|---|---|
| 480p → 1080P | ≈2.25 | x4 | 倍率 >2.2，x2 不够 |
| 720p → 1080P | 1.5 | x2 | 原生 2x 再 LANCZOS 微缩，计算量约为 x4 的 1/4 |
| 1080p → 2K/4K | 1.33–2.0 | x2 | 避免 x4 过度锐化与 4 倍算力 |
| 小图 → 4K | 4–8 | x4 + LANCZOS 补足 | 只跑一次模型，剩余倍率用 LANCZOS，不做二次模型串联 |

- 经验阈值 **scale ≤ 2.2 用 x2，否则 x4**（⚠️ 工程推断，需用样张回归校准）。
- 好处：4K 输出时若用 x4 会先生成 8K 中间图（内存/耗时爆炸），双模型直接把中间图控制在 2 倍。
- 两个模型都是 64MiB 级，磁盘成本可接受；按需懒加载，切换时释放显存。

### 4.3 为什么保留 PIL 兜底

- **可用性优先**：无 CUDA（纯 CPU 机器）、权重缺失/损坏、显存 OOM 且 tile 降到下限仍失败时，任务不能整体失败；回退 `LANCZOS + UnsharpMask` 仍能交付目标尺寸。
- **契约要求**：`engine.backend` 已在 `docs/API.md` 定义为 `realesrgan | pil`，前端显示"AI 超分 / 兼容模式"徽章；兜底是产品行为的一部分，不是异常路径。
- **降级要可见**：每个文件返回 `note`（如"权重缺失，已使用 Lanczos 兼容模式"），避免用户误以为 AI 生效。

### 4.4 为什么用 LANCZOS 做"精确缩放"

- 模型原生只有 2x/4x，而目标是精确像素（含 8px 对齐）。做法是**模型出图后只做一次 LANCZOS 重采样**到目标尺寸，避免"模型 4x → 缩 0.6 → 再对齐"的多次重采样累积模糊。
- LANCZOS 在缩小/放大都保留较多高频，配合 Real-ESRGAN 输出不会明显发糊；alpha 同用 LANCZOS，保证边缘一致。
- 与官方 `--outscale` 的行为一致（官方也是 LANCZOS4 收尾，✅ 源码验证），可复现官方观感。

### 4.5 本期明确不做（取舍）

| 不做的功能 | 原因 |
|---|---|
| GFPGAN / `--face_enhance` | 额外 300MB+ 依赖与模型；对人像会"换脸感"，且 licensing/肖像风险更高 |
| 视频超分 | 需要帧抽取、音频复用、长任务队列，超出本期 Web 图片工具范围 |
| TTA（`-x`） | 约 4–8 倍耗时换极小画质提升，8GB 卡不划算 |
| 多模型自动路由（按图像内容选 anime/photo） | 需要分类器与标注，误判代价高；改为用户显式选"动漫模式"更可控（二期） |
| 云端商用 API | 与"本地离线、不上传图片"的产品前提冲突 |

### 4.6 后续升级路线（按投入产出排序）

> **升级 PyTorch 前先看**：`torch.load` 的 `weights_only` 默认值在 torch 2.6 起改为 `True`（⚠️ 上游行为变更，本项目 2.5.1 尚为 `False`）。权重只含张量字典，`weights_only=True` 理论上可加载，但升级时需实测 `params_ema` 路径并显式传参，避免"升级后权重加载失败"。

1. **ONNX Runtime 后端**（中期）：把 RRDBNet 导出 ONNX，去掉 PyTorch 运行时依赖，Windows/Linux 部署更轻；可用 DirectML/CUDA/TensorRT EP。需验证动态 shape 与 tile 推理的算子支持（⚠️ 待验证）。
2. **TensorRT**（性能）：RTX 4060 Laptop 上固定 shape 引擎通常有 1.5–2× 加速；代价是首次构建引擎耗时、shape 需分桶、驱动/版本耦合（⚠️ 经验值，需实测）。
3. **Real-CUGAN 动漫后端**（体验）：MIT、2x/3x/4x、多档降噪，作为"动漫模式"比 anime6B 更现代；需要迁移其旧版 torch 权重加载代码（其发行包基于 torch 1.x）。
4. **HAT / SwinIR 高质量慢速档**（画质上限）：仅在"高质量/离线批处理"场景启用；HAT 20.8M 参数、显存门槛高，需 tile + 可能 fp16；SwinIR real-world L 271.8MB 只适合作为可选下载。
5. **ncnn-vulkan 便携后端**（分发）：给无 Python 的客户机提供绿色包；需要进程调用、临时文件协议、退出清理，和现有 job 模型对接成本中等。

---

## 5. 可借鉴的工程经验

### 5.1 Upscayl：模型管理与产品化

- **模型与二进制解耦**：`custom-models` 仓库按 `<name>.param` + `<name>.bin` 成对分发，24 个模型覆盖 1.2–66.8MB（✅ API 实测）。本项目可沿用"模型目录 + 清单 JSON（名称/倍率/体积/sha256/适用类型）"的结构，便于后续加 Real-CUGAN/HAT。
- **模型元数据驱动 UI**：每个模型标注 scale 与适用场景，前端只展示"通用/动漫/快速"标签，不暴露文件名。
- **批处理队列与失败隔离**：Upscayl 的 batch 是"先全部超分、再统一后处理"，中断时已完成的图仍保留；对应本项目"每文件独立状态 + 失败不拖垮整批"。
- **不要复制代码**：AGPL-3.0 只可参考交互与流程设计。

### 5.2 Real-ESRGAN：tiling 的正确实现

- **tile_pad 是接缝的关键**：每块外扩 10px 再裁掉，比"无 pad 直接拼"干净一个量级；这是本项目必须移植的细节。
- **mod_pad 用 reflect 补齐倍率整数倍**，避免右/下边缘半像素错位。
- **先算目标再选模型**：官方 `--outscale` 的思路是"模型出图 → LANCZOS 收尾"，而不是反复缩放；与本项目 `plan_target` 的 `scale` 计算天然契合。
- **OOM 可恢复**：捕获 `torch.cuda.OutOfMemoryError` 后清 cache、降 tile 重试当前图（一次），仍失败再回退 PIL；比整任务失败体验好。

### 5.3 OpenCV dnn_superres：轻量兜底与 CPU 路径

- ESPCN 仅 0.1MB、FSRCNN 0.04MB，CPU 上 256×256 不到 0.01s（✅ 官方 README 数据），适合做"快速预览"或"无 torch 环境"的第二兜底。
- 但只支持整数倍率（2/3/4，LapSRN 另支持 8x），且无 tiling/alpha 处理；若引入，需要自己写分块与拼接（✅ 模块 README 与模型列表验证）。
- 结论：本期不引入（PIL 已覆盖兜底），可作为"未来去掉 torch 依赖"的备选。

### 5.4 异步任务 / 进度 / 生命周期

- **单 worker 串行队列**：GPU 只有一块，串行最稳；任务状态 `queued → running → done/failed/cancelled`，与 `docs/API.md` 一致。
- **细粒度进度**：按"文件数 + tile 数"双维度上报（如 `file 2/5, tile 7/16`），前端轮询或 SSE 都够用；取消标志在每个 tile 循环检查（Real-ESRGAN 的 tile 循环是天然检查点）。
- **原子落盘**：先写 `*.part` 再 `os.replace`，避免用户下载到半个文件；ZIP 用流式 `zipfile` 写入临时文件后原子改名。
- **缓存生命周期**：`cache/uploads`、`cache/results` 分离；停止脚本先停服务再清图片（保留 `models/` 权重，权重永不删除）；启动时按 TTL 清理残留。
- **可观测性**：每次推理记录 `model/tile/耗时/峰值显存/是否回退`，出问题时能直接定位是 OOM 还是权重问题。

### 5.5 前端与 API 细节

- 目标档位只让用户选"短边/长边定值"，另一边由服务端按宽高比计算并 8px 对齐（`docs/API.md` §5 已定义），前端展示实际输出像素，避免"选 4K 却得到 3832"的困惑。
- 结果页提供"另存为"单张 + ZIP 打包；ZIP 条目名 `原名_tier.ext`，避免重名覆盖。
- `engine.backend` 徽章 + 每文件 `note`，把"AI / 兼容模式"透明化。

---

## 6. 参考资料

> 说明：本机网络对 `raw.githubusercontent.com` 解析被拦截，以下优先使用 `github.com` 页面、GitHub Release/API、PyPI、Hugging Face 页面。所有链接在调研时（2026-10-19）可访问；`hf-mirror.com` 未验证。

**Real-ESRGAN 主仓库与代码**
1. 仓库主页（模型列表、安装、ncnn 说明）：https://github.com/xinntao/Real-ESRGAN
2. LICENSE（BSD-3-Clause）：https://github.com/xinntao/Real-ESRGAN/blob/master/LICENSE
3. 推理入口（tile/tile_pad/pre_pad/alpha 参数默认值）：https://github.com/xinntao/Real-ESRGAN/blob/master/inference_realesrgan.py
4. `RealESRGANer`（tiling、alpha、LANCZOS4 收尾实现）：https://github.com/xinntao/Real-ESRGAN/blob/master/realesrgan/utils.py
5. Release v0.1.0（x4plus 63.94MiB）：https://github.com/xinntao/Real-ESRGAN/releases/tag/v0.1.0
6. Release v0.2.1（x2plus 63.96MiB）：https://github.com/xinntao/Real-ESRGAN/releases/tag/v0.2.1
7. Release v0.2.2.4（anime_6B 17.11MiB）：https://github.com/xinntao/Real-ESRGAN/releases/tag/v0.2.2.4
8. Release v0.2.5.0（general-x4v3 4.66MiB、animevideov3、ncnn 便携包 43.37MiB）：https://github.com/xinntao/Real-ESRGAN/releases/tag/v0.2.5.0

**basicsr / torchvision 兼容问题**
9. Issue #768（`functional_tensor` ModuleNotFoundError 完整栈，2024-03-17）：https://github.com/xinntao/Real-ESRGAN/issues/768
10. Issue #859（torchvision 版本限制与 shim 方案）：https://github.com/xinntao/Real-ESRGAN/issues/859
11. Issue #891（作者发布 `basicsr-fixed`，2025-01-20）：https://github.com/xinntao/Real-ESRGAN/issues/891
12. PyPI `basicsr` 1.4.2（2022-08-30 最后版本）：https://pypi.org/project/basicsr/
13. PyPI `realesrgan` 0.3.0（依赖 basicsr/facexlib/gfpgan）：https://pypi.org/project/realesrgan/
14. PyPI `basicsr-fixed`：https://pypi.org/project/basicsr-fixed/
15. BasicSR（Apache-2.0，RRDBNet 来源）：https://github.com/XPixelGroup/BasicSR

**ncnn / 便携推理**
16. Real-ESRGAN-ncnn-vulkan（CLI 参数 `-s/-t/-g/-j/-x`）：https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan
17. waifu2x-ncnn-vulkan（MIT，2025-09-15 发行包 33.9MiB）：https://github.com/nihui/waifu2x-ncnn-vulkan
18. realcugan-ncnn-vulkan（MIT，20220728 发行包 43.8MiB）：https://github.com/nihui/realcugan-ncnn-vulkan

**动漫向**
19. waifu2x 原仓库（MIT）：https://github.com/nagadomi/waifu2x
20. Real-CUGAN（MIT，2x/3x/4x、多档降噪、低显存模式）：https://github.com/bilibili/ailab/tree/main/Real-CUGAN
21. Real-CUGAN Release（`updated_weights.zip` 51.5MiB 等）：https://github.com/bilibili/ailab/releases/tag/Real-CUGAN

**学术 SOTA**
22. SwinIR（Apache-2.0，权重与 `--tile` 用法）：https://github.com/JingyunLiang/SwinIR
23. SwinIR Releases（权重体积实测）：https://github.com/JingyunLiang/SwinIR/releases/tag/v0.0
24. HAT（Apache-2.0，HAT 20.8M/102.4G，batch=4 约 20GB，tile 模式）：https://github.com/XPixelGroup/HAT
25. DAT（Apache-2.0，DAT-S/DAT-light 参数表）：https://github.com/zhengchen1999/DAT
26. BSRGAN（Apache-2.0，盲退化 x4）：https://github.com/cszn/BSRGAN
27. ESRGAN 初代（Apache-2.0）：https://github.com/xinntao/ESRGAN

**GUI / 节点式 / 模型生态**
28. Upscayl（AGPL-3.0，Vulkan 依赖说明）：https://github.com/upscayl/upscayl
29. Upscayl 自定义模型仓库（24 个模型 1.2–66.8MB）：https://github.com/upscayl/custom-models
30. upscayl-ncnn 后端（AGPL-3.0）：https://github.com/upscayl/upscayl-ncnn
31. chaiNNer（GPL-3.0，PyTorch/ONNX/NCNN/TensorRT 支持）：https://github.com/chaiNNer-org/chaiNNer
32. OpenModelDB（社区超分模型数据库）：https://openmodeldb.info/
33. Waifu2x-Extension-GUI（Windows GUI，许可待确认）：https://github.com/AaronFeng753/Waifu2x-Extension-GUI

**OpenCV dnn_superres**
34. 模块说明与四类模型优缺点/速度（EDSR 38.5MB、ESPCN 0.1MB、FSRCNN 0.04MB、LapSRN 1–5MB）：https://github.com/opencv/opencv_contrib/tree/4.x/modules/dnn_superres
35. EDSR 模型：https://github.com/Saafke/EDSR_Tensorflow
36. ESPCN 模型：https://github.com/fannymonori/TF-ESPCN
37. FSRCNN 模型：https://github.com/Saafke/FSRCNN_Tensorflow
38. LapSRN 模型：https://github.com/fannymonori/TF-LapSRN

**备用权重源**
39. Hugging Face `amd/realesrgan-x4plus`（bsd-3-clause，含 `RealESRGAN_x4plus.pth`）：https://huggingface.co/amd/realesrgan-x4plus
40. HF 镜像站（仅候选，未验证）：https://hf-mirror.com

---

## 附：关键数字速查（均 ✅ 已验证）

| 项 | 数值 |
|---|---|
| RealESRGAN_x4plus 权重 | 67,040,989 B（63.94 MiB），HTTP 200 |
| RealESRGAN_x2plus 权重 | 67,061,725 B（63.96 MiB），HTTP 200 |
| RealESRGAN_x4plus_anime_6B | 17,938,799 B（17.11 MiB），HTTP 200 |
| realesr-general-x4v3 | 4,885,111 B（4.66 MiB），HTTP 200 |
| Real-ESRGAN ncnn Windows 包 | 43.37 MiB（v0.2.5.0） |
| waifu2x-ncnn-vulkan Windows 包 | 33.9 MiB（20250915） |
| realcugan-ncnn-vulkan Windows 包 | 43.8 MiB（20220728） |
| Upscayl Windows 安装包 | 248.8 MiB（v2.15.0） |
| HAT 参数量 / 计算量 | 20.8M / 102.4G MACs（64×64 输入） |
| SwinIR real-world L 权重 | 271.8 MiB |
| basicsr 最后版本 | 1.4.2（2022-08-30） |
| torchvision 移除 functional_tensor | 0.17（2024-03，对应 issue #768 报错时间）⚠️ 版本号由时间线推断，报错本身已验证 |
