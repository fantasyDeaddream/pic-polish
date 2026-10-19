# PicPolish API v1（前后端契约 / 单一事实来源）

服务默认监听 `http://127.0.0.1:8765`（可用环境变量 `PIC_POLISH_HOST` / `PIC_POLISH_PORT` 覆盖，
启动脚本在端口被占用时会自动顺延并打印实际地址）。

所有请求/响应均为 UTF-8 JSON（图片下载接口除外）。错误统一为 FastAPI 风格：

```json
{ "detail": "错误说明" }
```

HTTP 状态码约定：`400` 参数错误 / `404` 任务或文件不存在 / `409` 任务状态冲突 /
`413` 文件过大 / `415` 不支持的图片类型 / `500` 处理失败 / `503` 引擎不可用。

---

## 1. 目标分辨率档位（tiers）

| id | label | 基准边 rule | 目标值 value | 说明 |
|----|-------|------------|-------------|------|
| `720p`  | 720P  | `short`（短边） | 720  | 短边固定 720，长边按比例 |
| `1080p` | 1080P | `short`（短边） | 1080 | 短边固定 1080，长边按比例 |
| `2k`    | 2K    | `long`（长边）  | 2560 | 长边固定 2560，短边按比例 |
| `4k`    | 4K    | `long`（长边）  | 3840 | 长边固定 3840，短边按比例 |

### 1.1 输出尺寸计算规则（后端唯一实现，前端只展示）

输入原图 `(W, H)`，档位 `tier`，对齐粒度 `align`（默认 8）：

1. `rule == "short"` 时，基准边 = `min(W, H)`；`rule == "long"` 时，基准边 = `max(W, H)`。
2. `scale = value / 基准边`。
3. **基准边固定为目标值**（720/1080/2560/3840），另一条边：
   - 若 `W <= H`（竖图/方图）：`out_w = value`（短边档）或 `out_h = value`（长边档是 H），
     另一条边 = `round(另一条原边 * scale)`；
   - 若 `W > H`（横图）：`out_h = value`（短边档）或 `out_w = value`（长边档是 W），
     另一条边 = `round(另一条原边 * scale)`。
4. **对齐调优**：把"另一条边"吸附到最近的 `align` 倍数（`round(v/align)*align`，最小 `align`），
   偏差 ≤ `align/2` 像素；基准边本身已是 8 的整数倍（720/1080/2560/3840），不参与吸附。
   这满足"无法整除时自动调优、允许细小差异"的要求；`aligned=true` 表示发生过吸附。
5. `mode`：`scale > 1` → `upscale`；`scale < 1` → `downscale`；相等 → `same`。
6. 宽高最小值为 `align`，最大单边 16384（超限 400）。
7. 保持原始宽高比，不做裁剪、不做旋转（EXIF 方向按显示方向处理）。

示例（align=8）：

| 原尺寸 | tier | 计算 | 输出 |
|--------|------|------|------|
| 1280x720 | 1080p | 短边 720→1080，scale 1.5 | **1920x1080** |
| 1920x1080 | 2k | 长边 1920→2560，scale 1.3333 | **2560x1440** |
| 1000x667 | 1080p | 短边 667→1080，长边 1000*1080/667=1619.2 → 1619 → 吸附 1620? | 见实现：`round(1619.2)=1619`，吸附到 1616（最近的 8 倍数，偏差 3） |
| 1080x1920（竖） | 720p | 短边 W=1080→720，H=1920*720/1080=1280 | **720x1280** |
| 3840x2160 | 1080p | 短边 2160→1080，scale 0.5 | **1920x1080**（downscale，直接高质量缩放） |

> 说明：吸附规则以 `app/target.py` 的 `plan_target()` 为准；前端必须展示后端返回的
> `output.width/height`，不得自行计算。

---

## 2. 接口一览

| 方法 | 路径 | 用途 |
|------|------|------|
| GET | `/api/health` | 健康检查 + 引擎/GPU/限制信息 |
| GET | `/api/targets` | 档位定义 |
| GET | `/api/models` | 可用超分模型列表（推荐用途 / 降噪能力 / 许可，见 §14） |
| POST | `/api/plan` | 只计算输出尺寸（不上传文件，前端预览用） |
| POST | `/api/jobs` | 上传图片并创建超分任务（multipart） |
| GET | `/api/jobs/{job_id}` | 任务状态 / 进度 / 每张图结果 |
| GET | `/api/jobs/{job_id}/files/{index}` | 结果图（inline 预览） |
| GET | `/api/jobs/{job_id}/files/{index}/download` | 结果图（attachment 下载/另存为） |
| GET | `/api/jobs/{job_id}/download.zip` | 全部结果打包下载 |
| POST | `/api/jobs/{job_id}/cancel` | 取消排队/进行中的任务 |
| DELETE | `/api/jobs/{job_id}` | 删除任务及其缓存文件 |
| GET | `/` | 网页 UI（重定向到 `/web/index.html`） |
| GET | `/web/*` | 静态前端资源 |

---

## 3. `GET /api/health`

```json
{
  "status": "ok",
  "app": "PicPolish",
  "version": "1.0.0",
  "engine": {
    "backend": "realesrgan",
    "device": "cuda",
    "gpu": "NVIDIA GeForce RTX 4060 Laptop GPU",
    "model": "RealESRGAN_x4plus",
    "native_scale": 4
  },
  "models": {
    "default": "auto",
    "available": ["auto", "realesrgan-x4plus", "realesrgan-x2plus", "realesrgan-anime6b", "realcugan", "realcugan-conservative"]
  },
  "limits": { "max_files": 20, "max_file_mb": 50, "max_total_mb": 200, "max_pixels": 20000000 },
  "cache": { "jobs": 1, "result_mb": 3.2 }
}
```

`engine.backend` 取值：`realesrgan`（PyTorch + CUDA/CPU 权重推理）、`pil`（无模型兜底：
Lanczos + 轻锐化）。`device` 取值 `cuda` / `cpu`。前端据 `backend` 显示"AI 超分 / 兼容模式"徽章。

## 4. `GET /api/targets`

```json
{
  "targets": [
    {"id": "720p",  "label": "720P",  "rule": "short", "value": 720,  "hint": "短边固定 720 像素"},
    {"id": "1080p", "label": "1080P", "rule": "short", "value": 1080, "hint": "短边固定 1080 像素"},
    {"id": "2k",    "label": "2K",    "rule": "long",  "value": 2560, "hint": "长边固定 2560 像素"},
    {"id": "4k",    "label": "4K",    "rule": "long",  "value": 3840, "hint": "长边固定 3840 像素"}
  ]
}
```

## 5. `POST /api/plan`

请求 JSON：

```json
{
  "target": "1080p",
  "align": 8,
  "images": [
    {"name": "a.jpg", "width": 1280, "height": 720},
    {"name": "b.png", "width": 1080, "height": 1920}
  ]
}
```

响应：

```json
{
  "target": {"id": "1080p", "label": "1080P", "rule": "short", "value": 1080, "align": 8},
  "images": [
    {
      "index": 0, "name": "a.jpg",
      "origin": {"width": 1280, "height": 720},
      "output": {"width": 1920, "height": 1080},
      "scale": 1.5, "mode": "upscale", "aligned": false,
      "note": "短边 720 → 1080；长边 1280 → 1920"
    }
  ]
}
```

字段说明：`aligned=true` 表示另一条边做过 8 像素吸附；`note` 为中文说明，前端可直接展示。
`width/height` 必须为正整数，否则 400。

## 6. `POST /api/jobs`（multipart/form-data）

| 字段 | 类型 | 必填 | 默认 | 说明 |
|------|------|------|------|------|
| `files` | file[] | ✔ | - | 1..max_files 张图片；支持 jpg/jpeg/png/webp/bmp/tif/tiff |
| `target` | string | ✔ | - | `720p` / `1080p` / `2k` / `4k` |
| `align` | int | ✘ | 8 | 输出尺寸对齐粒度，允许 2..64 |
| `model` | string | ✘ | `auto` | 模型 id，见 §14；`auto`=按倍率自动选 Real-ESRGAN x2/x4 |
| `denoise` | bool | ✘ | false | 预处理降噪：`realesrgan-*` 走 OpenCV；`realcugan` 等价于 `denoise_level=2` |
| `denoise_level` | int | ✘ | 0 | Real-CUGAN 原生降噪等级 0..3（0=关闭；仅 `realcugan` 生效，rrdb 模型忽略） |
| `format` | string | ✘ | `auto` | `auto`/`png`/`jpg`/`webp`；`auto`=PNG 输入或带透明通道→png，其余→jpg |
| `quality` | int | ✘ | 95 | jpg/webp 质量 70..100 |
| `tile` | int | ✘ | 0 | 0=自动；否则 64..1024 的推理分块 |

响应 `202 Accepted`（任务入队）：

```json
{
  "job_id": "9f2c4a1b7d3e",
  "status": "queued",
  "created_at": "2026-10-19T14:30:00+08:00",
  "queue_position": 0,
  "target": {"id": "1080p", "label": "1080P", "rule": "short", "value": 1080, "align": 8},
  "model": {"id": "auto", "label": "智能推荐", "denoise": false, "denoise_level": 0},
  "engine": {"backend": "realesrgan", "device": "cuda", "gpu": "...", "model": "RealESRGAN_x4plus", "native_scale": 4},
  "total": 2,
  "files": [
    {
      "index": 0, "name": "a.jpg", "size": 345678,
      "original": {"width": 1280, "height": 720},
      "output": {"width": 1920, "height": 1080},
      "scale": 1.5, "mode": "upscale", "aligned": false,
      "model": "auto", "model_used": "RealESRGAN_x2plus", "denoise_level": 0,
      "status": "pending", "progress": 0,
      "preview_url": "/api/jobs/9f2c4a1b7d3e/files/0",
      "download_url": "/api/jobs/9f2c4a1b7d3e/files/0/download"
    }
  ]
}
```

校验失败（未选档位、空文件、类型不支持、超限）返回 4xx 且不创建任务。
同一时刻只允许 1 个任务占用 GPU 串行推理，后来任务进入队列。

## 7. `GET /api/jobs/{job_id}`

```json
{
  "job_id": "9f2c4a1b7d3e",
  "status": "running",
  "progress": 37.5,
  "current_index": 1,
  "total": 2,
  "created_at": "...", "started_at": "...", "finished_at": null,
  "error": null,
  "target": {...},
  "engine": {...},
  "files": [
    {
      "index": 0, "name": "a.jpg", "size": 345678,
      "original": {"width": 1280, "height": 720},
      "output": {"width": 1920, "height": 1080},
      "scale": 1.5, "mode": "upscale", "aligned": false,
      "model": "auto", "model_used": "RealESRGAN_x2plus", "denoise_level": 0,
      "status": "done", "progress": 100,
      "output_size": 812345, "output_format": "jpg",
      "note": null, "error": null,
      "preview_url": "/api/jobs/9f2c4a1b7d3e/files/0",
      "download_url": "/api/jobs/9f2c4a1b7d3e/files/0/download"
    },
    {"index": 1, "...": "status=running, progress=55.0"}
  ],
  "elapsed_sec": 12.3
}
```

- 任务 `status`：`queued` | `running` | `done` | `error` | `cancelled`。
- 文件 `status`：`pending` | `running` | `done` | `error` | `cancelled`。
- 任务级 `progress` 为全部文件加权百分比（0..100，一位小数）。
- 前端轮询间隔建议 700ms；任务进入 `done/error/cancelled` 后停止轮询。
- 任务完成前，`preview_url/download_url` 访问返回 404/409，前端只对 `status=done` 的文件启用保存。

## 8. 图片读取接口

### `GET /api/jobs/{job_id}/files/{index}`
返回结果图字节流，`Content-Type` 按实际格式（`image/png` / `image/jpeg` / `image/webp`），
`Cache-Control: no-cache`。用于 `<img src>` 内联预览与对比。

### `GET /api/jobs/{job_id}/files/{index}/download`
同上，但带 `Content-Disposition: attachment; filename="a_1080p.jpg"`（RFC 5987 中文安全）。
可选 query：
- `filename`：自定义另存文件名（服务端会 sanitize，保留真实扩展名）；
- `zip=1` 不支持（用 zip 接口）。

### `GET /api/jobs/{job_id}/download.zip`
返回 `application/zip`，文件名 `PicPolish_{job_id前8位}.zip`；条目名 `{原文件名去扩展}_{tier}.{ext}`，
重名自动加 `_2`、`_3` 后缀。

## 9. `POST /api/jobs/{job_id}/cancel`
排队中：直接标记 `cancelled`；运行中：当前文件结束后停止（正在推理的 tile 循环每轮检查取消标志）。
响应 `{"job_id": "...", "status": "cancelled"}`；已结束任务返回 409。

## 10. `DELETE /api/jobs/{job_id}`
删除任务内存记录 + `cache/uploads/<job_id>`、`cache/results/<job_id>`、`cache/tmp/<job_id>`。
响应 `{"deleted": true, "job_id": "...", "freed_bytes": 1234567}`。

> 注意：任务仍在 `running`（包括刚 `cancel` 后 worker 正在收尾当前文件）时调用 DELETE 会返回
> `409`，通常 <1 秒后即可重试成功。前端可据此提示"正在收尾，请稍候"。

---

## 11. 调用示例

### curl（Windows PowerShell 7 / WSL 均可用）

```bash
# 健康检查
curl -s http://127.0.0.1:8765/api/health

# 只算尺寸
curl -s -X POST http://127.0.0.1:8765/api/plan \
  -H "Content-Type: application/json" \
  -d '{"target":"1080p","images":[{"name":"a.jpg","width":1280,"height":720}]}'

# 创建任务（多图）
curl -s -X POST http://127.0.0.1:8765/api/jobs \
  -F "files=@samples/photo1.jpg" -F "files=@samples/photo2.jpg" \
  -F "target=1080p" -F "format=auto"

# 轮询
curl -s http://127.0.0.1:8765/api/jobs/<job_id>

# 下载结果 / 打包
curl -sO -J http://127.0.0.1:8765/api/jobs/<job_id>/files/0/download
curl -s -o out.zip http://127.0.0.1:8765/api/jobs/<job_id>/download.zip
```

### Python（requests）

```python
import time, requests
base = "http://127.0.0.1:8765"
with open("samples/photo1.jpg", "rb") as f:
    r = requests.post(f"{base}/api/jobs",
                      files=[("files", ("photo1.jpg", f, "image/jpeg"))],
                      data={"target": "2k", "format": "auto"})
job = r.json()
while True:
    st = requests.get(f"{base}/api/jobs/{job['job_id']}").json()
    if st["status"] in ("done", "error", "cancelled"):
        break
    time.sleep(0.7)
out = requests.get(f"{base}/api/jobs/{job['job_id']}/files/0/download")
open("out.jpg", "wb").write(out.content)
```

---

## 12. 缓存目录与生命周期（启动/终止脚本据此清理）

```
cache/
  uploads/<job_id>/<index>_<safe_name>    # 上传原图
  results/<job_id>/<stem>_<tier>.<ext>    # 超分结果
  tmp/<job_id>/                           # 中间文件
.runtime/server.json                      # {pid, host, port, url, started_at, engine, log}
logs/server.log / logs/server.err.log     # 服务日志
models/RealESRGAN_x4plus.pth              # 权重（永不删除；下同）
models/RealESRGAN_x2plus.pth
models/RealESRGAN_x4plus_anime_6B.pth
models/realcugan/up{2,3,4}x-latest-*.pth  # Real-CUGAN v3 多档降噪权重
```

- 任务默认保留 24h（`PIC_POLISH_KEEP_HOURS`），过期由后台清理线程删除。
- **终止脚本必须清空 `cache/uploads`、`cache/results`、`cache/tmp` 下所有文件**（保留目录），
  防止持续占用磁盘；服务正常退出时也会清理。
- 停止脚本是幂等的：服务未运行也执行缓存清理并返回 0。

## 13. 限制与错误信息（前端需可读展示）

| 场景 | HTTP | detail 示例 |
|------|------|-------------|
| 档位非法 | 400 | `未知的 target：foo（可选 720p/1080p/2k/4k）` |
| 非图片 | 415 | `不支持的文件类型：a.txt` |
| 单文件超限 | 413 | `单个文件超过 50MB 限制` |
| 文件数超限 | 400 | `一次最多 20 张图片` |
| 图片损坏 | 400 | `无法解析图片：a.jpg` |
| 引擎不可用 | 503 | `超分引擎初始化失败：...` |
| 任务不存在 | 404 | `任务不存在：xxx` |
| 结果未就绪 | 409 | `结果尚未生成` |
| 模型 id 非法 | 400 | `未知的 model：foo（可选 auto/realesrgan-x4plus/...）` |
| 请求的模型权重缺失 | 400 | `模型 realcugan 的权重缺失，请运行 scripts/fetch_models.py 下载` |

---

## 14. 模型与降噪（v1.1 新增，向后兼容）

### 14.1 `GET /api/models`

响应（顺序固定；`available=false` 表示本地权重缺失，可运行 `scripts/fetch_models.py` 下载；
该接口不加载任何模型，秒回）：

```json
{
  "default": "auto",
  "models": [
    {
      "id": "auto", "label": "智能推荐", "family": "rrdb", "native_scales": [2, 4],
      "description": "按目标倍率自动选择 Real-ESRGAN x2plus / x4plus",
      "tags": ["通用"], "recommended_for": ["不确定选哪个时"],
      "denoise": {"supported": true, "native": false, "levels": [0, 1], "default": 0},
      "license": "BSD-3-Clause", "size_mb": 127.9, "available": true,
      "source": "https://github.com/xinntao/Real-ESRGAN"
    },
    {"id": "realesrgan-x4plus", "label": "Real-ESRGAN 通用照片", "...": "..."},
    {"id": "realesrgan-x2plus", "label": "Real-ESRGAN 低倍率", "...": "..."},
    {"id": "realesrgan-anime6b", "label": "Real-ESRGAN 动漫", "...": "..."},
    {"id": "realcugan", "label": "Real-CUGAN（动漫/视频）", "...": "..."},
    {"id": "realcugan-conservative", "label": "Real-CUGAN 保守", "...": "..."}
  ]
}
```

### 14.2 模型推荐

| 场景 | 推荐模型 |
|------|----------|
| 照片、风景、人像 | `auto` / `realesrgan-x4plus` |
| 只需要 ≤2 倍放大 | `realesrgan-x2plus` |
| 动漫截图、插画、二次元 | `realesrgan-anime6b` 或 `realcugan` |
| 动漫/视频截图、噪点明显、需要降噪 | `realcugan`（降噪等级 1~3） |
| 不想过度锐化/风格化 | `realcugan-conservative` |

### 14.3 降噪语义

- `realesrgan-*` / `auto`：`denoise=true` 时用 OpenCV `fastNlMeansDenoisingColored` 预处理；
  `denoise_level` 仅允许 0/1；未安装 opencv 时忽略并在 `files[i].note` 说明。
- `realcugan`：`denoise_level` 0..3 选择原生降噪权重：
  2x → no-denoise / denoise1x / denoise2x / denoise3x；3x/4x → no-denoise（0）或 denoise3x（1..3）。
  `denoise=true` 且未给等级时等价于 `denoise_level=2`。
- `realcugan-conservative`：固定 conservative 权重，不接受降噪等级。

### 14.4 任务与结果中的模型字段

- 请求：`model`、`denoise`、`denoise_level`（见 §6）。
- 任务响应新增：`"model": {"id", "label", "denoise", "denoise_level"}`。
- 文件响应新增：`"model": "auto"`（请求值）、`"model_used": "RealESRGAN_x2plus"`（实际解析）、
  `"denoise_level": 0`。
- 请求的模型权重缺失 → 400（detail 提示运行 `scripts/fetch_models.py`）；`auto` 在权重缺失时
  自动降级为可用模型或 `pil` 引擎，并在 `note` 中说明。

### 14.5 许可（分发注意）

| 组件 | 许可 |
|------|------|
| Real-ESRGAN 代码/权重 | BSD-3-Clause |
| BasicSR（RRDBNet 结构来源） | Apache-2.0 |
| Real-CUGAN 代码/权重 | MIT（Copyright (c) 2022 bilibili） |
| 本项目代码 | MIT |

权重不随代码分发；首次启动由 `scripts/fetch_models.py` 下载（也支持手动放置）。
商用/再分发前请阅读 `THIRD_PARTY_NOTICES.md` 与各上游 LICENSE。
