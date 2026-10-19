/* ==========================================================================
 * PicPolish — web frontend logic (vanilla JS, no dependencies, offline ready)
 *
 * Talks to the local FastAPI service documented in docs/API.md:
 *   GET    /api/health                    引擎/限制信息
 *   GET    /api/models                    超分模型列表（推荐用途 / 降噪能力 / 许可）
 *   POST   /api/plan                      目标尺寸预览
 *   POST   /api/jobs                      multipart 上传并创建任务（含 model/denoise）
 *   GET    /api/jobs/{id}                 任务进度（700ms 轮询）
 *   POST   /api/jobs/{id}/cancel          取消任务
 *   DELETE /api/jobs/{id}                 删除任务 + 清理缓存
 *   GET    /api/jobs/{id}/files/{i}       结果预览
 *   GET    /api/jobs/{id}/files/{i}/download
 *   GET    /api/jobs/{id}/download.zip
 * ========================================================================== */
(function () {
  'use strict';

  /* ------------------------------------------------------------------ */
  /* Constants                                                           */
  /* ------------------------------------------------------------------ */

  var API = '/api';
  var ALIGN = 8;
  var POLL_INTERVAL_MS = 700;
  var PLAN_DEBOUNCE_MS = 180;
  var MAX_POLL_ERRORS = 5;

  var DEFAULT_LIMITS = { max_files: 20, max_file_mb: 50, max_total_mb: 200, max_pixels: 20000000 };

  var TIERS = [
    { id: '720p', label: '720P', rule: 'short', value: 720 },
    { id: '1080p', label: '1080P', rule: 'short', value: 1080 },
    { id: '2k', label: '2K', rule: 'long', value: 2560 },
    { id: '4k', label: '4K', rule: 'long', value: 3840 }
  ];
  var TIER_MAP = {};
  TIERS.forEach(function (tier) { TIER_MAP[tier.id] = tier; });

  var ALLOWED_EXT = ['jpg', 'jpeg', 'png', 'webp', 'bmp', 'tif', 'tiff'];
  var MIME_BY_EXT = {
    jpg: 'image/jpeg',
    jpeg: 'image/jpeg',
    png: 'image/png',
    webp: 'image/webp',
    bmp: 'image/bmp',
    tif: 'image/tiff',
    tiff: 'image/tiff'
  };

  var FILE_STATUS_TEXT = {
    pending: '等待中',
    running: '处理中',
    done: '已完成',
    error: '失败',
    cancelled: '已取消'
  };
  var JOB_STATUS_TEXT = {
    queued: '排队中',
    running: '处理中',
    done: '已完成',
    error: '失败',
    cancelled: '已取消'
  };
  var MODE_TEXT = { upscale: '放大', downscale: '缩小', same: '保持原尺寸' };

  var DENOISE_LEVEL_LABELS = { 0: '0 关闭', 1: '1 轻度', 2: '2 中度', 3: '3 强力' };

  /* /api/models 加载失败时使用的兜底项（仅 auto，保证页面仍可提交） */
  var FALLBACK_AUTO_MODEL = {
    id: 'auto',
    label: '智能推荐',
    family: 'rrdb',
    native_scales: [2, 4],
    description: '按目标倍率自动选择 Real-ESRGAN x2plus / x4plus',
    tags: ['通用', '默认'],
    recommended_for: ['不确定选哪个时', '照片 / 人像'],
    denoise: { supported: true, native: false, levels: [0, 1], default: 0 },
    license: 'BSD-3-Clause',
    size_mb: 127.9,
    available: true
  };

  var ELEMENT_IDS = [
    'engineBadge', 'engineLabel', 'engineDetail',
    'tierGrid',
    'modelGrid', 'modelStatus', 'denoisePanel', 'denoiseCheckboxRow', 'denoiseCheckbox',
    'denoiseLevelRow', 'denoiseLevelLabel', 'denoiseLevelGroup', 'denoiseFixedRow', 'denoiseFixedHint',
    'uploadHint', 'dropzone', 'fileInput', 'fileListHead', 'fileCount', 'planStatus',
    'clearFilesBtn', 'fileList', 'startBar', 'startSummary', 'startBtn',
    'progressPanel', 'cancelBtn', 'jobConfigText', 'jobStatusText', 'totalProgressBar',
    'totalProgressFill', 'totalProgressText', 'progressList',
    'resultPanel', 'resultSummary', 'resultGrid', 'resultNotices', 'resultFooter',
    'zipBtn', 'deleteJobBtn', 'toastStack'
  ];

  /* ------------------------------------------------------------------ */
  /* State                                                               */
  /* ------------------------------------------------------------------ */

  var state = {
    tier: '1080p',
    limits: Object.assign({}, DEFAULT_LIMITS),
    health: null,
    models: [],
    modelsLoaded: false,
    modelsFallback: false,
    modelsRequest: 0,
    selectedModel: 'auto',
    denoise: false,
    denoiseLevel: 0,
    files: [],
    nextUid: 1,
    healthRequest: 0,
    planTimer: null,
    planAbort: null,
    planSeq: 0,
    submitting: false,
    job: null,
    jobId: null,
    polling: false,
    pollTimer: null,
    pollErrors: 0,
    cancelling: false
  };

  var el = {};

  function ApiError(message, status) {
    this.name = 'ApiError';
    this.message = message;
    this.status = status || 0;
    if (Error.captureStackTrace) Error.captureStackTrace(this, ApiError);
  }
  ApiError.prototype = Object.create(Error.prototype);
  ApiError.prototype.constructor = ApiError;

  /* ------------------------------------------------------------------ */
  /* Small DOM / format helpers                                          */
  /* ------------------------------------------------------------------ */

  function h(tag, attrs) {
    var node = document.createElement(tag);
    if (attrs) {
      Object.keys(attrs).forEach(function (key) {
        var value = attrs[key];
        if (value === null || value === undefined || value === false) return;
        if (key === 'class') { node.className = value; return; }
        if (key === 'text') { node.textContent = value; return; }
        if (key === 'style' && typeof value === 'object') { Object.assign(node.style, value); return; }
        if (key.slice(0, 2) === 'on' && typeof value === 'function') {
          node.addEventListener(key.slice(2).toLowerCase(), value);
          return;
        }
        node.setAttribute(key, value === true ? '' : String(value));
      });
    }
    var children = Array.prototype.slice.call(arguments, 2);
    appendChildren(node, children);
    return node;
  }

  function appendChildren(node, children) {
    children.forEach(function (child) {
      if (child === null || child === undefined || child === false) return;
      if (Array.isArray(child)) { appendChildren(node, child); return; }
      node.append(child instanceof Node ? child : document.createTextNode(String(child)));
    });
  }

  function clamp(value, min, max) {
    var number = Number(value);
    if (!Number.isFinite(number)) return min;
    return Math.min(max, Math.max(min, number));
  }

  function formatBytes(bytes) {
    var number = Number(bytes);
    if (!Number.isFinite(number) || number < 0) return '—';
    if (number < 1024) return number + ' B';
    var units = ['KB', 'MB', 'GB'];
    var value = number / 1024;
    var index = 0;
    while (value >= 1024 && index < units.length - 1) {
      value /= 1024;
      index += 1;
    }
    return (value >= 100 ? value.toFixed(0) : value.toFixed(1)) + ' ' + units[index];
  }

  function formatPercent(value) {
    var number = clamp(value, 0, 100);
    var rounded = Math.round(number * 10) / 10;
    return (Number.isInteger(rounded) ? String(rounded) : rounded.toFixed(1)) + '%';
  }

  function formatScale(scale) {
    var number = Number(scale);
    if (!Number.isFinite(number)) return '';
    return '×' + number.toFixed(2);
  }

  function formatDimensions(width, height) {
    var w = Number(width);
    var ht = Number(height);
    if (!Number.isFinite(w) || !Number.isFinite(ht) || w <= 0 || ht <= 0) return '未知尺寸';
    return w + ' × ' + ht;
  }

  function formatDuration(seconds) {
    var value = Number(seconds);
    if (!Number.isFinite(value) || value < 0) return '';
    if (value < 60) return (Math.round(value * 10) / 10).toFixed(1) + ' 秒';
    return Math.floor(value / 60) + ' 分 ' + Math.round(value % 60) + ' 秒';
  }

  function extOf(name) {
    var match = /\.([^./\\]+)$/.exec(String(name || ''));
    return match ? match[1].toLowerCase() : '';
  }

  function stemOf(name) {
    var base = String(name || '');
    var index = base.lastIndexOf('.');
    if (index > 0) base = base.slice(0, index);
    return base || 'image';
  }

  function extLabel(name) {
    var ext = extOf(name);
    return ext ? ext.toUpperCase() : 'IMG';
  }

  function mimeForExt(ext) {
    return MIME_BY_EXT[ext] || 'application/octet-stream';
  }

  function isActiveStatus(status) {
    return status === 'queued' || status === 'running';
  }

  function isBusy() {
    return state.submitting || Boolean(state.job && isActiveStatus(state.job.status));
  }

  function detailToText(detail) {
    if (detail === null || detail === undefined) return '';
    if (typeof detail === 'string') return detail;
    if (Array.isArray(detail)) {
      return detail.map(function (item) {
        if (item && typeof item === 'object') return item.msg || item.message || JSON.stringify(item);
        return String(item);
      }).join('；');
    }
    if (typeof detail === 'object') return detail.message || detail.msg || JSON.stringify(detail);
    return String(detail);
  }

  function extractDetail(payload) {
    if (payload && typeof payload === 'object' && 'detail' in payload) return detailToText(payload.detail);
    return '';
  }

  /* ------------------------------------------------------------------ */
  /* Network                                                             */
  /* ------------------------------------------------------------------ */

  function apiFetch(path, options) {
    var opts = options || {};
    var method = opts.method || 'GET';
    var body = opts.body || null;
    var headers = opts.headers || null;
    var timeout = Number(opts.timeout) > 0 ? Number(opts.timeout) : 15000;
    var externalSignal = opts.signal || null;

    var controller = new AbortController();
    var timedOut = false;
    var timer = window.setTimeout(function () {
      timedOut = true;
      controller.abort();
    }, timeout);

    var onExternalAbort = null;
    if (externalSignal) {
      if (externalSignal.aborted) {
        window.clearTimeout(timer);
        return Promise.reject(new ApiError('请求已取消', 0));
      }
      onExternalAbort = function () { controller.abort(); };
      externalSignal.addEventListener('abort', onExternalAbort, { once: true });
    }

    return fetch(path, {
      method: method,
      body: body,
      headers: headers || undefined,
      signal: controller.signal,
      cache: 'no-store'
    }).then(function (response) {
      window.clearTimeout(timer);
      if (externalSignal && onExternalAbort) externalSignal.removeEventListener('abort', onExternalAbort);
      if (!response.ok) {
        return response.text().then(function (text) {
          var detail = '';
          try {
            detail = extractDetail(JSON.parse(text));
          } catch (ignored) {
            detail = '';
          }
          throw new ApiError(detail || '请求失败（HTTP ' + response.status + '）', response.status);
        });
      }
      var contentType = response.headers.get('content-type') || '';
      if (contentType.indexOf('application/json') !== -1) {
        return response.json().catch(function () {
          throw new ApiError('服务响应格式异常（JSON 解析失败）', 0);
        });
      }
      return response;
    }, function (error) {
      window.clearTimeout(timer);
      if (externalSignal && onExternalAbort) externalSignal.removeEventListener('abort', onExternalAbort);
      if (timedOut) throw new ApiError('请求超时，请检查本地服务是否正常', 0);
      if (externalSignal && externalSignal.aborted) throw new ApiError('请求已取消', 0);
      if (error && error.name === 'AbortError') throw new ApiError('请求已取消', 0);
      throw new ApiError('无法连接到本地服务，请确认 PicPolish 服务已启动', 0);
    });
  }

  /* ------------------------------------------------------------------ */
  /* Toasts                                                             */
  /* ------------------------------------------------------------------ */

  function toast(message, variant, duration) {
    if (!el.toastStack) return;
    var kind = variant || 'error';
    var timeout = Number(duration) > 0 ? Number(duration) : 5600;
    while (el.toastStack.children.length >= 4) {
      el.toastStack.removeChild(el.toastStack.firstElementChild);
    }
    var item = document.createElement('div');
    item.className = 'toast toast--' + kind;
    item.setAttribute('role', kind === 'error' ? 'alert' : 'status');

    var text = document.createElement('p');
    text.className = 'toast-text';
    text.textContent = String(message);

    var close = document.createElement('button');
    close.type = 'button';
    close.className = 'toast-close';
    close.setAttribute('aria-label', '关闭提示');
    close.textContent = '×';

    var timer = 0;
    function dismiss() {
      window.clearTimeout(timer);
      if (!item.parentNode) return;
      item.classList.add('is-leaving');
      window.setTimeout(function () {
        if (item.parentNode) item.parentNode.removeChild(item);
      }, 190);
    }
    close.addEventListener('click', dismiss);
    timer = window.setTimeout(dismiss, timeout);
    item.addEventListener('mouseenter', function () { window.clearTimeout(timer); });
    item.addEventListener('mouseleave', function () {
      window.clearTimeout(timer);
      timer = window.setTimeout(dismiss, 2200);
    });

    item.append(text, close);
    el.toastStack.append(item);
  }

  /* ------------------------------------------------------------------ */
  /* Engine / health                                                     */
  /* ------------------------------------------------------------------ */

  function setEngineBadge(kind, label, detail, title) {
    if (!el.engineBadge) return;
    el.engineBadge.classList.remove('is-loading', 'is-ai', 'is-compat', 'is-error');
    el.engineBadge.classList.add(kind === 'ai' ? 'is-ai' : kind === 'compat' ? 'is-compat' : kind === 'error' ? 'is-error' : 'is-loading');
    if (el.engineLabel) el.engineLabel.textContent = label;
    if (el.engineDetail) el.engineDetail.textContent = detail || '';
    el.engineBadge.title = title || '点击重新检测引擎状态';
  }

  function renderEngine(engine) {
    if (!engine || typeof engine !== 'object') {
      setEngineBadge('error', '引擎状态未知', '', '点击重新检测引擎状态');
      return;
    }
    var backend = String(engine.backend || '');
    if (backend === 'realesrgan') {
      var device = engine.gpu
        ? String(engine.gpu).replace(/^NVIDIA\s+/i, '')
        : (engine.device === 'cuda' ? 'CUDA GPU' : 'CPU');
      setEngineBadge(
        'ai',
        'AI 超分',
        '· ' + device,
        'Real-ESRGAN 引擎' + (engine.model ? '（' + engine.model + '）' : '') +
          ' · ' + (engine.device === 'cuda' ? 'CUDA' : 'CPU') + ' · 点击重新检测'
      );
    } else if (backend === 'pil') {
      setEngineBadge(
        'compat',
        '兼容模式',
        '· Lanczos 缩放',
        '未加载 AI 模型，使用高质量 Lanczos 缩放兜底 · 点击重新检测'
      );
    } else {
      setEngineBadge('error', '引擎状态未知', '', '点击重新检测引擎状态');
    }
  }

  function pickLimits(raw) {
    var limits = {};
    ['max_files', 'max_file_mb', 'max_total_mb', 'max_pixels'].forEach(function (key) {
      var value = Number(raw && raw[key]);
      if (Number.isFinite(value) && value > 0) limits[key] = Math.floor(value);
    });
    return limits;
  }

  function loadHealth(options) {
    var opts = options || {};
    var requestId = ++state.healthRequest;
    if (el.engineBadge) {
      el.engineBadge.disabled = true;
      el.engineBadge.classList.add('is-loading');
    }
    if (el.engineLabel) el.engineLabel.textContent = '检测引擎中…';
    if (el.engineDetail) el.engineDetail.textContent = '';

    return apiFetch(API + '/health', { timeout: 10000 }).then(function (data) {
      if (requestId !== state.healthRequest) return null;
      state.health = data || {};
      var limits = pickLimits(data && data.limits);
      state.limits = Object.assign({}, DEFAULT_LIMITS, limits);
      renderEngine(data && data.engine);
      updateUploadHint();
      updateStartBar();
      return data;
    }).catch(function (error) {
      if (requestId !== state.healthRequest) return null;
      setEngineBadge('error', '服务未连接', '', '无法连接本地服务，点击重试');
      if (!opts.silent) toast('无法获取引擎状态：' + error.message, 'error', 6500);
      return null;
    }).then(function (data) {
      if (requestId === state.healthRequest) {
        if (el.engineBadge) el.engineBadge.disabled = false;
      }
      return data;
    });
  }

  function updateUploadHint() {
    if (!el.uploadHint) return;
    var limits = state.limits;
    el.uploadHint.textContent = '支持 JPG / PNG / WebP / BMP / TIFF，单张 ≤ ' + limits.max_file_mb +
      'MB，合计 ≤ ' + limits.max_total_mb + 'MB，一次最多 ' + limits.max_files + ' 张';
  }

  /* ------------------------------------------------------------------ */
  /* Tier selection                                                      */
  /* ------------------------------------------------------------------ */

  function selectTier(tierId, options) {
    var tier = TIER_MAP[tierId];
    if (!tier) return;
    var opts = options || {};
    var changed = state.tier !== tierId;
    state.tier = tierId;
    if (el.tierGrid) {
      Array.prototype.forEach.call(el.tierGrid.querySelectorAll('.tier-card'), function (card) {
        var active = card.getAttribute('data-tier') === tierId;
        card.classList.toggle('is-active', active);
        card.setAttribute('aria-checked', active ? 'true' : 'false');
        card.tabIndex = active ? 0 : -1;
      });
    }
    if (changed || opts.force) {
      if (opts.immediate) void refreshPlan();
      else schedulePlanRefresh();
    }
    updateStartBar();
  }

  function initTierCards() {
    if (!el.tierGrid) return;
    el.tierGrid.addEventListener('click', function (event) {
      var card = event.target.closest ? event.target.closest('.tier-card') : null;
      if (!card) return;
      if (isBusy()) {
        toast('任务处理中，完成或取消后才能切换档位', 'warn');
        return;
      }
      selectTier(card.getAttribute('data-tier'));
    });
    el.tierGrid.addEventListener('keydown', function (event) {
      var keys = ['ArrowRight', 'ArrowLeft', 'ArrowDown', 'ArrowUp', 'Home', 'End'];
      if (keys.indexOf(event.key) === -1) return;
      event.preventDefault();
      var cards = Array.prototype.slice.call(el.tierGrid.querySelectorAll('.tier-card'));
      if (!cards.length) return;
      var current = cards.findIndex(function (card) {
        return card.getAttribute('data-tier') === state.tier;
      });
      var next = current < 0 ? 0 : current;
      if (event.key === 'ArrowRight' || event.key === 'ArrowDown') next = (current + 1 + cards.length) % cards.length;
      else if (event.key === 'ArrowLeft' || event.key === 'ArrowUp') next = (current - 1 + cards.length) % cards.length;
      else if (event.key === 'Home') next = 0;
      else if (event.key === 'End') next = cards.length - 1;
      var target = cards[next];
      if (target) {
        selectTier(target.getAttribute('data-tier'));
        target.focus();
      }
    });
    selectTier(state.tier, { force: true, immediate: true });
  }

  /* ------------------------------------------------------------------ */
  /* Model selection + denoise (GET /api/models, API.md §14)             */
  /* ------------------------------------------------------------------ */

  function modelById(modelId) {
    var found = null;
    state.models.forEach(function (model) {
      if (model && model.id === modelId) found = model;
    });
    return found;
  }

  function modelLabelById(modelId) {
    var model = modelById(modelId);
    return model && model.label ? model.label : '';
  }

  function currentModel() {
    var model = modelById(state.selectedModel);
    if (model) return model;
    if (state.models.length) return state.models[0];
    return FALLBACK_AUTO_MODEL;
  }

  /* auto 由服务端自动降级，权重缺失时也应保持可选；其余模型缺权重则禁用 */
  function isModelSelectable(model) {
    if (!model) return false;
    if (model.id === 'auto') return true;
    return model.available !== false;
  }

  function modelDenoiseInfo(model) {
    var info = (model && model.denoise) || {};
    var levels = Array.isArray(info.levels) && info.levels.length
      ? info.levels.map(Number)
      : [0];
    return {
      supported: info.supported === true,
      native: info.native === true,
      levels: levels,
      defaultLevel: Number.isFinite(Number(info.default)) ? Number(info.default) : levels[0]
    };
  }

  function renderModelStatus(text, variant) {
    if (!el.modelStatus) return;
    el.modelStatus.textContent = text || '';
    el.modelStatus.classList.remove('is-pending', 'is-error');
    if (variant) el.modelStatus.classList.add('is-' + variant);
  }

  function buildModelCard(model) {
    var selected = model.id === state.selectedModel;
    var unavailable = model.available === false;
    var locked = unavailable && model.id !== 'auto';
    var tags = Array.isArray(model.tags) ? model.tags : [];
    var recommended = Array.isArray(model.recommended_for) ? model.recommended_for : [];
    var metaParts = [];
    if (Number.isFinite(Number(model.size_mb)) && Number(model.size_mb) > 0) {
      metaParts.push((Math.round(Number(model.size_mb) * 10) / 10).toFixed(1) + ' MB');
    }
    if (model.license) metaParts.push(String(model.license));

    var children = [
      h('span', { class: 'model-card-top' },
        h('span', { class: 'model-label', text: model.label || model.id }),
        h('span', { class: 'model-badges' },
          model.id === 'auto' ? h('span', { class: 'model-badge', text: '默认推荐' }) : null,
          unavailable ? h('span', { class: 'model-badge is-warn', text: '未下载' }) : null
        )
      )
    ];
    if (model.description) {
      children.push(h('span', { class: 'model-desc', text: model.description }));
    }
    if (tags.length) {
      children.push(h('span', { class: 'model-tags' }, tags.map(function (tag) {
        return h('span', { class: 'model-tag', text: String(tag) });
      })));
    }
    if (recommended.length) {
      children.push(h('span', { class: 'model-reco', text: '适合：' + recommended.join('、') }));
    }
    if (metaParts.length) {
      children.push(h('span', { class: 'model-meta', text: metaParts.join(' · ') }));
    }
    if (unavailable) {
      children.push(h('span', {
        class: 'model-missing',
        text: model.id === 'auto'
          ? '权重缺失：将自动降级为可用模型 / 兼容模式；可运行 scripts/fetch_models.py 下载'
          : '权重缺失：运行 scripts/fetch_models.py 下载'
      }));
    }

    return h('button', {
      class: 'model-card' + (selected ? ' is-active' : '') + (unavailable ? ' is-unavailable' : ''),
      type: 'button',
      role: 'radio',
      'aria-checked': selected ? 'true' : 'false',
      'data-model': model.id,
      'data-unavailable': locked ? '1' : '0',
      disabled: locked || isBusy(),
      onClick: function () { chooseModel(model.id); }
    }, children);
  }

  function renderModelCards() {
    if (!el.modelGrid) return;
    el.modelGrid.replaceChildren();
    state.models.forEach(function (model) {
      el.modelGrid.append(buildModelCard(model));
    });
    syncBusyState();
  }

  function chooseModel(modelId) {
    if (isBusy()) {
      toast('任务处理中，完成或取消后才能切换模型', 'warn');
      return;
    }
    var target = modelById(modelId);
    if (!target) return;
    if (!isModelSelectable(target)) {
      toast('模型「' + (target.label || target.id) + '」权重缺失，请先运行 scripts/fetch_models.py 下载', 'error', 6500);
      return;
    }
    if (state.selectedModel === modelId) return;
    state.selectedModel = modelId;
    renderModelCards();
    updateDenoiseUI();
    updateStartBar();
  }

  function renderDenoiseLevels(model) {
    if (!el.denoiseLevelGroup) return;
    var info = modelDenoiseInfo(model);
    var levels = info.levels;
    if (levels.indexOf(Number(state.denoiseLevel)) === -1) {
      state.denoiseLevel = levels.indexOf(info.defaultLevel) !== -1 ? info.defaultLevel : levels[0];
    }
    el.denoiseLevelGroup.replaceChildren();
    levels.forEach(function (level) {
      var value = Number(level);
      var active = value === Number(state.denoiseLevel);
      el.denoiseLevelGroup.append(h('button', {
        class: 'seg-btn' + (active ? ' is-active' : ''),
        type: 'button',
        role: 'radio',
        'aria-checked': active ? 'true' : 'false',
        'data-level': String(value),
        text: DENOISE_LEVEL_LABELS[value] || String(value),
        onClick: function () { chooseDenoiseLevel(value); }
      }));
    });
  }

  function chooseDenoiseLevel(level) {
    if (isBusy()) return;
    var info = modelDenoiseInfo(currentModel());
    if (!info.native || info.levels.indexOf(Number(level)) === -1) return;
    state.denoiseLevel = Number(level);
    updateDenoiseUI();
    updateStartBar();
  }

  function isConservativeModel(model) {
    return Boolean(model && model.id === 'realcugan-conservative');
  }

  function denoiseMode(model) {
    /* 'level' = Real-CUGAN 原生降噪等级；'checkbox' = OpenCV 预处理；'fixed' = 固定策略 */
    var info = modelDenoiseInfo(model);
    if (isConservativeModel(model)) return 'fixed';
    if (info.native) {
      var adjustable = info.levels.filter(function (level) { return Number(level) > 0; });
      return adjustable.length ? 'level' : 'fixed';
    }
    if (info.supported) return 'checkbox';
    return 'fixed';
  }

  function updateDenoiseUI() {
    var model = currentModel();
    var mode = denoiseMode(model);
    if (el.denoiseCheckboxRow) el.denoiseCheckboxRow.hidden = true;
    if (el.denoiseLevelRow) el.denoiseLevelRow.hidden = true;
    if (el.denoiseFixedRow) el.denoiseFixedRow.hidden = true;

    if (mode === 'level') {
      /* Real-CUGAN：降噪等级 0..3（原生降噪权重） */
      renderDenoiseLevels(model);
      if (el.denoiseLevelRow) el.denoiseLevelRow.hidden = false;
      if (el.denoiseLevelLabel) el.denoiseLevelLabel.textContent = '降噪等级';
    } else if (mode === 'checkbox') {
      /* Real-ESRGAN 系：OpenCV 预处理降噪复选框 */
      if (el.denoiseCheckbox) el.denoiseCheckbox.checked = Boolean(state.denoise);
      if (el.denoiseCheckboxRow) el.denoiseCheckboxRow.hidden = false;
    } else {
      if (el.denoiseFixedHint) {
        el.denoiseFixedHint.textContent = isConservativeModel(model)
          ? '保守模型使用固定策略（固定 conservative 权重），不提供降噪等级。'
          : '该模型不支持可调降噪设置。';
      }
      if (el.denoiseFixedRow) el.denoiseFixedRow.hidden = false;
    }
    syncBusyState();
  }

  function submissionOptions() {
    var model = currentModel();
    var info = modelDenoiseInfo(model);
    var mode = denoiseMode(model);
    var modelId = model && model.id ? model.id : 'auto';
    var denoise = false;
    var denoiseLevel = 0;
    if (mode === 'level') {
      var level = Number(state.denoiseLevel);
      denoiseLevel = info.levels.indexOf(level) !== -1 ? level : info.defaultLevel;
    } else if (mode === 'checkbox') {
      denoise = Boolean(state.denoise);
      denoiseLevel = denoise ? 1 : 0;
    }
    return { model: modelId, denoise: denoise, denoiseLevel: denoiseLevel };
  }

  function denoiseTextFor(denoise, denoiseLevel, model) {
    var mode = denoiseMode(model);
    if (mode === 'level') {
      var level = Number(denoiseLevel);
      return level > 0 ? '降噪等级 ' + level : '降噪关闭';
    }
    if (mode === 'checkbox') return denoise ? 'OpenCV 降噪' : '无降噪';
    return '固定保守策略';
  }

  function submissionSummary() {
    var model = currentModel();
    var options = submissionOptions();
    var label = model && model.label ? model.label : options.model;
    return { label: label, denoiseText: denoiseTextFor(options.denoise, options.denoiseLevel, model) };
  }

  function loadModels() {
    var requestId = ++state.modelsRequest;
    renderModelStatus('正在加载模型列表…', 'pending');
    return apiFetch(API + '/models', { timeout: 12000 }).then(function (data) {
      if (requestId !== state.modelsRequest) return;
      var models = data && Array.isArray(data.models) ? data.models.filter(function (model) {
        return model && typeof model === 'object' && model.id;
      }) : [];
      if (!models.length) throw new ApiError('模型列表为空', 0);
      state.models = models;
      state.modelsLoaded = true;
      state.modelsFallback = false;
      var defaultId = data && data.default ? String(data.default) : 'auto';
      var preferred = modelById(defaultId);
      if (!preferred || !isModelSelectable(preferred)) {
        preferred = models.filter(isModelSelectable)[0] || models[0];
      }
      state.selectedModel = preferred.id;
      renderModelCards();
      updateDenoiseUI();
      updateStartBar();
      var availableCount = models.filter(function (model) { return model.available !== false; }).length;
      if (availableCount < models.length) {
        renderModelStatus('已加载 ' + availableCount + ' / ' + models.length + ' 个可用模型', '');
      } else {
        renderModelStatus('已加载 ' + models.length + ' 个模型', '');
      }
    }).catch(function (error) {
      if (requestId !== state.modelsRequest) return;
      state.models = [FALLBACK_AUTO_MODEL];
      state.selectedModel = 'auto';
      state.modelsLoaded = true;
      state.modelsFallback = true;
      renderModelCards();
      updateDenoiseUI();
      updateStartBar();
      renderModelStatus('模型列表加载失败，已降级为智能推荐', 'error');
      toast('加载模型列表失败，已降级为「智能推荐」：' + error.message, 'error', 7500);
    });
  }

  function initModels() {
    state.models = [FALLBACK_AUTO_MODEL];
    state.selectedModel = 'auto';
    if (el.modelGrid) {
      el.modelGrid.addEventListener('click', function (event) {
        var card = event.target.closest ? event.target.closest('.model-card') : null;
        if (!card) return;
        chooseModel(card.getAttribute('data-model'));
      });
    }
    if (el.denoiseCheckbox) {
      el.denoiseCheckbox.addEventListener('change', function () {
        if (isBusy()) {
          el.denoiseCheckbox.checked = Boolean(state.denoise);
          return;
        }
        state.denoise = Boolean(el.denoiseCheckbox.checked);
        updateStartBar();
      });
    }
    renderModelCards();
    updateDenoiseUI();
    void loadModels();
  }

  /* ------------------------------------------------------------------ */
  /* Plan preview (POST /api/plan, debounced + stale response dropped)   */
  /* ------------------------------------------------------------------ */

  function setPlanStatus(text, variant) {
    if (!el.planStatus) return;
    el.planStatus.textContent = text || '';
    el.planStatus.classList.remove('is-pending', 'is-error');
    if (variant) el.planStatus.classList.add('is-' + variant);
  }

  function schedulePlanRefresh() {
    window.clearTimeout(state.planTimer);
    state.planTimer = window.setTimeout(function () {
      state.planTimer = null;
      void refreshPlan();
    }, PLAN_DEBOUNCE_MS);
  }

  function cancelPendingPlan() {
    window.clearTimeout(state.planTimer);
    state.planTimer = null;
    state.planSeq += 1;
    if (state.planAbort) {
      state.planAbort.abort();
      state.planAbort = null;
    }
  }

  function refreshPlan() {
    window.clearTimeout(state.planTimer);
    state.planTimer = null;
    if (state.planAbort) {
      state.planAbort.abort();
      state.planAbort = null;
    }
    var seq = ++state.planSeq;
    var planFiles = state.files.filter(function (entry) {
      return entry.width > 0 && entry.height > 0;
    });

    state.files.forEach(function (entry) {
      if (entry.width > 0 && entry.height > 0) return;
      entry.plan = null;
      entry.planError = null;
    });

    if (!planFiles.length) {
      setPlanStatus('', '');
      renderFileList();
      return Promise.resolve();
    }

    var controller = new AbortController();
    state.planAbort = controller;
    planFiles.forEach(function (entry) { entry.planError = null; });
    setPlanStatus('正在计算目标尺寸…', 'pending');
    renderFileList();

    var payload = {
      target: state.tier,
      align: ALIGN,
      images: planFiles.map(function (entry) {
        return { name: entry.name, width: entry.width, height: entry.height };
      })
    };

    return apiFetch(API + '/plan', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      timeout: 15000,
      signal: controller.signal
    }).then(function (data) {
      if (seq !== state.planSeq) return;
      var images = data && Array.isArray(data.images) ? data.images : [];
      state.files.forEach(function (entry) {
        entry.plan = null;
        entry.planError = null;
      });
      images.forEach(function (plan, fallbackIndex) {
        if (!plan || typeof plan !== 'object') return;
        var planIndex = Number.isInteger(plan.index) ? plan.index : fallbackIndex;
        var target = planFiles[planIndex];
        if (target) target.plan = plan;
      });
      var missing = planFiles.filter(function (entry) { return !entry.plan; });
      if (missing.length) {
        missing.forEach(function (entry) { entry.planError = '服务端未返回该图的尺寸'; });
        setPlanStatus('部分图片的目标尺寸计算失败', 'error');
      } else {
        var tier = TIER_MAP[state.tier];
        var skipped = state.files.length - planFiles.length;
        setPlanStatus(
          '已按 ' + (tier ? tier.label : state.tier) + ' 计算目标尺寸' +
            (skipped > 0 ? '（' + skipped + ' 张尺寸未知，跳过预览）' : ''),
          ''
        );
      }
    }).catch(function (error) {
      if (seq !== state.planSeq) return;
      if (controller.signal.aborted) return;
      planFiles.forEach(function (entry) {
        entry.plan = null;
        entry.planError = error.message;
      });
      setPlanStatus('目标尺寸计算失败：' + error.message, 'error');
      toast('无法计算目标尺寸：' + error.message, 'error');
    }).then(function () {
      if (seq === state.planSeq) {
        state.planAbort = null;
        renderFileList();
      }
    });
  }

  /* ------------------------------------------------------------------ */
  /* Image selection                                                     */
  /* ------------------------------------------------------------------ */

  function readImageDimensions(file) {
    if (typeof window.createImageBitmap === 'function') {
      return window.createImageBitmap(file).then(function (bitmap) {
        var dimensions = { width: bitmap.width, height: bitmap.height };
        if (typeof bitmap.close === 'function') bitmap.close();
        return dimensions;
      }).catch(function () {
        return readDimensionsWithImage(file);
      });
    }
    return readDimensionsWithImage(file);
  }

  function readDimensionsWithImage(file) {
    return new Promise(function (resolve, reject) {
      var url = URL.createObjectURL(file);
      var image = new Image();
      image.onload = function () {
        var dimensions = { width: image.naturalWidth, height: image.naturalHeight };
        URL.revokeObjectURL(url);
        resolve(dimensions);
      };
      image.onerror = function () {
        URL.revokeObjectURL(url);
        reject(new Error('无法读取图片尺寸'));
      };
      image.src = url;
    });
  }

  function createThumbUrl(file) {
    try {
      return URL.createObjectURL(file);
    } catch (ignored) {
      return '';
    }
  }

  function revokeThumbUrl(entry) {
    if (entry && entry.thumbUrl) {
      try { URL.revokeObjectURL(entry.thumbUrl); } catch (ignored) { /* noop */ }
      entry.thumbUrl = '';
    }
  }

  function revokeAllThumbUrls() {
    state.files.forEach(revokeThumbUrl);
  }

  function addFiles(fileListLike) {
    var incoming = Array.prototype.slice.call(fileListLike || []);
    if (!incoming.length) return Promise.resolve();
    if (isBusy()) {
      toast('任务处理中，暂时不能添加图片', 'warn');
      return Promise.resolve();
    }

    var limits = state.limits;
    var problems = [];
    var accepted = [];
    var totalBytes = state.files.reduce(function (sum, entry) { return sum + entry.size; }, 0);
    var knownKeys = {};
    state.files.forEach(function (entry) { knownKeys[entry.key] = true; });

    incoming.forEach(function (file) {
      var name = file && file.name ? file.name : '图片';
      var ext = extOf(name);
      if (ALLOWED_EXT.indexOf(ext) === -1) {
        problems.push('不支持的文件类型：' + name);
        return;
      }
      if (file.type && file.type.indexOf('image/') !== 0) {
        problems.push('不支持的文件类型：' + name);
        return;
      }
      if (!file.size) {
        problems.push('空文件已跳过：' + name);
        return;
      }
      var key = name + '|' + file.size + '|' + (file.lastModified || 0);
      if (knownKeys[key]) {
        problems.push('已跳过重复文件：' + name);
        return;
      }
      if (file.size > limits.max_file_mb * 1024 * 1024) {
        problems.push('「' + name + '」超过 ' + limits.max_file_mb + 'MB 单文件限制');
        return;
      }
      if (state.files.length + accepted.length >= limits.max_files) {
        problems.push('一次最多 ' + limits.max_files + ' 张图片');
        return;
      }
      if (totalBytes + file.size > limits.max_total_mb * 1024 * 1024) {
        problems.push('图片总大小超过 ' + limits.max_total_mb + 'MB 限制');
        return;
      }
      accepted.push(file);
      totalBytes += file.size;
      knownKeys[key] = true;
    });

    if (!accepted.length) {
      if (problems.length) toast(problems.join('；'), 'error', 7500);
      updateStartBar();
      return Promise.resolve();
    }

    var additions = accepted.map(function (file) {
      return {
        uid: 'f' + (state.nextUid++),
        key: file.name + '|' + file.size + '|' + (file.lastModified || 0),
        file: file,
        name: file.name || '图片',
        size: file.size,
        thumbUrl: createThumbUrl(file),
        width: 0,
        height: 0,
        dimError: false,
        plan: null,
        planError: null
      };
    });

    state.files = state.files.concat(additions);
    renderFileList();
    if (problems.length) toast(problems.join('；'), 'warn', 7000);

    return Promise.all(additions.map(function (entry) {
      return readImageDimensions(entry.file).then(function (dimensions) {
        if (state.files.indexOf(entry) === -1) return;
        entry.width = dimensions.width;
        entry.height = dimensions.height;
        entry.dimError = false;
      }).catch(function () {
        if (state.files.indexOf(entry) === -1) return;
        entry.width = 0;
        entry.height = 0;
        entry.dimError = true;
      });
    })).then(function () {
      renderFileList();
      schedulePlanRefresh();
    });
  }

  function removeFile(uid) {
    if (isBusy()) {
      toast('任务处理中，暂时不能修改图片列表', 'warn');
      return;
    }
    var index = state.files.findIndex(function (entry) { return entry.uid === uid; });
    if (index === -1) return;
    var entry = state.files[index];
    revokeThumbUrl(entry);
    state.files.splice(index, 1);
    renderFileList();
    schedulePlanRefresh();
  }

  function clearFiles() {
    if (isBusy()) {
      toast('任务处理中，暂时不能修改图片列表', 'warn');
      return;
    }
    cancelPendingPlan();
    revokeAllThumbUrls();
    state.files = [];
    setPlanStatus('', '');
    renderFileList();
  }

  /* ------------------------------------------------------------------ */
  /* File list rendering                                                 */
  /* ------------------------------------------------------------------ */

  function buildThumb(entry) {
    var thumb = h('div', { class: 'file-thumb' });
    if (entry.thumbUrl) {
      var image = h('img', { src: entry.thumbUrl, alt: '', loading: 'lazy', decoding: 'async' });
      image.addEventListener('error', function () {
        if (image.parentNode) image.parentNode.removeChild(image);
        thumb.classList.add('is-fallback');
        thumb.setAttribute('data-ext', extLabel(entry.name));
      });
      thumb.append(image);
    } else {
      thumb.classList.add('is-fallback');
      thumb.setAttribute('data-ext', extLabel(entry.name));
    }
    return thumb;
  }

  function buildPlanLine(entry) {
    if (entry.dimError) {
      return h('p', {
        class: 'plan-error',
        text: '浏览器无法读取该格式的尺寸，仍可提交（由服务端解码后处理）'
      });
    }
    if (!entry.width || !entry.height) {
      return h('p', { class: 'plan-pending', text: '正在读取图片尺寸…' });
    }
    if (entry.planError) {
      return h('p', { class: 'plan-error', text: '目标尺寸计算失败：' + entry.planError });
    }
    var plan = entry.plan;
    if (!plan || !plan.output) {
      return h('p', { class: 'plan-pending', text: '正在计算目标尺寸…' });
    }
    var modeText = MODE_TEXT[plan.mode] || '';
    var modeClass = plan.mode === 'upscale' ? 'is-up' : plan.mode === 'downscale' ? 'is-down' : 'is-same';
    var line = h('div', { class: 'file-plan' },
      h('span', { class: 'plan-arrow', text: '目标尺寸' }),
      h('span', { class: 'plan-target', text: formatDimensions(plan.output.width, plan.output.height) }),
      h('span', { class: 'plan-ratio', text: formatScale(plan.scale) })
    );
    if (modeText) line.append(h('span', { class: 'mode-pill ' + modeClass, text: modeText }));
    if (plan.aligned) {
      line.append(h('span', {
        class: 'align-pill',
        title: '另一条边已吸附到最近的 8px 网格，偏差不超过 4px',
        text: '已对齐 8px'
      }));
    }
    return line;
  }

  function buildFileRow(entry) {
    var busy = isBusy();
    var originText;
    if (entry.width > 0 && entry.height > 0) {
      originText = '原图 ' + formatDimensions(entry.width, entry.height);
    } else if (entry.dimError) {
      originText = '原图尺寸未知';
    } else {
      originText = '正在读取图片尺寸…';
    }

    var main = h('div', { class: 'file-main' },
      h('div', { class: 'file-title' },
        h('span', { class: 'file-name', title: entry.name, text: entry.name }),
        h('span', { class: 'file-size', text: formatBytes(entry.size) })
      ),
      h('div', { class: 'file-origin', text: originText }),
      buildPlanLine(entry)
    );
    if (entry.plan && entry.plan.note) {
      main.append(h('p', { class: 'file-note', text: entry.plan.note }));
    }

    var remove = h('button', {
      class: 'remove-file',
      type: 'button',
      'aria-label': '移除 ' + entry.name,
      title: '移除这张图片',
      disabled: busy,
      text: '×',
      onClick: function () { removeFile(entry.uid); }
    });

    return h('li', { class: 'file-row', 'data-uid': entry.uid }, buildThumb(entry), main, remove);
  }

  function renderFileList() {
    if (!el.fileList) return;
    el.fileList.replaceChildren();
    state.files.forEach(function (entry) {
      el.fileList.append(buildFileRow(entry));
    });
    var count = state.files.length;
    if (el.fileListHead) el.fileListHead.hidden = count === 0;
    if (el.fileCount) el.fileCount.textContent = String(count);
    updateStartBar();
  }

  /* ------------------------------------------------------------------ */
  /* Busy state / start bar                                              */
  /* ------------------------------------------------------------------ */

  function syncBusyState() {
    var busy = isBusy();
    document.body.classList.toggle('is-busy', busy);

    if (el.dropzone) {
      el.dropzone.classList.toggle('is-disabled', busy);
      el.dropzone.setAttribute('aria-disabled', busy ? 'true' : 'false');
      el.dropzone.tabIndex = busy ? -1 : 0;
    }
    if (el.fileInput) el.fileInput.disabled = busy;
    if (el.clearFilesBtn) el.clearFilesBtn.disabled = busy;
    if (el.tierGrid) {
      Array.prototype.forEach.call(el.tierGrid.querySelectorAll('.tier-card'), function (card) {
        card.disabled = busy;
      });
    }
    if (el.modelGrid) {
      Array.prototype.forEach.call(el.modelGrid.querySelectorAll('.model-card'), function (card) {
        var locked = card.getAttribute('data-unavailable') === '1';
        card.disabled = busy || locked;
        card.classList.toggle('is-busy', busy);
      });
    }
    if (el.denoiseCheckbox) el.denoiseCheckbox.disabled = busy;
    if (el.denoiseLevelGroup) {
      Array.prototype.forEach.call(el.denoiseLevelGroup.querySelectorAll('.seg-btn'), function (button) {
        button.disabled = busy;
      });
    }
    if (el.fileList) {
      Array.prototype.forEach.call(el.fileList.querySelectorAll('.remove-file'), function (button) {
        button.disabled = busy;
      });
    }
    if (el.cancelBtn) {
      var active = Boolean(state.job && isActiveStatus(state.job.status));
      el.cancelBtn.hidden = !active;
      el.cancelBtn.disabled = !active || state.cancelling;
      el.cancelBtn.textContent = state.cancelling ? '正在取消…' : '取消任务';
    }
  }

  function updateStartBar() {
    var count = state.files.length;
    var busy = isBusy();
    if (el.fileListHead) el.fileListHead.hidden = count === 0;
    if (el.fileCount) el.fileCount.textContent = String(count);
    if (el.startBar) el.startBar.hidden = count === 0;
    if (el.startBtn) {
      el.startBtn.disabled = count === 0 || busy;
      el.startBtn.textContent = state.submitting ? '正在提交…' : (busy ? '处理中…' : '开始处理');
    }
    if (el.startSummary) {
      if (!count) {
        el.startSummary.textContent = '';
      } else {
        var totalBytes = state.files.reduce(function (sum, entry) { return sum + entry.size; }, 0);
        var tier = TIER_MAP[state.tier];
        var rule = tier ? (tier.rule === 'short' ? '短边 ' : '长边 ') + tier.value + 'px' : '';
        var modelSummary = submissionSummary();
        el.startSummary.textContent = '已选 ' + count + ' 张 · 合计 ' + formatBytes(totalBytes) +
          ' · 目标 ' + (tier ? tier.label : state.tier) + '（' + rule + '）' +
          ' · 模型 ' + modelSummary.label + '（' + modelSummary.denoiseText + '）';
      }
    }
    syncBusyState();
  }

  function validateSelection() {
    var limits = state.limits;
    if (!state.files.length) return '请先选择至少一张图片';
    if (state.files.length > limits.max_files) return '一次最多 ' + limits.max_files + ' 张图片';
    var totalBytes = 0;
    var problem = '';
    state.files.forEach(function (entry) {
      if (problem) return;
      if (ALLOWED_EXT.indexOf(extOf(entry.name)) === -1) {
        problem = '不支持的文件类型：' + entry.name;
        return;
      }
      if (entry.size > limits.max_file_mb * 1024 * 1024) {
        problem = '「' + entry.name + '」超过 ' + limits.max_file_mb + 'MB 单文件限制';
        return;
      }
      totalBytes += entry.size;
    });
    if (problem) return problem;
    if (totalBytes > limits.max_total_mb * 1024 * 1024) {
      return '图片总大小超过 ' + limits.max_total_mb + 'MB 限制';
    }
    return '';
  }

  /* ------------------------------------------------------------------ */
  /* Progress rendering                                                  */
  /* ------------------------------------------------------------------ */

  function setTotalProgress(percent) {
    var value = clamp(percent, 0, 100);
    if (el.totalProgressFill) el.totalProgressFill.style.width = value + '%';
    if (el.totalProgressBar) el.totalProgressBar.setAttribute('aria-valuenow', String(Math.round(value)));
    if (el.totalProgressText) el.totalProgressText.textContent = formatPercent(value);
  }

  function setProgressFillState(statusName) {
    if (!el.totalProgressFill) return;
    el.totalProgressFill.classList.remove('is-done', 'is-error', 'is-cancelled');
    if (statusName) el.totalProgressFill.classList.add('is-' + statusName);
  }

  function describeJob(job) {
    var total = Number(job.total) || (Array.isArray(job.files) ? job.files.length : 0);
    var current = Number(job.current_index);
    var status = String(job.status || '');
    if (status === 'queued') {
      var position = Number(job.queue_position);
      if (Number.isFinite(position) && position > 0) return '排队中 · 前面还有 ' + position + ' 个任务';
      return '排队中 · 等待开始';
    }
    if (status === 'running') {
      if (Number.isFinite(current) && current >= 0 && current < total) {
        var file = Array.isArray(job.files) ? job.files[current] : null;
        return '处理中 · 第 ' + (current + 1) + '/' + total + ' 张' +
          (file && file.name ? '：' + file.name : '');
      }
      return '处理中 · 共 ' + total + ' 张';
    }
    if (status === 'done') {
      return '全部完成' + (job.elapsed_sec !== null && job.elapsed_sec !== undefined
        ? ' · 用时 ' + formatDuration(job.elapsed_sec) : '');
    }
    if (status === 'cancelled') return '任务已取消';
    if (status === 'error') return '处理失败' + (job.error ? '：' + job.error : '');
    return JOB_STATUS_TEXT[status] || '处理中';
  }

  function progressFileMeta(file) {
    var parts = [];
    if (file.original && file.output && Number(file.original.width) > 0 && Number(file.output.width) > 0) {
      parts.push(formatDimensions(file.original.width, file.original.height) + ' → ' +
        formatDimensions(file.output.width, file.output.height));
    }
    if (file.status === 'done') {
      if (file.output_format) parts.push(String(file.output_format).toUpperCase());
      if (Number(file.output_size) > 0) parts.push(formatBytes(file.output_size));
    } else if (file.status === 'running') {
      parts.push(Math.round(clamp(file.progress, 0, 100)) + '%');
    } else if (file.status === 'error' && file.error) {
      parts.push(file.error);
    } else if (file.note) {
      parts.push(file.note);
    }
    return parts.join(' · ');
  }

  function progressFillClass(status) {
    if (status === 'done') return 'is-done';
    if (status === 'error') return 'is-error';
    if (status === 'cancelled') return 'is-cancelled';
    return '';
  }

  function renderProgressList(job) {
    if (!el.progressList) return;
    el.progressList.replaceChildren();
    var files = Array.isArray(job.files) ? job.files : [];
    files.forEach(function (file, index) {
      var status = String(file.status || 'pending');
      var percent = clamp(file.progress, 0, 100);
      var row = h('li', { class: 'progress-row' + (status === 'running' ? ' is-current' : '') },
        h('div', { class: 'progress-row-head' },
          h('span', { class: 'progress-file-name', text: (index + 1) + '. ' + (file.name || '图片') }),
          h('span', { class: 'status-pill status-' + status, text: FILE_STATUS_TEXT[status] || status })
        ),
        h('div', { class: 'progress-row-meta', text: progressFileMeta(file) }),
        h('div', { class: 'progress-row-track' },
          h('div', {
            class: 'progress-row-fill' + (progressFillClass(status) ? ' ' + progressFillClass(status) : ''),
            style: { width: percent + '%' }
          })
        )
      );
      el.progressList.append(row);
    });
  }

  function describeJobConfig(job) {
    var parts = [];
    if (job.target && job.target.label) parts.push('目标 ' + job.target.label);
    var jobModel = job.model && job.model.id ? modelById(String(job.model.id)) : currentModel();
    var requestedLabel = job.model && job.model.label
      ? job.model.label
      : (jobModel && jobModel.label ? jobModel.label : '');
    if (requestedLabel) parts.push('模型 ' + requestedLabel);
    if (job.model && typeof job.model === 'object') {
      var denoiseLevel = Number(job.model.denoise_level);
      parts.push(denoiseTextFor(job.model.denoise === true, Number.isFinite(denoiseLevel) ? denoiseLevel : 0, jobModel));
    }
    return parts.join(' · ');
  }

  function resetProgressDisplay() {
    if (el.progressList) el.progressList.replaceChildren();
    setTotalProgress(0);
    setProgressFillState('');
    if (el.jobConfigText) el.jobConfigText.textContent = '';
    if (el.jobStatusText) {
      el.jobStatusText.textContent = '准备中…';
      el.jobStatusText.removeAttribute('data-status');
    }
  }

  function applyJob(job) {
    state.job = job;
    if (el.progressPanel) el.progressPanel.hidden = false;
    setTotalProgress(job.progress);
    if (el.jobConfigText) el.jobConfigText.textContent = describeJobConfig(job);
    if (el.jobStatusText) {
      el.jobStatusText.textContent = describeJob(job);
      el.jobStatusText.setAttribute('data-status', String(job.status || ''));
    }
    setProgressFillState(
      job.status === 'done' ? 'done' : job.status === 'error' ? 'error' : job.status === 'cancelled' ? 'cancelled' : ''
    );
    renderProgressList(job);
    updateStartBar();
  }

  /* ------------------------------------------------------------------ */
  /* Polling                                                             */
  /* ------------------------------------------------------------------ */

  function stopPolling() {
    state.polling = false;
    if (state.pollTimer !== null) {
      window.clearTimeout(state.pollTimer);
      state.pollTimer = null;
    }
  }

  function schedulePoll(delay) {
    if (!state.jobId) return;
    if (state.pollTimer !== null) window.clearTimeout(state.pollTimer);
    state.pollTimer = window.setTimeout(pollOnce, Number(delay) || 0);
  }

  function startPolling() {
    stopPolling();
    if (!state.jobId) return;
    state.polling = true;
    state.pollErrors = 0;
    schedulePoll(0);
    syncBusyState();
  }

  function ensurePolling() {
    if (!state.jobId || state.polling) return;
    state.polling = true;
    schedulePoll(200);
  }

  function pollOnce() {
    state.pollTimer = null;
    var jobId = state.jobId;
    if (!jobId) {
      state.polling = false;
      return;
    }
    apiFetch(API + '/jobs/' + encodeURIComponent(jobId), { timeout: 12000 }).then(function (job) {
      if (state.jobId !== jobId) return;
      state.pollErrors = 0;
      applyJob(job);
      if (isActiveStatus(job.status)) {
        if (state.polling) schedulePoll(POLL_INTERVAL_MS);
      } else {
        handleJobFinished(job);
      }
    }).catch(function (error) {
      if (state.jobId !== jobId) return;
      if (error.status === 404) {
        stopPolling();
        state.job = null;
        state.jobId = null;
        hideResults();
        if (el.progressPanel) el.progressPanel.hidden = true;
        resetProgressDisplay();
        updateStartBar();
        toast('任务不存在或已被清理，请重新提交', 'error');
        return;
      }
      state.pollErrors += 1;
      if (state.pollErrors <= MAX_POLL_ERRORS && state.polling) {
        if (el.jobStatusText) el.jobStatusText.textContent = '获取状态失败，正在重试（' + state.pollErrors + '/' + MAX_POLL_ERRORS + '）…';
        schedulePoll(Math.min(5000, POLL_INTERVAL_MS * 2 * state.pollErrors));
      } else {
        state.polling = false;
        if (el.jobStatusText) el.jobStatusText.textContent = '无法获取任务状态：' + error.message;
        syncBusyState();
        toast('无法获取任务状态：' + error.message + '。请稍后刷新页面重试。', 'error', 9000);
      }
    });
  }

  function hideResults() {
    if (el.resultPanel) el.resultPanel.hidden = true;
    if (el.resultGrid) el.resultGrid.replaceChildren();
    if (el.resultNotices) el.resultNotices.replaceChildren();
  }

  function handleJobFinished(job) {
    stopPolling();
    state.cancelling = false;
    renderResults(job);
    if (el.jobStatusText) el.jobStatusText.textContent = describeJob(job);
    updateStartBar();
    var doneCount = (job.files || []).filter(function (file) { return file.status === 'done'; }).length;
    if (job.status === 'done') {
      toast('处理完成' + (job.elapsed_sec !== null && job.elapsed_sec !== undefined
        ? '，用时 ' + formatDuration(job.elapsed_sec) : '') + '，共 ' + doneCount + ' 张，结果可另存为',
        'success', 5200);
    } else if (job.status === 'cancelled') {
      toast('任务已取消' + (doneCount ? '，已完成的 ' + doneCount + ' 张结果仍可保存' : ''), 'warn', 5200);
    } else if (job.status === 'error') {
      toast('处理失败：' + (job.error || '未知错误'), 'error', 9000);
    }
  }

  /* ------------------------------------------------------------------ */
  /* Results                                                             */
  /* ------------------------------------------------------------------ */

  function buildOutputFilename(job, file) {
    var stem = (stemOf(file.name) || ('image_' + (Number(file.index) + 1))).trim() || 'image';
    var tier = job && job.target && job.target.id ? job.target.id : state.tier;
    var rawExt = String(file.output_format || 'png').toLowerCase();
    var ext = /^(png|jpg|jpeg|webp)$/.test(rawExt) ? (rawExt === 'jpeg' ? 'jpg' : rawExt) : 'png';
    return stem + '_' + tier + '.' + ext;
  }

  function triggerDownload(url, filename) {
    var link = document.createElement('a');
    link.href = url;
    if (filename) link.download = filename;
    link.rel = 'noopener';
    link.style.display = 'none';
    document.body.append(link);
    link.click();
    window.setTimeout(function () {
      if (link.parentNode) link.parentNode.removeChild(link);
    }, 1000);
  }

  function saveFileAs(job, file) {
    if (!file || !file.download_url) {
      toast('该结果还没有可用的下载地址，请稍后重试', 'error');
      return;
    }
    var filename = buildOutputFilename(job, file);
    var ext = extOf(filename) || 'png';
    var mime = mimeForExt(ext);

    if (typeof window.showSaveFilePicker !== 'function') {
      triggerDownload(file.download_url, filename);
      toast('浏览器不支持自定义保存位置，已开始下载 ' + filename, 'info', 4200);
      return;
    }

    /* 必须在用户手势内先调用 showSaveFilePicker，再 fetch 文件内容；重复弹窗会打断流程 */
    var pickerPromise;
    try {
      pickerPromise = window.showSaveFilePicker({
        suggestedName: filename,
        types: [
          (function () {
            var accept = {};
            accept[mime] = ['.' + ext];
            return { description: '图片文件', accept: accept };
          })()
        ]
      });
    } catch (error) {
      triggerDownload(file.download_url, filename);
      return;
    }

    pickerPromise.then(function (handle) {
      return apiFetch(file.download_url, { timeout: 120000 }).then(function (response) {
        return response.blob();
      }).then(function (blob) {
        return handle.createWritable().then(function (writable) {
          return writable.write(blob).then(function () {
            return writable.close();
          });
        });
      }).then(function () {
        toast('已另存为 ' + filename, 'success');
      });
    }).catch(function (error) {
      if (error && error.name === 'AbortError') return;
      if (error && error.name === 'SecurityError') {
        triggerDownload(file.download_url, filename);
        return;
      }
      toast('保存失败：' + (error && error.message ? error.message : '未知错误'), 'error');
    });
  }

  function saveZipAs() {
    var job = state.job;
    var jobId = state.jobId;
    if (!job || !jobId) {
      toast('当前没有可打包的结果', 'info');
      return;
    }
    var doneCount = (job.files || []).filter(function (file) { return file.status === 'done'; }).length;
    if (!doneCount) {
      toast('还没有可下载的结果', 'warn');
      return;
    }
    var url = API + '/jobs/' + encodeURIComponent(jobId) + '/download.zip';
    var filename = 'PicPolish_' + String(jobId).slice(0, 8) + '.zip';

    if (typeof window.showSaveFilePicker !== 'function') {
      triggerDownload(url, filename);
      toast('浏览器不支持自定义保存位置，已开始下载 ZIP', 'info', 4200);
      return;
    }

    var pickerPromise;
    try {
      pickerPromise = window.showSaveFilePicker({
        suggestedName: filename,
        types: [{ description: 'ZIP 压缩包', accept: { 'application/zip': ['.zip'] } }]
      });
    } catch (error) {
      triggerDownload(url, filename);
      return;
    }

    pickerPromise.then(function (handle) {
      return apiFetch(url, { timeout: 180000 }).then(function (response) {
        return response.blob();
      }).then(function (blob) {
        return handle.createWritable().then(function (writable) {
          return writable.write(blob).then(function () {
            return writable.close();
          });
        });
      }).then(function () {
        toast('ZIP 已另存为 ' + filename, 'success');
      });
    }).catch(function (error) {
      if (error && error.name === 'AbortError') return;
      if (error && error.name === 'SecurityError') {
        triggerDownload(url, filename);
        return;
      }
      toast('保存 ZIP 失败：' + (error && error.message ? error.message : '未知错误'), 'error');
    });
  }

  function resultModelText(job, file) {
    var used = file.model_used ? String(file.model_used) : '';
    var requestedLabel = job.model && job.model.label
      ? job.model.label
      : modelLabelById(file.model);
    if (used) {
      if (requestedLabel && requestedLabel !== used) return requestedLabel + ' → ' + used;
      return used;
    }
    return requestedLabel || '未记录';
  }

  function resultDenoiseText(job, file) {
    var jobModel = job.model && job.model.id ? modelById(String(job.model.id)) : currentModel();
    var denoise = Boolean(job.model && job.model.denoise);
    var level = 0;
    if (file.denoise_level !== null && file.denoise_level !== undefined && Number.isFinite(Number(file.denoise_level))) {
      level = Number(file.denoise_level);
    } else if (job.model && Number.isFinite(Number(job.model.denoise_level))) {
      level = Number(job.model.denoise_level);
    }
    if (!job.model && file.denoise_level === null && file.denoise_level === undefined) return '';
    return denoiseTextFor(denoise, level, jobModel);
  }

  function buildResultCard(job, file) {
    var figure = h('div', { class: 'result-figure' });
    var image = h('img', {
      src: file.preview_url,
      alt: (file.name || '图片') + ' 的处理结果',
      loading: 'lazy',
      decoding: 'async'
    });
    image.addEventListener('error', function () {
      if (image.parentNode) image.parentNode.removeChild(image);
      figure.classList.add('is-error');
    });
    figure.append(image);

    var formatText = file.output_format ? String(file.output_format).toUpperCase() : '图片';
    var metaParts = [formatText];
    if (Number(file.output_size) > 0) metaParts.push(formatBytes(file.output_size));

    var dims = null;
    if (file.original && file.output) {
      dims = h('div', { class: 'result-dims' },
        h('span', { text: formatDimensions(file.original.width, file.original.height) + ' → ' }),
        h('strong', { text: formatDimensions(file.output.width, file.output.height) })
      );
    }

    var modelText = resultModelText(job, file);
    var denoiseValue = resultDenoiseText(job, file);

    var info = h('div', { class: 'result-info' },
      h('div', { class: 'result-name-row' },
        h('span', { class: 'result-name', title: file.name || '', text: file.name || '图片' }),
        h('span', { class: 'format-pill', text: formatText })
      ),
      dims,
      h('div', { class: 'result-model' },
        h('span', { text: '模型 ' }),
        h('strong', { text: modelText }),
        denoiseValue
          ? h('span', { class: 'result-model-sep', text: ' · ' })
          : null,
        denoiseValue ? h('span', { text: denoiseValue }) : null
      ),
      h('div', { class: 'result-meta', text: metaParts.join(' · ') }),
      file.note ? h('p', { class: 'result-note', text: file.note }) : null,
      h('div', { class: 'result-actions' },
        h('button', {
          class: 'btn btn-primary btn-sm',
          type: 'button',
          text: '另存为…',
          title: '选择保存位置并自定义文件名',
          onClick: function () { saveFileAs(job, file); }
        }),
        h('a', {
          class: 'btn btn-ghost btn-sm',
          href: file.download_url,
          download: '',
          text: '下载',
          title: '直接下载到浏览器默认下载目录'
        })
      )
    );

    return h('article', { class: 'result-card' }, figure, info);
  }

  function renderResults(job) {
    if (!el.resultPanel || !el.resultGrid) return;
    var files = Array.isArray(job.files) ? job.files : [];
    var doneFiles = files.filter(function (file) { return file.status === 'done' && file.preview_url; });
    var terminal = !isActiveStatus(job.status);
    if (!doneFiles.length && !terminal) {
      el.resultPanel.hidden = true;
      return;
    }

    el.resultPanel.hidden = false;
    var label = job.target && job.target.label
      ? job.target.label
      : (TIER_MAP[state.tier] ? TIER_MAP[state.tier].label : state.tier);
    var modelLabel = job.model && job.model.label
      ? job.model.label
      : (currentModel() && currentModel().label ? currentModel().label : '');
    var total = Number(job.total) || files.length;
    if (el.resultSummary) {
      if (doneFiles.length) {
        var summaryParts = [doneFiles.length + ' / ' + total + ' 张完成', '目标 ' + label];
        if (modelLabel) summaryParts.push('模型 ' + modelLabel);
        summaryParts.push('文件保存在本机缓存');
        el.resultSummary.textContent = summaryParts.join(' · ');
      } else {
        el.resultSummary.textContent = '没有可保存的结果 · 目标 ' + label;
      }
    }

    el.resultGrid.replaceChildren();
    doneFiles.forEach(function (file) {
      el.resultGrid.append(buildResultCard(job, file));
    });

    if (el.resultNotices) {
      el.resultNotices.replaceChildren();
      files.forEach(function (file) {
        if (file.status === 'done') return;
        var reason = file.status === 'error'
          ? (file.error || '处理失败')
          : file.status === 'cancelled'
            ? '已取消'
            : '未处理';
        el.resultNotices.append(h('li', {
          class: 'result-notice',
          text: (file.name || '图片') + '：' + reason
        }));
      });
    }

    if (el.zipBtn) el.zipBtn.disabled = doneFiles.length === 0;
    if (el.deleteJobBtn) el.deleteJobBtn.disabled = false;
  }

  /* ------------------------------------------------------------------ */
  /* Submit / cancel / delete                                            */
  /* ------------------------------------------------------------------ */

  function startJob() {
    if (state.submitting) return;
    if (isBusy()) {
      toast('任务正在处理中，请等待完成或取消', 'warn');
      return;
    }
    if (!state.files.length) {
      toast('请先选择至少一张图片', 'warn');
      return;
    }
    var validation = validateSelection();
    if (validation) {
      toast(validation, 'error', 7000);
      return;
    }
    var selectedModel = currentModel();
    if (!isModelSelectable(selectedModel)) {
      toast('所选模型权重缺失，请先运行 scripts/fetch_models.py 下载', 'error', 7000);
      return;
    }
    var options = submissionOptions();

    state.submitting = true;
    updateStartBar();
    var previousJobId = state.jobId;

    var form = new FormData();
    state.files.forEach(function (entry) {
      form.append('files', entry.file, entry.name);
    });
    form.append('target', state.tier);
    form.append('align', String(ALIGN));
    form.append('format', 'auto');
    form.append('model', options.model);
    form.append('denoise', options.denoise ? 'true' : 'false');
    form.append('denoise_level', String(options.denoiseLevel));

    apiFetch(API + '/jobs', { method: 'POST', body: form, timeout: 300000 }).then(function (job) {
      if (!job || typeof job !== 'object' || !job.job_id) {
        throw new ApiError('服务返回的任务信息不完整', 0);
      }
      stopPolling();
      state.job = job;
      state.jobId = job.job_id;
      state.cancelling = false;
      state.pollErrors = 0;

      if (previousJobId && previousJobId !== job.job_id) {
        /* 上一个任务已结束：尽力清理其缓存，失败不影响当前任务 */
        apiFetch(API + '/jobs/' + encodeURIComponent(previousJobId), { method: 'DELETE', timeout: 10000 })
          .catch(function () { /* 忽略 */ });
      }

      hideResults();
      if (el.progressPanel) el.progressPanel.hidden = false;
      resetProgressDisplay();
      applyJob(job);
      startPolling();
      toast('任务已提交，正在排队处理', 'success');
      if (el.progressPanel && typeof el.progressPanel.scrollIntoView === 'function') {
        el.progressPanel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      }
    }).catch(function (error) {
      toast('提交任务失败：' + error.message, 'error', 9000);
    }).then(function () {
      state.submitting = false;
      updateStartBar();
    });
  }

  function cancelCurrentJob() {
    var jobId = state.jobId;
    if (!jobId) {
      toast('当前没有正在运行的任务', 'info');
      return;
    }
    if (!state.job || !isActiveStatus(state.job.status)) {
      toast('任务已结束，无需取消', 'info');
      return;
    }
    if (state.cancelling) return;
    state.cancelling = true;
    syncBusyState();

    apiFetch(API + '/jobs/' + encodeURIComponent(jobId) + '/cancel', { method: 'POST', timeout: 15000 })
      .then(function (data) {
        toast('已发送取消请求，当前图片处理结束后停止', 'info', 4200);
        state.cancelling = false;
        syncBusyState();
        if (!state.polling || (data && data.status === 'cancelled')) ensurePolling();
      })
      .catch(function (error) {
        state.cancelling = false;
        syncBusyState();
        if (error.status === 409) {
          toast('任务已结束，正在同步最终状态', 'info');
          ensurePolling();
        } else {
          toast('取消失败：' + error.message, 'error');
        }
      });
  }

  function deleteCurrentJob() {
    var jobId = state.jobId;
    if (!jobId) {
      toast('当前没有可删除的任务', 'info');
      return;
    }
    if (isBusy()) {
      toast('任务正在处理中，请先取消任务', 'warn');
      return;
    }
    if (!window.confirm('确定删除这个任务吗？服务端缓存的原图与结果会被清理，且无法恢复。')) return;

    if (el.deleteJobBtn) el.deleteJobBtn.disabled = true;
    apiFetch(API + '/jobs/' + encodeURIComponent(jobId), { method: 'DELETE', timeout: 20000 })
      .then(function (data) {
        stopPolling();
        state.job = null;
        state.jobId = null;
        state.cancelling = false;
        state.pollErrors = 0;
        hideResults();
        if (el.progressPanel) el.progressPanel.hidden = true;
        resetProgressDisplay();
        updateStartBar();
        var freed = data && Number(data.freed_bytes);
        toast('任务已删除' + (Number.isFinite(freed) && freed > 0 ? '，释放 ' + formatBytes(freed) : ''), 'success');
      })
      .catch(function (error) {
        toast('删除任务失败：' + error.message, 'error');
      })
      .then(function () {
        if (el.deleteJobBtn) el.deleteJobBtn.disabled = false;
      });
  }

  /* ------------------------------------------------------------------ */
  /* Event wiring                                                        */
  /* ------------------------------------------------------------------ */

  function initUpload() {
    if (!el.dropzone || !el.fileInput) return;

    el.dropzone.addEventListener('click', function (event) {
      if (event.target === el.fileInput) return;
      if (isBusy()) {
        toast('任务处理中，暂时不能添加图片', 'warn');
        return;
      }
      el.fileInput.click();
    });

    el.dropzone.addEventListener('keydown', function (event) {
      if (event.key !== 'Enter' && event.key !== ' ') return;
      event.preventDefault();
      if (isBusy()) return;
      el.fileInput.click();
    });

    el.fileInput.addEventListener('change', function () {
      /* 必须在清空 input.value 之前把 FileList 复制成数组：FileList 是实时的，
         清空 value 会把同一个列表里的文件一起清掉。 */
      var picked = Array.prototype.slice.call(el.fileInput.files || []);
      el.fileInput.value = '';
      if (picked.length) void addFiles(picked);
    });

    function handleDragOver(event) {
      event.preventDefault();
      if (event.dataTransfer) event.dataTransfer.dropEffect = 'copy';
      if (!isBusy()) el.dropzone.classList.add('is-dragover');
    }

    el.dropzone.addEventListener('dragenter', handleDragOver);
    el.dropzone.addEventListener('dragover', handleDragOver);
    el.dropzone.addEventListener('dragleave', function (event) {
      if (!event.relatedTarget || !el.dropzone.contains(event.relatedTarget)) {
        el.dropzone.classList.remove('is-dragover');
      }
    });
    el.dropzone.addEventListener('drop', function (event) {
      event.preventDefault();
      el.dropzone.classList.remove('is-dragover');
      if (isBusy()) {
        toast('任务处理中，暂时不能添加图片', 'warn');
        return;
      }
      var files = event.dataTransfer && event.dataTransfer.files;
      if (files && files.length) void addFiles(files);
    });

    /* 防止把图片拖到页面其它区域时浏览器直接打开文件 */
    window.addEventListener('dragover', function (event) {
      var types = event.dataTransfer && event.dataTransfer.types;
      if (types && Array.prototype.indexOf.call(types, 'Files') !== -1) event.preventDefault();
    });
    window.addEventListener('drop', function (event) {
      if (event.dataTransfer && event.dataTransfer.files && event.dataTransfer.files.length) {
        event.preventDefault();
      }
    });

    if (el.clearFilesBtn) {
      el.clearFilesBtn.addEventListener('click', clearFiles);
    }
  }

  function initActions() {
    if (el.startBtn) {
      el.startBtn.addEventListener('click', function () { startJob(); });
    }
    if (el.cancelBtn) {
      el.cancelBtn.addEventListener('click', function () { cancelCurrentJob(); });
    }
    if (el.zipBtn) {
      el.zipBtn.addEventListener('click', function () { saveZipAs(); });
    }
    if (el.deleteJobBtn) {
      el.deleteJobBtn.addEventListener('click', function () { deleteCurrentJob(); });
    }
    if (el.engineBadge) {
      el.engineBadge.addEventListener('click', function () {
        if (el.engineBadge.disabled) return;
        void loadHealth({ silent: true });
      });
    }
  }

  function cacheElements() {
    ELEMENT_IDS.forEach(function (id) {
      var node = document.getElementById(id);
      if (!node) console.warn('[PicPolish] 缺少页面元素 #' + id);
      el[id] = node;
    });
  }

  /* ------------------------------------------------------------------ */
  /* Bootstrap                                                           */
  /* ------------------------------------------------------------------ */

  function init() {
    cacheElements();
    initTierCards();
    initModels();
    initUpload();
    initActions();
    updateUploadHint();
    renderFileList();
    updateStartBar();
    setPlanStatus('', '');
    void loadHealth({ silent: false });
    document.documentElement.setAttribute('data-app', 'ready');
  }

  function safeInit() {
    try {
      init();
    } catch (error) {
      document.documentElement.setAttribute('data-app', 'error');
      console.error('[PicPolish] 页面初始化失败', error);
      try {
        var banner = document.createElement('p');
        banner.className = 'noscript';
        banner.textContent = '页面初始化失败：' + (error && error.message ? error.message : '未知错误') +
          '。请刷新页面重试。';
        var main = document.getElementById('main');
        if (main) main.prepend(banner);
      } catch (ignored) { /* noop */ }
    }
  }

  window.addEventListener('unhandledrejection', function (event) {
    console.error('[PicPolish] 未处理的异步错误', event.reason);
  });

  window.addEventListener('beforeunload', function () {
    stopPolling();
    cancelPendingPlan();
    revokeAllThumbUrls();
  });

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', safeInit, { once: true });
  } else {
    safeInit();
  }
})();
