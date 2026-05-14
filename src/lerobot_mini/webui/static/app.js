/**
 * LeRobot Mini WebUI - Pure WebSocket Frontend
 *
 * WebSocket /ws:
 *   C->S JSON:   {"cmd": "switch_mode"}, {"cmd": "save", "label": 1}, ...
 *   S->C JSON:   status snapshots (mode, recording, episode, frames, ...)
 *   S->C Binary: [4B cam_name_len LE][cam_name UTF-8][JPEG data]
 */

(function () {
  'use strict';

  // ------------------------------------------------------------------
  // State
  // ------------------------------------------------------------------
  let ws = null;
  let isRecording = false;
  let canvasCtx = {};       // cam_name -> { canvas, ctx }
  let pendingCameras = [];  // cameras not yet rendered
  let reconnectTimer = null;
  let reconnectDelay = 1000;

  // ------------------------------------------------------------------
  // DOM elements
  // ------------------------------------------------------------------
  const els = {
    cameraContainer: document.getElementById('camera-container'),
    recordingIndicator: document.getElementById('recording-indicator'),
    modeDisplay: document.getElementById('mode-display'),
    controlDisplay: document.getElementById('control-display'),
    episodeNum: document.getElementById('episode-num'),
    frameCount: document.getElementById('frame-count'),
    elapsedTime: document.getElementById('elapsed-time'),
    inferenceStatus: document.getElementById('inference-status'),
    connectionStatus: document.getElementById('connection-status'),
    historyList: document.getElementById('history-list'),
    saveDialog: document.getElementById('save-dialog'),
    btnRecord: document.getElementById('btn-record'),
  };

  // ------------------------------------------------------------------
  // WebSocket connection
  // ------------------------------------------------------------------

  function connect() {
    if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
      return;
    }

    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const url = protocol + '//' + location.host + '/ws';
    ws = new WebSocket(url);
    ws.binaryType = 'blob';

    ws.onopen = function () {
      console.log('[WebUI] WebSocket connected');
      setConnected(true);
      reconnectDelay = 1000;
      if (pendingCameras.length > 0) {
        initCanvases(pendingCameras);
        pendingCameras = [];
      }
    };

    ws.onmessage = function (event) {
      if (event.data instanceof Blob) {
        handleBinaryFrame(event.data);
      } else {
        try {
          const msg = JSON.parse(event.data);
          if (msg.error) {
            console.warn('[WebUI] Server error:', msg.error);
          } else if (msg.ack) {
            console.log('[WebUI] Ack:', msg.ack);
          } else {
            updateStatus(msg);
          }
        } catch (e) {
          console.error('[WebUI] Parse error:', e);
        }
      }
    };

    ws.onclose = function () {
      console.log('[WebUI] WebSocket disconnected');
      setConnected(false);
      scheduleReconnect();
    };

    ws.onerror = function (err) {
      console.error('[WebUI] WebSocket error:', err);
    };
  }

  function scheduleReconnect() {
    if (reconnectTimer) return;
    console.log('[WebUI] Reconnecting in', reconnectDelay, 'ms');
    reconnectTimer = setTimeout(function () {
      reconnectTimer = null;
      reconnectDelay = Math.min(reconnectDelay * 2, 10000);
      connect();
    }, reconnectDelay);
  }

  function setConnected(ok) {
    if (ok) {
      els.connectionStatus.textContent = '● 已连接';
      els.connectionStatus.className = 'connected';
    } else {
      els.connectionStatus.textContent = '● 未连接';
      els.connectionStatus.className = 'disconnected';
    }
  }

  // ------------------------------------------------------------------
  // Binary frame decoding
  // ------------------------------------------------------------------

  function handleBinaryFrame(blob) {
    blob.arrayBuffer().then(function (buf) {
      const view = new DataView(buf);
      if (buf.byteLength < 4) return;

      const nameLen = view.getUint32(0, true); // little-endian
      if (buf.byteLength < 4 + nameLen) return;

      const nameBytes = new Uint8Array(buf, 4, nameLen);
      const camName = new TextDecoder().decode(nameBytes);

      const jpegData = new Uint8Array(buf, 4 + nameLen);
      const blob = new Blob([jpegData], { type: 'image/jpeg' });
      const url = URL.createObjectURL(blob);

      drawFrame(camName, url);
    }).catch(function (e) {
      console.error('[WebUI] Frame decode error:', e);
    });
  }

  function drawFrame(camName, url) {
    var entry = canvasCtx[camName];
    if (!entry) return;

    var img = new Image();
    img.onload = function () {
      var cw = entry.canvas.width;
      var ch = entry.canvas.height;
      if (cw !== img.naturalWidth || ch !== img.naturalHeight) {
        entry.canvas.width = img.naturalWidth;
        entry.canvas.height = img.naturalHeight;
      }
      entry.ctx.drawImage(img, 0, 0);
      URL.revokeObjectURL(url);
    };
    img.onerror = function () {
      URL.revokeObjectURL(url);
    };
    img.src = url;
  }

  // ------------------------------------------------------------------
  // Canvas initialization
  // ------------------------------------------------------------------

  function initCanvases(cameraNames) {
    els.cameraContainer.innerHTML = '';
    canvasCtx = {};

    cameraNames.forEach(function (cam) {
      var div = document.createElement('div');
      div.className = 'camera-feed';

      var canvas = document.createElement('canvas');
      canvas.className = 'camera-canvas';

      // Match the CSS aspect-ratio (4/3) initially; canvas will resize on first frame
      var label = document.createElement('div');
      label.className = 'camera-label';
      label.textContent = cam;

      div.appendChild(canvas);
      div.appendChild(label);
      els.cameraContainer.appendChild(div);

      canvasCtx[cam] = { canvas: canvas, ctx: canvas.getContext('2d') };
    });
  }

  // ------------------------------------------------------------------
  // Status update (JSON from server)
  // ------------------------------------------------------------------

  function updateStatus(msg) {
    // Camera list — init canvases on first status with cameras
    if (msg.cameras && msg.cameras.length > 0) {
      var currentCams = Object.keys(canvasCtx);
      var same = currentCams.length === msg.cameras.length &&
        currentCams.every(function (c, i) { return c === msg.cameras[i]; });
      if (!same) {
        initCanvases(msg.cameras);
      }
    }

    // Mode
    if (msg.mode) {
      els.modeDisplay.textContent = msg.mode.toUpperCase();
      els.modeDisplay.className = 'mode ' + (msg.mode === 'policy' ? 'policy' : 'teleop');
    }

    // Control mode
    if (msg.control) {
      els.controlDisplay.textContent = msg.control.toUpperCase();
    }

    // Recording
    isRecording = msg.recording;
    els.recordingIndicator.classList.toggle('recording', isRecording);
    els.btnRecord.classList.toggle('recording', isRecording);
    var btnDesc = els.btnRecord.querySelector('.btn-desc');
    if (btnDesc) btnDesc.textContent = isRecording ? '停止' : '录制';

    // Episode info
    els.episodeNum.textContent = msg.episode || 0;
    els.frameCount.textContent = msg.frames || 0;
    els.elapsedTime.textContent = (msg.elapsed || 0).toFixed(1) + 's';

    // Inference status
    if (typeof msg.inference_ok !== 'undefined') {
      els.inferenceStatus.textContent = msg.inference_ok ? '推理: OK' : '推理: ERR';
      els.inferenceStatus.style.color = msg.inference_ok ? '#16a34a' : '#ef4444';
    }

    // History
    if (msg.history) {
      updateHistory(msg.history);
    }
  }

  // ------------------------------------------------------------------
  // History
  // ------------------------------------------------------------------

  function updateHistory(history) {
    els.historyList.innerHTML = '';
    history.slice().reverse().forEach(function (item) {
      var div = document.createElement('div');
      var cls = 'history-item';
      if (item.success === 1) cls += ' success';
      else if (item.success === 0) cls += ' failure';
      div.className = cls;
      div.textContent = '#' + item.episode + ' ' + item.frames + '帧';
      els.historyList.appendChild(div);
    });
  }

  // ------------------------------------------------------------------
  // Commands
  // ------------------------------------------------------------------

  function sendCmd(cmd, payload) {
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      console.warn('[WebUI] Not connected');
      return;
    }
    var msg = Object.assign({ cmd: cmd }, payload || {});
    ws.send(JSON.stringify(msg));
  }

  // ------------------------------------------------------------------
  // Save dialog
  // ------------------------------------------------------------------

  function showSaveDialog() {
    els.saveDialog.classList.remove('hidden');
  }

  function hideSaveDialog() {
    els.saveDialog.classList.add('hidden');
  }

  function sendSave(label) {
    sendCmd('save', { label: label });
    hideSaveDialog();
  }

  // ------------------------------------------------------------------
  // Event listeners
  // ------------------------------------------------------------------

  document.querySelectorAll('.btn[data-cmd]').forEach(function (btn) {
    btn.addEventListener('click', function (e) {
      e.preventDefault();
      var cmd = btn.dataset.cmd;

      if (cmd === 'save_prompt') {
        if (!isRecording) {
          alert('请先开始录制');
          return;
        }
        showSaveDialog();
        return;
      }

      sendCmd(cmd);
    });
  });

  document.querySelectorAll('.btn-label-choice').forEach(function (btn) {
    btn.addEventListener('click', function (e) {
      e.preventDefault();
      var label = parseInt(btn.dataset.label, 10);
      sendSave(label);
    });
  });

  document.getElementById('btn-cancel-save').addEventListener('click', function (e) {
    e.preventDefault();
    hideSaveDialog();
  });

  // Prevent double-tap zoom on iOS
  document.addEventListener('touchend', function (e) {
    e.preventDefault();
    e.target.click();
  }, { passive: false });

  // ------------------------------------------------------------------
  // Init
  // ------------------------------------------------------------------

  connect();
})();
