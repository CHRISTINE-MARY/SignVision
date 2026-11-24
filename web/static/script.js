/**
 * SignVision — Real-Time Sign Language Tracking & Translation
 * Uses @mediapipe/tasks-vision (HandLandmarker) for in-browser landmark detection.
 */

// ── MediaPipe Tasks Vision CDN (ES module) ────────────────────────────────────
const MP_CDN = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm';
import {
  HandLandmarker,
  FilesetResolver,
  DrawingUtils
} from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/vision_bundle.mjs';

// Empty origin: this page's host (local). Set origin: the Oracle VM.
function apiOrigin() {
  return String(window.SIGNVISION_API_ORIGIN || '').trim().replace(/\/$/, '');
}

function apiUrl(path) {
  const origin = apiOrigin();
  return origin ? `${origin}${path}` : path;
}

function wsUrl() {
  const origin = apiOrigin();
  if (origin) {
    const url = new URL(origin);
    const protocol = url.protocol === 'https:' ? 'wss:' : 'ws:';
    return `${protocol}//${url.host}/ws`;
  }
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${protocol}//${window.location.host}/ws`;
}

// ── Hand connections (21-point skeleton) ─────────────────────────────────────
const HAND_CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],
  [0,5],[5,6],[6,7],[7,8],
  [0,9],[9,10],[10,11],[11,12],
  [0,13],[13,14],[14,15],[15,16],
  [0,17],[17,18],[18,19],[19,20],
  [5,9],[9,13],[13,17]
];

// ── UI Elements ───────────────────────────────────────────────────────────────
const systemStatusPill = document.getElementById('system-status-pill');
const systemStatusText = document.getElementById('system-status-text');
const modeBadge        = document.getElementById('mode-badge');

const tabWebcam   = document.getElementById('tab-webcam');
const tabUpload   = document.getElementById('tab-upload');
const webcamPanel = document.getElementById('webcam-panel');
const uploadPanel = document.getElementById('upload-panel');

const webcamVideo       = document.getElementById('webcam-video');
const overlayCanvas     = document.getElementById('overlay-canvas');
const hiddenCanvas      = document.getElementById('hidden-canvas');
const cameraPlaceholder = document.getElementById('camera-placeholder');
const liveIndicator     = document.getElementById('live-indicator');
const landmarksStatus   = document.getElementById('landmarks-status');

const startBtn        = document.getElementById('start-btn');
const stopBtn         = document.getElementById('stop-btn');
const toggleSkeleton  = document.getElementById('toggle-skeleton');
const toggleMirror    = document.getElementById('toggle-mirror');
const speakBtn        = document.getElementById('speak-btn');
const copyBtn         = document.getElementById('copy-btn');

const glossWebcam    = document.getElementById('gloss-webcam');
const textWebcam     = document.getElementById('text-webcam');
const confidenceVal  = document.getElementById('confidence-val');
const confidenceFill = document.getElementById('confidence-fill');

const dropZone          = document.getElementById('drop-zone');
const fileInput         = document.getElementById('file-input');
const browseBtn         = document.getElementById('browse-btn');
const previewContainer  = document.getElementById('preview-container');
const previewVideo      = document.getElementById('preview-video');
const selectedFileName  = document.getElementById('selected-file-name');
const removeFileBtn     = document.getElementById('remove-file-btn');
const processBtn        = document.getElementById('process-btn');
const uploadProgressCard = document.getElementById('upload-progress-card');
const progressStatus    = document.getElementById('progress-status');

const glossUpload      = document.getElementById('gloss-upload');
const textUpload       = document.getElementById('text-upload');
const metricFrames     = document.getElementById('metric-frames');
const metricLatency    = document.getElementById('metric-latency');
const metricConfidence = document.getElementById('metric-confidence');

// ── State ────────────────────────────────────────────────────────────────────
let mediaStream      = null;
let websocket        = null;
let isStreaming      = false;
let handLandmarker   = null;
let drawingUtils     = null;
let animFrameId      = null;
let fallbackInterval = null;
let lastVideoTime    = -1;

let lastValidPredictionTime = 0;
let lastSendTime            = 0;
let predictionBuffer        = [];
let currentVocabMode        = 'combined';

const SEND_THROTTLE_MS = 100; // ~10 fps to backend
const WAITING_TEXT = 'Waiting for a sign...';

// POSE indices we forward (must match backend pipeline)
const POSE_INDICES = [0, 11, 12, 13, 14, 15, 16];

// ── MediaPipe HandLandmarker initialiser ──────────────────────────────────────
async function initHandLandmarker() {
  try {
    const vision = await FilesetResolver.forVisionTasks(MP_CDN);
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
        delegate: 'GPU'
      },
      runningMode:           'VIDEO',
      numHands:              2,
      minHandDetectionConfidence: 0.35,
      minHandPresenceConfidence:  0.35,
      minTrackingConfidence:      0.35
    });
    drawingUtils = new DrawingUtils(overlayCanvas.getContext('2d'));
    console.log('[SignVision] HandLandmarker initialised (GPU).');
    return true;
  } catch (gpuErr) {
    console.warn('[SignVision] GPU delegate failed, falling back to CPU:', gpuErr);
    try {
      const vision = await FilesetResolver.forVisionTasks(MP_CDN);
      handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
          delegate: 'CPU'
        },
        runningMode:           'VIDEO',
        numHands:              2,
        minHandDetectionConfidence: 0.35,
        minHandPresenceConfidence:  0.35,
        minTrackingConfidence:      0.35
      });
      drawingUtils = new DrawingUtils(overlayCanvas.getContext('2d'));
      console.log('[SignVision] HandLandmarker initialised (CPU).');
      return true;
    } catch (err) {
      console.error('[SignVision] HandLandmarker init failed:', err);
      return false;
    }
  }
}

// ── Health check ──────────────────────────────────────────────────────────────
async function checkServerHealth() {
  try {
    const res  = await fetch(apiUrl('/api/health'));
    if (res.ok) {
      const data = await res.json();
      if (data.model_mode === 'transformer') {
        modeBadge.textContent = 'PyTorch Transformer: Online';
        modeBadge.style.borderColor = 'rgba(16,185,129,0.4)';
      } else if (data.model_mode === 'baseline_classifier') {
        modeBadge.textContent = 'Sign Classifier (98.6% Acc): Online';
        modeBadge.style.borderColor = 'rgba(6,182,212,0.4)';
      } else {
        modeBadge.textContent = 'Demo Mode (No Checkpoint)';
        modeBadge.style.borderColor = 'rgba(245,158,11,0.4)';
      }
    }
  } catch {
    systemStatusPill.style.background = 'rgba(244,63,94,0.15)';
    systemStatusPill.style.color = '#f43f5e';
    systemStatusText.textContent = 'Server Offline';
  }
}
checkServerHealth();

// ── Tab switcher ──────────────────────────────────────────────────────────────
tabWebcam.addEventListener('click', () => {
  tabWebcam.classList.add('active');
  tabUpload.classList.remove('active');
  webcamPanel.classList.add('active');
  uploadPanel.classList.remove('active');
});
tabUpload.addEventListener('click', () => {
  tabUpload.classList.add('active');
  tabWebcam.classList.remove('active');
  uploadPanel.classList.add('active');
  webcamPanel.classList.remove('active');
});

// ── Mirror toggle ─────────────────────────────────────────────────────────────
toggleMirror.addEventListener('change', (e) => {
  webcamVideo.classList.toggle('mirror', e.target.checked);
});

// ── Vocabulary Mode Switcher ──────────────────────────────────────────────────
const vocabPills = document.querySelectorAll('.vocab-pill');
const vocabModeHint = document.getElementById('vocab-mode-hint');

vocabPills.forEach(pill => {
  pill.addEventListener('click', () => {
    vocabPills.forEach(p => p.classList.remove('active'));
    pill.classList.add('active');
    currentVocabMode = pill.getAttribute('data-mode') || 'combined';

    if (vocabModeHint) {
      if (currentVocabMode === 'words') {
        vocabModeHint.textContent = 'Recognizing one ASL word (Hello, Yes, Love...)';
      } else if (currentVocabMode === 'numbers') {
        vocabModeHint.textContent = 'Recognizing one number word (Zero–Nine)';
      } else {
        vocabModeHint.textContent = 'One sign, one English word';
      }
    }

    predictionBuffer = [];
    clearWordDisplay();
  });
});

// ── Start Camera ──────────────────────────────────────────────────────────────
startBtn.addEventListener('click', async () => {
  startBtn.disabled = true;
  systemStatusText.textContent = 'Loading MediaPipe...';

  // Init handLandmarker on first use
  if (!handLandmarker) {
    const ok = await initHandLandmarker();
    if (!ok) {
      alert('MediaPipe HandLandmarker failed to load. Will use backend-side detection.');
    }
  }

  try {
    systemStatusText.textContent = 'Starting Camera...';
    mediaStream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30 } },
      audio: false
    });

    webcamVideo.srcObject = mediaStream;
    await new Promise((resolve) => {
      webcamVideo.onloadedmetadata = () => { webcamVideo.play(); resolve(); };
    });

    const vw = webcamVideo.videoWidth  || 640;
    const vh = webcamVideo.videoHeight || 480;
    overlayCanvas.width  = vw;
    overlayCanvas.height = vh;

    const vc = document.querySelector('.video-container');
    if (vc && vw && vh) vc.style.aspectRatio = `${vw} / ${vh}`;

    cameraPlaceholder.classList.add('hidden');
    liveIndicator.classList.remove('hidden');
    stopBtn.disabled = false;

    // WebSocket
    websocket = new WebSocket(wsUrl());

    websocket.onopen = () => {
      isStreaming = true;
      systemStatusText.textContent = 'Live Tracking Active';
      systemStatusPill.className = 'status-pill status-active';
      if (handLandmarker) {
        startLandmarkLoop();
      } else {
        startFallbackLoop();
      }
    };
    websocket.onmessage = (e) => handleSocketMessage(e.data);
    websocket.onclose   = () => { if (isStreaming) stopCamera(); };
    websocket.onerror   = (err) => { console.error('WS error', err); systemStatusText.textContent = 'Socket Error'; };

  } catch (err) {
    console.error('Camera startup error:', err);
    alert('Could not start camera. Check browser permissions.');
    startBtn.disabled = false;
    systemStatusPill.className = 'status-pill status-ready';
    systemStatusText.textContent = 'Camera Blocked';
  }
});

stopBtn.addEventListener('click', stopCamera);

function stopCamera() {
  isStreaming = false;

  if (animFrameId) { cancelAnimationFrame(animFrameId); animFrameId = null; }
  if (fallbackInterval) { clearInterval(fallbackInterval); fallbackInterval = null; }

  if (websocket && websocket.readyState === WebSocket.OPEN) websocket.close();
  websocket = null;

  if (mediaStream) { mediaStream.getTracks().forEach(t => t.stop()); mediaStream = null; }
  webcamVideo.srcObject = null;

  cameraPlaceholder.classList.remove('hidden');
  liveIndicator.classList.add('hidden');
  landmarksStatus.classList.add('hidden');
  startBtn.disabled = false;
  stopBtn.disabled  = true;

  systemStatusPill.className = 'status-pill status-ready';
  systemStatusText.textContent = 'System Ready';

  const ctx = overlayCanvas.getContext('2d');
  ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
}

// ── Main landmark loop (MediaPipe Tasks Vision) ───────────────────────────────
function startLandmarkLoop() {
  const ctx = overlayCanvas.getContext('2d');
  lastVideoTime = -1;

  function loop() {
    if (!isStreaming) return;
    animFrameId = requestAnimationFrame(loop);

    if (webcamVideo.readyState < 2) return;
    if (webcamVideo.currentTime === lastVideoTime) return;
    lastVideoTime = webcamVideo.currentTime;

    // Detect hands
    const nowMs = performance.now();
    let result;
    try {
      result = handLandmarker.detectForVideo(webcamVideo, nowMs);
    } catch (e) {
      console.warn('Landmark detection error:', e);
      return;
    }

    const w = overlayCanvas.width;
    const h = overlayCanvas.height;
    ctx.clearRect(0, 0, w, h);

    const hasHands = result.landmarks && result.landmarks.length > 0;
    landmarksStatus.classList.toggle('hidden', !hasHands);

    // Draw skeleton mesh
    if (toggleSkeleton.checked && hasHands) {
      ctx.save();
      if (toggleMirror.checked) { ctx.translate(w, 0); ctx.scale(-1, 1); }

      for (let hi = 0; hi < result.landmarks.length; hi++) {
        const lms = result.landmarks[hi];
        const handedness = result.handednesses[hi]?.[0]?.categoryName || 'Right';
        const isLeft = handedness === 'Left';

        // Draw connections
        const connColor = isLeft ? 'rgba(16,185,129,0.85)' : 'rgba(139,92,246,0.85)';
        ctx.strokeStyle = connColor;
        ctx.lineWidth   = 2.5;
        for (const [a, b] of HAND_CONNECTIONS) {
          const pa = lms[a], pb = lms[b];
          ctx.beginPath();
          ctx.moveTo(pa.x * w, pa.y * h);
          ctx.lineTo(pb.x * w, pb.y * h);
          ctx.stroke();
        }

        // Draw landmarks
        const dotColor   = isLeft ? '#34d399' : '#c084fc';
        const fillColor  = isLeft ? '#059669' : '#7c3aed';
        for (const lm of lms) {
          ctx.beginPath();
          ctx.arc(lm.x * w, lm.y * h, 4, 0, 2 * Math.PI);
          ctx.fillStyle   = fillColor;
          ctx.fill();
          ctx.strokeStyle = dotColor;
          ctx.lineWidth   = 1.5;
          ctx.stroke();
        }
      }
      ctx.restore();
    }

    // Send keypoints to backend at ~10 fps
    if (nowMs - lastSendTime > SEND_THROTTLE_MS) {
      lastSendTime = nowMs;
      sendKeypointsToBackend(result, hasHands);
    }
  }

  animFrameId = requestAnimationFrame(loop);
}

// ── Build 49-point vector from HandLandmarker result & send via WebSocket ─────
function sendKeypointsToBackend(result, hasHands) {
  if (!websocket || websocket.readyState !== WebSocket.OPEN) return;

  // 7 pose placeholders (NaN — server skips them; hand signals carry the sign)
  const kp49 = Array.from({ length: 7 }, () => [NaN, NaN, NaN]);

  // 21 left + 21 right hand points
  const leftHand  = Array.from({ length: 21 }, () => [NaN, NaN, NaN]);
  const rightHand = Array.from({ length: 21 }, () => [NaN, NaN, NaN]);

  if (result.landmarks) {
    for (let hi = 0; hi < result.landmarks.length; hi++) {
      const lms    = result.landmarks[hi];
      const handed = result.handednesses[hi]?.[0]?.categoryName || 'Right';
      const arr    = handed === 'Left' ? leftHand : rightHand;
      for (let i = 0; i < Math.min(lms.length, 21); i++) {
        arr[i] = [lms[i].x, lms[i].y, lms[i].z || 0];
      }
    }
  }

  const payload = kp49.concat(leftHand, rightHand); // 7+21+21 = 49

  websocket.send(JSON.stringify({
    type: 'keypoints',
    keypoints: payload,
    has_hands: hasHands,
    mode: currentVocabMode
  }));
}

// ── Fallback: send raw JPEG frames when HandLandmarker unavailable ────────────
function startFallbackLoop() {
  console.log('[SignVision] Using backend-side frame detection fallback.');
  const hCtx = hiddenCanvas.getContext('2d');
  hiddenCanvas.width  = 480;
  hiddenCanvas.height = 360;

  fallbackInterval = setInterval(() => {
    if (!isStreaming || !websocket || websocket.readyState !== WebSocket.OPEN) return;
    hCtx.drawImage(webcamVideo, 0, 0, hiddenCanvas.width, hiddenCanvas.height);
    const dataUrl = hiddenCanvas.toDataURL('image/jpeg', 0.65);
    websocket.send(JSON.stringify({ type: 'frame', frame: dataUrl, mode: currentVocabMode }));
  }, 120);
}

// ── WebSocket message handler ─────────────────────────────────────────────────
function handleSocketMessage(rawJson) {
  try {
    const data = JSON.parse(rawJson);

    if (data.type === 'prediction') {
      applyPrediction(data);
    }

    if (data.type === 'frame_result') {
      landmarksStatus.classList.toggle('hidden', !data.has_landmarks);

      // Draw server-side skeleton in fallback mode
      if (toggleSkeleton.checked && data.landmarks && !handLandmarker) {
        drawFallbackSkeleton(data.landmarks);
      }

      if (data.prediction) {
        applyPrediction(data.prediction);
      }
    }
  } catch (err) {
    console.error('WS parse error:', err);
  }
}

// ── Draw server-supplied skeleton (fallback only) ─────────────────────────────
function drawFallbackSkeleton(landmarks) {
  const ctx = overlayCanvas.getContext('2d');
  const w   = overlayCanvas.width;
  const h   = overlayCanvas.height;
  ctx.clearRect(0, 0, w, h);

  function drawPoints(pts, color) {
    if (!pts || pts.length === 0) return;
    ctx.fillStyle = color;
    for (const p of pts) {
      ctx.beginPath();
      ctx.arc(p.x * w, p.y * h, 5, 0, 2 * Math.PI);
      ctx.fill();
    }
  }

  drawPoints(landmarks.left_hand,  '#34d399');
  drawPoints(landmarks.right_hand, '#c084fc');
}

// ── HTML Escaper Helper ───────────────────────────────────────────────────────
function escapeHtml(str) {
  const d = document.createElement('div');
  d.textContent = String(str ?? '');
  return d.innerHTML;
}

function asWord(text) {
  const cleaned = String(text || '').trim().replace(/[.!?]+$/g, '');
  if (!cleaned) return '';
  return cleaned.charAt(0).toUpperCase() + cleaned.slice(1);
}

function clearWordDisplay() {
  glossWebcam.textContent = '—';
  textWebcam.textContent = WAITING_TEXT;
  confidenceVal.textContent = '0%';
  confidenceFill.style.width = '0%';
}

function showWord(gloss, text, digit, conf) {
  let displayHtml = escapeHtml(gloss.toUpperCase());
  if (digit !== null) {
    displayHtml += ` <span class="digit-subbadge">#${escapeHtml(digit)}</span>`;
  }
  glossWebcam.innerHTML = displayHtml;
  textWebcam.textContent = asWord(text || gloss);
  confidenceVal.textContent = `${conf}%`;
  confidenceFill.style.width = `${conf}%`;
}

// ── Show the current word only (majority vote smooths camera jitter) ─────────
function applyPrediction(pred) {
  const gloss = (pred.gloss || '').trim();
  const text = (pred.text || gloss).trim();
  const digit = (pred.digit !== undefined && pred.digit !== null && pred.digit !== '') ? String(pred.digit) : null;
  const conf = Math.round((pred.confidence || 0) * 100);

  if (!gloss || gloss === '...') {
    if (performance.now() - lastValidPredictionTime > 550) {
      predictionBuffer = [];
      clearWordDisplay();
    }
    return;
  }

  lastValidPredictionTime = performance.now();
  predictionBuffer.push({ gloss, text, digit, conf });
  if (predictionBuffer.length > 5) predictionBuffer.shift();

  const counts = {};
  const latest = {};
  for (const item of predictionBuffer) {
    counts[item.gloss] = (counts[item.gloss] || 0) + 1;
    latest[item.gloss] = item;
  }

  let dominantGloss = null;
  let maxCount = 0;
  for (const [g, count] of Object.entries(counts)) {
    if (count > maxCount) {
      maxCount = count;
      dominantGloss = g;
    }
  }

  // Show a word only after it wins two recent frames, so a flicker is not displayed.
  if (dominantGloss && maxCount >= 2 && latest[dominantGloss].conf >= 45) {
    const shown = latest[dominantGloss];
    showWord(shown.gloss, shown.text, shown.digit, shown.conf);
  }
}

copyBtn?.addEventListener('click', async () => {
  const text = textWebcam.textContent;
  if (!text || text === WAITING_TEXT) return;
  try {
    await navigator.clipboard.writeText(text);
    const orig = copyBtn.textContent;
    copyBtn.textContent = 'Copied';
    setTimeout(() => { copyBtn.textContent = orig; }, 1200);
  } catch (e) { console.warn('Clipboard error:', e); }
});

speakBtn?.addEventListener('click', () => {
  const text = textWebcam.textContent;
  if (!text || text === WAITING_TEXT) return;
  if ('speechSynthesis' in window) {
    window.speechSynthesis.cancel();
    const utt = new SpeechSynthesisUtterance(text);
    utt.rate = 0.95;
    window.speechSynthesis.speak(utt);
  }
});

// ── Video file upload ─────────────────────────────────────────────────────────
browseBtn.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('click', () => fileInput.click());

['dragenter', 'dragover'].forEach(ev => {
  dropZone.addEventListener(ev, (e) => { e.preventDefault(); e.stopPropagation(); dropZone.classList.add('dragover'); });
});
['dragleave', 'drop'].forEach(ev => {
  dropZone.addEventListener(ev, (e) => { e.preventDefault(); e.stopPropagation(); dropZone.classList.remove('dragover'); });
});

dropZone.addEventListener('drop', (e) => {
  if (e.dataTransfer.files.length > 0) handleSelectedFile(e.dataTransfer.files[0]);
});

fileInput.addEventListener('change', () => {
  if (fileInput.files.length > 0) handleSelectedFile(fileInput.files[0]);
});

let currentUploadFile = null;

function handleSelectedFile(file) {
  currentUploadFile = file;
  selectedFileName.textContent = file.name;
  previewVideo.src = URL.createObjectURL(file);
  dropZone.classList.add('hidden');
  previewContainer.classList.remove('hidden');
  uploadProgressCard.classList.add('hidden');
}

removeFileBtn.addEventListener('click', () => {
  currentUploadFile = null;
  fileInput.value   = '';
  previewVideo.pause();
  previewVideo.src  = '';
  previewContainer.classList.add('hidden');
  dropZone.classList.remove('hidden');
  uploadProgressCard.classList.add('hidden');
});

processBtn.addEventListener('click', async () => {
  if (!currentUploadFile) return;
  processBtn.disabled = true;
  uploadProgressCard.classList.remove('hidden');
  progressStatus.textContent = 'Recognizing one word...';

  const fd = new FormData();
  fd.append('file', currentUploadFile);
  fd.append('mode', currentVocabMode);

  try {
    const res    = await fetch(apiUrl('/api/upload'), { method: 'POST', body: fd });
    const result = await res.json();

    if (!res.ok || result.error) throw new Error(result.error || 'Failed to process video');

    const glossText = (result.gloss || '').trim();
    if (!glossText || glossText === '...') {
      glossUpload.textContent = '—';
      textUpload.textContent = 'No confident word';
    } else if (result.digit !== null && result.digit !== undefined) {
      glossUpload.innerHTML = `${escapeHtml(glossText.toUpperCase())} <span class="digit-subbadge">#${escapeHtml(result.digit)}</span>`;
      textUpload.textContent = asWord(result.text || glossText);
    } else {
      glossUpload.textContent = glossText.toUpperCase();
      textUpload.textContent = asWord(result.text || glossText);
    }
    metricFrames.textContent     = result.total_frames || '0';
    metricLatency.textContent    = `${result.processing_time_sec}s`;
    metricConfidence.textContent = `${Math.round((result.confidence || 0.9) * 100)}%`;

    progressStatus.textContent = 'Word recognized';
    setTimeout(() => { uploadProgressCard.classList.add('hidden'); processBtn.disabled = false; }, 1200);
  } catch (err) {
    console.error('Upload error:', err);
    alert(`Error: ${err.message}`);
    uploadProgressCard.classList.add('hidden');
    processBtn.disabled = false;
  }
});
