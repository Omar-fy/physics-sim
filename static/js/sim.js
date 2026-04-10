/**
 * PHYSIM — Frontend Renderer
 *
 * Polls the Flask REST API for simulation state on every animation frame,
 * renders to Canvas 2D, and sends user interaction events back to the server.
 *
 * API calls:
 *   POST /api/simulate   — create new simulation
 *   GET  /api/state      — fetch entity positions each frame
 *   GET  /api/metrics    — fetch physics metrics (~2 Hz)
 *   PUT  /api/params     — push slider updates
 *   POST /api/reset      — reinitialise
 *   POST /api/interact   — send mouse force
 */

'use strict';

const BASE = '';   // same-origin; Flask serves on port 5000

// ── Canvas setup ──────────────────────────────────────────────────────────────
const canvas = document.getElementById('sim-canvas');
const ctx    = canvas.getContext('2d');

function resizeCanvas() {
  const wrap = document.getElementById('canvas-wrap');
  canvas.width  = wrap.clientWidth;
  canvas.height = wrap.clientHeight;
}
window.addEventListener('resize', resizeCanvas);
resizeCanvas();

// ── State ─────────────────────────────────────────────────────────────────────
let mode       = 'particles';
let paused     = false;
let simState   = null;
let curPalette = 0;
let localTick  = 0;

const opts = { trails: true, glow: true, velcolor: true };

const palettes = [
  ['#00e5ff','#0080ff','#a259ff'],
  ['#ff3d71','#ff6b35','#ffaa00'],
  ['#a259ff','#7c3aed','#ec4899'],
  ['#39ff14','#00ff88','#00e5ff'],
  ['#ff9500','#ff3d71','#ffdd00'],
];

// FPS tracking
let fpsCount = 0, fpsTime = performance.now(), fps = 0;

// ── Parameter schemas per mode ────────────────────────────────────────────────
const PARAMS = {
  particles: [
    { key:'count',   label:'Count',   min:10,   max:800,  step:1,    def:300,  cls:''   },
    { key:'speed',   label:'Speed',   min:0.5,  max:10,   step:0.1,  def:2.0,  cls:''   },
    { key:'size',    label:'Size',    min:0.5,  max:8,    step:0.1,  def:2.5,  cls:'r2' },
    { key:'decay',   label:'Decay',   min:0.7,  max:1.0,  step:0.01, def:0.96, cls:'r3' },
    { key:'gravity', label:'Gravity', min:-0.5, max:0.5,  step:0.01, def:0.05, cls:'r4' },
  ],
  flocking: [
    { key:'count',      label:'Boids',      min:10,  max:400, step:1,    def:150, cls:''   },
    { key:'cohesion',   label:'Cohesion',   min:0,   max:2,   step:0.05, def:0.5, cls:'r2' },
    { key:'separation', label:'Separation', min:0,   max:3,   step:0.05, def:1.5, cls:'r3' },
    { key:'alignment',  label:'Alignment',  min:0,   max:2,   step:0.05, def:1.0, cls:'r4' },
    { key:'perception', label:'Perception', min:10,  max:150, step:1,    def:50,  cls:''   },
  ],
  gravity: [
    { key:'n_bodies',  label:'Bodies',    min:2,  max:12, step:1,   def:5,    cls:''   },
    { key:'G',         label:'G Constant',min:0.1,max:5,  step:0.1, def:1.0,  cls:'r2' },
    { key:'softening', label:'Softening', min:1,  max:50, step:1,   def:10,   cls:'r3' },
    { key:'trail_len', label:'Trails',    min:0,  max:200,step:1,   def:60,   cls:'r4' },
  ],
};

const CANVAS_INFO = {
  particles: 'MODE: PARTICLES\nCLICK: Attract\nRIGHT: Repel',
  flocking:  'MODE: FLOCKING\nCLICK: Guide flock\nDRAG: Lead boids',
  gravity:   'MODE: N-BODY\nCLICK: Attract bodies\nDRAG: Pull',
};

// ── API helpers ───────────────────────────────────────────────────────────────
async function api(method, path, body) {
  const opts = { method, headers: { 'Content-Type': 'application/json' } };
  if (body) opts.body = JSON.stringify(body);
  const t0  = performance.now();
  const res = await fetch(BASE + path, opts);
  const ms  = (performance.now() - t0).toFixed(0);
  const data = await res.json().catch(() => ({}));
  log(method === 'GET' ? 'out' : 'out', `${method} ${path}`);
  log('in', `${res.status} OK  (${ms}ms)  ${JSON.stringify(data).slice(0,80)}`);
  return data;
}

// ── Log ───────────────────────────────────────────────────────────────────────
function log(type, msg) {
  const el  = document.createElement('div');
  const ts  = new Date().toTimeString().slice(0, 8);
  el.className   = `log-entry ${type}`;
  el.textContent = `[${ts}] ${msg}`;
  const container = document.getElementById('api-log');
  container.appendChild(el);
  container.scrollTop = container.scrollHeight;
  if (container.children.length > 60) container.firstChild.remove();
}

// ── Simulation create / reset ─────────────────────────────────────────────────
async function createSim(m) {
  mode = m || mode;
  await api('POST', '/api/simulate', { mode });
  buildParamControls();
  document.getElementById('canvas-info').textContent = CANVAS_INFO[mode];
  document.getElementById('footer-mode').textContent = 'MODE: ' + mode.toUpperCase();
  document.querySelectorAll('.mode-tab').forEach(t => {
    t.classList.toggle('active', t.dataset.mode === mode);
  });
}

async function resetSim() {
  await api('POST', '/api/reset');
}

// ── Parameter controls ────────────────────────────────────────────────────────
function buildParamControls() {
  const container = document.getElementById('param-controls');
  container.innerHTML = '';
  (PARAMS[mode] || []).forEach(p => {
    const valId = `val-${p.key}`;
    const sldId = `sld-${p.key}`;
    container.innerHTML += `
      <div class="ctrl-row">
        <div class="ctrl-label">${p.label}</div>
        <div class="ctrl-value" id="${valId}">${p.def}</div>
      </div>
      <input type="range" id="${sldId}" class="${p.cls}"
        min="${p.min}" max="${p.max}" step="${p.step}" value="${p.def}">
    `;
  });
  // Wire sliders
  (PARAMS[mode] || []).forEach(p => {
    const sld = document.getElementById(`sld-${p.key}`);
    const val = document.getElementById(`val-${p.key}`);
    sld.addEventListener('input', () => {
      val.textContent = sld.value;
      pushParams();
    });
  });
}

function currentParams() {
  const out = {};
  (PARAMS[mode] || []).forEach(p => {
    const el = document.getElementById(`sld-${p.key}`);
    if (el) out[p.key] = parseFloat(el.value);
  });
  return out;
}

let paramDebounce;
function pushParams() {
  clearTimeout(paramDebounce);
  paramDebounce = setTimeout(async () => {
    await api('PUT', '/api/params', currentParams());
  }, 250);
}

// ── Metrics poll ──────────────────────────────────────────────────────────────
async function pollMetrics() {
  try {
    const m = await fetch(BASE + '/api/metrics').then(r => r.json());
    document.getElementById('m-entities').textContent = m.count ?? '–';
    document.getElementById('m-speed').textContent    = (m.avg_speed ?? 0).toFixed(2);
    document.getElementById('m-ke').textContent       = (m.kinetic_e ?? 0).toFixed(0);
    document.getElementById('m-tick').textContent     = m.tick ?? '–';
    document.getElementById('footer-tick').textContent = 'TICK: ' + (m.tick ?? 0);
  } catch { /* server might be restarting */ }
}
setInterval(pollMetrics, 500);

// ── State fetch + render loop ─────────────────────────────────────────────────
let rendering = false;

async function fetchAndRender() {
  if (paused) { requestAnimationFrame(fetchAndRender); return; }

  const t0 = performance.now();

  try {
    const state = await fetch(BASE + '/api/state').then(r => r.json());
    simState = state;
  } catch { /* skip frame */ }

  if (simState) render(simState);

  const framems = (performance.now() - t0).toFixed(1);
  document.getElementById('m-framems').textContent = framems + ' ms';

  fpsCount++;
  const now = performance.now();
  if (now - fpsTime >= 500) {
    fps = Math.round(fpsCount * 1000 / (now - fpsTime));
    fpsCount = 0; fpsTime = now;
    document.getElementById('m-fps').textContent = fps;
    document.getElementById('footer-fps').textContent = 'FPS: ' + fps;
  }

  localTick++;
  requestAnimationFrame(fetchAndRender);
}

// ── Renderer ──────────────────────────────────────────────────────────────────
function render(state) {
  const pal = palettes[curPalette];

  if (opts.trails) {
    ctx.fillStyle = 'rgba(5,8,16,0.18)';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  } else {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  }

  if (mode === 'particles' && state.positions) {
    const pos = state.positions, vel = state.velocities, hues = state.hues;
    const sz  = state.size || 2.5;
    for (let i = 0; i < pos.length; i++) {
      const [x, y]   = pos[i];
      const [vx, vy] = vel ? vel[i] : [0, 0];
      const spd = Math.sqrt(vx*vx + vy*vy);
      const color = opts.velcolor
        ? `hsl(${200 + spd * 20},100%,60%)`
        : pal[Math.floor((hues[i] || 0) / 120) % 3];
      ctx.shadowBlur  = opts.glow ? 8 : 0;
      ctx.shadowColor = color;
      ctx.fillStyle   = color;
      ctx.beginPath();
      ctx.arc(x, y, sz, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  else if (mode === 'flocking' && state.positions) {
    const pos  = state.positions;
    const hdgs = state.headings || [];
    const spds = state.speeds   || [];
    for (let i = 0; i < pos.length; i++) {
      const [x, y] = pos[i];
      const ang    = hdgs[i] || 0;
      const spd    = spds[i] || 2;
      const sz     = 6;
      const color  = opts.velcolor ? `hsl(${180 + spd * 30},100%,60%)` : pal[0];
      ctx.shadowBlur  = opts.glow ? 6 : 0;
      ctx.shadowColor = color;
      ctx.fillStyle   = color;
      ctx.beginPath();
      ctx.moveTo(x + Math.cos(ang)*sz*1.5,  y + Math.sin(ang)*sz*1.5);
      ctx.lineTo(x + Math.cos(ang+2.4)*sz,  y + Math.sin(ang+2.4)*sz);
      ctx.lineTo(x + Math.cos(ang-2.4)*sz,  y + Math.sin(ang-2.4)*sz);
      ctx.closePath();
      ctx.fill();
    }
  }

  else if (mode === 'gravity' && state.positions) {
    const pos    = state.positions;
    const masses = state.masses  || [];
    const trails = state.trails  || [];
    // Draw trails
    if (opts.trails) {
      trails.forEach((trail, i) => {
        const color = pal[i % pal.length];
        for (let t = 1; t < trail.length; t++) {
          const alpha = t / trail.length;
          ctx.beginPath();
          ctx.strokeStyle = color + Math.round(alpha * 150).toString(16).padStart(2,'0');
          ctx.lineWidth   = 1;
          ctx.moveTo(trail[t-1][0], trail[t-1][1]);
          ctx.lineTo(trail[t][0],   trail[t][1]);
          ctx.stroke();
        }
      });
    }
    // Draw bodies
    pos.forEach(([x, y], i) => {
      const r     = Math.sqrt(masses[i] || 10) * 2;
      const color = pal[i % pal.length];
      ctx.shadowBlur  = opts.glow ? 20 : 0;
      ctx.shadowColor = color;
      ctx.fillStyle   = color;
      ctx.beginPath();
      ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.fill();
    });
  }

  ctx.shadowBlur = 0;
}

// ── Mouse interaction ─────────────────────────────────────────────────────────
let mouseX = 0, mouseY = 0, mouseDown = false, mouseButton = 0;

canvas.addEventListener('mousemove', e => {
  const r = canvas.getBoundingClientRect();
  mouseX = (e.clientX - r.left) * (canvas.width  / r.width);
  mouseY = (e.clientY - r.top)  * (canvas.height / r.height);
});

canvas.addEventListener('mousedown', e => {
  mouseDown = true; mouseButton = e.button;
  api('POST', '/api/interact', { x: mouseX, y: mouseY, attract: e.button === 0 });
});
canvas.addEventListener('mouseup',        () => mouseDown = false);
canvas.addEventListener('contextmenu',    e  => e.preventDefault());

// ── UI wiring ─────────────────────────────────────────────────────────────────
document.querySelectorAll('.mode-tab').forEach(tab => {
  tab.addEventListener('click', () => createSim(tab.dataset.mode));
});

document.getElementById('btn-reset').addEventListener('click', resetSim);
document.getElementById('btn-pause').addEventListener('click', () => {
  paused = !paused;
  document.getElementById('btn-pause').textContent = paused ? 'Resume' : 'Pause';
});
document.getElementById('btn-clear').addEventListener('click', () => {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  simState = null;
});

// Toggles
document.querySelectorAll('.toggle').forEach(tog => {
  tog.addEventListener('click', () => {
    const key = tog.dataset.key;
    opts[key] = !opts[key];
    tog.classList.toggle('on', opts[key]);
  });
});

// Palettes
document.querySelectorAll('.color-swatch').forEach(sw => {
  sw.addEventListener('click', () => {
    curPalette = parseInt(sw.dataset.idx);
    document.querySelectorAll('.color-swatch').forEach((s, i) =>
      s.classList.toggle('active', i === curPalette));
  });
});

// API request builder
document.getElementById('btn-send').addEventListener('click', async () => {
  try {
    const body = JSON.parse(document.getElementById('api-input').value || '{}');
    const path = body.mode ? '/api/simulate' : '/api/params';
    const meth = body.mode ? 'POST' : 'PUT';
    const res  = await api(meth, path, body);
    if (body.mode) { mode = body.mode; buildParamControls(); }
  } catch { log('err', '400 Bad Request: Invalid JSON'); }
});

document.getElementById('btn-metrics').addEventListener('click', async () => {
  const m = await api('GET', '/api/metrics');
  document.getElementById('api-input').value = JSON.stringify(m, null, 2);
});

// ── Server health ─────────────────────────────────────────────────────────────
async function checkServer() {
  try {
    const s   = await fetch(BASE + '/api/status').then(r => r.json());
    const badge = document.getElementById('server-badge');
    badge.textContent = '● CONNECTED';
    badge.classList.add('ok');
    log('in', `Server ready — ${s.server}`);
  } catch {
    log('err', 'Cannot reach Flask server on :5000');
  }
}

// ── Boot ──────────────────────────────────────────────────────────────────────
(async () => {
  await checkServer();
  await createSim('particles');
  fetchAndRender();
})();
