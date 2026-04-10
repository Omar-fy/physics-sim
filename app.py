"""
PHYSIM — Flask REST API
=======================
Exposes a clean REST interface over the three physics engines.

Endpoints
---------
GET  /                        → serve frontend
GET  /api/status              → server health
POST /api/simulate            → create / replace simulation
GET  /api/state               → current entity positions
GET  /api/metrics             → live physics metrics
PUT  /api/params              → hot-update simulation parameters
POST /api/step                → advance N ticks (headless / scripting)
POST /api/reset               → reinitialise current simulation
POST /api/interact            → inject mouse force
"""

from __future__ import annotations

import time
from flask import Flask, jsonify, request, render_template, abort
from flask_cors import CORS

from core import (
    ParticleSystem, ParticleConfig,
    FlockSystem,    FlockConfig,
    GravitySystem,  GravityConfig,
)
from config import DEFAULT_WIDTH, DEFAULT_HEIGHT, DEBUG

app = Flask(__name__)
CORS(app)

# ── Simulation registry ────────────────────────────────────────────────────────
_sim: ParticleSystem | FlockSystem | GravitySystem | None = None
_mode: str = "particles"
_tick: int = 0
_started: float = time.time()

_mode_map = {
    "particles": (ParticleSystem, ParticleConfig),
    "flocking":  (FlockSystem,    FlockConfig),
    "gravity":   (GravitySystem,  GravityConfig),
}


def _get_sim():
    global _sim
    if _sim is None:
        _sim = ParticleSystem(DEFAULT_WIDTH, DEFAULT_HEIGHT)
    return _sim


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/status")
def status():
    return jsonify({
        "server":  "PHYSIM v2.4.1",
        "status":  "running",
        "mode":    _mode,
        "tick":    _tick,
        "uptime":  round(time.time() - _started, 1),
        "modes":   list(_mode_map.keys()),
    })


@app.post("/api/simulate")
def create_simulation():
    """Create or replace the active simulation."""
    global _sim, _mode, _tick
    body = request.get_json(silent=True) or {}
    mode = body.get("mode", _mode)

    if mode not in _mode_map:
        abort(400, f"Unknown mode '{mode}'. Choose from: {list(_mode_map.keys())}")

    SimClass, CfgClass = _mode_map[mode]
    cfg_fields = {k: v for k, v in body.items() if k != "mode"}

    try:
        cfg  = CfgClass(**{k: v for k, v in cfg_fields.items() if hasattr(CfgClass(), k)})
        _sim = SimClass(DEFAULT_WIDTH, DEFAULT_HEIGHT, cfg)
    except (TypeError, ValueError) as exc:
        abort(422, str(exc))

    _mode = mode
    _tick = 0
    return jsonify({"status": "created", "mode": mode}), 201


@app.get("/api/state")
def get_state():
    return jsonify(_get_sim().state_json())


@app.get("/api/metrics")
def get_metrics():
    m = _get_sim().metrics()
    m["tick"] = _tick
    m["mode"] = _mode
    return jsonify(m)


@app.put("/api/params")
def update_params():
    global _sim, _mode
    body = request.get_json(silent=True) or {}

    new_mode = body.pop("mode", None)
    if new_mode and new_mode != _mode:
        # Mode switch — rebuild sim
        if new_mode not in _mode_map:
            abort(400, f"Unknown mode '{new_mode}'")
        SimClass, CfgClass = _mode_map[new_mode]
        _sim  = SimClass(DEFAULT_WIDTH, DEFAULT_HEIGHT)
        _mode = new_mode

    if body:
        _get_sim().apply_config(body)

    return jsonify({"status": "updated", "mode": _mode})


@app.post("/api/step")
def advance():
    """Advance simulation N ticks server-side (useful for scripting / testing)."""
    global _tick
    body  = request.get_json(silent=True) or {}
    n     = int(body.get("n", 1))
    mouse = body.get("mouse")          # optional [x, y]
    sim   = _get_sim()

    for _ in range(n):
        if _mode == "particles":
            sim.step(mouse_pos=mouse)
        elif _mode == "flocking":
            sim.step(mouse_pos=mouse)
        else:
            sim.step(mouse_pos=mouse)
        _tick += 1

    return jsonify({"status": "stepped", "ticks": n, "total_tick": _tick,
                    **sim.metrics()})


@app.post("/api/reset")
def reset():
    global _sim, _tick
    SimClass, CfgClass = _mode_map[_mode]
    _sim  = SimClass(DEFAULT_WIDTH, DEFAULT_HEIGHT)
    _tick = 0
    return jsonify({"status": "reset", "mode": _mode})


@app.post("/api/interact")
def interact():
    """Inject a single mouse-force event."""
    global _tick
    body    = request.get_json(silent=True) or {}
    mx, my  = body.get("x", 0), body.get("y", 0)
    attract = body.get("attract", True)
    sim     = _get_sim()

    if _mode == "particles":
        sim.step(mouse_pos=(mx, my), attract=attract)
    else:
        sim.step(mouse_pos=(mx, my))

    _tick += 1
    return jsonify({"status": "ok", "tick": _tick})


# ── Error handlers ─────────────────────────────────────────────────────────────

@app.errorhandler(400)
@app.errorhandler(422)
def bad_request(e):
    return jsonify({"error": str(e.description)}), e.code


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=DEBUG, port=5000, host="0.0.0.0")
