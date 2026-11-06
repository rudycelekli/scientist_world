"""Swarm dashboard — Flask web UI + REST API.

Provides a status dashboard and REST API for managing the queue of
papers being built into environments. Mirrors the pattern from
nexus/daemon.py and hive/dashboard.py.

The API has no authentication: whoever can reach the port can submit,
cancel and pause papers. The daemon therefore binds to the interface
named by ``dashboard_host`` in the config, loopback by default.
"""

from __future__ import annotations

import logging
import time

from flask import Flask, jsonify, request, render_template_string

from swarm.config import SwarmConfig, load_config

log = logging.getLogger(__name__)

_daemon_start_time = 0.0


def create_app(
    config: SwarmConfig | None = None,
    scheduler=None,
) -> Flask:
    """Create the Flask application.

    Args:
        config: Swarm configuration. Loaded from disk if None.
        scheduler: Scheduler instance. If None, the app provides
            read-only endpoints only.

    Returns:
        Configured Flask app.
    """
    global _daemon_start_time
    _daemon_start_time = time.time()

    if config is None:
        config = load_config()

    app = Flask(__name__)
    app.config["JSONIFY_PRETTYPRINT_REGULAR"] = True

    # ── HTML dashboard ────────────────────────────────────────────

    @app.route("/")
    def index():
        return render_template_string(DASHBOARD_HTML)

    # ── REST API ──────────────────────────────────────────────────

    @app.route("/api/status")
    def api_status():
        if scheduler is None:
            return jsonify({"error": "no scheduler"}), 503
        try:
            status = scheduler.get_status()
            status["uptime"] = time.time() - _daemon_start_time
            status["nexus_url"] = config.nexus.url
            return jsonify(status)
        except Exception as e:
            log.exception("Error in /api/status")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/caps", methods=["GET", "POST", "DELETE"])
    def api_caps():
        if scheduler is None:
            return jsonify({"error": "no scheduler"}), 503
        try:
            if request.method == "GET":
                return jsonify(scheduler.get_caps())
            if request.method == "DELETE":
                bucket = request.args.get("bucket")
                if bucket:
                    return jsonify(scheduler.set_cap(bucket, None))
                return jsonify(scheduler.clear_caps())
            # POST: {"bucket": "<name>", "value": <int|null>}
            # or {"caps": {"bucket": value, ...}} for batch.
            data = request.get_json(force=True) or {}
            if "caps" in data:
                state = scheduler.get_caps()
                for b, v in (data.get("caps") or {}).items():
                    state = scheduler.set_cap(b, None if v is None else int(v))
                return jsonify(state)
            bucket = data.get("bucket")
            if not bucket:
                return jsonify({"error": "bucket required"}), 400
            raw = data.get("value")
            value = None if raw is None else int(raw)
            return jsonify(scheduler.set_cap(bucket, value))
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            log.exception("Error in /api/caps")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/papers")
    def api_papers():
        if scheduler is None:
            return jsonify({"error": "no scheduler"}), 503
        try:
            return jsonify(scheduler.get_papers())
        except Exception as e:
            log.exception("Error in /api/papers")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/dashboard")
    def api_dashboard():
        if scheduler is None:
            return jsonify({"error": "no scheduler"}), 503
        try:
            status = scheduler.get_status()
            status["uptime"] = time.time() - _daemon_start_time
            status["nexus_url"] = config.nexus.url
            papers = scheduler.get_papers()
            return jsonify({"status": status, "papers": papers})
        except Exception as e:
            log.exception("Error in /api/dashboard")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/papers/<arxiv_id>")
    def api_paper(arxiv_id):
        if scheduler is None:
            return jsonify({"error": "no scheduler"}), 503
        try:
            paper = scheduler.get_paper(arxiv_id)
            if paper is None:
                return jsonify({"error": "not found"}), 404
            return jsonify(paper)
        except Exception as e:
            log.exception("Error in /api/papers/%s", arxiv_id)
            return jsonify({"error": str(e)}), 500

    @app.route("/api/papers", methods=["POST"])
    def api_submit_paper():
        if scheduler is None:
            return jsonify({"error": "no scheduler"}), 503
        try:
            data = request.get_json(force=True) or {}
            arxiv_id = data.get("arxiv_id")
            if not arxiv_id:
                return jsonify({"error": "arxiv_id is required"}), 400

            gpu_budget = data.get("gpu_budget")
            date = data.get("date")
            topics = data.get("topics")
            profile = data.get("profile")
            workflow = data.get("workflow")
            job = scheduler.enqueue(arxiv_id, gpu_budget=gpu_budget, date=date, topics=topics, profile=profile, workflow=workflow)
            return jsonify({"arxiv_id": job.arxiv_id, "status": job.status}), 201
        except ValueError as e:
            return jsonify({"error": str(e)}), 409
        except FileNotFoundError as e:
            return jsonify({"error": str(e)}), 404
        except Exception as e:
            log.exception("Error in POST /api/papers")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/papers/<arxiv_id>/continue", methods=["POST"])
    def api_continue_paper(arxiv_id):
        if scheduler is None:
            return jsonify({"error": "no scheduler"}), 503
        try:
            data = request.get_json(force=True) if request.data else {}
            profile = data.get("profile") if data else None
            no_workflow = data.get("no_workflow", False) if data else False
            fresh = data.get("fresh", False) if data else False
            queue_front = data.get("queue_front", False) if data else False
            job = scheduler.continue_paper(arxiv_id, profile=profile,
                                           no_workflow=no_workflow, fresh=fresh,
                                           queue_front=queue_front)
            return jsonify({"arxiv_id": job.arxiv_id, "status": job.status}), 200
        except ValueError as e:
            return jsonify({"error": str(e)}), 409
        except FileNotFoundError as e:
            return jsonify({"error": str(e)}), 404
        except Exception as e:
            log.exception("Error in POST /api/papers/%s/continue", arxiv_id)
            return jsonify({"error": str(e)}), 500

    @app.route("/api/papers/<arxiv_id>/recover", methods=["POST"])
    def api_recover_paper(arxiv_id):
        if scheduler is None:
            return jsonify({"error": "no scheduler"}), 503
        try:
            data = request.get_json(force=True) if request.data else {}
            profile = data.get("profile") if data else None
            no_workflow = data.get("no_workflow", False) if data else False
            fresh = data.get("fresh", False) if data else False
            job = scheduler.recover_paper(
                arxiv_id, profile=profile, no_workflow=no_workflow, fresh=fresh,
            )
            return jsonify({"arxiv_id": job.arxiv_id, "status": job.status}), 200
        except ValueError as e:
            return jsonify({"error": str(e)}), 409
        except FileNotFoundError as e:
            return jsonify({"error": str(e)}), 404
        except Exception as e:
            log.exception("Error in POST /api/papers/%s/recover", arxiv_id)
            return jsonify({"error": str(e)}), 500

    @app.route("/api/recover-all", methods=["POST"])
    def api_recover_all():
        if scheduler is None:
            return jsonify({"error": "no scheduler"}), 503
        try:
            data = request.get_json(force=True) if request.data else {}
            profile = data.get("profile") if data else None
            no_workflow = data.get("no_workflow", False) if data else False
            fresh = data.get("fresh", False) if data else False
            result = scheduler.recover_all_papers(
                profile=profile, no_workflow=no_workflow, fresh=fresh,
            )
            return jsonify(result), 200
        except Exception as e:
            log.exception("Error in POST /api/recover-all")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/papers/<arxiv_id>/iterate", methods=["POST"])
    def api_iterate_paper(arxiv_id):
        if scheduler is None:
            return jsonify({"error": "no scheduler"}), 503
        try:
            data = request.get_json(force=True) if request.data else {}
            profile = data.get("profile") if data else None
            role = data.get("role", "reproducer") if data else "reproducer"
            gpu_budget = data.get("gpu_budget") if data else None
            workflow_step = data.get("workflow_step") if data else None
            workflow = data.get("workflow") if data else None
            queue_front = data.get("queue_front", False) if data else False
            job = scheduler.iterate_paper(
                arxiv_id,
                profile=profile,
                role=role,
                gpu_budget=gpu_budget,
                workflow_step_override=workflow_step,
                workflow_override=workflow,
                queue_front=queue_front,
            )
            return jsonify({"arxiv_id": job.arxiv_id, "status": job.status}), 200
        except ValueError as e:
            return jsonify({"error": str(e)}), 409
        except FileNotFoundError as e:
            return jsonify({"error": str(e)}), 404
        except Exception as e:
            log.exception("Error in POST /api/papers/%s/iterate", arxiv_id)
            return jsonify({"error": str(e)}), 500

    @app.route("/api/papers/<arxiv_id>/cancel", methods=["POST"])
    def api_cancel_paper(arxiv_id):
        if scheduler is None:
            return jsonify({"error": "no scheduler"}), 503
        try:
            ok = scheduler.cancel_paper(arxiv_id)
            if ok:
                return jsonify({"arxiv_id": arxiv_id, "status": "cancelling"}), 200
            return jsonify({"error": f"Paper {arxiv_id} not active or queued"}), 404
        except Exception as e:
            log.exception("Error in POST /api/papers/%s/cancel", arxiv_id)
            return jsonify({"error": str(e)}), 500

    @app.route("/api/pause-all", methods=["POST"])
    def api_pause_all():
        if scheduler is None:
            return jsonify({"error": "no scheduler"}), 503
        try:
            counts = scheduler.prepare_all_for_restart()
            return jsonify({
                "status": "prepared",
                "queued": counts["queued"],
                "active": counts["active"],
            }), 200
        except Exception as e:
            log.exception("Error in POST /api/pause-all")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/workers")
    def api_workers():
        if scheduler is None:
            return jsonify({"error": "no scheduler"}), 503
        try:
            status = scheduler.get_status()
            return jsonify({
                "pool_size": status["pool_size"],
                "max_working_workers": status["max_working_workers"],
                "max_working_return_overflow": status["max_working_return_overflow"],
                "max_admitted_workers": status["max_admitted_workers"],
                "max_building_workers": status["max_building_workers"],
                "max_installing_workers": status["max_installing_workers"],
                "max_runtime_workers": status["max_runtime_workers"],
                "working_count": status["working_count"],
                "admitted_count": status["admitted_count"],
                "waiting_count": status["waiting_count"],
                "building_count": status["building_count"],
                "installing_count": status["installing_count"],
                "runtime_count": status["runtime_count"],
                "split_runtime_caps": status["split_runtime_caps"],
                "active": status["active_count"],
                "idle": status["pool_size"] - status["active_count"],
            })
        except Exception as e:
            log.exception("Error in /api/workers")
            return jsonify({"error": str(e)}), 500

    return app


# ── Dashboard HTML ────────────────────────────────────────────────────────

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Swarm Dashboard</title>
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
    --bg: #fdf6ec;
    --card: #fffcf7;
    --border: #e8dcc8;
    --accent: #d97706;
    --accent-light: #f59e0b;
    --success: #16a34a;
    --warning: #d97706;
    --error: #dc2626;
    --blue: #2563eb;
    --amber: #d97706;
    --gray: #64748b;
    --text: #1c1917;
    --text2: #57534e;
    --text3: #a8a29e;
    --radius: 10px;
    --shadow: 0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.04);
}

body {
    background: var(--bg);
    color: var(--text);
    font-family: system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif;
    line-height: 1.5;
    min-height: 100vh;
}

/* Header */
.header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 16px 24px;
    background: var(--card);
    border-bottom: 1px solid var(--border);
    position: sticky;
    top: 0;
    z-index: 100;
}

.header-left {
    display: flex;
    align-items: center;
    gap: 16px;
}

.logo {
    font-size: 22px;
    font-weight: 700;
    letter-spacing: 3px;
    color: var(--accent);
    font-family: 'Courier New', monospace;
}

.logo-sub {
    font-size: 12px;
    color: var(--text2);
    font-weight: 400;
}

.header-pills {
    display: flex;
    gap: 10px;
    font-size: 12px;
}

.pill {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 4px 10px;
    border-radius: 12px;
    font-weight: 500;
    font-family: 'Courier New', monospace;
}

.pill-ok { background: rgba(22,163,74,0.12); color: var(--success); }
.pill-err { background: rgba(220,38,38,0.12); color: var(--error); }
.pill-neutral { background: rgba(217,119,6,0.1); color: var(--accent); }

.pill-dot {
    width: 6px; height: 6px; border-radius: 50%;
}
.pill-ok .pill-dot { background: var(--success); }
.pill-err .pill-dot { background: var(--error); }

.header-right {
    display: flex;
    align-items: center;
    gap: 16px;
    font-size: 13px;
    color: var(--text2);
}

.refresh-indicator {
    display: flex;
    align-items: center;
    gap: 6px;
}

.pulse-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: var(--success);
    animation: pulse 2s ease-in-out infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.4; transform: scale(0.8); }
}

.pulse-dot.error { background: var(--error); animation: none; }

/* Error banner */
.error-banner {
    display: none;
    padding: 10px 24px;
    background: rgba(239, 68, 68, 0.15);
    border-bottom: 1px solid rgba(239, 68, 68, 0.3);
    color: var(--error);
    font-size: 13px;
}
.error-banner.visible { display: block; }

/* Main */
.main {
    max-width: 1400px;
    margin: 0 auto;
    padding: 24px;
}

/* Overview cards */
.cards {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 16px;
    margin-bottom: 28px;
}

.card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 18px;
    box-shadow: var(--shadow);
}

.card-label {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--text2);
    margin-bottom: 6px;
}

.card-value {
    font-size: 28px;
    font-weight: 700;
    font-family: 'Courier New', monospace;
}

.card-sub {
    font-size: 12px;
    color: var(--text3);
    margin-top: 4px;
}

.tag {
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: var(--text3);
    vertical-align: middle;
}

/* Sections */
.section {
    margin-bottom: 28px;
}

.section-header {
    display: flex;
    align-items: baseline;
    gap: 8px;
    margin-bottom: 12px;
}

.section-title {
    font-size: 14px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--text2);
}

.section-count {
    font-size: 13px;
    color: var(--text3);
    font-family: 'Courier New', monospace;
}

/* Badges */
.badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.badge-queued { background: rgba(217,119,6,0.15); color: var(--amber); }
.badge-active { background: rgba(37,99,235,0.15); color: var(--blue); }
.badge-waiting { background: rgba(217,119,6,0.15); color: var(--amber); }
.badge-building { background: rgba(124,58,237,0.15); color: #7c3aed; }
.badge-installing { background: rgba(6,182,212,0.15); color: #0891b2; }
.badge-working { background: rgba(37,99,235,0.15); color: var(--blue); }
.badge-wrapping_up { background: rgba(14,165,233,0.15); color: #0284c7; }
.badge-done { background: rgba(22,163,74,0.15); color: var(--success); }
.badge-error { background: rgba(220,38,38,0.15); color: var(--error); }
.badge-idle { background: rgba(100,116,139,0.15); color: var(--gray); }
/* Applied only to failure-retry blocked badges (e.g. install failed,
   bouncing back to working). Normal forward-flow blocked badges are
   left untouched — capacity pressure isn't interesting visually. */
.badge-blocked {
    position: relative;
    padding-right: 12px;
    box-shadow: inset -3px 0 0 #f59e0b;
}
.badge-blocked::after {
    content: "";
    position: absolute;
    top: 0;
    right: 0;
    width: 0;
    height: 0;
    border-top: 8px solid #f59e0b;
    border-left: 8px solid transparent;
}

/* Milestone badges */
.milestone {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.5px;
    background: rgba(217,119,6,0.1);
    color: var(--accent);
}

.milestone-none { background: rgba(100,116,139,0.1); color: var(--gray); }
.milestone-method_runs { background: rgba(37,99,235,0.1); color: var(--blue); }
.milestone-core_claim { background: rgba(22,163,74,0.1); color: var(--success); }
.milestone-full { background: rgba(22,163,74,0.2); color: var(--success); }

/* Tables */
.table-wrap {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    overflow: hidden;
    box-shadow: var(--shadow);
}

td.error-td {
    max-width: 180px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    font-size: 11px;
}

.done-row { cursor: pointer; }
.done-row:hover { background: rgba(217,119,6,0.04); }

.done-detail { display: none; }
.done-detail.open { display: table-row; }
.done-detail td {
    padding: 0;
    background: #fdf6ec;
}

.done-detail-content {
    padding: 12px 16px;
    font-size: 12px;
    color: var(--text2);
    white-space: pre-wrap;
    word-break: break-word;
    max-height: 200px;
    overflow-y: auto;
}

table {
    width: 100%;
    border-collapse: collapse;
}

thead th {
    text-align: left;
    padding: 10px 14px;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--text3);
    border-bottom: 1px solid var(--border);
    background: rgba(0,0,0,0.02);
}

tbody td {
    padding: 8px 14px;
    font-size: 12px;
    border-bottom: 1px solid var(--border);
    font-family: 'Courier New', monospace;
}

tbody tr:last-child td { border-bottom: none; }
tbody tr:hover { background: rgba(217,119,6,0.04); }

.empty-state {
    padding: 24px;
    text-align: center;
    color: var(--text3);
    font-size: 13px;
}

/* Progress bars */
.progress-row {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 8px;
}

.progress-label {
    font-size: 11px;
    font-weight: 600;
    color: var(--text3);
    min-width: 48px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.progress-track {
    flex: 1;
    height: 6px;
    background: var(--border);
    border-radius: 3px;
    overflow: hidden;
}

.progress-fill {
    height: 100%;
    border-radius: 3px;
    transition: width 0.5s ease;
}

.progress-fill-pool { background: linear-gradient(90deg, var(--accent), var(--accent-light)); }

.progress-value {
    font-size: 11px;
    font-family: 'Courier New', monospace;
    color: var(--text2);
    min-width: 48px;
    text-align: right;
}

/* Responsive */
@media (max-width: 768px) {
    .main { padding: 12px; }
    .cards { grid-template-columns: repeat(2, 1fr); gap: 10px; }
    .header { padding: 12px 16px; flex-wrap: wrap; gap: 8px; }
    .header-pills { display: none; }
}
</style>
</head>
<body>

<div class="header">
    <div class="header-left">
        <div class="logo">SWARM <span class="logo-sub">Builder Orchestrator</span></div>
        <div class="header-pills">
            <span class="pill pill-neutral" id="pillPool">
                Workers: --/--
            </span>
            <span class="pill pill-neutral" id="pillQueue">Queue: --</span>
        </div>
    </div>
    <div class="header-right">
        <div class="refresh-indicator">
            <div class="pulse-dot" id="pulseDot"></div>
            <span>Auto-refresh</span>
        </div>
        <span id="lastUpdated">--</span>
    </div>
</div>

<div class="error-banner" id="errorBanner"></div>

<div class="main">

    <!-- Overview cards -->
    <div class="cards">
        <div class="card">
            <div class="card-label">Workers</div>
            <div class="card-value"><span id="activeWorkers">--</span> <span class="tag">/ <span id="poolSize">--</span></span></div>
            <div class="card-sub">active / pool size</div>
        </div>
        <div class="card">
            <div class="card-label">Queue</div>
            <div class="card-value" id="queueSize">--</div>
            <div class="card-sub">papers waiting</div>
        </div>
        <div class="card">
            <div class="card-label">Completed</div>
            <div class="card-value" id="completedCount">--</div>
            <div class="card-sub" id="completedSub">--</div>
        </div>
        <div class="card">
            <div class="card-label">Uptime</div>
            <div class="card-value" id="uptime">--</div>
            <div class="card-sub" id="nexusSub">Nexus: --</div>
        </div>
    </div>

    <!-- Worker pool utilization -->
    <div class="section">
        <div class="progress-row">
            <span class="progress-label">Pool</span>
            <div class="progress-track"><div class="progress-fill progress-fill-pool" id="poolBar" style="width:0%"></div></div>
            <span class="progress-value" id="poolValue">--</span>
        </div>
    </div>

    <!-- Active papers -->
    <div class="section">
        <div class="section-header">
            <span class="section-title">Active</span>
            <span class="section-count" id="activeCount">(0)</span>
        </div>
        <div class="table-wrap">
            <table>
                <thead>
                    <tr>
                        <th>Paper</th>
                        <th>Profile</th>
                        <th>Iter</th>
                        <th>Milestone</th>
                        <th>Worker</th>
                        <th>State Time</th>
                        <th>GPU Budget</th>
                        <th>Elapsed</th>
                    </tr>
                </thead>
                <tbody id="activeBody">
                    <tr><td colspan="6" class="empty-state">No active papers</td></tr>
                </tbody>
            </table>
        </div>
    </div>

    <!-- Queued papers -->
    <div class="section">
        <div class="section-header">
            <span class="section-title">Queued</span>
            <span class="section-count" id="queuedCount">(0)</span>
        </div>
        <div class="table-wrap">
            <table>
                <thead>
                    <tr>
                        <th>Paper</th>
                        <th>GPU Budget</th>
                    </tr>
                </thead>
                <tbody id="queuedBody">
                    <tr><td colspan="2" class="empty-state">No papers in queue</td></tr>
                </tbody>
            </table>
        </div>
    </div>

    <!-- Completed papers -->
    <div class="section">
        <div class="section-header">
            <span class="section-title">Completed</span>
            <span class="section-count" id="doneCount">(0)</span>
        </div>
        <div class="table-wrap">
            <table>
                <thead>
                    <tr>
                        <th>Paper</th>
                        <th>Profile</th>
                        <th>Iter</th>
                        <th>Status</th>
                        <th>Milestone</th>
                        <th>Duration</th>
                        <th>Error</th>
                        <th>Completed</th>
                    </tr>
                </thead>
                <tbody id="doneBody">
                    <tr><td colspan="8" class="empty-state">No completed papers</td></tr>
                </tbody>
            </table>
        </div>
    </div>

</div>

<script>
(function() {
    "use strict";

    function formatDuration(seconds) {
        if (seconds === null || seconds === undefined || seconds < 0) return "--";
        seconds = Math.floor(seconds);
        var h = Math.floor(seconds / 3600);
        var m = Math.floor((seconds % 3600) / 60);
        var s = seconds % 60;
        if (h > 0) return h + "h " + m + "m";
        if (m > 0) return m + "m " + s + "s";
        return s + "s";
    }

    function formatUptime(seconds) {
        if (!seconds) return "--";
        seconds = Math.floor(seconds);
        var d = Math.floor(seconds / 86400);
        var h = Math.floor((seconds % 86400) / 3600);
        var m = Math.floor((seconds % 3600) / 60);
        if (d > 0) return d + "d " + h + "h";
        if (h > 0) return h + "h " + m + "m";
        return m + "m";
    }

    function escapeHtml(str) {
        if (!str) return "";
        var d = document.createElement("div");
        d.textContent = str;
        return d.innerHTML;
    }

    function milestoneClass(m) {
        if (!m || m === "none") return "milestone milestone-none";
        if (m === "method_runs") return "milestone milestone-method_runs";
        if (m === "core_claim" || m === "core_claim_plus") return "milestone milestone-core_claim";
        if (m === "full" || m === "near_complete" || m === "majority") return "milestone milestone-full";
        return "milestone";
    }

    // DOM refs
    var $activeWorkers = document.getElementById("activeWorkers");
    var $poolSize = document.getElementById("poolSize");
    var $queueSize = document.getElementById("queueSize");
    var $completedCount = document.getElementById("completedCount");
    var $completedSub = document.getElementById("completedSub");
    var $uptime = document.getElementById("uptime");
    var $nexusSub = document.getElementById("nexusSub");
    var $poolBar = document.getElementById("poolBar");
    var $poolValue = document.getElementById("poolValue");
    var $activeCount = document.getElementById("activeCount");
    var $activeBody = document.getElementById("activeBody");
    var $queuedCount = document.getElementById("queuedCount");
    var $queuedBody = document.getElementById("queuedBody");
    var $doneCount = document.getElementById("doneCount");
    var $doneBody = document.getElementById("doneBody");
    var $lastUpdated = document.getElementById("lastUpdated");
    var $pulseDot = document.getElementById("pulseDot");
    var $errorBanner = document.getElementById("errorBanner");
    var $pillPool = document.getElementById("pillPool");
    var $pillQueue = document.getElementById("pillQueue");

    var errorCount = 0;

    function fetchJSON(url, cb) {
        fetch(url)
            .then(function(r) {
                if (!r.ok) throw new Error("HTTP " + r.status);
                return r.json();
            })
            .then(function(data) {
                errorCount = 0;
                $errorBanner.classList.remove("visible");
                $pulseDot.classList.remove("error");
                cb(data);
            })
            .catch(function(err) {
                errorCount++;
                if (errorCount >= 3) {
                    $errorBanner.textContent = "Failed to reach API: " + err.message;
                    $errorBanner.classList.add("visible");
                    $pulseDot.classList.add("error");
                }
            });
    }

    function renderStatus(data) {
        if (data.error) return;

        $activeWorkers.textContent = data.active_count;
        $poolSize.textContent = data.pool_size;
        $queueSize.textContent = data.queue_size;
        $completedCount.textContent = data.completed_count;
        $uptime.textContent = formatUptime(data.uptime);

        if (data.nexus_url) {
            $nexusSub.textContent = "Nexus: " + data.nexus_url;
        }

        var poolPct = data.pool_size > 0
            ? Math.round((data.active_count / data.pool_size) * 100)
            : 0;
        $poolBar.style.width = poolPct + "%";
        var workingCap = data.max_working_return_overflow > 0
            ? data.max_working_workers + "+" + data.max_working_return_overflow
            : String(data.max_working_workers);
        var runtimeDetail = data.split_runtime_caps
            ? "building " + data.building_count + "/" + data.max_building_workers +
              ", installing " + data.installing_count + "/" + data.max_installing_workers +
              ", wait " + data.waiting_count +
              ", admitted " + data.admitted_count + "/" + data.max_admitted_workers
            : "runtime " + data.runtime_count + "/" + data.max_runtime_workers;
        $poolValue.textContent =
            data.active_count + " / " + data.pool_size +
            " (working " + data.working_count + "/" + workingCap +
            ", " + runtimeDetail + ")";

        // Header pills
        $pillPool.textContent = data.split_runtime_caps
            ? "Workers: " + data.active_count + "/" + data.pool_size +
              " | W " + data.working_count + "/" + workingCap +
              " | Build " + data.building_count + "/" + data.max_building_workers +
              " | Install " + data.installing_count + "/" + data.max_installing_workers +
              " | Wait " + data.waiting_count +
              " | Admit " + data.admitted_count + "/" + data.max_admitted_workers
            : "Workers: " + data.active_count + "/" + data.pool_size +
              " | W " + data.working_count + "/" + workingCap +
              " | R " + data.runtime_count + "/" + data.max_runtime_workers;
        $pillQueue.textContent = "Queue: " + data.queue_size;

        if (data.shutdown) {
            $pillPool.className = "pill pill-err";
        } else if (data.active_count > 0) {
            $pillPool.className = "pill pill-ok";
        } else {
            $pillPool.className = "pill pill-neutral";
        }
    }

    function renderPapers(papers) {
        if (papers.error) return;

        var active = [];
        var queued = [];
        var done = [];
        for (var i = 0; i < papers.length; i++) {
            var p = papers[i];
            if (p.phase === "active") active.push(p);
            else if (p.phase === "queued") queued.push(p);
            else done.push(p);
        }

        // Count done vs error
        var doneOk = 0, doneErr = 0;
        for (var j = 0; j < done.length; j++) {
            if (done[j].status === "error") doneErr++;
            else doneOk++;
        }
        $completedSub.textContent = doneOk + " done, " + doneErr + " error";

        // Active
        $activeCount.textContent = "(" + active.length + ")";
        if (active.length === 0) {
            $activeBody.innerHTML = '<tr><td colspan="8" class="empty-state">No active papers</td></tr>';
        } else {
            var html = "";
            var now = Date.now() / 1000;
            for (var a = 0; a < active.length; a++) {
                var ap = active[a];
                var iterElapsed = ap.started_at ? (now - ap.started_at) : 0;
                var totalElapsed = (ap.total_duration_sec || 0) + iterElapsed;
                var elapsed = totalElapsed > 0 ? formatDuration(totalElapsed) : "--";
                if (ap.iteration > 1 && iterElapsed > 0) {
                    elapsed += ' <span style="font-size:10px;color:var(--text3)">(iter ' + formatDuration(iterElapsed) + ')</span>';
                }
                html += "<tr>";
                html += "<td>" + escapeHtml(ap.arxiv_id) + "</td>";
                html += "<td>" + escapeHtml(ap.profile || "--") + "</td>";
                html += "<td>" + (ap.iteration != null ? ap.iteration : 0) + "</td>";
                html += '<td><span class="' + milestoneClass(ap.milestone) + '">' + escapeHtml(ap.milestone || "none") + "</span></td>";
                var wsClasses = 'badge badge-' + (ap.worker_status || "idle");
                var wsTitle = "";
                if (ap.blocked_on) {
                    // Failure retry = wants to return to "working" from
                    // anything other than waiting/idle (waiting→working
                    // is the normal hive-job-done path; idle→working is
                    // a fresh worker, not thrashing).
                    var isFailureRetry = (ap.blocked_on === "working"
                        && ap.worker_status !== "waiting"
                        && ap.worker_status !== "idle");
                    if (isFailureRetry) {
                        wsClasses += " badge-blocked";
                        wsTitle = ' title="failed ' + escapeHtml(ap.worker_status) + ', retrying → working"';
                    } else {
                        wsTitle = ' title="queued for ' + escapeHtml(ap.blocked_on) + '"';
                    }
                }
                var wsBadge = '<span class="' + wsClasses + '"' + wsTitle + '>' + escapeHtml(ap.worker_status || "idle") + "</span>";
                if (ap.last_job_resources && (ap.worker_status === "waiting" || ap.worker_status === "building" || ap.worker_status === "installing")) {
                    wsBadge += ' <span style="font-size:10px;color:var(--text3)">' + escapeHtml(ap.last_job_resources) + '</span>';
                    if (ap.last_job_id) {
                        var shortId = ap.last_job_id.length > 8 ? ap.last_job_id.substring(0, 8) : ap.last_job_id;
                        wsBadge += ' <span style="font-size:10px;color:var(--text3)" title="' + escapeHtml(ap.last_job_id) + '">[' + escapeHtml(shortId) + ']</span>';
                    } else if (ap.worker_status === "waiting") {
                        wsBadge += ' <span style="font-size:10px;color:var(--text3)">[queuing]</span>';
                    }
                }
                html += "<td>" + wsBadge + "</td>";
                // State time: current stretch + total for this state
                var ws = ap.worker_status || "idle";
                var stCurrent = ap.worker_status_elapsed || 0;
                var stTotal = (ap.worker_status_totals && ap.worker_status_totals[ws]) || 0;
                var stTotalWithCurrent = stTotal + stCurrent;
                var stateTimeStr = formatDuration(stCurrent);
                if (stTotal > 0) {
                    stateTimeStr += ' <span style="font-size:10px;color:var(--text3)">(' + formatDuration(stTotalWithCurrent) + ' total)</span>';
                }
                html += "<td>" + stateTimeStr + "</td>";
                html += "<td>" + (ap.gpu_budget || "--") + "h</td>";
                html += "<td>" + elapsed + "</td>";
                html += "</tr>";
            }
            $activeBody.innerHTML = html;
        }

        // Queued
        $queuedCount.textContent = "(" + queued.length + ")";
        if (queued.length === 0) {
            $queuedBody.innerHTML = '<tr><td colspan="2" class="empty-state">No papers in queue</td></tr>';
        } else {
            var qhtml = "";
            for (var q = 0; q < queued.length; q++) {
                var qp = queued[q];
                qhtml += "<tr>";
                qhtml += "<td>" + escapeHtml(qp.arxiv_id) + "</td>";
                qhtml += "<td>" + (qp.gpu_budget || "--") + "h</td>";
                qhtml += "</tr>";
            }
            $queuedBody.innerHTML = qhtml;
        }

        // Completed — sort by finished_at descending (latest first)
        done.sort(function(a, b) { return (b.finished_at || 0) - (a.finished_at || 0); });
        $doneCount.textContent = "(" + done.length + ")";
        if (done.length === 0) {
            $doneBody.innerHTML = '<tr><td colspan="8" class="empty-state">No completed papers</td></tr>';
        } else {
            var dhtml = "";
            for (var d = 0; d < done.length; d++) {
                var dp = done[d];
                var totalSec = dp.total_duration_sec > 0 ? dp.total_duration_sec
                    : (dp.duration_sec > 0 ? dp.duration_sec
                    : (dp.finished_at && dp.started_at ? dp.finished_at - dp.started_at : 0));
                var dur = totalSec > 0 ? formatDuration(totalSec) : "--";
                var statusCls = dp.status === "done" ? "badge-done" : "badge-error";
                var completedAt = "--";
                if (dp.finished_at) {
                    var dt = new Date(dp.finished_at * 1000);
                    var mm = String(dt.getMonth() + 1).padStart(2, "0");
                    var dd = String(dt.getDate()).padStart(2, "0");
                    var hh = String(dt.getHours()).padStart(2, "0");
                    var mi = String(dt.getMinutes()).padStart(2, "0");
                    completedAt = mm + "-" + dd + " " + hh + ":" + mi;
                }
                var errText = dp.error || "";
                dhtml += '<tr class="done-row" data-did="' + escapeHtml(dp.arxiv_id) + "_" + d + '">';
                dhtml += "<td>" + escapeHtml(dp.arxiv_id) + "</td>";
                dhtml += "<td>" + escapeHtml(dp.profile || "--") + "</td>";
                dhtml += "<td>" + (dp.iteration != null ? dp.iteration : 0) + "</td>";
                dhtml += '<td><span class="badge ' + statusCls + '">' + escapeHtml(dp.status || "--") + "</span></td>";
                dhtml += '<td><span class="' + milestoneClass(dp.milestone) + '">' + escapeHtml(dp.milestone || "none") + "</span></td>";
                dhtml += "<td>" + dur + "</td>";
                dhtml += '<td class="error-td">' + escapeHtml(errText) + "</td>";
                dhtml += "<td>" + completedAt + "</td>";
                dhtml += "</tr>";
                if (errText) {
                    dhtml += '<tr class="done-detail" data-ddetail="' + escapeHtml(dp.arxiv_id) + "_" + d + '">';
                    dhtml += '<td colspan="8"><div class="done-detail-content">' + escapeHtml(errText) + "</div></td>";
                    dhtml += "</tr>";
                }
            }
            $doneBody.innerHTML = dhtml;
            var doneRows = $doneBody.querySelectorAll(".done-row");
            for (var dr = 0; dr < doneRows.length; dr++) {
                doneRows[dr].addEventListener("click", function() {
                    var did = this.getAttribute("data-did");
                    var detail = $doneBody.querySelector('[data-ddetail="' + did + '"]');
                    if (detail) detail.classList.toggle("open");
                });
            }
        }
    }

    function refresh() {
        fetchJSON("/api/dashboard", function(data) {
            if (data.error) return;
            renderStatus(data.status || {});
            renderPapers(data.papers || []);
            $lastUpdated.textContent = new Date().toLocaleTimeString();
        });
    }

    refresh();
    setInterval(refresh, 10000);
})();
</script>
</body>
</html>
"""
