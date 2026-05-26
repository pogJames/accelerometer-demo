"""Worker 3 — Flask dashboard. Also the entry point that wires W1, W2, W3.

project4 adds:
- /train page (feature-extractor + lightweight head training)
- /api/recordings, /api/record/start|status|cancel endpoints (from project3)
- /api/train/start|status|cancel endpoints (new)
- a per-port reader subprocess that owns the local RecordingManager — main
  process talks to it over a RemoteRecorder RPC proxy (raw samples never
  cross the IPC boundary)
- a TrainerManager that borrows the InferenceWorker's TFLite interpreter to
  compute prototypes, then hot-reloads the head — no app restart needed

Run with:  python app.py
Dashboard: http://localhost:8000/
"""

import json
import multiprocessing as mp
import os
import queue
import sys
import threading
import time

# Bundled deps (if any) live in ./site-packages on the embedded device —
# we used to ship pymodbus this way; the path insert is harmless when the
# directory is empty and the safety net stays in case something else gets
# bundled later. On Linux (fork), child processes inherit sys.path; on
# Windows (spawn, the default), reader_process_main re-applies it because
# spawn gives the child a fresh interpreter with no inherited state.
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_path, "site-packages"))

from flask import Flask, Response, jsonify, render_template, request

from state import RollingPredictions, SnapshotBus, ROLLING_WINDOW
from sensor_reader import (reader_process_main, ALLOWED_PORTS,
                           get_existed_serial_ports, SAMPLE_RATE)
from inference import InferenceWorker
from recorder import list_existing_labels, MIN_SAMPLES, DATA_DIR
from trainer import TrainerManager, MIN_WINDOWS, WINDOW_SIZE
from classifier import DEFAULT_HEAD_PATH


SSE_HEARTBEAT_S = 15
RPC_TIMEOUT_S = 5.0


class RemoteRecorder:
    """Thin RPC wrapper. Reads exactly like a local RecordingManager from
    Flask's perspective; under the hood every call round-trips over an
    mp.Queue pair to the reader subprocess that owns the real
    RecordingManager. Serialized with a lock so concurrent Flask requests
    can't interleave a put-then-get pair."""

    def __init__(self, req_q, resp_q, port):
        self._req_q = req_q
        self._resp_q = resp_q
        self._port = port               # for debug strings only
        self._next_id = 0
        self._lock = threading.Lock()

    def _call(self, op, **kwargs):
        with self._lock:
            # Drain any stale responses from earlier timed-out RPCs so the
            # req_id match below stays meaningful.
            try:
                while True:
                    self._resp_q.get_nowait()
            except queue.Empty:
                pass

            req_id = self._next_id
            self._next_id += 1
            self._req_q.put({"op": op, "req_id": req_id, "kwargs": kwargs})

            try:
                resp = self._resp_q.get(timeout=RPC_TIMEOUT_S)
            except queue.Empty:
                raise RuntimeError(
                    f"recorder RPC timed out after {RPC_TIMEOUT_S}s "
                    f"(port={self._port}, op={op})")

        if resp.get("req_id") != req_id:
            raise RuntimeError(
                f"out-of-order recorder RPC response "
                f"(expected req_id={req_id}, got {resp.get('req_id')})")
        if not resp["ok"]:
            err_cls = {"ValueError": ValueError,
                       "RuntimeError": RuntimeError}.get(
                resp.get("error_type"), RuntimeError)
            raise err_cls(resp["error"])
        return resp["result"]

    def start(self, name, target_samples, port, mode):
        return self._call("start", name=name, target_samples=target_samples,
                          port=port, mode=mode)

    def cancel(self):
        return self._call("cancel")

    def status(self):
        return self._call("status")


def build_app():
    ports = [p for p in get_existed_serial_ports() if p in ALLOWED_PORTS]
    print(f"[app] sensors detected: {ports}")

    # Make sure data/ exists relative to project4 so recordings land predictably
    # regardless of where the user launches python from.
    data_dir = os.path.join(current_path, DATA_DIR)
    os.makedirs(data_dir, exist_ok=True)
    print(f"[app] data dir: {data_dir}")

    head_path = os.path.join(current_path, DEFAULT_HEAD_PATH)

    # mp.Queue crosses the process boundary; pickling cost is ~1–2 ms for
    # a (2604, 3) float32 window — negligible vs the 167 ms hop interval.
    # Cap at 16 so 4 readers (max_qsize=12 in sensor_reader) have headroom
    # before put() would block; drop-oldest on the writer side kicks in
    # well before we hit the maxsize cap.
    window_queue = mp.Queue(maxsize=16)
    bus = SnapshotBus()
    rolling_predictions = {p: RollingPredictions(on_latch=bus.bump) for p in ports}

    # Per-port reader subprocess + RPC channel for the recorder it owns.
    req_qs   = {p: mp.Queue() for p in ports}
    resp_qs  = {p: mp.Queue() for p in ports}
    stop_events = {p: mp.Event() for p in ports}
    # active_events start CLEAR — every reader is paused on boot. The UI
    # (or /api/record/start) flips exactly one event at a time so only one
    # sensor is ever pulling Modbus traffic; switching ports resets the
    # paused port's FIFO via a sample-rate rewrite inside the subprocess.
    active_events = {p: mp.Event() for p in ports}
    reader_procs = {}
    for p in ports:
        proc = mp.Process(
            target=reader_process_main,
            args=(p, window_queue, req_qs[p], resp_qs[p],
                  stop_events[p], active_events[p], data_dir),
            daemon=True,
            name=f"reader-{p.replace('/', '_')}",
        )
        proc.start()
        reader_procs[p] = proc
        print(f"[app] spawned reader process for {p} (pid={proc.pid})")

    # No CPU pinning for the main process either — the kernel schedules
    # Flask + InferenceWorker alongside the reader subprocesses freely.

    recorders = {p: RemoteRecorder(req_qs[p], resp_qs[p], p) for p in ports}

    # Which port's subprocess owns the active (or most-recent) recording
    # session. Set on successful start(); never cleared, so /status keeps
    # reporting the right subprocess's last_finished snapshot.
    record_state = {"port": None}
    record_state_lock = threading.Lock()

    def current_recorder():
        with record_state_lock:
            p = record_state["port"]
        return recorders.get(p)

    # Any subset of readers can be active at once. The active_events dict is
    # the source of truth — no separate state tracking needed. Lock serialises
    # set/clear pairs so concurrent /api/active_port requests don't race the
    # rolling-buffer clear with each other.
    active_state_lock = threading.Lock()

    def activate(port):
        """Wake the reader for `port`. No-op if already active.
        Clears the port's rolling buffer so the dashboard immediately shows
        'waiting…' instead of stale predictions from a previous activation."""
        if port not in ports:
            raise ValueError(f"unknown port: {port!r}")
        with active_state_lock:
            if active_events[port].is_set():
                return
            rolling_predictions[port].clear()
            active_events[port].set()

    def deactivate(port):
        """Pause the reader for `port`. No-op if already inactive."""
        if port not in ports:
            raise ValueError(f"unknown port: {port!r}")
        with active_state_lock:
            active_events[port].clear()

    inferer = InferenceWorker(window_queue, rolling_predictions,
                              head_path=head_path)
    inferer.start()

    trainer = TrainerManager(data_dir=data_dir, head_path=head_path)

    def snapshot_payload():
        now = time.time()
        out = {}
        for p in ports:
            snap = rolling_predictions[p].displayable()
            if snap is None:
                out[p] = {
                    "display_seq": 0,
                    "majority_class_id": None,
                    "majority_class_name": None,
                    "majority_count": 0,
                    "window_count": 0,
                    "recent": [],
                    "latest_ts": None,
                }
            else:
                out[p] = {
                    "display_seq": snap["display_seq"],
                    "majority_class_id": snap["majority_class_id"],
                    "majority_class_name": snap["majority_class_name"],
                    "majority_count": snap["majority_count"],
                    "window_count": snap["window_count"],
                    "recent": snap["recent"],
                    "latest_ts": snap["latest_ts"],
                }
        head = inferer.head
        live_labels = head.labels if head is not None else ["untrained"]
        label_colors = head.label_color_map() if head is not None else {"untrained": -1}
        with active_state_lock:
            active_ports = [p for p in ports if active_events[p].is_set()]
        return {
            "ports": out,
            "inference_mode": inferer.mode,
            "class_labels": live_labels,
            "label_colors": label_colors,
            "active_ports": active_ports,
            "now": now,
        }

    app = Flask(__name__,
                static_folder="static",
                template_folder="templates")

    # Make inference_mode + ports available to every template (sidebar uses them).
    @app.context_processor
    def inject_globals():
        return {
            "inference_mode": inferer.mode,
            "ports": ports,
        }

    @app.route("/")
    def index():
        head = inferer.head
        labels = head.labels if head is not None else ["untrained"]
        return render_template("dashboard.html",
                               active_page="live",
                               class_labels=labels,
                               rolling_window=ROLLING_WINDOW)

    @app.route("/record")
    def record_page():
        return render_template("record.html",
                               active_page="record",
                               sample_rate=SAMPLE_RATE,
                               min_samples=MIN_SAMPLES)

    @app.route("/train")
    def train_page():
        head = inferer.head
        head_labels = head.labels if head is not None else []
        return render_template("train.html",
                               active_page="train",
                               window_size=WINDOW_SIZE,
                               min_windows=MIN_WINDOWS,
                               sample_rate=SAMPLE_RATE,
                               head_labels=head_labels,
                               backbone_variant=os.path.basename(inferer.model_path),
                               backbone_present=(inferer._interp is not None))

    @app.route("/api/metrics")
    def metrics():
        return jsonify(snapshot_payload())

    @app.route("/api/stream")
    def stream():
        def gen():
            last_seq = bus.current_seq()
            yield f"data: {json.dumps(snapshot_payload())}\n\n"
            while True:
                new_seq = bus.wait_for_change(last_seq, timeout=SSE_HEARTBEAT_S)
                if new_seq > last_seq:
                    last_seq = new_seq
                    yield f"data: {json.dumps(snapshot_payload())}\n\n"
                else:
                    yield ": heartbeat\n\n"

        return Response(gen(), mimetype="text/event-stream", headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        })

    @app.route("/api/recordings")
    def recordings():
        labels = list_existing_labels(data_dir)
        # Decorate each label with its window count so the train page can
        # filter by eligibility without re-reading the file.
        for entry in labels:
            entry["windows"] = entry["samples"] // WINDOW_SIZE
            entry["eligible"] = entry["windows"] >= MIN_WINDOWS
        return jsonify({
            "labels": labels,
            "ports": ports,
            "sample_rate": SAMPLE_RATE,
            "min_samples": MIN_SAMPLES,
            "min_windows": MIN_WINDOWS,
            "window_size": WINDOW_SIZE,
            "data_dir": data_dir,
        })

    @app.route("/api/record/start", methods=["POST"])
    def record_start():
        body = request.get_json(silent=True) or {}
        name = body.get("name", "")
        target_samples = body.get("target_samples")
        port = body.get("port")
        mode = body.get("mode", "append")

        if port not in ports:
            return jsonify({"error": f"unknown port: {port!r}. Available: {ports}"}), 400
        if not isinstance(target_samples, int) or target_samples <= 0:
            return jsonify({"error": "target_samples must be a positive integer"}), 400

        # Recording can't proceed unless the reader for this port is awake —
        # feed() runs inside the reader subprocess, so a paused reader means
        # zero samples committed. Activating is additive (multiple sensors
        # can be active at once); this does not pause anything else.
        activate(port)

        try:
            session = recorders[port].start(name=name, target_samples=target_samples,
                                            port=port, mode=mode)
        except (ValueError, RuntimeError) as e:
            return jsonify({"error": str(e)}), 400

        with record_state_lock:
            record_state["port"] = port
        return jsonify({"session": session})

    @app.route("/api/active_port", methods=["POST"])
    def api_set_active_port():
        """Toggle one port's active state. Body: {port: "/dev/ttyUSB0",
        active: true|false}. Activating one port no longer deactivates
        others — any subset may be active simultaneously."""
        body = request.get_json(silent=True) or {}
        port = body.get("port")
        active = bool(body.get("active"))
        try:
            if active:
                activate(port)
            else:
                deactivate(port)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        return jsonify({"port": port, "active": active})

    @app.route("/api/record/status")
    def record_status():
        rec = current_recorder()
        if rec is None:
            return jsonify({"session": None})
        try:
            session = rec.status()
        except RuntimeError as e:
            return jsonify({"error": str(e)}), 500
        return jsonify({"session": session})

    @app.route("/api/record/stream")
    def record_stream():
        # Time-driven 1 Hz tick, not event-driven like /api/stream — recording
        # progress is gated by elapsed seconds, not by inference latches.
        def gen():
            while True:
                rec = current_recorder()
                if rec is None:
                    session = None
                else:
                    try:
                        session = rec.status()
                    except RuntimeError:
                        session = None
                yield f"data: {json.dumps({'session': session})}\n\n"
                time.sleep(1.0)

        return Response(gen(), mimetype="text/event-stream", headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        })

    @app.route("/api/record/cancel", methods=["POST"])
    def record_cancel():
        rec = current_recorder()
        if rec is None:
            return jsonify({"session": None})
        try:
            session = rec.cancel()
        except RuntimeError as e:
            return jsonify({"error": str(e)}), 500
        return jsonify({"session": session})

    @app.route("/api/train/status")
    def train_status():
        head = inferer.head
        return jsonify({
            "session": trainer.status(),
            "head_labels": head.labels if head is not None else [],
            "head_colors": head.label_color_map() if head is not None else {},
            "backbone": os.path.basename(inferer.model_path),
            "backbone_mode": inferer.mode,
        })

    @app.route("/api/train/start", methods=["POST"])
    def train_start():
        body = request.get_json(silent=True) or {}
        labels = body.get("labels") or []
        if not isinstance(labels, list):
            return jsonify({"error": "labels must be a list"}), 400
        try:
            session = trainer.start(inference_worker=inferer, selected_labels=labels)
        except (ValueError, RuntimeError) as e:
            return jsonify({"error": str(e)}), 400
        return jsonify({"session": session})

    @app.route("/api/train/cancel", methods=["POST"])
    def train_cancel():
        session = trainer.cancel()
        return jsonify({"session": session})

    return app


if __name__ == "__main__":
    app = build_app()
    app.run(host="0.0.0.0", port=8000, threaded=True, debug=False)
