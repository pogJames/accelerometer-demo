"""Worker 1 — Modbus reader, one subprocess per sensor.

Runs in its OWN process (not a thread). The serial-buffer poll loop is
GIL-bound; sharing the GIL with Flask / InferenceWorker / GC pauses lets
the sensor's hardware FIFO (max 65535 regs ≈ 2.8 s at 7812 Hz) overflow
and silently drop samples. A dedicated process means nothing in the main
interpreter can stall this read loop.

The subprocess also owns the RecordingManager — raw samples never cross
the IPC boundary. `RecordingManager.feed()` runs locally; its writer
thread releases the GIL during `ndarray.tofile()`, so disk I/O doesn't
stall the modbus poll either. Control plane (start/cancel/status) is
RPC-style over a request/response queue pair drained by a sibling thread.

Streams raw XYZ samples from input register 0x02 (FC04) and emits
WINDOW_SIZE-sample sliding windows (HOP_SIZE hop) to the inference queue.
Read pattern mirrors DAQ_Modbus_MultiChs_v1.3.py; queue uses drop-oldest.
"""

import glob
import os
import queue
import sys
import threading
import time

import numpy as np
import serial

# Up to 4 sensors. Add /dev/ttyUSB1..USB3 here as more come online.
ALLOWED_PORTS = ['/dev/ttyUSB0']

WINDOW_SIZE   = 2604     # 1/2 sec at 7812 Hz — must match the trained backbone
HOP_SIZE      = 1302     # 50 % overlap → ~6 window emits/sec (inference path)
# Second, faster cadence for the live waveform: emit a small chunk of fresh
# samples every RAW_CHUNK_SIZE (≈36 Hz/port at 7812 Hz). The waveform ring
# buffer (waveform.py) accumulates these; inference keeps using the slower
# 2604-window stream above.
RAW_CHUNK_SIZE = 217
MAX_PACKET    = 41 * 3   # Modbus packet upper bound: 41 XYZ triplets
TURN_GRAVITY  = 8192     # raw int16 / 8192 -> G value
SAMPLE_RATE   = 7812
# Precomputed float32 scale so the per-iteration decode `raw * SCALE` is one
# numpy ufunc pass (int16 → float32 cast fused with the multiply). Was:
# `(raw / TURN_GRAVITY).astype(np.float32)` — two passes plus a float64 temp.
SCALE         = np.float32(1.0 / TURN_GRAVITY)

# pyserial blocks on read() until N bytes arrive or this timeout expires.
# 50 ms is well above the sensor's actual response time at 3 Mbaud
# (request ~30 µs, processing a few ms, response ~700 µs — total normally
# <10 ms) but turns a bad-packet stall into a perceptually invisible
# blip: ~50 ms timeout + one MAX_PACKET catchup read ≈ 100 ms total.
# If you start seeing total_fails climbing during normal (non-impact)
# operation, bump this up — the sensor's tail latency might be longer.
READ_TIMEOUT_S          = 0.05
# After this many CONSECUTIVE failed reads, drop + reopen the serial port.
# Picks up wedged FTDI/CDC drivers and resyncs the framer.
RECONNECT_AFTER_FAILS   = 5

# ── Computed-metric registers (FC03 holding registers) ────────────────────
# Addresses + scaling mirror the validated Rust client (src/types.rs,
# src/modbus.rs). Each metric is its OWN FC03 read of `count` registers at its
# base address — the addresses alias, so we do NOT read one contiguous block.
REG_TEMPERATURE        = 0x0014   # 1 reg, value / 100  -> °C
REG_GRAVITY_RMS        = 0x001E   # 3 regs (x,y,z), / 1000
REG_GRAVITY_PEAK       = 0x001F   # 3 regs, / 1000
REG_GRAVITY_CREST      = 0x0020   # 3 regs, / 1000
REG_GRAVITY_SKEWNESS   = 0x0021   # 3 regs, / 1000  (slow: 2-5 s update)
REG_GRAVITY_KURTOSIS   = 0x0022   # 3 regs, / 1000  (slow: 2-5 s update)
REG_GRAVITY_PRIM_FREQ  = 0x003D   # 1 reg, raw Hz
REG_VELOCITY_RMS       = 0x0032   # 3 regs, / 100
REG_VELOCITY_PEAK      = 0x0033   # 3 regs, / 100
REG_VELOCITY_CREST     = 0x0034   # 3 regs, / 100
REG_VELOCITY_PRIM_FREQ = 0x003C   # 1 reg, raw Hz

# Metric polling cadence. We poll kurtosis (the slowest metric) at this rate
# and only emit a full batch when it changes — so the whole metric set updates
# together at the slow 2-5 s kurtosis cadence rather than spamming the fast
# ones. METRIC_FALLBACK_S forces a refresh even if kurtosis sits perfectly
# still. (The Rust reference instead polls everything at a flat 5 Hz; swap
# KURT change-detection for a fixed interval here if that's preferred.)
KURT_POLL_INTERVAL_S = 0.4
METRIC_FALLBACK_S    = 5.0
# FC03 metric reads are genuinely SLOW + variable on this sensor (~0.5-1 s,
# occasionally more; FC04 raw reads on the same port are fast, so it's the
# sensor's holding-register handler, not the wire). The vendor's Rust client
# uses a 5 s read timeout for exactly this — and we MUST too, because a
# premature timeout leaves the slow response in-flight; the next request then
# reads that stale reply ("wrong byte count") and the stream desyncs forever.
# So: generous timeout (don't bail on a slow-but-coming response) + a line
# drain to resync if anything does go wrong. The timeout is swapped in only
# around metric reads and restored to READ_TIMEOUT_S for raw streaming.
#
# Consequence: a full sweep of ~12 metrics costs a several-second floor. We
# can't batch it away — the metric registers ALIAS (each base address is a
# selector read with count=3; a block read returns garbage), which is why the
# Rust reads each metric separately.
METRIC_READ_TIMEOUT_S = 5.0
METRIC_READ_RETRIES   = 3
METRIC_READ_GAP_S     = 0.001


class _MetricsAbort(Exception):
    """Raised to bail out of an in-progress (slow) metric sweep when raw
    streaming is requested or metrics are turned off — so a page switch to the
    waveform view isn't blocked behind a multi-second FC03 batch."""


def _emit_window(window_queue, port, window, max_qsize):
    # Drop-oldest on full so the inference worker always sees the freshest window.
    # mp.Queue.qsize() raises NotImplementedError on macOS — tolerate that.
    while True:
        try:
            if window_queue.qsize() < max_qsize:
                break
        except NotImplementedError:
            break
        print(f"[{port}] Warning! Queue Overwrite")
        try:
            window_queue.get_nowait()
        except Exception:
            break
    window_queue.put((port, window))


def _emit_raw_chunk(raw_queue, port, chunk):
    """Push a small fresh-sample chunk to the waveform raw_queue. Drop-oldest
    on full (best-effort, never blocks the read loop). qsize() may be
    unimplemented on some platforms — tolerate that."""
    if raw_queue is None:
        return
    try:
        if raw_queue.full():
            try:
                raw_queue.get_nowait()
            except Exception:
                pass
        raw_queue.put_nowait((port, chunk))
    except Exception:
        pass


def _control_listener(recorder, req_q, resp_q, stop_event):
    """Sibling thread inside the reader subprocess. Drains the request
    queue, dispatches start/cancel/status to the local RecordingManager,
    and writes a response back. Decoupled from the read loop so an RPC
    can't stall the modbus poll."""
    while not stop_event.is_set():
        try:
            req = req_q.get(timeout=0.5)
        except queue.Empty:
            continue
        op = req.get("op")
        req_id = req.get("req_id")
        try:
            if op == "start":
                result = recorder.start(**req.get("kwargs", {}))
            elif op == "cancel":
                result = recorder.cancel()
            elif op == "status":
                result = recorder.status()
            else:
                raise ValueError(f"unknown op: {op!r}")
            resp_q.put({"req_id": req_id, "ok": True, "result": result})
        except Exception as e:
            resp_q.put({"req_id": req_id, "ok": False,
                        "error": str(e), "error_type": type(e).__name__})


def reader_process_main(port, window_queue, req_q, resp_q,
                        stop_event, active_event, data_dir,
                        metrics_event=None, metrics_queue=None, raw_queue=None,
                        port_baud=3000000, bytesize=8, parity='N', stopbits=1,
                        timeout=READ_TIMEOUT_S, sample_rate=SAMPLE_RATE,
                        window_size=WINDOW_SIZE, hop_size=HOP_SIZE, max_qsize=12):
    """Subprocess entry point. Owns the pyserial client + RecordingManager
    for one serial port.

    port — e.g. '/dev/ttyUSB0'.
    window_queue — mp.Queue, push `(port, np.ndarray)` for inference.
    req_q / resp_q — mp.Queue pair, main process RPCs in / responses out.
    stop_event — mp.Event, set by main process to request shutdown.
    active_event — mp.Event, main process sets this when this port should be
        actively reading. When clear, the reader sleeps and burns no Modbus
        traffic. On every inactive→active transition we re-write the
        sample-rate register, which the sensor uses as a FIFO reset so the
        first window after resume is fresh.
    data_dir — absolute path, where recordings get written.
    """
    # `spawn` start method gives the child a fresh interpreter — re-apply
    # the bundled-deps path before importing project modules.
    current_path = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(current_path, "site-packages"))
    from fast_modbus import (read_input_registers, read_holding_registers,
                              write_single_register, ModbusError)
    from recorder import RecordingManager

    print(f"[{port}] reader subprocess started pid={os.getpid()} ppid={os.getppid()}")

    # No CPU pinning — let the kernel schedule readers across cores
    # however it sees fit. SCHED_FIFO below still preempts normal-priority
    # processes when a reader has work; that's how we keep the read loop
    # responsive without binding to a specific core.

    # Real-time priority so the reader preempts Flask threads whenever
    # both are runnable on the same core (fallback if affinity-pinning
    # isn't enough). Requires root or CAP_SYS_NICE; we just log and
    # continue on permission errors so dev machines aren't blocked.
    if hasattr(os, "sched_setscheduler") and hasattr(os, "SCHED_FIFO"):
        try:
            os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(50))
            print(f"[{port}] scheduler=SCHED_FIFO priority=50")
        except (OSError, PermissionError) as e:
            print(f"[{port}] sched_setscheduler failed (run as root or grant "
                  f"CAP_SYS_NICE): {e}")

    recorder = RecordingManager(data_dir=data_dir)
    threading.Thread(target=_control_listener,
                     args=(recorder, req_q, resp_q, stop_event),
                     daemon=True,
                     name=f"control-{port}").start()

    # Direct pyserial — no pymodbus wrapper. fast_modbus does the framing.
    client = serial.Serial(port=port,
                            baudrate=port_baud,
                            bytesize=bytesize,
                            parity=parity,
                            stopbits=stopbits,
                            timeout=timeout)

    chip = read_input_registers(client, 1, 0x80, 3)
    print(f"[{port}] ChipID: {hex(chip[0])}, {hex(chip[1])}, {hex(chip[2])}")
    print(f"[{port}] SampleRate: {sample_rate}")
    write_single_register(client, 1, 0x01, sample_rate)

    data_len = read_input_registers(client, 1, 0x02, 1)[0]
    print(f"[{port}] Initial buffer length: {data_len}")

    # Pre-allocated ring buffer. Holds at most WINDOW_SIZE valid samples
    # plus one MAX_PACKET worth of just-read tail before the next emit
    # compacts it down. window_size * 2 gives plenty of headroom and keeps
    # the math obvious. Per-iteration code now does a slice-assign instead
    # of np.row_stack — no allocation in the hot path.
    buffer = np.empty((window_size * 2, 3), dtype=np.float32)
    fill = 0
    last_log_t = time.time()
    emits_since_log = 0
    fail_count = 0
    total_fails = 0
    last_fail_logged_t = 0.0

    # Waveform raw-chunk accumulator: collect decoded samples and emit a fresh
    # chunk every RAW_CHUNK_SIZE (~36 Hz) to raw_queue, independent of the
    # 2604-window inference emission below.
    raw_accum = []
    raw_accum_n = 0

    def safe_read(count):
        """Returns the registers list on success, or None on any failure
        (timeout, CRC, exception response). Tracks consecutive failures
        so the outer loop can decide when to reconnect."""
        nonlocal fail_count, total_fails, last_fail_logged_t
        try:
            r = read_input_registers(client, 1, 0x02, count)
        except (ModbusError, serial.SerialException, OSError) as e:
            fail_count += 1
            total_fails += 1
            last_fail_logged_t = time.time()
            print(f"[{port}] modbus read failed (#{fail_count}, total={total_fails}, "
                  f"data_len={data_len}, count={count}): {type(e).__name__}: {e}")
            return None
        fail_count = 0
        return r

    def reconnect():
        """Close and reopen the serial port. Cures wedged FTDI drivers and
        framer desync. Doesn't re-write the sample-rate register because
        the sensor retains it across our local reconnect."""
        nonlocal fail_count
        print(f"[{port}] {fail_count} consecutive failures — closing + reopening serial")
        try:
            client.close()
        except Exception as e:
            print(f"[{port}] close raised (ignoring): {e}")
        time.sleep(0.1)
        try:
            client.open()
            print(f"[{port}] reconnected")
        except Exception as e:
            print(f"[{port}] reconnect failed: {e} — sleeping 1s before retry")
            time.sleep(1.0)
        fail_count = 0

    def _drain():
        """Read and discard until the line is quiet, then flush. Resyncs after
        a slow/late response so a stale reply can't be mis-parsed as the next
        request's response (the 'wrong byte count' desync)."""
        old_to = client.timeout
        client.timeout = 0.03
        try:
            while client.read(256):
                pass
        except Exception:
            pass
        finally:
            client.timeout = old_to
        try:
            client.reset_input_buffer()
        except Exception:
            pass

    def _abort_metrics():
        """Raw streaming takes priority: bail out of metric reads the moment
        raw is requested (page switch to waveform/inference, or a recording),
        metrics are turned off, or we're shutting down. Metric reads share the
        serial line with raw, so they must yield rather than block it."""
        return (stop_event.is_set()
                or active_event.is_set()
                or (metrics_event is not None and not metrics_event.is_set()))

    def _hold(addr, count):
        """FC03 read with retries. Reads are slow + occasionally desync; on any
        failure we drain the line before retrying so a stale late response
        doesn't corrupt the next read. Bails via _MetricsAbort if raw streaming
        is requested mid-sweep. Raises the last error if all attempts fail."""
        last_err = None
        for attempt in range(METRIC_READ_RETRIES):
            if _abort_metrics():
                raise _MetricsAbort()
            if attempt:
                _drain()
            time.sleep(METRIC_READ_GAP_S)
            try:
                return read_holding_registers(client, 1, addr, count)
            except (ModbusError, serial.SerialException, OSError) as e:
                last_err = e
        raise last_err

    def _triple(addr, divisor):
        """One FC03 read of 3 registers → [x, y, z] scaled. Mirrors the Rust
        client: registers are read unsigned. NOTE skewness can be negative on
        the wire — if it reads wrong, view as int16 here (flagged for the
        hardware-validation pass)."""
        regs = _hold(addr, 3)
        return [regs[0] / divisor, regs[1] / divisor, regs[2] / divisor]

    def read_metrics_batch():
        """Full FC03 sweep of every computed metric → one snapshot dict.
        Raises ModbusError on any failed read (caller skips the emit)."""
        temp = _hold(REG_TEMPERATURE, 1)[0] / 100.0
        g_freq = float(_hold(REG_GRAVITY_PRIM_FREQ, 1)[0])
        v_freq = float(_hold(REG_VELOCITY_PRIM_FREQ, 1)[0])
        return {
            "ts": time.time(),
            "temperature": temp,
            "gravity": {
                "rms":      _triple(REG_GRAVITY_RMS, 1000.0),
                "peak":     _triple(REG_GRAVITY_PEAK, 1000.0),
                "crest":    _triple(REG_GRAVITY_CREST, 1000.0),
                "skewness": _triple(REG_GRAVITY_SKEWNESS, 1000.0),
                "kurtosis": _triple(REG_GRAVITY_KURTOSIS, 1000.0),
                "primary_freq": g_freq,
            },
            "velocity": {
                "rms":   _triple(REG_VELOCITY_RMS, 100.0),
                "peak":  _triple(REG_VELOCITY_PEAK, 100.0),
                "crest": _triple(REG_VELOCITY_CREST, 100.0),
                "primary_freq": v_freq,
            },
        }

    # Metric polling state (used only when metrics_event is set).
    metrics_were_on = False
    last_kurt_regs = None       # raw kurtosis registers, for change detection
    last_kurt_poll_t = 0.0
    last_metric_emit_t = 0.0

    def poll_metrics_step():
        """Called once per loop iteration when metrics are active. Polls
        kurtosis at KURT_POLL_INTERVAL_S; emits a full batch when it changes,
        on the first poll, or after METRIC_FALLBACK_S of no change."""
        nonlocal last_kurt_regs, last_kurt_poll_t, last_metric_emit_t
        now = time.time()
        if now - last_kurt_poll_t < KURT_POLL_INTERVAL_S:
            return
        last_kurt_poll_t = now
        # FC03 metric reads are slow — give them a generous timeout, then
        # restore the fast raw-read timeout no matter how we exit.
        client.timeout = METRIC_READ_TIMEOUT_S
        try:
            _drain()    # start each poll cycle on a clean, synced line
            try:
                kurt = tuple(_hold(REG_GRAVITY_KURTOSIS, 3))
            except _MetricsAbort:
                return
            except (ModbusError, serial.SerialException, OSError) as e:
                print(f"[{port}] kurtosis poll failed: {e}")
                return
            changed = (kurt != last_kurt_regs)
            fallback = (now - last_metric_emit_t >= METRIC_FALLBACK_S)
            if not (changed or fallback):
                return
            try:
                snap = read_metrics_batch()
            except _MetricsAbort:
                return    # raw took priority mid-sweep — drop this partial batch
            except (ModbusError, serial.SerialException, OSError) as e:
                print(f"[{port}] metrics batch failed: {e}")
                return
            last_kurt_regs = kurt
            last_metric_emit_t = now
            if metrics_queue is not None:
                try:
                    metrics_queue.put_nowait((port, snap))
                except Exception:
                    pass    # full / closed — drop, next emit replaces it anyway
        finally:
            client.timeout = timeout

    was_active = False
    while not stop_event.is_set():
        raw_on = active_event.is_set()
        met_on = metrics_event.is_set() if metrics_event is not None else False

        if not raw_on and not met_on:
            if was_active:
                print(f"[{port}] deactivated — reader paused")
                was_active = False
                fill = 0
                raw_accum = []; raw_accum_n = 0
            if metrics_were_on:
                metrics_were_on = False
                last_kurt_regs = None
            # Block on the raw event so we wake the instant the main process
            # activates us. (Metrics activation is caught on the next tick;
            # 0.5 s latency to first metric is fine.)
            active_event.wait(timeout=0.5)
            continue

        # Metrics activation transition — force a fresh first emit.
        if met_on and not metrics_were_on:
            print(f"[{port}] metrics polling active")
            metrics_were_on = True
            last_kurt_regs = None
            last_metric_emit_t = 0.0
            last_kurt_poll_t = 0.0
        elif not met_on and metrics_were_on:
            metrics_were_on = False

        # Poll metrics FIRST so the raw block's early-`continue`s below can't
        # starve it. poll_metrics_step() self-gates on KURT_POLL_INTERVAL_S.
        if met_on:
            poll_metrics_step()

        if not raw_on:
            # Metrics-only mode: no FC04 streaming. Sleep so we don't busy-spin
            # between kurtosis polls.
            if was_active:
                was_active = False
                fill = 0
                raw_accum = []; raw_accum_n = 0
            time.sleep(0.05)
            continue

        if not was_active:
            # Just transitioned inactive → active. Re-write the sample-rate
            # register; the sensor uses this as a FIFO reset so subsequent
            # reads return fresh samples instead of whatever was sitting in
            # the on-chip buffer while we were paused.
            print(f"[{port}] activated — resetting FIFO via sample-rate write")
            try:
                write_single_register(client, 1, 0x01, sample_rate)
            except (ModbusError, serial.SerialException, OSError) as e:
                print(f"[{port}] FIFO reset failed: {e}; retry in 0.5s")
                time.sleep(0.5)
                continue
            regs = safe_read(1)
            data_len = regs[0] if regs is not None else 0
            last_log_t = time.time()
            emits_since_log = 0
            was_active = True

        # Three-branch read (mirror DAQ_Modbus_MultiChs_v1.3.py:109-117)
        if data_len >= MAX_PACKET:
            regs = safe_read(1 + MAX_PACKET)
        elif data_len <= 6:                # < 2 complete XYZ triplets
            time.sleep(0.001)
            regs = safe_read(1)
            if regs is not None:
                data_len = regs[0]
            elif fail_count >= RECONNECT_AFTER_FAILS:
                reconnect()
                data_len = 0
            continue
        else:
            regs = safe_read(1 + data_len)

        if regs is None:
            if fail_count >= RECONNECT_AFTER_FAILS:
                reconnect()
                data_len = 0      # force the small-read branch on next iter
            continue

        data_len = regs[0]                # updated remaining length for next iteration

        # Decode: list[int] → uint16 → int16 view (no copy, two's-complement
        # reinterpret) → float32 array via fused cast+scale ufunc.
        raw = np.array(regs[1:], dtype=np.uint16).view(np.int16)
        samples = (raw * SCALE).reshape(-1, 3)

        # Recording tap — local to this process, no IPC. feed() is cheap
        # when no session is active (one-line port-match check). Keep `samples`
        # as a separate small array so the recorder's stage-1 chunk list can
        # hold the reference safely while we compact the ring buffer below.
        recorder.feed(port, samples)

        # Waveform tap (fast cadence): accumulate fresh samples and emit a
        # small chunk every RAW_CHUNK_SIZE to the waveform ring buffer.
        if raw_queue is not None:
            raw_accum.append(samples)
            raw_accum_n += samples.shape[0]
            if raw_accum_n >= RAW_CHUNK_SIZE:
                chunk = np.concatenate(raw_accum, axis=0)
                _emit_raw_chunk(raw_queue, port, chunk)
                raw_accum = []
                raw_accum_n = 0

        # Slice-assign into the pre-allocated buffer instead of np.row_stack —
        # no allocation, no full-buffer memcpy per iteration.
        n = samples.shape[0]
        buffer[fill:fill + n] = samples
        fill += n

        # Sliding-window emission: emit whenever we have >= WINDOW_SIZE
        # samples, then drop the oldest HOP_SIZE and wait for HOP_SIZE new.
        while fill >= window_size:
            _emit_window(window_queue, port, buffer[:window_size].copy(), max_qsize)
            # Compact the tail. .copy() avoids overlapping-slice UB when
            # fill > 2*hop_size (happens whenever a packet pushes fill past
            # window_size by more than one MAX_PACKET).
            new_fill = fill - hop_size
            if new_fill > 0:
                buffer[:new_fill] = buffer[hop_size:fill].copy()
            fill = new_fill
            emits_since_log += 1

        now = time.time()
        if now - last_log_t >= 1.0:
            print(f"[{port}] emit rate: {emits_since_log}/s · data_len={data_len} "
                  f"· buffer={fill} · total_fails={total_fails}")
            emits_since_log = 0
            last_log_t = now

    client.close()


def get_existed_serial_ports():
    if sys.platform.startswith('win'):
        candidates = ['COM%s' % (i + 1) for i in range(256)]
    elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
        candidates = glob.glob('/dev/ttyUSB*')
        # Drop USB-serial latency from 16ms to 1ms for sustained 7812 Hz throughput.
        for port in candidates:
            try:
                dev = port.split('/')[-1]
                with open(f'/sys/bus/usb-serial/devices/{dev}/latency_timer', 'w') as f:
                    f.write('1\n')
            except OSError as e:
                print(f"[{port}] latency_timer write failed: {e}")
    elif sys.platform.startswith('darwin'):
        candidates = glob.glob('/dev/tty.*')
    else:
        raise EnvironmentError('Unsupported platform')

    available = []
    for port in candidates:
        try:
            s = serial.Serial(port)
            s.close()
            available.append(port)
        except (OSError, serial.SerialException):
            pass
    return available
