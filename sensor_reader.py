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
RAW_CHUNK_SIZE = 217     # fast waveform cadence (~36 Hz/port at 7812 Hz)
MAX_PACKET    = 41 * 3   # Modbus packet upper bound: 41 XYZ triplets
TURN_GRAVITY  = 8192     # raw int16 / 8192 -> G value
SAMPLE_RATE   = 7812
SCALE         = np.float32(1.0 / TURN_GRAVITY)   # fused int16→float32 decode

READ_TIMEOUT_S          = 0.05   # see pages/notes.md for the 50 ms rationale
RECONNECT_AFTER_FAILS   = 5      # consecutive fails → drop + reopen the port

# ── Computed-metric registers (FC03 holding registers) ────────────────────
# Each metric is its OWN FC03 read at its base address — the addresses ALIAS,
# so a contiguous block read returns garbage. Mirrors the Rust client. pages/notes.md
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

# Poll kurtosis (slowest metric) and emit a full batch when it changes;
# fallback forces a refresh even if it sits still. See pages/notes.md.
KURT_POLL_INTERVAL_S = 0.4
METRIC_FALLBACK_S    = 5.0
# FC03 reads are slow + variable (~0.5-1 s); a premature timeout desyncs the
# stream forever. Generous timeout + line drain to resync. pages/notes.md.
METRIC_READ_TIMEOUT_S = 5.0
METRIC_READ_RETRIES   = 3
METRIC_READ_GAP_S     = 0.001


class _MetricsAbort(Exception):
    # Bail out of an in-progress slow metric sweep when raw streaming is
    # requested or metrics are turned off.
    pass


def _emit_window(window_queue, port, window, max_qsize):
    # Drop-oldest on full so inference sees the freshest window.
    # qsize() raises NotImplementedError on macOS — tolerate that.
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
    # Best-effort drop-oldest push to the waveform queue; never blocks the read loop.
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
    # Sibling thread: drains RPC requests, dispatches to the local
    # RecordingManager. Decoupled so an RPC can't stall the modbus poll.
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
    # Subprocess entry point; owns the pyserial client + RecordingManager for
    # one port. See pages/notes.md for the process model and arg contract.
    #
    # `spawn` gives the child a fresh interpreter — re-apply the bundled-deps
    # path before importing project modules.
    current_path = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(current_path, "site-packages"))
    from fast_modbus import (read_input_registers, read_holding_registers,
                              write_single_register, ModbusError)
    from recorder import RecordingManager

    print(f"[{port}] reader subprocess started pid={os.getpid()} ppid={os.getppid()}")

    # Real-time priority so the reader preempts Flask threads on a shared core.
    # Requires root / CAP_SYS_NICE; log and continue otherwise.
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

    # Pre-allocated ring: WINDOW_SIZE valid samples + one MAX_PACKET tail
    # before the next emit compacts it. Slice-assign, no alloc in the hot path.
    buffer = np.empty((window_size * 2, 3), dtype=np.float32)
    fill = 0
    last_log_t = time.time()
    emits_since_log = 0
    fail_count = 0
    total_fails = 0
    last_fail_logged_t = 0.0

    raw_accum = []
    raw_accum_n = 0

    def safe_read(count):
        # Returns registers on success, None on any failure. Tracks consecutive
        # failures so the outer loop can decide when to reconnect.
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
        # Cures wedged FTDI drivers and framer desync. Sensor retains the
        # sample-rate register across a local reconnect, so no re-write.
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
        # Read+discard until the line is quiet, then flush — resyncs after a
        # slow/late response so a stale reply isn't mis-parsed as the next one.
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
        # Raw streaming takes priority — bail out of metric reads when raw is
        # requested, metrics are off, or we're shutting down.
        return (stop_event.is_set()
                or active_event.is_set()
                or (metrics_event is not None and not metrics_event.is_set()))

    def _hold(addr, count):
        # FC03 read with retries; drain before each retry to clear a stale late
        # response. Bails via _MetricsAbort if raw is requested mid-sweep.
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
        # One FC03 read of 3 regs → [x, y, z] scaled, read unsigned.
        # NOTE skewness can be negative on the wire — flagged for hw validation.
        regs = _hold(addr, 3)
        return [regs[0] / divisor, regs[1] / divisor, regs[2] / divisor]

    def read_metrics_batch():
        # Full FC03 sweep → one snapshot dict. Raises ModbusError on any read.
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

    metrics_were_on = False
    last_kurt_regs = None
    last_kurt_poll_t = 0.0
    last_metric_emit_t = 0.0

    def poll_metrics_step():
        # Once-per-iteration when metrics active: poll kurtosis at
        # KURT_POLL_INTERVAL_S, emit a full batch on change / first poll /
        # METRIC_FALLBACK_S.
        nonlocal last_kurt_regs, last_kurt_poll_t, last_metric_emit_t
        now = time.time()
        if now - last_kurt_poll_t < KURT_POLL_INTERVAL_S:
            return
        last_kurt_poll_t = now
        client.timeout = METRIC_READ_TIMEOUT_S
        try:
            _drain()
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
                return
            except (ModbusError, serial.SerialException, OSError) as e:
                print(f"[{port}] metrics batch failed: {e}")
                return
            last_kurt_regs = kurt
            last_metric_emit_t = now
            if metrics_queue is not None:
                try:
                    metrics_queue.put_nowait((port, snap))
                except Exception:
                    pass
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
            # Block on raw event to wake instantly on activation; metrics
            # activation is caught on the next tick.
            active_event.wait(timeout=0.5)
            continue

        if met_on and not metrics_were_on:
            print(f"[{port}] metrics polling active")
            metrics_were_on = True
            last_kurt_regs = None
            last_metric_emit_t = 0.0
            last_kurt_poll_t = 0.0
        elif not met_on and metrics_were_on:
            metrics_were_on = False

        # Poll metrics FIRST so the raw block's early-continues can't starve it.
        if met_on:
            poll_metrics_step()

        if not raw_on:
            if was_active:
                was_active = False
                fill = 0
                raw_accum = []; raw_accum_n = 0
            time.sleep(0.05)
            continue

        if not was_active:
            # inactive → active: re-write sample-rate register; the sensor uses
            # this as a FIFO reset so reads return fresh samples.
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
                data_len = 0
            continue

        data_len = regs[0]                # updated remaining length for next iter

        # list[int] → uint16 → int16 view (no-copy two's-complement) → float32
        # via fused cast+scale ufunc.
        raw = np.array(regs[1:], dtype=np.uint16).view(np.int16)
        samples = (raw * SCALE).reshape(-1, 3)

        # Recording tap — local, no IPC. Keep `samples` as its own small array
        # so the recorder can hold the reference while we compact below.
        recorder.feed(port, samples)

        if raw_queue is not None:
            raw_accum.append(samples)
            raw_accum_n += samples.shape[0]
            if raw_accum_n >= RAW_CHUNK_SIZE:
                chunk = np.concatenate(raw_accum, axis=0)
                _emit_raw_chunk(raw_queue, port, chunk)
                raw_accum = []
                raw_accum_n = 0

        n = samples.shape[0]
        buffer[fill:fill + n] = samples
        fill += n

        while fill >= window_size:
            _emit_window(window_queue, port, buffer[:window_size].copy(), max_qsize)
            # .copy() avoids overlapping-slice UB when fill > 2*hop_size.
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
        # Drop USB-serial latency 16ms→1ms for sustained 7812 Hz throughput.
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
