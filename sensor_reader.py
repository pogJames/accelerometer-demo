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
ALLOWED_PORTS = ['/dev/ttyUSB0', '/dev/ttyUSB1', '/dev/ttyUSB2', '/dev/ttyUSB3']

WINDOW_SIZE   = 2604     # 1/2 sec at 7812 Hz — must match the trained backbone
HOP_SIZE      = 1302     # 50 % overlap → 4 emits/sec
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
    from fast_modbus import (read_input_registers, write_single_register,
                              ModbusError)
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

    was_active = False
    while not stop_event.is_set():
        if not active_event.is_set():
            if was_active:
                print(f"[{port}] deactivated — reader paused")
                was_active = False
                fill = 0
            # Block on the event so we wake up the instant the main process
            # activates us, instead of spinning every 50 ms.
            active_event.wait(timeout=0.5)
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
