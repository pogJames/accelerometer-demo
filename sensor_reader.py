"""Worker 1 — Modbus reader, one subprocess per sensor.

Runs in its OWN process (not a thread). pymodbus's busy-poll loop on the
serial buffer is GIL-bound; sharing the GIL with Flask / InferenceWorker /
GC pauses lets the sensor's hardware FIFO (max 65535 regs ≈ 2.8 s at
7812 Hz) overflow and silently drop samples. A dedicated process means
nothing in the main interpreter can stall this read loop.

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
import subprocess
import sys
import threading
import time

import numpy as np
import serial

# Up to 4 sensors. Add /dev/ttyUSB1..USB3 here as more come online.
ALLOWED_PORTS = ['/dev/ttyUSB0', 'COM3', 'COM4']

WINDOW_SIZE   = 2604     # 1/2 sec at 7812 Hz — must match the trained backbone
HOP_SIZE      = 1302     # 50 % overlap → 4 emits/sec
MAX_PACKET    = 41 * 3   # Modbus packet upper bound: 41 XYZ triplets
TURN_GRAVITY  = 8192     # raw int16 / 8192 -> G value
SAMPLE_RATE   = 7812

# pymodbus stalls during impact-induced CRC errors / brief sensor silences.
# Cap each failed read at this many seconds; pair with retries=0 on the
# client so a single bad frame doesn't compound into ~12 s of pymodbus
# internal retry (default in pymodbus 3.x).
#
# 50 ms is well above the sensor's actual response time at 3 Mbaud
# (request ~30 µs, processing a few ms, response ~700 µs — total normally
# <10 ms) but turns a bad-packet stall into a perceptually invisible
# blip: ~50 ms timeout + one MAX_PACKET catchup read ≈ 100 ms total.
# If you start seeing total_fails climbing during normal (non-impact)
# operation, bump this up — the sensor's tail latency might be longer.
READ_TIMEOUT_S          = 0.05
MODBUS_RETRIES          = 0
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
    can't stall pymodbus."""
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


def reader_process_main(port, window_queue, req_q, resp_q, stop_event, data_dir,
                        port_baud=3000000, bytesize=8, parity='N', stopbits=1,
                        timeout=READ_TIMEOUT_S, sample_rate=SAMPLE_RATE,
                        window_size=WINDOW_SIZE, hop_size=HOP_SIZE, max_qsize=3):
    """Subprocess entry point. Owns the pymodbus client + RecordingManager
    for one serial port.

    window_queue — mp.Queue, push `(port, np.ndarray)` for inference.
    req_q / resp_q — mp.Queue pair, main process RPCs in / responses out.
    stop_event — mp.Event, set by main process to request shutdown.
    data_dir — absolute path, where recordings get written.
    """
    # `spawn` start method gives the child a fresh interpreter — the
    # sys.path mutation app.py does for bundled pymodbus is NOT inherited.
    # Re-apply it before importing pymodbus / recorder.
    current_path = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(current_path, "site-packages"))
    from pymodbus.client import ModbusSerialClient as ModbusClient
    from recorder import RecordingManager

    print(f"[{port}] reader subprocess started pid={os.getpid()} ppid={os.getppid()}")

    # iMX93 has 2 cores. Pin the reader to CPU 1 so Flask + InferenceWorker
    # on the main process (CPU 0) can't preempt the modbus read loop. Without
    # this, process separation alone still loses cycles to the kernel
    # scheduler when main is busy serving HTTP — that's why data_len was
    # spiking during page loads.
    if hasattr(os, "sched_setaffinity") and (os.cpu_count() or 1) > 1:
        try:
            non_zero_cpus = set(range(1, os.cpu_count()))
            os.sched_setaffinity(0, non_zero_cpus)
            print(f"[{port}] cpu affinity set to {sorted(non_zero_cpus)}")
        except (OSError, PermissionError) as e:
            print(f"[{port}] sched_setaffinity failed: {e}")

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

    client = ModbusClient(port=port,
                          baudrate=port_baud,
                          bytesize=bytesize,
                          parity=parity,
                          stopbits=stopbits,
                          timeout=timeout,
                          retries=MODBUS_RETRIES)
    client.connect()

    chip = client.read_input_registers(0x80, count=3, device_id=1).registers
    print(f"[{port}] ChipID: {hex(chip[0])}, {hex(chip[1])}, {hex(chip[2])}")
    print(f"[{port}] SampleRate: {sample_rate}")
    client.write_register(0x01, sample_rate, device_id=1)

    result = client.read_input_registers(0x02, count=1, device_id=1)
    data_len = result.registers[0]
    print(f"[{port}] Initial buffer length: {data_len}")

    # Drain the sensor FIFO once before emitting — otherwise pre-startup
    # backlog gets pumped through the model at 10× real-time as stale data.
    drain_total = 0
    drain_iters = 0
    while data_len > 6 and drain_iters < 200:
        count = min(data_len, MAX_PACKET)
        result = client.read_input_registers(0x02, count=1 + count, device_id=1)
        data_len = result.registers[0]
        drain_total += count
        drain_iters += 1
    print(f"[{port}] FIFO drain: discarded {drain_total} stale samples in {drain_iters} reads (data_len now {data_len})")

    buffer = np.empty((0, 3), dtype=np.float32)
    last_log_t = time.time()
    emits_since_log = 0
    fail_count = 0
    total_fails = 0
    last_fail_logged_t = 0.0

    def safe_read(count):
        """Returns the registers list on success, or None on any failure
        (timeout, CRC, error response, exception). Tracks consecutive
        failures so the outer loop can decide when to reconnect."""
        nonlocal fail_count, total_fails, last_fail_logged_t
        try:
            r = client.read_input_registers(0x02, count=count, device_id=1)
        except Exception as e:
            fail_count += 1
            total_fails += 1
            last_fail_logged_t = time.time()
            print(f"[{port}] modbus read raised (#{fail_count}, total={total_fails}, "
                  f"data_len={data_len}, count={count}): {type(e).__name__}: {e}")
            return None
        if r is None or (hasattr(r, "isError") and r.isError()) \
                     or not getattr(r, "registers", None):
            fail_count += 1
            total_fails += 1
            last_fail_logged_t = time.time()
            print(f"[{port}] modbus error response (#{fail_count}, total={total_fails}, "
                  f"data_len={data_len}, count={count}): {r}")
            return None
        fail_count = 0
        return r.registers

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
            client.connect()
            print(f"[{port}] reconnected")
        except Exception as e:
            print(f"[{port}] reconnect failed: {e} — sleeping 1s before retry")
            time.sleep(1.0)
        fail_count = 0

    while not stop_event.is_set():
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

        raw = np.array(regs[1:], dtype=np.uint16).astype(np.int16)
        samples = (raw / TURN_GRAVITY).reshape(-1, 3).astype(np.float32)

        # Recording tap — local to this process, no IPC. feed() is cheap
        # when no session is active (one-line port-match check).
        recorder.feed(port, samples)

        buffer = np.row_stack((buffer, samples))

        # Sliding-window emission: emit whenever we have >= WINDOW_SIZE
        # samples, then drop the oldest HOP_SIZE and wait for HOP_SIZE new.
        while buffer.shape[0] >= window_size:
            _emit_window(window_queue, port, buffer[:window_size].copy(), max_qsize)
            buffer = buffer[hop_size:]
            emits_since_log += 1

        now = time.time()
        if now - last_log_t >= 1.0:
            print(f"[{port}] emit rate: {emits_since_log}/s · data_len={data_len} "
                  f"· buffer={buffer.shape[0]} · total_fails={total_fails}")
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
            cmd = ('sudo bash -c "echo 1 > /sys/bus/usb-serial/devices/ttyUSB'
                   + port.split('USB')[-1] + '/latency_timer"')
            print(cmd)
            subprocess.run(cmd, shell=True, check=True, executable='/bin/bash')
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
