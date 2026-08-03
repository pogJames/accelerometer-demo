# Getting started

How to run the demo on a **Matrix800** gateway.

See also: [architecture.md](architecture.md) for how it works,
[api.md](api.md) to build on top of it.

---

## 1. What you need

- A **Matrix800** gateway with its Linux image, on your network.
- One or more **accelerometers** on RS485. Each uses one USB port
  (`/dev/ttyUSB0`, `/dev/ttyUSB1`, ...).

You can also run it on a normal PC with no sensor and no NPU. It starts in
*stub* mode (see [step 5](#5-first-run)) so you can look at the pages. Real data
needs the board.

---

## 2. Install

```bash
git clone https://github.com/pogJames/accelerometer-demo
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 3. Choose your NPU driver

> **Read this step before you run on the board.** The Matrix800 has one NPU, but its Linux image uses **one of two drivers**.

Check which driver your board is currently using:

```bash
ls /dev/ethos*    # exists → NXP driver
ls /dev/accel     # exists → Mesa driver
```

Then set three values in `inference.py` to match:

| | **NXP driver** (default) | **Mesa driver** |
|---|---|---|
| Runtime import (`_try_load_interpreter`) | `from tflite_runtime.interpreter import ...` | `from ai_edge_litert.interpreter import ...` |
| `DELEGATE_PATH` | `/usr/lib/libethosu_delegate.so` | `/usr/local/lib/aarch64-linux-gnu/libteflon.so` |
| `NPU_MODEL_PATH` | `models/vibration_backbone_int8_vela.tflite` | `models/vibration_backbone_int8.tflite` |

The code is set for the **NXP driver**. If your board uses the **Mesa driver**,
change those three lines. There is no auto-detect.

Both model files are already in `models/`. You only pick which one to load.
See [architecture.md → NPU backend](architecture.md#npu-backend) for why a
mismatch is silent.

---

## 4. Connect the sensors

List your ports:

```bash
ls /dev/ttyUSB*
```

Set which ports the app reads. Edit `ALLOWED_PORTS` in `sensor_reader.py`. The
default is one port. List every sensor you have:

```python
ALLOWED_PORTS = ['/dev/ttyUSB0', '/dev/ttyUSB1', '/dev/ttyUSB2', '/dev/ttyUSB3']
```

**Serial permission.** Reading `/dev/ttyUSB*` needs access -> run as `root`

---

## 5. First run

```bash
python app.py
```

The server listens on `0.0.0.0:80`. **Port 80 needs root.** -> run as `root`

Open a browser to the board: `http://<board-ip>/`.

Check the **inference mode** in the sidebar:

- **`npu`** — driver, delegate, and model loaded. It works.
- **`stub`** — runtime, delegate, or model file not found. The app serves a fixed placeholder class so the pages still load. This is normal on a PC. On the board it means step 3 is not done. The startup log prints which file is missing.

This build has **no CPU fallback**. The mode is `npu` or `stub`.

---

## 6. Try it

1. Open `/`. You should see a live waveform for each sensor. A flat line means
   the sensor is not sending data. Check the wiring and `ALLOWED_PORTS`.
2. Record two motions: `/record` → set a label (for example `steady`), record a
   few seconds. Repeat for a second label.
3. Train: `/train` → pick 2 or more labels → **Train model** (takes seconds).
4. Open `/inference`. It now shows the live class.

Full page and API reference: [api.md](api.md).

---

## 7. Run as a service *(optional)*

To keep the dashboard running after reboot, add a `systemd` unit. Example
`/etc/systemd/system/npu-python.service`:

```ini
[Unit]
Description=NPU vibration dashboard
After=network.target

[Service]
Type=simple
WorkingDirectory=/root/project4
ExecStart=/root/project4/.venv/bin/python app.py
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now npu-python
journalctl -u npu-python -f          # view the log
```

---

## Troubleshooting

| Problem | Cause and fix |
|---|---|
| Sidebar shows **`stub`** on the board | Runtime, delegate, or model missing or mismatched. Read the startup log — it names the missing file. Re-check [step 3](#3-choose-your-npu-driver). |
| Result never changes or looks random | Usually the wrong driver or model. Test the `.tflite` alone: `python smoketest.py` (push zeros, ones, random — outputs must differ). See [architecture.md → NPU backend](architecture.md#npu-backend). |
| **`Permission denied`** on `/dev/ttyUSB*` | Not root and not in `dialout` group — see [step 4](#4-connect-the-sensors). |
| **`Permission denied`** on port 80 | Port 80 needs root — use `sudo` or `setcap`, see [step 5](#5-first-run). |
| No ports found / flat waveform | `ALLOWED_PORTS` does not match the real ports. Check `ls /dev/ttyUSB*` and wiring. |
| `total_fails` rising in the log | Sensor is slow to answer. Raise `READ_TIMEOUT_S` in `sensor_reader.py`. See [modules.md → sensor_reader](modules.md#sensor_readerpy). |
