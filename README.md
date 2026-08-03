# accelerometer-demo

**English** · [繁體中文](README.zh-TW.md)

Real-time vibration monitor + classifier. A tri-axial accelerometer streams raw
XYZ over Modbus RTU → a frozen CNN backbone extracts 128-d embeddings → a
cosine-similarity head classifies motion → live waveform, FFT spectrum, sensor
metrics and inference results surface in a Flask dashboard. The backbone is
never retrained; the head is computed in seconds from your own recordings.

## Hardware

1. Matrix 800
2. 12–24 V power brick w/ terminal block (Matrix 800)
3. Tri-axial accelerometer
4. 5 V power brick (accelerometer)

## Setup & launch

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python app.py                   # http://localhost (port 80)
```

Off the target board (no `tflite_runtime` / Ethos-U delegate) inference runs in
`stub` mode — everything else works for development. The app auto-detects serial
ports in `ALLOWED_PORTS` (`/dev/ttyUSB0..3`), one reader subprocess per sensor.

## Pages

| URL | What |
|---|---|
| `/` | Live waveform — raw scroll + FFT spectrum, per sensor |
| `/inference` | Live classification cards (majority vote over last 7) |
| `/metrics` | Per-sensor metric tables (gravity / velocity / temperature) |
| `/record` | Record raw XYZ to `data/<label>.bin` |
| `/train` | Train the classifier head (≥ 2 labels, seconds) |
| `/settings` | Per-port friendly aliases |

## More docs

| Doc | Contents |
|---|---|
| [architecture.md](pages/architecture.md) | Worker/data-flow diagram, dual cadence, inference modes |
| [http-api.md](pages/http-api.md) | SSE streams + REST endpoints |
| [pages.md](pages/pages.md) | Full per-page UI reference |
| [tuning.md](pages/tuning.md) | Key files + tuning-knob constants |
| [notes.md](pages/notes.md) | Per-module design rationale (the code's `See pages/notes.md`) + NPU porting |
| [setup.md](pages/setup.md) | Board / OS image setup |
