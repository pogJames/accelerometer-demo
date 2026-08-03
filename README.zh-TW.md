# npu-classifier

[English](README.md) · **繁體中文**

即時振動監控 + 分類器。三軸加速度計透過 Modbus RTU 串流原始 XYZ 取樣 → 凍結的 CNN 主幹（backbone）擷取 128 維嵌入向量 → 餘弦相似度分類頭（head）判斷運動類別 → 即時波形、FFT 頻譜、感測器指標與推論結果全部呈現在 Flask 儀表板上。主幹不需重新訓練；分類頭可從你自己的錄製資料在數秒內算出。


## 硬體設定

| 項目 | 說明 |
|---|---|
| 感測器 | 三軸加速度計，RS485/Modbus RTU |
| 介面 | USB–RS485 轉接器 → `/dev/ttyUSB0..3`（最多 4 顆感測器） |
| 鮑率（Baud rate） | **3 Mbps**（原始串流所必需） |
| 取樣率 | 標稱 7812 Hz（連線時寫入感測器暫存器 `0x01`） |
| Modbus FC04 | 暫存器 `0x02` — FIFO 計數 + 原始 XYZ 取樣 |
| Modbus FC03 | 計算後指標（RMS、峰值、峰度…）— 參見 `sensor_reader.py` 暫存器對照表 |


## 安裝與啟動

```bash
# 1. 建立虛擬環境並安裝相依套件
python -m venv .venv
source .venv/bin/activate       # Windows：.venv\Scripts\activate
pip install -r requirements.txt

# 2. 執行
python app.py                   # http://localhost（port 80）
FORCE_CPU=1 python app.py       # 繞過 NPU delegate（除錯／開發機）
```

App 會自動偵測 `ALLOWED_PORTS`（`/dev/ttyUSB0..3`）中列出的序列埠。每偵測到一顆感測器就產生一個 reader 子行程。


## 架構

```
W1  sensor_reader.py             每個 port 一個子行程
     ────────────────────────
     FC04 register 0x02   → 解碼 → /8192 → G
       │                         ├─→ recorder.feed()          （本地，無 IPC）
       │                         ├─→ 累積 2604 取樣視窗  ──┐
       │                         │   (hop=1302, ~6 Hz/port)           │
       │                         └─→ 累積 217 取樣區塊  ──┐ │
       │                             (~36 Hz/port)                  │ │
       │                                                            │ │
     FC03 指標批次  (峰度變化偵測, ~2-5 s) ─────────┐  │ │
                                                                 │  │ │
                                                                 │  │ │
                                                 metrics_queue ──┘  │ │
                                                    window_queue ───┘ │
                                                        raw_queue ────┘
                                                          │
                                  ────── 主行程 ────┴──────
                                       ↓               ↓                 ↓
                            InferenceWorker     raw drain →       metrics drain →
                            (分類)             WaveformAggregator   每個 port 的
                              ↓                .append() 到 ring     latest dict
                            RollingPreds       ↓                     ↓
                              ↓                display-tick 30 Hz   MetricsBus
                            SnapshotBus         render_tick()
                                                ↓
                                               WaveformBus

W3  app.py + Flask       （提供以下頁面 + HTTP API — 參見 § HTTP API）
     /                  即時波形（uPlot，原始捲動 + FFT）
     /inference         即時分類卡片
     /metrics           各感測器指標表
     /record            資料錄製（可就地刪除）
     /train             訓練分類頭
     /settings          各 port 別名
```

**Reader 的雙節奏。** `window_queue` 以 ~6 Hz 傳送完整的 2604 取樣視窗給模型；`raw_queue` 以 ~36 Hz 傳送 217 取樣的小區塊給波形。推論與波形完全解耦——彼此不會互相阻塞。

**環形緩衝波形。** `WaveformAggregator` 為每個 port 維護 2 秒的環形取樣緩衝。display-tick 執行緒每個 tick 讀取最後 N 個取樣（原始波形在大範圍重疊視窗下平順捲動），並以 ~6 Hz 從最後 `WINDOW_SIZE` 個取樣重新計算 FFT。

**推論模式**（顯示於側邊欄）：
- `npu` — 已載入 Ethos-U delegate，Vela int8 主幹跑在 NXP NPU 上
- `cpu` — 透過 `tflite_runtime` 跑 float32 主幹，不使用 delegate（`FORCE_CPU=1` 或沒有可用的 delegate）
- `stub` — 沒有模型檔或沒有 `tflite_runtime`；儀表板仍可正常渲染

**移植到其他 NPU。** 這是針對 NXP i.MX / Ethos-U 的單一路徑建置；所有旋鈕都在 `inference.py`，以 `PORT:` 標記。重新指定目標 = 改 3 個地方：

| 項目 | `inference.py` | NXP（預設） | Matrix800 / VeriSilicon |
|---|---|---|---|
| Runtime import | `_try_load_interpreter` `# PORT: runtime` | `tflite_runtime.interpreter` | `ai_edge_litert.interpreter` |
| Delegate `.so` | `DELEGATE_PATH` | `/usr/lib/libethosu_delegate.so` | `/usr/local/lib/aarch64-linux-gnu/libteflon.so` |
| NPU 模型 | `NPU_MODEL_PATH` | `..._int8_vela.tflite` | `..._int8.tflite`（非 Vela） |

`CPU_MODEL_PATH` fallback 與 `FORCE_CPU=1` 繞過不變。


## HTTP API

所有請求／回應主體皆為 JSON。錯誤時回傳 `{"error": "..."}` 並帶 4xx/5xx 狀態碼。port 為原始的 `/dev/ttyUSBx` 字串。

### SSE 串流

長連線的 `text/event-stream`；每個 `data:` 訊框是一份 JSON 快照，於狀態變化時推送（閒置時送出 heartbeat 註解）。在頁面啟用對應 port 之前，串流不會產生資料——參見 [頁面](#頁面)。

| 端點 | 推送觸發時機 | 訊框內容 |
|---|---|---|
| `GET /stream/inference` | 新的多數決鎖存（latch） | `{ports:{<port>:{majority_class_name, majority_count, window_count, recent[], display_seq, latest_ts}}, class_labels[], label_colors{}, active_ports[], inference_mode, now}` |
| `GET /stream/waveform` | 30 Hz display tick | `{ports:{<port>:{raw, fft, raw_seq, fft_seq, ts}}, raw_axis[], freq_axis_hz[], sample_rate, window_size, fft_bins, …}` |
| `GET /stream/metrics` | FC03 批次（~2–5 s） | `{ports:{<port>:{temperature, gravity{rms,peak,crest,skewness,kurtosis,primary_freq}, velocity{rms,peak,crest,primary_freq}, ts}}, open_ports[], now}` |

### 啟用與設定（POST）

| 端點 | 請求主體 | 回應 | 作用 |
|---|---|---|---|
| `POST /api/active_port` | `{port, active: bool}` | `{port, active}` | 開啟／關閉某 port 的 **原始 + 推論** 串流 |
| `POST /api/metrics_active` | `{port, active: bool}` | `{port, active}` | 開啟／關閉 **FC03 指標** 輪詢（與原始串流獨立） |
| `POST /api/waveform_config` | `{fft_max_hz?, raw_samples?}` | 套用後的值（`{fft_max_hz, fft_bins, raw_samples}`） | 即時波形／FFT 調整旋鈕 |
| `POST /api/port_alias` | `{port, alias}` | `{port, alias}` | 設定某 port 的顯示別名 |

### 錄製

| 端點 | 請求主體 | 回應 |
|---|---|---|
| `GET /api/recordings` | — | `{labels:[{name, samples, windows, eligible}], ports[], sample_rate, min_samples, min_windows, window_size, data_dir}` |
| `GET /api/record/status` | — | `{session: {status, name, port, progress, elapsed_s, samples_written, target_samples, file_path} \| null}`（用戶端 ~1 Hz 輪詢） |
| `POST /api/record/start` | `{name, target_samples, port, mode:"append"\|"overwrite"}` | `{session}` |
| `POST /api/record/cancel` | — | `{session}`（保留目前已寫入的取樣） |
| `POST /api/recordings/delete` | `{name}` | `{deleted}` — 若該標籤正在錄製則回 `409` |

### 訓練

| 端點 | 請求主體 | 回應 |
|---|---|---|
| `GET /api/train/status` | — | `{session, head_labels[], head_colors{}, backbone, backbone_mode}` |
| `POST /api/train/start` | `{labels:[...]}`（≥ 2 個符合條件） | `{session}` |
| `POST /api/train/cancel` | — | `{session}` |

### 單次快照（JSON）

非串流的 GET，回傳與對應 `/stream/*` 相同的內容——用於 `curl`／`jq` 除錯，不必維持 SSE 長連線。無論是否有 port 被啟用，都會反映目前快取的狀態。

| 端點 | 回應 |
|---|---|
| `GET /api/inference` | 目前的 `/stream/inference` 內容 |
| `GET /api/metrics` | 目前的 `/stream/metrics` 內容 |
| `GET /api/waveform` | 目前的 `/stream/waveform` 內容 |


## 頁面

### 即時波形 `/`

2 × 2 的各感測器 uPlot 畫布格線。

- **原始模式** — 捲動的時域軌跡，X/Y/Z 疊圖，「最後 N 個取樣」可調整至最多 1 秒，更新率可選 10/30/60 fps。
- **FFT 模式** — 幅值頻譜，FFT 範圍以 Hz 為單位可調（上限 800 Hz）。
- 每張圖上方有各軸統計小標籤（原始模式顯示 min..max，FFT 模式顯示峰值頻率）。點擊小標籤可隱藏／顯示該軸。
- 滑鼠停在圖上會出現內嵌提示，顯示該點的取樣值。

### 即時推論 `/inference`

每顆感測器一張手風琴式卡片。顯示最後 7 筆結果的多數決類別、最新信心值，以及一條顏色編碼的圓點條。由 SSE 驅動，不輪詢。

### 指標資料 `/metrics`

每顆感測器一張卡片，顯示溫度 + 重力表（RMS / 峰值 / 峰值因數 / 偏度 / 峰度 × XYZ + 主頻）+ 速度表（RMS / 峰值 / 峰值因數 × XYZ + 主頻）。Reader 以 ~2 Hz 輪詢峰度，僅在峰度變化時（或 5 秒 fallback）才發出一整批所有指標——因此頁面上每個欄位都一起以 ~2–5 秒的慢節奏更新。

啟用 `/metrics` **不會** 在該 port 上啟動原始串流——只做 FC03 輪詢。`/` 與 `/inference` 頁面會另外啟用原始串流；同一 port 上原始串流一律優先於指標輪詢。

### 資料錄製 `/record`

將原始 XYZ 取樣錄製到 `data/<name>.bin`（float32 小端序，3 通道交錯）。「既有資料」表格中每一列都有垃圾桶圖示的刪除按鈕（會跳出確認提示）。支援附加（Append）／覆寫（Overwrite）模式；進度透過 SSE 串流。

> 若要將錄製的 `.bin` 轉成 `.csv`（`x,y,z`，不含時間戳），請手動執行轉換腳本：`python bin2csv.py data/<name>.bin`（也接受 glob 或目錄）。

### 訓練模型 `/train`

「訓練」= 為每個標籤計算一個平均嵌入向量。無梯度，只需數秒。選擇 ≥ 2 個符合條件的標籤 → 點擊 **Train model** → 寫出 `classifier_head.json` → `InferenceWorker` 熱重載，無需重啟 app。每次重新訓練都會隨機重新指派顏色槽。

### 設定 `/settings`

每個偵測到的 port 一列；輸入好記的別名（例如「左前馬達」），即時存入 `port_aliases.json`。別名會在每個顯示 port 名稱的頁面上呈現；原始的 `/dev/ttyUSBx` 路徑仍為內部 ID。


## 主要檔案

| 檔案 | 角色 |
|---|---|
| `app.py` | Flask app，串接 reader 子行程 + InferenceWorker + drain + display tick，托管所有 SSE 端點 |
| `sensor_reader.py` | 每個 port 一個 reader 子行程：FC04 雙節奏原始串流 + FC03 指標批次 |
| `fast_modbus.py` | 精簡 Modbus RTU 用戶端 — 直接以 pyserial 實作 FC03 / FC04 / FC06，不用 pymodbus |
| `waveform.py` | `WaveformAggregator` — 每個 port 的環形緩衝，`append()` + 30 Hz `render_tick()`（原始波形平滑／降取樣，FFT 節流至 ~6 Hz） |
| `inference.py` | `InferenceWorker` — 單一分類迴圈：視窗 → 主幹嵌入 → 分類頭判斷 → rolling.push |
| `classifier.py` | `ClassifierHead` — 餘弦相似度，載入／存成 JSON |
| `trainer.py` | `TrainerManager` — 原型（prototype）計算，背景執行緒 |
| `recorder.py` | `RecordingManager` + `delete_label()` — 兩階段佇列，二進位寫入執行緒 |
| `state.py` | `RollingPredictions`、`SnapshotBus`、`LatestSlot` |
| `port_aliases.json` | 別名覆寫 — 由 `/settings` 寫入，載入到每個模板的 context |
| `models/vibration_backbone_int8.tflite` | 凍結的 int8 CNN 主幹（NPU 目標） |
| `classifier_head.json` | 訓練好的分類頭 — 由 trainer 寫入，供 inference 讀取 |
| `data/` | 錄製檔 — `<label>.bin` float32 原始 XYZ |


## 調整旋鈕

| 常數 | 檔案 | 預設 | 作用 |
|---|---|---|---|
| `WINDOW_SIZE` | `sensor_reader.py` | 2604 | **模型** 輸入視窗（取樣數）。須與主幹相符——未重新訓練前勿更動。 |
| `HOP_SIZE` | `sensor_reader.py` | 1302 | 推論用的視窗跨步（~6 emits/s）。 |
| `RAW_CHUNK_SIZE` | `sensor_reader.py` | 217 | 波形路徑的新鮮取樣區塊大小（~36 emits/s）。 |
| `WINDOW_SIZE` | `waveform.py` | 2604 | **FFT** 視窗（取樣數）。與模型視窗獨立——調整以取得更細的頻率解析度。 |
| `RING_CAPACITY` | `waveform.py` | `SAMPLE_RATE*2` | 每個 port 的環形取樣緩衝（~2 s）。須 ≥ 最大的 FFT/原始視窗。 |
| `FFT_INTERVAL_S` | `waveform.py` | `1/6` | display tick 上的 FFT 重算間隔。 |
| `KURT_POLL_INTERVAL_S` | `sensor_reader.py` | 0.4 | 峰度變化偵測的輪詢節奏。 |
| `METRIC_FALLBACK_S` | `sensor_reader.py` | 5.0 | 即使峰度完全不動，也強制發出一批指標。 |
| `METRIC_READ_TIMEOUT_S` | `sensor_reader.py` | 5.0 | FC03 的序列讀取逾時（與原廠 Rust 參考實作一致）。 |
| `ROLLING_WINDOW` | `state.py` | 7 | 滾動多數決緩衝中的預測筆數。 |
| `DISPLAY_REFRESH_EVERY` | `state.py` | 2 | 推論儀表板每次刷新之間的推送次數。 |
| `TRAIN_HOP` | `trainer.py` | 651 | 訓練視窗跨步（75 % 重疊）。 |
| `MIN_WINDOWS` | `trainer.py` | 6 | 允許訓練時每個標籤所需的最少視窗數。 |
