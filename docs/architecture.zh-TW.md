# Architecture

[English](architecture.md) · **繁體中文**

系統如何組成，以及為何這樣設計。各檔案的註記見 [modules.md](modules.zh-TW.md)；
HTTP 介面見 [api.md](api.zh-TW.md)。

---

## 三個 worker

整條管線是三個 worker，以佇列解耦。

```
W1  sensor_reader.py             one subprocess per port
     FC04 register 0x02 → decode → /8192 → G
       ├─→ recorder.feed()               (local, no IPC)
       ├─→ 2604-sample window  ─┐  hop 1302, ~6 Hz/port  ──→ window_queue ─┐
       └─→ 217-sample chunk  ─┐ │  ~36 Hz/port            ──→ raw_queue ──┐ │
     FC03 metric batch ─┐     │ │  (kurtosis change, ~2–5 s) → metrics_queue │ │
                        │     │ │                                        │  │ │
    ──── main process ──┴─────┴─┴────────────────────────────────────────┴──┴─┘
        InferenceWorker      raw drain →           metrics drain →
        (classify)           WaveformAggregator    per-port latest
          ↓                    ↓ 30 Hz tick          ↓
        RollingPredictions   WaveformBus           MetricsBus
          ↓
        SnapshotBus

W3  app.py — Flask, wires it all, serves pages + SSE
```

- **W1 — `sensor_reader.py`。** 每個埠一個子行程。讀取 Modbus，輸出供推論用的
  `(WINDOW_SIZE, 3)` 視窗，以及供波形用的小型原始片段。它在**自己的行程**中執行，
  因為序列輪詢迴圈受 GIL 綁定：若與 Flask、推論或 GC 停頓共用 GIL，會使感測器的
  硬體 FIFO 溢位並掉取樣點。此子行程也擁有 recorder；原始取樣點永遠不跨越行程
  邊界。控制（start / cancel / status）透過一對 `mp.Queue` 以 RPC 進行。
- **W2 — `inference.py`。** 單一執行緒。執行凍結的骨幹網路加上餘弦相似度分類頭。
  只做分類——波形有自己的佇列與抽取執行緒，所以 W2 永不與它競爭。
- **W3 — `app.py`。** 把一切接起來，並透過 SSE 提供儀表板。

### 雙節奏

reader 以兩種速率輸出，完全解耦：

- `window_queue`——供模型用的完整 2604 取樣點視窗，約 6 Hz。
- `raw_queue`——供波形用的小型 217 取樣點新鮮片段，約 36 Hz。

推論與波形彼此永不阻塞。

### 波形環形緩衝區

`WaveformAggregator` 為每個埠保存約 2 秒的環形緩衝區。30 Hz 的顯示 tick 每次
讀取最後 N 個取樣點（原始波形平滑捲動），並以約 6 Hz 從最後的 FFT 視窗取樣點
重算 FFT。

---

<a id="npu-backend"></a>
## NPU 後端

兩種模式，顯示在側邊欄：

- **`npu`**——驅動、delegate、int8 骨幹網路皆已載入。唯一真正的後端。
- **`stub`**——執行環境、模型或 delegate 缺少，或 delegate 載入失敗。回傳一個
  固定分類讓儀表板仍能顯示。開發用機器一律落在這裡。

本版本**沒有 CPU 後備**。

### 無聲垃圾陷阱

一個 float32 的 `.tflite`，或對板子驅動而言錯誤的 delegate/模型，**不會報錯也
不會警告**。它會載入、執行，並對每一筆輸入回傳位元完全相同的輸出。兩條規則
可避免它：

1. 讓執行環境、delegate、模型對應板子的 NPU 驅動。見
   [getting-started → 步驟 3](getting-started.zh-TW.md#3-choose-your-npu-driver)。
2. 正確地餵給模型 int8。量化輸入時要**先 clip 再 cast**：
   `np.clip(np.round(x/scale + zp), -128, 127).astype(np.int8)`。單純的
   `.astype` 會把超出範圍的值繞回（200 → −56），餵給 NPU 垃圾資料。

骨幹網路已經以 `Dense(N, softmax)` 結尾——輸出就是機率向量。分類流程中**不要**
再套一次 softmax。

<a id="same-npu-two-drivers"></a>
### 同一顆 NPU，兩種驅動

Matrix800 只有一顆 NPU。它的 Linux 映像使用兩種驅動其中之一，而每種都需要
不同的執行環境、delegate 與模型檔。這三個旋鈕都在 `inference.py` 中，標記為
`# PORT`。完整表格見
[getting-started → 步驟 3](getting-started.zh-TW.md#3-choose-your-npu-driver)。

---

## 檔案

| 檔案 | 角色 |
|---|---|
| `app.py` | Flask app；串接 reader 子行程 + InferenceWorker + 抽取執行緒 + 顯示 tick；提供所有 SSE 端點 |
| `sensor_reader.py` | 每個埠的 reader 子行程：FC04 雙節奏原始串流 + FC03 指標批次 |
| `fast_modbus.py` | 最小 Modbus RTU client（pyserial 上的 FC03 / FC04 / FC06，不用 pymodbus） |
| `waveform.py` | `WaveformAggregator`——每埠環形緩衝區，30 Hz render tick |
| `inference.py` | `InferenceWorker`——視窗 → 骨幹嵌入 → 分類頭分類 → 推送 |
| `classifier.py` | `ClassifierHead`——餘弦相似度，JSON 讀寫 |
| `trainer.py` | `TrainerManager`——原型計算，背景執行緒 |
| `recorder.py` | `RecordingManager` + `delete_label()`——兩階段佇列，二進位寫入執行緒 |
| `state.py` | `RollingPredictions`、`SnapshotBus`、`LatestSlot` |
| `bin2csv.py` | 將 `.bin` 錄製轉為 CSV |
| `port_aliases.json` | 友善名稱覆寫，由 `/settings` 寫入 |
| `models/vibration_backbone_int8_vela.tflite` | Vela 編譯的 int8 骨幹網路——NXP 驅動（Ethos-U） |
| `models/vibration_backbone_int8.tflite` | 非 Vela 的 int8 骨幹網路——Mesa 驅動 |
| `classifier_head.json` | 訓練好的分類頭——由 trainer 寫入，inference 讀取 |
| `data/` | 錄製檔——`<label>.bin`，float32 原始 XYZ |

各檔案設計註記：[modules.md](modules.zh-TW.md)。

---

## 調校旋鈕

| 常數 | 檔案 | 預設 | 作用 |
|---|---|---|---|
| `WINDOW_SIZE` | `sensor_reader.py` | 2604 | **模型**輸入視窗（取樣點）。對應骨幹網路——未重訓練前勿改。 |
| `HOP_SIZE` | `sensor_reader.py` | 1302 | 推論的視窗步幅（約每秒 6 次輸出）。 |
| `RAW_CHUNK_SIZE` | `sensor_reader.py` | 217 | 波形用的新鮮取樣片段（約每秒 36 次輸出）。 |
| `WINDOW_SIZE` | `waveform.py` | 3906 | **FFT** 視窗（取樣點）。與模型視窗獨立——改動可得到更細的頻率解析度。 |
| `RING_CAPACITY` | `waveform.py` | `SAMPLE_RATE*2` | 每埠環形緩衝區（約 2 秒）。必須 ≥ 最大的 FFT/原始視窗。 |
| `FFT_INTERVAL_S` | `waveform.py` | `1/4` | 顯示 tick 中的 FFT 重算間隔。 |
| `KURT_POLL_INTERVAL_S` | `sensor_reader.py` | 0.4 | Kurtosis 變化偵測的輪詢節奏。 |
| `METRIC_FALLBACK_S` | `sensor_reader.py` | 5.0 | 若 kurtosis 靜止不動，強制輸出一次指標。 |
| `METRIC_READ_TIMEOUT_S` | `sensor_reader.py` | 5.0 | FC03 的序列讀取逾時。 |
| `ROLLING_WINDOW` | `state.py` | 7 | 多數決緩衝區中的預測數量。 |
| `DISPLAY_REFRESH_EVERY` | `state.py` | 2 | 推論儀表板兩次刷新之間的推送數。 |
| `TRAIN_HOP` | `trainer.py` | 651 | 訓練視窗步幅（75% 重疊）。 |
| `MIN_WINDOWS` | `trainer.py` | 6 | 每個標籤可訓練所需的最少視窗數。 |

> `WINDOW_SIZE` **同時**存在於 `sensor_reader.py`（模型輸入）與 `waveform.py`
> （FFT 視窗）。兩者獨立——不要把它們合而為一。
