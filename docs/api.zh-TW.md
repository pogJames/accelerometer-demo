# API

[English](api.md) · **繁體中文**

用於在本示範之上進行延伸開發的 HTTP 介面：儀表板頁面、SSE 串流與控制端點。

所有請求與回應的內容皆為 JSON。錯誤會回傳 `{"error": "..."}` 並附上 4xx/5xx
狀態碼。埠使用原始的 `/dev/ttyUSBx` 字串。

在頁面（或 `/api/active_port` 呼叫）啟用某個埠之前，該埠不會串流任何資料。

---

## 頁面

| 網址 | 顯示內容 |
|---|---|
| `/` | 即時波形——原始捲動 + FFT 頻譜，每個感測器一個 |
| `/inference` | 即時分類卡片（對最近 7 個視窗做多數決） |
| `/metrics` | 每個感測器的指標表（重力 / 速度 / 溫度） |
| `/record` | 錄製原始 XYZ 至 `data/<label>.bin` |
| `/train` | 訓練分類頭（2 個以上標籤，數秒完成） |
| `/settings` | 每個埠的友善名稱 |

各頁細節：[見下方頁面細節](#page-details)。

---

## SSE 串流

長連線的 `text/event-stream`。每個 `data:` 幀是一份 JSON 快照，在有變化時
推送。閒置時送出註解形式的心跳。

| 端點 | 推送時機 | 幀內容（欄位） |
|---|---|---|
| `GET /stream/inference` | 新的多數決鎖存 | `ports{<port>:{majority_class_name, majority_count, window_count, recent[], display_seq, latest_ts}}, class_labels[], label_colors{}, active_ports[], inference_mode, now` |
| `GET /stream/waveform` | 30 Hz 顯示 tick | `ports{<port>:{raw, fft, raw_seq, fft_seq, ts}}, raw_axis[], freq_axis_hz[], sample_rate, window_size, fft_bins, ...` |
| `GET /stream/metrics` | FC03 批次（約 2–5 秒） | `ports{<port>:{temperature, gravity{rms,peak,crest,skewness,kurtosis,primary_freq}, velocity{rms,peak,crest,primary_freq}, ts}}, open_ports[], now` |

從終端機追蹤某個串流：

```bash
curl -N http://<board-ip>/stream/inference
```

---

## 控制與設定（POST）

| 端點 | 主體 | 作用 |
|---|---|---|
| `POST /api/active_port` | `{port, active}` | 開啟／關閉某埠的**原始波形 + 推論**串流 |
| `POST /api/metrics_active` | `{port, active}` | 開啟／關閉**指標**串流（獨立於原始串流） |
| `POST /api/waveform_config` | `{fft_max_hz?, raw_samples?}` | 設定即時波形/FFT 參數 |
| `POST /api/port_alias` | `{port, alias}` | 設定某埠的友善名稱 |

---

## 錄製

| 端點 | 主體 | 回傳 |
|---|---|---|
| `GET /api/recordings` | — | `{labels:[{name, samples, windows, eligible}], ports[], sample_rate, min_samples, min_windows, window_size, data_dir}` |
| `GET /api/record/status` | — | `{session}` 或 `null`（約 1 Hz 輪詢） |
| `POST /api/record/start` | `{name, target_samples, port, mode:"append"\|"overwrite"}` | `{session}` |
| `POST /api/record/cancel` | — | `{session}`（保留已寫入的取樣點） |
| `POST /api/recordings/delete` | `{name}` | `{deleted}`——若該標籤正在錄製則回 `409` |

`session` 為 `{status, name, port, progress, elapsed_s, samples_written, target_samples, file_path}`。

---

## 訓練

| 端點 | 主體 | 回傳 |
|---|---|---|
| `GET /api/train/status` | — | `{session, head_labels[], head_colors{}, backbone, backbone_mode}` |
| `POST /api/train/start` | `{labels:[...]}`（2 個以上符合資格） | `{session}` |
| `POST /api/train/cancel` | — | `{session}` |

---

## 一次性 JSON

非串流的 GET，回傳與對應 `/stream/*` 相同的內容，供 `curl`/`jq` 除錯。無論
是否有埠處於啟用狀態，都會反映目前快取的狀態。

| 端點 | 回傳 |
|---|---|
| `GET /api/inference` | 目前的 `/stream/inference` 內容 |
| `GET /api/metrics` | 目前的 `/stream/metrics` 內容 |
| `GET /api/waveform` | 目前的 `/stream/waveform` 內容 |

```bash
curl -s http://<board-ip>/api/inference | jq
```

---

<a id="page-details"></a>
## 頁面細節

### `/` — 即時波形

<p float="left">
    <img src="waveform.png" width="50%">
</p>

- **原始模式**——捲動的時域曲線，X/Y/Z 疊加。「最近 N 個取樣點」最多 1 秒。
  更新率 10/30/60 fps。
- **FFT 模式**——幅值頻譜，範圍以 Hz 設定（上限 800 Hz）。
- 每軸統計標籤（原始模式為 min..max，FFT 模式為峰值頻率）。點擊標籤可
  隱藏/顯示該軸。滑鼠停在圖上可看該點取樣值。

### `/inference` — 即時分類

<p float="left">
    <img src="inference.png" width="50%">
</p>

每個感測器一張卡片。顯示最近 7 個結果的多數決分類、最新信心值，以及一條
彩色點條。由 SSE 驅動，無輪詢。

### `/metrics` — 指標表

<p float="left">
    <img src="metrics.png" width="50%">
</p>

每個感測器一張卡片：溫度、一個重力表（RMS / Peak / Crest / Skewness /
Kurtosis × XYZ + 主頻）與一個速度表（RMS / Peak / Crest × XYZ + 主頻）。所有
欄位以約 2–5 秒的慢節奏一起更新。

啟用 `/metrics` 只會開始 FC03 輪詢，不會啟動原始串流。原始串流（來自 `/` 或
`/inference`）在同一個埠上永遠優先於指標輪詢。

### `/record` — 錄製資料

<p float="left">
    <img src="record.png" width="50%">
</p>

將原始 XYZ 錄製至 `data/<name>.bin`（float32 小端序、3 通道交錯）。有 Append
與 Overwrite 模式。進度透過 SSE 串流。表格每一列都有刪除按鈕。

將 `.bin` 轉為 CSV（`x,y,z`，無時間戳）：

```bash
python bin2csv.py data/<name>.bin      # 也接受 glob 或目錄
```

### `/train` — 訓練分類頭

<p float="left">
    <img src="train.png" width="50%">
</p>

「訓練」是為每個標籤計算一個平均嵌入向量。無梯度，數秒完成。選 2 個以上
符合資格的標籤 → **Train model** → 寫入 `classifier_head.json`，推論 worker
會熱重載，無需重啟。每次訓練會重新指派顏色配置。

### `/settings` — 埠名稱

每個偵測到的埠一列。輸入友善名稱（例如「Front-left motor」），會立即存到
`port_aliases.json` 並顯示在每個頁面。原始的 `/dev/ttyUSBx` 路徑仍是內部 ID。
