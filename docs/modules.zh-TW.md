# Modules

[English](modules.md) · **繁體中文**

供修改內部程式者參考的各檔案設計註記。系統層級的視角見
[architecture.md](architecture.zh-TW.md)。

---

## app.py

Worker 3——Flask 儀表板，也是串接 W1/W2/W3 的進入點。每個埠執行一個 reader
子行程（各自擁有本地的 `RecordingManager`，主行程透過 `RemoteRecorder` RPC
存取）、一個 `InferenceWorker`，以及一個 `TrainerManager`（它借用 worker 的
interpreter 來計算原型，再熱重載分類頭）。執行：`python app.py` →
`http://<board-ip>/`（綁定 `0.0.0.0:80`）。

- 插入 `site-packages` 路徑讓嵌入式建置可隨附打包的相依套件。空的時候無害。
  在 Linux（fork）子行程會繼承 `sys.path`；在 Windows（spawn）由
  `reader_process_main` 重新套用。
- 佇列上限：一個 `(2604,3)` 視窗的 pickle 約 1–2 ms，相對於 167 ms 的 hop
  間隔，所以上限 16 可讓 4 個 reader 在 `put()` 阻塞前有餘裕；drop-oldest 會
  先觸發。指標佇列上限很小（約 2–5 秒輸出一次）；原始片段佇列上限為 64。
- Reader 開關：當使用者為推論開啟某埠**或**有錄製進行中時，reader 才執行。
  `inference_open` / `recording_ports` / `metrics_open` 分開追蹤；原始串流
  （`active_events`）是前兩者的聯集，FC03（`metrics_events`）追蹤第三者——所以
  `/metrics` 可以輪詢而不必付出 3 Mbps 原始串流的成本。在使用者未為推論開啟的
  埠上錄製，結束後不得留下「開啟」狀態，因此 `begin_recording` /
  `end_recording` 不會動到 `inference_open`。
- `record_state["port"]` 在啟動時設定且永不清除，所以 `/status` 會持續回報
  正確子行程的 `last_finished`。
- 錄製在子行程中到達目標時自動停止；主行程透過狀態輪詢
  （`release_if_finished`）才得知。
- `/api/*` 的 GET 是 `/stream/*` 的非串流版兄弟，供 curl/jq 使用而不需維持連線。

## sensor_reader.py

從輸入暫存器 `0x02`（FC04）串流原始 XYZ，並向推論佇列輸出 `WINDOW_SIZE` 的
滑動視窗（步幅 `HOP_SIZE`）。讀取模式仿照經驗證的 Rust client；佇列使用
drop-oldest。

`reader_process_main` 的參數：
- `port`——例如 `/dev/ttyUSB0`。
- `window_queue`——推送供推論用的 `(port, ndarray)`。
- `req_q` / `resp_q`——RPC 請求進 / 回應出。
- `stop_event`——請求關閉。
- `active_event`——當該埠應讀取原始資料時設定。清除時 reader 休眠且不送任何
  Modbus 流量。每次由非啟用→啟用的轉換都會重寫取樣率暫存器，感測器會將其
  視為 FIFO 重置，所以恢復後的第一個視窗是新鮮的。
- `data_dir`——錄製寫入的位置。

註記：
- `spawn` 給子行程一個全新的 interpreter，所以它會在 import 專案模組前重新
  套用 `site-packages` 路徑。
- 不做 CPU 綁定；核心會把 reader 排程到各核心上。當 reader 有工作時，
  `SCHED_FIFO` 優先權 50 會搶佔一般行程（需要 root / `CAP_SYS_NICE`；否則
  記錄並略過）。
- `READ_TIMEOUT_S = 0.05`：在 3 Mbaud 下，50 ms 遠高於感測器的回應時間，卻能
  把壞封包卡住轉成約 100 ms 的短暫中斷。若正常運作時 `total_fails` 攀升，調高它。
- **FC03 指標暫存器會別名。** 每個指標各自是一次
  `read_holding_registers(base, count=3)`——連續區塊讀取會得到垃圾。位址與
  縮放比仿照 Rust client（`src/types.rs`、`src/modbus.rs`）。
- 指標輪詢：kurtosis（最慢的指標）以 `KURT_POLL_INTERVAL_S` 輪詢；只有在它
  變化時（或首次輪詢，或超過 `METRIC_FALLBACK_S`）才輸出整批，讓整組以約
  2–5 秒的慢節奏一起更新。FC03 讀取慢且不穩定（約 0.5–1 秒）；過早逾時會讓
  慢回應仍在傳輸中，下一個請求便讀到那個過期回應，使串流失步。因此
  `METRIC_READ_TIMEOUT_S = 5.0` 加上一次線路清空，僅在指標讀取前後切入。原始
  串流永遠優先：一旦有原始資料請求，`_MetricsAbort` 立刻中止緩慢的掃描。
- `skewness` 在線路上可能為負——目前以無號讀取；若讀值錯誤，改以 int16 檢視
  （已標記待硬體驗證）。

## fast_modbus.py

pyserial 上的最小 Modbus RTU client——只有 FC03 / FC04 / FC06，不用 pymodbus。
足夠這顆感測器使用，別無其他。

## waveform.py

每埠即時波形聚合器。由快速原始串流餵入（sensor_reader 約 217 取樣點片段、
約 36 Hz/埠 → app.py 抽取執行緒 → `append(port, chunk)`）進入每埠環形緩衝區。
約 30 Hz 的顯示 tick 呼叫 `render_tick()`，它每個 tick 平滑並抽取原始視圖
（平滑捲動），並每隔 `FFT_INTERVAL_S`（約 6 Hz）重算 FFT。`snapshot()` 是給
SSE 處理器用的廉價讀取。

- FFT 使用 pyfftw 並快取計畫（aarch64 上的 NEON SIMD），若無 pyfftw 則退回
  `numpy.fft.rfft`。
- 這裡的 `WINDOW_SIZE` **只是 FFT 視窗**——與模型輸入視窗（`sensor_reader.py`
  中的 `WINDOW_SIZE`）獨立。不要把它們合而為一。
- Savitzky-Golay 係數（`_savgol_coeffs`）不依賴 scipy 推導：建構 Vandermonde
  矩陣 `A[i,k] = i**k`（`i` 從 `-half` 到 `+half`），取 `pinv(A)` 的第 0 列。
  等同於 `scipy.signal.savgol_coeffs(W, P)`。`_savgol` 以邊緣複製方式填補，
  避免兩端被拉向零。
- `WaveformBus` 與 `state.SnapshotBus` 分開，讓預測與波形的 SSE 不會互相誤觸。

## inference.py

每個 `(WINDOW, 3)` 視窗 → 骨幹網路 → 128 維嵌入 → `head.predict()` →
`(class_id, similarity, class_name)`。後端只有 NPU（見
[architecture → NPU 後端](architecture.zh-TW.md#npu-backend)）。若執行環境、
模型或 delegate 缺少，會退回 `stub`（class 0，信心 1.0）。無 CPU 後備。

- `embed()` 是公開的，讓 trainer.py 可借用 interpreter 計算原型；它在 TFLite
  呼叫期間持有 `_invoke_lock`。
- `reload_head()` 原子性地從磁碟換入分類頭，讓重新訓練能顯現新標籤而無需重啟。
- 量化輸入時**先 clip 再 cast**：
  `np.clip(np.round(x/scale + zp), -128, 127).astype(np.int8)`。單純的
  `.astype` 會把超出範圍的值繞回（200 → −56）。
- 骨幹網路已以 `Dense(N, softmax)` 結尾——不要再套一次 softmax。
- 三個 `# PORT` 常數用來選擇 NPU 驅動堆疊（見
  [architecture → 同一顆 NPU，兩種驅動](architecture.zh-TW.md#same-npu-two-drivers)）。

## classifier.py

CPU 端的分類頭。骨幹網路把一個視窗轉成 128 維嵌入；分類頭以餘弦相似度對比
各類別原型來分類。純 numpy。

以 JSON 保存在骨幹網路旁（`classifier_head.json`）：

```json
{ "labels": ["circle","shake","steady"], "prototypes": [[...128 floats...], ...] }
```

每個標籤也會分到一個顏色槽 `0..PALETTE_SIZE-1`，每次訓練重新指派，所以重訓練
在視覺上刻意變得明顯。`save()` 先寫入暫存檔再改名，讓部分寫入不會破壞 worker
即將重載的分類頭。

## trainer.py

對每個選定標籤：把每個 `(WINDOW, 3)` 視窗跑過骨幹網路，平均成一個 128 維
原型，全部寫入 `classifier_head.json`，並熱重載當前的分類頭。無 TF、無梯度、
數秒完成。單例 + 背景執行緒 + 狀態的模式仿照 recorder.py。

- `TRAIN_HOP = WINDOW_SIZE // 4`（75% 重疊）→ 每個動作週期 4 個相位對齊，
  約為非重疊切片的 4 倍樣本量，原型變異更低。此處 50% 重疊只會命中 2 個相位
  位置（`2604/1302 = 2`）——更差。
- 顏色指派使用 Python 的 RNG（非 numpy 的），所以外部的 numpy 種子無法讓
  洗牌穩定。

## recorder.py

在即時推論的同時把原始 XYZ 錄成二進位。兩階段佇列：`feed()`（來自 W1）在
短暫鎖下把資料附加到第一階段的 list，不做 I/O；一旦累積 `FLUSH_SIZE` 個
取樣點，該片段就進入由專屬執行緒抽取的寫入佇列，約每秒 flush 一次。
`ndarray.tofile` 在寫入期間釋放 GIL——這正是採用二進位格式的原因；純 Python
的 CSV 格式化會在整個片段期間持有 GIL，餓死 reader 的 Modbus 輪詢。

- 二進位格式：原始 float32 小端序、3 通道（x, y, z）交錯、無標頭、`.bin`。
  取樣點數 = `file_size // 12`。附加即 `open("ab")`。
- 舊的 `.csv` 檔仍可被 trainer.py 讀取；新錄製一律寫成 `.bin`。
- `delete_label` **不會**重新淨化名稱（那會弄壞例如開頭底線）——它比對
  `list_existing_labels` 回傳的內容，拒絕路徑分隔符，並確認解析後的路徑就
  直接位於 `data_dir` 之下。
- `cancel()` 把剩餘的第一階段緩衝交給寫入器，然後最多阻塞
  `WRITER_JOIN_TIMEOUT_S`，讓 `last_finished` 回報真實的磁碟上數量。
- **已知未解 bug：** recorder 會在 `data/*.bin` 中產生週期性的約 9400 取樣點
  接縫。Reader 本身健康（`data_len` 穩定、`total_fails=0`）。懷疑是
  `_writer_queue` / 狀態輪詢的鎖競爭。

## state.py

三個 worker 共用的基礎元件。

- `load_class_labels()` / `CLASS_LABELS` 只是啟動時的提示（樣板、顏色對應）。
  每筆預測的權威類別名稱來自呼叫當下的即時分類頭，所以這裡的值過期也無妨。
- `SnapshotBus` 在每次鎖存時喚醒任意數量的 SSE client。每次鎖存都會遞增一個
  單調計數器並通知所有等待者；SSE 產生器追蹤自己上次看到的計數器，所以在多
  client 時不會漏掉通知。
- `RollingPredictions` 保存最近 N 個結果，以及每 `DISPLAY_REFRESH_EVERY` 次
  推送才刷新一次的可顯示快照。多數類別以最近度打破平手（反向走訪 deque）。
  `on_latch` 回呼在鎖**之外**觸發（`bus.bump()` 會取自己的鎖）。切換啟用中的
  感測器時會呼叫 `clear()`，以免過期預測影響新的多數決。

## bin2csv.py

把 recorder 的 `.bin`（原始 float32 XYZ、3 通道交錯、無標頭）轉成帶 `x,y,z`
標頭的 `.csv`。無時間戳——`.bin` 格式從未儲存它們。

```bash
python bin2csv.py data/steady.bin data/shake.bin   # 指定檔案
python bin2csv.py data/*.bin                        # shell glob
python bin2csv.py data                              # 目錄中的每個 .bin
```
