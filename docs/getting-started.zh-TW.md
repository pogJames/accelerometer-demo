# Getting started

[English](getting-started.md) · **繁體中文**

如何在 **Matrix800** 閘道器上執行本示範。

另見：[architecture.md](architecture.zh-TW.md) 了解運作方式，
[api.md](api.zh-TW.md) 進行延伸開發。

---

## 1. 你需要什麼

- 一台已安裝 Linux 映像、連上你網路的 **Matrix800** 閘道器。
- 一個以上接在 RS485 上的**加速度計**。每個使用一個 USB 埠
  （`/dev/ttyUSB0`、`/dev/ttyUSB1`、...）。

你也可以在一般 PC 上執行，不需感測器也不需 NPU。它會以 *stub* 模式啟動
（見[步驟 5](#5-first-run)），讓你瀏覽各頁面。真實資料則需要實機板。

---

## 2. 安裝

```bash
git clone https://github.com/pogJames/accelerometer-demo
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

<a id="3-choose-your-npu-driver"></a>
## 3. 選擇你的 NPU 驅動程式

> **在實機上執行前請先讀這個步驟。** Matrix800 只有一顆 NPU，但它的 Linux
> 映像會使用**兩種驅動程式其中之一**。

檢查你的板子目前使用哪個驅動程式：

```bash
ls /dev/ethos*    # 存在 → NXP 驅動
ls /dev/accel     # 存在 → Mesa 驅動
```

接著在 `inference.py` 中設定三個值以對應：

| | **NXP 驅動**（預設） | **Mesa 驅動** |
|---|---|---|
| 執行環境 import（`_try_load_interpreter`） | `from tflite_runtime.interpreter import ...` | `from ai_edge_litert.interpreter import ...` |
| `DELEGATE_PATH` | `/usr/lib/libethosu_delegate.so` | `/usr/local/lib/aarch64-linux-gnu/libteflon.so` |
| `NPU_MODEL_PATH` | `models/vibration_backbone_int8_vela.tflite` | `models/vibration_backbone_int8.tflite` |

程式碼預設為 **NXP 驅動**。若你的板子使用 **Mesa 驅動**，請修改那三行。
沒有自動偵測。

兩個模型檔都已在 `models/` 中，你只需選擇載入哪一個。
關於為何設錯不會報錯，請見
[architecture.md → NPU 後端](architecture.zh-TW.md#npu-backend)。

---

<a id="4-connect-the-sensors"></a>
## 4. 連接感測器

列出你的埠：

```bash
ls /dev/ttyUSB*
```

設定 app 讀取哪些埠。編輯 `sensor_reader.py` 中的 `ALLOWED_PORTS`。預設為
一個埠。把你所有的感測器都列出來：

```python
ALLOWED_PORTS = ['/dev/ttyUSB0', '/dev/ttyUSB1', '/dev/ttyUSB2', '/dev/ttyUSB3']
```

**序列埠權限。** 讀取 `/dev/ttyUSB*` 需要存取權 -> 以 `root` 執行

---

<a id="5-first-run"></a>
## 5. 首次執行

```bash
python app.py
```

伺服器監聽 `0.0.0.0:80`。**埠 80 需要 root。** -> 以 `root` 執行

用瀏覽器開啟板子：`http://<board-ip>/`。

檢查側邊欄的**推論模式**：

- **`npu`**——驅動、delegate、模型都已載入，運作正常。
- **`stub`**——找不到執行環境、delegate 或模型檔。App 會提供一個固定的佔位
  分類，讓頁面仍可顯示。在 PC 上這是正常的。在板子上則代表步驟 3 尚未完成。
  啟動記錄檔會印出缺少哪個檔案。

本版本**沒有 CPU 後備**。模式只有 `npu` 或 `stub`。

---

## 6. 試用

1. 開啟 `/`。你應該看到每個感測器的即時波形。平直的線代表感測器沒有送出
   資料，請檢查接線與 `ALLOWED_PORTS`。
2. 錄兩種動作：`/record` → 設一個標籤（例如 `steady`），錄幾秒。再錄第二個
   標籤。
3. 訓練：`/train` → 選 2 個以上標籤 → **Train model**（數秒完成）。
4. 開啟 `/inference`，現在會顯示即時分類。

完整頁面與 API 參考：[api.md](api.zh-TW.md)。

---

## 7. 以服務方式執行 *(選用)*

若要讓儀表板在重新開機後持續執行，加入一個 `systemd` 服務單元。範例
`/etc/systemd/system/npu-python.service`：

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
journalctl -u npu-python -f          # 查看記錄
```

---

## 疑難排解

| 問題 | 原因與解法 |
|---|---|
| 板子上側邊欄顯示 **`stub`** | 執行環境、delegate 或模型缺少或不匹配。讀啟動記錄檔——它會指出缺少的檔案。重新確認[步驟 3](#3-choose-your-npu-driver)。 |
| 結果從不變化或看似隨機 | 通常是驅動或模型錯誤。單獨測試該 `.tflite`：`python smoketest.py`（輸入全 0、全 1、隨機值——輸出必須不同）。見 [architecture.md → NPU 後端](architecture.zh-TW.md#npu-backend)。 |
| `/dev/ttyUSB*` 出現 **`Permission denied`** | 非 root 且不在 `dialout` 群組——見[步驟 4](#4-connect-the-sensors)。 |
| 埠 80 出現 **`Permission denied`** | 埠 80 需要 root——用 `sudo` 或 `setcap`，見[步驟 5](#5-first-run)。 |
| 找不到埠／波形平直 | `ALLOWED_PORTS` 與實際的埠不符。檢查 `ls /dev/ttyUSB*` 與接線。 |
| 記錄檔中 `total_fails` 上升 | 感測器回應慢。調高 `sensor_reader.py` 中的 `READ_TIMEOUT_S`。見 [modules.md → sensor_reader](modules.zh-TW.md#sensor_readerpy)。 |
