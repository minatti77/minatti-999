# STELLA v1.33 — スーパーエージェント向け使い方ガイド

## 概要

| ファイル | 役割 |
|---|---|
| `GINANDTONIC.py` | STELLA 本体（OCR → 予想 → 分析）|
| `stella_agent.py` | Python API ラッパー（エージェントから呼ぶクラス）|
| `stella_tool.py` | OpenAI tools / LangChain ツール定義 |

---

## 1. 最小構成での予想（Python から直接呼ぶ）

```python
from stella_agent import StellaAgent

agent = StellaAgent()

result = agent.predict(
    rating_img="path/to/rating.png",        # レイティング表画像（必須）
    speed_index_img="path/to/speed.png",    # スピード指数表画像（必須）
)

if result["success"]:
    print(result["markdown"])   # 予想 Markdown が得られる
else:
    print("エラー:", result["error"])
```

---

## 2. 全画像を渡す場合

```python
result = agent.predict(
    rating_img="rating.png",
    speed_index_img="speed.png",
    meta_img="meta.png",                    # レース情報
    factor_img="factor.png",               # ファクター表
    ai_time_img="ai_time.png",             # AI展開予測
    pred_times_imgs=["pred1.png", "pred2.png"],  # 予測タイム表（複数可）
    track_imgs=["track.png"],              # 馬場状態
    training_keibabook_imgs=["train.png"], # 調教
    comments_csv="comments.csv",           # 厩舎コメント CSV
    out_md="output/prediction.md",         # Markdown 保存先
    params_json="params.json",             # パラメータ上書き
    ocr_cache_dir="cache/ocr",             # OCR キャッシュ
    result_dir="output/results",           # 予想結果 JSON 保存先
    race_id="2025_阪神_11R",               # レース ID
    fast=True,                             # 高速モード
)
```

---

## 3. OpenAI スーパーエージェントでの使い方

### 3-1. セットアップ

```python
import openai, json
from stella_tool import STELLA_TOOLS, call_stella_tool

client = openai.OpenAI()
```

### 3-2. Responses API（推奨）

```python
messages = [{"role": "user", "content": "rating.png と speed.png でレースを予想して"}]

response = client.responses.create(
    model="gpt-4o",
    input=messages,
    tools=STELLA_TOOLS,
)

# ツール呼び出しを処理するループ
while True:
    tool_calls = [item for item in response.output if item.type == "function_call"]
    if not tool_calls:
        # テキスト応答で終了
        for item in response.output:
            if hasattr(item, "content"):
                for c in item.content:
                    if c.type == "output_text":
                        print(c.text)
        break

    # ツールを実行して結果を返す
    tool_results = []
    for tc in tool_calls:
        result = call_stella_tool(tc.name, json.loads(tc.arguments))
        tool_results.append({
            "type": "function_call_output",
            "call_id": tc.call_id,
            "output": result,
        })

    # 次のターンへ
    messages = list(response.output) + tool_results
    response = client.responses.create(
        model="gpt-4o",
        input=messages,
        tools=STELLA_TOOLS,
    )
```

### 3-3. Chat Completions API

```python
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "rating.png で予想して"}],
    tools=STELLA_TOOLS,
    tool_choice="auto",
)

# ツール呼び出しがあれば実行
if response.choices[0].message.tool_calls:
    for tc in response.choices[0].message.tool_calls:
        result = call_stella_tool(tc.function.name, json.loads(tc.function.arguments))
        print(result)
```

---

## 4. 利用可能なツール一覧

| ツール名 | 説明 | 必須引数 |
|---|---|---|
| `stella_predict` | レース予想を実行 | `rating_img`, `speed_index_img` |
| `stella_analyze` | 事後分析レポート生成 (R41) | `result_dir` |
| `stella_tune_confidence` | confidence 自動チューニング (R42) | `result_dir` |
| `stella_wide_roi` | ワイド ROI 最適化 (R43) | `result_dir` |
| `stella_dashboard` | 総合ダッシュボード (R45) | `result_dir` |
| `stella_full_analysis` | 全分析一括実行 (R48) | `result_dir` |
| `stella_record_result` | レース結果を記録 (R39) | `race_id`, `result_dir`, `win_num`, `place_nums` |

---

## 5. 典型的なワークフロー

```
① 競馬アプリのスクリーンショットを撮影
        ↓
② stella_predict(rating_img, speed_index_img, ...) 実行
        ↓
③ 予想 Markdown を確認して馬券購入
        ↓
④ レース終了後 → stella_record_result() で結果を記録
        ↓
⑤ 数レース蓄積したら stella_full_analysis() で成績確認・チューニング
```

---

## 6. コマンドラインから直接実行（確認用）

```bash
# 予想
python3 GINANDTONIC.py \
  --rating_img rating.png \
  --speed_index_img speed.png \
  --out_md prediction.md

# 事後分析
python3 GINANDTONIC.py \
  --analyze_results output/results \
  --analyze_out_md analysis.md

# 全分析一括
python3 GINANDTONIC.py \
  --full_analysis output/results \
  --full_analysis_out_md full_report.md

# stella_agent.py 単体テスト
python3 stella_agent.py \
  --rating_img rating.png \
  --speed_index_img speed.png

# ツール定義一覧確認
python3 stella_tool.py
```

---

## 7. 返り値の構造

### `predict()` の返り値

```python
{
    "success":  True,              # 成功/失敗
    "markdown": "## 予想...",      # 予想 Markdown 全文
    "out_md":   "prediction.md",  # 出力ファイルパス
    "stdout":   "...",
    "stderr":   "...",
    "error":    None,              # エラー時は文字列
}
```

### `analyze()` / `dashboard()` 等の返り値

```python
{
    "success":  True,
    "markdown": "## 分析レポート...",
    "out_md":   "report.md",
    "stdout":   "...",
    "stderr":   "...",
    "error":    None,
}
```

### `tune_confidence()` の返り値

```python
{
    "success": True,
    "tuning":  { "high_threshold": 0.38, ... },  # チューニング結果辞書
    "out_json": "tuning_result.json",
    ...
}
```

---

## 8. 注意事項

- **Tesseract OCR** が必要です: `sudo apt-get install tesseract-ocr tesseract-ocr-jpn`
- **依存ライブラリ**: `pip install lightgbm opencv-python-headless pytesseract pandas numpy`
- 画像は **スマートフォンのスクリーンショット**（Chrome/Firefox）に対応しています
- `fast=True` を指定するとモンテカルロのサンプル数を減らして高速化できます
- `ocr_cache_dir` を指定すると同じ画像への再 OCR を省略できます（高速化）
