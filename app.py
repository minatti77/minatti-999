#!/usr/bin/env python3
"""
app.py  —  STELLA v1.33 スーパーエージェント Web UI
「競馬予想」と入力するだけで予想が始まるチャット形式インターフェース
"""
import os
import sys
import uuid
import subprocess
import threading
import time
import json
from pathlib import Path
from flask import (
    Flask, request, jsonify, render_template,
    send_from_directory, Response, stream_with_context
)
from werkzeug.utils import secure_filename

# ── パス設定 ──────────────────────────────────────
BASE_DIR    = Path(__file__).parent.resolve()
UPLOAD_DIR  = BASE_DIR / "uploads"
OUTPUT_DIR  = BASE_DIR / "output"
GINANDTONIC = BASE_DIR / "GINANDTONIC.py"

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100MB

# ── 実行中セッション管理 ─────────────────────────
_jobs: dict = {}   # job_id -> {"status", "log", "markdown", "error", "start_time"}
_lock = threading.Lock()

# ── ユーティリティ ────────────────────────────────
def allowed(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXT

def save_upload(f) -> Path:
    uid  = uuid.uuid4().hex
    name = secure_filename(f.filename)
    ext  = Path(name).suffix.lower() or ".jpg"
    dest = UPLOAD_DIR / f"{uid}{ext}"
    f.save(dest)
    return dest

def run_stella(job_id: str, rating_img: str, speed_index_img: str,
               meta_img: str = None, factor_img: str = None,
               ai_time_img: str = None,
               pred_times_imgs: list = None,
               track_imgs: list = None,
               training_imgs: list = None,
               comments_csv: str = None,
               fast: bool = False):
    """バックグラウンドで GINANDTONIC.py を実行する。"""
    out_md = str(OUTPUT_DIR / f"{job_id}.md")

    cmd = [
        sys.executable, str(GINANDTONIC),
        "--rating_img",      rating_img,
        "--speed_index_img", speed_index_img,
        "--out_md",          out_md,
        "--lang",            "jpn",
    ]
    if meta_img:        cmd += ["--meta_img",      meta_img]
    if factor_img:      cmd += ["--factor_img",    factor_img]
    if ai_time_img:     cmd += ["--ai_time_img",   ai_time_img]
    if pred_times_imgs: cmd += ["--pred_times_imgs"] + pred_times_imgs
    if track_imgs:      cmd += ["--track_imgs"]    + track_imgs
    if training_imgs:   cmd += ["--training_keibabook_imgs"] + training_imgs
    if comments_csv:    cmd += ["--comments_csv",  comments_csv]
    if fast:            cmd.append("--fast")

    with _lock:
        _jobs[job_id] = {
            "status": "running",
            "log": "STELLA 起動中...\n",
            "markdown": "",
            "error": "",
            "start_time": time.time()
        }

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(BASE_DIR),
            bufsize=1
        )
        for line in proc.stdout:
            with _lock:
                _jobs[job_id]["log"] += line
        proc.wait(timeout=600)

        md_path = Path(out_md)
        if md_path.exists():
            md_text = md_path.read_text(encoding="utf-8")
        else:
            md_text = ""

        elapsed = time.time() - _jobs[job_id]["start_time"]

        with _lock:
            if proc.returncode == 0 and md_text:
                _jobs[job_id]["status"]   = "done"
                _jobs[job_id]["markdown"] = md_text
                _jobs[job_id]["elapsed"]  = round(elapsed, 1)
            elif proc.returncode == 0 and not md_text:
                # 出力ファイルなし → ログから結果を取る
                log = _jobs[job_id]["log"]
                _jobs[job_id]["status"]   = "done"
                _jobs[job_id]["markdown"] = f"## STELLA 実行ログ\n\n```\n{log}\n```"
                _jobs[job_id]["elapsed"]  = round(elapsed, 1)
            else:
                _jobs[job_id]["status"]   = "error"
                _jobs[job_id]["error"]    = (
                    f"終了コード {proc.returncode}\n\n"
                    f"ログ (末尾500文字):\n{_jobs[job_id]['log'][-500:]}"
                )

    except subprocess.TimeoutExpired:
        try:
            proc.kill()
        except Exception:
            pass
        with _lock:
            _jobs[job_id]["status"] = "error"
            _jobs[job_id]["error"]  = "タイムアウト（600秒）\nOCR処理に時間がかかりすぎています。"
    except Exception as e:
        with _lock:
            _jobs[job_id]["status"] = "error"
            _jobs[job_id]["error"]  = str(e)


# ── ルーティング ──────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    """
    テキストメッセージを受け取る。
    「競馬予想」を含むとき → 予想モードへ誘導するレスポンスを返す。
    それ以外 → 汎用ガイドを返す。
    """
    data = request.get_json(force=True)
    msg  = str(data.get("message", "")).strip()

    if not msg:
        return jsonify({"reply": "メッセージを入力してください。"})

    keiba_keywords = [
        "競馬予想", "よそう", "予想して", "予想お願い",
        "stella", "STELLA", "馬券", "レース予想",
        "競馬", "予想", "レース", "馬", "keiba"
    ]

    if any(k.lower() in msg.lower() for k in keiba_keywords):
        return jsonify({
            "reply": (
                "🏇 **STELLA 競馬予想モード** へようこそ！\n\n"
                "以下の画像をアップロードして予想を開始できます：\n\n"
                "| 画像 | 必須 |\n"
                "|---|---|\n"
                "| 📊 レイティング表 | ✅ 必須 |\n"
                "| ⚡ スピード指数表 | ✅ 必須 |\n"
                "| 📋 レース情報 | 任意 |\n"
                "| 🔍 ファクター表 | 任意 |\n"
                "| 🤖 AI展開予測 | 任意 |\n\n"
                "👇 右側のパネルから画像を選択してください。"
            ),
            "action": "show_upload"
        })

    elif any(k in msg for k in ["使い方", "ヘルプ", "help", "Help", "?", "？"]):
        return jsonify({
            "reply": (
                "## 📖 STELLA の使い方\n\n"
                "1. **「競馬予想」** と入力する\n"
                "2. 右パネルで画像をアップロード（ドラッグ&ドロップ対応）\n"
                "3. **「予想を開始する」** ボタンをクリック\n"
                "4. OCR処理後、予想結果が表示されます\n\n"
                "### 必須画像\n"
                "- **レイティング表**（`--rating_img`）\n"
                "- **スピード指数表**（`--speed_index_img`）\n\n"
                "### 任意画像（精度向上）\n"
                "- レース情報・ファクター表・AI展開予測・馬場状態"
            )
        })

    else:
        return jsonify({
            "reply": (
                "こんにちは！🏇 **STELLA 競馬予想** アシスタントです。\n\n"
                "**「競馬予想」** と入力すると予想を開始できます。\n\n"
                "- `競馬予想` → レース予想を開始\n"
                "- `使い方` / `ヘルプ` → 使い方を表示"
            )
        })


@app.route("/predict", methods=["POST"])
def predict():
    """
    画像ファイルを受け取って予想ジョブを開始する。
    job_id を返す（進捗は /status/<job_id> でポーリング）。
    """
    rating_file = request.files.get("rating_img")
    speed_file  = request.files.get("speed_index_img")

    if not rating_file or not rating_file.filename:
        return jsonify({"error": "レイティング表画像（rating_img）が必要です"}), 400
    if not allowed(rating_file.filename):
        return jsonify({"error": "対応していないファイル形式です（jpg/png/bmp/webp）"}), 400
    if not speed_file or not speed_file.filename:
        return jsonify({"error": "スピード指数表画像（speed_index_img）が必要です"}), 400
    if not allowed(speed_file.filename):
        return jsonify({"error": "対応していないファイル形式です（jpg/png/bmp/webp）"}), 400

    # 画像を保存
    rating_path = save_upload(rating_file)
    speed_path  = save_upload(speed_file)

    def opt_save(key):
        f = request.files.get(key)
        if f and f.filename and allowed(f.filename):
            return save_upload(f)
        return None

    meta_path   = opt_save("meta_img")
    factor_path = opt_save("factor_img")
    aitime_path = opt_save("ai_time_img")

    pred_paths  = [save_upload(f) for f in request.files.getlist("pred_times_imgs")
                   if f.filename and allowed(f.filename)]
    track_paths = [save_upload(f) for f in request.files.getlist("track_imgs")
                   if f.filename and allowed(f.filename)]
    train_paths = [save_upload(f) for f in request.files.getlist("training_imgs")
                   if f.filename and allowed(f.filename)]

    fast = request.form.get("fast", "false").lower() == "true"

    job_id = uuid.uuid4().hex
    t = threading.Thread(
        target=run_stella,
        kwargs=dict(
            job_id=job_id,
            rating_img=str(rating_path),
            speed_index_img=str(speed_path),
            meta_img=str(meta_path) if meta_path else None,
            factor_img=str(factor_path) if factor_path else None,
            ai_time_img=str(aitime_path) if aitime_path else None,
            pred_times_imgs=[str(p) for p in pred_paths] or None,
            track_imgs=[str(p) for p in track_paths] or None,
            training_imgs=[str(p) for p in train_paths] or None,
            fast=fast,
        ),
        daemon=True,
    )
    t.start()

    return jsonify({"job_id": job_id})


@app.route("/status/<job_id>")
def status(job_id):
    """ジョブの進捗・結果を返す。"""
    with _lock:
        job = _jobs.get(job_id)
    if not job:
        return jsonify({"status": "not_found"}), 404

    # 実行時間も返す
    result = dict(job)
    if job["status"] == "running":
        result["elapsed"] = round(time.time() - job.get("start_time", time.time()), 1)
    return jsonify(result)


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(str(UPLOAD_DIR), filename)


@app.route("/health")
def health():
    return jsonify({"status": "ok", "ginandtonic": str(GINANDTONIC.exists())})


# ── 起動 ──────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    print(f"[STELLA] サーバー起動: http://0.0.0.0:{port}", flush=True)
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
