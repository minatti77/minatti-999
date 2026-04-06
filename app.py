#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
app.py — GIN AND TONIC STELLA v2.0 — netkeita.com品質 競馬AI予想Webアプリ

netkeita.com のような8項目ランク指数マトリクス表示と
GINANDTONIC.py の全分析エンジンを統合した高精度予想アプリ。
"""

import os
import sys
import uuid
import subprocess
import threading
import time
import json
import re
from pathlib import Path
from flask import (
    Flask, request, jsonify, render_template,
    send_from_directory, Response
)
from werkzeug.utils import secure_filename

# ── パス設定 ──────────────────────────────────────
BASE_DIR    = Path(__file__).parent.resolve()
UPLOAD_DIR  = BASE_DIR / "uploads"
OUTPUT_DIR  = BASE_DIR / "output"
RESULT_DIR  = BASE_DIR / "output" / "results"
STATIC_DIR  = BASE_DIR / "static"
GINANDTONIC = BASE_DIR / "GINANDTONIC.py"

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)

ALLOWED_IMG_EXT = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
ALLOWED_PDF_EXT = {".pdf"}
ALLOWED_CSV_EXT = {".csv"}
ALLOWED_EXT     = ALLOWED_IMG_EXT | ALLOWED_PDF_EXT | ALLOWED_CSV_EXT

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200MB

# ── 実行中セッション管理 ─────────────────────────
_jobs: dict = {}
_lock = threading.Lock()

# ── 結果履歴管理 ──────────────────────────────────
_history: list = []


def allowed_img(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_IMG_EXT

def allowed_any(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXT

def save_upload(f) -> Path:
    uid  = uuid.uuid4().hex
    name = secure_filename(f.filename)
    ext  = Path(name).suffix.lower() or ".jpg"
    dest = UPLOAD_DIR / f"{uid}{ext}"
    f.save(dest)
    return dest


# ─────────────────────────────────────────────────
#  STELLA 実行（画像モード）
# ─────────────────────────────────────────────────
def run_stella(job_id: str, rating_img: str, speed_index_img: str,
               meta_img: str = None, factor_img: str = None,
               ai_time_img: str = None,
               pred_times_imgs: list = None,
               track_imgs: list = None,
               training_imgs: list = None,
               comments_csv: str = None,
               race_id: str = None,
               fast: bool = False):
    """バックグラウンドで GINANDTONIC.py を実行する（画像入力モード）。"""
    out_md = str(OUTPUT_DIR / f"{job_id}.md")

    cmd = [
        sys.executable, str(GINANDTONIC),
        "--rating_img",      rating_img,
        "--speed_index_img", speed_index_img,
        "--out_md",          out_md,
        "--result_dir",      str(RESULT_DIR),
        "--lang",            "jpn",
    ]
    if meta_img:        cmd += ["--meta_img",      meta_img]
    if factor_img:      cmd += ["--factor_img",    factor_img]
    if ai_time_img:     cmd += ["--ai_time_img",   ai_time_img]
    if pred_times_imgs: cmd += ["--pred_times_imgs"] + list(pred_times_imgs)
    if track_imgs:      cmd += ["--track_imgs"]    + list(track_imgs)
    if training_imgs:   cmd += ["--training_keibabook_imgs"] + list(training_imgs)
    if comments_csv:    cmd += ["--comments_csv",  comments_csv]
    if race_id:         cmd += ["--race_id",       race_id]
    if fast:            cmd.append("--fast")

    _run_cmd(job_id, cmd, mode="image")


# ─────────────────────────────────────────────────
#  STELLA 実行（PDF入力モード）
# ─────────────────────────────────────────────────
def run_stella_pdf(job_id: str, pdf_path: str,
                   fast: bool = False, dpi: int = 200):
    """バックグラウンドで GINANDTONIC.py を PDF 入力モードで実行する。"""
    out_md  = str(OUTPUT_DIR / f"{job_id}.md")
    pdf_img_dir = str(UPLOAD_DIR / f"pdf_{job_id}")

    cmd = [
        sys.executable, str(GINANDTONIC),
        "--pdf",         pdf_path,
        "--pdf_dpi",     str(dpi),
        "--pdf_out_dir", pdf_img_dir,
        "--out_md",      out_md,
        "--result_dir",  str(RESULT_DIR),
        "--lang",        "jpn",
    ]
    if fast:
        cmd.append("--fast")

    _run_cmd(job_id, cmd, mode="pdf")


def _run_cmd(job_id: str, cmd: list, mode: str = "image") -> None:
    """コマンドを実行しジョブ状態を管理する共通ヘルパー。"""
    with _lock:
        _jobs[job_id] = {
            "status":     "running",
            "log":        "",
            "markdown":   "",
            "error":      "",
            "start_time": time.time(),
            "mode":       mode,
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

        md_path = Path(str(OUTPUT_DIR / f"{job_id}.md"))
        md_text = md_path.read_text(encoding="utf-8") if md_path.exists() else ""

        elapsed = time.time() - _jobs[job_id]["start_time"]

        # 結果JSONから構造化データを抽出
        structured = _extract_structured_data(job_id, md_text)

        with _lock:
            if proc.returncode == 0 and md_text:
                _jobs[job_id]["status"]     = "done"
                _jobs[job_id]["markdown"]   = md_text
                _jobs[job_id]["elapsed"]    = round(elapsed, 1)
                _jobs[job_id]["structured"] = structured
                _add_history(job_id, md_text, structured)
            elif proc.returncode == 0:
                log = _jobs[job_id]["log"]
                _jobs[job_id]["status"]   = "done"
                _jobs[job_id]["markdown"] = f"## STELLA 実行ログ\n\n```\n{log}\n```"
                _jobs[job_id]["elapsed"]  = round(elapsed, 1)
            else:
                _jobs[job_id]["status"] = "error"
                _jobs[job_id]["error"]  = (
                    f"終了コード {proc.returncode}\n\n"
                    f"ログ (末尾500文字):\n{_jobs[job_id]['log'][-500:]}"
                )

    except subprocess.TimeoutExpired:
        try: proc.kill()
        except Exception: pass
        with _lock:
            _jobs[job_id]["status"] = "error"
            _jobs[job_id]["error"]  = "タイムアウト（600秒）"
    except Exception as e:
        with _lock:
            _jobs[job_id]["status"] = "error"
            _jobs[job_id]["error"]  = str(e)


def _extract_structured_data(job_id: str, md_text: str) -> dict:
    """Markdownと結果JSONから構造化データを抽出する。"""
    result = {
        "race_info": {},
        "anchor": {},
        "horses": [],
        "wide": [],
        "trio": {},
        "confidence": {},
        "rankings": [],
    }

    # 結果JSONファイルを探す
    try:
        for fp in sorted(RESULT_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
            try:
                obj = json.loads(fp.read_text(encoding="utf-8"))
                meta = obj.get("meta", {}) or {}
                pred = obj.get("prediction", {}) or {}
                scores = obj.get("scores", []) or []

                result["race_info"] = meta
                result["anchor"] = {
                    "num": pred.get("anchor_num", ""),
                    "name": pred.get("anchor_name", ""),
                    "score": pred.get("anchor_score"),
                    "p_place_est": pred.get("p_place_est"),
                    "p_place_est_pct": pred.get("p_place_est_pct"),
                    "place_ok": pred.get("place_ok", False),
                    "confidence_level": pred.get("confidence_level"),
                    "confidence_score_gap": pred.get("confidence_score_gap"),
                }
                result["wide"] = pred.get("wide_nums", [])
                result["trio"] = pred.get("trio_form", {})
                result["horses"] = scores

                # ランク指数を計算
                if scores:
                    result["rankings"] = _compute_rank_indices(scores)

                break  # 最新1件だけ使う
            except Exception:
                continue
    except Exception:
        pass

    # Markdownからレース情報を補完
    if md_text:
        result["race_info"] = _extract_race_info_from_md(md_text, result.get("race_info", {}))

    return result


def _compute_rank_indices(scores: list) -> list:
    """netkeita.com風の8項目ランク指数（S〜D）を計算する。"""
    if not scores:
        return []

    # 各スコア項目のマッピング
    score_fields = {
        "総合":   "AnchorScore",
        "能力":   "Ability",
        "展開":   "PosFit",
        "適性":   "MapFit",
        "安定":   "Consist" if any("Consist" in (s or {}) for s in scores) else "SAS",
        "上り":   "TimeFit",
        "調教":   "TrainingScore",
        "EV":    "p_place_est_pct",
    }

    # 各馬の指数を計算
    rankings = []
    for horse in scores:
        if not horse or not horse.get("num"):
            continue

        entry = {
            "num":  str(horse.get("num", "")),
            "name": str(horse.get("name", "")),
            "anchor_score": horse.get("AnchorScore"),
            "sas": horse.get("SAS"),
            "ranks": {},
            "raw_scores": {},
        }

        for label, field in score_fields.items():
            val = horse.get(field)
            entry["raw_scores"][label] = val

        rankings.append(entry)

    # 相対ランク（S〜D）を付与
    n = len(rankings)
    if n == 0:
        return []

    for label in score_fields:
        vals = []
        for r in rankings:
            v = r["raw_scores"].get(label)
            try:
                v = float(v) if v is not None else None
            except (ValueError, TypeError):
                v = None
            vals.append(v)

        # ソートしてランク付け
        valid_vals = [(i, v) for i, v in enumerate(vals) if v is not None]
        if not valid_vals:
            for r in rankings:
                r["ranks"][label] = "-"
            continue

        valid_vals.sort(key=lambda x: x[1], reverse=True)
        total = len(valid_vals)

        for rank_idx, (orig_idx, _) in enumerate(valid_vals):
            pct = rank_idx / max(total - 1, 1)
            if pct <= 0.15:
                grade = "S"
            elif pct <= 0.35:
                grade = "A"
            elif pct <= 0.60:
                grade = "B"
            elif pct <= 0.80:
                grade = "C"
            else:
                grade = "D"
            rankings[orig_idx]["ranks"][label] = grade

        # None値のランクは "-"
        for i, v in enumerate(vals):
            if v is None:
                rankings[i]["ranks"][label] = "-"

    return rankings


def _extract_race_info_from_md(md_text: str, base_info: dict) -> dict:
    """Markdownからレース情報を抽出する。"""
    info = dict(base_info or {})
    lines = md_text.split("\n")
    for line in lines[:30]:
        # レース名/情報パターン
        m = re.match(r'^#+\s+.*?(\d+R)\s*(.*)', line)
        if m:
            info.setdefault("race", m.group(1))
            if m.group(2).strip():
                info.setdefault("race_name", m.group(2).strip())

        # 日付パターン
        m = re.search(r'(\d{4}[/.-]\d{1,2}[/.-]\d{1,2})', line)
        if m:
            info.setdefault("date", m.group(1))

        # 競馬場
        for venue in ["東京", "中山", "阪神", "京都", "中京", "小倉", "新潟", "福島", "札幌", "函館",
                       "船橋", "大井", "川崎", "園田", "高知", "佐賀", "門別", "浦和", "名古屋", "笠松"]:
            if venue in line:
                info.setdefault("venue", venue)

    return info


def _add_history(job_id: str, md_text: str, structured: dict = None) -> None:
    """予想結果を履歴に追加する。"""
    global _history

    title = "予想結果"
    for line in md_text.splitlines():
        m = re.match(r'^#{1,2}\s+(.+)', line)
        if m:
            title = m.group(1)[:60]
            break

    race_info = (structured or {}).get("race_info", {})
    anchor = (structured or {}).get("anchor", {})

    entry = {
        "job_id":      job_id,
        "title":       title,
        "timestamp":   time.strftime("%Y-%m-%d %H:%M"),
        "venue":       race_info.get("venue", ""),
        "race":        race_info.get("race", ""),
        "date":        race_info.get("date", ""),
        "anchor_num":  anchor.get("num", ""),
        "anchor_name": anchor.get("name", ""),
        "confidence":  anchor.get("confidence_level", ""),
        "preview":     md_text[:300],
    }
    _history.insert(0, entry)
    _history = _history[:50]


# ── ルーティング ──────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    rating_file = request.files.get("rating_img")
    speed_file  = request.files.get("speed_index_img")

    if not rating_file or not rating_file.filename:
        return jsonify({"error": "レイティング表画像が必要です"}), 400
    if not speed_file or not speed_file.filename:
        return jsonify({"error": "スピード指数表画像が必要です"}), 400

    rating_path = save_upload(rating_file)
    speed_path  = save_upload(speed_file)

    def opt_save(key):
        f = request.files.get(key)
        if f and f.filename and allowed_img(f.filename):
            return save_upload(f)
        return None

    meta_path   = opt_save("meta_img")
    factor_path = opt_save("factor_img")
    aitime_path = opt_save("ai_time_img")

    pred_paths  = [save_upload(f) for f in request.files.getlist("pred_times_imgs")
                   if f.filename and allowed_img(f.filename)]
    track_paths = [save_upload(f) for f in request.files.getlist("track_imgs")
                   if f.filename and allowed_img(f.filename)]
    train_paths = [save_upload(f) for f in request.files.getlist("training_imgs")
                   if f.filename and allowed_img(f.filename)]

    fast    = request.form.get("fast", "false").lower() == "true"
    race_id = request.form.get("race_id", "").strip() or None

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
            race_id=race_id,
        ),
        daemon=True,
    )
    t.start()
    return jsonify({"job_id": job_id})


@app.route("/predict_pdf", methods=["POST"])
def predict_pdf():
    pdf_file = request.files.get("pdf")
    if not pdf_file or not pdf_file.filename:
        return jsonify({"error": "PDFファイルが必要です"}), 400

    ext = Path(secure_filename(pdf_file.filename)).suffix.lower()
    if ext != ".pdf":
        return jsonify({"error": "PDFファイルのみ対応"}), 400

    pdf_path = save_upload(pdf_file)
    fast = request.form.get("fast", "false").lower() == "true"
    dpi  = int(request.form.get("dpi", "200") or "200")

    job_id = uuid.uuid4().hex
    t = threading.Thread(
        target=run_stella_pdf,
        kwargs=dict(job_id=job_id, pdf_path=str(pdf_path), fast=fast, dpi=dpi),
        daemon=True,
    )
    t.start()
    return jsonify({"job_id": job_id})


@app.route("/status/<job_id>")
def status(job_id):
    with _lock:
        job = _jobs.get(job_id)
    if not job:
        return jsonify({"status": "not_found"}), 404

    result = dict(job)
    if job["status"] == "running":
        result["elapsed"] = round(time.time() - job.get("start_time", time.time()), 1)
    return jsonify(result)


@app.route("/history")
def history():
    return jsonify({"history": _history})


@app.route("/results")
def results_list():
    """保存済みの全予想結果JSONのサマリーを返す。"""
    items = []
    try:
        for fp in sorted(RESULT_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
            try:
                obj = json.loads(fp.read_text(encoding="utf-8"))
                meta = obj.get("meta", {}) or {}
                pred = obj.get("prediction", {}) or {}
                actual = obj.get("actual", {}) or {}
                items.append({
                    "race_id": obj.get("race_id", fp.stem),
                    "created_at": obj.get("created_at", ""),
                    "venue": meta.get("venue", ""),
                    "race": meta.get("race", ""),
                    "date": meta.get("date", ""),
                    "anchor_num": pred.get("anchor_num", ""),
                    "anchor_name": pred.get("anchor_name", ""),
                    "place_ok": pred.get("place_ok"),
                    "confidence": pred.get("confidence_level"),
                    "place_hit": actual.get("place_hit"),
                    "trio_hit": actual.get("trio_hit"),
                })
            except Exception:
                continue
    except Exception:
        pass
    return jsonify({"results": items})


@app.route("/result/<job_id>")
def get_result(job_id):
    """特定ジョブのMarkdown+構造化データ結果を返す。"""
    md_path = OUTPUT_DIR / f"{job_id}.md"
    md_text = ""
    if md_path.exists():
        md_text = md_path.read_text(encoding="utf-8")

    with _lock:
        job = _jobs.get(job_id)

    structured = {}
    if job:
        structured = job.get("structured", {})
        if not md_text:
            md_text = job.get("markdown", "")

    if md_text:
        return jsonify({"success": True, "markdown": md_text, "structured": structured})

    return jsonify({"success": False, "error": "結果が見つかりません"}), 404


@app.route("/record_result", methods=["POST"])
def record_result():
    """レース結果を記録する。"""
    data = request.get_json(force=True)
    race_id  = data.get("race_id", "").strip()
    rank_1st = data.get("rank_1st", "").strip()
    rank_2nd = data.get("rank_2nd", "").strip()
    rank_3rd = data.get("rank_3rd", "").strip()

    if not race_id:
        return jsonify({"error": "race_idが必要です"}), 400

    # GINANDTONIC の record_race_actual を使用
    try:
        sys.path.insert(0, str(BASE_DIR))
        from GINANDTONIC import record_race_actual
        ok = record_race_actual(str(RESULT_DIR), race_id, rank_1st or None, rank_2nd or None, rank_3rd or None)
        return jsonify({"success": ok})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(str(UPLOAD_DIR), filename)


@app.route("/health")
def health():
    try:
        import pytesseract
        tess_ok = True
    except Exception:
        tess_ok = False

    try:
        import fitz
        pdf_ok = True
    except ImportError:
        pdf_ok = False

    return jsonify({
        "status":         "ok",
        "ginandtonic":    GINANDTONIC.exists(),
        "pdf_support":    pdf_ok,
        "tesseract":      tess_ok,
        "active_jobs":    len([j for j in _jobs.values() if j["status"] == "running"]),
        "total_history":  len(_history),
        "total_results":  len(list(RESULT_DIR.glob("*.json"))),
    })


# ── 起動 ──────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    print(f"[STELLA] サーバー起動: http://0.0.0.0:{port}", flush=True)
    print(f"[STELLA] GINANDTONIC.py: {GINANDTONIC.exists()}", flush=True)
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
