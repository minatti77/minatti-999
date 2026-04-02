#!/usr/bin/env python3
"""
stella_agent.py  –  STELLA v1.33 スーパーエージェント向けインターフェース
==========================================================================

スーパーエージェント（AI Agent）から GINANDTONIC.py を簡単に呼び出すための
薄いラッパーモジュールです。

## 基本的な使い方

```python
from stella_agent import StellaAgent

agent = StellaAgent()

# 最小構成：レイティング画像 + スピード指数画像 を渡すだけ
result = agent.predict(
    rating_img="rating.png",
    speed_index_img="speed.png",
)
print(result["markdown"])   # 予想 Markdown
print(result["success"])    # True / False
print(result["error"])      # エラー文字列（成功時は None）
```

## 全画像を渡す場合

```python
result = agent.predict(
    rating_img="rating.png",
    speed_index_img="speed.png",
    meta_img="meta.png",
    factor_img="factor.png",
    ai_time_img="ai_time.png",
    pred_times_imgs=["pred1.png", "pred2.png"],
    track_imgs=["track1.png"],
    training_keibabook_imgs=["train1.png"],
    comments_csv="comments.csv",
    out_md="output.md",         # Markdown を保存したいパス（省略可）
    params_json="params.json",  # パラメータ JSON（省略可）
    ocr_cache_dir="cache/ocr",  # OCR キャッシュ（省略可）
    fast=True,                  # 高速モード（省略可）
)
```

## 事後分析・チューニング

```python
# R41 事後分析
analysis = agent.analyze(result_dir="output/results")

# R42 confidence チューニング
tuning = agent.tune_confidence(result_dir="output/results")

# R45 ダッシュボード
dashboard = agent.dashboard(result_dir="output/results")

# R48 全分析一括
full = agent.full_analysis(result_dir="output/results")
```
"""

from __future__ import annotations

import subprocess
import sys
import os
import json
import tempfile
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional


# ───────────────────────────────────────────────
# パス解決
# ───────────────────────────────────────────────
_HERE = Path(__file__).parent.resolve()
_GINANDTONIC = _HERE / "GINANDTONIC.py"


# ───────────────────────────────────────────────
# ヘルパー
# ───────────────────────────────────────────────
def _run(cmd: List[str], timeout: int = 300) -> Dict[str, Any]:
    """サブプロセスで cmd を実行し、結果を辞書で返す。"""
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(_HERE),
        )
        return {
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "success": proc.returncode == 0,
        }
    except subprocess.TimeoutExpired:
        return {
            "returncode": -1,
            "stdout": "",
            "stderr": f"タイムアウト（{timeout}秒）",
            "success": False,
        }
    except Exception as e:
        return {
            "returncode": -1,
            "stdout": "",
            "stderr": str(e),
            "success": False,
        }


def _read_md(path: str) -> str:
    """Markdown ファイルを読み込む。"""
    try:
        return Path(path).read_text(encoding="utf-8")
    except Exception:
        return ""


# ───────────────────────────────────────────────
# StellaAgent クラス
# ───────────────────────────────────────────────
class StellaAgent:
    """
    STELLA v1.33 を呼び出すエージェント向けクラス。

    Parameters
    ----------
    python : str
        使用する Python インタープリタのパス（デフォルト: sys.executable）。
    default_timeout : int
        デフォルトのタイムアウト秒数（デフォルト: 300）。
    """

    def __init__(
        self,
        python: str = sys.executable,
        default_timeout: int = 300,
    ) -> None:
        self.python = python
        self.default_timeout = default_timeout

    # ────────────────────────────────────────
    # メイン予想
    # ────────────────────────────────────────
    def predict(
        self,
        rating_img: str,
        speed_index_img: str,
        meta_img: Optional[str] = None,
        factor_img: Optional[str] = None,
        ai_time_img: Optional[str] = None,
        pred_times_imgs: Optional[List[str]] = None,
        track_imgs: Optional[List[str]] = None,
        training_keibabook_imgs: Optional[List[str]] = None,
        entries_img: Optional[str] = None,
        entries_csv: Optional[str] = None,
        comments_csv: Optional[str] = None,
        comments_inline: Optional[str] = None,
        out_md: Optional[str] = None,
        params_json: Optional[str] = None,
        ocr_cache_dir: Optional[str] = None,
        analysis_cache_dir: Optional[str] = None,
        run_cache_dir: Optional[str] = None,
        result_dir: Optional[str] = None,
        race_id: Optional[str] = None,
        fast: bool = False,
        lang: str = "jpn",
        ocr_workers: Optional[int] = None,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        スクリーンショット画像を受け取り、予想 Markdown を返す。

        Parameters
        ----------
        rating_img : str
            レイティング表スクリーンショットのパス（必須）。
        speed_index_img : str
            スピード指数表スクリーンショットのパス（必須）。
        （以下すべて省略可）
        meta_img, factor_img, ai_time_img, pred_times_imgs,
        track_imgs, training_keibabook_imgs, entries_img,
        entries_csv, comments_csv, comments_inline,
        out_md, params_json, ocr_cache_dir, analysis_cache_dir,
        run_cache_dir, result_dir, race_id, fast, lang,
        ocr_workers, timeout

        Returns
        -------
        dict
            {
                "success": bool,
                "markdown": str,   # 予想 Markdown 全文
                "out_md": str,     # 出力ファイルパス
                "stdout": str,
                "stderr": str,
                "error": str | None,
            }
        """
        # out_md が未指定なら一時ファイルへ
        _tmp_dir = None
        if out_md is None:
            _tmp_dir = tempfile.mkdtemp(prefix="stella_")
            out_md = str(Path(_tmp_dir) / "prediction.md")

        cmd = [
            self.python, str(_GINANDTONIC),
            "--rating_img", rating_img,
            "--speed_index_img", speed_index_img,
            "--out_md", out_md,
            "--lang", lang,
        ]

        # オプション引数を追加
        if meta_img:
            cmd += ["--meta_img", meta_img]
        if factor_img:
            cmd += ["--factor_img", factor_img]
        if ai_time_img:
            cmd += ["--ai_time_img", ai_time_img]
        if pred_times_imgs:
            cmd += ["--pred_times_imgs"] + pred_times_imgs
        if track_imgs:
            cmd += ["--track_imgs"] + track_imgs
        if training_keibabook_imgs:
            cmd += ["--training_keibabook_imgs"] + training_keibabook_imgs
        if entries_img:
            cmd += ["--entries_img", entries_img]
        if entries_csv:
            cmd += ["--entries_csv", entries_csv]
        if comments_csv:
            cmd += ["--comments_csv", comments_csv]
        if comments_inline:
            cmd += ["--comments_inline", comments_inline]
        if params_json:
            cmd += ["--params_json", params_json]
        if ocr_cache_dir:
            cmd += ["--ocr_cache_dir", ocr_cache_dir]
        if analysis_cache_dir:
            cmd += ["--analysis_cache_dir", analysis_cache_dir]
        if run_cache_dir:
            cmd += ["--run_cache_dir", run_cache_dir]
        if result_dir:
            cmd += ["--result_dir", result_dir]
        if race_id:
            cmd += ["--race_id", race_id]
        if ocr_workers is not None:
            cmd += ["--ocr_workers", str(ocr_workers)]
        if fast:
            cmd.append("--fast")

        result = _run(cmd, timeout=timeout or self.default_timeout)
        markdown = _read_md(out_md)

        # 一時ディレクトリのクリーンアップ（out_md を読んだ後）
        # ※ caller が markdown を使う前にディレクトリを消さないよう、ここでは削除しない
        # （必要なら caller 側で del result 後に _tmp_dir を消す）

        return {
            "success": result["success"] and bool(markdown),
            "markdown": markdown,
            "out_md": out_md,
            "stdout": result["stdout"],
            "stderr": result["stderr"],
            "error": result["stderr"] if not result["success"] else None,
            "_tmp_dir": _tmp_dir,  # caller がクリーンアップ用に参照可
        }

    # ────────────────────────────────────────
    # 事後分析 (R41)
    # ────────────────────────────────────────
    def analyze(
        self,
        result_dir: str,
        out_md: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """R41 事後分析レポートを生成する。"""
        _tmp_dir = None
        if out_md is None:
            _tmp_dir = tempfile.mkdtemp(prefix="stella_analyze_")
            out_md = str(Path(_tmp_dir) / "analysis_report.md")

        cmd = [
            self.python, str(_GINANDTONIC),
            "--analyze_results", result_dir,
            "--analyze_out_md", out_md,
        ]
        result = _run(cmd, timeout=timeout or self.default_timeout)
        markdown = _read_md(out_md)
        return {
            "success": result["success"],
            "markdown": markdown,
            "out_md": out_md,
            "stdout": result["stdout"],
            "stderr": result["stderr"],
            "error": result["stderr"] if not result["success"] else None,
        }

    # ────────────────────────────────────────
    # confidence チューニング (R42)
    # ────────────────────────────────────────
    def tune_confidence(
        self,
        result_dir: str,
        out_json: Optional[str] = None,
        min_high_n: int = 3,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """R42 confidence パラメータを自動チューニングする。"""
        _tmp_dir = None
        if out_json is None:
            _tmp_dir = tempfile.mkdtemp(prefix="stella_tune_")
            out_json = str(Path(_tmp_dir) / "tuning_result.json")

        cmd = [
            self.python, str(_GINANDTONIC),
            "--tune_confidence", result_dir,
            "--tune_out_json", out_json,
            "--tune_min_high_n", str(min_high_n),
        ]
        result = _run(cmd, timeout=timeout or self.default_timeout)
        tuning_data = {}
        try:
            tuning_data = json.loads(Path(out_json).read_text(encoding="utf-8"))
        except Exception:
            pass
        return {
            "success": result["success"],
            "tuning": tuning_data,
            "out_json": out_json,
            "stdout": result["stdout"],
            "stderr": result["stderr"],
            "error": result["stderr"] if not result["success"] else None,
        }

    # ────────────────────────────────────────
    # ワイド ROI 最適化 (R43)
    # ────────────────────────────────────────
    def wide_roi(
        self,
        result_dir: str,
        out_md: Optional[str] = None,
        stake: int = 100,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """R43 ワイド ROI 最適化分析を実行する。"""
        _tmp_dir = None
        if out_md is None:
            _tmp_dir = tempfile.mkdtemp(prefix="stella_wide_roi_")
            out_md = str(Path(_tmp_dir) / "wide_roi_report.md")

        cmd = [
            self.python, str(_GINANDTONIC),
            "--wide_roi", result_dir,
            "--wide_roi_out_md", out_md,
            "--wide_roi_stake", str(stake),
        ]
        result = _run(cmd, timeout=timeout or self.default_timeout)
        return {
            "success": result["success"],
            "markdown": _read_md(out_md),
            "out_md": out_md,
            "stdout": result["stdout"],
            "stderr": result["stderr"],
            "error": result["stderr"] if not result["success"] else None,
        }

    # ────────────────────────────────────────
    # 三連複 ROI 最適化 (R44)
    # ────────────────────────────────────────
    def trio_roi(
        self,
        result_dir: str,
        out_md: Optional[str] = None,
        stake: int = 100,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """R44 三連複 ROI 最適化分析を実行する。"""
        _tmp_dir = None
        if out_md is None:
            _tmp_dir = tempfile.mkdtemp(prefix="stella_trio_roi_")
            out_md = str(Path(_tmp_dir) / "trio_roi_report.md")

        cmd = [
            self.python, str(_GINANDTONIC),
            "--trio_roi", result_dir,
            "--trio_roi_out_md", out_md,
            "--trio_roi_stake", str(stake),
        ]
        result = _run(cmd, timeout=timeout or self.default_timeout)
        return {
            "success": result["success"],
            "markdown": _read_md(out_md),
            "out_md": out_md,
            "stdout": result["stdout"],
            "stderr": result["stderr"],
            "error": result["stderr"] if not result["success"] else None,
        }

    # ────────────────────────────────────────
    # 総合ダッシュボード (R45)
    # ────────────────────────────────────────
    def dashboard(
        self,
        result_dir: str,
        out_md: Optional[str] = None,
        stake_place: int = 1000,
        stake_wide: int = 100,
        stake_trio: int = 100,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """R45 総合ダッシュボードレポートを生成する。"""
        _tmp_dir = None
        if out_md is None:
            _tmp_dir = tempfile.mkdtemp(prefix="stella_dashboard_")
            out_md = str(Path(_tmp_dir) / "dashboard_report.md")

        cmd = [
            self.python, str(_GINANDTONIC),
            "--dashboard", result_dir,
            "--dashboard_out_md", out_md,
            "--dashboard_stake_place", str(stake_place),
            "--dashboard_stake_wide", str(stake_wide),
            "--dashboard_stake_trio", str(stake_trio),
        ]
        result = _run(cmd, timeout=timeout or self.default_timeout)
        return {
            "success": result["success"],
            "markdown": _read_md(out_md),
            "out_md": out_md,
            "stdout": result["stdout"],
            "stderr": result["stderr"],
            "error": result["stderr"] if not result["success"] else None,
        }

    # ────────────────────────────────────────
    # 全分析一括 (R48)
    # ────────────────────────────────────────
    def full_analysis(
        self,
        result_dir: str,
        out_md: Optional[str] = None,
        stake_place: int = 1000,
        stake_wide: int = 100,
        stake_trio: int = 100,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """R48 全分析（R41〜R47）を一括実行して統合レポートを生成する。"""
        _tmp_dir = None
        if out_md is None:
            _tmp_dir = tempfile.mkdtemp(prefix="stella_full_")
            out_md = str(Path(_tmp_dir) / "full_analysis_report.md")

        cmd = [
            self.python, str(_GINANDTONIC),
            "--full_analysis", result_dir,
            "--full_analysis_out_md", out_md,
            "--full_analysis_stake_place", str(stake_place),
            "--full_analysis_stake_wide", str(stake_wide),
            "--full_analysis_stake_trio", str(stake_trio),
        ]
        result = _run(cmd, timeout=timeout or self.default_timeout)
        return {
            "success": result["success"],
            "markdown": _read_md(out_md),
            "out_md": out_md,
            "stdout": result["stdout"],
            "stderr": result["stderr"],
            "error": result["stderr"] if not result["success"] else None,
        }

    # ────────────────────────────────────────
    # 実際の結果を記録 (R39 record)
    # ────────────────────────────────────────
    def record_result(
        self,
        race_id: str,
        result_dir: str,
        win_num: str,
        place_nums: List[str],
        wide_pairs: Optional[List[List[str]]] = None,
        trio_nums: Optional[List[str]] = None,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        レース結果を JSON として保存する（R39）。

        Parameters
        ----------
        race_id : str    レース ID
        result_dir : str 保存先ディレクトリ
        win_num : str    1着馬番
        place_nums : list[str]  複勝対象馬番（1〜3着）
        wide_pairs : list[list[str]]  ワイド的中ペア（省略可）
        trio_nums : list[str]   三連複的中馬番3頭（省略可）
        """
        import importlib.util
        spec = importlib.util.spec_from_file_location("GINANDTONIC", str(_GINANDTONIC))
        mod = importlib.util.module_from_spec(spec)
        sys.modules["GINANDTONIC"] = mod
        spec.loader.exec_module(mod)

        try:
            mod.record_race_actual(
                race_id=race_id,
                result_dir=result_dir,
                win_num=str(win_num),
                place_nums=[str(x) for x in place_nums],
                wide_pairs=[[str(a), str(b)] for a, b in (wide_pairs or [])],
                trio_nums=[str(x) for x in (trio_nums or [])],
            )
            return {"success": True, "error": None}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ────────────────────────────────────────
    # CLI コマンド文字列を返すだけ（デバッグ用）
    # ────────────────────────────────────────
    def build_command(
        self,
        rating_img: str,
        speed_index_img: str,
        **kwargs,
    ) -> str:
        """predict() が実行するコマンド文字列を返す（確認・デバッグ用）。"""
        r = self.predict(
            rating_img=rating_img,
            speed_index_img=speed_index_img,
            **kwargs,
        )
        return r


# ───────────────────────────────────────────────
# スタンドアロン実行（動作確認）
# ───────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="StellaAgent 動作確認")
    ap.add_argument("--rating_img",      required=True, help="レイティング表画像")
    ap.add_argument("--speed_index_img", required=True, help="スピード指数表画像")
    ap.add_argument("--meta_img",        default=None)
    ap.add_argument("--factor_img",      default=None)
    ap.add_argument("--ai_time_img",     default=None)
    ap.add_argument("--out_md",          default=None, help="出力 Markdown パス")
    ap.add_argument("--fast",            action="store_true")
    args = ap.parse_args()

    agent = StellaAgent()
    result = agent.predict(
        rating_img=args.rating_img,
        speed_index_img=args.speed_index_img,
        meta_img=args.meta_img,
        factor_img=args.factor_img,
        ai_time_img=args.ai_time_img,
        out_md=args.out_md,
        fast=args.fast,
    )

    if result["success"]:
        print("=== 予想結果 ===")
        print(result["markdown"][:2000], "..." if len(result["markdown"]) > 2000 else "")
    else:
        print("=== エラー ===")
        print(result["stderr"])
        sys.exit(1)
