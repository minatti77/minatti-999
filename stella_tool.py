#!/usr/bin/env python3
"""
stella_tool.py  –  STELLA v1.33 OpenAI / スーパーエージェント向けツール定義
============================================================================

OpenAI の Function Calling / Responses API の tools 形式、および
汎用エージェントフレームワーク（LangChain, AutoGen, CrewAI 等）向けに
STELLA の各機能をツールとして定義します。

## OpenAI Responses API での使い方

```python
from stella_tool import STELLA_TOOLS, call_stella_tool

client = openai.OpenAI()
response = client.responses.create(
    model="gpt-4o",
    input=[{"role": "user", "content": "このレースを予想して"}],
    tools=STELLA_TOOLS,
)

# ツール呼び出しを処理
for item in response.output:
    if item.type == "function_call":
        result = call_stella_tool(item.name, json.loads(item.arguments))
        print(result)
```

## LangChain での使い方

```python
from stella_tool import get_langchain_tools
tools = get_langchain_tools()
agent = initialize_agent(tools, llm, ...)
```
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# stella_agent をインポート
_HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(_HERE))
from stella_agent import StellaAgent

_agent = StellaAgent()


# ═══════════════════════════════════════════════════════════════════
#  OpenAI tools 形式の定義
# ═══════════════════════════════════════════════════════════════════

STELLA_TOOLS: List[Dict[str, Any]] = [
    # ──────────────────────────────────────────
    # 1. 予想実行
    # ──────────────────────────────────────────
    {
        "type": "function",
        "name": "stella_predict",
        "description": (
            "STELLA v1.33 競馬予想エンジンを実行する。"
            "スクリーンショット画像のパスを受け取り、"
            "複勝・ワイド・三連複フォーメーションの予想 Markdown を返す。"
            "rating_img（レイティング表）と speed_index_img（スピード指数表）が必須。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "rating_img": {
                    "type": "string",
                    "description": "レイティング表スクリーンショットのファイルパス（必須）",
                },
                "speed_index_img": {
                    "type": "string",
                    "description": "スピード指数表スクリーンショットのファイルパス（必須）",
                },
                "meta_img": {
                    "type": "string",
                    "description": "レース情報スクリーンショットのパス（任意）",
                },
                "factor_img": {
                    "type": "string",
                    "description": "ファクター表スクリーンショットのパス（任意）",
                },
                "ai_time_img": {
                    "type": "string",
                    "description": "AI展開予測・タイム予測スクリーンショットのパス（任意）",
                },
                "pred_times_imgs": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "予測タイム表スクリーンショットのパスリスト（任意）",
                },
                "track_imgs": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "馬場状態スクリーンショットのパスリスト（任意）",
                },
                "training_keibabook_imgs": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "keibabook調教ページスクリーンショットのパスリスト（任意）",
                },
                "comments_csv": {
                    "type": "string",
                    "description": "厩舎コメント CSV ファイルのパス（num,type,text 形式）（任意）",
                },
                "out_md": {
                    "type": "string",
                    "description": "予想 Markdown の出力ファイルパス（省略時は自動生成）",
                },
                "params_json": {
                    "type": "string",
                    "description": "パラメータ JSON ファイルのパス（省略可）",
                },
                "ocr_cache_dir": {
                    "type": "string",
                    "description": "OCR キャッシュディレクトリのパス（省略可）",
                },
                "result_dir": {
                    "type": "string",
                    "description": "予想結果 JSON の保存先ディレクトリ（省略可）",
                },
                "race_id": {
                    "type": "string",
                    "description": "レース ID（省略時は自動生成）",
                },
                "fast": {
                    "type": "boolean",
                    "description": "高速モード（サンプル数を減らして高速化）。デフォルト false",
                },
            },
            "required": ["rating_img", "speed_index_img"],
        },
    },

    # ──────────────────────────────────────────
    # 2. 事後分析 (R41)
    # ──────────────────────────────────────────
    {
        "type": "function",
        "name": "stella_analyze",
        "description": (
            "STELLA R41: 保存済み予想結果 JSON から事後分析レポートを生成する。"
            "複勝・ワイドの的中率・ROI・ペース別統計などをまとめた Markdown を返す。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "result_dir": {
                    "type": "string",
                    "description": "予想結果 JSON が保存されているディレクトリのパス（必須）",
                },
                "out_md": {
                    "type": "string",
                    "description": "分析レポートの出力 Markdown パス（省略可）",
                },
            },
            "required": ["result_dir"],
        },
    },

    # ──────────────────────────────────────────
    # 3. confidence チューニング (R42)
    # ──────────────────────────────────────────
    {
        "type": "function",
        "name": "stella_tune_confidence",
        "description": (
            "STELLA R42: 過去の予想結果から confidence パラメータを自動チューニングする。"
            "最適な HIGH/MEDIUM/LOW の閾値を JSON で返す。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "result_dir": {
                    "type": "string",
                    "description": "予想結果 JSON ディレクトリのパス（必須）",
                },
                "out_json": {
                    "type": "string",
                    "description": "チューニング結果 JSON の出力パス（省略可）",
                },
                "min_high_n": {
                    "type": "integer",
                    "description": "有効 HIGH 件数の最小要件（デフォルト: 3）",
                },
            },
            "required": ["result_dir"],
        },
    },

    # ──────────────────────────────────────────
    # 4. ワイド ROI 最適化 (R43)
    # ──────────────────────────────────────────
    {
        "type": "function",
        "name": "stella_wide_roi",
        "description": (
            "STELLA R43: ワイド ROI 最適化分析を実行する。"
            "ポイント数・信頼度レベルごとの ROI を分析した Markdown を返す。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "result_dir": {
                    "type": "string",
                    "description": "予想結果 JSON ディレクトリのパス（必須）",
                },
                "out_md": {
                    "type": "string",
                    "description": "レポートの出力 Markdown パス（省略可）",
                },
                "stake": {
                    "type": "integer",
                    "description": "ワイド 1 点あたりの賭け金（円）。デフォルト: 100",
                },
            },
            "required": ["result_dir"],
        },
    },

    # ──────────────────────────────────────────
    # 5. 総合ダッシュボード (R45)
    # ──────────────────────────────────────────
    {
        "type": "function",
        "name": "stella_dashboard",
        "description": (
            "STELLA R45: 総合ダッシュボードレポートを生成する。"
            "複勝・ワイド・三連複の統合成績サマリーを Markdown で返す。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "result_dir": {
                    "type": "string",
                    "description": "予想結果 JSON ディレクトリのパス（必須）",
                },
                "out_md": {
                    "type": "string",
                    "description": "ダッシュボードレポートの出力 Markdown パス（省略可）",
                },
                "stake_place": {
                    "type": "integer",
                    "description": "複勝 1 レースの賭け金（円）。デフォルト: 1000",
                },
                "stake_wide": {
                    "type": "integer",
                    "description": "ワイド 1 点の賭け金（円）。デフォルト: 100",
                },
                "stake_trio": {
                    "type": "integer",
                    "description": "三連複 1 点の賭け金（円）。デフォルト: 100",
                },
            },
            "required": ["result_dir"],
        },
    },

    # ──────────────────────────────────────────
    # 6. 全分析一括 (R48)
    # ──────────────────────────────────────────
    {
        "type": "function",
        "name": "stella_full_analysis",
        "description": (
            "STELLA R48: R41〜R47 の全分析を一括実行して統合レポートを生成する。"
            "包括的な成績分析・チューニング推奨・ROI サマリーを Markdown で返す。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "result_dir": {
                    "type": "string",
                    "description": "予想結果 JSON ディレクトリのパス（必須）",
                },
                "out_md": {
                    "type": "string",
                    "description": "統合レポートの出力 Markdown パス（省略可）",
                },
                "stake_place": {
                    "type": "integer",
                    "description": "複勝 1 レースの賭け金（円）。デフォルト: 1000",
                },
                "stake_wide": {
                    "type": "integer",
                    "description": "ワイド 1 点の賭け金（円）。デフォルト: 100",
                },
                "stake_trio": {
                    "type": "integer",
                    "description": "三連複 1 点の賭け金（円）。デフォルト: 100",
                },
            },
            "required": ["result_dir"],
        },
    },

    # ──────────────────────────────────────────
    # 7. レース結果を記録 (R39)
    # ──────────────────────────────────────────
    {
        "type": "function",
        "name": "stella_record_result",
        "description": (
            "STELLA R39: レースの実際の結果を記録する。"
            "1着・複勝対象馬番・ワイド的中ペア・三連複的中馬番を保存する。"
            "事後分析（R41〜R48）のために必ず予想後に記録すること。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "race_id": {
                    "type": "string",
                    "description": "レース ID（必須）",
                },
                "result_dir": {
                    "type": "string",
                    "description": "結果を保存するディレクトリのパス（必須）",
                },
                "win_num": {
                    "type": "string",
                    "description": "1着馬番（必須）",
                },
                "place_nums": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "複勝対象馬番（1〜3着の馬番リスト）（必須）",
                },
                "wide_pairs": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "description": "ワイド的中ペアのリスト（例: [['3','7'],['3','11']]）（任意）",
                },
                "trio_nums": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "三連複的中馬番3頭のリスト（例: ['3','7','11']）（任意）",
                },
            },
            "required": ["race_id", "result_dir", "win_num", "place_nums"],
        },
    },
]


# ═══════════════════════════════════════════════════════════════════
#  ツール呼び出しルーター
# ═══════════════════════════════════════════════════════════════════

def call_stella_tool(name: str, arguments: Dict[str, Any]) -> str:
    """
    ツール名と引数を受け取り、対応する StellaAgent メソッドを呼び出す。
    エージェントに返す文字列（JSON または Markdown）を返す。

    Parameters
    ----------
    name : str
        STELLA_TOOLS に定義されたツール名。
    arguments : dict
        ツールの引数辞書。

    Returns
    -------
    str
        エージェントへの返答文字列。
    """
    try:
        if name == "stella_predict":
            result = _agent.predict(**arguments)
            if result["success"]:
                return result["markdown"] or "（予想 Markdown が空です）"
            else:
                return f"エラー: {result['error']}\n\nstderr:\n{result['stderr']}"

        elif name == "stella_analyze":
            result = _agent.analyze(**arguments)
            if result["success"]:
                return result["markdown"] or "（分析レポートが空です）"
            else:
                return f"エラー: {result['error']}"

        elif name == "stella_tune_confidence":
            result = _agent.tune_confidence(**arguments)
            if result["success"]:
                return json.dumps(result["tuning"], ensure_ascii=False, indent=2)
            else:
                return f"エラー: {result['error']}"

        elif name == "stella_wide_roi":
            result = _agent.wide_roi(**arguments)
            if result["success"]:
                return result["markdown"] or "（レポートが空です）"
            else:
                return f"エラー: {result['error']}"

        elif name == "stella_dashboard":
            result = _agent.dashboard(**arguments)
            if result["success"]:
                return result["markdown"] or "（ダッシュボードが空です）"
            else:
                return f"エラー: {result['error']}"

        elif name == "stella_full_analysis":
            result = _agent.full_analysis(**arguments)
            if result["success"]:
                return result["markdown"] or "（統合レポートが空です）"
            else:
                return f"エラー: {result['error']}"

        elif name == "stella_record_result":
            result = _agent.record_result(**arguments)
            if result["success"]:
                return "レース結果を記録しました。"
            else:
                return f"記録エラー: {result['error']}"

        else:
            return f"未知のツール名: {name}"

    except Exception as e:
        return f"ツール実行エラー [{name}]: {e}"


# ═══════════════════════════════════════════════════════════════════
#  LangChain 向けラッパー（オプション）
# ═══════════════════════════════════════════════════════════════════

def get_langchain_tools():
    """
    LangChain の Tool リストを返す。
    langchain がインストールされていない場合は空リストを返す。
    """
    try:
        from langchain.tools import Tool

        tools = []
        for spec in STELLA_TOOLS:
            fname = spec["name"]
            fdesc = spec["description"]

            def _make_func(tool_name):
                def _func(input_str: str) -> str:
                    try:
                        args = json.loads(input_str)
                    except Exception:
                        # 単純文字列が来た場合の fallback
                        args = {"rating_img": input_str, "speed_index_img": input_str}
                    return call_stella_tool(tool_name, args)
                return _func

            tools.append(Tool(
                name=fname,
                description=fdesc,
                func=_make_func(fname),
            ))
        return tools
    except ImportError:
        return []


# ═══════════════════════════════════════════════════════════════════
#  動作確認用 CLI
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=== STELLA_TOOLS 定義一覧 ===")
    for t in STELLA_TOOLS:
        props = list(t["parameters"]["properties"].keys())
        req   = t["parameters"].get("required", [])
        print(f"\n[{t['name']}]")
        print(f"  説明: {t['description'][:60]}...")
        print(f"  必須引数: {req}")
        print(f"  全引数:   {props}")
    print(f"\n合計: {len(STELLA_TOOLS)} ツール")
