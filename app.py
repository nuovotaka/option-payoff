# app.py
# オプション戦略 損益グラフ（BS 付き）＋ yfinance 原資産
# 取引単位の自動推測＋最大損失 / 最大利益表示付き
# プレミアムは常に正の値で入力
# K は銘柄ごとの刻み幅（株式100円 / N225系125円 / TOPIX系25円）で、
# ティッカー or 中心価格変更時にだけ自動初期化（session_state 利用）
# Flet版機能を追加：数値入力正規化、横軸範囲コントロール強化、ストライクタグ表示

import math
import re
import unicodedata
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from dataclasses import dataclass
from typing import List, Literal
from scipy.stats import norm
import yfinance as yf
from enum import Enum


OptionType = Literal["call", "put"]
SideType = Literal["long", "short"]


def sanitize_numeric_text(val) -> str:
    """全角数字・記号を半角に正規化し、数値以外を除去"""
    normalized = unicodedata.normalize("NFKC", str(val or ""))
    return re.sub(r"[^0-9.-]", "", normalized)


def safe_float(val, default=0.0) -> float:
    """文字列を安全にfloatに変換"""
    try:
        def_val = float(default) if default is not None else 0.0
    except (ValueError, TypeError):
        def_val = 0.0

    if val is None:
        return def_val
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip()
    if not s:
        return def_val
    s = sanitize_numeric_text(s)
    try:
        return float(s)
    except (ValueError, TypeError):
        cleaned = s
        if cleaned.count(".") > 1:
            parts = cleaned.split(".")
            cleaned = parts[0] + "." + "".join(parts[1:])
        try:
            return float(cleaned)
        except (ValueError, TypeError):
            return def_val


def safe_int(val, default=1) -> int:
    """文字列を安全にintに変換"""
    try:
        def_val = int(default) if default is not None else 1
    except (ValueError, TypeError):
        def_val = 1
    f = safe_float(val, default=float(def_val))
    return int(round(f))


def get_last_close(ticker: str) -> float | None:
    try:
        data = yf.Ticker(ticker).history(period="5d")
        if data.empty:
            return None
        return float(data["Close"].iloc[-1])
    except Exception:
        return None


def guess_contract_size_from_ticker(ticker: str) -> int:
    t = ticker.upper()
    if "N225" in t:
        if "MICRO" in t:
            return 10
        if "MINI" in t:
            return 100
        return 1000
    return 100


def guess_strike_step_from_ticker(ticker: str) -> float:
    t = ticker.upper()
    if "TOPIX" in t or "TPX" in t:
        return 25.0
    if "N225" in t:
        return 125.0
    return 100.0


def round_to_step(x: float, step: float) -> float:
    if step <= 0:
        return x
    return round(x / step) * step


def bs_price(
    option_type: OptionType,
    S: np.ndarray,
    K: float,
    r: float,
    sigma: float,
    T: float
) -> np.ndarray:
    S = np.array(S, dtype=float)

    if T <= 0 or sigma <= 0:
        if option_type == "call":
            intrinsic = np.maximum(S - K, 0.0)
        else:
            intrinsic = np.maximum(K - S, 0.0)
        return intrinsic

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


@dataclass
class OptionPosition:
    option_type: OptionType
    side: SideType        # "long" or "short"
    strike: float
    premium: float        # 常に >= 0
    quantity: int = 1
    contract_size: int = 1
    r: float = 0.0
    sigma: float = 0.2
    T: float = 0.1

    def get_tag_color(self):
        """Call/Put × Long/Short でタグの色と塗りを決定"""
        if self.option_type == "call":
            base_color = "#22c55e"  # 緑
        else:
            base_color = "#ef4444"  # 赤

        if self.side == "short":
            # ショートは塗りつぶし
            return base_color, "white"
        else:
            # ロングは枠線のみ
            return "white", base_color

    def payoff_at_maturity(self, S: np.ndarray) -> np.ndarray:
        if self.option_type == "call":
            intrinsic = np.maximum(S - self.strike, 0.0)
        else:
            intrinsic = np.maximum(self.strike - S, 0.0)

        if self.side == "long":
            pl_per_contract = intrinsic - self.premium
        else:
            pl_per_contract = -intrinsic + self.premium

        return pl_per_contract * self.quantity * self.contract_size

    def pl_now_bs(self, S: np.ndarray) -> np.ndarray:
        theo = bs_price(
            option_type=self.option_type,
            S=S,
            K=self.strike,
            r=self.r,
            sigma=self.sigma,
            T=self.T
        )

        if self.side == "long":
            position_value_now = +theo
            initial_cf_per_contract = -self.premium
        else:
            position_value_now = -theo
            initial_cf_per_contract = +self.premium

        pl_per_contract = position_value_now + initial_cf_per_contract
        return pl_per_contract * self.quantity * self.contract_size


def total_payoff_maturity(positions: List[OptionPosition], S: np.ndarray) -> np.ndarray:
    total = np.zeros_like(S, dtype=float)
    for p in positions:
        total += p.payoff_at_maturity(S)
    return total


def total_pl_now_bs(positions: List[OptionPosition], S: np.ndarray) -> np.ndarray:
    total = np.zeros_like(S, dtype=float)
    for p in positions:
        total += p.pl_now_bs(S)
    return total


# ============================
# 戦略プリセット定義
# ============================
class StrategyPreset(str, Enum):
    NONE = "手動で設定する"

    # 基本4ポジション
    LONG_CALL = "ロング・コール（強気）"
    SHORT_CALL = "ショート・コール（弱気〜横ばい）"
    LONG_PUT = "ロング・プット（弱気）"
    SHORT_PUT = "ショート・プット（強気〜横ばい）"

    # バーティカル・スプレッド
    BULL_CALL_SPREAD = "ブル・コール・スプレッド（緩やかな強気）"
    BEAR_PUT_SPREAD = "ベア・プット・スプレッド（緩やかな弱気）"
    BULL_PUT_SPREAD = "ブル・プット・スプレッド（クレジット・強気〜横ばい）"
    BEAR_CALL_SPREAD = "ベア・コール・スプレッド（クレジット・弱気〜横ばい）"

    # ボラティリティ系（変動狙い）
    LONG_STRADDLE = "ロング・ストラドル（変動大きめ予想）"
    SHORT_STRADDLE = "ショート・ストラドル（変動小さめ予想・高リスク）"
    LONG_STRANGLE = "ロング・ストラングル（大化け・急変狙い、低コスト）"
    SHORT_STRANGLE = "ショート・ストラングル（買いなし・上級者向け）"

    # レンジ相場向け・クレジット系
    IRON_CONDOR = "アイアン・コンドル（広めレンジでコツコツ）"
    IRON_BUTTERFLY = "アイアン・バタフライ（狭いレンジをピンポイント）"

    # 現物組み合わせ・ヘッジ系（※ここではオプション部分のみ）
    COVERED_CALL = "カバード・コール（現物保有＋コール売り）"
    PROTECTIVE_PUT = "プロテクティブ・プット（現物下落の保険）"
    COLLAR = "カラー（低コストの現物保全）"
    CASH_SECURED_PUT = "キャッシュ・セキュアード・プット（安値での指値買い代用）"

    # 上級者向け（レシオ・カレンダー）
    PUT_RATIO_SPREAD = "プット・レシオ・スプレッド（限定的な下落〜横ばい）"
    CALENDAR_SPREAD = "カレンダー・スプレッド（満期ズレ・時間価値狙い）"

    # ジェイド・リザード系
    JADE_LIZARD = "ジェイド・リザード（レンジ〜やや強気・上値限定）"
    SAFE_JADE_LIZARD = "ジェイド・リザード（安全版・両サイド損失限定）"


def build_positions_from_preset(
    preset: StrategyPreset,
    spot_center: float,
    contract_size: int,
    r: float,
    ticker: str,
) -> List[OptionPosition]:
    """
    プリセット名から OptionPosition のリストを作る。
    プレミアム・σ・T は「雰囲気用の仮置き値」。
    現物を使う戦略は、ここではオプション部分のみを構成。
    """
    positions: List[OptionPosition] = []

    sigma = 0.2
    T_short = 30 / 365      # 近い満期
    T_long = 90 / 365       # 遠い満期（カレンダー用など）

    step = guess_strike_step_from_ticker(ticker)
    k_atm = round_to_step(spot_center, step)
    k_low = round_to_step(spot_center * 0.95, step)
    k_high = round_to_step(spot_center * 1.05, step)
    k_very_low = round_to_step(spot_center * 0.9, step)
    k_very_high = round_to_step(spot_center * 1.1, step)

    # ---- 基本4ポジション ----
    if preset == StrategyPreset.LONG_CALL:
        positions.append(
            OptionPosition("call", "long", k_atm, 300.0, 1, contract_size, r, sigma, T_short)
        )

    elif preset == StrategyPreset.SHORT_CALL:
        positions.append(
            OptionPosition("call", "short", k_atm, 300.0, 1, contract_size, r, sigma, T_short)
        )

    elif preset == StrategyPreset.LONG_PUT:
        positions.append(
            OptionPosition("put", "long", k_atm, 300.0, 1, contract_size, r, sigma, T_short)
        )

    elif preset == StrategyPreset.SHORT_PUT:
        positions.append(
            OptionPosition("put", "short", k_atm, 300.0, 1, contract_size, r, sigma, T_short)
        )

    # ---- バーティカル・スプレッド ----
    elif preset == StrategyPreset.BULL_CALL_SPREAD:
        # 低いKコール買い、高いKコール売り（デビット）
        positions.append(
            OptionPosition("call", "long", k_low, 400.0, 1, contract_size, r, sigma, T_short)
        )
        positions.append(
            OptionPosition("call", "short", k_high, 200.0, 1, contract_size, r, sigma, T_short)
        )

    elif preset == StrategyPreset.BEAR_PUT_SPREAD:
        # 高いKプット買い、低いKプット売り（デビット）
        positions.append(
            OptionPosition("put", "long", k_high, 400.0, 1, contract_size, r, sigma, T_short)
        )
        positions.append(
            OptionPosition("put", "short", k_low, 200.0, 1, contract_size, r, sigma, T_short)
        )

    elif preset == StrategyPreset.BULL_PUT_SPREAD:
        # 高いKプット売り、低いKプット買い（クレジット）
        positions.append(
            OptionPosition("put", "short", k_high, 400.0, 1, contract_size, r, sigma, T_short)
        )
        positions.append(
            OptionPosition("put", "long", k_low, 200.0, 1, contract_size, r, sigma, T_short)
        )

    elif preset == StrategyPreset.BEAR_CALL_SPREAD:
        # 低いKコール売り、高いKコール買い（クレジット）
        positions.append(
            OptionPosition("call", "short", k_low, 400.0, 1, contract_size, r, sigma, T_short)
        )
        positions.append(
            OptionPosition("call", "long", k_high, 200.0, 1, contract_size, r, sigma, T_short)
        )

    # ---- ボラティリティ系 ----
    elif preset == StrategyPreset.LONG_STRADDLE:
        # ATM コール買い＋ATM プット買い
        positions.append(
            OptionPosition("call", "long", k_atm, 300.0, 1, contract_size, r, sigma, T_short)
        )
        positions.append(
            OptionPosition("put", "long", k_atm, 300.0, 1, contract_size, r, sigma, T_short)
        )

    elif preset == StrategyPreset.SHORT_STRADDLE:
        # ATM コール売り＋ATM プット売り（高リスク）
        positions.append(
            OptionPosition("call", "short", k_atm, 300.0, 1, contract_size, r, sigma, T_short)
        )
        positions.append(
            OptionPosition("put", "short", k_atm, 300.0, 1, contract_size, r, sigma, T_short)
        )

    elif preset == StrategyPreset.LONG_STRANGLE:
        # OTM コール買い＋OTM プット買い（大きな変動狙い）
        positions.append(
            OptionPosition("call", "long", k_high, 180.0, 1, contract_size, r, sigma, T_short)
        )
        positions.append(
            OptionPosition("put", "long", k_low, 180.0, 1, contract_size, r, sigma, T_short)
        )

    elif preset == StrategyPreset.SHORT_STRANGLE:
        # OTM コール売り＋OTM プット売り（プレミアム取り・高リスク）
        positions.append(
            OptionPosition("call", "short", k_high, 250.0, 1, contract_size, r, sigma, T_short)
        )
        positions.append(
            OptionPosition("put", "short", k_low, 250.0, 1, contract_size, r, sigma, T_short)
        )

    # ---- レンジ相場向け・クレジット系 ----
    elif preset == StrategyPreset.IRON_CONDOR:
        # 下側：プットクレジットスプレッド（内側売り / 外側買い）
        positions.append(
            OptionPosition("put", "short", k_low, 200.0, 1, contract_size, r, sigma, T_short)
        )
        positions.append(
            OptionPosition("put", "long", k_very_low, 100.0, 1, contract_size, r, sigma, T_short)
        )
        # 上側：コールクレジットスプレッド（内側売り / 外側買い）
        positions.append(
            OptionPosition("call", "short", k_high, 200.0, 1, contract_size, r, sigma, T_short)
        )
        positions.append(
            OptionPosition("call", "long", k_very_high, 100.0, 1, contract_size, r, sigma, T_short)
        )

    elif preset == StrategyPreset.IRON_BUTTERFLY:
        # 中央 ATM コール & プットのショート
        positions.append(
            OptionPosition("call", "short", k_atm, 300.0, 1, contract_size, r, sigma, T_short)
        )
        positions.append(
            OptionPosition("put", "short", k_atm, 300.0, 1, contract_size, r, sigma, T_short)
        )
        # 外側のコール買い & プット買い
        positions.append(
            OptionPosition("call", "long", k_high, 150.0, 1, contract_size, r, sigma, T_short)
        )
        positions.append(
            OptionPosition("put", "long", k_low, 150.0, 1, contract_size, r, sigma, T_short)
        )

    # ---- 現物組み合わせ・ヘッジ系（オプション部分のみ）----
    elif preset == StrategyPreset.COVERED_CALL:
        # 本来は「現物ロング + OTM コールショート」
        # ここでは OTM コールショートだけを描画
        positions.append(
            OptionPosition("call", "short", k_high, 200.0, 1, contract_size, r, sigma, T_short)
        )

    elif preset == StrategyPreset.PROTECTIVE_PUT:
        # 本来は「現物ロング + ATM/OTM プットロング」
        positions.append(
            OptionPosition("put", "long", k_low, 200.0, 1, contract_size, r, sigma, T_short)
        )

    elif preset == StrategyPreset.COLLAR:
        # 現物ロング + 下側プットロング + 上側コールショート
        # ここではオプション2本のみ
        positions.append(
            OptionPosition("put", "long", k_low, 150.0, 1, contract_size, r, sigma, T_short)
        )
        positions.append(
            OptionPosition("call", "short", k_high, 150.0, 1, contract_size, r, sigma, T_short)
        )

    elif preset == StrategyPreset.CASH_SECURED_PUT:
        # 現金担保付きプット売り（安値で指値買いの代用）
        positions.append(
            OptionPosition("put", "short", k_low, 250.0, 1, contract_size, r, sigma, T_short)
        )

    # ---- 上級者向け ----
    elif preset == StrategyPreset.PUT_RATIO_SPREAD:
        # 例：高いKプット1枚ロング、低いKプット2枚ショート（3レッグ）
        positions.append(
            OptionPosition("put", "long", k_high, 350.0, 1, contract_size, r, sigma, T_short)
        )
        positions.append(
            OptionPosition("put", "short", k_low, 220.0, 2, contract_size, r, sigma, T_short)
        )

    elif preset == StrategyPreset.CALENDAR_SPREAD:
        # 同一Kで近い限月をショート、遠い限月をロング（2レッグ）
        positions.append(
            OptionPosition("call", "short", k_atm, 200.0, 1, contract_size, r, sigma, T_short)
        )
        positions.append(
            OptionPosition("call", "long", k_atm, 250.0, 1, contract_size, r, sigma, T_long)
        )

    # ---- ジェイド・リザード系 ----
    elif preset == StrategyPreset.JADE_LIZARD:
        # シンプルなジェイド・リザード：
        # プット売り（k_low）＋ショート・コール・スプレッド（k_high〜k_very_high）
        positions.append(
            OptionPosition("put", "short", k_low, 260.0, 1, contract_size, r, sigma, T_short)
        )
        positions.append(
            OptionPosition("call", "short", k_high, 220.0, 1, contract_size, r, sigma, T_short)
        )
        positions.append(
            OptionPosition("call", "long", k_very_high, 120.0, 1, contract_size, r, sigma, T_short)
        )

    elif preset == StrategyPreset.SAFE_JADE_LIZARD:
        # 安全版ジェイド・リザード：
        # ジェイド・リザードに加え、さらに外側のプット買いで下もスプレッド化（両サイド損失限定）
        positions.append(
            OptionPosition("put", "short", k_low, 240.0, 1, contract_size, r, sigma, T_short)
        )
        positions.append(
            OptionPosition("put", "long", k_very_low, 120.0, 1, contract_size, r, sigma, T_short)
        )
        positions.append(
            OptionPosition("call", "short", k_high, 220.0, 1, contract_size, r, sigma, T_short)
        )
        positions.append(
            OptionPosition("call", "long", k_very_high, 120.0, 1, contract_size, r, sigma, T_short)
        )

    return positions


def main():
    st.set_page_config(page_title="オプション戦略 損益グラフ", layout="wide")
    st.title("オプション戦略 損益グラフ（BS 付き）")

    # session_stateの初期化
    if "spot_range" not in st.session_state:
        st.session_state["spot_range"] = 4000.0
    if "auto_range_enabled" not in st.session_state:
        st.session_state["auto_range_enabled"] = False

    # --- サイドバー: 原資産 & レンジ設定 ---
    st.sidebar.header("原資産レンジ設定")

    default_ticker = "^N225"
    ticker = st.sidebar.text_input(
        "原資産ティッカー（例: ^N225, 7203.T）",
        value=default_ticker
    )

    auto_fetch = st.sidebar.checkbox("ティッカーから中心価格を自動取得する", value=True)

    if auto_fetch:
        last_close = get_last_close(ticker)
        if last_close is not None:
            spot_center = st.sidebar.number_input(
                "中心となる原資産価格",
                value=float(round(last_close, -1)),
                step=100.0
            )
            st.sidebar.caption(f"yfinance 終値: 約 {last_close:.2f}")
        else:
            st.sidebar.warning("価格を取得できませんでした。手動で入力してください。")
            spot_center = st.sidebar.number_input(
                "中心となる原資産価格",
                value=32000.0,
                step=100.0
            )
    else:
        spot_center = st.sidebar.number_input(
            "中心となる原資産価格",
            value=32000.0,
            step=100.0
        )

    # 横軸範囲コントロール（拡張版）
    st.sidebar.subheader("横軸表示範囲")
    
    # 自動調整ボタン
    if st.sidebar.button("🎯 全体表示（自動調整）"):
        st.session_state["auto_range_enabled"] = True
    
    # クイック範囲選択ボタン
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("±5%"):
            st.session_state["spot_range"] = spot_center * 0.05
            st.session_state["auto_range_enabled"] = False
        if st.button("±10%"):
            st.session_state["spot_range"] = spot_center * 0.10
            st.session_state["auto_range_enabled"] = False
    with col2:
        if st.button("±20%"):
            st.session_state["spot_range"] = spot_center * 0.20
            st.session_state["auto_range_enabled"] = False
        if st.button("±35%"):
            st.session_state["spot_range"] = spot_center * 0.35
            st.session_state["auto_range_enabled"] = False
    
    spot_range = st.sidebar.number_input(
        "上下レンジ幅（±）",
        value=float(st.session_state["spot_range"]),
        step=500.0,
        min_value=100.0
    )
    st.session_state["spot_range"] = spot_range
    
    st.sidebar.caption(f"表示範囲: {spot_center - spot_range:,.0f} 〜 {spot_center + spot_range:,.0f}")
    
    spot_step = st.sidebar.number_input(
        "刻み幅（グラフ用の S の刻み）",
        value=100.0,
        step=50.0
    )

    spot_min = spot_center - spot_range
    spot_max = spot_center + spot_range
    spots = np.arange(spot_min, spot_max + spot_step, spot_step)

    # --- サイドバー: 共通パラメータ ---
    st.sidebar.header("共通パラメータ")

    default_contract_size = guess_contract_size_from_ticker(ticker)
    contract_size = st.sidebar.number_input(
        "取引単位（1枚あたりの原資産数量）",
        value=default_contract_size,
        step=10
    )

    r = st.sidebar.number_input(
        "無リスク金利 r（年率, 例: 0.0075 = 0.75%）",
        value=0.0075,
        step=0.001,
        format="%.4f"
    )

    # --- サイドバー: 戦略プリセット ---
    st.sidebar.header("戦略プリセット")
    preset = st.sidebar.selectbox(
        "代表的な戦略を選択（任意）",
        options=list(StrategyPreset),
        format_func=lambda x: x.value
    )

    # 前回プリセットとの比較（変更検知）
    prev_preset = st.session_state.get("prev_preset")
    preset_changed = (prev_preset != preset)
    st.session_state["prev_preset"] = preset

    # --- サイドバー: レッグ数 ---
    st.sidebar.header("レッグ数")

    if preset == StrategyPreset.NONE:
        n_legs = st.sidebar.slider(
            "ポジションの数",
            min_value=1,
            max_value=6,   # レシオなども考慮して少し広げておく
            value=2
        )
        preset_positions: List[OptionPosition] | None = None
    else:
        preset_positions = build_positions_from_preset(
            preset=preset,
            spot_center=spot_center,
            contract_size=contract_size,
            r=r,
            ticker=ticker,
        )
        n_legs = len(preset_positions)
        st.sidebar.caption(f"この戦略のレッグ数: {n_legs}")

    # プリセット内容を dict に展開
    preset_dict: dict[int, OptionPosition] = {}
    if preset_positions is not None:
        for idx, p in enumerate(preset_positions, start=1):
            preset_dict[idx] = p

    # --- K の初期化ロジック（ティッカー or 中心価格変更時）---
    prev_ticker = st.session_state.get("prev_ticker")
    prev_spot_center = st.session_state.get("prev_spot_center")

    ticker_or_center_changed = (prev_ticker != ticker) or (prev_spot_center != spot_center)
    if ticker_or_center_changed:
        st.session_state["prev_ticker"] = ticker
        st.session_state["prev_spot_center"] = spot_center

        strike_step_for_init = guess_strike_step_from_ticker(ticker)
        default_strike_for_init = round_to_step(spot_center, strike_step_for_init)

        for i in range(1, n_legs + 1):
            st.session_state[f"strike_{i}"] = default_strike_for_init

    # --- プリセット変更時: session_state を上書きしてレッグを同期 ---
    if preset_changed and preset_positions is not None:
        for i in range(1, n_legs + 1):
            base = preset_dict.get(i)
            if base is None:
                continue
            st.session_state[f"type_{i}"] = base.option_type
            st.session_state[f"side_{i}"] = base.side
            st.session_state[f"strike_{i}"] = base.strike
            st.session_state[f"premium_{i}"] = base.premium
            st.session_state[f"qty_{i}"] = base.quantity
            st.session_state[f"sigma_{i}"] = base.sigma
            st.session_state[f"Tdays_{i}"] = int(base.T * 365)

    positions: List[OptionPosition] = []

    # --- pending strikeの反映（widgetインスタンス化の前に実行）---
    for i in range(1, n_legs + 1):
        pending_key = f"_pending_strike_{i}"
        state_key = f"strike_{i}"
        if pending_key in st.session_state:
            st.session_state[state_key] = st.session_state[pending_key]
            del st.session_state[pending_key]

    # --- メイン: 各レッグ入力 ---
    st.subheader("ポジション入力")

    # ストライク価格の視覚的表示
    if n_legs > 0:
        st.markdown("#### ストライク価格の分布")
        strike_info = []
        for i in range(1, n_legs + 1):
            state_key = f"strike_{i}"
            if state_key in st.session_state:
                strike_val = st.session_state[state_key]
                type_key = f"type_{i}"
                side_key = f"side_{i}"
                opt_type = st.session_state.get(type_key, "call")
                side = st.session_state.get(side_key, "long")
                strike_info.append((strike_val, opt_type, side, i))
        
        if strike_info:
            strike_info.sort(key=lambda x: x[0])
            cols = st.columns(len(strike_info))
            for idx, (strike_val, opt_type, side, leg_num) in enumerate(strike_info):
                with cols[idx]:
                    color = "🟢" if opt_type == "call" else "🔴"
                    style = "買" if side == "long" else "売"
                    st.metric(
                        label=f"レッグ {leg_num}",
                        value=f"{strike_val:,.0f}",
                        delta=f"{color} {opt_type.upper()} {style}"
                    )

    for i in range(1, n_legs + 1):
        with st.expander(f"レッグ {i} の設定", expanded=(i == 1)):
            col1, col2, col3 = st.columns(3)
            with col1:
                opt_type = st.selectbox(
                    "種類",
                    options=["call", "put"],
                    key=f"type_{i}"
                )
            with col2:
                side = st.selectbox(
                    "売買",
                    options=["long", "short"],
                    key=f"side_{i}"
                )
            with col3:
                strike_step = guess_strike_step_from_ticker(ticker)
                state_key = f"strike_{i}"
                
                # 初期化
                if state_key not in st.session_state:
                    st.session_state[state_key] = round_to_step(spot_center, strike_step)

                strike = st.number_input(
                    f"行使価格 K（刻み {strike_step:.0f} 円）",
                    value=st.session_state[state_key],
                    step=strike_step,
                    key=state_key
                )

            col4, col5, col6 = st.columns(3)
            with col4:
                premium = st.number_input(
                    "プレミアム（1枚あたりの金額, 常に +）",
                    value=300.0,
                    step=10.0,
                    key=f"premium_{i}"
                )
            with col5:
                quantity = st.number_input(
                    "枚数",
                    min_value=1,
                    max_value=2000,
                    value=1,
                    step=1,
                    key=f"qty_{i}"
                )
            with col6:
                sigma = st.number_input(
                    "ボラティリティ σ（年率）",
                    value=0.2,
                    step=0.05,
                    format="%.3f",
                    key=f"sigma_{i}"
                )

            T_days = st.number_input(
                "残存日数（日）",
                value=30,
                step=1,
                key=f"Tdays_{i}"
            )
            T = T_days / 365.0

            positions.append(
                OptionPosition(
                    option_type=opt_type,
                    side=side,
                    strike=strike,
                    premium=premium,
                    quantity=quantity,
                    contract_size=contract_size,
                    r=r,
                    sigma=sigma,
                    T=T
                )
            )

    st.markdown("---")

    # --- ストライクダイヤル（全レッグ統合版）---
    st.subheader("🎯 ストライク価格ダイヤル調整")
    st.caption("スライダーで各レッグのストライク価格を視覚的に調整できます")
    
    strike_step = guess_strike_step_from_ticker(ticker)
    dial_min = spot_center * 0.7
    dial_max = spot_center * 1.3
    
    for i in range(1, n_legs + 1):
        state_key = f"strike_{i}"
        if state_key not in st.session_state:
            continue
            
        col_label, col_slider, col_buttons = st.columns([1, 3, 2])
        
        with col_label:
            opt_type = st.session_state.get(f"type_{i}", "call")
            side = st.session_state.get(f"side_{i}", "long")
            color = "🟢" if opt_type == "call" else "🔴"
            style = "買" if side == "long" else "売"
            st.markdown(f"**レッグ {i}**  \n{color} {opt_type.upper()} {style}")
        
        with col_slider:
            # ダイヤル専用のキーを使用
            dial_key = f"strike_dial_{i}"
            current_strike = st.session_state.get(state_key, spot_center)
            
            # on_changeコールバックで更新
            def update_strike_from_dial(leg_num=i):
                dial_k = f"strike_dial_{leg_num}"
                strike_k = f"strike_{leg_num}"
                if dial_k in st.session_state:
                    # 次回のレンダリング時に反映されるように別キーに保存
                    st.session_state[f"_pending_strike_{leg_num}"] = st.session_state[dial_k]
            
            st.slider(
                f"ストライク調整 (現在: {current_strike:,.0f})",
                min_value=dial_min,
                max_value=dial_max,
                value=float(current_strike),
                step=strike_step,
                key=dial_key,
                format="%.0f",
                label_visibility="collapsed",
                on_change=update_strike_from_dial,
                args=(i,)
            )
        
        with col_buttons:
            col_atm, col_otm_low, col_otm_high = st.columns(3)
            
            def set_strike_value(leg_num, multiplier):
                new_val = round_to_step(spot_center * multiplier, strike_step)
                st.session_state[f"_pending_strike_{leg_num}"] = new_val
                st.session_state[f"strike_dial_{leg_num}"] = new_val
            
            with col_atm:
                st.button("ATM", key=f"atm_{i}", help="At The Money",
                         on_click=set_strike_value, args=(i, 1.0))
            with col_otm_low:
                st.button("-5%", key=f"otm_low_{i}", help="5%下のOTM",
                         on_click=set_strike_value, args=(i, 0.95))
            with col_otm_high:
                st.button("+5%", key=f"otm_high_{i}", help="5%上のOTM",
                         on_click=set_strike_value, args=(i, 1.05))

    st.markdown("---")

    # --- グラフ描画 ---
    if st.button("グラフを更新", type="primary"):
        if not positions:
            st.warning("少なくとも 1 つポジションを入力してください。")
            return

        # 自動範囲調整が有効な場合
        if st.session_state.get("auto_range_enabled", False):
            strikes = [p.strike for p in positions]
            if strikes:
                strike_step = guess_strike_step_from_ticker(ticker)
                max_dist = max([abs(k - spot_center) for k in strikes] + [strike_step * 2])
                auto_range = float(round_to_step(max_dist * 1.5, strike_step))
                auto_range = max(auto_range, strike_step * 3)
                st.session_state["spot_range"] = auto_range
                spot_range = auto_range
                spots = np.arange(spot_center - spot_range, spot_center + spot_range + spot_step, spot_step)
                st.session_state["auto_range_enabled"] = False
                st.rerun()

        total_now = total_pl_now_bs(positions, spots)
        total_maturity = total_payoff_maturity(positions, spots)

        max_now = float(np.max(total_now))
        min_now = float(np.min(total_now))
        max_mat = float(np.max(total_maturity))
        min_mat = float(np.min(total_maturity))

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=spots,
                y=total_now,
                mode="lines",
                name="Total P/L Now (BS)",
                line=dict(color="orange", width=3)
            )
        )

        fig.add_trace(
            go.Scatter(
                x=spots,
                y=total_maturity,
                mode="lines",
                name="Total P/L at Maturity",
                line=dict(color="blue", width=2, dash="dash")
            )
        )

        fig.add_hline(y=0, line_color="gray", line_width=1)
        fig.add_vline(x=spot_center, line_color="green", line_width=1, line_dash="dot")

        # ストライク価格に縦線とタグを追加
        for pos in positions:
            bgcolor, textcolor = pos.get_tag_color()
            label_text = f"{pos.option_type[0].upper()}{pos.strike:,.0f}"
            if pos.side == "short":
                label_text += "売"
            else:
                label_text += "買"
            
            fig.add_vline(
                x=pos.strike,
                line_color=textcolor if pos.side == "long" else bgcolor,
                line_width=1,
                line_dash="dot",
                opacity=0.5
            )
            
            # ストライクラベルをアノテーションで追加
            fig.add_annotation(
                x=pos.strike,
                y=max_now * 0.9,
                text=label_text,
                showarrow=False,
                bgcolor=bgcolor,
                bordercolor=textcolor,
                font=dict(color=textcolor, size=10),
                borderwidth=2,
                borderpad=4
            )

        fig.add_hline(
            y=max_now,
            line_color="orange",
            line_width=1,
            line_dash="dot",
            annotation_text=f"最大利益(期中) ≈ {max_now:,.0f}",
            annotation_position="top left"
        )
        fig.add_hline(
            y=min_now,
            line_color="orange",
            line_width=1,
            line_dash="dot",
            annotation_text=f"最大損失(期中) ≈ {min_now:,.0f}",
            annotation_position="bottom left"
        )

        fig.add_hline(
            y=max_mat,
            line_color="blue",
            line_width=1,
            line_dash="dot",
            annotation_text=f"最大利益(満期) ≈ {max_mat:,.0f}",
            annotation_position="top right"
        )
        fig.add_hline(
            y=min_mat,
            line_color="blue",
            line_width=1,
            line_dash="dot",
            annotation_text=f"最大損失(満期) ≈ {min_mat:,.0f}",
            annotation_position="bottom right"
        )

        fig.update_layout(
            title=f"オプション戦略 損益グラフ（原資産: {ticker}）",
            xaxis_title="原資産価格 S",
            yaxis_title="損益",
            template="plotly_white",
            legend=dict(x=0.01, y=0.99),
            height=600
        )

        st.plotly_chart(fig, use_container_width=True)

        # サマリー情報を見やすく表示
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("最大利益(期中)", f"{max_now:,.0f}")
        with col2:
            st.metric("最大損失(期中)", f"{min_now:,.0f}")
        with col3:
            st.metric("最大利益(満期)", f"{max_mat:,.0f}")
        with col4:
            st.metric("最大損失(満期)", f"{min_mat:,.0f}")

        st.caption(
            "🟠 オレンジ実線: 期中（BS 理論価格） / "
            "🔵 青破線: 満期損益 / 🟢 緑点線: 中心価格 / "
            "⚪ 点線: ストライク価格"
        )


if __name__ == "__main__":
    main()
