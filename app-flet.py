# app-flet.py
# オプション戦略 損益グラフ（BS 付き）＋ yfinance 原資産
# Flet版

import io
import math
import os
import platform
import tempfile
from dataclasses import dataclass
from enum import Enum
from typing import List, Literal

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from scipy.stats import norm
import yfinance as yf
import flet as ft

import re
import unicodedata

OptionType = Literal["call", "put"]
SideType = Literal["long", "short"]


def safe_float(val, default=0.0) -> float:
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
    s = unicodedata.normalize("NFKC", s)
    s = s.replace(",", "").replace("¥", "").replace("\\", "").replace(" ", "")
    try:
        return float(s)
    except (ValueError, TypeError):
        cleaned = re.sub(r"[^0-9.-]", "", s)
        if cleaned.count(".") > 1:
            parts = cleaned.split(".")
            cleaned = parts[0] + "." + "".join(parts[1:])
        try:
            return float(cleaned)
        except (ValueError, TypeError):
            return def_val


def safe_int(val, default=1) -> int:
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
    side: SideType
    strike: float
    premium: float
    quantity: int = 1
    contract_size: int = 1
    r: float = 0.0
    sigma: float = 0.2
    T: float = 0.1

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


class StrategyPreset(str, Enum):
    NONE = "手動で設定する"
    LONG_CALL = "ロング・コール（強気）"
    SHORT_CALL = "ショート・コール（弱気〜横ばい）"
    LONG_PUT = "ロング・プット（弱気）"
    SHORT_PUT = "ショート・プット（強気〜横ばい）"
    BULL_CALL_SPREAD = "ブル・コール・スプレッド（緩やかな強気）"
    BEAR_PUT_SPREAD = "ベア・プット・スプレッド（緩やかな弱気）"
    BULL_PUT_SPREAD = "ブル・プット・スプレッド（クレジット・強気〜横ばい）"
    BEAR_CALL_SPREAD = "ベア・コール・スプレッド（クレジット・弱気〜横ばい）"
    LONG_STRADDLE = "ロング・ストラドル（変動大きめ予想）"
    SHORT_STRADDLE = "ショート・ストラドル（変動小さめ予想・高リスク）"
    LONG_STRANGLE = "ロング・ストラングル（大化け・急変狙い、低コスト）"
    SHORT_STRANGLE = "ショート・ストラングル（買いなし・上級者向け）"
    IRON_CONDOR = "アイアン・コンドル（広めレンジでコツコツ）"
    IRON_BUTTERFLY = "アイアン・バタフライ（狭いレンジをピンポイント）"
    COVERED_CALL = "カバード・コール（現物保有＋コール売り）"
    PROTECTIVE_PUT = "プロテクティブ・プット（現物下落の保険）"
    COLLAR = "カラー（低コストの現物保全）"
    CASH_SECURED_PUT = "キャッシュ・セキュアード・プット（安値での指値買い代用）"
    PUT_RATIO_SPREAD = "プット・レシオ・スプレッド（限定的な下落〜横ばい）"
    CALENDAR_SPREAD = "カレンダー・スプレッド（満期ズレ・時間価値狙い）"


def resolve_strategy_preset(val: str | StrategyPreset | None) -> StrategyPreset:
    if not val:
        return StrategyPreset.NONE
    if isinstance(val, StrategyPreset):
        return val
    # メンバー名で検索 (例: "LONG_CALL")
    if val in StrategyPreset.__members__:
        return StrategyPreset[val]
    # 値で検索 (例: "ロング・コール（強気）")
    for p in StrategyPreset:
        if p.value == val or p.name == val:
            return p
    return StrategyPreset.NONE


def build_positions_from_preset(
    preset: StrategyPreset | str,
    spot_center: float,
    contract_size: int,
    r: float,
    ticker: str,
) -> List[OptionPosition]:
    preset = resolve_strategy_preset(preset)
    positions: List[OptionPosition] = []

    sigma = 0.2
    T_short = 30 / 365
    T_long = 90 / 365

    step = guess_strike_step_from_ticker(ticker)
    k_atm = round_to_step(spot_center, step)
    k_low = round_to_step(spot_center * 0.95, step)
    k_high = round_to_step(spot_center * 1.05, step)
    k_very_low = round_to_step(spot_center * 0.9, step)
    k_very_high = round_to_step(spot_center * 1.1, step)

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
    elif preset == StrategyPreset.BULL_CALL_SPREAD:
        positions.append(
            OptionPosition("call", "long", k_low, 400.0, 1, contract_size, r, sigma, T_short)
        )
        positions.append(
            OptionPosition("call", "short", k_high, 200.0, 1, contract_size, r, sigma, T_short)
        )
    elif preset == StrategyPreset.BEAR_PUT_SPREAD:
        positions.append(
            OptionPosition("put", "long", k_high, 400.0, 1, contract_size, r, sigma, T_short)
        )
        positions.append(
            OptionPosition("put", "short", k_low, 200.0, 1, contract_size, r, sigma, T_short)
        )
    elif preset == StrategyPreset.BULL_PUT_SPREAD:
        positions.append(
            OptionPosition("put", "short", k_high, 400.0, 1, contract_size, r, sigma, T_short)
        )
        positions.append(
            OptionPosition("put", "long", k_low, 200.0, 1, contract_size, r, sigma, T_short)
        )
    elif preset == StrategyPreset.BEAR_CALL_SPREAD:
        positions.append(
            OptionPosition("call", "short", k_low, 400.0, 1, contract_size, r, sigma, T_short)
        )
        positions.append(
            OptionPosition("call", "long", k_high, 200.0, 1, contract_size, r, sigma, T_short)
        )
    elif preset == StrategyPreset.LONG_STRADDLE:
        positions.append(
            OptionPosition("call", "long", k_atm, 300.0, 1, contract_size, r, sigma, T_short)
        )
        positions.append(
            OptionPosition("put", "long", k_atm, 300.0, 1, contract_size, r, sigma, T_short)
        )
    elif preset == StrategyPreset.SHORT_STRADDLE:
        positions.append(
            OptionPosition("call", "short", k_atm, 300.0, 1, contract_size, r, sigma, T_short)
        )
        positions.append(
            OptionPosition("put", "short", k_atm, 300.0, 1, contract_size, r, sigma, T_short)
        )
    elif preset == StrategyPreset.LONG_STRANGLE:
        positions.append(
            OptionPosition("call", "long", k_high, 180.0, 1, contract_size, r, sigma, T_short)
        )
        positions.append(
            OptionPosition("put", "long", k_low, 180.0, 1, contract_size, r, sigma, T_short)
        )
    elif preset == StrategyPreset.SHORT_STRANGLE:
        positions.append(
            OptionPosition("call", "short", k_high, 250.0, 1, contract_size, r, sigma, T_short)
        )
        positions.append(
            OptionPosition("put", "short", k_low, 250.0, 1, contract_size, r, sigma, T_short)
        )
    elif preset == StrategyPreset.IRON_CONDOR:
        positions.append(
            OptionPosition("put", "short", k_low, 200.0, 1, contract_size, r, sigma, T_short)
        )
        positions.append(
            OptionPosition("put", "long", k_very_low, 100.0, 1, contract_size, r, sigma, T_short)
        )
        positions.append(
            OptionPosition("call", "short", k_high, 200.0, 1, contract_size, r, sigma, T_short)
        )
        positions.append(
            OptionPosition("call", "long", k_very_high, 100.0, 1, contract_size, r, sigma, T_short)
        )
    elif preset == StrategyPreset.IRON_BUTTERFLY:
        positions.append(
            OptionPosition("call", "short", k_atm, 300.0, 1, contract_size, r, sigma, T_short)
        )
        positions.append(
            OptionPosition("put", "short", k_atm, 300.0, 1, contract_size, r, sigma, T_short)
        )
        positions.append(
            OptionPosition("call", "long", k_high, 150.0, 1, contract_size, r, sigma, T_short)
        )
        positions.append(
            OptionPosition("put", "long", k_low, 150.0, 1, contract_size, r, sigma, T_short)
        )
    elif preset == StrategyPreset.COVERED_CALL:
        positions.append(
            OptionPosition("call", "short", k_high, 200.0, 1, contract_size, r, sigma, T_short)
        )
    elif preset == StrategyPreset.PROTECTIVE_PUT:
        positions.append(
            OptionPosition("put", "long", k_low, 200.0, 1, contract_size, r, sigma, T_short)
        )
    elif preset == StrategyPreset.COLLAR:
        positions.append(
            OptionPosition("put", "long", k_low, 150.0, 1, contract_size, r, sigma, T_short)
        )
        positions.append(
            OptionPosition("call", "short", k_high, 150.0, 1, contract_size, r, sigma, T_short)
        )
    elif preset == StrategyPreset.CASH_SECURED_PUT:
        positions.append(
            OptionPosition("put", "short", k_low, 250.0, 1, contract_size, r, sigma, T_short)
        )
    elif preset == StrategyPreset.PUT_RATIO_SPREAD:
        positions.append(
            OptionPosition("put", "long", k_high, 350.0, 1, contract_size, r, sigma, T_short)
        )
        positions.append(
            OptionPosition("put", "short", k_low, 220.0, 2, contract_size, r, sigma, T_short)
        )
    elif preset == StrategyPreset.CALENDAR_SPREAD:
        positions.append(
            OptionPosition("call", "short", k_atm, 200.0, 1, contract_size, r, sigma, T_short)
        )
        positions.append(
            OptionPosition("call", "long", k_atm, 250.0, 1, contract_size, r, sigma, T_long)
        )

    return positions


def main(page: ft.Page):
    page.title = "オプション戦略 損益グラフ（BS 付き）"
    page.scroll = "adaptive"
    page.padding = 20

    # 状態管理
    state = {
        "ticker": "^N225",
        "spot_center": 32000.0,
        "spot_range": 4000.0,
        "spot_step": 100.0,
        "contract_size": 1000,
        "r": 0.0075,
        "preset": StrategyPreset.NONE,
        "n_legs": 2,
        "legs": {},
        "prev_ticker": None,
        "prev_spot_center": None,
        "prev_preset": None,
    }

    # UI要素の参照
    ticker_field = ft.TextField(
        label="原資産ティッカー（例: ^N225, 7203.T）",
        value=state["ticker"],
        width=270
    )

    auto_fetch_check = ft.Checkbox(
        label="ティッカーから中心価格を自動取得する",
        value=True
    )

    spot_center_field = ft.TextField(
        label="中心となる原資産価格",
        value=str(state["spot_center"]),
        keyboard_type=ft.KeyboardType.NUMBER,
        width=200
    )

    yfinance_info = ft.Text("")

    spot_range_field = ft.TextField(
        label="上下レンジ幅",
        value=str(state["spot_range"]),
        keyboard_type=ft.KeyboardType.NUMBER,
        width=200
    )

    spot_step_field = ft.TextField(
        label="刻み幅",
        value=str(state["spot_step"]),
        keyboard_type=ft.KeyboardType.NUMBER,
        width=200
    )

    contract_size_field = ft.TextField(
        label="取引単位",
        value=str(state["contract_size"]),
        keyboard_type=ft.KeyboardType.NUMBER,
        width=200
    )

    r_field = ft.TextField(
        label="無リスク金利 r（年率）",
        value=str(state["r"]),
        keyboard_type=ft.KeyboardType.NUMBER,
        width=200
    )

    legs_slider = ft.Slider(
        min=1,
        max=6,
        divisions=5,
        value=2,
        label="{value}",
        width=300
    )

    legs_container = ft.Column()
    chart_container = ft.Column()
    results_text = ft.Text("")

    # 横軸範囲コントロール用UI
    range_info_text = ft.Text(
        f"表示範囲: {state['spot_center'] - state['spot_range']:,.0f} 〜 {state['spot_center'] + state['spot_range']:,.0f}（中心: {state['spot_center']:,.0f} ±{state['spot_range']:,.0f}）",
        size=14,
        weight=ft.FontWeight.BOLD,
        color=ft.Colors.BLUE_GREY_700,
    )

    range_slider = ft.Slider(
        min=10,
        max=50000,
        value=state["spot_range"],
        label="{value}",
        expand=True,
    )

    def update_range_slider_bounds(target_val: float):
        # 目標レンジ幅に合わせて min と max を動的に適正化
        cur_min = min(float(range_slider.min), target_val * 0.2, 50.0)
        cur_min = max(5.0, round(cur_min / 5.0) * 5.0)
        cur_max = max(float(range_slider.max), target_val * 2.5, 1000.0)
        range_slider.min = cur_min
        range_slider.max = cur_max
        range_slider.value = max(cur_min, min(target_val, cur_max))

    def update_range_display():
        center = state["spot_center"]
        rng = state["spot_range"]
        range_info_text.value = f"表示範囲: {center - rng:,.0f} 〜 {center + rng:,.0f}（中心: {center:,.0f} ±{rng:,.0f}）"

    def set_auto_range(e=None):
        update_state_from_fields()
        strikes = []
        for leg in state["legs"].values():
            if "strike" in leg:
                try:
                    strikes.append(float(leg["strike"]))
                except (ValueError, TypeError):
                    pass
        center = state["spot_center"]
        strike_step = guess_strike_step_from_ticker(state["ticker"])
        if strikes:
            max_dist = max([abs(k - center) for k in strikes] + [strike_step * 2])
            auto_range = float(round_to_step(max_dist * 1.5, strike_step))
            auto_range = max(auto_range, strike_step * 3)
        else:
            auto_range = max(float(round_to_step(center * 0.15, strike_step)), strike_step * 5)

        state["spot_range"] = auto_range
        spot_range_field.value = str(auto_range)
        update_range_slider_bounds(auto_range)
        update_range_display()
        generate_chart()

    def set_quick_range(val: float):
        update_state_from_fields()
        state["spot_range"] = val
        spot_range_field.value = str(val)
        update_range_slider_bounds(val)
        update_range_display()
        generate_chart()

    def set_quick_range_percent(pct: float):
        update_state_from_fields()
        center = state["spot_center"]
        strike_step = guess_strike_step_from_ticker(state["ticker"])
        val = float(round_to_step(center * (pct / 100.0), strike_step))
        val = max(val, strike_step * 2)
        set_quick_range(val)

    def on_range_slider_change(e):
        val = float(e.control.value)
        state["spot_range"] = val
        spot_range_field.value = str(val)
        update_range_display()
        generate_chart()

    range_slider.on_change = on_range_slider_change

    range_control_card = ft.Card(
        content=ft.Container(
            content=ft.Column([
                ft.Row([
                    ft.Row([
                        ft.Icon(ft.Icons.AUTO_GRAPH, color=ft.Colors.PRIMARY),
                        ft.Text("横軸表示範囲（原資産価格 S）", weight=ft.FontWeight.BOLD, size=16),
                    ]),
                    range_info_text,
                ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                ft.Row([
                    ft.Text("レンジ幅: ±"),
                    range_slider,
                ]),
                ft.Row([
                    ft.FilledTonalButton(
                        "全体表示（自動調整）",
                        icon=ft.Icons.FIT_SCREEN,
                        on_click=set_auto_range
                    ),
                    ft.OutlinedButton("±5%", on_click=lambda e: set_quick_range_percent(5)),
                    ft.OutlinedButton("±10%", on_click=lambda e: set_quick_range_percent(10)),
                    ft.OutlinedButton("±20%", on_click=lambda e: set_quick_range_percent(20)),
                    ft.OutlinedButton("±35%", on_click=lambda e: set_quick_range_percent(35)),
                ], wrap=True),
            ]),
            padding=15
        )
    )

    def update_state_from_fields():
        state["ticker"] = ticker_field.value or "^N225"
        try:
            state["spot_center"] = float(spot_center_field.value)
        except (ValueError, TypeError):
            pass
        try:
            state["spot_range"] = float(spot_range_field.value)
        except (ValueError, TypeError):
            pass
        try:
            state["spot_step"] = float(spot_step_field.value)
        except (ValueError, TypeError):
            pass
        try:
            state["contract_size"] = int(float(contract_size_field.value))
        except (ValueError, TypeError):
            pass
        try:
            state["r"] = float(r_field.value)
        except (ValueError, TypeError):
            pass

    def update_ticker_info(e=None, force_k_update=True):
        state["ticker"] = ticker_field.value or "^N225"
        old_center = state.get("spot_center", 32000.0)

        # 1. yfinance から中心価格を取得
        if auto_fetch_check.value:
            last_close = get_last_close(state["ticker"])
            if last_close is not None:
                new_center = float(round(last_close, -1) if last_close > 1000 else round(last_close, 1))
                state["spot_center"] = new_center
                spot_center_field.value = str(state["spot_center"])
                yfinance_info.value = f"yfinance 終値: 約 {last_close:,.2f}"
            else:
                yfinance_info.value = "価格を取得できませんでした"
        else:
            try:
                state["spot_center"] = float(spot_center_field.value)
            except (ValueError, TypeError):
                pass

        center = state["spot_center"]

        # 2. 取引単位 & 刻み幅の推定
        state["contract_size"] = guess_contract_size_from_ticker(state["ticker"])
        contract_size_field.value = str(state["contract_size"])

        strike_step = guess_strike_step_from_ticker(state["ticker"])
        state["spot_step"] = strike_step
        spot_step_field.value = str(strike_step)

        # 3. レンジ幅の適正化（価格水準が大幅に変化した場合や、マイナスになる場合に自動調整）
        suggested_range = float(round_to_step(center * 0.15, strike_step))
        suggested_range = max(suggested_range, strike_step * 5)
        if state["spot_range"] >= center or abs(center - old_center) > (old_center * 0.1):
            state["spot_range"] = suggested_range
            spot_range_field.value = str(suggested_range)
            update_range_slider_bounds(suggested_range)
        else:
            update_range_slider_bounds(state["spot_range"])

        # 4. 行使価格 K の自動変更
        if force_k_update:
            if state.get("preset", StrategyPreset.NONE) != StrategyPreset.NONE:
                # 戦略プリセットが選択されている場合: 新しい原資産価格で戦略の全レッグを再構築
                preset_positions = build_positions_from_preset(
                    preset=state["preset"],
                    spot_center=center,
                    contract_size=state["contract_size"],
                    r=state["r"],
                    ticker=state["ticker"],
                )
                state["n_legs"] = len(preset_positions)
                legs_slider.value = state["n_legs"]
                state["legs"] = {}
                for i, pos in enumerate(preset_positions, start=1):
                    state["legs"][i] = {
                        "type": pos.option_type,
                        "side": pos.side,
                        "strike": pos.strike,
                        "premium": pos.premium,
                        "quantity": pos.quantity,
                        "sigma": pos.sigma,
                        "T_days": int(pos.T * 365)
                    }
            else:
                # 手動設定の場合: 新しい原資産価格に合わせて各行使価格をシフト
                ratio = center / old_center if old_center > 0 else 1.0
                atm_strike = round_to_step(center, strike_step)
                for i in range(1, state["n_legs"] + 1):
                    if i in state["legs"]:
                        old_k = state["legs"][i].get("strike", old_center)
                        if old_center > 0 and abs(ratio - 1.0) > 0.03:
                            # 3%以上の価格変化があれば比率でシフトして刻みに丸める
                            state["legs"][i]["strike"] = round_to_step(old_k * ratio, strike_step)
                        elif "strike" not in state["legs"][i]:
                            state["legs"][i]["strike"] = atm_strike
                    else:
                        state["legs"][i] = {
                            "type": "call",
                            "side": "long",
                            "strike": atm_strike,
                            "premium": 300.0 if center > 10000 else float(round_to_step(center * 0.02, 1.0)),
                            "quantity": 1,
                            "sigma": 0.2,
                            "T_days": 30
                        }

        # 5. UI再構築 & レンジ表示更新 & グラフ再描画
        build_legs_ui()
        update_range_display()
        generate_chart()

    def on_preset_change(e=None):
        val = None
        if e is not None:
            if hasattr(e, "control") and e.control and hasattr(e.control, "value") and e.control.value:
                val = e.control.value
            elif hasattr(e, "data") and e.data:
                val = e.data
        if not val and "preset_dropdown" in locals():
            val = preset_dropdown.value

        selected_preset = resolve_strategy_preset(val)
        state["preset"] = selected_preset
        
        if selected_preset != StrategyPreset.NONE:
            update_state_from_fields()
            preset_positions = build_positions_from_preset(
                preset=selected_preset,
                spot_center=state["spot_center"],
                contract_size=state["contract_size"],
                r=state["r"],
                ticker=state["ticker"],
            )
            state["n_legs"] = len(preset_positions)
            legs_slider.value = state["n_legs"]
            
            # レッグデータを初期化
            state["legs"] = {}
            for i, pos in enumerate(preset_positions, start=1):
                state["legs"][i] = {
                    "type": pos.option_type,
                    "side": pos.side,
                    "strike": pos.strike,
                    "premium": pos.premium,
                    "quantity": pos.quantity,
                    "sigma": pos.sigma,
                    "T_days": int(pos.T * 365)
                }
            
            # UIを再構築して全体が収まる自動レンジで再描画
            build_legs_ui()
            set_auto_range()
        else:
            page.update()

    preset_dropdown = ft.Dropdown(
        label="戦略プリセット",
        options=[
            ft.dropdown.Option(key=p.name, text=p.value)
            for p in StrategyPreset
        ],
        value=StrategyPreset.NONE.name,
        width=400,
    )

    def on_legs_change(e):
        state["n_legs"] = int(legs_slider.value)
        build_legs_ui()
        page.update()

    def build_legs_ui():
        legs_container.controls.clear()
        
        for i in range(1, state["n_legs"] + 1):
            if i not in state["legs"]:
                strike_step = guess_strike_step_from_ticker(state["ticker"])
                state["legs"][i] = {
                    "type": "call",
                    "side": "long",
                    "strike": round_to_step(state["spot_center"], strike_step),
                    "premium": 300.0,
                    "quantity": 1,
                    "sigma": 0.2,
                    "T_days": 30
                }
            
            leg_data = state["legs"][i]
            
            type_dropdown = ft.Dropdown(
                label="種類",
                options=[
                    ft.dropdown.Option("call", "Call"),
                    ft.dropdown.Option("put", "Put"),
                ],
                value=leg_data["type"],
                width=150,
            )
            type_handler = lambda e, idx=i: update_leg(idx, "type", e.control.value if (e and hasattr(e, "control") and e.control) else type_dropdown.value)
            for ev in ["on_select", "on_change", "on_selected"]:
                if hasattr(type_dropdown, ev):
                    setattr(type_dropdown, ev, type_handler)
            
            side_dropdown = ft.Dropdown(
                label="売買",
                options=[
                    ft.dropdown.Option("long", "Long"),
                    ft.dropdown.Option("short", "Short"),
                ],
                value=leg_data["side"],
                width=150,
            )
            side_handler = lambda e, idx=i: update_leg(idx, "side", e.control.value if (e and hasattr(e, "control") and e.control) else side_dropdown.value)
            for ev in ["on_select", "on_change", "on_selected"]:
                if hasattr(side_dropdown, ev):
                    setattr(side_dropdown, ev, side_handler)
            
            strike_field = ft.TextField(
                label="行使価格 K",
                value=str(leg_data["strike"]),
                keyboard_type=ft.KeyboardType.NUMBER,
                width=150,
                on_change=lambda e, idx=i: update_leg(
                    idx, "strike", safe_float(e.control.value, default=state["legs"].get(idx, {}).get("strike", 0.0))
                )
            )
            
            premium_field = ft.TextField(
                label="プレミアム",
                value=str(leg_data["premium"]),
                keyboard_type=ft.KeyboardType.NUMBER,
                width=150,
                on_change=lambda e, idx=i: update_leg(
                    idx, "premium", safe_float(e.control.value, default=state["legs"].get(idx, {}).get("premium", 0.0))
                )
            )
            
            quantity_field = ft.TextField(
                label="枚数",
                value=str(leg_data["quantity"]),
                keyboard_type=ft.KeyboardType.NUMBER,
                width=100,
                on_change=lambda e, idx=i: update_leg(
                    idx, "quantity", safe_int(e.control.value, default=state["legs"].get(idx, {}).get("quantity", 1))
                )
            )
            
            sigma_field = ft.TextField(
                label="ボラティリティ σ",
                value=str(leg_data["sigma"]),
                keyboard_type=ft.KeyboardType.NUMBER,
                width=150,
                on_change=lambda e, idx=i: update_leg(
                    idx, "sigma", safe_float(e.control.value, default=state["legs"].get(idx, {}).get("sigma", 0.2))
                )
            )
            
            t_days_field = ft.TextField(
                label="残存日数（日）",
                value=str(leg_data["T_days"]),
                keyboard_type=ft.KeyboardType.NUMBER,
                width=150,
                on_change=lambda e, idx=i: update_leg(
                    idx, "T_days", safe_int(e.control.value, default=state["legs"].get(idx, {}).get("T_days", 30))
                )
            )
            
            leg_card = ft.Card(
                content=ft.Container(
                    content=ft.Column([
                        ft.Text(f"レッグ {i}", weight=ft.FontWeight.BOLD),
                        ft.Row([type_dropdown, side_dropdown, strike_field]),
                        ft.Row([premium_field, quantity_field, sigma_field]),
                        ft.Row([t_days_field]),
                    ]),
                    padding=15
                )
            )
            
            legs_container.controls.append(leg_card)

    def update_leg(leg_idx, field, value):
        if leg_idx in state["legs"]:
            state["legs"][leg_idx][field] = value

    def generate_chart(e=None):
        update_state_from_fields()

        # ポジション構築
        positions = []
        for i in range(1, state["n_legs"] + 1):
            if i in state["legs"]:
                leg = state["legs"][i]
                positions.append(
                    OptionPosition(
                        option_type=leg["type"],
                        side=leg["side"],
                        strike=leg["strike"],
                        premium=leg["premium"],
                        quantity=leg["quantity"],
                        contract_size=state["contract_size"],
                        r=state["r"],
                        sigma=leg["sigma"],
                        T=leg["T_days"] / 365.0
                    )
                )

        if not positions:
            results_text.value = "少なくとも 1 つポジションを入力してください"
            chart_container.controls.clear()
            page.update()
            return

        # 計算
        spot_min = state["spot_center"] - state["spot_range"]
        spot_max = state["spot_center"] + state["spot_range"]
        
        # なめらかな曲線を生成（最低 500 点）
        n_points = max(int((spot_max - spot_min) / 50.0) + 1, 500)
        spots = np.linspace(spot_min, spot_max, n_points)

        total_now = total_pl_now_bs(positions, spots)
        total_maturity = total_payoff_maturity(positions, spots)

        max_now = float(np.max(total_now))
        min_now = float(np.min(total_now))
        max_mat = float(np.max(total_maturity))
        min_mat = float(np.min(total_maturity))

        # チャート作成（matplotlib）
        system = platform.system()
        if system == 'Darwin':  # macOS
            plt.rcParams['font.family'] = ['Hiragino Sans', 'Hiragino Maru Gothic Pro', 'AppleGothic']
        elif system == 'Windows':
            plt.rcParams['font.family'] = ['MS Gothic', 'Yu Gothic', 'Meiryo']
        else:  # Linux等
            plt.rcParams['font.family'] = ['Noto Sans CJK JP', 'IPAGothic', 'DejaVu Sans']

        plt.rcParams['axes.unicode_minus'] = False  # マイナス記号の文字化け対策

        fig, ax = plt.subplots(figsize=(10, 5.5))
        ax.plot(spots, total_now, label='期中損益 (BS 理論価格)', color='#ff7f0e', linewidth=2.5)
        ax.plot(spots, total_maturity, label='満期損益', color='#1f77b4', linewidth=2, linestyle='--')
        ax.axhline(y=0, color='gray', linewidth=1)
        ax.axvline(x=state["spot_center"], color='green', linewidth=1.2, linestyle=':', label=f'中心価格 ({state["spot_center"]:,.0f})')

        # 各レッグの行使価格（K）にマーカー垂直線を引く
        strikes = sorted(list({p.strike for p in positions}))
        for k in strikes:
            if spot_min <= k <= spot_max:
                ax.axvline(x=k, color='#9467bd', linewidth=0.8, linestyle=':', alpha=0.6)

        # 最大利益・損失ライン
        ax.axhline(y=max_now, color='#ff7f0e', linewidth=0.8, linestyle=':', alpha=0.6)
        ax.axhline(y=min_now, color='#ff7f0e', linewidth=0.8, linestyle=':', alpha=0.6)
        ax.axhline(y=max_mat, color='#1f77b4', linewidth=0.8, linestyle=':', alpha=0.6)
        ax.axhline(y=min_mat, color='#1f77b4', linewidth=0.8, linestyle=':', alpha=0.6)

        # 横軸・縦軸のフォーマット改善（3桁カンマ区切り、詳細すぎる目盛りの整理）
        ax.set_xlim(spot_min, spot_max)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"{x:,.0f}"))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, p: f"{y:,.0f}"))

        # 目盛りの数を多すぎず（6〜8本程度に）キリの良い数値で自動調整
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=8, steps=[1, 2, 2.5, 5, 10]))
        ax.tick_params(axis='x', rotation=15, labelsize=9)
        ax.tick_params(axis='y', labelsize=9)

        ax.set_xlabel('原資産価格 S', fontsize=11)
        ax.set_ylabel('損益 (円)', fontsize=11)
        ax.set_title(f'オプション戦略 損益グラフ（原資産: {state["ticker"]}）', fontsize=13, weight='bold')
        ax.legend(loc='best', framealpha=0.85)
        ax.grid(True, alpha=0.3, linestyle='--')
        fig.tight_layout()

        # メモリ上にPNG画像として出力し、ft.Image に渡す
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        plt.close(fig)
        img_bytes = buf.getvalue()

        chart_container.controls.clear()
        chart_container.controls.append(
            ft.Image(src=img_bytes, width=800, height=440)
        )

        results_text.value = (
            f"最大利益（期中）: {max_now:,.0f}\n"
            f"最大損失（期中）: {min_now:,.0f}\n"
            f"最大利益（満期）: {max_mat:,.0f}\n"
            f"最大損失（満期）: {min_mat:,.0f}"
        )
        update_range_display()

        page.update()

    def on_spot_range_field_change(e):
        try:
            val = float(spot_range_field.value)
            if val > 0:
                state["spot_range"] = val
                update_range_slider_bounds(val)
                update_range_display()
        except (ValueError, TypeError):
            pass

    def on_spot_center_field_change(e):
        try:
            val = float(spot_center_field.value)
            if val > 0 and abs(val - state["spot_center"]) > 0.01:
                state["spot_center"] = val
                update_ticker_info(force_k_update=True)
        except (ValueError, TypeError):
            pass

    # イベントハンドラー設定
    ticker_field.on_submit = lambda e: update_ticker_info(force_k_update=True)
    ticker_field.on_blur = lambda e: update_ticker_info(force_k_update=True)
    auto_fetch_check.on_change = lambda e: update_ticker_info(force_k_update=True)
    legs_slider.on_change = on_legs_change
    spot_range_field.on_change = on_spot_range_field_change
    spot_center_field.on_submit = on_spot_center_field_change
    for ev in ["on_select", "on_change", "on_selected"]:
        if hasattr(preset_dropdown, ev):
            setattr(preset_dropdown, ev, on_preset_change)

    # 初期レッグUI構築
    build_legs_ui()

    # レイアウト
    page.add(
        ft.Row([
            # サイドバー（左側）
            ft.Container(
                content=ft.Column([
                    ft.Text("原資産レンジ設定", size=20, weight=ft.FontWeight.BOLD),
                    ft.Row([
                        ticker_field,
                        ft.IconButton(
                            icon=ft.Icons.SEARCH,
                            tooltip="価格を取得して行使価格Kを更新",
                            on_click=lambda e: update_ticker_info(force_k_update=True),
                        )
                    ]),
                    auto_fetch_check,
                    spot_center_field,
                    yfinance_info,
                    spot_range_field,
                    spot_step_field,
                    ft.Divider(),
                    ft.Text("共通パラメータ", size=20, weight=ft.FontWeight.BOLD),
                    contract_size_field,
                    r_field,
                    ft.Divider(),
                    ft.Text("戦略プリセット", size=20, weight=ft.FontWeight.BOLD),
                    preset_dropdown,
                    ft.Divider(),
                    ft.Text("レッグ数", size=20, weight=ft.FontWeight.BOLD),
                    legs_slider,
                ], scroll="adaptive"),
                width=400,
                padding=20
            ),
            
            # メインエリア（右側）
            ft.Container(
                content=ft.Column([
                    ft.Text("ポジション入力", size=24, weight=ft.FontWeight.BOLD),
                    legs_container,
                    ft.FilledButton(
                        "グラフを更新",
                        on_click=generate_chart,
                        icon=ft.Icons.REFRESH
                    ),
                    ft.Divider(),
                    range_control_card,
                    chart_container,
                    results_text,
                ], scroll="adaptive"),
                expand=True,
                padding=20
            )
        ], expand=True)
    )
    
    # 初期情報更新 & 初回グラフ描画
    update_ticker_info()
    generate_chart()


if __name__ == "__main__":
    import flet.app as fapp
    fapp.run(main)

